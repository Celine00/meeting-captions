import argparse
import asyncio
import json
import os
import signal
import sys
import threading
import time
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import sounddevice as sd

import numpy as np
import requests
import certifi
import sounddevice as sd
from dotenv import load_dotenv
from faster_whisper import WhisperModel
from jinja2 import Template
from markdown import markdown
from scipy.io import wavfile
from scipy import signal as scipy_signal
import websockets

# Load environment variables from .env file
load_dotenv()

# -----------------------------
# Doubao (Volcengine Ark) config
# -----------------------------
# API domain by region; cn-beijing => ark.cn-beijing.volces.com :contentReference[oaicite:1]{index=1}
# Chat Completions endpoint & auth format :contentReference[oaicite:2]{index=2}
DOUBAO_BASE_URL = os.environ.get(
    "DOUBAO_BASE_URL",
    "https://ark.cn-beijing.volces.com/api/v3/chat/completions",
)

DEFAULT_DOUBAO_MODEL = os.environ.get("DOUBAO_MODEL", "doubao-seed-1-8-251228")  # example shown in docs :contentReference[oaicite:3]{index=3}


# -----------------------------
# Storage
# -----------------------------
def make_run_dir(root: Path) -> Path:
    now = datetime.now()
    day_dir = root / now.strftime("%Y-%m-%d")
    run_dir = day_dir / f"{now.strftime('%H-%M-%S')}_zoom"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")



# -----------------------------
# Audio device helpers
# -----------------------------
def list_audio_devices() -> None:
    print(sd.query_devices())


def _pick_input_device_by_predicate(prefer_channels: int, predicate) -> Tuple[int, str, int, float]:
    devices = sd.query_devices()
    candidates = []
    for i, d in enumerate(devices):
        name = d.get("name", "")
        in_ch = int(d.get("max_input_channels", 0) or 0)
        if in_ch <= 0:
            continue
        if predicate(name, d):
            dsr = float(d.get("default_samplerate", 0) or 0)
            candidates.append((i, name, in_ch, dsr))

    if not candidates:
        raise RuntimeError("No matching input device found.")

    # Prefer devices that can satisfy prefer_channels, then prefer larger input channel count
    candidates.sort(key=lambda x: (x[2] >= prefer_channels, x[2]), reverse=True)
    idx, name, in_ch, dsr = candidates[0]
    ch = min(prefer_channels, in_ch)
    if ch <= 0:
        raise RuntimeError(f"Selected device has no input channels: {name}")
    if dsr <= 0:
        # PortAudio sometimes returns 0; fall back to 48000 as a safe default for virtual devices
        dsr = 48000.0
    return idx, name, ch, dsr


def resolve_input_device(device_arg: str, prefer_channels: int) -> Tuple[int, str, int, int]:
    """
    Resolve an input device from a CLI arg.

    device_arg can be:
      - an integer index, e.g. "7"
      - "blackhole" (auto-pick BlackHole input)
      - a substring to match device name, e.g. "MacBook Microphone"
    Returns: (device_index, device_name, channels, default_samplerate_int)
    """
    device_arg = (device_arg or "").strip()

    # Index path
    if device_arg.isdigit():
        idx = int(device_arg)
        info = sd.query_devices(idx)
        name = info.get("name", str(idx))
        in_ch = int(info.get("max_input_channels", 0) or 0)
        if in_ch <= 0:
            raise RuntimeError(f"Device {idx} ({name}) has 0 input channels. Pick an input device, not Multi-Output/output-only.")
        ch = min(prefer_channels, in_ch)
        dsr = int(float(info.get("default_samplerate", 0) or 0) or 48000)
        return idx, name, ch, dsr

    # BlackHole convenience
    if device_arg.lower() in {"blackhole", "bh"} or "blackhole" in device_arg.lower():
        idx, name, ch, dsr = _pick_input_device_by_predicate(
            prefer_channels,
            lambda n, d: "blackhole" in (n or "").lower(),
        )
        return idx, name, ch, int(dsr)

    # Generic substring match (case-insensitive)
    idx, name, ch, dsr = _pick_input_device_by_predicate(
        prefer_channels,
        lambda n, d: device_arg.lower() in (n or "").lower(),
    )
    return idx, name, ch, int(dsr)


def resample_audio(mono: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """High-quality resample for float32 mono audio using polyphase filtering."""
    if sr_in == sr_out:
        return mono.astype(np.float32, copy=False)

    g = math.gcd(sr_in, sr_out)
    up = sr_out // g
    down = sr_in // g
    y = scipy_signal.resample_poly(mono, up, down).astype(np.float32)
    return y

# -----------------------------
# Audio ring buffer
# -----------------------------
class RingBuffer:
    def __init__(self, capacity_samples: int, channels: int):
        self.capacity = capacity_samples
        self.channels = channels
        self.buf = np.zeros((capacity_samples, channels), dtype=np.float32)
        self.write_pos = 0
        self.filled = False
        self.lock = threading.Lock()
        self.total_written = 0  # samples since start

    def push(self, frames: np.ndarray) -> None:
        # frames: (n, channels)
        n = frames.shape[0]
        if n == 0:
            return
        with self.lock:
            end = self.write_pos + n
            if end < self.capacity:
                self.buf[self.write_pos:end, :] = frames
            else:
                first = self.capacity - self.write_pos
                self.buf[self.write_pos:, :] = frames[:first, :]
                remain = n - first
                self.buf[:remain, :] = frames[first:, :]
                self.filled = True
            self.write_pos = (self.write_pos + n) % self.capacity
            if self.total_written + n >= self.capacity:
                self.filled = True
            self.total_written += n

    def get_last(self, n_samples: int) -> np.ndarray:
        with self.lock:
            available = self.capacity if self.filled else min(self.write_pos, self.capacity)
            n = min(n_samples, available)
            if n <= 0:
                return np.zeros((0, self.channels), dtype=np.float32)

            start = (self.write_pos - n) % self.capacity
            if start < self.write_pos and not (self.filled and start > self.write_pos):
                # contiguous
                out = self.buf[start:start + n, :].copy()
            else:
                # wrapped
                part1 = self.buf[start:, :]
                part2 = self.buf[:self.write_pos, :]
                out = np.vstack([part1, part2])[-n:, :].copy()
            return out

    def seconds_written(self, sr: int) -> float:
        with self.lock:
            return self.total_written / float(sr)


# -----------------------------
# WebSocket broadcaster
# -----------------------------
class Broadcaster:
    def __init__(self):
        self.clients = set()
        self.queue: "asyncio.Queue[Dict[str, Any]]" = asyncio.Queue()

    async def handler(self, websocket):
        self.clients.add(websocket)
        try:
            await websocket.send(json.dumps({"type": "hello", "ts": time.time()}))
            async for _ in websocket:
                # UI can send pings/commands later; ignore for MVP
                pass
        finally:
            self.clients.discard(websocket)

    async def fanout_loop(self):
        while True:
            msg = await self.queue.get()
            if msg.get("type") == "__stop__":
                break
            dead = []
            payload = json.dumps(msg, ensure_ascii=False)
            for ws in list(self.clients):
                try:
                    await ws.send(payload)
                except Exception:
                    dead.append(ws)
            for ws in dead:
                self.clients.discard(ws)


# -----------------------------
# Transcript sink
# -----------------------------
@dataclass
class SegmentRec:
    t0: float
    t1: float
    text: str
    avg_logprob: Optional[float] = None


class TranscriptWriter:
    def __init__(self, run_dir: Path):
        self.jsonl_path = run_dir / "transcript.jsonl"
        self.md_path = run_dir / "transcript.md"
        self._jsonl = self.jsonl_path.open("a", encoding="utf-8")

    def append(self, seg: SegmentRec) -> None:
        self._jsonl.write(json.dumps(seg.__dict__, ensure_ascii=False) + "\n")
        self._jsonl.flush()

    def close(self) -> None:
        self._jsonl.close()

    def render_markdown(self) -> str:
        lines = []
        for line in self.jsonl_path.read_text(encoding="utf-8").splitlines():
            obj = json.loads(line)
            t0 = obj["t0"]
            text = obj["text"].strip()
            mmss = f"{int(t0//60):02d}:{int(t0%60):02d}"
            lines.append(f"- **{mmss}** {text}")
        md = "# Transcript\n\n" + "\n".join(lines) + "\n"
        self.md_path.write_text(md, encoding="utf-8")
        return md


# -----------------------------
# Doubao summarize
# -----------------------------
def doubao_chat(messages: List[Dict[str, str]], model: str, timeout: int = 90) -> str:
    token = os.environ.get("DOUBAO_API_KEY")
    if not token:
        raise RuntimeError("Missing env DOUBAO_API_KEY")

    payload = {"model": model, "messages": messages}

    try:
        resp = requests.post(
            DOUBAO_BASE_URL,
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json=payload,
            timeout=timeout,
            verify=certifi.where(),  # Use certifi's certificate bundle
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]
    except requests.exceptions.HTTPError as e:
        # Add more context to the error
        error_msg = f"Doubao API error: {e}"
        try:
            error_data = resp.json()
            error_msg += f"\nResponse: {error_data}"
        except:
            error_msg += f"\nResponse text: {resp.text[:200]}"
        raise RuntimeError(error_msg) from e


def chunk_text(text: str, max_chars: int = 12000) -> List[str]:
    # 简单按字符切；更精细可按段落/句子
    text = text.strip()
    if len(text) <= max_chars:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        # 尽量在换行处分割
        cut = text.rfind("\n", start, end)
        if cut <= start + 2000:  # 换行太靠前就不强求
            cut = end
        chunks.append(text[start:cut].strip())
        start = cut
    return [c for c in chunks if c]

def summarize_transcript(transcript_md: str, model: str) -> str:
    chunks = chunk_text(transcript_md, max_chars=12000)

    partials = []
    for i, ch in enumerate(chunks, 1):
        msg = [
            {"role": "system", "content": "You write concise, actionable meeting minutes in English."},
            {
                "role": "user",
                "content": f"""Summarize this meeting transcript chunk ({i}/{len(chunks)}).

Return Markdown with:
- Key points (bullets)
- Decisions (bullets)
- Action items (bullets, include owner if mentioned)

Chunk:
{ch}
""",
            },
        ]
        partials.append(doubao_chat(msg, model=model))

    if len(partials) == 1:
        combined = partials[0]
    else:
        partials_md = os.linesep.join(["---" + os.linesep + p for p in partials])

        msg = [
            {"role": "system", "content": "You consolidate meeting minutes into a final coherent set of notes."},
            {
                "role": "user",
                "content": f"""Merge and deduplicate these partial summaries into final meeting notes.

Output Markdown with sections:
1) TL;DR (3-5 bullets)
2) Decisions
3) Action Items (table: Owner | Due | Task)
4) Open Questions / Risks

Partials:
{partials_md}
""",
            },
        ]
        combined = doubao_chat(msg, model=model)

    return combined


REPORT_TEMPLATE = Template("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Meeting Report - {{ started_at }}</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            line-height: 1.6;
            color: #333;
            background: #f5f5f5;
            padding: 20px;
        }
        .container {
            max-width: 900px;
            margin: 0 auto;
            background: white;
            padding: 40px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 { color: #2c3e50; margin-bottom: 10px; font-size: 2em; }
        h2 { color: #34495e; margin-top: 30px; margin-bottom: 15px; border-bottom: 2px solid #3498db; padding-bottom: 5px; }
        .meta {
            background: #ecf0f1;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 30px;
            font-size: 0.9em;
        }
        .meta strong { color: #2c3e50; }
        .summary { background: #e8f4f8; padding: 20px; border-left: 4px solid #3498db; margin-bottom: 30px; }
        .transcript { background: #fafafa; padding: 20px; border-left: 4px solid #95a5a6; }
        ul { margin-left: 20px; margin-bottom: 15px; }
        li { margin-bottom: 8px; }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }
        th {
            background-color: #3498db;
            color: white;
        }
        tr:nth-child(even) { background-color: #f9f9f9; }
        code {
            background: #f4f4f4;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Monaco', 'Courier New', monospace;
            font-size: 0.9em;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>📝 Meeting Report</h1>
        <div class="meta">
            <strong>Date:</strong> {{ started_at }}<br>
            <strong>Audio Device:</strong> {{ device }}<br>
            <strong>Whisper Model:</strong> {{ whisper_model }}
        </div>

        <div class="summary">
            <h2>📊 Summary</h2>
            {{ summary_html|safe }}
        </div>

        <div class="transcript">
            <h2>📜 Transcript</h2>
            {{ transcript_html|safe }}
        </div>
    </div>
</body>
</html>
""")


def build_report(run_dir: Path, meta: Dict[str, Any], summary_md: str, transcript_md: str) -> Path:
    summary_html = markdown(summary_md, extensions=["tables", "fenced_code"])
    transcript_html = markdown(transcript_md, extensions=["tables", "fenced_code"])

    html = REPORT_TEMPLATE.render(
        started_at=meta.get("started_at", ""),
        device=meta.get("audio_device_name", "Unknown"),
        whisper_model=meta.get("whisper_model", ""),
        summary_html=summary_html,
        transcript_html=transcript_html,
    )
    out = run_dir / "report.html"
    out.write_text(html, encoding="utf-8")
    return out


# -----------------------------
# Main capture + transcribe
# -----------------------------
async def run_capture(
    device_index: int,
    device_name: str,
    sr: int,
    channels: int,
    whisper_model_name: str,
    compute_type: str,
    window_s: float,
    step_s: float,
    ws_port: int,
    out_root: Path,
    doubao_model: str,
    save_wav: bool,
    no_summary: bool = False,
    stop_event: asyncio.Event = None,
):
    run_dir = make_run_dir(out_root)
    started_at = datetime.now().isoformat(timespec="seconds")

    meta = {
        "started_at": started_at,
        "audio_device_index": int(device_index),
        "audio_device_name": str(device_name),
        "sample_rate": sr,
        "channels": channels,
        "whisper_model": whisper_model_name,
        "compute_type": compute_type,
        "window_s": window_s,
        "step_s": step_s,
        "ws_port": ws_port,
        "doubao_model": doubao_model,
    }
    write_json(run_dir / "meta.json", meta)

    # Whisper (M1 Pro: CPU + int8 usually best for speed/mem)
    model = WhisperModel(whisper_model_name, device="cpu", compute_type=compute_type)

    # Ring buffer holds last window + some slack
    capacity = int(sr * max(window_s * 2, 20))
    ring = RingBuffer(capacity_samples=capacity, channels=channels)

    # Optional raw wav capture (for debug)
    wav_frames: List[np.ndarray] = []

    def audio_cb(indata, frames, time_info, status):
        if status:
            # print status only if needed
            pass
        ring.push(indata.copy())
        if save_wav:
            wav_frames.append(indata.copy())

    # WebSocket
    broadcaster = Broadcaster()
    ws_server = await websockets.serve(broadcaster.handler, "127.0.0.1", ws_port)
    fanout_task = asyncio.create_task(broadcaster.fanout_loop())

    tw = TranscriptWriter(run_dir)

    # Dedupe state: last emitted absolute end time
    last_emitted_t1 = 0.0

    print(f"[RUN] {run_dir}")
    print(f"[AUDIO] input={device_index} | {device_name} | sr={sr} | ch={channels}")
    print(f"[WS]  ws://127.0.0.1:{ws_port}")
    print(f"[WAV] save_wav={save_wav}")
    print("Press Ctrl+C to stop.\n")

    # Start audio stream
    stream = sd.InputStream(
        device=device_index,
        samplerate=sr,
        channels=channels,
        dtype="float32",
        callback=audio_cb,
        blocksize=0,
    )

    try:
        with stream:
            try:
                while not (stop_event and stop_event.is_set()):
                    await asyncio.sleep(step_s)

                    audio = ring.get_last(int(sr * window_s))  # (n, ch)
                    if audio.shape[0] < int(sr * min(2.0, window_s)):
                        continue

                    # downmix to mono
                    mono = np.mean(audio, axis=1).astype(np.float32)

                    # output rms to debug
                    rms = float(np.sqrt(np.mean(mono * mono)))
                    print(f"[LEVEL] rms={rms:.6f}")

                    # compute window absolute start/end in "recording time"
                    t_end = ring.seconds_written(sr)
                    t_start = max(0.0, t_end - window_s)

                    # transcribe
                    mono_for_whisper = resample_audio(mono, sr_in=sr, sr_out=16000)

                    segments, info = model.transcribe(
                        mono_for_whisper,
                        language="en",
                        vad_filter=True,
                        beam_size=2,
                        word_timestamps=False,
                    )

                    new_count = 0
                    for seg in segments:
                        abs_t0 = t_start + float(seg.start)
                        abs_t1 = t_start + float(seg.end)
                        text = (seg.text or "").strip()
                        if not text:
                            continue

                        # Only emit if it extends beyond last emitted (with small tolerance)
                        if abs_t1 <= last_emitted_t1 + 0.15:
                            continue

                        last_emitted_t1 = max(last_emitted_t1, abs_t1)
                        rec = SegmentRec(
                            t0=abs_t0,
                            t1=abs_t1,
                            text=text,
                            avg_logprob=getattr(seg, "avg_logprob", None),
                        )
                        tw.append(rec)

                        await broadcaster.queue.put(
                            {"type": "caption", "t0": rec.t0, "t1": rec.t1, "text": rec.text}
                        )

                        # Console view (optional)
                        mmss = f"{int(rec.t0//60):02d}:{int(rec.t0%60):02d}"
                        print(f"[{mmss}] {rec.text}")
                        new_count += 1

                    if new_count == 0:
                        # keepalive for UI if you want
                        await broadcaster.queue.put({"type": "tick", "t": time.time()})

            except Exception as e:
                print(f"\n[ERROR] {e}")
                import traceback
                traceback.print_exc()
    finally:
        print("[CLEANUP] Starting cleanup...")
        # Shutdown WS
        try:
            print("[CLEANUP] Closing WebSocket server...")
            ws_server.close()
            await ws_server.wait_closed()
            print("[CLEANUP] WebSocket server closed")
        except Exception as e:
            print(f"[CLEANUP] WS shutdown error (ignored): {e}")

        try:
            print("[CLEANUP] Stopping broadcaster...")
            await broadcaster.queue.put({"type": "__stop__"})
            try:
                await fanout_task
            except asyncio.CancelledError:
                print("[CLEANUP] Fanout task was cancelled")
            print("[CLEANUP] Broadcaster stopped")
        except Exception as e:
            print(f"[CLEANUP] Broadcaster shutdown error (ignored): {e}")

        print("[CLEANUP] Closing transcript writer...")
        tw.close()
        print("[CLEANUP] Transcript writer closed")

        # Save wav if enabled
        if save_wav:
            if wav_frames:
                print(f"[WAV] Saving {len(wav_frames)} audio chunks to audio.wav...")
                raw = np.concatenate(wav_frames, axis=0)  # (n, ch)
                # convert to int16 PCM
                pcm = np.clip(raw, -1.0, 1.0)
                pcm16 = (pcm * 32767.0).astype(np.int16)
                wavfile.write(str(run_dir / "audio.wav"), sr, pcm16)
                print(f"[WAV] Saved audio.wav ({pcm16.shape[0]} samples, {pcm16.shape[0]/sr:.1f}s)")
            else:
                print("[WAV] Warning: --save-wav enabled but no audio frames were captured")

        # Render transcript.md
        transcript_md = tw.render_markdown()

        # Summarize via Doubao
        summary_md = ""
        if no_summary:
            print("[SUM] Skipping summary (--no-summary flag)")
            summary_md = "# Summary\n\n*Summary generation skipped.*\n"
        else:
            try:
                print("[SUM] Calling Doubao LLM...")
                summary_md = summarize_transcript(transcript_md, model=doubao_model)
                (run_dir / "summary.md").write_text(summary_md, encoding="utf-8")
                print("[SUM] Summary saved")
            except Exception as e:
                print(f"[SUM] Warning: Failed to generate summary: {e}")
                summary_md = "# Summary\n\n*Summary generation failed. See transcript below.*\n"
                (run_dir / "summary.md").write_text(f"Error: {e}\n", encoding="utf-8")

        # Build report.html
        try:
            report_path = build_report(run_dir, meta, summary_md, transcript_md)
            print(f"[DONE] report: {report_path}")
            print("Tip: open it with:")
            print(f"open '{report_path}'")
        except Exception as e:
            print(f"[DONE] Report generation failed: {e}")
            print(f"[DONE] Transcript saved to: {run_dir / 'transcript.md'}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=str, default="blackhole", help="Input device: index (e.g. 7) or name substring (e.g. blackhole)")
    p.add_argument("--list-devices", action="store_true", help="List audio devices and exit")
    p.add_argument("--sr", type=int, default=0, help="Capture sample rate. 0 = use device default")
    p.add_argument("--channels", type=int, default=2, help="BlackHole usually 2ch")
    p.add_argument("--whisper-model", type=str, default="small")
    p.add_argument("--compute-type", type=str, default="int8", help="M1 Pro CPU recommended: int8")
    p.add_argument("--window-s", type=float, default=8.0)
    p.add_argument("--step-s", type=float, default=1.0)
    p.add_argument("--ws-port", type=int, default=8765)
    p.add_argument(
        "--out-root",
        type=str,
        default="/Users/celinezou/Celine00/meeting-captions/records",
    )
    p.add_argument("--doubao-model", type=str, default=DEFAULT_DOUBAO_MODEL)
    p.add_argument("--no-summary", action="store_true", help="skip LLM summary generation")
    p.add_argument("--save-wav", action="store_true", help="save audio.wav for debugging")
    args = p.parse_args()

    if args.list_devices:
        list_audio_devices()
        return

    # Resolve input device robustly (indexes can change when plugging/unplugging headsets)
    device_index, device_name, channels, device_default_sr = resolve_input_device(
        args.device, prefer_channels=args.channels
    )

    # Auto SR: use device default unless user overrides
    sr = int(args.sr) if int(args.sr) > 0 else int(device_default_sr)

    out_root = Path(args.out_root).expanduser()
    out_root.mkdir(parents=True, exist_ok=True)

    # Create event loop to handle signals properly
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Event to signal shutdown
    stop_event = asyncio.Event()

    def signal_handler(sig, frame):
        print("\n[STOP] Stopping...")
        stop_event.set()

    # Register signal handler
    signal.signal(signal.SIGINT, signal_handler)

    try:
        loop.run_until_complete(
            run_capture(
                device_index=device_index,
                device_name=device_name,
                sr=sr,
                channels=channels,
                whisper_model_name=args.whisper_model,
                compute_type=args.compute_type,
                window_s=args.window_s,
                step_s=args.step_s,
                ws_port=args.ws_port,
                out_root=out_root,
                doubao_model=args.doubao_model,
                save_wav=args.save_wav,
                no_summary=args.no_summary,
                stop_event=stop_event,
            )
        )
    finally:
        loop.close()


if __name__ == "__main__":
    main()
