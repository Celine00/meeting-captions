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

# Core libraries (fast imports)
import numpy as np
import requests
import certifi
import sounddevice as sd
from dotenv import load_dotenv

# Heavy imports - imported at module level for better error messages
# but actual model loading is deferred
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
    text: str  # Original transcription (usually English)
    text_zh: Optional[str] = None  # Chinese translation (if enabled)
    avg_logprob: Optional[float] = None


class TranscriptWriter:
    def __init__(self, run_dir: Path):
        self.jsonl_path = run_dir / "transcript.jsonl"
        self.run_dir = run_dir
        self._jsonl = self.jsonl_path.open("a", encoding="utf-8")

    def append(self, seg: SegmentRec) -> None:
        self._jsonl.write(json.dumps(seg.__dict__, ensure_ascii=False) + "\n")
        self._jsonl.flush()

    def close(self) -> None:
        self._jsonl.close()

    def render_markdown(self, lang: str = "en") -> str:
        """Render transcript as markdown in specified language"""
        lines = []
        for line in self.jsonl_path.read_text(encoding="utf-8").splitlines():
            obj = json.loads(line)
            t0 = obj["t0"]

            # Choose text based on language
            if lang == "zh" and "text_zh" in obj and obj["text_zh"]:
                text = obj["text_zh"].strip()
            else:
                text = obj["text"].strip()

            mmss = f"{int(t0//60):02d}:{int(t0%60):02d}"
            lines.append(f"- **{mmss}** {text}")

        title = "# 会议记录\n\n" if lang == "zh" else "# Transcript\n\n"
        md = title + "\n".join(lines) + "\n"

        # Save to appropriate file
        md_file = self.run_dir / f"transcript-{lang}.md"
        md_file.write_text(md, encoding="utf-8")
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


async def translate_to_chinese_async(text: str, model: str) -> str:
    """Translate English text to Chinese using Doubao LLM"""
    try:
        messages = [
            {
                "role": "system",
                "content": "You are a professional translator. Translate the following text to Simplified Chinese. Return ONLY the translation, no explanations or additional text."
            },
            {"role": "user", "content": text}
        ]
        # Use asyncio to run synchronous doubao_chat in executor
        loop = asyncio.get_event_loop()
        translation = await loop.run_in_executor(
            None,
            lambda: doubao_chat(messages, model=model, timeout=30)
        )
        return translation.strip()
    except Exception as e:
        print(f"[TRANSLATE] Warning: Translation failed: {e}")
        return text  # Fallback to original text


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

def summarize_transcript(transcript_md: str, model: str, lang: str = "en") -> str:
    """Generate meeting summary in specified language"""
    chunks = chunk_text(transcript_md, max_chars=12000)

    # Language-specific system prompts and instructions
    if lang == "zh":
        system_prompt = "你是一个专业的会议纪要撰写助手。请用简体中文撰写简洁、可操作的会议纪要。"
        summary_instruction = """请总结这段会议记录（第{}/{}段）。

使用Markdown格式返回：
- 关键要点（项目符号列表）
- 决策事项（项目符号列表）
- 行动项（项目符号列表，如有提及请包含负责人）

会议记录：
{}
"""
        consolidate_system = "你负责将多个部分总结合并去重，生成最终的会议纪要。"
        consolidate_instruction = """请将以下部分总结合并去重，生成最终的会议纪要。

使用Markdown格式，包含以下章节：
1) 核心要点（3-5个要点）
2) 决策事项
3) 行动项（表格：负责人 | 截止日期 | 任务）
4) 待解决问题 / 风险

部分总结：
{}
"""
    else:  # English
        system_prompt = "You write concise, actionable meeting minutes in English."
        summary_instruction = """Summarize this meeting transcript chunk ({}/{}).

Return Markdown with:
- Key points (bullets)
- Decisions (bullets)
- Action items (bullets, include owner if mentioned)

Chunk:
{}
"""
        consolidate_system = "You consolidate meeting minutes into a final coherent set of notes."
        consolidate_instruction = """Merge and deduplicate these partial summaries into final meeting notes.

Output Markdown with sections:
1) TL;DR (3-5 bullets)
2) Decisions
3) Action Items (table: Owner | Due | Task)
4) Open Questions / Risks

Partials:
{}
"""

    partials = []
    for i, ch in enumerate(chunks, 1):
        msg = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": summary_instruction.format(i, len(chunks), ch),
            },
        ]
        partials.append(doubao_chat(msg, model=model))

    if len(partials) == 1:
        combined = partials[0]
    else:
        partials_md = os.linesep.join(["---" + os.linesep + p for p in partials])

        msg = [
            {"role": "system", "content": consolidate_system},
            {
                "role": "user",
                "content": consolidate_instruction.format(partials_md),
            },
        ]
        combined = doubao_chat(msg, model=model)

    return combined


REPORT_TEMPLATE = Template("""<!DOCTYPE html>
<html lang="{{ lang_code }}">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }} - {{ started_at }}</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, 'PingFang SC', 'Microsoft YaHei', sans-serif;
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
        <h1>{{ title }}</h1>
        <div class="meta">
            <strong>{{ date_label }}:</strong> {{ started_at }}<br>
            <strong>{{ device_label }}:</strong> {{ device }}<br>
            <strong>{{ model_label }}:</strong> {{ whisper_model }}
        </div>

        <div class="summary">
            <h2>{{ summary_title }}</h2>
            {{ summary_html|safe }}
        </div>

        <div class="transcript">
            <h2>{{ transcript_title }}</h2>
            {{ transcript_html|safe }}
        </div>
    </div>
</body>
</html>
""")


def build_report(run_dir: Path, meta: Dict[str, Any], summary_md: str, transcript_md: str, lang: str = "en") -> Path:
    """Build HTML report in specified language"""
    summary_html = markdown(summary_md, extensions=["tables", "fenced_code"])
    transcript_html = markdown(transcript_md, extensions=["tables", "fenced_code"])

    # Language-specific labels
    if lang == "zh":
        title = "📝 会议报告"
        date_label = "日期"
        device_label = "音频设备"
        model_label = "Whisper模型"
        summary_title = "📊 会议总结"
        transcript_title = "📜 会议记录"
        lang_code = "zh-CN"
    else:
        title = "📝 Meeting Report"
        date_label = "Date"
        device_label = "Audio Device"
        model_label = "Whisper Model"
        summary_title = "📊 Summary"
        transcript_title = "📜 Transcript"
        lang_code = "en"

    html = REPORT_TEMPLATE.render(
        title=title,
        lang_code=lang_code,
        date_label=date_label,
        device_label=device_label,
        model_label=model_label,
        summary_title=summary_title,
        transcript_title=transcript_title,
        started_at=meta.get("started_at", ""),
        device=meta.get("audio_device_name", "Unknown"),
        whisper_model=meta.get("whisper_model", ""),
        summary_html=summary_html,
        transcript_html=transcript_html,
    )

    out = run_dir / f"report-{lang}.html"
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
    caption_lang: str = "en",
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
        "caption_lang": caption_lang,
    }
    write_json(run_dir / "meta.json", meta)

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

    print(f"\n[RUN] {run_dir}")
    print(f"[AUDIO] input={device_index} | {device_name} | sr={sr} | ch={channels}")
    print(f"[WS]  ws://127.0.0.1:{ws_port}")
    print(f"[WAV] save_wav={save_wav}")

    # Start audio stream IMMEDIATELY
    stream = sd.InputStream(
        device=device_index,
        samplerate=sr,
        channels=channels,
        dtype="float32",
        callback=audio_cb,
        blocksize=0,
    )

    print(f"\n[AUDIO] Capturing started")
    print(f"[MODEL] Loading Whisper '{whisper_model_name}' in background...")

    # Load Whisper model in parallel with audio capture
    # This async task runs while we're buffering audio
    loop = asyncio.get_event_loop()
    model_future = loop.run_in_executor(
        None,
        lambda: WhisperModel(whisper_model_name, device="cpu", compute_type=compute_type)
    )

    print("\n[READY] Press Ctrl+C to stop.\n")

    try:
        with stream:
            # Wait for model to load (with progress updates)
            model = None
            model_loaded = False

            try:
                while not (stop_event and stop_event.is_set()):
                    await asyncio.sleep(step_s)

                    # Check if model is ready (non-blocking)
                    if not model_loaded:
                        if model_future.done():
                            model = await model_future
                            print(f"[MODEL] Whisper model loaded and ready")
                            model_loaded = True
                        else:
                            # Still loading, show buffering status
                            buffered_s = ring.seconds_written(sr)
                            print(f"[MODEL] Loading... (buffered {buffered_s:.1f}s of audio)")
                            continue

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

                        # Create segment record
                        rec = SegmentRec(
                            t0=abs_t0,
                            t1=abs_t1,
                            text=text,
                            avg_logprob=getattr(seg, "avg_logprob", None),
                        )

                        # Translate to Chinese if needed
                        if caption_lang == "zh":
                            rec.text_zh = await translate_to_chinese_async(rec.text, doubao_model)

                        # Save to JSONL
                        tw.append(rec)

                        # Broadcast via WebSocket
                        await broadcaster.queue.put({
                            "type": "caption",
                            "t0": rec.t0,
                            "t1": rec.t1,
                            "text": rec.text,
                            "text_zh": rec.text_zh,
                            "display_lang": caption_lang
                        })

                        # Console output - show language based on setting
                        mmss = f"{int(rec.t0//60):02d}:{int(rec.t0%60):02d}"
                        display_text = rec.text_zh if (caption_lang == "zh" and rec.text_zh) else rec.text
                        print(f"[{mmss}] {display_text}")
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

        # Render transcripts
        print("[TRANSCRIPT] Rendering markdown...")
        transcript_en = tw.render_markdown(lang="en")

        # Generate Chinese transcript only if translations exist
        transcript_zh = None
        if caption_lang == "zh":
            transcript_zh = tw.render_markdown(lang="zh")

        print("[TRANSCRIPT] Transcript files saved")

        # Generate summaries in both languages (unless --no-summary)
        summary_en = ""
        summary_zh = ""

        if no_summary:
            print("[SUM] Skipping summary (--no-summary flag)")
            summary_en = "# Summary\n\n*Summary generation skipped.*\n"
            summary_zh = "# 总结\n\n*已跳过总结生成。*\n"
        else:
            try:
                print("[SUM] Generating English summary...")
                summary_en = summarize_transcript(transcript_en, model=doubao_model, lang="en")
                (run_dir / "summary-en.md").write_text(summary_en, encoding="utf-8")
                print("[SUM] English summary saved")

                print("[SUM] Generating Chinese summary...")
                summary_zh = summarize_transcript(transcript_en, model=doubao_model, lang="zh")
                (run_dir / "summary-zh.md").write_text(summary_zh, encoding="utf-8")
                print("[SUM] Chinese summary saved")
            except Exception as e:
                print(f"[SUM] Warning: Failed to generate summaries: {e}")
                summary_en = "# Summary\n\n*Summary generation failed.*\n"
                summary_zh = "# 总结\n\n*总结生成失败。*\n"
                (run_dir / "summary-en.md").write_text(f"Error: {e}\n", encoding="utf-8")
                (run_dir / "summary-zh.md").write_text(f"Error: {e}\n", encoding="utf-8")

        # Build HTML reports
        try:
            print("[REPORT] Building HTML reports...")
            report_en = build_report(run_dir, meta, summary_en, transcript_en, lang="en")
            report_zh = build_report(run_dir, meta, summary_zh, transcript_zh or transcript_en, lang="zh")
            print(f"[DONE] English report: {report_en}")
            print(f"[DONE] Chinese report: {report_zh}")
            print("Tip: open them with:")
            print(f"open '{report_en}'")
            print(f"open '{report_zh}'")
        except Exception as e:
            print(f"[DONE] Report generation failed: {e}")
            print(f"[DONE] Transcripts saved to: {run_dir}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=str, default="blackhole", help="Input device: index (e.g. 7) or name substring (e.g. blackhole)")
    p.add_argument("--list-devices", action="store_true", help="List audio devices and exit")
    p.add_argument("--sr", type=int, default=0, help="Capture sample rate. 0 = use device default")
    p.add_argument("--channels", type=int, default=2, help="BlackHole usually 2ch")
    p.add_argument("--whisper-model", type=str, default="base", help="Whisper model: tiny, base, small, medium, large (default: base for faster startup)")
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
    p.add_argument(
        "--caption-lang",
        type=str,
        choices=["en", "zh"],
        default="en",
        help="Caption display language: en=English (default), zh=Chinese (translates in real-time)"
    )
    args = p.parse_args()

    if args.list_devices:
        list_audio_devices()
        return

    # Provide immediate feedback
    print("\n" + "="*60)
    print("  MeetingCaptions - Real-time Transcription System")
    print("="*60)
    print(f"\n[CONFIG] Whisper model: {args.whisper_model}")
    print(f"[CONFIG] Doubao model: {args.doubao_model}")
    print(f"[CONFIG] Caption language: {args.caption_lang}")
    print(f"[CONFIG] Summary: {'disabled' if args.no_summary else 'enabled'}")

    print("\n[INIT] Resolving audio device...")
    # Resolve input device robustly (indexes can change when plugging/unplugging headsets)
    device_index, device_name, channels, device_default_sr = resolve_input_device(
        args.device, prefer_channels=args.channels
    )
    print(f"[INIT] Audio device: {device_name} (index={device_index})")

    # Auto SR: use device default unless user overrides
    sr = int(args.sr) if int(args.sr) > 0 else int(device_default_sr)

    out_root = Path(args.out_root).expanduser()
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"[INIT] Output directory: {out_root}")

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
                caption_lang=args.caption_lang,
                stop_event=stop_event,
            )
        )
    finally:
        loop.close()


if __name__ == "__main__":
    main()
