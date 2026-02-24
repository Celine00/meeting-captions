import argparse
import queue
import time
from typing import Optional

import numpy as np
import sounddevice as sd


def resolve_input_device(device_arg: str) -> int:
    device_arg = (device_arg or "").strip()
    if device_arg.isdigit():
        idx = int(device_arg)
        info = sd.query_devices(idx)
        if int(info.get("max_input_channels", 0) or 0) <= 0:
            raise RuntimeError(f"Device {idx} is not an input device: {info.get('name', idx)}")
        return idx

    devices = sd.query_devices()
    for i, d in enumerate(devices):
        name = str(d.get("name", "") or "")
        if int(d.get("max_input_channels", 0) or 0) <= 0:
            continue
        if device_arg.lower() in name.lower():
            return i
    raise RuntimeError(f"No matching input device for '{device_arg}'")


def main():
    p = argparse.ArgumentParser(description="Monitor input audio RMS/peak per second.")
    p.add_argument("--device", type=str, default="blackhole", help="Input device index or name substring")
    p.add_argument("--sr", type=int, default=0, help="Sample rate (0 = device default)")
    p.add_argument("--channels", type=int, default=2, help="Requested channels")
    p.add_argument("--seconds", type=int, default=30, help="Monitoring duration")
    args = p.parse_args()

    idx = resolve_input_device(args.device)
    info = sd.query_devices(idx)
    name = str(info.get("name", idx))
    max_in = int(info.get("max_input_channels", 0) or 0)
    channels = min(max_in, int(args.channels))
    if channels <= 0:
        raise RuntimeError(f"Selected device has no input channels: {name}")

    sr = int(args.sr) if int(args.sr) > 0 else int(float(info.get("default_samplerate", 0) or 0) or 48000)
    q: "queue.Queue[np.ndarray]" = queue.Queue()

    def cb(indata, frames, time_info, status):
        if status:
            print(f"[WARN] {status}")
        q.put(indata.copy())

    print(f"[MONITOR] device={idx} | {name} | sr={sr} | ch={channels}")
    print("[MONITOR] Ctrl+C to stop")

    started = time.time()
    bucket = []
    last_tick = started
    total_samples = 0

    try:
        with sd.InputStream(
            device=idx,
            samplerate=sr,
            channels=channels,
            dtype="float32",
            blocksize=0,
            callback=cb,
        ):
            while True:
                now = time.time()
                if now - started >= float(args.seconds):
                    break

                timeout = max(0.0, 1.0 - (now - last_tick))
                try:
                    frame = q.get(timeout=timeout)
                    bucket.append(frame)
                    total_samples += frame.shape[0]
                except queue.Empty:
                    pass

                now = time.time()
                if now - last_tick >= 1.0:
                    if bucket:
                        x = np.concatenate(bucket, axis=0)
                        ch_rms = np.sqrt(np.mean(x * x, axis=0))
                        ch_peak = np.max(np.abs(x), axis=0)
                        rms_txt = ",".join(f"{float(v):.6f}" for v in ch_rms.tolist())
                        peak_txt = ",".join(f"{float(v):.4f}" for v in ch_peak.tolist())
                        print(f"[LEVEL] rms=[{rms_txt}] peak=[{peak_txt}]")
                    else:
                        print("[LEVEL] no audio frames")
                    bucket = []
                    last_tick = now
    except KeyboardInterrupt:
        pass

    print(f"[DONE] captured_samples={total_samples} (~{total_samples/float(sr):.1f}s)")


if __name__ == "__main__":
    main()
