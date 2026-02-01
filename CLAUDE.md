# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A real-time meeting transcription system that captures audio (typically via BlackHole virtual audio device), transcribes it using Faster-Whisper, and generates meeting summaries using Doubao (Volcengine Ark) LLM. The system provides live WebSocket-based captions and generates comprehensive HTML reports with transcripts and summaries.

## Development Commands

### Initial Setup

```bash
# Activate virtual environment
pyenv activate meeting-captions

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
# Edit .env and add your DOUBAO_API_KEY
```

### Running the Application

```bash
# Basic usage with BlackHole device
python3 app.py --device blackhole

# List available audio devices
python3 app.py --list-devices
# Or use the helper script
python3 list_devices.py

# Full configuration example
python3 app.py \
  --device blackhole \
  --sr 48000 \
  --channels 2 \
  --whisper-model small \
  --compute-type int8 \
  --window-s 8.0 \
  --step-s 1.0 \
  --ws-port 8765 \
  --doubao-model doubao-pro-32k-240615 \
  --save-wav
```

### Environment Setup

Environment variables are loaded from `.env` file (create from `.env.example`):
- `DOUBAO_API_KEY` (required) - API key for Doubao (Volcengine Ark) LLM service
- `DOUBAO_BASE_URL` (optional) - Defaults to `https://ark.cn-beijing.volces.com/api/v3/chat/completions`
- `DOUBAO_MODEL` (optional) - Defaults to `doubao-pro-32k-240615`

### Dependencies

Core dependencies (see `requirements.txt`):
- `faster-whisper` - Local Whisper transcription
- `sounddevice` - Audio I/O
- `numpy`, `scipy` - Audio processing and resampling
- `websockets` - Real-time caption broadcasting
- `jinja2` - HTML report templating
- `markdown` - Markdown to HTML conversion
- `requests` - Doubao API calls
- `python-dotenv` - Load environment variables from .env file

## Architecture

### Audio Capture Pipeline

1. **Device Resolution** (`resolve_input_device`): Flexible device selection by index, name substring, or "blackhole" shorthand
2. **Ring Buffer** (`RingBuffer`): Thread-safe circular buffer storing recent audio (default: 2x window size or 20s)
3. **Audio Callback**: Continuously captures audio frames in real-time
4. **Resampling** (`resample_audio`): High-quality polyphase resampling to 16kHz for Whisper

### Transcription Loop (async)

- **Sliding Window**: Processes audio in overlapping windows (default: 8s window, 1s step)
- **Whisper Transcription**: Uses `faster-whisper` with VAD filtering on CPU + int8 quantization (optimized for M1 Pro)
- **Deduplication**: Tracks `last_emitted_t1` to prevent duplicate segments across windows
- **Absolute Timestamps**: Converts Whisper's relative timestamps to absolute recording time

### Output Generation

1. **Real-time**: WebSocket broadcasts (`ws://127.0.0.1:8765`) for live caption display
2. **Storage**: JSONL transcript log (`transcript.jsonl`) with structured segment records
3. **Post-processing** (on Ctrl+C):
   - Renders `transcript.md` with `MM:SS` timestamps
   - Chunked summarization via Doubao (handles long transcripts by splitting at ~12k chars)
   - HTML report generation combining transcript + summary

### File Structure

```
records/
└── YYYY-MM-DD/
    └── HH-MM-SS_zoom/
        ├── meta.json          # Run configuration
        ├── transcript.jsonl   # Raw segment records
        ├── transcript.md      # Human-readable transcript
        ├── summary.md         # LLM-generated meeting notes
        ├── report.html        # Combined HTML report
        └── audio.wav          # Optional raw audio (--save-wav)
```

## Key Implementation Details

### Device Selection Strategy

The `resolve_input_device` function supports three patterns:
- Numeric index (e.g., `--device 7`)
- "blackhole" keyword for auto-detection
- Substring matching (e.g., `--device "MacBook Microphone"`)

Device selection auto-detects channel count and sample rate. Virtual devices like BlackHole may return 0 for sample rate; the code defaults to 48kHz in such cases.

### Audio Quality & Resampling

- Uses scipy's `resample_poly` for high-quality polyphase filtering
- Automatically downmixes multi-channel to mono for Whisper
- Resamples to 16kHz (Whisper's expected input rate)
- RMS level monitoring printed during capture for debugging

### Transcription Deduplication

Critical for sliding window approach:
- Each window's segments get timestamps relative to window start
- Converted to absolute time: `abs_t = t_window_start + seg.start`
- Only emit if `abs_t1 > last_emitted_t1 + 0.15s` tolerance
- Prevents duplicate text when windows overlap

### Doubao Integration

Summary generation uses a two-phase approach:
1. **Chunking**: Splits long transcripts (~12k char chunks) with newline-aware splitting
2. **Partial Summaries**: Each chunk gets key points, decisions, action items
3. **Consolidation**: If multiple chunks, a second pass merges into final format:
   - TL;DR (3-5 bullets)
   - Decisions
   - Action Items (Owner | Due | Task table)
   - Open Questions / Risks

### HTML Report Generation

The `REPORT_TEMPLATE` is a Jinja2 template that generates a styled HTML report combining:
- Meeting metadata (date, device, model)
- Summary section with blue highlight
- Full transcript with gray highlight
- Responsive design with clean typography

## Testing Considerations

When testing or debugging:
- Use `--save-wav` to capture raw audio for replay/analysis
- Check RMS levels in console output to verify audio capture
- Use `--list-devices` if device indexes change (headset plug/unplug)
- WebSocket port 8765 must be available
- Doubao API requires valid auth token in `DOUBAO_API_KEY`


<claude-mem-context>
# Recent Activity

<!-- This section is auto-generated by claude-mem. Edit content outside the tags. -->

*No recent activity*
</claude-mem-context>