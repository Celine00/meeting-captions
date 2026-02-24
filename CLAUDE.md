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

### Performance Utilities

```bash
# Benchmark model loading times to choose the right model
python3 benchmark_models.py

# Pre-download models to avoid first-run delay
python3 preload_model.py base    # Pre-load base model
python3 preload_model.py tiny    # Pre-load tiny model
python3 preload_model.py small   # Pre-load small model
```

### Running the Application

```bash
# Default production config (BlackHole + medium + English + save wav)
python3 app.py

# Equivalent explicit command
python3 app.py \
  --device blackhole \
  --whisper-model medium \
  --save-wav \
  --language en \
  --beam-size 1 \
  --min-rms 0.01 \
  --window-s 2 \
  --step-s 1

# List available audio devices
python3 app.py --list-devices
# Or use the helper script
python3 list_devices.py

# Pre-meeting input-level check (recommended)
python3 monitor_levels.py --device blackhole --seconds 15 --channels 2

# Full configuration example
python3 app.py \
  --device blackhole \
  --sr 48000 \
  --channels 2 \
  --whisper-model medium \
  --compute-type int8 \
  --window-s 2.0 \
  --step-s 1.0 \
  --language en \
  --beam-size 1 \
  --min-rms 0.01 \
  --ws-port 8765 \
  --doubao-model doubao-pro-32k-240615 \
  --save-wav

# Skip summary generation
python3 app.py --device blackhole --no-summary
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
2. **Ring Buffer** (`RingBuffer`): Thread-safe circular buffer storing recent audio (default: max(2x window, 120s) for model-load backfill)
3. **Audio Callback**: Continuously captures audio frames in real-time
4. **Channel Selection + Gain** (`to_mono_best_channel`): picks strongest channel to avoid cancellation, optional gain before decode
5. **Resampling** (`resample_audio`): High-quality polyphase resampling to 16kHz for Whisper

### Transcription Loop (async)

- **Sliding Window**: Processes audio in overlapping windows (default: 4s window, 2s step)
- **Whisper Transcription**: Uses `faster-whisper` on CPU + int8 quantization (optimized for Apple Silicon)
- **Default language**: fixed English (`--language en`) with optional `auto` for multilingual meetings
- **Low-level noise gate**: `--min-rms` threshold skips decode on very quiet windows to reduce hallucinations
- **Deduplication**: Tracks `last_emitted_t1` to prevent duplicate segments across windows
- **Absolute Timestamps**: Converts Whisper's relative timestamps to absolute recording time

### Output Generation

1. **Real-time**: WebSocket broadcasts (`ws://127.0.0.1:8765`) for live caption display
2. **Storage**: JSONL transcript log (`transcript.jsonl`) with structured segment records
3. **Post-processing** (on Ctrl+C):
   - Renders bilingual transcripts with `MM:SS` timestamps
   - Chunked summarization via Doubao (handles long transcripts by splitting at ~12k chars)
   - HTML report generation combining transcript + summary

### Language Handling

The system supports both fixed language and auto-detection:

**Default (recommended for English meetings):**
- `--language en`
- Faster and more stable than auto-detect for single-language meetings

**Auto-detection mode (`--language auto`):**
- Whisper detects language per window
- Supports English, Chinese, and many other languages
- Useful for mixed-language meetings

**Console Output**:
- Displays language tags in real-time: `[00:05][EN] Hello everyone`
- Language tags: `[EN]` for English, `[ZH]` for Chinese, etc.
- Shows transcribed text in the detected language

**Mixed Transcript**:
- `transcript-mixed.md`: ALL content with language tags [EN]/[ZH] (original languages)
- Shows what was actually said in each language
- No translation delays during post-processing

**Summary Generation**:
- Generates BOTH English and Chinese summaries from mixed transcript
- Language-specific prompts handle mixed-language input intelligently
- LLM understands and summarizes content across all languages
- Each summary saved separately: `summary-en.md` and `summary-zh.md`

**Data Storage**:
- JSONL stores complete language metadata per segment
- Fields: `text` (original), `detected_language`, `language_confidence`
- WebSocket broadcasts include language information for real-time UI

**File Output**:
- `transcript-mixed.md`: All content with language tags
- `summary-en.md`: English summary of full meeting
- `summary-zh.md`: Chinese summary of full meeting
- `report-en.html`: English HTML report (shows mixed transcript)
- `report-zh.html`: Chinese HTML report (shows mixed transcript)

### File Structure

```
records/
└── YYYY-MM-DD/
    └── HH-MM-SS_zoom/
        ├── meta.json             # Run configuration (includes language + auto_detect_language flag)
        ├── transcript.jsonl      # Raw segment records with detected_language, language_confidence
        ├── transcript-mixed.md   # All content with language tags
        ├── summary-en.md         # English summary of full meeting
        ├── summary-zh.md         # Chinese summary of full meeting
        ├── report-en.html        # English HTML report
        ├── report-zh.html        # Chinese HTML report
        └── audio.wav             # Optional raw audio (--save-wav)
```

## Key Implementation Details

### Startup Performance

The application uses **lazy initialization** for optimal startup time:

**Startup sequence (~2-3 seconds to start capturing)**:
1. Configuration display and validation (~1s)
2. Audio device resolution (~0.5s)
3. WebSocket server start (~0.5s)
4. **Audio capture begins immediately** (~2s total)
5. Whisper model loads in background (parallel with audio buffering)

**Key optimizations**:
- **Audio capture starts immediately** - no waiting for model load
- **Parallel model loading** - Whisper loads while audio buffers
- **Progress feedback** - shows buffering status during model load
- **Buffered backfill** - first decode pass includes buffered audio captured during model load
- **Default 'medium' model** - higher quality, with 2s/1s window-step tuned for lower-latency real-time output

**Model loading times** (background, doesn't block startup):

*First-time download (includes internet download):*
- `tiny`: ~8-10 seconds
- `base`: ~20-25 seconds
- `small`: ~30-40 seconds
- `medium`: ~45-60 seconds (environment dependent)

*Cached model loading (typical performance):*
- `tiny`: ~0.5-1 second
- `base`: ~0.5-1 second
- `small`: ~1-2 seconds
- `medium`: ~2-5 seconds (default)
- `large`: ~4-5 seconds

**First-time use**: Models auto-download from Hugging Face on first use and are cached at `~/.cache/huggingface/hub/`.

**Performance tip**: If low-latency matters more than quality, switch to `--whisper-model small`.

### Device Selection Strategy

The `resolve_input_device` function supports three patterns:
- Numeric index (e.g., `--device 7`)
- "blackhole" keyword for auto-detection
- Substring matching (e.g., `--device "MacBook Microphone"`)

Device selection auto-detects channel count and sample rate. Virtual devices like BlackHole may return 0 for sample rate; the code defaults to 48kHz in such cases.

### Audio Quality & Resampling

- Uses scipy's `resample_poly` for high-quality polyphase filtering
- Selects strongest input channel to avoid destructive channel averaging
- Optional `--input-gain` boosts quiet virtual-device signals before decode
- Resamples to 16kHz (Whisper's expected input rate)
- RMS level diagnostics can be enabled during capture via `--verbose-levels`
- `--min-rms` can suppress noise-only windows

### Transcription Deduplication

Critical for sliding window approach:
- Each window's segments get timestamps relative to window start
- Converted to absolute time: `abs_t = t_window_start + seg.start`
- Only emit if `abs_t1 > last_emitted_t1 + 0.15s` tolerance
- Prevents duplicate text when windows overlap

### Doubao Integration

**Summary Generation**:
Summary generation uses a two-phase approach with language-specific prompts:
1. **Chunking**: Splits long transcripts (~12k char chunks) with newline-aware splitting
2. **Partial Summaries**: Each chunk gets key points, decisions, action items
   - English: "You write concise, actionable meeting minutes in English."
   - Chinese: "你是一个专业的会议纪要撰写助手。请用简体中文撰写简洁、可操作的会议纪要。"
3. **Consolidation**: If multiple chunks, a second pass merges into final format:
   - English: TL;DR (3-5 bullets), Decisions, Action Items (table), Open Questions/Risks
   - Chinese: 核心要点, 决策事项, 行动项（表格）, 待解决问题/风险

### HTML Report Generation

The `REPORT_TEMPLATE` is a Jinja2 template that generates a styled HTML report combining:
- Meeting metadata (date, device, model)
- Summary section with blue highlight
- Full transcript with gray highlight
- Responsive design with clean typography

## Testing Considerations

When testing or debugging:
- Use `--save-wav` to capture raw audio for replay/analysis
- Use `monitor_levels.py` before meetings to confirm non-zero capture levels
- Use `--verbose-levels` if you need per-window RMS diagnostics in console output
- Use `--list-devices` if device indexes change (headset plug/unplug)
- WebSocket port 8765 must be available
- Doubao API requires valid auth token in `DOUBAO_API_KEY`


<claude-mem-context>
# Recent Activity

<!-- This section is auto-generated by claude-mem. Edit content outside the tags. -->

*No recent activity*
</claude-mem-context>
