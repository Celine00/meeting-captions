# Meeting Captions

Real-time meeting transcription system that captures audio, transcribes it using Faster-Whisper, and generates meeting summaries using Doubao (Volcengine Ark) LLM.

## Features

- **Fast startup**: Audio capture begins in ~2-3 seconds (model loads in background)
- Real-time audio capture from virtual audio devices (e.g., BlackHole)
- Live transcription using Faster-Whisper (optimized for Apple Silicon)
- **Bilingual support**: English transcription with optional Chinese translation
- WebSocket-based live caption broadcasting (English or Chinese)
- **Automatic bilingual summaries**: Both English and Chinese meeting notes
- HTML report generation with transcript and summary in both languages

## Setup

### 1. Activate Virtual Environment

```bash
pyenv activate meeting-captions
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your Doubao API key
# Get your API key from: https://console.volcengine.com/ark
```

Edit `.env` and set:
```
DOUBAO_API_KEY=your_actual_api_key_here
```

### 4. Set Up Audio Routing

For macOS, install [BlackHole](https://existential.audio/blackhole/) to route system audio:

```bash
brew install blackhole-2ch
```

Configure a Multi-Output Device in Audio MIDI Setup to hear audio while recording.

## Usage

### List Available Audio Devices

```bash
python app.py --list-devices
```

### Basic Usage

```bash
# Use BlackHole device with English captions (default)
python app.py --device blackhole

# Use Chinese captions (real-time translation)
python app.py --device blackhole --caption-lang zh

# Or specify device by name
python app.py --device MeetingAgg

# Or specify device by index
python app.py --device 8
```

### Advanced Options

```bash
python app.py \
  --device blackhole \
  --caption-lang zh \
  --whisper-model small \
  --compute-type int8 \
  --window-s 8.0 \
  --step-s 1.0 \
  --save-wav \
  --no-summary
```

**Options:**
- `--device` - Audio input device (index, name substring, or "blackhole")
- `--caption-lang` - Caption language: `en` (English, default) or `zh` (Chinese with translation)
- `--whisper-model` - Whisper model size: `tiny`, `base` (default, fast), `small`, `medium`, `large`
  - **Recommendation**: Use `base` for best startup speed (~3-5s load) with good accuracy
  - Use `small` for better accuracy (~8-10s load)
- `--compute-type` - Quantization (int8, int16, float16, float32)
- `--window-s` - Transcription window size in seconds (default: 8.0)
- `--step-s` - Step size between transcriptions (default: 1.0)
- `--save-wav` - Save raw audio for debugging
- `--no-summary` - Skip LLM summary generation
- `--ws-port` - WebSocket port (default: 8765)

### Stop Recording

Press `Ctrl+C` to stop. The system will automatically:
1. Save transcripts (English + Chinese if `--caption-lang zh`)
2. Generate bilingual summaries using Doubao
3. Create HTML reports in both languages
4. Save the WAV file (if `--save-wav` was specified)

## Output

Recordings are saved in `records/YYYY-MM-DD/HH-MM-SS_zoom/` with bilingual support:

### Default (English captions: `--caption-lang en`)
```
records/2026-02-01/18-02-31_zoom/
├── meta.json             # Run configuration
├── transcript.jsonl      # Raw segment records (English only)
├── transcript-en.md      # English transcript
├── summary-en.md         # English meeting summary
├── summary-zh.md         # Chinese meeting summary
├── report-en.html        # English HTML report
├── report-zh.html        # Chinese HTML report
└── audio.wav             # Raw audio (if --save-wav used)
```

### With Chinese captions (`--caption-lang zh`)
```
records/2026-02-01/18-02-31_zoom/
├── meta.json             # Run configuration (includes caption_lang)
├── transcript.jsonl      # Raw records with both text and text_zh
├── transcript-en.md      # English transcript
├── transcript-zh.md      # Chinese transcript (translated)
├── summary-en.md         # English meeting summary
├── summary-zh.md         # Chinese meeting summary
├── report-en.html        # English HTML report
├── report-zh.html        # Chinese HTML report
└── audio.wav             # Raw audio (if --save-wav used)
```

### File Descriptions

- **meta.json**: Run configuration including caption language setting
- **transcript.jsonl**: Raw segment records with timestamps
  - With `--caption-lang en`: Contains `text` field (English)
  - With `--caption-lang zh`: Contains both `text` (English) and `text_zh` (Chinese) fields
- **transcript-en.md**: Human-readable English transcript with timestamps
- **transcript-zh.md**: Chinese transcript (only if `--caption-lang zh`)
- **summary-en.md**: LLM-generated English meeting notes (always)
- **summary-zh.md**: LLM-generated Chinese meeting notes (always)
- **report-en.html**: Combined English HTML report
- **report-zh.html**: Combined Chinese HTML report

## WebSocket API

Live captions are broadcast on `ws://127.0.0.1:8765` (default port).

### Message Format

**English captions** (`--caption-lang en`):
```json
{
  "type": "caption",
  "t0": 5.2,
  "t1": 8.1,
  "text": "Hello, this is a test.",
  "text_zh": null,
  "display_lang": "en"
}
```

**Chinese captions** (`--caption-lang zh`):
```json
{
  "type": "caption",
  "t0": 5.2,
  "t1": 8.1,
  "text": "Hello, this is a test.",
  "text_zh": "你好，这是一个测试。",
  "display_lang": "zh"
}
```

The `display_lang` field indicates which language should be displayed to the user. Both `text` (original English) and `text_zh` (Chinese translation) are provided when available, allowing clients to choose which to display or show both.

## Performance

**Startup time**: ~2-3 seconds until audio capture begins
- Audio capture starts immediately
- Whisper model loads in background (shows progress: "Loading... buffered X.Xs")
- First transcription appears after model loads + window fills (~10-15 seconds total)

**Model load times** (background, doesn't block startup):

*First-time download (downloads model from internet):*
- `tiny`: ~8-10s download + load
- `base`: ~20-25s download + load (default)
- `small`: ~30-40s download + load

*Subsequent runs (cached models):*
- `tiny`: ~0.5-1s load ⚡
- `base`: ~0.5-1s load (default, best balance)
- `small`: ~1-2s load
- `medium`: ~2-3s load
- `large`: ~4-5s load

**Recommendation**: Use default `base` model for best balance of speed + accuracy. After first download, it loads in under 1 second.

## Troubleshooting

- **No audio captured**: Check device selection with `--list-devices`
- **Low RMS levels**: Verify audio routing through BlackHole/virtual device
- **Doubao API errors**: Verify `DOUBAO_API_KEY` in `.env` file
- **Whisper slow on M1**: Use `--compute-type int8` (default)
- **Slow startup**: Model is loading in background; audio is being captured. First transcription appears once model loads (see "Performance" section)
