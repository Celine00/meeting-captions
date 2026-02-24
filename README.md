# Meeting Captions

Real-time meeting transcription system that captures audio, transcribes it using Faster-Whisper, and generates meeting summaries using Doubao (Volcengine Ark) LLM.

## Features

- **Fast startup**: Audio capture begins in ~2-3 seconds (model loads in background)
- **Fast shutdown**: No translation delays - shutdown completes in seconds, not minutes
- Real-time audio capture from virtual audio devices (e.g., BlackHole)
- Live transcription using Faster-Whisper (optimized for Apple Silicon)
- **Automatic language detection**: Supports English, Chinese, and 90+ languages without manual configuration
- **Mixed-language meetings**: Automatically detects and transcribes each language natively
- Input level diagnostics with `monitor_levels.py`
- WebSocket-based live caption broadcasting with language tags
- **Bilingual summaries**: Generates both English and Chinese summaries from mixed-language content
- HTML report generation with mixed transcript and summaries

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

Recommended routing (Zoom/Teams/Meet):
1. In **Audio MIDI Setup**, create a Multi-Output Device that includes:
   - `BlackHole 2ch`
   - Your headphones/speakers
2. In your meeting app, set **Speaker/Output** to this Multi-Output Device.
3. In this app, capture from `--device blackhole`.

## Usage

### List Available Audio Devices

```bash
python app.py --list-devices
```

### Verify Audio Routing (Recommended Before Meetings)

```bash
# 10-15s quick check: you should see non-zero RMS while others are speaking
python monitor_levels.py --device blackhole --seconds 15 --channels 2
```

If RMS stays near zero, meeting audio is not routed to the capture device yet.

### Basic Usage

```bash
# Run with default production config
# (blackhole + medium + English + save wav + beam 1 + min-rms 0.01 + 2s/1s window-step)
python app.py

# Explicit device (same defaults otherwise)
python app.py --device blackhole

# Or specify device by index
python app.py --device 4

# You can also use your own aggregate device name if it is correctly routed
python app.py --device MeetingAgg
```

### Validated Real-Time English Command

```bash
python app.py \
  --device blackhole \
  --whisper-model medium \
  --save-wav \
  --language en \
  --beam-size 1 \
  --min-rms 0.01 \
  --window-s 2 \
  --step-s 1
```

### Advanced Options

```bash
python app.py \
  --device blackhole \
  --whisper-model small \
  --compute-type int8 \
  --language en \
  --beam-size 1 \
  --min-rms 0.01 \
  --input-gain 1.0 \
  --window-s 2.0 \
  --step-s 1.0 \
  --save-wav \
  --no-summary
```

**Options:**
- `--device` - Audio input device (index, name substring, or "blackhole")
- `--whisper-model` - Whisper model size: `tiny`, `base`, `small`, `medium` (default), `large`
  - **Recommendation**: Keep `medium` for better quality in meetings
  - Use `small` if you want lower CPU usage and faster first token
- `--compute-type` - Quantization (int8, int16, float16, float32)
- `--language` - Whisper language (e.g. `auto`, `en`, `zh`)
- `--beam-size` - Decoder beam size (`1` is fastest)
- `--min-rms` - Skip decode when input RMS is below threshold (helps avoid low-level hallucinations)
- `--input-gain` - Linear gain before Whisper for quiet virtual-device input
- `--no-vad` - Disable Whisper VAD filter (use with caution in noisy environments)
- `--window-s` - Transcription window size in seconds (default: 2.0)
- `--step-s` - Step size between transcriptions (default: 1.0)
- `--verbose-levels` - Print per-window `[LEVEL]` RMS diagnostics (off by default)
- `--save-wav` - Save raw audio for debugging
- `--no-save-wav` - Disable wav saving (enabled by default)
- `--no-summary` - Skip LLM summary generation
- `--ws-port` - WebSocket port (default: 8765)

### Stop Recording

Press `Ctrl+C` once for graceful stop. The system will automatically:
1. Analyze language distribution
2. Save mixed transcript with language tags
3. Generate summaries in both languages from mixed transcript
4. Create HTML reports in both languages
5. Save the WAV file (if `--save-wav` was specified)

Press `Ctrl+C` a second time within 1.5 seconds for fast shutdown:
1. Stop quickly after core cleanup
2. Keep transcript output on disk
3. Skip language analysis, summary generation, and HTML report generation

Graceful stop may take longer because it generates summaries/reports. Use double `Ctrl+C` for a quick exit during testing.

## Output

Recordings are saved in `records/YYYY-MM-DD/HH-MM-SS_zoom/`.
Default uses fixed English (`--language en`). Use `--language auto` for mixed-language meetings.

```
records/2026-02-01/18-02-31_zoom/
├── meta.json             # Run configuration (includes language, beam_size, min_rms, auto_detect_language)
├── transcript.jsonl      # Raw segment records with detected_language, language_confidence
├── transcript-mixed.md   # All segments with [EN]/[ZH] language tags (original languages)
├── summary-en.md         # English summary of full meeting
├── summary-zh.md         # Chinese summary of full meeting
├── report-en.html        # English HTML report (shows mixed transcript)
├── report-zh.html        # Chinese HTML report (shows mixed transcript)
└── audio.wav             # Raw audio (if --save-wav used)
```

### File Descriptions

- **meta.json**: Run configuration (device/model/lang/threshold settings)
- **transcript.jsonl**: Raw segment records with:
  - `text`: Transcribed text in detected language
  - `detected_language`: Language code (e.g., "en", "zh")
  - `language_confidence`: Confidence score (0.0-1.0)
  - `t0`, `t1`: Absolute timestamps
- **transcript-mixed.md**: All segments with language tags showing what was actually said
  - Example: `- **00:05** [EN] Hello everyone`
  - Example: `- **00:12** [ZH] 今天我们讨论项目`
- **summary-en.md**: LLM-generated English meeting notes from mixed transcript
- **summary-zh.md**: LLM-generated Chinese meeting notes from mixed transcript
- **report-en.html**: Combined English HTML report with mixed transcript
- **report-zh.html**: Combined Chinese HTML report with mixed transcript

## WebSocket API

Live captions are broadcast on `ws://127.0.0.1:8765` (default port).

### Message Format

**Caption with language detection**:
```json
{
  "type": "caption",
  "t0": 5.2,
  "t1": 8.1,
  "text": "Hello, this is a test.",
  "detected_language": "en",
  "language_confidence": 0.97
}
```

**Example Chinese caption**:
```json
{
  "type": "caption",
  "t0": 12.5,
  "t1": 15.3,
  "text": "今天我们讨论项目进度。",
  "detected_language": "zh",
  "language_confidence": 0.95
}
```

Fields:
- `text`: Transcribed text in the detected language
- `detected_language`: ISO language code (e.g., "en", "zh", "es")
- `language_confidence`: Detection confidence (0.0-1.0)

## Performance

**Startup time**: ~2-3 seconds until audio capture begins
- Audio capture starts immediately
- Whisper model loads in background (shows progress: "Loading... buffered X.Xs")
- First transcription appears after model loads + window fills (~10-15 seconds total)

**Model load times** (background, doesn't block startup):

*First-time download (downloads model from internet):*
- `tiny`: ~8-10s download + load
- `base`: ~20-25s download + load
- `small`: ~30-40s download + load
- `medium`: ~45-60s download + load (environment dependent, default)

*Subsequent runs (cached models):*
- `tiny`: ~0.5-1s load ⚡
- `base`: ~0.5-1s load
- `small`: ~1-2s load
- `medium`: ~2-5s load (default)
- `large`: ~4-5s load

**Recommendation**: Use default `medium` for better quality. If latency matters more, switch to `small`.

## Troubleshooting

- **No audio captured**: Check device selection with `--list-devices`
- **Low RMS levels**: Verify audio routing through BlackHole/virtual device with `monitor_levels.py`
- **Hallucinated text (e.g., “Thanks for watching!”)**: Usually low-level noise instead of meeting speech. Use `--language en`, raise `--min-rms`, and fix system audio routing.
- **`BlackHole` WAV is silent**: Meeting app output is not set to your Multi-Output device that includes BlackHole.
- **Doubao API errors**: Verify `DOUBAO_API_KEY` in `.env` file
- **Whisper slow on M1**: Use `--compute-type int8` (default)
- **Slow startup**: Model is loading in background; audio is being captured. First transcription appears once model loads (see "Performance" section)
