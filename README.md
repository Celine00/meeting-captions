# Meeting Captions

Real-time meeting transcription system that captures audio, transcribes it using Faster-Whisper, and generates meeting summaries using Doubao (Volcengine Ark) LLM.

## Features

- Real-time audio capture from virtual audio devices (e.g., BlackHole)
- Live transcription using Faster-Whisper (optimized for Apple Silicon)
- WebSocket-based live caption broadcasting
- Automatic meeting summarization with action items and decisions
- HTML report generation with transcript and summary

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
# Use BlackHole device (auto-detected)
python app.py --device blackhole

# Or specify device by name
python app.py --device MeetingAgg

# Or specify device by index
python app.py --device 8
```

### Advanced Options

```bash
python app.py \
  --device blackhole \
  --whisper-model small \
  --compute-type int8 \
  --window-s 8.0 \
  --step-s 1.0 \
  --save-wav
```

**Options:**
- `--device` - Audio input device (index, name substring, or "blackhole")
- `--whisper-model` - Whisper model size (tiny, base, small, medium, large)
- `--compute-type` - Quantization (int8, int16, float16, float32)
- `--window-s` - Transcription window size in seconds (default: 8.0)
- `--step-s` - Step size between transcriptions (default: 1.0)
- `--save-wav` - Save raw audio for debugging
- `--ws-port` - WebSocket port (default: 8765)

### Stop Recording

Press `Ctrl+C` to stop. The system will automatically:
1. Save the transcript
2. Generate a summary using Doubao
3. Create an HTML report
4. Save the WAV file (if `--save-wav` was specified)

## Output

Recordings are saved in `records/YYYY-MM-DD/HH-MM-SS_zoom/`:

```
records/2026-02-01/18-02-31_zoom/
├── meta.json          # Run configuration
├── transcript.jsonl   # Raw segment records
├── transcript.md      # Human-readable transcript
├── summary.md         # LLM-generated meeting notes
├── report.html        # Combined HTML report
└── audio.wav          # Raw audio (if --save-wav used)
```

## WebSocket API

Live captions are broadcast on `ws://127.0.0.1:8765` (default port).

Message format:
```json
{
  "type": "caption",
  "t0": 5.2,
  "t1": 8.1,
  "text": "Hello, this is a test."
}
```

## Troubleshooting

- **No audio captured**: Check device selection with `--list-devices`
- **Low RMS levels**: Verify audio routing through BlackHole/virtual device
- **Doubao API errors**: Verify `DOUBAO_API_KEY` in `.env` file
- **Whisper slow on M1**: Use `--compute-type int8` (default)
