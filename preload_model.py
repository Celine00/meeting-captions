#!/usr/bin/env python3
"""
Pre-load and cache Whisper model to improve startup time.

Usage:
    python3 preload_model.py base     # Pre-load base model
    python3 preload_model.py small    # Pre-load small model
"""

import sys
from faster_whisper import WhisperModel

def preload_model(model_name: str = "base", compute_type: str = "int8"):
    """Pre-load and cache a Whisper model"""
    print(f"Pre-loading Whisper model '{model_name}' with compute_type '{compute_type}'...")
    print("This will download and cache the model for faster subsequent startups.")
    print("")

    model = WhisperModel(model_name, device="cpu", compute_type=compute_type)
    print(f"✓ Model '{model_name}' loaded and cached successfully!")
    print(f"  Subsequent runs will use the cached model.")

    # Test transcription to ensure everything works
    print("\nTesting model with sample audio...")
    import numpy as np
    # 1 second of silence at 16kHz
    test_audio = np.zeros(16000, dtype=np.float32)
    segments, info = model.transcribe(test_audio, language="en")
    print(f"✓ Model test successful (language: {info.language})")

    return model

if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "base"
    compute_type = sys.argv[2] if len(sys.argv) > 2 else "int8"
    preload_model(model_name, compute_type)
