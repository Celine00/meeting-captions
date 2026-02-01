#!/usr/bin/env python3
"""
Benchmark Whisper model loading times to help choose the right model.

Usage:
    python3 benchmark_models.py
"""

import time
from faster_whisper import WhisperModel

models = ["tiny", "base", "small"]

print("Benchmarking Whisper model load times...")
print("=" * 60)

results = []

for model_name in models:
    print(f"\nLoading '{model_name}' model...")
    start = time.time()

    try:
        model = WhisperModel(model_name, device="cpu", compute_type="int8")
        elapsed = time.time() - start
        results.append((model_name, elapsed))
        print(f"  ✓ Loaded in {elapsed:.2f} seconds")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        results.append((model_name, None))

print("\n" + "=" * 60)
print("\nSummary:")
print("-" * 60)
for model_name, elapsed in results:
    if elapsed is not None:
        print(f"  {model_name:10s}: {elapsed:6.2f}s")
    else:
        print(f"  {model_name:10s}: FAILED")

print("\nRecommendation:")
print("-" * 60)
best = min((m, t) for m, t in results if t is not None)
print(f"  Fastest model: '{best[0]}' ({best[1]:.2f}s)")
print(f"  Use: python3 app.py --whisper-model {best[0]}")
print("\nNote: Larger models provide better accuracy but take longer to load.")
print("      'tiny' is great for most meetings, 'base' for better accuracy.")
