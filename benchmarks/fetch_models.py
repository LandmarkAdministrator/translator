#!/usr/bin/env python
"""Download every benchmark candidate once, into the repo's model cache.

Downloading on one machine and copying over the LAN saves pulling ~80 GB
through the church's internet twice.
"""
import sys, time
from huggingface_hub import snapshot_download

CACHE = "models/translation"
REPOS = [
    ("opus-es", "Helsinki-NLP/opus-mt-en-es"),
    ("opus-ht", "Helsinki-NLP/opus-mt-en-ht"),
    ("opus-ru", "Helsinki-NLP/opus-mt-en-ru"),
    ("madlad-3b", "google/madlad400-3b-mt"),
    ("qwen3-8b", "Qwen/Qwen3-8B"),
    ("aya-101", "CohereLabs/aya-101"),
    ("mistral-nemo-12b", "mistralai/Mistral-Nemo-Instruct-2407"),
]
for name, repo in REPOS:
    t0 = time.time()
    print(f">>> {name}  {repo}", flush=True)
    try:
        # Weights only: skip the duplicate .bin/.h5/.msgpack copies of the
        # same checkpoint, which double or triple the download for nothing.
        p = snapshot_download(repo, cache_dir=CACHE,
                              ignore_patterns=["*.bin", "*.h5", "*.msgpack",
                                               "*.onnx", "*consolidated*"])
        print(f"    ok in {time.time() - t0:.0f}s -> {p}", flush=True)
    except Exception as e:
        print(f"    FAILED {type(e).__name__}: {e}", flush=True)
print("all done", flush=True)
