#!/usr/bin/env python3
"""Download every model the production stack uses, so the first service start
does not spend minutes fetching them.

    ./venv/bin/python scripts/prefetch_models.py

Loads, under the environment scripts/run_production.sh exports: NLLB-200 (once,
shared), the voice for every language enabled in config/settings.yaml (or the
page's languages when there is no settings.yaml), and the streaming ASR model
by starting the NeMo server once and letting it exit. Everything lands in the
same caches the service uses.
"""
from __future__ import annotations

import os
import struct
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))


def main() -> int:
    from test_pipeline_config import production_env, settings_languages, site_languages
    os.environ.update(production_env())
    langs = settings_languages() or site_languages()
    print(f"languages: {', '.join(langs)}")

    from pipeline.translation import TranslationService
    from pipeline.tts import TTSService
    for code in langs:
        print(f"\n== {code}: translation")
        t = TranslationService(source_language="en", target_language=code,
                               download_root=str(ROOT / "models" / "translation"))
        t.load()
        t.unload()
        print(f"== {code}: voice")
        v = TTSService(language=code, download_root=str(ROOT / "models" / "tts"))
        v.load()
        v.unload()

    print("\n== streaming ASR (parakeet-unified-en-0.6b via the NeMo server)")
    python = os.environ.get("UNIFIED_PYTHON", os.path.expanduser("~/nemo-venv/bin/python"))
    if not Path(python).exists():
        print(f"   skipped: {python} not found (create the NeMo venv first)")
        return 0
    os.environ.setdefault("HF_HOME", str(ROOT / "models" / "asr" / "parakeet"))
    proc = subprocess.Popen([python, str(ROOT / "src" / "pipeline" / "unified_asr_server.py")],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    ready = proc.stdout.readline()
    if b'"ready": true' not in ready:
        print(f"   server did not come up: {ready.decode(errors='replace').strip()}")
        return 1
    proc.stdin.write(struct.pack("<I", 0xFFFFFFFF))     # flush = end of stream
    proc.stdin.flush()
    proc.stdout.readline()
    proc.stdin.close()
    proc.wait(timeout=60)
    print("   model present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
