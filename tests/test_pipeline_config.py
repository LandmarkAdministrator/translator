#!/usr/bin/env python3
"""Every language the site offers must build a pipeline — checked before a
service, not during one.

On 2026-09-06 Russian was added to settings.yaml without RU_TTS in
scripts/run_production.sh. TTSService raised in __init__ while the pipelines
loaded and the whole service crash-looped, taking Spanish and Creole down with
it. A dry run caught it four hours before the service; this catches it in a
second, with no models loaded.

What it checks, under the environment run_production.sh exports:
  * a TranslationService and a TTSService can be constructed for every
    language enabled in config/settings.yaml and every language the web page
    offers audio for in config/site.json (the page's list is tracked in git;
    settings.yaml is per-machine and may be absent)
  * the coordinator accepts those languages
  * the audio routing and the schedule pass the admin panel's own validation

    ./venv/bin/python tests/test_pipeline_config.py
"""
from __future__ import annotations

import json
import os
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

FAIL: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'OK  ' if ok else 'FAIL'}  {name}" + (f"  — {detail}" if detail and not ok else ""))
    if not ok:
        FAIL.append(name)


def production_env() -> dict[str, str]:
    """The literal `export NAME="value"` lines of run_production.sh, plus the
    defaults of its `export NAME="${NAME:-default}"` lines."""
    env: dict[str, str] = {}
    for line in (ROOT / "scripts" / "run_production.sh").read_text().splitlines():
        m = re.match(r'^export ([A-Z_]+)="\$\{[A-Z_]+:-([^}]*)\}"\s*$', line)
        if m:
            env[m.group(1)] = m.group(2)
            continue
        m = re.match(r'^export ([A-Z_]+)="([^"$]*)"\s*$', line)
        if m:
            env[m.group(1)] = m.group(2)
    return env


def site_languages() -> dict[str, str]:
    """code -> label for every language the page offers audio for."""
    site = json.loads((ROOT / "config" / "site.json").read_text())
    return {l["code"]: l.get("label", l["code"]) for l in site["languages"] if l.get("audio")}


def settings_languages() -> dict[str, str]:
    path = ROOT / "config" / "settings.yaml"
    if not path.exists():
        return {}
    import yaml
    raw = yaml.safe_load(path.read_text()) or {}
    return {l["code"]: l.get("name", l["code"]) for l in raw.get("languages", []) if l.get("enabled")}


def main() -> int:
    env = production_env()
    print(f"environment from run_production.sh: "
          f"{', '.join(k for k in sorted(env) if k.endswith('_TTS') or k.startswith('NLLB'))}")
    os.environ.update(env)

    langs = site_languages()
    from_settings = settings_languages()
    langs.update(from_settings)
    print(f"languages: {', '.join(f'{c} ({n})' for c, n in langs.items())}"
          + ("" if from_settings else "   (no settings.yaml here; page list only)"))

    from pipeline.coordinator import PipelineConfig, TranslationCoordinator
    from pipeline.translation import TranslationService
    from pipeline.tts import TTSService

    scratch = tempfile.mkdtemp(prefix="pipeline-config-")
    print("\n1. each language resolves a translation model and a TTS voice")
    configs = []
    for code, name in langs.items():
        try:
            TranslationService(source_language="en", target_language=code,
                               download_root=f"{scratch}/translation")
            check(f"{code}: translation ({name})", True)
        except Exception as e:
            check(f"{code}: translation ({name})", False, f"{type(e).__name__}: {e}")
        try:
            t = TTSService(language=code, download_root=f"{scratch}/tts")
            backend = "mms" if t._use_mms else "kokoro" if t._use_kokoro else "piper"
            check(f"{code}: TTS -> {backend} {t._model_name}", True)
        except Exception as e:
            check(f"{code}: TTS ({name})", False,
                  f"{type(e).__name__}: {e}  [is {code.upper()}_TTS exported in run_production.sh?]")
        configs.append(PipelineConfig(language_code=code, language_name=name))

    print("\n2. the coordinator accepts the language set")
    try:
        TranslationCoordinator(languages=configs)
        check("TranslationCoordinator(...) constructs", True)
    except Exception as e:
        check("TranslationCoordinator(...) constructs", False, f"{type(e).__name__}: {e}")

    print("\n3. routing and schedule pass the admin panel's validation")
    from web import config_api
    if (ROOT / "config" / "settings.yaml").exists():
        audio = config_api.read_audio()
        err = audio.get("error") or config_api.validate_audio(audio)
        check("settings.yaml audio routing", err is None, err or "")
    else:
        print("  skip  settings.yaml absent on this machine")
    sched = config_api.read_schedule()
    err = sched.get("error") or config_api.validate_schedule(sched)
    check("schedule.conf", err is None, err or "")

    print()
    if FAIL:
        print(f"FAILED: {len(FAIL)} check(s): " + "; ".join(FAIL))
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
