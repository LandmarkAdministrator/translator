"""Reading and writing the settings a service operator needs to change.

Two files, both edited from /admin so the sound room does not need SSH:

  config/schedule.conf   service windows and the archive drain lead time
  config/settings.yaml   audio input and per-language output routing

Everything here validates before writing and writes atomically. A malformed
schedule means no services until someone notices, and a wrong audio device
means a silent auditorium, so nothing is written that has not been checked.
"""
from __future__ import annotations

import os
import re
import shutil
from pathlib import Path
from typing import Any, Optional

PROJECT = Path(__file__).resolve().parent.parent.parent
SCHEDULE = Path(os.environ.get("TRANSLATOR_SCHEDULE_CONF", PROJECT / "config" / "schedule.conf"))
SETTINGS = Path(os.environ.get("TRANSLATOR_SETTINGS_YAML", PROJECT / "config" / "settings.yaml"))

DAYS = ["", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
HHMM = re.compile(r"^([01]\d|2[0-3]):([0-5]\d)$")
MAX_WINDOWS = 20


# --------------------------------------------------------------- schedule
def read_schedule() -> dict:
    windows, drain = [], 30
    try:
        for line in SCHEDULE.read_text().splitlines():
            parts = line.split()
            if not parts or parts[0].startswith("#"):
                continue
            if parts[0] == "drain_min" and len(parts) >= 2 and parts[1].isdigit():
                drain = int(parts[1])
            elif parts[0] == "window" and len(parts) >= 4:
                day, start, end = parts[1], parts[2], parts[3]
                if day.isdigit() and 1 <= int(day) <= 7 and HHMM.match(start) and HHMM.match(end):
                    windows.append({"day": int(day), "start": start, "end": end})
    except FileNotFoundError:
        pass
    return {"drain_min": drain, "windows": windows}


def validate_schedule(data: Any) -> Optional[str]:
    """Return an error message, or None if the schedule is safe to write."""
    if not isinstance(data, dict):
        return "Schedule must be an object."
    windows = data.get("windows")
    if not isinstance(windows, list) or not windows:
        return "At least one service window is required."
    if len(windows) > MAX_WINDOWS:
        return f"Too many windows (limit {MAX_WINDOWS})."
    drain = data.get("drain_min", 30)
    if not isinstance(drain, int) or not (0 <= drain <= 180):
        return "Drain minutes must be a whole number between 0 and 180."
    seen = []
    for w in windows:
        if not isinstance(w, dict):
            return "Each window must be an object."
        day, start, end = w.get("day"), str(w.get("start", "")), str(w.get("end", ""))
        if not isinstance(day, int) or not (1 <= day <= 7):
            return "Day must be 1 (Monday) to 7 (Sunday)."
        if not HHMM.match(start) or not HHMM.match(end):
            return f"Times must be HH:MM in 24-hour form (got {start!r}–{end!r})."
        s, e = _min(start), _min(end)
        if e <= s:
            return f"{DAYS[day]} {start}–{end}: the end must come after the start."
        for od, os_, oe in seen:
            if od == day and s < oe and os_ < e:
                return (f"{DAYS[day]} {start}–{end} overlaps another window "
                        f"that day; merge them instead.")
        seen.append((day, s, e))
    return None


def write_schedule(data: dict) -> None:
    err = validate_schedule(data)
    if err:
        raise ValueError(err)
    lines = [
        "# Service schedule — edited from /admin. See scripts/ops/"
        "translate-window-check.sh, which reads this every five minutes.",
        "#   window DAY HH:MM HH:MM    DAY is 1=Mon .. 7=Sun",
        "#   drain_min N               stop claiming archive jobs this long before a window",
        "",
        f"drain_min {int(data.get('drain_min', 30))}",
        "",
    ]
    for w in sorted(data["windows"], key=lambda w: (w["day"], _min(str(w["start"])))):
        lines.append(f"# {DAYS[w['day']]}")
        lines.append(f"window {w['day']} {w['start']} {w['end']}")
    _atomic_write(SCHEDULE, "\n".join(lines) + "\n")


def _min(hhmm: str) -> int:
    h, m = hhmm.split(":")
    return int(h) * 60 + int(m)


# --------------------------------------------------------------- devices
def list_audio_devices() -> dict:
    """Enumerate inputs and outputs. Never raises.

    A device already open by another process can vanish from enumeration
    entirely — the Behringer does exactly that while translation is running —
    so the configured names are always returned too, marked present or not.
    """
    out: dict = {"inputs": [], "outputs": [], "error": None}
    try:
        import sounddevice as sd
        for i, d in enumerate(sd.query_devices()):
            entry = {"index": i, "name": d["name"]}
            if d["max_input_channels"] > 0:
                out["inputs"].append(entry)
            if d["max_output_channels"] > 0:
                out["outputs"].append({**entry, "channels": d["max_output_channels"]})
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
    return out


def read_audio() -> dict:
    cfg = {"input_device": "", "languages": []}
    try:
        import yaml
        raw = yaml.safe_load(SETTINGS.read_text()) or {}
        cfg["input_device"] = raw.get("input_device", "")
        for l in raw.get("languages", []):
            cfg["languages"].append({
                "code": l.get("code"), "name": l.get("name"),
                "output_device": l.get("output_device", "default"),
                "output_channel": l.get("output_channel"),
                "enabled": bool(l.get("enabled", False)),
            })
    except Exception as e:
        cfg["error"] = f"{type(e).__name__}: {e}"
    return cfg


def validate_audio(data: Any) -> Optional[str]:
    if not isinstance(data, dict):
        return "Audio settings must be an object."
    if not str(data.get("input_device", "")).strip():
        return "An input device is required."
    langs = data.get("languages")
    if not isinstance(langs, list) or not langs:
        return "At least one language is required."
    claimed: dict = {}
    for l in langs:
        if not isinstance(l, dict) or not l.get("code"):
            return "Each language needs a code."
        ch = l.get("output_channel")
        if ch not in (None, 0, 1):
            return f"{l['code']}: channel must be left, right, or both."
        dev = str(l.get("output_device", "")).strip()
        if not dev:
            return f"{l['code']}: an output device is required."
        if not l.get("enabled"):
            continue
        # Two languages on the same device and channel would talk over each
        # other; 'both' claims the whole device.
        key = (dev, ch)
        for (odev, och), other in claimed.items():
            if odev != dev:
                continue
            if och == ch or och is None or ch is None:
                return (f"{l['code']} and {other} would both play on "
                        f"{dev}" + (f" channel {ch}" if ch is not None else "") +
                        " — give one of them a different device or channel.")
        claimed[key] = l["code"]
    return None


def write_audio(data: dict) -> None:
    err = validate_audio(data)
    if err:
        raise ValueError(err)
    import yaml
    text = SETTINGS.read_text()
    raw = yaml.safe_load(text) or {}
    raw["input_device"] = data["input_device"]
    by_code = {l["code"]: l for l in data["languages"]}
    for l in raw.get("languages", []):
        upd = by_code.get(l.get("code"))
        if upd:
            l["output_device"] = upd["output_device"]
            l["output_channel"] = upd["output_channel"]
            l["enabled"] = bool(upd["enabled"])
    # Edit values in place so the file's comments survive: the routing notes
    # (why Russian is on the onboard jack, why devices are matched by name)
    # live between the entries, and yaml.safe_dump would erase them. The
    # patched text must load back to exactly what we meant to write; if it
    # does not — an unfamiliar layout — fall back to a clean dump.
    patched = _patch_yaml_values(text, raw)
    if patched is None or yaml.safe_load(patched) != raw:
        patched = yaml.safe_dump(raw, sort_keys=False, allow_unicode=True)
    _atomic_write(SETTINGS, patched)


_TOP_KEY = re.compile(r"^(input_device):\s*(.*?)\s*$")
_ENTRY = re.compile(r"^(\s*)-\s+code:\s*(\S+)\s*$")
_FIELD = re.compile(r"^(\s+)(output_device|output_channel|enabled):\s*(.*?)\s*$")


def _patch_yaml_values(text: str, raw: dict) -> Optional[str]:
    """Rewrite just the values write_audio changes, line by line, keeping every
    other line (comments included) byte for byte. Understands the one layout
    settings.yaml has always had — a flat mapping with a `languages` list whose
    items start `- code: xx` — and returns None for anything else."""
    import json
    langs = {l.get("code"): l for l in raw.get("languages", [])}
    out, code = [], None
    for line in text.splitlines():
        m = _TOP_KEY.match(line)
        if m:
            code = None
            out.append(f"input_device: {json.dumps(raw['input_device'], ensure_ascii=False)}")
            continue
        m = _ENTRY.match(line)
        if m:
            code = m.group(2).strip("'\"")
            out.append(line)
            continue
        m = _FIELD.match(line) if code in langs else None
        if m:
            value = langs[code][m.group(2)]
            out.append(f"{m.group(1)}{m.group(2)}: {json.dumps(value, ensure_ascii=False)}")
            continue
        if line and not line[0].isspace() and not line.startswith("#") and not line.startswith("-"):
            code = None            # another top-level key ends the list
        out.append(line)
    return "\n".join(out) + "\n"


def _atomic_write(path: Path, text: str) -> None:
    """Write via a temp file and keep one backup, so a failure mid-write cannot
    leave the service with a truncated schedule or device list."""
    if path.exists():
        shutil.copy2(path, path.with_suffix(path.suffix + ".bak"))
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)
