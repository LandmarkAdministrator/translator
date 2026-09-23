"""Spell numbers out in words before they reach a voice that cannot read digits.

Why this exists
---------------
The MMS-TTS voices are character-level VITS models whose vocabulary is
whatever their training text contained.  `facebook/mms-tts-hat` has a
32-character vocabulary::

    " '-abcdefghijklmnoprstuvwyz|àèò—"

There are no digits in it, and no `:` `.` `,` either.  `VitsTokenizer`
normalizes input against that vocabulary and silently DROPS anything it does
not know, so "Daniel chapit 6" is spoken as "Daniel chapit" — every chapter
and verse reference loses its numbers.  (Verified 2026-09-21 by reading
`vocab.json` out of the model cache on the Translate PC, not from the docs.)

`facebook/mms-tts-rus` has the same class of gap: only a few digits survive,
so "3:16" comes out as "1".

NLLB keeps numbers as digits when it translates ("Danyèl chapit nimewo 6",
"Sòm 119 vèsè 105", "глава 6"), so the fix belongs between translation and
synthesis: rewrite digits as words in the target language, then synthesize.

Kokoro (Spanish) verbalizes digits itself and Piper goes through espeak-ng,
which does too — so this runs on the MMS path only.  Calling it for a
language with no converter returns the text unchanged.

Accuracy note
-------------
Every Creole numeral below is representable in the 32-character vocabulary
above (no `q`, no `x`, only `à è ò` as accents) — that is checked by
`tests/test_number_words.py`.  The *wording* follows standard Kreyòl
numerals and matches the four examples recorded in `TODO.md` item 8
("sis", "disèt", "san diznèf", "mil sisan onz"), but it has not been through
a native speaker yet.  Same caveat as the short-phrase dictionary in
`translate_short_dict.py` — see `TODO.md` item 1.
"""
from __future__ import annotations

import re
from typing import Callable, Optional

__all__ = ["verbalize", "ht_cardinal", "ru_cardinal", "has_converter"]


# --- Haitian Creole ---------------------------------------------------------

_HT_0_19 = [
    "zewo", "en", "de", "twa", "kat", "senk", "sis", "sèt", "uit", "nèf",
    "dis", "onz", "douz", "trèz", "katòz", "kenz", "sèz", "disèt", "dizuit",
    "diznèf",
]

# Kreyòl follows French: 70 = 60+10, 80 = 4x20, 90 = 80+10.
# Each regular decade has three stems:
#   exact   — the bare decade ("ven")
#   linking — before 1 ("vent" + "eyen")
#   joining — before 2..7 ("venn" + "de")
# 8 and 9 take the linking stem ("ventuit", "ventnèf") because the joining
# stem collides with the vowel.
_HT_DECADES = {
    20: ("ven", "vent", "venn"),
    30: ("trant", "trant", "trann"),
    40: ("karant", "karant", "karann"),
    50: ("senkant", "senkant", "senkann"),
    60: ("swasant", "swasant", "swasann"),
}

_HT_HUNDREDS = [
    "", "san", "desan", "twasan", "katsan", "senksan", "sisan", "sètsan",
    "uitsan", "nèfsan",
]


def _ht_under_100(n: int) -> str:
    if n < 20:
        return _HT_0_19[n]
    if n < 70:
        decade = (n // 10) * 10
        unit = n % 10
        exact, linking, joining = _HT_DECADES[decade]
        if unit == 0:
            return exact
        if unit == 1:
            return linking + "eyen"
        if unit in (8, 9):
            return linking + _HT_0_19[unit]
        return joining + _HT_0_19[unit]
    if n < 80:
        # 70..79 = "swasann" + 10..19
        return "swasann" + _HT_0_19[n - 60]
    # 80..99 = "katreven" + 0..19
    rest = n - 80
    if rest == 0:
        return "katreven"
    if rest == 1:
        return "katreveneyen"
    return "katreven" + _HT_0_19[rest]


def _ht_under_1000(n: int) -> str:
    hundreds, rest = divmod(n, 100)
    parts = []
    if hundreds:
        parts.append(_HT_HUNDREDS[hundreds])
    if rest or not hundreds:
        parts.append(_ht_under_100(rest))
    return " ".join(parts)


def ht_cardinal(n: int) -> str:
    """Haitian Creole cardinal for a non-negative integer below 10^9."""
    if n < 0:
        return "mwens " + ht_cardinal(-n)
    if n < 1000:
        return _ht_under_1000(n)

    millions, rest = divmod(n, 1_000_000)
    parts: list[str] = []
    if millions:
        if millions == 1:
            parts.append("en milyon")
        else:
            parts.append(f"{_ht_under_1000(millions)} milyon")
    thousands, hundreds = divmod(rest, 1000)
    if thousands:
        # 1000 is bare "mil", not "en mil".
        parts.append("mil" if thousands == 1 else f"{_ht_under_1000(thousands)} mil")
    if hundreds:
        parts.append(_ht_under_1000(hundreds))
    return " ".join(parts)


# --- Russian ----------------------------------------------------------------

_num2words: Optional[Callable] = None
_num2words_checked = False


def _load_num2words() -> Optional[Callable]:
    """Import num2words lazily.  Absent → callers leave digits alone rather
    than crash a live service over a missing optional dependency."""
    global _num2words, _num2words_checked
    if not _num2words_checked:
        _num2words_checked = True
        try:
            from num2words import num2words as _fn
            _num2words = _fn
        except Exception:
            _num2words = None
    return _num2words


def ru_cardinal(n: int) -> Optional[str]:
    """Russian cardinal, nominative case.  Not always grammatical in context
    (Russian declines numerals by case), but always intelligible — which is
    the whole bar here, since the alternative is the number vanishing.

    Returns None when num2words is unavailable.
    """
    fn = _load_num2words()
    if fn is None:
        return None
    try:
        return fn(n, lang="ru")
    except Exception:
        return None


# --- Language table ---------------------------------------------------------

# Per language: cardinal fn, the word between chapter and verse, the word
# joining a range, and the word for "percent".
_LANGS = {
    "ht": {
        "cardinal": ht_cardinal,
        "verse": "vèsè",
        "range": "a",
        "percent": "pousan",
    },
    "ru": {
        "cardinal": ru_cardinal,
        "verse": "стих",
        "range": "до",
        "percent": "процентов",
    },
}


def has_converter(lang: str) -> bool:
    """True when this language has a working converter right now."""
    spec = _LANGS.get(lang)
    if spec is None:
        return False
    if lang == "ru":
        return _load_num2words() is not None
    return True


# Digits grouped in threes by a comma: "1,000" -> "1000".  Only when the
# groups are exactly three long, so "Genesis 1,2" is left alone.
_THOUSAND_SEP = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")
_VERSE_REF = re.compile(r"(\d+)\s*:\s*(\d+)")
_RANGE = re.compile(r"(\d+)\s*[-–—]\s*(\d+)")
_PERCENT = re.compile(r"(\d+)\s*%")
_INTEGER = re.compile(r"\d+")

# Above this, spelling it out is longer than it is useful and almost
# certainly not a real quantity (a phone number, an ID).  Left as digits,
# which the voice drops — no worse than today, and it never produces a
# thirty-word noun phrase mid-sermon.
_MAX = 999_999_999


def verbalize(text: str, lang: str) -> str:
    """Rewrite digits in `text` as words in `lang`.

    Unknown language, or a converter that cannot load, returns `text`
    unchanged.  Safe to call on text with no digits (returns it as-is).
    """
    if not text or not any(ch.isdigit() for ch in text):
        return text
    spec = _LANGS.get(lang)
    if spec is None:
        return text

    cardinal = spec["cardinal"]

    def say(raw: str) -> Optional[str]:
        try:
            n = int(raw)
        except ValueError:
            return None
        if n > _MAX:
            return None
        return cardinal(n)

    out = _THOUSAND_SEP.sub("", text)

    def _verse(m: re.Match) -> str:
        a, b = say(m.group(1)), say(m.group(2))
        if a is None or b is None:
            return m.group(0)
        return f"{a} {spec['verse']} {b}"

    def _range(m: re.Match) -> str:
        a, b = say(m.group(1)), say(m.group(2))
        if a is None or b is None:
            return m.group(0)
        return f"{a} {spec['range']} {b}"

    def _percent(m: re.Match) -> str:
        a = say(m.group(1))
        return m.group(0) if a is None else f"{a} {spec['percent']}"

    def _plain(m: re.Match) -> str:
        a = say(m.group(0))
        return m.group(0) if a is None else a

    out = _VERSE_REF.sub(_verse, out)
    out = _RANGE.sub(_range, out)
    out = _PERCENT.sub(_percent, out)
    out = _INTEGER.sub(_plain, out)
    return out
