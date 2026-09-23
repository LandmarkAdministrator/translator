"""Tests for src/pipeline/number_words.py.

Run: ./venv/bin/python tests/test_number_words.py
No model loading, no GPU — pure text, runs in milliseconds.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline.number_words import (  # noqa: E402
    ht_cardinal,
    has_converter,
    verbalize,
)

# The exact vocabulary of facebook/mms-tts-hat, read from vocab.json in the
# model cache on 2026-09-21.  Anything the converter emits that is not in
# here would be silently dropped by VitsTokenizer — which is the bug we are
# fixing, so producing such a character would be a regression.
MMS_HAT_VOCAB = set(" '-abcdefghijklmnoprstuvwyz|àèò—")

FAILURES: list[str] = []


def check(label: str, got, want) -> None:
    if got != want:
        FAILURES.append(f"{label}: got {got!r}, want {want!r}")


def test_todo_examples() -> None:
    """The four Creole numerals recorded in TODO.md item 8."""
    check("6", ht_cardinal(6), "sis")
    check("17", ht_cardinal(17), "disèt")
    check("119", ht_cardinal(119), "san diznèf")
    check("1611", ht_cardinal(1611), "mil sisan onz")


def test_ht_units_and_teens() -> None:
    for n, want in enumerate(
        ["zewo", "en", "de", "twa", "kat", "senk", "sis", "sèt", "uit", "nèf",
         "dis", "onz", "douz", "trèz", "katòz", "kenz", "sèz", "disèt",
         "dizuit", "diznèf"]
    ):
        check(f"ht {n}", ht_cardinal(n), want)


def test_ht_decades() -> None:
    cases = {
        20: "ven", 21: "venteyen", 22: "vennde", 27: "vennsèt",
        28: "ventuit", 29: "ventnèf",
        30: "trant", 31: "tranteyen", 33: "tranntwa", 39: "trantnèf",
        40: "karant", 41: "karanteyen", 48: "karantuit",
        50: "senkant", 55: "senkannsenk",
        60: "swasant", 66: "swasannsis",
        70: "swasanndis", 71: "swasannonz", 77: "swasanndisèt",
        79: "swasanndiznèf",
        80: "katreven", 81: "katreveneyen", 85: "katrevensenk",
        90: "katrevendis", 91: "katrevenonz", 99: "katrevendiznèf",
    }
    for n, want in cases.items():
        check(f"ht {n}", ht_cardinal(n), want)


def test_ht_hundreds_and_thousands() -> None:
    cases = {
        100: "san", 101: "san en", 105: "san senk", 150: "san senkant",
        200: "desan", 600: "sisan", 999: "nèfsan katrevendiznèf",
        1000: "mil", 1005: "mil senk", 2000: "de mil",
        2026: "de mil vennsis",
    }
    for n, want in cases.items():
        check(f"ht {n}", ht_cardinal(n), want)


def test_ht_output_is_speakable() -> None:
    """Every character the converter can emit must be in the MMS vocabulary.

    This is the test that actually guards the bug: a numeral containing a
    character the tokenizer does not know would be dropped just like the
    digit it replaced.
    """
    bad: dict[int, set] = {}
    for n in list(range(0, 1001)) + [1611, 2026, 12345, 100000, 999999]:
        stray = set(ht_cardinal(n)) - MMS_HAT_VOCAB
        if stray:
            bad[n] = stray
    if bad:
        FAILURES.append(f"characters outside the MMS-hat vocabulary: {bad}")


def test_verse_references() -> None:
    check(
        "verse 5:17",
        verbalize("5:17", "ht"),
        "senk vèsè disèt",
    )
    check(
        "chapter line",
        verbalize("Danyèl chapit nimewo 6", "ht"),
        "Danyèl chapit nimewo sis",
    )
    check(
        "psalm line",
        verbalize("Sòm 119 vèsè 105", "ht"),
        "Sòm san diznèf vèsè san senk",
    )


def test_ranges_and_separators() -> None:
    check("range", verbalize("vèsè 3-5", "ht"), "vèsè twa a senk")
    check("thousand sep", verbalize("1,000", "ht"), "mil")
    # Not a thousands separator — groups aren't three long.
    check("not a sep", verbalize("1,2", "ht"), "en,de")


def test_passthrough() -> None:
    check("no digits", verbalize("Amen.", "ht"), "Amen.")
    check("empty", verbalize("", "ht"), "")
    check("unknown lang", verbalize("chapter 6", "de"), "chapter 6")
    # Spanish goes through Kokoro, which reads digits itself.
    check("es untouched", verbalize("capítulo 6", "es"), "capítulo 6")


def test_absurd_numbers_left_alone() -> None:
    check("huge", verbalize("5551234567890", "ht"), "5551234567890")


def test_russian_if_available() -> None:
    if not has_converter("ru"):
        print("  (skipped Russian: num2words not installed here)")
        return
    got = verbalize("глава 6", "ru")
    if "6" in got:
        FAILURES.append(f"ru: digit survived verbalization: {got!r}")
    got = verbalize("3:16", "ru")
    if "стих" not in got or any(c.isdigit() for c in got):
        FAILURES.append(f"ru verse ref not converted: {got!r}")


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}):")
        for f in FAILURES:
            print("  -", f)
        return 1
    print(f"OK — {len(tests)} test groups passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
