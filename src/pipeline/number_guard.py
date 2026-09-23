"""Keep numbers alive across translation.

Measured 2026-09-23 over the 56-minute test suite: Opus-MT carried only 59% of
the numbers into Haitian Creole, against NLLB's 100%. "Turn in your hymn books
again to 413" came back as "pou 4 1513"; "a scan at 7:30" lost the time
altogether. For a service built on chapter, verse and hymn numbers that is the
first fault a listener would notice.

Two obvious designs were measured and rejected:

* **Placeholders.** Replace each number with a token, translate, put them
  back. Tokens like `X0`, `NUM0` and `#0#` survive Spanish (100%) and Russian
  (93%), but **not one of them survives Creole** — the model drops or mangles
  every marker it has not seen in training.
* **Spelling numbers out in English first.** "Chapter 6" to "chapter six"
  works, but larger numbers are then mistranslated as words: "when I was 17"
  became "sèt an" (seven), "at 30" became "nan tren" (in the train), and
  "turned 30" lost the number entirely. Worse than doing nothing.

What does work is to leave the digits in place, check the result, and repair
only when the check fails: translate the sentence around its numbers and put
the digits back verbatim. The repair costs one extra pass, and only on the
sentences that actually lost something.

The page keeps real digits, which is what a reader wants. Spelling them out
for the voices is a separate stage, since the Creole and Russian voices cannot
pronounce digits at all (TODO item 8).
"""

from __future__ import annotations

import re
from typing import Callable, Iterable

# 6, 413, 7:30, 5:17, 1,500, 2.5 — a run of digits with separators inside it.
NUMBER = re.compile(r"\d+(?:[.,:/]\d+)*")

# Spacing left behind once a sentence has been cut apart and rejoined.
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.;:!?%)\]»…])")
_SPACE_AFTER_OPEN = re.compile(r"([(\[«¿¡])\s+")
_MULTI_SPACE = re.compile(r"\s{2,}")


def numbers(text: str) -> list[str]:
    """Every number in the text, in order, with thousands separators removed."""
    return [m.group(0).replace(",", "") for m in NUMBER.finditer(text)]


def _spelled(n: int, lang: str) -> set[str]:
    """The number written out in the target language, where we can."""
    forms: set[str] = set()
    try:
        from pipeline import number_words
        if lang == "ht":
            forms.add(number_words.ht_cardinal(n))
        elif lang == "ru":
            forms.add(number_words.ru_cardinal(n) or "")
    except Exception:  # noqa: BLE001 - detection must never break a translation
        pass
    try:
        from num2words import num2words
        if lang in ("es", "ru", "fr", "pt"):
            forms.add(num2words(n, lang=lang))
    except Exception:  # noqa: BLE001
        pass
    return {f.lower() for f in forms if f}


def present(number: str, translated: str, lang: str = "") -> bool:
    """Is this source number still in the translation, in any fair form?

    A translator is allowed to write 1970 as "los años 70", to split 7:30 into
    seven and thirty, or to spell the figure out. Only an outright
    disappearance counts as a loss — an earlier version of this check called
    "a mediados de los años 70" a failure and then "repaired" a perfectly good
    sentence into "la mitad... 1970 s".
    """
    low = translated.lower()
    if number in translated:
        return True
    parts = [p for p in re.split(r"[.,:/]", number) if p]
    if len(parts) > 1 and all(p in translated for p in parts):
        return True
    if len(number) == 4 and number.isdigit() and number[-2:] in translated:
        return True                      # 1970 -> "años 70", "70-х"
    try:
        n = int(number)
    except ValueError:
        return False
    return any(f in low for f in _spelled(n, lang))


def keeps_numbers(source: str, translated: str, lang: str = "") -> bool:
    """Did every number survive, in some recognisable form?"""
    return all(present(n, translated, lang) for n in numbers(source))


def missing(source: str, translated: str, lang: str = "") -> list[str]:
    return [n for n in numbers(source) if not present(n, translated, lang)]


def tidy(text: str) -> str:
    text = _MULTI_SPACE.sub(" ", text.strip())
    text = _SPACE_BEFORE_PUNCT.sub(r"\1", text)
    text = _SPACE_AFTER_OPEN.sub(r"\1", text)
    if text and text[0].islower():
        text = text[0].upper() + text[1:]
    return text


def segment_translate(text: str, translate: Callable[[str], str]) -> str:
    """Translate the words around each number and put the digits back.

    The number cannot be lost because the model never sees it. Word order
    suffers a little where a language would inflect around the figure, which
    is why this runs only after the plain translation has already failed.
    """
    pieces: list[tuple[str, bool]] = []      # (text, is_number)
    last = 0
    for m in NUMBER.finditer(text):
        before = text[last:m.start()]
        if before.strip():
            piece = translate(before.strip()) if _has_letters(before) else before.strip()
            pieces.append((piece, False))
        pieces.append((m.group(0), True))
        last = m.end()
    tail = text[last:]
    if tail.strip():
        piece = translate(tail.strip()) if _has_letters(tail) else tail.strip()
        pieces.append((piece, False))

    out = ""
    for i, (piece, is_num) in enumerate(pieces):
        if not is_num and i < len(pieces) - 1:
            # A fragment translated on its own comes back as a finished
            # sentence; strip that so the number does not start a new one.
            piece = piece.rstrip().rstrip(".!?")
        if out and not is_num and out.rstrip()[-1:].isdigit() and piece[:1].isupper():
            if not _looks_proper(piece):
                piece = piece[0].lower() + piece[1:]
        if not out:
            out = piece
        elif piece[:1] in ",.;:!?)»":
            out += piece
        else:
            out += " " + piece
    return tidy(out)


def _looks_proper(piece: str) -> bool:
    """A capitalized fragment that is probably a name, not a sentence start."""
    first = piece.split()[0] if piece.split() else ""
    return len(first) > 1 and first[1:].islower() and first.rstrip(".,").isalpha() \
        and first.lower() in _PROPERISH


_PROPERISH = {"daniel", "danyèl", "dànyèl", "apple", "jezi", "bondye", "seyè",
              "jesus", "god", "lord", "macintosh", "reed", "woz", "amen"}


def _has_letters(text: str) -> bool:
    return any(c.isalpha() for c in text)


# Placeholders survive Spanish (100%) and Russian (93%) but nothing survives
# Creole, measured 2026-09-23. Where they work, masking is cheaper and gentler
# than cutting the sentence apart.
MASKABLE = {"es", "ru", "fr", "pt"}
_MASK = "X{}"


def mask_translate(text: str, translate, lang: str) -> str:
    nums = numbers(text)
    masked, idx = text, 0
    for m in list(NUMBER.finditer(text))[::-1]:
        masked = masked[:m.start()] + _MASK.format(len(nums) - 1 - idx) + masked[m.end():]
        idx += 1
    out = translate(masked)
    for i, n in enumerate(nums):
        out = out.replace(_MASK.format(i), n)
    return tidy(out)


def guard(text: str, translate, lang: str = "") -> tuple[str, str]:
    """Translate `text`, making sure its numbers survive.

    Returns the translation and what was done: `none` (no numbers), `clean`
    (nothing needed), `masked` (translated with placeholders), `repaired`
    (translated around the numbers), `partial` (still incomplete; the better
    attempt is returned).
    """
    if not numbers(text):
        return translate(text), "none"
    plain = translate(text)
    if keeps_numbers(text, plain, lang):
        return plain, "clean"
    if lang in MASKABLE:
        masked = mask_translate(text, translate, lang)
        if keeps_numbers(text, masked, lang):
            return masked, "masked"
    repaired = segment_translate(text, translate)
    if keeps_numbers(text, repaired, lang):
        return repaired, "repaired"
    if len(missing(text, repaired, lang)) < len(missing(text, plain, lang)):
        return repaired, "partial"
    return plain, "partial"


def audit(pairs: Iterable[tuple[str, str]], lang: str = "") -> dict:
    """Summarize number retention over many (source, translation) pairs."""
    total = lost = sentences = bad = 0
    for src, out in pairs:
        want = numbers(src)
        if not want:
            continue
        sentences += 1
        total += len(want)
        gone = len(missing(src, out, lang))
        lost += gone
        bad += 1 if gone else 0
    return {"sentences_with_numbers": sentences, "numbers": total, "lost": lost,
            "sentences_damaged": bad,
            "retention": 1.0 - (lost / total) if total else 1.0}
