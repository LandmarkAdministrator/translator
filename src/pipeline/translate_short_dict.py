"""
Exact-match overrides for very short translation inputs.

Both NLLB and Opus-MT hallucinate badly on short isolated utterances
(no surrounding context to anchor on). `Amen.` becomes `- No lo sé.` in
Spanish or `Mèsi anpil.` in Haitian — both wrong, both extremely visible
during a service.

This dictionary catches the most common offenders before they reach the
model. Lookup is exact-match (case- and punctuation-sensitive) so we
don't accidentally clobber legitimate translations.

The entries below were validated by the sister batch project's
multi-LLM voter panel over a 10-sermon corpus. See ADR-029 in
~/Projects/Multi-Bitrate-Sermons/DECISIONS.md for the methodology.
"""

from __future__ import annotations

from typing import Optional


# Key: exact English input.
# Value: (Spanish, Haitian Creole) tuple.
SHORT_INPUT_DICTIONARY: dict[str, tuple[str, str]] = {
    "Amen.":         ("Amén.",          "Amèn."),
    "Amen?":         ("¿Amén?",         "Amèn?"),
    "Thank you.":    ("Gracias.",       "Mèsi."),
    "Thanks.":       ("Gracias.",       "Mèsi."),
    "Jesus.":        ("Jesús.",         "Jezi."),
    "Well.":         ("Bueno.",         "Byen."),
    "Hello.":        ("Hola.",          "Bonjou."),
    "Why?":          ("¿Por qué?",      "Poukisa?"),
    "What?":         ("¿Qué?",          "Ki sa?"),
    "Good evening.": ("Buenas noches.", "Bonn aswè."),
}


# Map our 2-letter target codes to position in the value tuple.
_LANG_INDEX = {"es": 0, "ht": 1}


def lookup(text: str, target_language: str) -> Optional[str]:
    """Return a dictionary translation if one exists, else None.

    Callers should fall back to the model when this returns None.
    """
    if target_language not in _LANG_INDEX:
        return None
    entry = SHORT_INPUT_DICTIONARY.get(text)
    if entry is None:
        return None
    return entry[_LANG_INDEX[target_language]]
