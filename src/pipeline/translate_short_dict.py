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


# Key: exact English input. Value: translation per target language code.
#
# Keyed by language rather than positional so a new language is one key per
# entry instead of a tuple widened in eleven places — the tuple form silently
# mismatched if any row was missed.
#
# Spanish and Haitian Creole were validated by the sister batch project's
# multi-LLM voter panel over a 10-sermon corpus (ADR-029). Russian was added
# 2026-09-04 and has NOT been through that panel; it is conventional usage and
# should be reviewed by a native speaker before it is leaned on.
SHORT_INPUT_DICTIONARY: dict[str, dict[str, str]] = {
    "Amen.":         {"es": "Amén.",          "ht": "Amèn.",      "ru": "Аминь."},
    "Amen?":         {"es": "¿Amén?",         "ht": "Amèn?",      "ru": "Аминь?"},
    "Thank you.":    {"es": "Gracias.",       "ht": "Mèsi.",      "ru": "Спасибо."},
    "Thanks.":       {"es": "Gracias.",       "ht": "Mèsi.",      "ru": "Спасибо."},
    "Jesus.":        {"es": "Jesús.",         "ht": "Jezi.",      "ru": "Иисус."},
    "Well.":         {"es": "Bueno.",         "ht": "Byen.",      "ru": "Что ж."},
    "Hello.":        {"es": "Hola.",          "ht": "Bonjou.",    "ru": "Здравствуйте."},
    "Why?":          {"es": "¿Por qué?",      "ht": "Poukisa?",   "ru": "Почему?"},
    "What?":         {"es": "¿Qué?",          "ht": "Ki sa?",     "ru": "Что?"},
    "Good evening.": {"es": "Buenas noches.", "ht": "Bonn aswè.", "ru": "Добрый вечер."},
}


def lookup(text: str, target_language: str) -> Optional[str]:
    """Return a dictionary translation if one exists, else None.

    Callers should fall back to the model when this returns None.
    """
    entry = SHORT_INPUT_DICTIONARY.get(text)
    if entry is None:
        return None
    return entry.get(target_language)
