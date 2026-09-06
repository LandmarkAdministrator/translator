"""Text heuristics shared by pipeline stages.

`looks_like_loop` began life as the Whisper hallucination filter in the
retired batch ASR path. The translation stage kept using it on NLLB output
("Mr Mr Mr...", "Eisen Eisen Eisenh..."), which pulled the whole retired
module into the live import path; now it lives here on its own.
"""
from __future__ import annotations

from collections import Counter

_STOPWORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'of', 'to', 'in', 'on', 'at', 'for',
    'with', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'am',
    'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
    'my', 'your', 'his', 'our', 'their', 'this', 'that', 'these', 'those',
    'as', 'by', 'from', 'so', 'if', 'not', 'no', 'do', 'does', 'did',
    'have', 'has', 'had',
}


def looks_like_loop(text: str) -> bool:
    """True if the text looks like a decoder caught in a repetition loop.

    Two heuristics:
    - Content-word dominance: ignoring stopwords, one word makes up >40% of
      the content OR appears more than 6 times ("oh oh oh", "thank you thank
      you thank you").
    - Trigram loop: any 3-word phrase repeats more than 3 times ("out of the
      road out of the road...").
    """
    words = text.lower().split()
    if not words:
        return False

    content_words = [w.strip(".,!?;:'\"") for w in words]
    content_words = [w for w in content_words if w and w not in _STOPWORDS]

    if content_words:
        counts = Counter(content_words)
        top_word, top_count = counts.most_common(1)[0]
        if top_count > 6 or top_count > len(content_words) * 0.40:
            return True

    if len(words) >= 9:
        trigrams = [tuple(words[i:i + 3]) for i in range(len(words) - 2)]
        tg_counts = Counter(trigrams)
        if tg_counts.most_common(1)[0][1] > 3:
            return True

    return False
