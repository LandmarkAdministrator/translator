"""
Sentence buffer for streaming ASR output.

Parakeet's token-level LocalAgreement-2 commits emit tiny fragments (often
1-3 words: "Mr", ". Speaker", ", President Eisenhower,"). Feeding those
directly to a translator produces bad translations and — on NLLB especially —
can trigger repetition loops because the input has no context for the model
to anchor on.

SentenceBuffer sits between the ASR output and the translation pipelines,
accumulates fragments, and flushes a full utterance when one of these fires:

  1. The buffered text ends in sentence-final punctuation (. ? !) and the
     punctuation is not a recognized abbreviation like "Mr." or "i.e.".
  2. No new fragment has arrived for `silence_timeout` seconds — the speaker
     paused, so we assume the sentence is done even without punctuation.
  3. The buffer has been growing for `hard_timeout` seconds without break —
     safety valve for run-on speech or missed punctuation.

The emit tuple matches what the streaming callback expects from Parakeet's
feed(): (text, first_chunk_start_wall, accumulated_asr_time). This keeps the
downstream pipeline.process() call unchanged.
"""

from __future__ import annotations

import re
import time
from typing import List, Optional, Tuple


# Common English abbreviations that end in "." but do not end a sentence.
# Matched at end-of-string; case-insensitive. If the buffer ends with one of
# these we keep accumulating instead of flushing.
_ABBREV_TAIL = re.compile(
    r'(?:^|[\s"(\[\'])'                               # boundary
    r'(?:Mr|Mrs|Ms|Mx|Dr|St|Jr|Sr|Mt|Ft|Fr|Rev|Hon|'
    r'Prof|Gen|Col|Capt|Cpl|Sgt|Lt|Cmdr|Adm|Gov|Pres|'
    r'vs|etc|i\.e|e\.g|cf|No|Nos|Vol|pp|approx|est'
    r')\.\s*$',
    re.IGNORECASE,
)

# Sentence terminator at end of buffer (possibly followed by closing quote /
# bracket / whitespace).
_SENT_END = re.compile(r'[.?!]["\')\]]?\s*$')


def _clean_join(fragments: List[str]) -> str:
    """Join fragments with a space and normalize punctuation spacing.

    Parakeet commits sometimes contain leading punctuation (e.g. ", Mr.")
    and sometimes don't; blindly joining with space produces "word ," which
    we collapse here so the translator sees clean input.
    """
    if not fragments:
        return ""
    joined = " ".join(f.strip() for f in fragments if f and f.strip())
    # No space before , . ; : ! ?
    joined = re.sub(r"\s+([,.;:!?])", r"\1", joined)
    # Collapse multi-space
    joined = re.sub(r"\s+", " ", joined).strip()
    return joined


class SentenceBuffer:
    """Accumulate Parakeet fragments into sentences before translation."""

    def __init__(
        self,
        silence_timeout: float = 2.0,
        hard_timeout: float = 10.0,
        min_emit_chars: int = 2,
        min_emit_words: int = 3,
    ):
        """
        Args:
            silence_timeout: flush after this many seconds of no new fragments
                when the buffer has pending content.
            hard_timeout: flush if the buffer has been accumulating this many
                seconds since the first fragment, regardless of punctuation or
                word count — this is the safety valve.
            min_emit_chars: don't emit anything shorter than this (in chars,
                post-cleanup). Floor against stray punctuation like ".".
            min_emit_words: punctuation- and silence-triggered emits require at
                least this many alphanumeric words. Hard-timeout emits and
                shutdown flush() ignore this floor so no content is dropped.
        """
        self.silence_timeout = silence_timeout
        self.hard_timeout = hard_timeout
        self.min_emit_chars = min_emit_chars
        self.min_emit_words = min_emit_words

        self._frags: List[str] = []
        self._first_start_wall: float = 0.0       # start-wall of first fragment
        self._first_recv_monotonic: float = 0.0   # when we received the first frag
        self._last_recv_monotonic: float = 0.0    # when we received the most recent
        self._asr_accum: float = 0.0

    # ------------------------------------------------------------------ api

    def feed(
        self,
        text: str,
        start_wall: float,
        asr_time: float,
        now: Optional[float] = None,
    ) -> Optional[Tuple[str, float, float]]:
        """Add a fragment; return (sentence, start_wall, asr_accum) if ready.

        `now` is only used for testing; production callers can omit it.
        """
        if text is None:
            return None
        stripped = text.strip()
        if not stripped:
            # Still treat this as a tick in case a silence_timeout should fire.
            return self.tick(now=now)

        now = now if now is not None else time.monotonic()

        if not self._frags:
            self._first_start_wall = start_wall
            self._first_recv_monotonic = now
        self._frags.append(stripped)
        self._last_recv_monotonic = now
        self._asr_accum += asr_time

        # Punctuation flush takes priority — no need to wait for silence —
        # but only if we have enough words to be worth translating.
        if self._ends_sentence() and self._has_min_words():
            return self._emit()

        # Hard timeout can trip even on the arrival of a new fragment; this
        # is a safety valve against unbounded growth, so it ignores min_words.
        if (now - self._first_recv_monotonic) >= self.hard_timeout:
            return self._emit()

        return None

    def tick(self, now: Optional[float] = None) -> Optional[Tuple[str, float, float]]:
        """Call periodically (e.g. on each audio callback) to catch silence flushes."""
        if not self._frags:
            return None
        now = now if now is not None else time.monotonic()
        if (now - self._last_recv_monotonic) >= self.silence_timeout and self._has_min_words():
            return self._emit()
        if (now - self._first_recv_monotonic) >= self.hard_timeout:
            return self._emit()
        return None

    def flush(self) -> Optional[Tuple[str, float, float]]:
        """Force-emit whatever is buffered (shutdown)."""
        if not self._frags:
            return None
        return self._emit()

    # -------------------------------------------------------------- internals

    def _ends_sentence(self) -> bool:
        text = _clean_join(self._frags)
        if not _SENT_END.search(text):
            return False
        if _ABBREV_TAIL.search(text):
            return False
        return True

    def _has_min_words(self) -> bool:
        """Count alphanumeric-bearing words; ignore pure-punctuation tokens."""
        if self.min_emit_words <= 0:
            return True
        text = _clean_join(self._frags)
        words = [w for w in text.split() if any(c.isalnum() for c in w)]
        return len(words) >= self.min_emit_words

    def _emit(self) -> Optional[Tuple[str, float, float]]:
        text = _clean_join(self._frags)
        first_wall = self._first_start_wall
        asr = self._asr_accum
        self._frags = []
        self._first_start_wall = 0.0
        self._first_recv_monotonic = 0.0
        self._last_recv_monotonic = 0.0
        self._asr_accum = 0.0
        if len(text) < self.min_emit_chars:
            return None
        return (text, first_wall, asr)
