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

# Sentence-final punctuation opening a fragment: belongs to the sentence that
# was already emitted, not the one starting here.
_LEAD_PUNCT = re.compile(r'^\s*[.?!,;:]+\s*')


def _clean_join(fragments: List[str]) -> str:
    """Concatenate Parakeet fragments and normalize whitespace.

    Each fragment carries its own SentencePiece-style leading-space marker:
    new words start with a space, continuations of a previous word don't.
    We concatenate WITHOUT adding any separator — direct join preserves
    `" just"` + `"ice"` = `" justice"`. The previous space-join strategy
    incorrectly produced `"just ice"` for this case, which was the source
    of the long-standing `"J ustice"`, `"signifying ing"`, `"re volution"`
    subword artifacts.
    """
    if not fragments:
        return ""
    joined = "".join(f for f in fragments if f)
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
        max_buffer_chars: int = 800,
        max_emit_words: int = 40,
        silence_min_words: int = 1,
        strip_lead_punct: bool = True,
    ):
        """
        Args:
            silence_timeout: flush after this many seconds of no new fragments
                when the buffer has pending content.
            hard_timeout: flush if the buffer has been accumulating this many
                seconds since the first fragment, regardless of punctuation or
                word count — this is the time-based safety valve.
            min_emit_chars: don't emit anything shorter than this (in chars,
                post-cleanup). Floor against stray punctuation like ".".
            min_emit_words: the PUNCTUATION trigger requires at least this many
                words. A mark arriving mid-thought ("...Amen. And so") should
                not split the sentence; waiting for more words keeps the unit
                worth translating. Hard-timeout, size-cap, max_emit_words and
                shutdown flush() ignore this floor so no content is dropped.
            silence_min_words: the SILENCE trigger's own, much lower floor. A
                short utterance followed by a real pause is a complete sentence
                ("Amen.", "Let us pray."); holding it until the hard timeout
                just adds seconds of latency to the shortest lines. Set to 0 to
                flush anything at all on silence.
            max_emit_words: force a flush once the buffer reaches this many
                words even without punctuation. Bounds worst-case latency:
                unpunctuated runs otherwise grow until the hard timeout, and
                the audio for them lands long after it was spoken.
            max_buffer_chars: emit immediately when the joined buffer exceeds
                this length, even if hard_timeout hasn't tripped. Catches
                pause-prone speech that accumulates across multiple
                hard_timeout windows. ~800 chars is a long paragraph; NLLB
                quality drops past sentence-level inputs anyway.
        """
        self.silence_timeout = silence_timeout
        self.hard_timeout = hard_timeout
        self.min_emit_chars = min_emit_chars
        self.min_emit_words = min_emit_words
        self.max_buffer_chars = max_buffer_chars
        self.max_emit_words = max_emit_words
        self.silence_min_words = silence_min_words
        self.strip_lead_punct = strip_lead_punct

        self.last_reason: str = ""    # why the most recent emit fired
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
        if not text.strip():
            # Whitespace-only fragment: still tick so silence_timeout can fire.
            return self.tick(now=now)

        # Preserve leading whitespace — it carries word-boundary info from
        # SentencePiece tokenization. Stripping here would break the
        # `" just"` + `"ice"` = `" justice"` joining in _clean_join.
        now = now if now is not None else time.monotonic()

        if not self._frags:
            # A silence flush during the pause before the speaker's next word
            # sends the sentence before the ASR has emitted its closing mark;
            # that mark then arrives at the head of this fragment, belonging to
            # a sentence already downstream. Drop it rather than open the next
            # sentence with ". " — which reached congregants and, via NLLB,
            # their translations too.
            if self.strip_lead_punct:
                stripped = _LEAD_PUNCT.sub("", text, count=1)
                if stripped.strip():
                    text = stripped
            self._first_start_wall = start_wall
            self._first_recv_monotonic = now
        self._frags.append(text)
        self._last_recv_monotonic = now
        self._asr_accum += asr_time

        # Punctuation flush takes priority — no need to wait for silence —
        # but only if we have enough words to be worth translating.
        if self._ends_sentence() and self._has_min_words():
            self.last_reason = "punctuation"
            return self._emit()

        # Word cap: bound how late an unpunctuated run can arrive. Ignores the
        # min-words floor by definition (we are over it).
        if self.max_emit_words and self._word_count() >= self.max_emit_words:
            self.last_reason = "word_cap"
            return self._emit()

        # Hard timeout can trip even on the arrival of a new fragment; this
        # is a safety valve against unbounded growth, so it ignores min_words.
        if (now - self._first_recv_monotonic) >= self.hard_timeout:
            self.last_reason = "hard_timeout"
            return self._emit()

        # Size-based safety valve — same intent as hard_timeout but watches
        # accumulated text length instead of elapsed time.
        if self._joined_length() >= self.max_buffer_chars:
            self.last_reason = "size_cap"
            return self._emit()

        return None

    def tick(self, now: Optional[float] = None) -> Optional[Tuple[str, float, float]]:
        """Call periodically (e.g. on each audio callback) to catch silence flushes."""
        if not self._frags:
            return None
        now = now if now is not None else time.monotonic()
        if ((now - self._last_recv_monotonic) >= self.silence_timeout
                and self._word_count() >= self.silence_min_words):
            self.last_reason = "silence"
            return self._emit()
        if self.max_emit_words and self._word_count() >= self.max_emit_words:
            return self._emit()
        if (now - self._first_recv_monotonic) >= self.hard_timeout:
            return self._emit()
        if self._joined_length() >= self.max_buffer_chars:
            return self._emit()
        return None

    def flush(self) -> Optional[Tuple[str, float, float]]:
        """Force-emit whatever is buffered (shutdown).

        Ignores both min_emit_words AND min_emit_chars so no content is
        dropped at session end — whatever the speaker said last should
        reach translation.
        """
        if not self._frags:
            return None
        self.last_reason = "shutdown"
        return self._emit(force=True)

    # -------------------------------------------------------------- internals

    def _ends_sentence(self) -> bool:
        text = _clean_join(self._frags)
        if not _SENT_END.search(text):
            return False
        if _ABBREV_TAIL.search(text):
            return False
        return True

    def _word_count(self) -> int:
        text = _clean_join(self._frags)
        return len([w for w in text.split() if any(c.isalnum() for c in w)])

    def _has_min_words(self) -> bool:
        """Count alphanumeric-bearing words; ignore pure-punctuation tokens."""
        if self.min_emit_words <= 0:
            return True
        text = _clean_join(self._frags)
        words = [w for w in text.split() if any(c.isalnum() for c in w)]
        return len(words) >= self.min_emit_words

    def _joined_length(self) -> int:
        return len(_clean_join(self._frags))

    def _emit(self, force: bool = False) -> Optional[Tuple[str, float, float]]:
        text = _clean_join(self._frags)
        first_wall = self._first_start_wall
        asr = self._asr_accum
        self._frags = []
        self._first_start_wall = 0.0
        self._first_recv_monotonic = 0.0
        self._last_recv_monotonic = 0.0
        self._asr_accum = 0.0
        # Force path (shutdown flush) ignores the char floor. Non-force emits
        # from punctuation / silence / hard-timeout drop purely-punctuation
        # remnants like "." or "".
        if not force and len(text) < self.min_emit_chars:
            return None
        if not text:
            return None
        return (text, first_wall, asr)
