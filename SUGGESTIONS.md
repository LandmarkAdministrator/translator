# Suggestions for the translator project

These suggestions come from cross-referencing the live translator against the sister batch project at `~/Projects/Multi-Bitrate-Sermons` (sermon backfill: encode → ASR → NLLB translate → TTS → HLS package), where many of the same translation-quality and reliability problems were already solved.  Both projects translate the same congregation's audio, so problems carry across.

The numbered items below are roughly priority-ordered.  Each one is a *recommendation* — not a plan — pending your review.

---

## 1. Add a sermon-tuned `initial_prompt` to Whisper

**File:** `src/pipeline/asr.py` (`WhisperTransformersService.transcribe()` and the `ASRService.transcribe()` faster-whisper path).

**Change:** Pass an `initial_prompt` listing every Bible book name + KJV pronouns + recurring theological vocabulary.  Same prompt the batch pipeline uses (`scripts/pipeline/transcribe.py`):

> "This is a sermon from Landmark Baptist Church. The preacher may reference Genesis, Exodus, Leviticus, Numbers, Deuteronomy, Joshua, Judges, Ruth, Samuel, Kings, Chronicles, Ezra, Nehemiah, Esther, Job, Psalms, Proverbs, Ecclesiastes, Song of Solomon, Isaiah, Jeremiah, Lamentations, Ezekiel, Daniel, Hosea, Joel, Amos, Obadiah, Jonah, Micah, Nahum, Habakkuk, Zephaniah, Haggai, Zechariah, Malachi, Matthew, Mark, Luke, John, Acts, Romans, Corinthians, Galatians, Ephesians, Philippians, Colossians, Thessalonians, Timothy, Titus, Philemon, Hebrews, James, Peter, Jude, Revelation. Common terms: grace, justification, sanctification, atonement, brethren, the Lord, thee, thou, thy, repentance."

For `WhisperTransformersService` the prompt goes into `generate_kwargs` as `prompt` (or via the model's `forced_decoder_ids` setup, depending on transformers version).  For `faster-whisper` (`ASRService`) it's the `initial_prompt=` kwarg on `model.transcribe()`.

**Why:**
- Same audio domain as the batch project — same vocabulary problems.  Without the prompt, Whisper occasionally outputs "the" instead of "thee" (KJV pronoun), mangles "Habakkuk", "Ecclesiastes", etc.  The prompt steers the decoder.
- ~3 lines of code, no risk.  Worst case: the prompt is silently ignored on a model that doesn't accept it.

**Cost:** Trivial.  Verify with a 2-minute test sermon clip after change.

---

## 2. Drop a short-input dictionary in front of `TranslationService.translate()`

**File:** new `src/pipeline/translate_short_dict.py`, plus 5 lines in `translation.py`.

**Change:** Before calling `model.generate()` for translation, look up the EN cue in a small exact-match dictionary.  If hit, return the dict value; if miss, proceed with NLLB / Opus-MT as today.

The batch project's dictionary (`scripts/pipeline/translate_short_dict.py` — copy verbatim, ~30 lines) handles 10 cues:

```python
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
```

**Why:**
- Both NLLB and Opus-MT reliably hallucinate on these short isolated utterances because there's no surrounding context.  Without the dict, `Amen.` becomes `- No lo sé.` (Spanish) or `Mèsi anpil.` (HT) — both wrong, both highly visible during a service.
- The 10 entries were validated through a multi-LLM voter panel (Qwen 2.5:7B/14B + gemma 9B + NLLB back-translation) over a 10-sermon corpus.  See ADR-029 in `~/Projects/Multi-Bitrate-Sermons/DECISIONS.md` for the full rationale.
- Live translator amplifies the impact: a mistranslated `Amen.` mid-sermon plays through the room while the speaker pauses for response.  Catching it adds zero latency (dict lookup is O(1)).

**Cost:** ~30 lines + integration.  Maintenance: re-survey after every ~10 new sermons (the batch project does this).

---

## 3. Port the post-translation glossary regex pass

**File:** new `src/pipeline/translate_glossary.py` + 2 lines in `translation.py` after `model.generate()`.

**Change:** After NLLB/Opus-MT produces output, run a regex find-and-replace pass on it.  Two entry classes:
1. **Output-targeted**: catch generic NLLB failure-mode outputs that recur regardless of source.  Example: `^¿Qué es eso?$` → `Bueno.` (NLLB renders many short interjections this way).
2. **Input-targeted**: anchored to the specific NLLB output that a known EN input produces.

The batch project's glossary lives in `scripts/pipeline/translate_glossary.py` — pick the 5–10 most universal entries.  Don't port the input-targeted ones that overlap with the dictionary in #2 above (those are now redundant).

**Why:**
- Even with the dictionary, NLLB still produces failure modes that the dictionary can't catch in advance (long inputs that NLLB botches in a recognizable way).
- Glossary entries are deterministic, debuggable, ship as source rather than as opaque model state — easy to add/remove based on user reports.

**Caution from our project:** word-swap entries are higher-risk than full-sentence anchors.  If a noun gets swapped, audit gender/number agreement.  We had a bug where `cusp` → `borde` produced `en la borde` (gender mismatch — `borde` is masculine but the article was feminine).  Fix required article-bearing patterns instead of bare-word substitution.

**Cost:** ~50 lines + 5–10 entries to start.  Low risk if you keep entries to full-sentence-anchored regexes.

---

## 4. Native systemd watchdog instead of polling restart

**Files:** `systemd/translator.service` + `src/pipeline/coordinator.py` (add a daemon thread).

**Change:** Add to `translator.service`:
```ini
WatchdogSec=600
NotifyAccess=main
Restart=always
```
And in `coordinator.py`, start a daemon thread that calls `systemd.daemon.notify("WATCHDOG=1")` every 30s for the lifetime of the coordinator.

**Why:**
- Current setup uses `Restart=on-failure`.  systemd notices process *exits* but does not notice process *hangs*.  The batch project hit a real ROCm/HIP kernel-queue stall on 2026-05-06 — the worker was alive (180% CPU spinning in `_beam_search`) but completely wedged for 1h33m before manual intervention.  systemd would have left it sitting silently.
- The daemon-thread heartbeat works because PyTorch C++ kernels release the GIL, so a daemon Python thread keeps getting scheduled even when the main thread is busy in a long `model.generate()` call.  Empirically validated in our project — heartbeat stays at 1–3 s old throughout NLLB translate on the AMD iGPU.
- Native systemd path is cleaner than the cron-based supervisor we use (because translator is a single-unit service, not a fleet of workers).  No extra scripts, no cron entries.
- The `Restart=always` change means systemd also restarts after kernel OOM-kill (which we hit on the batch side when NLLB-3.3B + concurrent encode pushed past 16 GB).

**Cost:** ~10 lines of code + 3 lines of systemd config.  See ADR-031 in our DECISIONS.md for the full rationale on the three-layer scheme.

---

## 5. Whisper hang timeout (in-process)

**File:** `src/pipeline/asr_process.py` (parent side, since translator already runs Whisper in a child process).

**Change:** When the parent submits an audio chunk, also start a per-chunk timer.  If the result doesn't arrive within `audio_duration × N` seconds (suggest N=4, floor 600 s), call `self._process.terminate()` and start a fresh subprocess.

```python
# Pseudocode in ASRProcess.submit() / get_result()
deadline = time.monotonic() + max(audio_duration * 4.0, 600.0)
result = self._output_q.get(timeout=deadline - time.monotonic())
# on timeout: self._process.terminate(); self._restart_subprocess()
```

**Why:**
- Even with the systemd watchdog from #4, a Whisper hang inside the child process is invisible to the heartbeat thread (which runs in the parent).  In our project we put a `threading.Timer` *inside* the Whisper-running process to `os._exit(2)` on timeout — but translator's child-process design lets the parent do this more cleanly without losing the parent's state.
- Real-world frequency: rare but non-zero.  We hit one in ~50 sermons.  When it happens, the cost is hours of silent failure, which is unacceptable for live mode.

**Cost:** ~30 lines (timer + restart logic).  Test by injecting a synthetic infinite-loop Whisper call and verifying recovery.

---

## 6. RAM-budget warning when NLLB is enabled

**File:** `src/pipeline/translation.py` (`TranslationService.load()`, the NLLB branch).

**Change:** Right before `from_pretrained(NLLB-3.3B)`, check `psutil.virtual_memory().available`.  If < 14 GB and the requested model is `nllb-200-3.3B`, log a loud warning and recommend `nllb-200-distilled-1.3B` instead.

**Why:**
- We empirically OOM'd on this exact pattern: 16 GB RAM box + concurrent encode worker + NLLB-3.3B.  `transformers.from_pretrained` transiently uses ~14.8 GB RSS during the fp32→fp16 conversion in-place (we measured this on `old-translate` 2026-05-06).  Add anything else running and the kernel OOM-kills the process.
- Distilled-1.3B is a great middle ground for live mode: ~2.5 GB peak load, only marginally lower HT quality than 3.3B, and on the GPU it's ~5× faster.  Much better than the Opus-MT default for HT (Opus-MT en-ht is the weakest of the Opus-MT models).

**Cost:** ~5 lines.  Suggest also documenting in README that NLLB-3.3B requires ≥ 16 GB free RAM at load time, distilled-1.3B requires ~3 GB.

---

## 7. Recommend NLLB-200 distilled-1.3B for HT (TODO Phase 1)

**File:** TODO.md Phase 1 + `scripts/download_models.py` + the README's NLLB section.

**Change:** Where TODO Phase 1 currently says "Replace `opus-mt-en-ht` with Facebook NLLB-200 distilled 600M (same as old system)", consider distilled-1.3B instead of 600M.

**Why:**
- We A/B tested NLLB sizes on Haitian Creole (10-sermon corpus, 2026-04-30 to 2026-05-05):
  - 600M: distilled-loss noticeable on HT specifically; many idioms get garbled
  - 1.3B: clear quality jump for HT; Spanish slightly better too
  - 3.3B: marginal gain over 1.3B on HT, but ~3× the runtime and ~3× the VRAM
- 1.3B is the sweet spot for HT without blowing live mode's latency budget.  Per-sentence runtime on the AMD iGPU is ~1 s vs 0.5 s for 600M (acceptable in live mode); on RTX 3060 it's ~0.2 s.
- Memory: ~2.5 GB fp16 vs ~1.2 GB for 600M vs ~6.6 GB for 3.3B.  Comfortable on 16 GB RAM live boxes.

**Cost:** Documentation update + one line in `download_models.py`.

---

## 8. Consider Kokoro for Spanish TTS (TODO Phase 4)

**File:** TODO Phase 4 evaluation.

**Change:** Where Phase 4 considers Coqui xtts_v2 for Spanish, also evaluate **Kokoro 82M** (`hexgrad/Kokoro-82M`, voice `em_alex` — see batch project's `scripts/pipeline/tts.py`).

**Why:**
- We measured Kokoro at **RTF 7.95×** on CPU (Ryzen AI 9 HX 370) — significantly faster than Piper, comparable subjective quality, and CPU-only so no ROCm headaches.
- xtts_v2 has known ROCm issues (the batch project rejected it for that reason — see ADR-028).  Coqui's TODO entry says "Test ROCm compatibility before committing to this on the 890M machine" — Kokoro sidesteps that risk entirely.
- Kokoro is 82M parameters vs xtts_v2's ~330M — much smaller install footprint.

**Cost:** A/B test on CPU: same Spanish input through Kokoro and Piper, listen to both, pick.  ~1 hour.

**Caveat for live mode:** Kokoro per-sentence latency on CPU is ~50-150 ms vs Piper's ~30-80 ms.  Probably acceptable but verify.

---

## 9. AMD Ryzen AI NPU for Whisper — not viable today, but track

**File:** TODO.md (add as Phase 7 or note).

**Decision:** Don't pursue NPU acceleration of Whisper on the 890M boxes today.  Re-evaluate when AMD ships:
1. Linux support for `large-v3` on the NPU (currently only base/small/medium are supported per the September 2025 announcement)
2. KV caching enabled (currently disabled — limits performance)
3. Either Ubuntu 24.04 migration on the runtime boxes OR Debian 12 official support from AMD

**Why this came up:** The article at https://www.amd.com/en/developer/resources/technical-articles/2025/unlocking-on-device-asr-with-whisper-on-ryzen-ai-npus.html plus the Linux install docs at https://ryzenai.docs.amd.com/en/latest/linux.html were reviewed against our setup (Ryzen AI 9 HX 370 + Radeon 890M, Debian 12).

**Specific blockers:**
- AMD Linux flow requires Ubuntu 24.04 (we run Debian 12).  Probably possible to graft drivers onto Debian, but unsupported.
- AMD Linux flow requires kernel ≥ 6.10 (Debian 12 ships 6.1).  Backporting kernel = significant scope.
- AMD recommends 64 GB RAM for the model compilation step (we have 16–32 GB).
- AMD requires Python 3.12 (we use 3.11 because of torch+rocm wheel pinning).
- The NPU does NOT support Whisper `large-v3` — only base / small / medium.  Live translator currently uses `large-v3`.
- Performance ceiling: AMD's published RTF for Whisper-Small on NPU is **1.2** (slower than real-time).  Our measured RTF for Whisper-large-v3 on the 890M iGPU via PyTorch ROCm is ~**1.0–1.6**.  Even at quality parity, the NPU isn't faster.

**The NPU's claimed advantage** (offload from CPU/GPU, lower power) doesn't pay off in our live setup because the iGPU isn't shared with another GPU workload, and CPU isn't the bottleneck during live translation.

**Re-check in 6 months** (April 2026) for AMD's progress on the three blockers above.

---

## 10. Add libcuda.so symlink to the NVIDIA install path

**File:** `install.sh --cuda` branch (or `scripts/install_rocm.sh` parallel for CUDA).

**Change:** After installing the NVIDIA driver but before any `pip install torch`, ensure the unversioned `libcuda.so` symlink exists:
```bash
if [ ! -L /usr/lib/x86_64-linux-gnu/libcuda.so ] && [ -f /usr/lib/x86_64-linux-gnu/libcuda.so.1 ]; then
    sudo ln -sf libcuda.so.1 /usr/lib/x86_64-linux-gnu/libcuda.so
fi
```

**Why:**
- Fresh Debian/Ubuntu installs ship `libcuda.so.1` and `libcuda.so.<version>` but no unversioned `libcuda.so`.  PyTorch's Triton JIT compiles CUDA kernels at first GPU op via the linker, which looks for `libcuda.so` (no version).  Without the symlink: `cannot find -lcuda` linker errors flood the log every Whisper call.
- We hit this on `old-translate` (RTX 3060, fresh Debian 12).  The errors don't break Whisper (PyTorch falls back to non-Triton kernels) but they pollute logs and slow down the first call.
- One-line fix.  Idempotent (the `[ ! -L ... ]` test).

**Cost:** Trivial.

---

## 11. Sentence buffer: add a `max_buffer_chars` safety valve

**File:** `src/pipeline/sentence_buffer.py`.

**Change:** Add `max_buffer_chars` (suggest default 800) as a fourth flush trigger.  When the joined fragments exceed this length, emit immediately regardless of timing.

**Why:**
- Currently `hard_timeout=10s` is the only buffer-growth bound.  If a speaker pauses every 5–8 s the buffer can accumulate many fragments without ever tripping the hard timeout.
- 800 chars ≈ a long paragraph.  Beyond that, NLLB's quality starts dropping anyway (it was trained on sentence-level pairs).
- ~5-line change.

**Cost:** Trivial.

---

## 12. Hallucination filter on translation output too

**File:** `src/pipeline/translation.py` (after `tokenizer.decode()`).

**Change:** Reuse the existing `_is_hallucination()` heuristic from `asr.py` on translation output.  If the translation matches the pattern, return empty (skip TTS for this cue).

**Why:**
- You already filter Whisper hallucinations.  NLLB has its own loop modes — "Mr. Mr. Mr. Mr." style spirals that the `repetition_penalty=1.3 + no_repeat_ngram_size=3` settings catch *most* of the time but not always.
- Trivial reuse — the helper is already there in `asr.py`.

**Cost:** ~10 lines (import + call).

---

## What this document does NOT recommend

- **Switching ASR backend** (faster-whisper vs whisper-transformers vs Parakeet).  Current setup is fine; trade-off is clear and the existing config keeps both available.
- **Removing the stereo-channel split.**  Our pipeline doesn't have an equivalent — it's a clever solution to the single-output-device constraint.  Keep.
- **Adding a sheet-as-queue layer.**  Live translator is single-machine, single-purpose.  No benefit.
- **Adding HLS / segment streaming.**  Live mode pushes audio out a speaker, not over HTTP.  Different problem.

---

## Provenance

These suggestions come from the operational data and decision history of the sister project at `~/Projects/Multi-Bitrate-Sermons` (see its `DECISIONS.md` for ADR-001 through ADR-031, and `chunks/` for chunk-by-chunk design history).  Specific ADRs cited above:
- ADR-026: NLLB-200 + local LLM for transcripts; preserve `transcription.json` for re-translation
- ADR-028: TTS execution + Kokoro performance numbers
- ADR-029: Translation pipeline + dictionary + glossary, multi-LLM voter panel methodology
- ADR-031: ASR worker liveness + heartbeat + watchdog three-layer scheme

The batch project also has memory entries useful for context:
- `feedback_benchmark_isolation.md`: how to measure model speed without contention
- `project_old_translate_pc.md`: the RAM-budget OOM observation referenced in #6
- `feedback_no_paid_apis.md`: why hosted APIs are off the table (same constraint applies to live translator)
