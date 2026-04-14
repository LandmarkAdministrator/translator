# ASR A/B Test — Batch vs Streaming

Compare Batch and Streaming ASR modes on the same source audio using an
authoritative reference transcript. Measures English ASR accuracy (WER) and
end-to-end latency for both output languages.

## What you need

1. **A TV/speaker to play back the source audio**, and a mic picking it up
   (normal room setup — this tests real acoustic pickup, not file-fed audio).
2. **Two contrasting sources** with known, human-verified transcripts.
   Recommended pair:
   - **JFK Inaugural Address** (~14 min) — formal oratory, many pauses.
     Transcript: `reference_jfk_inaugural.txt` (already populated).
   - **Steve Jobs Stanford Commencement** (~15 min) — storytelling pace,
     fewer pauses, conversational. Paste into `reference_source2.txt`
     from <https://news.stanford.edu/2005/06/14/jobs-061505/>.

   Testing both gives you a read on how each ASR mode behaves across
   pacing styles — the formal speech stresses silence-based chunking
   (batch), the conversational delivery stresses the rolling window
   (streaming).
3. **`jiwer`** for WER:
   ```bash
   source ~/translator/venv/bin/activate
   pip install jiwer
   ```
   The analyzer falls back to a built-in Levenshtein WER if jiwer isn't
   installed, but jiwer is faster and gives identical numbers.

## Prepare the reference

Open `reference_jfk_inaugural.txt` and paste the canonical transcript
**after** the `---REFERENCE BEGINS---` marker. Everything before the marker
(and any line starting with `#`) is ignored by the analyzer.

You don't need to strip punctuation or normalize — the analyzer does it.

If you use a TED talk or different speech, just create another reference
file the same way (header optional — a plain text file works fine).

## Run the test

Keep environmental conditions **identical** across runs: same TV volume,
same mic position, same room, same time of day (HVAC noise matters).

### Run 1 — Batch mode

```bash
cd ~/translator
./translator
# When prompted, choose 1 (Batch). Play the source on TV start-to-finish.
# Press Ctrl+C when the speech ends.
```

### Run 2 — Streaming mode

```bash
cd ~/translator
./translator
# Choose 2 (Streaming). Play the same source the same way.
```

Each run produces a log at `~/translator/logs/translator_YYYY-MM-DD_HH-mm-ss.log`.

**Tip:** Run each mode **at least twice** — first run can include model
warmup artifacts. Compare the second runs.

## Analyze results

```bash
cd ~/translator
source venv/bin/activate

# Batch result
python tests/ab_test/analyze.py \
    logs/translator_<BATCH_TIMESTAMP>.log \
    tests/ab_test/reference_jfk_inaugural.txt

# Streaming result
python tests/ab_test/analyze.py \
    logs/translator_<STREAMING_TIMESTAMP>.log \
    tests/ab_test/reference_jfk_inaugural.txt
```

The analyzer detects batch vs streaming automatically from the log content.

## What the output means

```
English ASR accuracy (vs reference):
  WER           : 8.34%           ← word error rate, lower is better
  substitutions : 42              ← words replaced (misheard)
  deletions     : 18              ← words missed entirely
  insertions    : 5               ← words ASR added that weren't said
  ref words     : 1364            ← reference word count
  hyp words     : 1351            ← ASR word count
```

**WER interpretation** (ballpark, for real-room audio off a TV):
- `< 5%`  — excellent; transcription is publication-quality.
- `5–10%` — good; occasional minor mistakes, fine for live translation.
- `10–20%` — acceptable for casual use; some meaning loss.
- `> 20%` — poor; review mic placement, volume, or model size.

```
ASR latency:
  asr_time      : n=45   mean=0.612s  median=0.580s  p95=0.920s  max=1.10s
  chunk_dur     : n=45   mean=6.40s   median=5.80s   p95=9.80s   max=10.10s
  RTF           : n=45   mean=0.095   median=0.092   p95=0.141   max=0.17
```

- **RTF (Real-Time Factor)** = `asr_time / chunk_dur`. Must stay `< 1.0`
  on average or ASR will fall behind. `< 0.2` is the sweet spot.

```
[ES] pipeline latency:
  e2e           : n=43   mean=7.20s   median=6.90s   p95=11.4s   max=15.8s
  translate     : n=43   mean=0.510s  ...
  tts           : n=43   mean=0.180s  ...
  queue_was     : n=43   mean=0.0     ...            ← 0 = no backlog
```

- **e2e** is the full latency from chunk start → audio ready for playback.
- **queue_was > 0** means the pipeline was already behind when this event
  landed — watch for backlog growth across the run.

## Comparing batch vs streaming — what to look for

| Aspect | Batch | Streaming |
|---|---|---|
| **Accuracy (WER)** | Usually lower — has full silence-delimited chunk context | Usually higher — rolling window, re-transcribes partials |
| **Latency (e2e)** | Higher — waits for silence to cut chunks | Lower in principle — emits sooner |
| **Queue backlog risk** | Lower — predictable chunk cadence | Higher — depends on rolling buffer pacing |
| **Quality on short utterances** | Good | Can be unstable |
| **Quality on long pauses** | Good (natural cut) | Sometimes drops text at boundaries |

If streaming shows noticeably higher WER **and** equal/higher e2e, batch is
the right default for this source. If streaming is close on WER and
meaningfully lower on e2e, it may be worth using for sermons with tight
back-and-forth pacing.
