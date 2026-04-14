# ASR A/B Test — Batch vs Streaming

Compare Batch and Streaming ASR modes on the same source audio using an
authoritative reference transcript. Measures English ASR accuracy (WER) and
end-to-end latency for both output languages.

There are two ways to run this:

- **Automated (recommended)** — `run_ab_tests.sh` runs 8 back-to-back tests
  (JFK × 4 + Jobs × 4) unattended, with audio played locally and the
  translator controlled remotely over SSH. ~2h15m total.
- **Manual** — you drive each run by hand with the `./translator` launcher,
  changing mode in the menu each time.

---

## Automated mode (8 runs in ~2h15m)

### One-time setup on the local (playback) machine

1. **Both audio files** staged at `tests/ab_test/audio/`:
   - `JFK_Inaugural_Address.mp3` (~14.6 min)
   - `Steve_Jobs_Stanford.mp3` (~15.1 min, from YouTube `UF8uR6Z6KLc`)

   These are gitignored (large, copyright-hygiene) so they must be present
   on the local playback machine but are not committed.

2. **`ffplay`** installed (`apt install ffmpeg`).

3. **SSH key** at `/home/administrator/.ssh/mykey` authenticated to the
   test machine at `10.1.170.184`. Edit the top of `run_ab_tests.sh` if
   your host/user/key differs.

### One-time setup on the test (translator) machine

1. **`jiwer` installed** in the translator venv:
   ```bash
   ~/translator/venv/bin/pip install jiwer
   ```
2. **Both reference transcripts populated**:
   - `tests/ab_test/reference_jfk_inaugural.txt` — already populated.
   - `tests/ab_test/reference_jobs.txt` — paste the speech text from
     <https://news.stanford.edu/2005/06/14/jobs-061505/> after the
     `---REFERENCE BEGINS---` marker.
3. **Saved settings** (`config/settings.yaml`) — audio input/output devices
   already configured via `python run.py --setup`.

### Physical setup before you leave

1. Position the mic on the test machine near the local machine's speakers
   (or route a 3.5mm cable — test machine mic input hears whatever plays).
2. Set speaker volume so the mic reads clearly but doesn't clip.
   Test with one short playback and check `python run.py --verbose` briefly.
3. **Turn off** any notifications/sounds on the local machine (email pings,
   IM, Slack, etc.) — anything the mic hears during the 2 hours corrupts
   the transcription.
4. Disable screensaver/suspend on both machines or audio may cut out.

### Launch

```bash
cd ~/translator
bash tests/ab_test/run_ab_tests.sh 2>&1 | tee tests/ab_test/run_log.txt
```

Now walk away. Total time ≈ 2h15m.

The script runs 8 tests:

| # | Source | Mode |
|---|--------|------|
| 1 | JFK | Batch |
| 2 | JFK | Batch |
| 3 | JFK | Streaming |
| 4 | JFK | Streaming |
| 5 | Jobs | Batch |
| 6 | Jobs | Batch |
| 7 | Jobs | Streaming |
| 8 | Jobs | Streaming |

### After it finishes

Results land in `tests/ab_test/results/` on the local machine (SCP'd back
from remote):

- `runN_src_mode.log` — raw translator log per run
- `analysis/runN_src_mode.txt` — full analyzer output per run
- `SUMMARY.txt` — headline numbers for all 8 runs, side-by-side

Read `SUMMARY.txt` first.

---

## Manual mode

If you want to drive runs by hand (for a single-source test or
experimentation):

### Prepare the reference

Open `reference_jfk_inaugural.txt` (or `reference_jobs.txt`) and paste the
canonical transcript after the `---REFERENCE BEGINS---` marker. Lines
starting with `#` are ignored.

The analyzer lowercases, strips punctuation, and collapses whitespace on
both sides — no manual normalization needed.

### Run each mode

```bash
cd ~/translator
./translator         # Menu: pick 1 (Batch) or 2 (Streaming)
# Play the source on TV start-to-finish. Ctrl+C when the speech ends.
```

Each run produces `logs/translator_YYYY-MM-DD_HH-mm-ss.log`. Run each mode
at least **twice** — first run has model-warmup artifacts.

### Analyze

```bash
source venv/bin/activate
python tests/ab_test/analyze.py \
    logs/translator_<TIMESTAMP>.log \
    tests/ab_test/reference_jfk_inaugural.txt
```

The analyzer auto-detects batch vs streaming from log content.

---

## Interpreting the output

```
English ASR accuracy (vs reference):
  WER           : 8.34%           ← word error rate, lower is better
  substitutions : 42              ← words replaced (misheard)
  deletions     : 18              ← words missed entirely
  insertions    : 5               ← words ASR added that weren't said
  ref words     : 1364            ← reference word count
  hyp words     : 1351            ← ASR word count
```

**WER bands** (for real-room audio off speakers):
- `< 5%`  — excellent; publication-quality transcription.
- `5–10%` — good; occasional minor mistakes, fine for live translation.
- `10–20%` — acceptable for casual use; some meaning loss.
- `> 20%` — poor; review mic placement, volume, or model size.

```
ASR latency:
  asr_time      : n=45   mean=0.612s  median=0.580s  p95=0.920s  max=1.10s
  chunk_dur     : n=45   mean=6.40s   median=5.80s   p95=9.80s   max=10.10s
  RTF           : n=45   mean=0.095   median=0.092   p95=0.141   max=0.17
```

**RTF (Real-Time Factor)** = `asr_time / chunk_dur`. Must stay `< 1.0`
on average or ASR falls behind. `< 0.2` is the sweet spot.

```
[ES] pipeline latency:
  e2e           : n=43   mean=7.20s   median=6.90s   p95=11.4s   max=15.8s
  translate     : n=43   mean=0.510s  ...
  tts           : n=43   mean=0.180s  ...
  queue_was     : n=43   mean=0.0     ...            ← 0 = no backlog
```

- **e2e** is full latency from chunk start → audio ready to play.
- **queue_was > 0** means the pipeline was already behind when the event
  landed. Watch for backlog growth across the run.

## Comparing batch vs streaming

| Aspect | Batch | Streaming |
|---|---|---|
| **Accuracy (WER)** | Usually lower — full silence-delimited chunks | Usually higher — rolling window re-transcribes partials |
| **Latency (e2e)** | Higher — waits for silence | Lower in principle — emits sooner |
| **Queue backlog risk** | Lower — predictable cadence | Higher — depends on rolling buffer pacing |
| **Short utterances** | Good | Can be unstable |
| **Long pauses** | Good (natural cut) | Sometimes drops text at boundaries |

If streaming shows noticeably higher WER **and** equal/higher e2e, batch
is the right default for that source. If streaming is close on WER and
meaningfully lower on e2e, it may be worth using for sermons with tight
back-and-forth pacing.
