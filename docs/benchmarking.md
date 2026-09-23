# Benchmarking the live stack

**Written 2026-09-21.** Purpose: replace guesses about hardware with
measurements, so we can tell a prospective church "this box does three
languages, that one does five" and be right.

## The question this answers

Each language runs in its **own thread, with its own queue, its own
`TranslationService` and its own `TTSService`** (`LanguagePipeline` in
`coordinator.py`). So three languages do not cost three times one — they
overlap. How much they overlap is a property of the hardware, and it is
the number that decides how many languages a given box supports.

The benchmark measures it directly:

```
overlap = sum(stage durations) / (wall-clock time those stages occupy)
```

- **overlap ≈ 3.0** with three languages → truly parallel; a language is
  nearly free, and the limit is somewhere other than the GPU
- **overlap ≈ 1.0** → fully serialized; each language costs its full time
- **in between** → partial, which is what real hardware usually gives

## Setup

Nothing to install. Profiling is off unless `TRANSLATE_PROFILE` is set,
and the benchmark sets it. `src/pipeline/profiling.py` is a no-op
otherwise, so this is safe to leave in the production path.

Check the analysis maths first — it runs in milliseconds and needs no GPU:

```bash
./venv/bin/python tests/test_benchmark_stack.py
```

## Step 0: check the machine before trusting any number

**Run this first, on both machines, and keep the output with the
results.**

```bash
./venv/bin/python tests/device_report.py
```

It confirms torch sees the GPU, times a real matmul on GPU against CPU
(a HIP build that has silently fallen back to the host still *reports* a
cuda device, so placement alone proves nothing), checks whether
onnxruntime has any accelerated provider, and reports which ASR model
this box would actually use.

**ASR on CPU is expected, not a fault.** `tests/benchmarks/REPORT_20260515.md`
measured Parakeet TDT v3 at RTF 0.027 on CPU — 37x realtime — against
0.005 on CUDA, and that report's own production recommendation puts
Parakeet on CPU deliberately, to keep the GPU free for translation. A
CPU-only onnxruntime is the intended configuration.

The trap it catches is different: **a different ASR model entirely.** The NeMo unified Parakeet needs its
  own Python 3.11 venv, and `TODO.md` item 3 records that on ROCm hosts
  that venv is still built by hand. A box falling back to onnx-asr TDT is
  running a *different model* with no punctuation-driven sentence
  boundaries — which changes how many sentences reach translation, and
  therefore every downstream number

If the two machines disagree on either, the comparison measures the
software difference, not the hardware. Fix it or note it before running.

The benchmark embeds this report in every result JSON, and prints any
WARN or FAIL alongside the numbers, so a run can never quietly lose its
provenance.

## The audio

The default is the mixed-content suite already in the repo:
`tests/ab_test/audio/suite_v4.wav`, built by
`tests/comparison/build_suite.py`. 56 minutes across five tracks:

| Track | Length | Why it is in there |
|---|---|---|
| jfk | 14.0 min | Clear oratory, a known reference transcript |
| sermon | 10.0 min | The actual workload — your preacher, your mixer feed |
| jobs | 15.1 min | Different voice, different cadence |
| singing | 5.4 min | Congregational singing: the ASR's hardest case |
| libri-mix | 10.7 min | Clean read speech, the easy baseline |

Four configurations over the whole file is about four hours in realtime
mode, so **the sermon track is the default** — it is the representative
workload and makes a full comparison roughly an hour.

```bash
--track sermon   # default
--track singing  # the hard case
--track all      # the whole 56 minutes, for an overnight run
```

The cut is cached under `benchmarks/audio/`, so every configuration chews
byte-identical audio.

## Run it on the RTX 3060 (Translate PC)

```bash
cd ~/translator
source scripts/run_production.sh --print-env 2>/dev/null || true
./venv/bin/python tests/device_report.py
./venv/bin/python tests/benchmark_stack.py --suite compare --tag 3060
```

That runs four configurations: one language and three, each in realtime
and throughput mode. Expect roughly 4× the recording's length plus the
throughput runs, so budget an hour for a 15-minute sermon.

Do it when nothing else is on the GPU — stop the archive worker first, or
the numbers measure contention rather than the stack.

## Run it on the 890M (the laptop)

The Translate PC has its iGPU memory allocation reduced, so the 890M
measurement has to happen on the laptop.

```bash
cd ~/translator
./venv/bin/python tests/device_report.py
./venv/bin/python tests/benchmark_stack.py --suite compare --tag 890m
```

What the device report should say here: `run_production.sh` exports
`NLLB_DEVICE=cuda`, `KOKORO_DEVICE=cuda` and `MMS_DEVICE=cuda`, and on
ROCm torch maps `cuda` to HIP — so translation and both TTS backends do
use the 890M. That part is fine and is confirmed by the archive project's
own note that HIP mapping works across thousands of jobs.

ASR is the one to check, for the two reasons above.

Also: **plug the laptop in and set the governor to performance.** A
thermal- or power-throttled measurement is not the appliance's ceiling,
and a laptop on battery will quietly give you one.

## Compare

```bash
./venv/bin/python tests/benchmark_stack.py --report benchmarks/
```

Copy the `benchmarks/*.json` from one machine to the other, or to a third
place, and report over both at once.

## Reading the output

### Expect translation to be serialized

Before reading any overlap number, know what the code does.

`translation.py` shares one set of NLLB weights across every language
pipeline and guards `generate()` with a **single shared lock** — because
concurrent `generate()` on a shared model segfaults on ROCm. So the three
language threads queue up behind each other for translation.

TTS does not: each language owns its own `TTSService` instance, so those
genuinely overlap.

| Stage | Expected overlap with 3 languages | Why |
|---|---|---|
| `translate` | near **1.0x** | One shared model, one shared lock |
| `tts` | **above 1.0x** | Per-language instances, no shared lock |

If `translate` overlap comes back near 3.0x, something has changed in how
the weights are cached and it is worth understanding before trusting the
rest.

**This is also the biggest single lever on language count.** Per-language
translation models — Opus-MT already works this way, and M2M-100 418M
would — use a per-instance lock with no contention. Three 418M models is
comparable memory to one NLLB-1.3B and would translate three languages in
parallel instead of in series.

| Line | What it means |
| --- | --- |
| `OVERLAP: 3.0x` | The three languages ran genuinely concurrently |
| `duty cycle` | Fraction of the run with work in flight. Near 1.0 = saturated |
| `queue depth (0.8 -> 3.2)` | First half vs second half. Rising = falling behind |
| `speed-ups: 40%` | How often adaptive speed kicked in to catch up — a stress signal |
| `each extra language adds ~X%` | The headline: the marginal cost of a language |

**Realtime mode answers "does it work".** If queue depth climbs through
the run, the box is over-subscribed at that language count, no matter how
good the average latency looks.

**Throughput mode answers "by how much".** With `--no-realtime` the file
is fed as fast as the stack will take it, so the result is the ceiling.
The ratio between throughput wall time and the recording's real duration
is roughly the headroom factor.

## Turning this into a sales sheet

Once both machines are measured, the recipe for any new candidate box:

1. Run `--suite compare` with the same sermon
2. Read the marginal per-language cost and the realtime queue trend
3. The supported language count is the largest N where realtime queue
   depth stays flat, minus one for margin

That gives a defensible table: *this hardware supports N simultaneous
languages; for more you need the next tier*. It is also the honest answer
to a church that asks for a language we have not benchmarked.

## What this does not measure

- **The livestream packager's load.** It runs on VCN, separate silicon,
  and does not touch these numbers. Benchmark it separately if the
  appliance runs both at once
- **The sermon archive worker.** It is meant to pause during services;
  if an appliance runs it concurrently, that is a third measurement
- **Quality.** This is purely about speed. Translation quality after a
  model swap is `scripts/eval/run_multi_eval.py` in the archive project
- **Cold start.** Model loading happens before the first sentence and is
  excluded; it matters for restart-during-service, not for throughput
