# Runbook: translation model + stack benchmarks

**Written 2026-09-22.** Self-contained: everything needed to run these on
both machines and collect the results. No prior conversation required.

Intended reader: whoever (or whatever) has SSH to the fleet.

---

## Why these runs exist

Two separate questions, two separate harnesses.

**1. Can we replace NLLB-200?** NLLB-200 and MMS-TTS are CC-BY-NC-4.0 —
non-commercial. That blocks selling an appliance. The permissive
candidates are M2M-100 (MIT) and MADLAD-400 (Apache 2.0), both of which
cover `ht` and `ru`. We need speed, memory and quality numbers before
committing. → `tests/translate_bench.py`

**2. What does each additional language actually cost, and how does the
890M compare with the RTX 3060?** → `tests/benchmark_stack.py`

There is a specific thing to look for in (1), explained under *Reading
the results* below: translation is currently **serialized** across
languages, and per-language model instances may remove that.

---

## Rules

1. **Never run these during a service window.** `run_bench_safely.sh`
   refuses if `translate.service` is active, but check the schedule too.
2. **The sermon archive worker must be stopped.** The wrapper does this,
   waits for it, verifies it, and restarts it afterwards — including on
   Ctrl-C. Do not bypass with `--no-stop` unless you stopped it yourself.
3. **Do not modify `src/pipeline/translation.py`.** The benchmark loads
   models directly and deliberately does not touch the production path.
4. **Both machines must run the same code and the same input**, or the
   comparison is meaningless.

---

## Files this needs

New or changed. The laptop (`it-vivobook-s14`,
`~/Projects/translator`) already has all of them.

| Path | What |
|---|---|
| `tests/translate_bench.py` | model × device benchmark + quality dump |
| `tests/benchmark_stack.py` | full-stack benchmark |
| `tests/test_benchmark_stack.py` | unit tests for the analysis maths |
| `tests/device_report.py` | pre-flight: what is really on the GPU |
| `src/pipeline/profiling.py` | span recorder (no-op unless `TRANSLATE_PROFILE` is set) |
| `src/pipeline/coordinator.py` | **modified**: 4 lines, spans around translate/TTS |
| `scripts/overnight_translate_bench.sh` | the overnight loop |
| `scripts/run_bench_safely.sh` | wrapper: stop worker → bench → restart |
| `docs/benchmarking.md` | detail behind this runbook |
| `tests/comparison/suite/track_sermon.ref.txt` | **required input** — the English source text |

`tests/ab_test/audio/suite_v4.wav` (109 MB) is needed **only** for
`benchmark_stack.py`, not for the translation benchmark. Skip it unless
running the stack benchmark on that machine.

---

## Machine A — the laptop (ROCm + CPU)

```bash
cd ~/Projects/translator
chmod +x scripts/run_bench_safely.sh scripts/overnight_translate_bench.sh
./venv/bin/python tests/test_benchmark_stack.py     # should print 11 groups passed
./venv/bin/python tests/device_report.py            # read this before proceeding
./scripts/run_bench_safely.sh
```

Runs overnight. Leave it.

On this machine `--device cuda` **is ROCm** — torch maps cuda to HIP.
That is expected and correct; it is not a misconfiguration.

---

## Machine B — the RTX 3060 box (CUDA + CPU)

### Confirm which host that is first

`docs/DEPLOYMENT.md` describes the production translator as a Ryzen AI 9
HX 370 mini PC with an RTX 3060 over OCuLink. The sermons repo's README
lists `translate-pc` at 10.1.170.184 (Radeon 890M) and `old-translate` at
10.1.170.51 (RTX 3060). **These two records disagree**, probably because
hardware moved. Verify before deploying:

```bash
ssh administrator@<host> 'hostname; nvidia-smi --query-gpu=name --format=csv,noheader'
```

Use whichever box reports the RTX 3060.

### Deploy

The fleet boxes cannot `git pull` (operator note, sermons README), so
copy. Confirm the repo path on the target first — `docs/DEPLOYMENT.md`
installs to `~/translator`, which may differ from the laptop's
`~/Projects/translator`:

```bash
TARGET=administrator@<host>
REMOTE=~/translator                      # verify this path exists

rsync -av -e "ssh -i ~/.ssh/mykey" \
  tests/translate_bench.py \
  tests/benchmark_stack.py \
  tests/test_benchmark_stack.py \
  tests/device_report.py \
  "$TARGET:$REMOTE/tests/"

rsync -av -e "ssh -i ~/.ssh/mykey" \
  src/pipeline/profiling.py \
  src/pipeline/coordinator.py \
  "$TARGET:$REMOTE/src/pipeline/"

rsync -av -e "ssh -i ~/.ssh/mykey" \
  scripts/overnight_translate_bench.sh \
  scripts/run_bench_safely.sh \
  "$TARGET:$REMOTE/scripts/"

rsync -av -e "ssh -i ~/.ssh/mykey" \
  docs/benchmarking.md "$TARGET:$REMOTE/docs/"

# The English source text the translation benchmark reads.
rsync -av -e "ssh -i ~/.ssh/mykey" \
  tests/comparison/suite/ "$TARGET:$REMOTE/tests/comparison/suite/"

# Stale bytecode can mask new code (operator note, sermons README).
ssh "$TARGET" "find $REMOTE/src $REMOTE/tests -name __pycache__ -type d \
  -exec rm -rf {} + 2>/dev/null; true"
```

### Run

```bash
ssh administrator@<host>
cd ~/translator
chmod +x scripts/run_bench_safely.sh scripts/overnight_translate_bench.sh
./venv/bin/python tests/device_report.py
./scripts/run_bench_safely.sh
```

Use `tmux` or `nohup` — it runs for hours and an SSH drop would kill it:

```bash
tmux new -s bench './scripts/run_bench_safely.sh 2>&1 | tee /tmp/bench.log'
```

---

## Collect

Both machines write to `benchmarks/translate/`. Pull them into one place,
keeping them separate by host:

```bash
mkdir -p ~/bench-results/laptop ~/bench-results/3060
cp -r ~/Projects/translator/benchmarks/translate/* ~/bench-results/laptop/
rsync -av -e "ssh -i ~/.ssh/mykey" \
  administrator@<host>:~/translator/benchmarks/translate/ ~/bench-results/3060/

# One report across both
cd ~/Projects/translator
./venv/bin/python tests/translate_bench.py \
    --compare ~/bench-results/laptop ~/bench-results/3060 --samples 12
```

Each JSON carries its own `host`, `device` and full device report, so
runs cannot get confused once mixed.

---

## Reading the results

### Expect translation to be serialized today

`src/pipeline/translation.py` shares one set of NLLB weights across every
language pipeline and guards `generate()` with a **single shared lock**,
because concurrent `generate()` on a shared model segfaults on ROCm. So
three languages currently translate in series, not in parallel.

The `--threads` runs load a **separate model instance per language**,
which has no shared lock. The report's *SHARED MODEL vs PER-LANGUAGE
INSTANCES* section is the one to read first:

- **speedup > 1.4x** → per-language instances buy real parallelism. This
  matters more than the model choice, because it directly sets how many
  languages a box can carry. 3 × M2M-100 418M is roughly the memory of
  one NLLB-1.3B, so it may be close to free.
- **speedup ≈ 1.0x** → the GPU is the limit, not the lock, and languages
  stay roughly additive.

### Quality

The automated signals need no gold reference and catch the failures that
actually occur in this domain:

| Signal | Why it matters |
|---|---|
| number retention | "Daniel chapter 6 verse 10" losing its numbers is the failure people notice |
| loop rate | NLLB already spirals occasionally (`TODO.md` item 2) |
| copied-source rate | output identical to English = the model silently did not translate |
| length ratio | outliers flag truncation and runaway generation |

**None of these judge fluency.** The *SIDE BY SIDE* dump at the end of
the report is for that, and ultimately a native speaker is — the same
standing caveat as the Kreyòl numerals and the short-phrase dictionary.

### Device placement

For context, from `tests/benchmarks/REPORT_20260515.md`: Parakeet ASR runs
at **37× realtime on CPU** vs 200× on CUDA, and that report's production
recommendation deliberately places ASR on CPU to keep the GPU free for
translation. So CPU placement for a stage is often the right answer, not
a fallback. Expect the same question for translation.

---

## Known traps

**A different ASR model on each box.** The NeMo unified Parakeet
(`PARAKEET_MODEL=unified-remote`) needs a second Python 3.11 venv, and
`TODO.md` item 3 records that on ROCm hosts that venv is still built by
hand. If one box falls back to onnx-asr TDT it is running a *different
model* with no punctuation-driven sentence boundaries — which changes how
many sentences reach translation and therefore every downstream number.
`device_report.py` reports which one is active. This only affects
`benchmark_stack.py`; `translate_bench.py` does not use ASR.

**Model downloads.** First run pulls M2M-100 418M (~2 GB) and 1.2B
(~5 GB) from Hugging Face. `--nllb-only` banks the baseline first if
bandwidth is tight. `--with-madlad` adds a ~12 GB download.

**Re-running is safe.** Each configuration writes its own JSON and the
script skips anything already present. A crash costs only that run.
Delete a specific JSON to redo just that configuration.

**A long graceful stop is normal.** The archive worker finishes its
current sermon before exiting, and an encode job runs ~75 minutes. The
wrapper waits up to 90 minutes and reports progress. Raise with
`--wait=SECONDS`.

---

## Options

```bash
./scripts/run_bench_safely.sh --quick        # 30 sentences instead of 120
./scripts/run_bench_safely.sh --nllb-only    # baseline only, no downloads
./scripts/run_bench_safely.sh --with-madlad  # add the Apache 3B model
./scripts/run_bench_safely.sh --no-stop      # worker already stopped
./scripts/run_bench_safely.sh --no-restart   # leave the worker down
./scripts/run_bench_safely.sh --wait=10800   # allow 3h for a graceful stop
```

Single configuration, if something needs re-running by hand:

```bash
./venv/bin/python tests/translate_bench.py \
    --model m2m100-418m --device cuda --langs es ht ru \
    --sentences 120 --label m2m100-418m-cuda
```

Models: `nllb-1.3b`, `nllb-600m`, `m2m100-418m`, `m2m100-1.2b`,
`madlad-3b`, `opus-es`, `opus-ht`, or a raw Hugging Face id.

---

## The stack benchmark (separate, optional)

Answers the per-language-cost and 890M-vs-3060 questions for the *whole*
pipeline rather than translation alone. Needs `suite_v4.wav` on the
machine.

```bash
./venv/bin/python tests/benchmark_stack.py --suite compare --tag 3060
./venv/bin/python tests/benchmark_stack.py --report benchmarks/
```

Defaults to the 10-minute sermon track of the 56-minute suite;
`--track all` for the whole thing, `--track singing` for the hard case.
Full detail in `docs/benchmarking.md`.

---

## When it finishes

Report back with:

1. `benchmarks/translate/REPORT.txt` from both machines
2. The `device_report.py` output from both
3. Anything in the logs that looked wrong

The side-by-side Spanish and Creole output is the part that needs human
judgement; the rest is arithmetic.

---

## Run log — 2026-09-22 (translation benchmark, both machines)

Both machines ran the full default loop — 3 models × CPU/GPU × shared /
per-language instances, 120 sentences (342 translations per run), es/ht/ru —
with no failed configuration. Combined report:
`benchmarks/REPORT-combined-2026-09-22.txt` (per-machine results in
`~/bench-results/{laptop,3060}/` on the laptop).

**Which host.** The RTX 3060 is in the Translate PC, 10.1.170.184 (mini PC,
HX 370, card over OCuLink). `old-translate` 10.1.170.51 is decommissioned.
The sermons README is out of date on this point.

**Deviations from the steps above, and why:**

- *Not deployed into `~/translator` on the PC.* That is the production
  checkout, a clean git clone deployed with `git pull`; copying a modified
  `coordinator.py` into it would have changed the code the next service runs.
  The benchmark ran from `~/translator-bench` (a copy sharing the production
  venv; `models/translation` linked to `~/.cache/huggingface/hub`, where NLLB
  already is — the default cache dir there was empty and would have
  re-downloaded it). The PC *can* `git pull`; it just should not pull
  uncommitted benchmark code.
- *The worker was held through the scheduler, not by `stop.flag` alone.*
  Outside a window `translate-window-check.sh` clears `stop.flag` and
  relaunches the worker every five minutes (`docs/BACKLOG-CONTRACT.md`), so
  the runner set `~/translate-manual.flag` first, drained with `stop.flag`,
  ran `run_bench_safely.sh --no-stop --no-restart`, then removed both flags
  and let the scheduler relaunch the worker in its own unit. The wrapper's
  own restart calls `lbc-start-unified-worker.sh`, which uses `nohup`: inside
  a systemd job that worker would be killed when the job ends.
- *Laptop:* the two scripts were not executable yet; `.env.rocm` sourced
  with `set +u` (it references an unset `LD_LIBRARY_PATH`); performance power
  profile for the run, restored afterwards.

**Fixes to the benchmark code (applied on both machines, measurements untouched):**

- `device_report.py`: the GPU sanity matmul now runs fp16 on the GPU. At fp32
  the Radeon 890M is only ~1.3× the CPU, so a healthy iGPU was reported as
  a "silent CPU fallback"; at fp16 (what the benchmark and production load
  NLLB in) it is 4–7×. A real fallback is still caught.
- `translate_bench.py --compare`: skipped `device_report.json` (it crashed
  on it at the end of every run), and now keys everything by host — before,
  combining two machines let one machine's `cuda` run stand in for the
  other's. Adds a same-configuration-on-each-machine table and names each
  machine's GPU.

**Results (mean seconds per translation, one shared model):**

| model | RTX 3060 | Radeon 890M | CPU (HX 370) | VRAM |
|---|---|---|---|---|
| NLLB-1.3B (CC-BY-NC) | 0.161 | 0.806 | 2.67–2.84 | 2.6 GB |
| M2M-100 1.2B (MIT) | 0.159 | 0.774 | 2.29–2.30 | 2.4 GB |
| M2M-100 418M (MIT) | 0.085 | 0.386 | 1.15–1.18 | 0.95 GB |

- The 3060 is ~5× the 890M; the 890M is ~3–3.5× the CPU; the two CPUs match.
- **Per-language instances buy nothing on a GPU** (0.98–1.02× on both, VRAM
  ×3) and ~1.7× on CPU. The shared lock is not the limit; the GPU is.
  Languages stay roughly additive on the GPU.
- **Quality, automated:** every model keeps 100 % of numbers, never returns
  empty or copied output. Loops: M2M-100 418M 1.2 % (all Creole), M2M-100
  1.2B 0.3 % (one Creole "tanpri, tanpri, …"), NLLB 0.3 % (a false positive).
- **Quality, read (not a native speaker's judgement):** Spanish and Russian
  are serviceable from all three; NLLB is the only one that renders the
  Bible "passage" correctly (M2M: *paso* / *процесс*) and addresses God with
  *Ты* in Russian prayer. **Creole separates them:** M2M-100 418M is
  unusable (runaway "Te te te…", "vle vle vle"); M2M-100 1.2B makes
  meaning-changing errors a congregation would notice — *spirit of God* →
  *san Bondye* (blood), *overwhelmed* → *ankouraje* (encouraged),
  *spirit* → *espwa* (hope). NLLB's Creole is clearly the best of the three.
  M2M keeps "Brother Oar" as a name; NLLB translates it (Remador / Rem /
  Весло) — `TODO.md` item 2.
- **So:** M2M-100 1.2B matches NLLB on speed and memory, but not on Haitian
  Creole, the language that justifies the whole system. It is not a drop-in
  replacement. MADLAD-400 3B (Apache-2.0, `--with-madlad`) is the remaining
  permissive candidate and the next run worth doing; a Creole speaker should
  read the side-by-side before any switch.

Not run yet: the stack benchmark. The laptop has no NeMo venv, so it would
transcribe with onnx-asr TDT while the PC uses unified-remote — the trap
described above. Either build the NeMo venv on the laptop first, or run the
stack benchmark on the PC alone.

---

## Run log — 2026-09-22/23, round two: the permissive candidates

Question asked: which permissively licensed model can replace NLLB-200, given
that memory is no longer a hard constraint. Candidates from the user's list.
Combined report: `benchmarks/REPORT-combined-2026-09-23.txt`.

### What could not be measured, and why

| Candidate | Outcome |
|---|---|
| Mistral Large 3 | 675B mixture-of-experts. Not runnable here at any quantization |
| DeepSeek V3/V4 | 671B, same |
| Qwen3.6-27B | ~15 GB at 4-bit: over the 3060's 12 GB, at the laptop's ceiling |
| Aya-101 13B | Would not load on either machine |
| MADLAD-400 7B | Would not load on either machine |
| Mistral NeMo 12B | Ran on the 3060; would not load on the laptop |

**Why "13B at 4-bit" is not 7 GB.** bitsandbytes quantizes linear layers only.
A 250k-token vocabulary keeps a ~2 GB embedding, and an untied output head
another ~2 GB, in fp16. Aya-101 needed >11.6 GB and MADLAD-7B tried to
allocate 1.95 GB for its embedding with 1 GB free. Parameter count alone does
not predict fit; vocabulary size matters as much.

**The laptop's ceiling is half its RAM.** The Radeon 890M has 0.5 GB of
dedicated video memory in BIOS and borrows the rest through GTT, which the
kernel caps at half of system RAM: 15.2 GB of 30.5 GB today. Past that the
loader crawls rather than failing — Mistral NeMo spent six hours loading and
never finished. **64 GB of RAM would roughly double that ceiling** and put
Aya-101 and MADLAD-7B in reach.

### Two harness bugs found, both of which would have produced false verdicts

1. **MADLAD-400 was mis-wired by transformers 5** and emitted one repeated
   character for every input. Its checkpoint contains only
   `decoder.embed_tokens` and `lm_head`; with no `shared` tensor, transformers
   filled the encoder's embedding from `lm_head`, so the encoder embedded
   words with the output head. `translate_bench.py` now detects exactly this
   (encoder embedding equal to the head, different from the decoder's) and
   rebuilds it from the checkpoint's own table. Without that, MADLAD would
   have been written off.
2. **T5-family models overflow in fp16.** MADLAD in fp16 scored a 90% loop
   rate. They now get bfloat16 maths. (This was not the cause of bug 1, but
   it is a real trap.)

### Results

Mean seconds per translation, one sentence into one language:

| model | licence | RTX 3060 | Radeon 890M | CPU | GPU memory |
|---|---|---|---|---|---|
| **Opus-MT** (3 models) | CC-BY-4.0 | **0.029** | 0.064 | 0.215 | **441 MB** |
| M2M-100 418M | MIT | 0.085 | 0.386 | 1.179 | 952 MB |
| NLLB-1.3B | CC-BY-NC | 0.161 | 0.806 | 2.667 | 2.6 GB |
| M2M-100 1.2B | MIT | 0.159 | 0.774 | 2.293 | 2.4 GB |
| MADLAD-400 3B | Apache-2.0 | 0.347 | 1.530 | not run | 5.7 GB |
| Mistral NeMo 12B | Apache-2.0 | 0.993 | would not load | not run | 10.5 GB |
| Qwen3-8B | Apache-2.0 | 1.036 (4-bit) | 4.177 | not run | 8.2 GB |

Automated signals: Opus-MT is the only model with **no loops at all** and it
keeps every number; MADLAD 0.6% loops; Qwen 3.3% at 4-bit and 1.7% at 8-bit;
Mistral 1.1%.

**Per-language instances pay for Opus-MT**, unlike every other model: 2.05x on
CPU and 1.27x on GPU, for 22 MB more memory. Three separate small models have
no shared lock and barely touch the GPU. For the others, three instances on a
GPU gain nothing and triple the memory.

### Quality, read rather than scored

- **Creole ranking: NLLB > Opus-MT > MADLAD-3B > M2M-1.2B > Qwen3-8B ≈
  Mistral NeMo.** NLLB remains the most reliable. Opus-MT is close and
  occasionally paraphrases. MADLAD-3B is fluent but **hallucinated on two of
  ten short sentences** — "And tribulations of this life" became "11:11 Do not
  do anything evil" — which is disqualifying for live use without a
  short-input guard, exactly the failure the short-phrase dictionary exists to
  prevent. Qwen produced "Fèt Holy Spirit" and Mistral answered in French.
- **Spanish: every model is usable**, including Opus-MT.
- **Russian: all serviceable.** Opus-MT and Qwen keep "Brother Oar" as a name;
  NLLB and Mistral translate it to the word for an oar, the existing
  `TODO.md` item 2.
- **LLMs verbalize numbers** ("chapter six" rather than "chapter 6"), which
  the digit-retention signal counts as a loss but which is *better* for the
  MMS voices that cannot say digits (`TODO.md` item 8).

### Recommendation

**Opus-MT is the strongest permissive candidate**, and the result reshapes the
appliance: at 0.215 s per translation on CPU and 441 MB on GPU, translation
stops needing a GPU at all. It is CC-BY-4.0, so attribution is the only
obligation. Before adopting it, two guards from the existing to-do list become
prerequisites rather than nice-to-haves: the protected-names list, because
Opus truncated "Oar" to "O." in Creole, and the short-phrase dictionary.

MADLAD-400 3B is the fallback if a Creole speaker prefers its phrasing, but
its short-input hallucinations must be fenced first.

Neither general LLM is competitive for Creole at these sizes, contradicting
the common advice that a modern LLM beats a dedicated translator. That may
change at 27B and above, which needs more memory than either machine has.

**Still required: a native Creole speaker's judgement on NLLB vs Opus-MT.**
Everything above is speed, memory and failure-mode analysis. The fluency call
is not mine to make.
