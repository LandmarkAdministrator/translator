# Plan: one repository for the live translator and the sermon archive

**Written 2026-09-22. Nothing has been moved.** This is the plan to review
before anything changes.

## Why

The two projects already share a stack and a machine. Counting files that
mention each concern:

| Concern | archive | translator |
|---|---|---|
| NLLB translation | 14 | 24 |
| Kokoro / TTS | 7 | 11 |
| Whisper / ASR | 16 | 11 |
| Synology paths | 33 | 1 |
| `~/sermons/logs` | 15 | 4 |
| translation glossary | 10 | 0 — and `TODO.md` item 2 wants it |

They also share one developer machine, one GPU, one NAS, one log directory,
and a written contract for taking turns on that GPU
(`docs/BACKLOG-CONTRACT.md`). Fixes made on one side are copied by hand to
the other today; the glossary is the clearest example.

## The blocker: one repo is public, the other is private

| Repo | Visibility | Branch | Commits |
|---|---|---|---|
| `LandmarkAdministrator/translator` | **public**, MIT | `master` | 144 |
| `LandmarkAdministrator/Multi-Bitrate-Sermons` | **private** | `main` | 181 |

Merging the archive into the translator as it stands would publish the
archive's whole history: internal email addresses, search-key tooling,
infrastructure notes. **This decision comes before any command.**

Three ways out:

1. **Private monorepo (recommended).** Flip `translator` to private, merge
   the archive in. Nothing leaks: the translator's history is already public,
   and making a repo private does not publish anything. If a public artifact
   is still wanted for other churches, publish the live stack as a filtered
   export to a second, public repo, pushed by a script rather than developed
   there.
2. **New private repo**, both projects added as subtrees. Cleanest mental
   model, but the public URL and its stars/links are left behind.
3. **Stay separate**, and share only extracted code through a small package
   both repos depend on. Least disruption, but it keeps the two-checkout
   dance that motivated this.

## Decisions needed

- **D1 Visibility.** Option 1, 2 or 3 above.
- **D2 Default branch.** `translator` deploys to the PC with
  `git pull --ff-only` on `master`. Keeping `master` avoids touching the live
  deployment; renaming to `main` is a separate, later chore.
- **D3 Repo name.** Keep `translator`, or rename to something that covers
  both (`lbc-media`). Renaming on GitHub leaves a redirect, so this is cheap,
  but every remote on both machines needs updating.

## Target layout

Phase 1 and 2 move **nothing** in the live stack. Its paths, imports, units
and installer stay exactly as they are; only new directories appear.

```
translator/
  src/ scripts/ systemd/ tests/ config/ docs/   the live translator, untouched
  archive/                                      the sermon archive, as-is
      scripts/  plugin/  chunks/  docs/  *.md
  shared/                                       phase 3 only: code both use
```

The archive's scripts compute their project root from their own location, so
they keep working under `archive/` without edits.

## Phases

### Phase 0 — reconcile the PC's archive checkout (do this first)

`~/Multi-Bitrate-Sermons` on the Translate PC is **12 commits behind
`origin/main` and has 12+ modified tracked files** (`DECISIONS.md`,
`GLOSSARY.md`, `README.md`, `chunks/*`, `.env.example`, `.gitignore`).
Merging a repo whose deployed copy has diverged would bake that divergence
in. Decide per file: commit, or discard. Until this is clean, stop.

### Phase 1 — merge the histories (laptop only, no deployment)

```bash
cd ~/Projects/translator
git tag pre-merge-2026-09-22
git remote add archive ~/Projects/Multi-Bitrate-Sermons
git fetch archive
git subtree add --prefix=archive archive main
```

`git subtree` keeps every archive commit and needs no extra tooling. Both
histories stay intact and `git log -- archive/...` reads normally. If the
paths should also read correctly *inside* the old commits, `git-filter-repo`
can prefix them before the merge; that is tidier and slower, and it rewrites
hashes.

Then: `.nas-sync-ignore` gains `archive/.venv/` and `archive/.venv-tts/`, and
`.gitignore` absorbs the archive's ignores under the new prefix.

### Phase 2 — switch the Translate PC (quiet window: a Monday or Tuesday)

Not Wednesday or Sunday. The live service must be outside a window, and the
archive worker drained the usual way (manual flag, `stop.flag`, wait).

```bash
# on the PC, worker drained
mv ~/Multi-Bitrate-Sermons/.venv      ~/translator/archive/.venv
mv ~/Multi-Bitrate-Sermons/.venv-tts  ~/translator/archive/.venv-tts
mv ~/Multi-Bitrate-Sermons ~/Multi-Bitrate-Sermons.pre-merge
ln -s ~/translator/archive ~/Multi-Bitrate-Sermons     # every absolute path keeps working
cd ~/translator && git pull --ff-only
```

The symlink is the safety net: nothing has to be updated on the same day.
Then update the real references, one at a time, verifying each:

| Reference | Where |
|---|---|
| `SERMONS_DIR`, `WORKER_PAT`, `stop.flag` | `scripts/ops/translate-window-check.sh` and the copy in `~/bin` |
| worker launch line (`systemd-run … unified_worker.py`) | same script |
| archive paths in the admin panel's start/stop | `src/web/admin.py` |
| hourly index job, 4-hourly alert check | the PC's crontab |
| `SERMONS_DIR` fallbacks | `scripts/run_bench_safely.sh` |
| the contract itself | `docs/BACKLOG-CONTRACT.md` |

Remove the symlink only once nothing references the old path:
`grep -rl "Multi-Bitrate-Sermons" ~/bin ~/translator` returns nothing.

### Phase 3 — extract what is genuinely shared (ongoing, one piece at a time)

Order by payoff, each its own commit with tests:

1. **Translation glossary and short-phrase dictionary.** The archive has a
   proven glossary; the live stack needs one (`TODO.md` item 2).
2. **Model wrappers** — NLLB / M2M loading, Kokoro and MMS voices, including
   the number-verbalizer (`TODO.md` item 8), which both sides need.
3. **NAS and logging conventions** — mount checks, `~/sermons/logs`
   layout, the "NAS is slow, time-box any find" rule.
4. **Licence facts** — one table both projects read, since both ship models.

Not shared, deliberately: the two Python environments (their dependency sets
conflict, which is why the archive already carries `.venv` and `.venv-tts`),
the WordPress plugin, and the live stack's real-time code.

## Risks

- **Publishing private history.** Addressed by D1. Verify with
  `git log --stat` on the merged branch before the first push.
- **A diverged deployed checkout.** Phase 0.
- **The archive worker stops silently** because a path moved. The symlink
  prevents this on day one; the table above is the checklist afterwards.
- **The live service.** Untouched by phases 1 and 2 — its code does not move.
  Still, do phase 2 outside service windows, and run
  `tests/test_pipeline_config.py` plus a short pipeline dry run afterwards.
- **The NAS mirror** doubles in size if the archive venvs are not excluded.

## Rollback

Both original repos stay on GitHub, untouched, until phase 3 is finished and
proven. `pre-merge-2026-09-22` tags the translator. On the PC, point the
symlink back at `~/Multi-Bitrate-Sermons.pre-merge` and the old arrangement
returns.

## Effort

Phase 0: unknown until the 12 files are reviewed, probably an hour.
Phase 1: about an hour. Phase 2: an hour in a quiet window, plus a watchful
day. Phase 3: roughly half a day per extracted component, and it can stop
after the glossary if the value runs out.
