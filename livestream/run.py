"""Entry point: run the ingest and the publisher together.

    ./venv/bin/python -m livestream.run                      # live, from config
    ./venv/bin/python -m livestream.run --input-file x.mp4   # the same pipeline
                                                             #   over a recording
    ./venv/bin/python -m livestream.run --encoder x264 --check

`--input-file` mirrors `run_production.sh --input-file` in the translator:
the identical pipeline driven by a file instead of the live feed, which is
how this is developed and how a regression is reproduced without waiting
for a Sunday.
"""
from __future__ import annotations

import argparse
import logging
import shutil
import signal
import sys
import time
from pathlib import Path

from . import config as config_mod
from .ingest import Ingest, build_command
from .publisher import Publisher

log = logging.getLogger("livestream")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="livestream.run", description=__doc__)
    p.add_argument("--config", type=Path, default=None,
                   help="path to livestream.yaml (default: config/livestream.yaml)")
    p.add_argument("--input-file", default=None,
                   help="run over a recording instead of the live RTMP feed")
    p.add_argument("--encoder", choices=("vaapi", "nvenc", "x264"), default=None)
    p.add_argument("--delay", type=float, default=None,
                   help="seconds behind real time (default: from config)")
    p.add_argument("--staging-dir", type=Path, default=None)
    p.add_argument("--public-dir", type=Path, default=None)
    p.add_argument("--duration", type=float, default=None,
                   help="stop after N seconds; for tests")
    p.add_argument("--clean", action="store_true",
                   help="wipe staging and public before starting")
    p.add_argument("--check", action="store_true",
                   help="validate config, print the ffmpeg command, exit")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def _apply_overrides(cfg: config_mod.Config, args: argparse.Namespace) -> None:
    if args.input_file:
        cfg.source = "file"
        cfg.input_file = args.input_file
    if args.encoder:
        cfg.encoder = args.encoder
    if args.delay is not None:
        cfg.delay_seconds = args.delay
    if args.staging_dir:
        cfg.staging_dir = args.staging_dir
    if args.public_dir:
        cfg.public_dir = args.public_dir
    cfg.validate()


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    try:
        cfg = config_mod.load(args.config)
        _apply_overrides(cfg, args)
    except ValueError as e:
        log.error("configuration: %s", e)
        return 2

    if args.clean:
        for d in (cfg.staging_dir, cfg.public_dir):
            shutil.rmtree(d, ignore_errors=True)

    cfg.staging_dir.mkdir(parents=True, exist_ok=True)
    cfg.public_dir.mkdir(parents=True, exist_ok=True)

    if args.check:
        print(f"source       : {cfg.source} "
              f"({cfg.input_file or cfg.rtmp_url})")
        print(f"encoder      : {cfg.encoder}")
        print(f"renditions   : {', '.join(r.name for r in cfg.renditions)}")
        print(f"audio tracks : "
              f"{', '.join(t.language for t in cfg.active_audio())}")
        print(f"segment      : {cfg.segment_seconds:g}s")
        print(f"delay        : {cfg.delay_seconds:g}s")
        print(f"window       : {cfg.window_segments} segments "
              f"({cfg.window_segments * cfg.segment_seconds:g}s)")
        print(f"staging      : {cfg.staging_dir}")
        print(f"public       : {cfg.public_dir}")
        print()
        print(" ".join(build_command(cfg)))
        return 0

    ingest = Ingest(cfg).start()
    publisher = Publisher(cfg).start()
    log.info("running — publishing %.0fs behind real time", cfg.delay_seconds)

    stopping = {"now": False}

    def _handle(signum, _frame):
        log.info("signal %s — shutting down", signum)
        stopping["now"] = True

    signal.signal(signal.SIGINT, _handle)
    signal.signal(signal.SIGTERM, _handle)

    started = time.monotonic()
    try:
        while not stopping["now"]:
            if args.duration and time.monotonic() - started >= args.duration:
                log.info("--duration reached")
                break
            time.sleep(0.25)
    finally:
        ingest.stop()
        # One last pass so anything ffmpeg finalized on the way out is
        # published before the playlists are closed.
        try:
            publisher.tick()
        except Exception:  # noqa: BLE001
            log.exception("final publish pass failed")
        publisher.stop()
        publisher.finish()
    return 0


if __name__ == "__main__":
    sys.exit(main())
