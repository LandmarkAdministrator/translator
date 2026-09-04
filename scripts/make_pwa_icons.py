#!/usr/bin/env python3
"""Generate the PWA icon set with no image-library dependency.

Writes minimal PNGs by hand (zlib + struct), so the build has no Pillow
requirement. The mark is a speech bubble on the app's green ground — legible
at 48px on a home screen, which is the only size that really matters.

Usage: make_pwa_icons.py [OUTDIR]   (default src/web/static/icons)
"""
from __future__ import annotations

import struct
import sys
import zlib
from pathlib import Path

BG = (23, 94, 84)        # --accent, the app's green
FG = (247, 245, 240)     # warm off-white


def _chunk(tag: bytes, data: bytes) -> bytes:
    return (struct.pack(">I", len(data)) + tag + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))


def write_png(path: Path, size: int, maskable: bool) -> None:
    """Speech bubble centred on a solid ground.

    Maskable icons must keep their content inside the safe zone (the middle
    ~80%), because Android crops them to whatever shape the launcher uses.
    """
    pad = size * 0.22 if maskable else size * 0.14
    bx0, bx1 = pad, size - pad
    by0, by1 = pad, size - pad * 1.18
    radius = (bx1 - bx0) * 0.22
    tail_x, tail_w, tail_h = bx0 + (bx1 - bx0) * 0.28, (bx1 - bx0) * 0.18, size * 0.10

    rows = bytearray()
    for y in range(size):
        rows.append(0)  # PNG filter: none
        for x in range(size):
            px, py = x + 0.5, y + 0.5
            inside = bx0 <= px <= bx1 and by0 <= py <= by1
            if inside:  # round the corners
                for cx, cy in ((bx0 + radius, by0 + radius), (bx1 - radius, by0 + radius),
                               (bx0 + radius, by1 - radius), (bx1 - radius, by1 - radius)):
                    if ((px < bx0 + radius or px > bx1 - radius) and
                            (py < by0 + radius or py > by1 - radius)):
                        if (px - cx) ** 2 + (py - cy) ** 2 > radius ** 2:
                            near = (abs(px - cx) < radius * 1.2 and abs(py - cy) < radius * 1.2)
                            if near:
                                inside = False
            if not inside and by1 <= py <= by1 + tail_h:
                # triangular tail, narrowing toward the point
                t = (py - by1) / tail_h
                if tail_x <= px <= tail_x + tail_w * (1 - t):
                    inside = True
            rows.extend(FG if inside else BG)

    png = (b"\x89PNG\r\n\x1a\n"
           + _chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
           + _chunk(b"IDAT", zlib.compress(bytes(rows), 9))
           + _chunk(b"IEND", b""))
    path.write_bytes(png)


def main() -> None:
    out = Path(sys.argv[1] if len(sys.argv) > 1
               else Path(__file__).parent.parent / "src/web/static/icons")
    out.mkdir(parents=True, exist_ok=True)
    for size in (192, 512):
        write_png(out / f"icon-{size}.png", size, maskable=False)
        write_png(out / f"icon-{size}-maskable.png", size, maskable=True)
    write_png(out / "apple-touch-icon.png", 180, maskable=False)
    for f in sorted(out.iterdir()):
        print(f"  {f.name}: {f.stat().st_size} bytes")


if __name__ == "__main__":
    main()
