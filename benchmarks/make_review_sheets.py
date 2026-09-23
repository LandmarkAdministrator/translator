#!/usr/bin/env python
"""Blind A/B review sheets for a native speaker.

Both translations come from the guarded runs, so what the reviewer judges is
what the system would actually produce. Which model is A and which is B is
decided per sentence by a fixed seed, so no one can drift into recognising a
style; the key is written to a separate file the reviewer does not get.
"""
import json, random, re, subprocess, sys
from pathlib import Path
sys.path.insert(0, "tests")
from translate_bench import load_sentences

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "benchmarks" / "review"
R = Path.home() / "bench-results" / "3060"
LANGS = {"es": ("Spanish", "Español"), "ht": ("Haitian Creole", "Kreyòl Ayisyen")}
INTRO = {
    "es": ("Gracias por ayudarnos. Abajo hay frases del sermón en inglés y dos "
           "traducciones al español, A y B, hechas por dos programas distintos. "
           "Marque cuál suena mejor en español, o «igual» si las dos sirven. "
           "Piense en lo que oiría alguien en el culto: ¿se entiende bien?, "
           "¿suena natural?, ¿dice lo mismo que el inglés?"),
    "ht": ("Mèsi paske w ap ede nou. Anba a gen fraz an angle ki soti nan yon "
           "prèch, ak de tradiksyon an kreyòl, A ak B, ki soti nan de pwogram "
           "diferan. Make kiyès ki pi bon an kreyòl, oswa « menm bagay » si tou "
           "de bon. Panse ak sa yon moun ta tande nan sèvis la: èske li klè? "
           "èske li sonnen natirèl? èske li di menm bagay ak angle a?"),
}
CHOICES = {"es": ("A es mejor", "B es mejor", "Iguales"),
           "ht": ("A pi bon", "B pi bon", "Menm bagay")}
FINAL = {"es": "En general, ¿cuál traducción prefiere para nuestros cultos?",
         "ht": "An jeneral, ki tradiksyon ou pi renmen pou sèvis nou yo?"}

def build(lang: str, pairs, nllb, opus, rng):
    name, native = LANGS[lang]
    key = []
    items = []
    for n, k in enumerate(pairs, 1):
        a_is_nllb = rng.random() < 0.5
        a = (nllb if a_is_nllb else opus)[k]["out"].strip()
        b = (opus if a_is_nllb else nllb)[k]["out"].strip()
        key.append(f"{n:>3}.  A = {'NLLB' if a_is_nllb else 'Opus'}   B = {'Opus' if a_is_nllb else 'NLLB'}")
        c1, c2, c3 = CHOICES[lang]
        items.append(f'''<div class="item">
  <div class="num">{n}</div>
  <div class="en">{nllb[k]["src"]}</div>
  <div class="opt"><span class="tag">A</span>{a}</div>
  <div class="opt"><span class="tag">B</span>{b}</div>
  <div class="pick"><span class="box"></span>{c1}<span class="box"></span>{c2}<span class="box"></span>{c3}
     <span class="note">&nbsp;</span></div>
</div>''')
    html = f'''<!DOCTYPE html><html lang="{lang}"><head><meta charset="utf-8">
<title>{native} — A / B</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=PT+Serif:wght@700&family=PT+Sans:wght@400;700&display=swap">
<style>
@page {{ size: letter; margin: 0.7in 0.7in 0.6in; }}
body {{ font-family: "PT Sans", Calibri, sans-serif; color: #1F1D1A; font-size: 11pt; line-height: 1.35; }}
h1 {{ font-family: "PT Serif", Georgia, serif; color: #283A82; font-size: 20pt; margin: 0 0 2pt; }}
.sub {{ color: #5F5A4E; font-size: 10pt; margin-bottom: 10pt; }}
.intro {{ background: #DDD1B1; padding: 9pt 11pt; border-left: 4pt solid #C9A24C; margin-bottom: 14pt; }}
.item {{ border-bottom: 0.5pt solid #d8d3c6; padding: 7pt 0 6pt 26pt; position: relative;
         page-break-inside: avoid; }}
.num {{ position: absolute; left: 0; top: 7pt; color: #283A82; font-weight: 700; }}
.en {{ color: #5F5A4E; font-style: italic; margin-bottom: 3pt; }}
.opt {{ margin: 2pt 0; }}
.tag {{ display: inline-block; width: 15pt; height: 15pt; line-height: 15pt; text-align: center;
        background: #283A82; color: #fff; font-weight: 700; border-radius: 3pt; margin-right: 6pt;
        font-size: 9pt; }}
.pick {{ margin-top: 4pt; color: #5F5A4E; font-size: 9.5pt; }}
.box {{ display: inline-block; width: 10pt; height: 10pt; border: 1pt solid #5F5A4E;
        margin: 0 4pt 0 14pt; vertical-align: -1pt; }}
.pick .box:first-child {{ margin-left: 0; }}
.note {{ display: inline-block; border-bottom: 0.5pt dotted #b9b3a4; min-width: 45%; margin-left: 12pt; }}
.final {{ margin-top: 16pt; padding: 10pt 12pt; border: 1.5pt solid #283A82; page-break-inside: avoid; }}
.final b {{ color: #283A82; }}
.lines div {{ border-bottom: 0.5pt dotted #b9b3a4; height: 17pt; }}
</style></head><body>
<h1>{native}</h1>
<div class="sub">Landmark Baptist Church · live translation · which translation is better? ({name})</div>
<div class="intro">{INTRO[lang]}</div>
{''.join(items)}
<div class="final"><b>{FINAL[lang]}</b>
  <div class="pick"><span class="box"></span>A<span class="box"></span>B<span class="box"></span>{CHOICES[lang][2]}</div>
  <div class="lines"><div></div><div></div><div></div></div>
</div>
</body></html>'''
    return html, key

def main():
    sents, tracks = load_sentences(0, source="suite")
    nllb = {(r["i"], r["lang"]): r for r in json.loads((R / "nllb-1.3b-cuda-fp16-suite-guarded.json").read_text())["rows"]}
    opus = {(r["i"], r["lang"]): r for r in json.loads((R / "opus-cuda-fp16-suite-guarded.json").read_text())["rows"]}
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(20260923)
    keyfile = ["Which is A and which is B, per sentence. Do not give this to the reviewer.", ""]
    for lang in ("es", "ht"):
        church = [k for k in sorted(nllb) if k[1] == lang and tracks[k[0]] in ("sermon", "singing")
                  and nllb[k]["out"].strip() != opus[k]["out"].strip()
                  and 5 <= len(nllb[k]["src"].split()) <= 40]
        withnum = [k for k in church if re.search(r"\d", nllb[k]["src"])]
        rest = [k for k in church if k not in withnum]
        rng.shuffle(rest)
        chosen = sorted(withnum[:8] + rest[:40 - min(8, len(withnum))])
        html, key = build(lang, chosen, nllb, opus, rng)
        name = LANGS[lang][0].replace(" ", "-")
        src = OUT / f"review-{lang}.html"
        src.write_text(html, encoding="utf-8")
        pdf = OUT / f"Translation review - {name}.pdf"
        subprocess.run(["google-chrome", "--headless=new", "--disable-gpu", "--no-sandbox",
                        "--no-pdf-header-footer", f"--print-to-pdf={pdf}", f"file://{src}"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=180)
        keyfile += [f"=== {LANGS[lang][0]} ==="] + key + [""]
        print(f"{lang}: {len(chosen)} sentences -> {pdf.name}")
    (OUT / "ANSWER-KEY.txt").write_text("\n".join(keyfile), encoding="utf-8")
    print("key written to", OUT / "ANSWER-KEY.txt")

main()
