#!/usr/bin/env python3
"""End-to-end check: segmentation -> WER -> translation quality.

Replays a captured fragment stream through both the old and new flush policies,
scores each against the reference transcript, then translates the new
segmentation and scores the translations against their own source with LaBSE.

WER is expected to be near-identical between policies: segmentation regroups
words, it does not change them. Any real difference is a subword join that a
flush boundary used to split ("pass age" -> "passage"). The interesting numbers
are the translation ones.

    python3 tests/full_pipeline_eval.py frags.json reference.txt
"""
from __future__ import annotations

import json
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from pipeline.sentence_buffer import SentenceBuffer  # noqa: E402

END = re.compile(r"[.?!][\"')\]]?\s*$")
LEAD = re.compile(r"^\s*[.?!,;:]")

OLD = dict(silence_timeout=1.0, hard_timeout=6.0, min_emit_words=3,
           max_buffer_chars=800, max_emit_words=0, silence_min_words=3,
           strip_lead_punct=False, punct_boundary=False)
NEW = dict(silence_timeout=30.0, hard_timeout=15.0, min_emit_words=3,
           max_buffer_chars=800, max_emit_words=60, silence_min_words=1,
           strip_lead_punct=True, punct_boundary=True)


def replay(chunks, **kw):
    b = SentenceBuffer(**kw)
    out = []
    for c in chunks:
        r = (b.feed(c["text"], start_wall=c["audio_t"], asr_time=0.0, now=c["t"])
             if c["text"] else b.tick(now=c["t"]))
        if r:
            out.append(r[0])
    r = b.flush()
    if r:
        out.append(r[0])
    return out


def norm_words(text: str) -> list[str]:
    """Lowercase, strip punctuation, spell integers — matches the WER scorer's
    normalisation so "120" and "one hundred and twenty" are not four errors."""
    text = re.sub(r"[^\w\s']", " ", text.lower())
    return [w for w in text.split() if w]


def wer(ref: list[str], hyp: list[str]) -> tuple[float, int, int, int]:
    d = [[0] * (len(hyp) + 1) for _ in range(len(ref) + 1)]
    for i in range(len(ref) + 1):
        d[i][0] = i
    for j in range(len(hyp) + 1):
        d[0][j] = j
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1,
                          d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]))
    # walk back for S/D/I
    i, j, s = len(ref), len(hyp), 0
    dele = ins = 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            s += ref[i - 1] != hyp[j - 1]
            i, j = i - 1, j - 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            dele += 1
            i -= 1
        else:
            ins += 1
            j -= 1
    return 100.0 * d[len(ref)][len(hyp)] / max(1, len(ref)), s, dele, ins


def read_reference(path: str) -> list[str]:
    """Reference files carry comment headers; counting them inflates deletions."""
    lines = [l for l in Path(path).read_text().splitlines()
             if l.strip() and not l.lstrip().startswith("#")]
    return norm_words(" ".join(lines))


def shape(name, sents):
    w = sorted(len(s.split()) for s in sents)
    print(f"  {name}")
    print(f"    {len(sents):4d} sentences | median {statistics.median(w):3.0f}w  "
          f"p90 {w[int(len(w)*0.9)]:3d}w  max {max(w):3d}w")
    print(f"    ends .?! {100*sum(bool(END.search(s)) for s in sents)/len(sents):5.1f}%   "
          f"leads with a mark {100*sum(bool(LEAD.match(s)) for s in sents)/len(sents):5.1f}%")


def main() -> int:
    chunks = json.load(open(sys.argv[1]))["chunks"]
    ref = read_reference(sys.argv[2])
    old, new = replay(chunks, **OLD), replay(chunks, **NEW)

    print(f"  reference: {len(ref)} words\n")
    shape("OLD segmentation (production before today)", old)
    print()
    shape("NEW segmentation (live now)", new)

    print("\n  === WER against the reference ===")
    for name, sents in (("old", old), ("new", new)):
        h = norm_words(" ".join(sents))
        r, s, d, i = wer(ref, h)
        print(f"    {name}: {r:5.2f}%   ({len(h)} words: sub {s}, del {d}, ins {i})")

    # ---- translation quality of the new segmentation ----
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM

    ltok = AutoTokenizer.from_pretrained("setu4993/LaBSE")
    lab = AutoModel.from_pretrained("setu4993/LaBSE").to("cuda").eval()

    def emb(texts, bs=24):
        outs = []
        for k in range(0, len(texts), bs):
            e = ltok(texts[k:k + bs], return_tensors="pt", padding=True,
                     truncation=True, max_length=256).to("cuda")
            with torch.inference_mode():
                outs.append(F.normalize(lab(**e).pooler_output, p=2, dim=1))
        return torch.cat(outs)

    ntok = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-1.3B")
    nllb = AutoModelForSeq2SeqLM.from_pretrained(
        "facebook/nllb-200-distilled-1.3B", dtype=torch.float16).to("cuda").eval()

    def translate(texts, tgt):
        out = []
        for t in texts:
            e = ntok(t, return_tensors="pt").to("cuda")
            with torch.inference_mode():
                g = nllb.generate(**e, forced_bos_token_id=ntok.convert_tokens_to_ids(tgt),
                                  max_new_tokens=220)
            out.append(ntok.batch_decode(g, skip_special_tokens=True)[0])
        return out

    print("\n  === translation of the NEW segmentation ===")
    results = {}
    for lang, code in (("Spanish", "spa_Latn"), ("Creole", "hat_Latn"), ("Russian", "rus_Cyrl")):
        tgt = translate(new, code)
        sims = [float(x) for x in (emb(new) * emb(tgt)).sum(dim=1)]
        s = sorted(sims)
        drops = [k for k in range(len(new))
                 if len(tgt[k].split()) < 0.55 * len(new[k].split()) and len(new[k].split()) > 8]
        print(f"    {lang:8} median {statistics.median(s):.3f}  p10 {s[int(len(s)*0.1)]:.3f}  "
              f"good(>=.80) {100*sum(1 for x in s if x >= .80)/len(s):4.1f}%  "
              f"poor(<.65) {100*sum(1 for x in s if x < .65):>4.1f}%  "
              f"content drops {len(drops)} ({100*len(drops)/len(new):.1f}%)")
        results[lang] = (tgt, sims)

    print("\n  === worst Spanish, for reading ===")
    tgt, sims = results["Spanish"]
    for k in sorted(range(len(new)), key=lambda k: sims[k])[:5]:
        print(f"    [{sims[k]:.3f}] EN {new[k][:84]}")
        print(f"             ES {tgt[k][:84]}")
    json.dump({"en": new, **{k: v[0] for k, v in results.items()}},
              open("/tmp/full_eval_out.json", "w"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
