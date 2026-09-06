#!/usr/bin/env python3
"""Does the live Spanish and Creole actually mean what the preacher said?

Round-trip similarity was too blunt to answer that. This scores the English
source directly against its translation with LaBSE, a multilingual encoder that
puts a sentence and its faithful translation close together in one space, so no
back-translation is needed. Calibration on known pairs: a good translation
scores ~0.86-0.91, unrelated text ~0.08, and a statement mistranslated as a
question ~0.75.

Two questions:
  1. How good is what congregants actually received today?
  2. Would sending complete thoughts instead of pause-cut fragments improve it?

    python3 tests/eval_translation_quality.py path/to/sermon.log
"""
from __future__ import annotations

import re
import statistics
import sys

LEAD = re.compile(r"^\s*([.?!,;:]+)")
GOOD, POOR = 0.80, 0.65        # from the calibration above


def load_triples(path: str):
    """(english, spanish, creole) as production actually emitted them."""
    triples, cur = [], None
    for line in open(path, errors="replace"):
        m = re.search(r"\[(EN|ES|HT)\] (.*?)(?: \| (?:mode|e2e)=|$)", line)
        if not m:
            continue
        tag, txt = m.group(1), m.group(2).strip()
        if tag == "EN":
            if "streaming/sentence" not in line:
                continue
            cur = {"en": txt, "es": None, "ht": None}
            triples.append(cur)
        elif cur is not None and cur[tag.lower()] is None:
            cur[tag.lower()] = txt
    return [t for t in triples if t["es"] and t["ht"]]


def rejoin(sents):
    """Reconstruct complete thoughts: the marks opening fragments are the
    sentence boundaries the ASR reported, so glue on them and split there."""
    glued = ""
    for s in sents:
        m = LEAD.match(s)
        if m and glued:
            glued = glued.rstrip() + m.group(1)[0] + " " + s[m.end():].strip()
        else:
            glued = (glued + " " + s).strip() if glued else s
    return [x.strip() for x in re.split(r"(?<=[.?!])\s+", glued) if x.strip()]


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else "sermon.log"
    triples = load_triples(path)
    print(f"  {len(triples)} EN/ES/HT triples from the live service\n")

    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM

    ltok = AutoTokenizer.from_pretrained("setu4993/LaBSE")
    lab = AutoModel.from_pretrained("setu4993/LaBSE").to("cuda").eval()

    def emb(texts, bs=32):
        outs = []
        for i in range(0, len(texts), bs):
            e = ltok(texts[i:i + bs], return_tensors="pt", padding=True,
                     truncation=True, max_length=256).to("cuda")
            with torch.inference_mode():
                outs.append(F.normalize(lab(**e).pooler_output, p=2, dim=1))
        return torch.cat(outs)

    def score(src, tgt):
        a, b = emb(src), emb(tgt)
        return [float(x) for x in (a * b).sum(dim=1)]

    def report(name, sims):
        s = sorted(sims)
        print(f"  {name}")
        print(f"    median {statistics.median(s):.3f}   mean {statistics.mean(s):.3f}   "
              f"p10 {s[int(len(s)*0.1)]:.3f}   min {min(s):.3f}")
        print(f"    good (>={GOOD}) {100*sum(1 for x in s if x >= GOOD)/len(s):5.1f}%   "
              f"poor (<{POOR}) {100*sum(1 for x in s if x < POOR)/len(s):5.1f}%")

    en = [t["en"] for t in triples]
    es_sim = score(en, [t["es"] for t in triples])
    ht_sim = score(en, [t["ht"] for t in triples])

    print("  === WHAT CONGREGANTS RECEIVED TODAY ===")
    report("English -> Spanish", es_sim)
    report("English -> Creole ", ht_sim)

    print("\n  === worst Spanish, with the English that produced it ===")
    for i in sorted(range(len(triples)), key=lambda i: es_sim[i])[:6]:
        print(f"    [{es_sim[i]:.3f}] EN {triples[i]['en'][:82]}")
        print(f"             ES {triples[i]['es'][:82]}")

    print("\n  === worst Creole ===")
    for i in sorted(range(len(triples)), key=lambda i: ht_sim[i])[:4]:
        print(f"    [{ht_sim[i]:.3f}] EN {triples[i]['en'][:82]}")
        print(f"             HT {triples[i]['ht'][:82]}")

    # Does completeness help? Re-translate reconstructed thoughts and compare.
    print("\n  === would complete thoughts do better? ===")
    thoughts = rejoin(en)
    ntok = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-1.3B")
    nllb = AutoModelForSeq2SeqLM.from_pretrained(
        "facebook/nllb-200-distilled-1.3B", dtype=torch.float16).to("cuda").eval()

    def translate(texts, tgt):
        out = []
        for t in texts:
            e = ntok(t, return_tensors="pt").to("cuda")
            with torch.inference_mode():
                g = nllb.generate(**e, forced_bos_token_id=ntok.convert_tokens_to_ids(tgt),
                                  max_new_tokens=200)
            out.append(ntok.batch_decode(g, skip_special_tokens=True)[0])
        return out

    print(f"    {len(en)} pause-cut fragments -> {len(thoughts)} complete thoughts")
    t_es = translate(thoughts, "spa_Latn")
    t_ht = translate(thoughts, "hat_Latn")
    report("thoughts -> Spanish", score(thoughts, t_es))
    report("thoughts -> Creole ", score(thoughts, t_ht))
    return 0


if __name__ == "__main__":
    sys.exit(main())
