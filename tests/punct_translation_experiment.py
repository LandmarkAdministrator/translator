#!/usr/bin/env python3
"""Does the punctuation/fragmentation problem hurt translation? (experiment 2)

Experiment 1 was invalid: the sample contained no leading marks, and its
"complete thoughts" variant collapsed to a single run-on because these
sentences so rarely end in punctuation. This one uses a run that actually
exhibits the problem, and reconstructs thoughts by splitting at the marks the
ASR reported — which are the real sentence boundaries.
"""
import json, re, sys
from difflib import SequenceMatcher
# Input: a JSON file with "asis" (segments as they were sent to translation)
# and "thoughts" (the same text regrouped at the ASR's sentence marks).
#     python3 tests/punct_translation_experiment.py exp2.json
if len(sys.argv) < 2:
    sys.exit(__doc__ + "\nusage: punct_translation_experiment.py EXPERIMENT_JSON")
LEAD=re.compile(r"^\s*[.?!,;:]+\s*")
d=json.load(open(sys.argv[1])); asis=d["asis"]; thoughts=d["thoughts"]

def norm(s): return [w for w in re.sub(r"[^\w\s]"," ",s.lower()).split() if w]
def sim(a,b):
    ta,tb=norm(a),norm(b)
    if not ta or not tb: return 0.0
    pool=list(tb); c=0
    for w in ta:
        if w in pool: pool.remove(w); c+=1
    p,r=c/len(ta),c/len(tb); f1=2*p*r/(p+r) if p+r else 0
    return (f1+SequenceMatcher(None," ".join(ta)," ".join(tb)).ratio())/2

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
M="facebook/nllb-200-distilled-1.3B"
tok=AutoTokenizer.from_pretrained(M)
mdl=AutoModelForSeq2SeqLM.from_pretrained(M, dtype=torch.float16).to("cuda").eval()
def tr(texts,src,tgt):
    out=[]; tok.src_lang=src
    for t in texts:
        e=tok(t,return_tensors="pt").to("cuda")
        with torch.inference_mode():
            g=mdl.generate(**e,forced_bos_token_id=tok.convert_tokens_to_ids(tgt),max_new_tokens=200)
        out.append(tok.batch_decode(g,skip_special_tokens=True)[0])
    return out

variants={
 "A": ("as sent today (stray marks + pause-cut)", asis),
 "B": ("marks stripped, same segmentation",       [LEAD.sub("",s).strip() or s for s in asis]),
 "C": ("complete thoughts, punctuated",           thoughts),
 "D": ("complete thoughts, punctuation stripped", [re.sub(r"[.?!,;:]+","",t).strip() for t in thoughts]),
}
ref=" ".join(thoughts)
res={}
for k,(label,v) in variants.items():
    es=tr(v,"eng_Latn","spa_Latn"); back=tr(es,"spa_Latn","eng_Latn")
    res[k]=(label,v,es,back,sim(" ".join(back),ref))
    print(f"  {k}. {label:42} segs={len(v):3d}  round-trip={res[k][4]:.3f}")
print()
for k in "ABC":
    label,v,es,back,_=res[k]
    print(f"  --- {k}: {label}")
    for s,t in list(zip(v,es))[:3]:
        print(f"     EN {s[:86]}")
        print(f"     ES {t[:86]}")
    print()
a,b,c,dd=(res[k][4] for k in "ABCD")
print("  === answers ===")
print(f"    1. stray marks cost:       {a:.3f} -> {b:.3f}   ({(b-a)*100:+.1f} pts)")
print(f"    2. complete thoughts gain: {b:.3f} -> {c:.3f}   ({(c-b)*100:+.1f} pts)")
print(f"    3. punctuation itself:     {c:.3f} -> {dd:.3f}   ({(dd-c)*100:+.1f} pts)")

