"""Build a leakage-controlled index of the two public otoscopy datasets.

- Maps source labels to 5 harmonised classes.
- Groups near-duplicate frames (perceptual hash, Hamming <= THR) so that
  splits are done at group level, never at image level.
- Caches all images resized to RES x RES as a uint8 tensor.
"""
import sys, os, json, glob
import numpy as np
from PIL import Image
import imagehash
import torch

ROOT = sys.argv[1]          # data dir
OUT = sys.argv[2]           # output dir
RES = 224
THR = 10

CLASSES = ["Normal", "Earwax", "AOM", "COM", "Myringosclerosis"]
MAP = {
    ("chile", "Normal"): "Normal",
    ("chile", "Earwax plug"): "Earwax",
    ("chile", "Chronic otitis media"): "COM",
    ("chile", "Myringosclerosis"): "Myringosclerosis",
    ("turkey", "Normal"): "Normal",
    ("turkey", "Earwax"): "Earwax",
    ("turkey", "AOM"): "AOM",
    ("turkey", "CSOM"): "COM",
    ("turkey", "Miringoskleroz"): "Myringosclerosis",
}

recs = []
for p in sorted(glob.glob(os.path.join(ROOT, "**", "*.*"), recursive=True)):
    if not p.lower().endswith((".jpg", ".png", ".jpeg")):
        continue
    site = "chile" if "/chile/" in p else "turkey"
    folder = os.path.basename(os.path.dirname(p))
    key = (site, folder)
    if key not in MAP:
        continue  # otitis externa, tympanostomy tubes: excluded
    recs.append(dict(path=p, site=site, label=CLASSES.index(MAP[key]), src_label=folder))

print("images", len(recs))
imgs = np.zeros((len(recs), RES, RES, 3), dtype=np.uint8)
hashes = []
for i, r in enumerate(recs):
    im = Image.open(r["path"]).convert("RGB")
    hashes.append(imagehash.phash(im, hash_size=16))
    imgs[i] = np.asarray(im.resize((RES, RES), Image.BICUBIC))

# union-find near duplicates within the same site
par = list(range(len(recs)))
def find(a):
    while par[a] != a:
        par[a] = par[par[a]]
        a = par[a]
    return a
ndup_pairs = 0
cross_label = 0
for i in range(len(recs)):
    for j in range(i + 1, len(recs)):
        if recs[i]["site"] != recs[j]["site"]:
            continue
        if hashes[i] - hashes[j] <= THR:
            ndup_pairs += 1
            if recs[i]["label"] != recs[j]["label"]:
                cross_label += 1
            par[find(i)] = find(j)
groups = [find(i) for i in range(len(recs))]
for r, g in zip(recs, groups):
    r["group"] = int(g)
ng = len(set(groups))
print("near-duplicate pairs", ndup_pairs, "cross-label pairs", cross_label, "groups", ng)

os.makedirs(OUT, exist_ok=True)
torch.save(torch.from_numpy(imgs).permute(0, 3, 1, 2).contiguous(), os.path.join(OUT, "images.pt"))
json.dump(dict(classes=CLASSES, records=recs, dup_pairs=ndup_pairs,
               cross_label_pairs=cross_label, n_groups=ng, thr=THR),
          open(os.path.join(OUT, "index.json"), "w"))
from collections import Counter
c = Counter((r["site"], CLASSES[r["label"]]) for r in recs)
for k in sorted(c):
    print(k, c[k])
