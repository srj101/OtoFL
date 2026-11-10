"""Sensitivity of near-duplicate detection to the Hamming threshold (256-bit pHash, within site)."""
import json, sys, numpy as np, imagehash
from PIL import Image
idx = json.load(open(sys.argv[1])); recs = idx["records"]
H = np.array([imagehash.phash(Image.open(r["path"]).convert("RGB"), hash_size=16).hash.flatten() for r in recs])
site = np.array([r["site"] for r in recs]); lab = np.array([r["label"] for r in recs])
off_tr = np.array(["/Training-validation/" in r["path"] for r in recs]); off_te = np.array(["/Testing/" in r["path"] for r in recs])
D = (H[:, None, :] != H[None, :, :]).sum(-1)
iu = np.triu_indices(len(recs), 1); same = site[iu[0]] == site[iu[1]]
out = []
for t in [0, 4, 6, 8, 10, 12, 14, 16, 20, 24, 32]:
    m = same & (D[iu] <= t)
    pairs = int(m.sum()); cross = int((lab[iu[0]][m] != lab[iu[1]][m]).sum())
    # official test images with a near-duplicate (direct match) in the official training set
    leak = int(((D[np.ix_(off_te, off_tr)] <= t).any(1)).sum())
    out.append(dict(thr=t, pairs=pairs, cross_label=cross, leaked_official_test=leak))
    print(out[-1])
json.dump(out, open(sys.argv[2], "w"), indent=1)
