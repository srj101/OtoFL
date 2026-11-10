"""Effect of near-duplicate leakage on Site-1 (Chile) test scores: official split, image-level random split,
and the group-aware split used throughout the paper (Chile-only Fenet5, 40 epochs, 3 seeds)."""
import json, sys, numpy as np
from sklearn.metrics import accuracy_score, f1_score
S = sys.argv[1]
recs = json.load(open(f"{S}/cache/index.json"))["records"]; G = np.array([r["group"] for r in recs])
site = np.array([0 if r["site"] == "chile" else 1 for r in recs]); y = np.array([r["label"] for r in recs])
out = {}
def summarise(name, runs, train_of):
    rows = []
    for r in runs:
        te = np.array(r["test_ids"]); p = np.array(r["probs"]).argmax(1); lab = np.array(r["labels"])
        m = site[te] == 0; te, p, lab = te[m], p[m], lab[m]
        trg = set(G[train_of(r)]); leak = np.array([g in trg for g in G[te]])
        d = dict(acc=accuracy_score(lab, p), f1=f1_score(lab, p, average="macro"), n=len(te), n_leak=int(leak.sum()),
                 acc_leak=accuracy_score(lab[leak], p[leak]) if leak.any() else np.nan,
                 acc_clean=accuracy_score(lab[~leak], p[~leak]) if (~leak).any() else np.nan)
        rows.append(d)
    agg = {k: (float(np.nanmean([d[k] for d in rows])), float(np.nanstd([d[k] for d in rows], ddof=1))) for k in rows[0]}
    out[name] = agg; print(name, {k: f"{v[0]*(100 if k not in ('n','n_leak') else 1):.1f}±{v[1]*(100 if k not in ('n','n_leak') else 1):.1f}" for k, v in agg.items()})
ids0 = np.where(site == 0)[0]
paths = [r["path"] for r in recs]
off_tr = np.array([i for i in ids0 if "/Training-validation/" in paths[i]])
from sklearn.model_selection import StratifiedKFold
def rand_tr(r):
    a, b = next(iter(StratifiedKFold(n_splits=5, shuffle=True, random_state=r["seed"]).split(ids0, y[ids0]))); return ids0[a]
def group_tr(r):
    te = set(r["test_ids"]); return np.array([i for i in ids0 if i not in te])
summarise("official", [json.load(open(f"{S}/res/leak/official_s{s}.json")) for s in range(3)], lambda r: off_tr)
summarise("random", [json.load(open(f"{S}/res/leak/random_s{s}.json")) for s in range(3)], rand_tr)
summarise("group", [json.load(open(f"{S}/res/local/site0_s{s}.json")) for s in range(3)], group_tr)
json.dump(out, open(f"{S}/ana/leakage.json", "w"), indent=1)
