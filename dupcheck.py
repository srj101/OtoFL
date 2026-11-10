"""Embedding-based near-duplicate check (second detector) on the group-aware splits."""
import os, sys, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import otofl
cache, emb, out = sys.argv[1], sys.argv[2], sys.argv[3]
X, y, site, grp, classes = otofl.load(cache)
paths = [r["path"] for r in json.load(open(f"{cache}/index.json"))["records"]]
E = dict(np.load(emb)); R = {}
n = len(y); same_site = site[:, None] == site[None, :]; iu = np.triu_indices(n, 1)
ph_pair = (grp[:, None] == grp[None, :]) & same_site
for name, F in E.items():
    Sm = F @ F.T
    ps = Sm[iu][ph_pair[iu]]
    thr = {"phash_min": float(ps.min()), "phash_p5": float(np.percentile(ps, 5)), "0.95": 0.95, "0.90": 0.90}
    info = dict(phash_pair_sim=dict(n=int(ps.size), min=float(ps.min()), p5=float(np.percentile(ps, 5)), median=float(np.median(ps))),
                nonpair_p99=float(np.percentile(Sm[iu][same_site[iu] & ~ph_pair[iu]], 99)))
    per_t = {}
    for tn, t in thr.items():
        L = (Sm >= t) & same_site; np.fill_diagonal(L, False)
        par = list(range(n))
        def f(i):
            while par[i] != i: par[i] = par[par[i]]; i = par[i]
            return i
        a, b = np.where(np.triu(L, 1))
        for i, j in zip(a, b): par[f(i)] = f(j)
        comp = np.array([f(i) for i in range(n)])
        cross = int(((y[a] != y[b])).sum())
        d = dict(t=t, pairs=int(len(a)), cross_label_pairs=cross,
                 clusters_site1=int(len(set(comp[site == 0]))), clusters_site2=int(len(set(comp[site == 1]))))
        tr_o = np.array([i for i in range(n) if site[i] == 0 and "/Training-validation/" in paths[i]])
        te_o = np.array([i for i in range(n) if site[i] == 0 and "/Testing/" in paths[i]])
        d["official_test_flagged"] = int((Sm[np.ix_(te_o, tr_o)].max(1) >= t).sum())
        seeds = {}
        for s in range(8):
            tr, te = otofl.split(y, site, grp, s)
            fl = []
            for st in (0, 1):
                a_ = te[site[te] == st]; b_ = tr[site[tr] == st]
                m = Sm[np.ix_(a_, b_)].max(1)
                fl += a_[m >= t].tolist()
            seeds[s] = dict(n_test=int(len(te)), flagged=sorted(int(v) for v in fl))
        d["seeds"] = seeds
        per_t[tn] = d
    info["thresholds"] = per_t
    R[name] = info
json.dump(R, open(out, "w"))
for k, v in R.items():
    print(k, v["phash_pair_sim"], "nonpair_p99", round(v["nonpair_p99"], 3))
    for tn, d in v["thresholds"].items():
        print("  ", tn, round(d["t"], 3), "pairs", d["pairs"], "cross", d["cross_label_pairs"], "clusters S1/S2", d["clusters_site1"], d["clusters_site2"],
              "official", d["official_test_flagged"], "flagged/seed", [len(x["flagged"]) for x in d["seeds"].values()])
