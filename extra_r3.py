"""Round-3 checks: (1) compute-matched comparisons restricted to test images NOT flagged by the
embedding duplicate detector (DINOv2, phash_p5 threshold); (2) per-class sensitivity at Site 2 for the
E3, 120-epoch local and FedPer models (S1). Two-level bootstrap (seeds, then near-duplicate groups)."""
import json, glob, sys, numpy as np
S = sys.argv[1]
IDX = json.load(open(f"{S}/cache/index.json"))["records"]; GRP = np.array([r["group"] for r in IDX])
DUP = json.load(open(f"{S}/ana2/dupcheck.json"))["dinov2_vitb14"]["thresholds"]["phash_p5"]["seeds"]
def load(pat): return {json.load(open(f))["seed"]: json.load(open(f)) for f in sorted(glob.glob(f"{S}/res2/{pat}_s*.json"))}
def comp(r0, r1):
    out = {}
    for sd in sorted(set(r0) & set(r1)):
        a, b = r0[sd], r1[sd]; P = np.array(a["probs"]); st = np.array(a["sites"]); ib = {t: i for i, t in enumerate(b["test_ids"])}
        P[st == 1] = np.array(b["probs"])[[ib[t] for t in a["test_ids"]]][st == 1]
        out[sd] = dict(probs=P, labels=a["labels"], sites=a["sites"], test_ids=a["test_ids"])
    return out
def cmg(y, p, g):
    ug, inv = np.unique(g, return_inverse=True); c = np.zeros((len(ug), 5, 5)); np.add.at(c, (inv, y, p), 1); return c
def mf1(c):
    tp = np.diagonal(c, axis1=-2, axis2=-1); t = c.sum(-1); pr = c.sum(-2); d = t + pr
    f = np.where(d > 0, 2 * tp / np.maximum(d, 1e-12), 0); pres = d > 0
    return (f * pres).sum(-1) / np.maximum(pres.sum(-1), 1)
def per_seed(r, sd, clean):
    te = np.array(r["test_ids"]); y = np.array(r["labels"]); p = np.array(r["probs"]).argmax(1)
    k = np.ones(len(te), bool)
    if clean: fl = set(DUP[str(sd)]["flagged"]); k = np.array([t not in fl for t in te])
    o = np.argsort(te[k]); return y[k][o], p[k][o], GRP[te[k]][o]
def paired(A, B, clean, Bn=10000, seed=0):
    rng = np.random.default_rng(seed); seeds = sorted(set(A) & set(B)); rep = np.zeros((len(seeds), Bn)); obs = []
    for j, sd in enumerate(seeds):
        ya, pa, ga = per_seed(A[sd], sd, clean); yb, pb, gb = per_seed(B[sd], sd, clean); assert (ga == gb).all()
        ca, cb = cmg(ya, pa, ga), cmg(yb, pb, gb); G = len(ca)
        obs.append(mf1(ca.sum(0)) - mf1(cb.sum(0)))
        w = np.stack([np.bincount(rng.integers(0, G, G), minlength=G) for _ in range(Bn)])
        rep[j] = mf1(np.tensordot(w, ca, (1, 0))) - mf1(np.tensordot(w, cb, (1, 0)))
    pick = rng.integers(0, len(seeds), (Bn, len(seeds))); d = rep[pick, np.arange(Bn)[:, None]].mean(1)
    p = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())); p = max(p, 1 / Bn)
    return dict(diff=float(np.mean(obs)), lo=float(np.percentile(d, 2.5)), hi=float(np.percentile(d, 97.5)), p=float(p), n=len(seeds))
def mean_f1(A, clean): return float(np.mean([mf1(cmg(*per_seed(A[s], s, clean)).sum(0)) for s in A]))
L120 = comp(load("tune/local/site0_e120"), load("tune/local/site1_e120"))
L40 = {k: v for k, v in comp(load("local/site0"), load("local/site1")).items() if k < 3}
M = {"S1": dict(local120=L120, local40=L40, fedbn_E3=load("tune/S1/fedbn_E3"), fedavg_E3=load("tune/S1/fedavg_E3"),
                fedbn={k: v for k, v in load("fl/S1/fedbn").items() if k < 3}, fedper=load("tune/S1/fedper")),
     "S2": dict(local120=load("tune/S2/local_E3"), fedbn_E3=load("tune/S2/fedbn_E3"), fedavg_E3=load("tune/S2/fedavg_E3")),
     "S3": dict(local120=load("tune/S3/local_E3"), fedbn_E3=load("tune/S3/fedbn_E3"), fedavg_E3=load("tune/S3/fedavg_E3"))}
out = {"compute_matched": {}, "perclass_site2": {}}
for st, m in M.items():
    for clean in (False, True):
        tag = "clean" if clean else "all"
        for a in ("fedbn_E3", "fedavg_E3"):
            r = paired(m[a], m["local120"], clean); r["f1_a"] = mean_f1(m[a], clean); r["f1_local120"] = mean_f1(m["local120"], clean)
            out["compute_matched"][f"{st}|{a}|{tag}"] = r
            print(st, a, "vs local120", tag, {k: round(100 * v, 1) if k not in ("p", "n") else v for k, v in r.items()})
def sens(A, site):
    se, rep = [], []
    for sd, r in A.items():
        y = np.array(r["labels"]); p = np.array(r["probs"]).argmax(1); s = np.array(r["sites"]); k = s == site
        se.append([((p[k] == c) & (y[k] == c)).sum() / max((y[k] == c).sum(), 1) if (y[k] == c).any() else np.nan for c in range(5)])
    return np.nanmean(se, 0)
for nm in ("local40", "local120", "fedbn", "fedbn_E3", "fedavg_E3", "fedper"):
    s2 = sens(M["S1"][nm], 1); s1 = sens(M["S1"][nm], 0)
    out["perclass_site2"][nm] = dict(site2=s2.tolist(), site1=s1.tolist())
    print(nm, "Site2 sens N/E/AOM/COM/My", np.round(100 * s2), " Site1", np.round(100 * s1))
json.dump(out, open(f"{S}/ana2/extra_r3.json", "w"), indent=1)
