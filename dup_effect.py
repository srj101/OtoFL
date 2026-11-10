"""Accuracy on test images flagged by the embedding detector (a same-site training image with cosine >= t)
vs. unflagged ones, for models trained on the group-aware splits (seeds 0-7)."""
import json, sys, glob, numpy as np
S = sys.argv[1]; R = json.load(open(f"{S}/ana2/dupcheck.json"))
recs = json.load(open(f"{S}/cache/index.json"))["records"]; site = np.array([0 if r["site"] == "chile" else 1 for r in recs])
def runs(pat): return {json.load(open(f))["seed"]: json.load(open(f)) for f in glob.glob(f"{S}/res2/{pat}_s*.json")}
sets = {"local Site1": runs("local/site0"), "local Site2": runs("local/site1"), "FedBN S1": runs("fl/S1/fedbn"),
        "FedAvg S1": runs("fl/S1/fedavg"), "AP-FedBN S1": runs("fl/S1/otofl")}
out = {}
for emb in ["dinov2_vitb14", "resnet50"]:
    for tn in ["phash_p5", "0.95"]:
        th = R[emb]["thresholds"][tn]
        for nm, rs in sets.items():
            for st in (0, 1):
                if nm.startswith("local") and int(nm[-1]) - 1 != st: continue
                a_f, a_c, n_f, n_c = [], [], [], []
                for sd, r in sorted(rs.items()):
                    fl = set(th["seeds"][str(sd)]["flagged"])
                    te = np.array(r["test_ids"]); p = np.array(r["probs"]).argmax(1); y = np.array(r["labels"])
                    k = site[te] == st; te, p, y = te[k], p[k], y[k]
                    f = np.array([t in fl for t in te])
                    if f.any(): a_f.append((p[f] == y[f]).mean())
                    if (~f).any(): a_c.append((p[~f] == y[~f]).mean())
                    n_f.append(f.sum()); n_c.append((~f).sum())
                key = f"{emb}|{tn}|{nm}|site{st+1}"
                out[key] = dict(n_seeds=len(rs), n_flag=float(np.mean(n_f)), n_clean=float(np.mean(n_c)),
                                acc_flag=float(np.mean(a_f)) if a_f else None, acc_clean=float(np.mean(a_c)) if a_c else None,
                                sd_flag=float(np.std(a_f, ddof=1)) if len(a_f) > 1 else None, sd_clean=float(np.std(a_c, ddof=1)) if len(a_c) > 1 else None)
                print(key, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out[key].items()})
json.dump(out, open(f"{S}/ana2/dup_effect.json", "w"), indent=1)
