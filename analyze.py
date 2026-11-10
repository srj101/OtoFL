"""Aggregate experiment JSONs into metrics, LaTeX tables and PDF figures.

Usage: python analyze.py RES_DIR OUT_DIR
Every number in the paper comes from this script. Missing runs are skipped.
"""
import sys, os, json, glob
import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score,
                             cohen_kappa_score, matthews_corrcoef, confusion_matrix,
                             precision_recall_fscore_support)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES, OUT = sys.argv[1], sys.argv[2]
FIG = os.path.join(OUT, "figures"); TAB = os.path.join(OUT, "tables")
os.makedirs(FIG, exist_ok=True); os.makedirs(TAB, exist_ok=True)
CLS = ["Normal", "Earwax", "AOM", "COM", "Myring."]

# fixed categorical order (validated reference palette), colour follows the method
COL = {"otofl": "#2a78d6", "fedavg": "#eb6834", "fedprox": "#1baf7a", "scaffold": "#eda100",
       "fedbn": "#e87ba4", "central": "#008300", "local": "#6250d6", "otofl_nobn": "#e34948",
       "otofl_fixedmu": "#6250d6"}
NAME = {"fedavg": "FedAvg", "fedprox": "FedProx", "scaffold": "SCAFFOLD", "fedbn": "FedBN",
        "otofl": "AP-FedBN", "otofl_nobn": "Adaptive $\\mu_k$, shared norm", "otofl_fixedmu": "FedBN + fixed $\\mu$"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.5,
                     "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
                     "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
                     "grid.color": GRID, "grid.linewidth": 0.5, "pdf.fonttype": 42, "savefig.dpi": 300,
                     "legend.frameon": False})


def load(pattern):
    out = {}
    for f in sorted(glob.glob(os.path.join(RES, pattern))):
        out[f] = json.load(open(f))
    return out


def metrics(r):
    P = np.array(r["probs"]); y = np.array(r["labels"]); s = np.array(r["sites"])
    pred = P.argmax(1)
    labels = list(range(P.shape[1]))
    m = dict(acc=accuracy_score(y, pred), bacc=balanced_accuracy_score(y, pred),
             f1=f1_score(y, pred, average="macro", labels=labels, zero_division=0),
             kappa=cohen_kappa_score(y, pred), mcc=matthews_corrcoef(y, pred))
    try:
        m["auc"] = roc_auc_score(y, P, multi_class="ovr", average="macro", labels=labels)
    except ValueError:
        m["auc"] = np.nan
    for si, nm in ((0, "acc_chile"), (1, "acc_turkey")):
        k = s == si
        m[nm] = accuracy_score(y[k], pred[k]) if k.any() else np.nan
    conf = P.max(1); bins = np.linspace(0, 1, 11); ece = 0
    for a, b in zip(bins[:-1], bins[1:]):
        k = (conf > a) & (conf <= b)
        if k.any():
            ece += k.mean() * abs((pred[k] == y[k]).mean() - conf[k].mean())
    m["ece"] = ece
    if "per_client" in r:
        accs = [c["acc"] for c in r["per_client"] if c["acc"] is not None and c["n_test"] >= 5]
        m["worst"] = min(accs) if accs else np.nan
    m["cm"] = confusion_matrix(y, pred, labels=labels)
    return m


def agg(runs, key):
    v = np.array([metrics(r)[key] for r in runs], dtype=float)
    return v.mean(), v.std(ddof=1) if len(v) > 1 else 0.0, len(v)


def fmt(mu, sd, n, pct=True, bold=False):
    if n == 0 or np.isnan(mu):
        return "--"
    s = f"{100*mu:.1f}$\\pm${100*sd:.1f}" if pct else f"{mu:.3f}$\\pm${sd:.3f}"
    return f"\\textbf{{{s}}}" if bold else s


def group(prefix, method):
    return [r for f, r in load(f"{prefix}/{method}_s*.json").items()]


_IDX = json.load(open(os.environ.get("OTOFL_INDEX", os.path.join(os.path.dirname(os.path.abspath(RES)), "cache", "index.json"))))
GRP = np.array([r["group"] for r in _IDX["records"]])


def _cm_by_group(y, pred, g, n_cls=5):
    """Per-group confusion counts: array (G, C, C) for the unique groups in g."""
    ug, inv = np.unique(g, return_inverse=True)
    cm = np.zeros((len(ug), n_cls, n_cls))
    np.add.at(cm, (inv, y, pred), 1)
    return cm


def _macro_f1(cm):
    """Macro-F1 from (..., C, C) confusion counts, averaged over classes present in truth or prediction
    (same convention as sklearn's f1_score(average='macro'))."""
    tp = np.diagonal(cm, axis1=-2, axis2=-1)
    t = cm.sum(-1); pr = cm.sum(-2)
    den = t + pr
    f1 = np.where(den > 0, 2 * tp / np.maximum(den, 1e-12), 0.0)
    present = den > 0
    return (f1 * present).sum(-1) / np.maximum(present.sum(-1), 1)


IMSITE = np.array([0 if r["site"] == "chile" else 1 for r in _IDX["records"]])
LAST = {}


def _strat_weights(gsite, n, rng):
    """Bootstrap counts for n replicates; whole groups are resampled within each site (stratified)."""
    G = len(gsite); w = np.zeros((n, G), dtype=np.int32)
    for s in np.unique(gsite):
        idx = np.where(gsite == s)[0]
        draw = idx[rng.integers(0, len(idx), (n, len(idx)))]
        w += np.stack([np.bincount(d, minlength=G) for d in draw])
    return w


def paired_boot(runs_a, runs_b, key="f1", B=10000, seed=0):
    """Two-level paired bootstrap of the macro-F1 difference (a - b).
    Level 1 resamples seeds with replacement (training variability: split, partition, initialisation).
    Level 2, inside every drawn seed, resamples whole near-duplicate groups of test images, stratified by
    site, with an independent draw for every slot (a seed drawn twice gets two different inner resamples).
    The statistic is the mean over seeds of the per-seed macro-F1 difference. Only seeds present in both
    arms are used; the number is recorded in LAST together with a seed-level paired t-interval."""
    rng = np.random.default_rng(seed)
    sa = {r["seed"]: r for r in runs_a}; sb = {r["seed"]: r for r in runs_b}
    seeds = sorted(set(sa) & set(sb))
    LAST.clear(); LAST.update(n=len(seeds), seeds_a=len(sa), seeds_b=len(sb))
    if not seeds:
        return np.nan, (np.nan, np.nan), np.nan
    per_seed = []
    for s in seeds:
        ra, rb = sa[s], sb[s]
        ia = {t: i for i, t in enumerate(ra["test_ids"])}
        order = [ia[t] for t in rb["test_ids"]]
        pa = np.array(ra["probs"])[order].argmax(1); pb = np.array(rb["probs"]).argmax(1)
        y = np.array(rb["labels"]); te = np.array(rb["test_ids"]); g = GRP[te]
        ug, first = np.unique(g, return_index=True)
        per_seed.append((_cm_by_group(y, pa, g), _cm_by_group(y, pb, g), IMSITE[te][first]))
    d_seed = np.array([_macro_f1(ca.sum(0)) - _macro_f1(cb.sum(0)) for ca, cb, _ in per_seed])
    obs = float(d_seed.mean())
    S_ = len(per_seed)
    pick = rng.integers(0, S_, (B, S_))
    diffs = np.zeros(B)
    for j, (ca, cb, gs) in enumerate(per_seed):
        slots = np.argwhere(pick == j)  # (b, slot) pairs that use seed j
        if len(slots) == 0:
            continue
        w = _strat_weights(gs, len(slots), rng)
        dj = _macro_f1(np.tensordot(w, ca, axes=(1, 0))) - _macro_f1(np.tensordot(w, cb, axes=(1, 0)))
        np.add.at(diffs, slots[:, 0], dj)
    diffs /= S_
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    from scipy import stats as _st
    if S_ > 1:
        se = d_seed.std(ddof=1) / np.sqrt(S_); tc = _st.t.ppf(0.975, S_ - 1)
        tp = 2 * _st.t.sf(abs(obs) / se, S_ - 1) if se > 0 else 0.0
        LAST.update(t_lo=obs - tc * se, t_hi=obs + tc * se, t_p=float(tp))
    LAST.update(per_seed=d_seed.tolist())
    return obs, (np.percentile(diffs, 2.5), np.percentile(diffs, 97.5)), max(p, 1 / B)


def sigrec(setting, a, vs, ra, rb):
    d, ci, p = paired_boot(ra, rb)
    out = dict(setting=setting, a=a, vs=vs, diff=d, lo=ci[0], hi=ci[1], p=p)
    out.update({k: v for k, v in LAST.items()})
    return out


summary = {}

# ------------------------------------------------------------ Table: centralised backbones
rows = []
CB = [("fenet5", "Fenet5 (scratch)"), ("mobilenet_v3_large", "MobileNetV3-L"),
      ("efficientnet_b0", "EfficientNet-B0"), ("resnet50", "ResNet-50"), ("densenet121", "DenseNet-121")]
cen = {k: group("central", k) for k, _ in CB}
best = {key: max((agg(v, key)[0] for v in cen.values() if v), default=0) for key in ["acc", "bacc", "f1", "auc", "mcc"]}
for k, nm in CB:
    v = cen[k]
    if not v:
        continue
    npar = v[0]["n_params"] / 1e6
    cells = [fmt(*agg(v, key), bold=abs(agg(v, key)[0] - best[key]) < 1e-9) for key in ["acc", "bacc", "f1"]]
    cells.append(fmt(*agg(v, "auc"), pct=False, bold=abs(agg(v, "auc")[0] - best["auc"]) < 1e-9))
    cells.append(fmt(*agg(v, "mcc"), pct=False, bold=abs(agg(v, "mcc")[0] - best["mcc"]) < 1e-9))
    rows.append(f"{nm} & {npar:.2f} & " + " & ".join(cells) + " \\\\")
    summary[f"central_{k}"] = {key: agg(v, key)[:2] for key in ["acc", "bacc", "f1", "auc", "mcc", "acc_chile", "acc_turkey", "ece"]}
open(os.path.join(TAB, "central.tex"), "w").write("\n".join(rows))

# ------------------------------------------------------------ Table: FL methods x settings
M = ["fedavg", "fedprox", "scaffold", "fedbn", "otofl"]
ST = [("S1", "fl/S1"), ("S2", "fl/S2"), ("S3", "fl/S3")]
fl = {(st, m): group(p, m) for st, p in ST for m in M}
rows = []
for st, _ in ST:
    fl[(st, "fedper")] = group(f"tune/{st}", "fedper")
bestf = {(st, key): max((agg(fl[(st, m)], key)[0] for m in M + ["fedper"] if len(fl[(st, m)]) > 3), default=0)
         for st, _ in ST for key in ["bacc", "f1", "worst"]}
for m in M:
    cells = []
    for st, _ in ST:
        v = fl[(st, m)]
        for key in ["bacc", "f1"] + (["worst"] if st == "S2" else []):
            if v:
                mu = agg(v, key)
                cells.append(fmt(*mu, bold=abs(mu[0] - bestf[(st, key)]) < 1e-9))
            else:
                cells.append("--")
    comm = f"{fl[('S2', m)][0]['comm_MB']:.0f}" if fl[("S2", m)] else "--"
    rows.append(f"{NAME[m]} & " + " & ".join(cells) + " \\\\")
    for st, _ in ST:
        if fl[(st, m)]:
            summary[f"fl_{st}_{m}"] = {key: agg(fl[(st, m)], key)[:2] for key in
                                       ["acc", "bacc", "f1", "auc", "mcc", "kappa", "worst", "acc_chile", "acc_turkey", "ece"]}
# FedPer (local normalisation and local classifier head), eight seeds
if all(len(fl[(st, "fedper")]) > 3 for st, _ in ST):
    cells = []
    for st, _ in ST:
        v = fl[(st, "fedper")]
        for key in ["bacc", "f1"] + (["worst"] if st == "S2" else []):
            mu = agg(v, key)
            cells.append(fmt(*mu, bold=abs(mu[0] - bestf[(st, key)]) < 1e-9))
        summary[f"fl_{st}_fedper"] = {key: agg(v, key)[:2] for key in
                                      ["acc", "bacc", "f1", "auc", "mcc", "kappa", "worst", "acc_chile", "acc_turkey", "ece"]}
    rows.append("FedPer$^{b}$ & " + " & ".join(cells) + " \\\\")
# personalised FedBN: brief local fine-tuning after federated training
for st, p_ in ST:
    fl[(st, "fedbn_ft")] = group(p_, "fedbn_ft")
if any(fl[(st, "fedbn_ft")] for st, _ in ST):
    cells = []
    for st, _ in ST:
        v = fl[(st, "fedbn_ft")]
        for key in ["bacc", "f1"] + (["worst"] if st == "S2" else []):
            cells.append(fmt(*agg(v, key)) if v else "--")
        if v:
            summary[f"fl_{st}_fedbn_ft"] = {key: agg(v, key)[:2] for key in
                                           ["acc", "bacc", "f1", "auc", "worst", "acc_chile", "acc_turkey", "ece"]}
    comm = f"{fl[('S2', 'fedbn_ft')][0]['comm_MB']:.0f}" if fl[("S2", "fedbn_ft")] else "--"
    rows.append("\\midrule FedBN + fine-tuning & " + " & ".join(cells) + " \\\\")
# centralised and local reference rows
loc = {s_: {r["seed"]: r for r in group("local", f"site{s_}")} for s_ in (0, 1)}
if loc[0] and loc[1]:
    comp = []
    for sd_ in sorted(loc[0]):
        a, b = loc[0][sd_], loc[1][sd_]
        P = np.array(a["probs"]); Pb = np.array(b["probs"]); st_ = np.array(a["sites"])
        ib = {t: i for i, t in enumerate(b["test_ids"])}
        Pb = Pb[[ib[t] for t in a["test_ids"]]]
        P[st_ == 1] = Pb[st_ == 1]
        comp.append(dict(probs=P.tolist(), labels=a["labels"], sites=a["sites"], seed=sd_, test_ids=a["test_ids"]))
    cells_ = [fmt(*agg(comp, "bacc")), fmt(*agg(comp, "f1"))]
    for st_ in ("S2", "S3"):
        v_ = group(f"fl/{st_}", "local")
        if v_:
            fl[(st_, "local")] = v_
            cells_ += [fmt(*agg(v_, k_)) for k_ in (["bacc", "f1"] + (["worst"] if st_ == "S2" else []))]
            summary[f"fl_{st_}_local"] = {k_: agg(v_, k_)[:2] for k_ in ["acc", "bacc", "f1", "worst", "acc_chile", "acc_turkey"]}
        else:
            cells_ += ["--"] * (3 if st_ == "S2" else 2)
    rows.append("\\midrule Local only (ref.) & " + " & ".join(cells_) + " \\\\")
    summary["local_composite"] = {key: agg(comp, key)[:2] for key in ["acc", "bacc", "f1", "auc", "acc_chile", "acc_turkey"]}
    fl[("S1", "local")] = comp
if cen["fenet5"]:
    c = cen["fenet5"]
    b, f = fmt(*agg(c, "bacc")), fmt(*agg(c, "f1"))
    rows.append(f"Centralised (ref.) & {b} & {f} & {b} & {f} & -- & {b} & {f} \\\\")
open(os.path.join(TAB, "fl.tex"), "w").write("\n".join(rows))

# significance: OtoFL vs each baseline (macro-F1), per setting
sig = []
for st, _ in ST:
    for a_, b_ in [("fedbn", "fedavg"), ("fedbn", "fedprox"), ("fedbn", "scaffold"), ("fedprox", "fedavg"),
                   ("otofl", "fedavg"), ("otofl", "fedbn"), ("fedbn_ft", "fedbn"), ("fedper", "fedbn")]:
        if fl[(st, a_)] and fl[(st, b_)]:
            sig.append(sigrec(st, a_, b_, fl[(st, a_)], fl[(st, b_)]))
for st_ in ("S1", "S2", "S3"):
    if fl.get((st_, "local")):
        for m in ["fedavg", "fedbn", "otofl", "fedbn_ft", "fedper"]:
            if not fl.get((st_, m)):
                continue
            sig.append(sigrec(st_, m, "local", fl[(st_, m)], fl[(st_, "local")]))
summary["significance"] = sig

# ------------------------------------------------------------ Table: ablation (S2)
abl = [("fedavg", "fl/S2", "FedAvg & shared & 0"),
       ("otofl_nobn", "abl/S2", "Adaptive $\\mu_k$ only & shared & $\\mu_k$, $\\lambda=9$"),
       ("fedbn", "fl/S2", "FedBN & local & 0"),
       ("otofl_fixedmu", "abl/S2", "FedBN + FedProx & local & 0.01"),
       ("lam3", "abl/S2", "AP-FedBN & local & $\\mu_k$, $\\lambda=3$"),
       ("otofl", "fl/S2", "AP-FedBN & local & $\\mu_k$, $\\lambda=9$"),
       ("lam27", "abl/S2", "AP-FedBN & local & $\\mu_k$, $\\lambda=27$")]
rows = []
for k, p, nm in abl:
    v = group(p, k)
    if not v:
        continue
    rows.append(f"{nm} & " + " & ".join(fmt(*agg(v, key)) for key in ["acc", "bacc", "f1", "worst"]) + " \\\\")
    summary[f"abl_{k}" if k != "fedbn" else "abl_fedbn"] = {key: agg(v, key)[:2] for key in ["acc", "bacc", "f1", "worst"]}
open(os.path.join(TAB, "ablation.tex"), "w").write("\n".join(rows))

# ------------------------------------------------------------ DP
dp = {}
for e in [1, 2, 4, 8, 0]:
    v = group("dp", f"eps{e}")
    if v:
        dp[e] = v
fav = group("dp", "fedavg_eps0")
rows = []
for e, v in dp.items():
    lab = "$\\infty$ (no DP)" if e == 0 else str(e)
    sig_ = np.mean([r["sigma"] for r in v]) if e else 0
    rows.append(f"{lab} & {sig_:.2f} & " + " & ".join(fmt(*agg(v, key)) for key in ["acc", "bacc", "f1"]) +
                f" & {fmt(*agg(v, 'auc'), pct=False)} \\\\")
    summary[f"dp_{e}"] = {key: agg(v, key)[:2] for key in ["acc", "bacc", "f1", "auc"]}
    summary[f"dp_{e}"]["sigma"] = float(sig_)
if fav:
    summary["dp_fedavg_nodp"] = {key: agg(fav, key)[:2] for key in ["acc", "bacc", "f1", "auc"]}
open(os.path.join(TAB, "dp.tex"), "w").write("\n".join(rows))

# ------------------------------------------------------------ MobileNetV3 FL (backbone generality)
for m in ["fedavg", "fedprox", "fedbn", "otofl"]:
    v = group("fl_mnv3/S2", m)
    if v:
        summary[f"mnv3_S2_{m}"] = {key: agg(v, key)[:2] for key in ["acc", "bacc", "f1", "worst"]}

# ------------------------------------------------------------ local-only & cross-site
for site in [0, 1]:
    v = group("local", f"site{site}")
    if v:
        summary[f"local_site{site}"] = {key: agg(v, key)[:2] for key in ["acc", "bacc", "f1", "acc_chile", "acc_turkey"]}

# ------------------------------------------------------------ scalability
for K in [5, 20]:
    for m in ["fedavg", "otofl"]:
        v = group(f"scale/K{K}", m)
        if v:
            summary[f"scale_K{K}_{m}"] = {key: agg(v, key)[:2] for key in ["acc", "bacc", "f1", "worst"]}

# ------------------------------------------------------------ tuned / added baselines (seeds 0-2)
TUNE = [("fedavg", "fl", "FedAvg"), ("fedavg_gn", "tune", "FedAvg + GroupNorm"),
        ("fedprox", "fl", "FedProx, $\\mu=0.01$"), ("fedprox_mu0.1", "tune", "FedProx, $\\mu=0.1$"),
        ("fedprox_mu1", "tune", "FedProx, $\\mu=1$"), ("scaffold", "fl", "SCAFFOLD, plain SGD"),
        ("scaffold_mom", "tune", "SCAFFOLD, momentum + wd"), ("silobn", "tune", "SiloBN"),
        ("fedper", "tune", "FedPer (local norm.\\ + head)"), ("fedbn", "fl", "FedBN"), ("otofl", "fl", "AP-FedBN")]
s012 = lambda v: [r for r in v if r["seed"] < 3]
tuned = {}
rows = []
for k, d_, nm in TUNE:
    cells = []
    for st, _ in ST:
        v = s012(group(f"{d_}/{st}", k))
        tuned[(st, k)] = v
        cells.append(fmt(*agg(v, "f1")) if v else "--")
        if v:
            summary[f"tune_{st}_{k}"] = {key: agg(v, key)[:2] for key in ["acc", "bacc", "f1", "worst", "acc_chile", "acc_turkey"]}
    rows.append(f"{nm} & " + " & ".join(cells) + " \\\\")
# S2 with three local epochs
for k, nm in [("fedavg", "FedAvg"), ("fedprox", "FedProx"), ("scaffold", "SCAFFOLD"), ("fedbn", "FedBN")]:
    v = s012(group("tune/S2", f"{k}_E3"))
    tuned[("S2", f"{k}_E3")] = v
    if v:
        summary[f"tune_S2_{k}_E3"] = {key: agg(v, key)[:2] for key in ["acc", "bacc", "f1", "worst"]}
v = s012(group("tune/S2", "otofl_proxfix")); tuned[("S2", "otofl_proxfix")] = v
if v:
    summary["tune_S2_otofl_proxfix"] = {key: agg(v, key)[:2] for key in ["acc", "bacc", "f1", "worst"]}
open(os.path.join(TAB, "tuned.tex"), "w").write("\n".join(rows))
sig_t = []
for st, _ in ST:
    for a_ in ["fedavg_gn", "fedprox_mu0.1", "fedprox_mu1", "scaffold_mom", "silobn", "fedper"]:
        if tuned.get((st, a_)) and tuned.get((st, "fedbn")):
            sig_t.append(sigrec(st, "fedbn", a_, tuned[(st, "fedbn")], tuned[(st, a_)]))
E3 = {}
for st in ("S1", "S2", "S3"):
    for k in ("fedavg", "fedprox", "scaffold", "fedbn", "local"):
        v = group(f"tune/{st}", f"{k}_E3")
        if v:
            E3[(st, k)] = v
    v = group(f"tune/{st}", "fedbn_R120")
    if v:
        E3[(st, "fedbn_R120")] = v
for (st, k), v in E3.items():
    summary[f"e3_{st}_{k}"] = {key: agg(v, key)[:2] for key in ["acc", "bacc", "f1", "worst", "acc_chile", "acc_turkey"]}
    summary[f"e3_{st}_{k}"]["n"] = len(v)
for k in ["fedavg", "fedprox", "scaffold", "fedbn"]:
    if E3.get(("S2", k)) and fl.get(("S2", k)):
        sig_t.append(sigrec("S2", f"{k}_E3", k, E3[("S2", k)], fl[("S2", k)]))
for st in ("S1", "S3"):
    for k in ("fedavg", "fedbn"):
        if E3.get((st, k)):
            sig_t.append(sigrec(st, f"{k}_E3", k, E3[(st, k)], fl[(st, k)]))
if tuned.get(("S2", "otofl_proxfix")):
    sig_t.append(sigrec("S2", "otofl_proxfix", "fedbn", tuned[("S2", "otofl_proxfix")], tuned[("S2", "fedbn")]))
# longer training budget: three local epochs per round vs. local training for 120 epochs
def composite(r0, r1):
    out = []
    for sd_ in sorted(set(r0) & set(r1)):
        a, b = r0[sd_], r1[sd_]
        P = np.array(a["probs"]); Pb = np.array(b["probs"]); st_ = np.array(a["sites"])
        ib = {t: i for i, t in enumerate(b["test_ids"])}
        P[st_ == 1] = Pb[[ib[t] for t in a["test_ids"]]][st_ == 1]
        out.append(dict(probs=P.tolist(), labels=a["labels"], sites=a["sites"], seed=sd_, test_ids=a["test_ids"]))
    return out
l120 = {t: {r["seed"]: r for r in group("tune/local", f"site{t}_e120")} for t in (0, 1)}
cm = {("S1", "local_E3"): composite(l120[0], l120[1]) if l120[0] and l120[1] else []}
for st in ("S1", "S2", "S3"):
    for k in ("fedavg", "fedbn", "fedbn_R120"):
        if E3.get((st, k)):
            cm[(st, f"{k}_E3" if k != "fedbn_R120" else k)] = E3[(st, k)]
    if st != "S1" and E3.get((st, "local")):
        cm[(st, "local_E3")] = E3[(st, "local")]
for (st, k), v in cm.items():
    if v:
        summary[f"cm_{st}_{k}"] = {key: agg(v, key)[:2] for key in ["acc", "bacc", "f1", "acc_chile", "acc_turkey"]}
        summary[f"cm_{st}_{k}"]["n"] = len(v)
cmrows = {}
for st in ("S1", "S2", "S3"):
    loc = cm.get((st, "local_E3"))
    if not loc:
        continue
    sd_ = {r["seed"] for r in loc}
    base = [r for r in fl[(st, "local")] if r["seed"] in sd_]
    summary[f"cm_{st}_local40"] = {key: agg(base, key)[:2] for key in ["f1"]}
    sig_t.append(sigrec(st, "local_E3", "local", loc, base))
    for a_ in ("fedbn_E3", "fedavg_E3", "fedbn_R120"):
        if cm.get((st, a_)):
            sig_t.append(sigrec(st, a_, "local_E3", cm[(st, a_)], loc))
    if cm.get((st, "fedbn_R120")) and cm.get((st, "fedbn_E3")):
        sig_t.append(sigrec(st, "fedbn_R120", "fedbn_E3", cm[(st, "fedbn_R120")], cm[(st, "fedbn_E3")]))
    if cm.get((st, "fedbn_R120")):
        sig_t.append(sigrec(st, "fedbn_R120", "fedbn", cm[(st, "fedbn_R120")], fl[(st, "fedbn")]))
# FedPer with all available seeds (main schedule and three local epochs per round)
FP = {}
for st in ("S1", "S2", "S3"):
    FP[(st, "fedper")] = group(f"tune/{st}", "fedper")
    FP[(st, "fedper_E3")] = group(f"tune/{st}", "fedper_E3")
    for k, v in FP.items():
        if k[0] == st and v:
            summary[f"fp_{st}_{k[1]}"] = {key: agg(v, key)[:2] for key in ["acc", "bacc", "f1", "worst", "acc_chile", "acc_turkey"]}
            summary[f"fp_{st}_{k[1]}"]["n"] = len(v)
    v8, vE = FP[(st, "fedper")], FP[(st, "fedper_E3")]
    if len(v8) > 3:
        sig_t.append(sigrec(st, "fedper8", "fedbn", v8, fl[(st, "fedbn")]))
        sig_t.append(sigrec(st, "fedper8", "local", v8, fl[(st, "local")]))
    if vE:
        cm[(st, "fedper_E3")] = vE
        if cm.get((st, "local_E3")):
            sig_t.append(sigrec(st, "fedper_E3", "local_E3", vE, cm[(st, "local_E3")]))
        if cm.get((st, "fedbn_E3")):
            sig_t.append(sigrec(st, "fedper_E3", "fedbn_E3", vE, cm[(st, "fedbn_E3")]))
        if len(v8) > 3:
            sig_t.append(sigrec(st, "fedper_E3", "fedper8", vE, v8))
# table rows for the longer-budget comparison
def _cell(v):
    return fmt(*agg(v, "f1")) if v else "--"
def _drow(st, a_):
    m = [s for s in sig_t if s["setting"] == st and s["a"] == a_ and s["vs"] == "local_E3"]
    if not m:
        return "--"
    s = m[0]
    return f"${100*s['diff']:+.1f}$ [{100*s['lo']:.1f}, {100*s['hi']:.1f}]"
R = []
seedsets = {st: sorted({r["seed"] for r in cm.get((st, "local_E3"), [])}) for st in ("S1", "S2", "S3")}
R.append("Local, 40 epochs (main) & " + " & ".join(_cell([r for r in fl[(st, "local")] if r["seed"] in seedsets[st]]) for st in ("S1", "S2", "S3")) + " \\\\")
R.append("Local, 120 epochs & " + " & ".join(_cell(cm.get((st, "local_E3"))) for st in ("S1", "S2", "S3")) + " \\\\")
R.append("FedAvg, E3 & " + " & ".join(_cell(cm.get((st, "fedavg_E3"))) for st in ("S1", "S2", "S3")) + " \\\\")
R.append("FedBN, E3 & " + " & ".join(_cell(cm.get((st, "fedbn_E3"))) for st in ("S1", "S2", "S3")) + " \\\\")
R.append("FedPer, E3 & " + " & ".join(_cell(cm.get((st, "fedper_E3"))) for st in ("S1", "S2", "S3")) + " \\\\")
R.append("FedBN, 120 rounds of 1 epoch$^{a}$ & " + " & ".join(_cell(cm.get((st, "fedbn_R120"))) for st in ("S1", "S2", "S3")) + " \\\\")
R.append("\\midrule")
R.append("$\\Delta$ FedAvg E3 $-$ local 120 & " + " & ".join(_drow(st, "fedavg_E3") for st in ("S1", "S2", "S3")) + " \\\\")
R.append("$\\Delta$ FedBN E3 $-$ local 120 & " + " & ".join(_drow(st, "fedbn_E3") for st in ("S1", "S2", "S3")) + " \\\\")
R.append("$\\Delta$ FedPer E3 $-$ local 120 & " + " & ".join(_drow(st, "fedper_E3") for st in ("S1", "S2", "S3")) + " \\\\")
open(os.path.join(TAB, "cm.tex"), "w").write("\n".join(R))
summary["significance_tuned"] = sig_t

# ------------------------------------------------------------ per-class sensitivity / specificity (S1)
def perclass(runs, site=None, B=4000, seed=1):
    """Sensitivity and specificity per class, averaged over seeds, with two-level bootstrap CIs."""
    rng = np.random.default_rng(seed)
    per = []
    for r in runs:
        y = np.array(r["labels"]); pr = np.array(r["probs"]).argmax(1); s_ = np.array(r["sites"])
        k = np.ones_like(y, bool) if site is None else s_ == site
        per.append(_cm_by_group(y[k], pr[k], GRP[np.array(r["test_ids"])][k]))
    def stat(cm):  # (..., C, C)
        tp = np.diagonal(cm, axis1=-2, axis2=-1); t = cm.sum(-1); p_ = cm.sum(-2); n = cm.sum((-1, -2))[..., None]
        sens = np.where(t > 0, tp / np.maximum(t, 1), np.nan)
        spec = np.where(n - t > 0, (n - t - p_ + tp) / np.maximum(n - t, 1), np.nan)
        return sens, spec
    obs = [stat(c.sum(0)) for c in per]
    se = np.nanmean([o[0] for o in obs], 0); sp = np.nanmean([o[1] for o in obs], 0)
    reps_se, reps_sp = [], []
    for c in per:
        G = len(c); w = np.stack([np.bincount(rng.integers(0, G, G), minlength=G) for _ in range(B)])
        a, b = stat(np.tensordot(w, c, axes=(1, 0))); reps_se.append(a); reps_sp.append(b)
    reps_se, reps_sp = np.array(reps_se), np.array(reps_sp)  # (S, B, C)
    pick = rng.integers(0, len(per), (B, len(per)))
    bse = np.nanmean(reps_se[pick, np.arange(B)[:, None]], 1); bsp = np.nanmean(reps_sp[pick, np.arange(B)[:, None]], 1)
    n_cls = per[0].sum((0, 2))
    ci = lambda x: (np.nanpercentile(x, 2.5, 0), np.nanpercentile(x, 97.5, 0))
    return dict(sens=se, spec=sp, sens_ci=ci(bse), spec_ci=ci(bsp), n=n_cls)
import warnings; warnings.filterwarnings("ignore", category=RuntimeWarning)
PC = [("Local", fl.get(("S1", "local"))), ("FedAvg", fl[("S1", "fedavg")]), ("FedBN", fl[("S1", "fedbn")]),
      ("AP-FedBN", fl[("S1", "otofl")]), ("FedBN + FT", fl[("S1", "fedbn_ft")]),
      ("FedPer", FP.get(("S1", "fedper"))) if len(FP.get(("S1", "fedper"), [])) > 3 else ("FedPer", None), ("Local, 120 ep.", cm.get(("S1", "local_E3"))), ("FedBN, E3", cm.get(("S1", "fedbn_E3"))), ("FedPer, E3", cm.get(("S1", "fedper_E3"))), ("Centralised", cen["fenet5"])]
rows = []; summary["perclass"] = {}
for site, sname in [(0, "Site 1"), (1, "Site 2")]:
    for nm, v in PC:
        if not v:
            continue
        pc = perclass(v, site)
        summary["perclass"][f"{sname}|{nm}"] = {k_: np.asarray(x).tolist() for k_, x in pc.items() if k_ not in ("sens_ci", "spec_ci")}
        summary["perclass"][f"{sname}|{nm}"]["sens_ci"] = [np.asarray(x).tolist() for x in pc["sens_ci"]]
        cells = []
        for c in range(5):
            if pc["n"][c] == 0:
                cells.append("--"); continue
            cells.append(f"{100*pc['sens'][c]:.0f} [{100*pc['sens_ci'][0][c]:.0f}, {100*pc['sens_ci'][1][c]:.0f}]")
        rows.append(f"{sname} & {nm} & " + " & ".join(cells) + " \\\\")
    ncls = perclass(PC[1][1], site)["n"]  # counts for one seed (per[0])
    summary["perclass"][f"{sname}|n_per_seed"] = ncls.tolist()
    rows.append("\\midrule")
open(os.path.join(TAB, "perclass.tex"), "w").write("\n".join(rows[:-1]))

json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=1, default=float)

# ================================================================ FIGURES

def savefig(fig, name):
    fig.savefig(os.path.join(FIG, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG, name + ".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)

# --- convergence curves (S1, S2): mean +- sd across seeds
fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.2), sharey=True)
for ax, (st, title) in zip(axes, [("S1", "S1: two hospitals"), ("S2", "S2: 10 clients, Dir(0.3)"), ("S3", "S3: 10 clients, IID")]):
    for m in M:
        v = fl[(st, m)]
        if not v:
            continue
        C = np.array([np.array(r["curve"])[:, 1] for r in v]); x = np.array(v[0]["curve"])[:, 0]
        mu, sd = C.mean(0), C.std(0)
        ax.plot(x, 100 * mu, color=COL[m], lw=1.6 if m == "fedbn" else 1.1, ls="--" if m == "otofl" else "-", label=NAME[m], zorder=3 if m == "fedbn" else 2)
        ax.fill_between(x, 100 * (mu - sd), 100 * (mu + sd), color=COL[m], alpha=0.12, lw=0)
    ax.set_title(title, loc="left", color=INK); ax.set_xlabel("Communication round"); ax.grid(axis="y")
axes[0].set_ylabel("Test accuracy (%)")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.1))
savefig(fig, "convergence")

# --- confusion matrices: centralised Fenet5 vs OtoFL S1 vs FedAvg S2 vs OtoFL S2 (summed over seeds)
panels = [("Centralised", cen["fenet5"]), ("FedAvg, S1", fl[("S1", "fedavg")]), ("FedBN, S1", fl[("S1", "fedbn")])]
panels = [(t, v) for t, v in panels if v]
if panels:
    fig, axes = plt.subplots(1, len(panels), figsize=(2.35 * len(panels), 2.3))
    axes = np.atleast_1d(axes)
    for ax, (t, v) in zip(axes, panels):
        cm = sum(metrics(r)["cm"] for r in v).astype(float)
        cmn = cm / cm.sum(1, keepdims=True)
        ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        for i in range(len(CLS)):
            for j in range(len(CLS)):
                ax.text(j, i, f"{100*cmn[i, j]:.0f}", ha="center", va="center", fontsize=6.5,
                        color="white" if cmn[i, j] > 0.55 else INK)
        ax.set_xticks(range(len(CLS)), CLS, rotation=40, ha="right"); ax.set_yticks(range(len(CLS)), CLS)
        ax.set_title(t, loc="left"); ax.set_xlabel("Predicted")
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.tick_params(length=0)
    axes[0].set_ylabel("True")
    fig.tight_layout(w_pad=0.6)
    savefig(fig, "confusion")

# --- privacy-utility
if dp:
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    E = [e for e in [1, 2, 4, 8] if e in dp]
    for key, lab, c in [("f1", "Macro-F1", "#2a78d6"), ("bacc", "Balanced acc.", "#eb6834")]:
        mu = [100 * agg(dp[e], key)[0] for e in E]; sd = [100 * agg(dp[e], key)[1] for e in E]
        ax.errorbar(E, mu, yerr=sd, color=c, marker="o", ms=4, lw=1.4, capsize=2, label=lab)
        if 0 in dp:
            ax.axhline(100 * agg(dp[0], key)[0], color=c, ls="--", lw=0.9)
    ax.set_xscale("log", base=2); ax.set_xticks(E, [str(e) for e in E])
    ax.set_xlabel("Privacy budget $\\varepsilon$ ($\\delta=10^{-4}$)"); ax.set_ylabel("Score (%)")
    ax.grid(axis="y"); ax.legend(loc="lower right")
    ax.text(E[0], 100 * agg(dp[0], "f1")[0] + 0.8, "dashed: no DP", color=INK2, fontsize=6.5) if 0 in dp else None
    savefig(fig, "privacy_utility")

# --- scalability
Ks = [5, 10, 20]
have = {(K, m): (group(f"scale/K{K}", m) if K != 10 else fl[("S2", m)]) for K in Ks for m in ["fedavg", "otofl"]}
if all(have[(K, m)] for K in Ks for m in ["fedavg", "otofl"]):
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    w = 0.36
    for i, m in enumerate(["fedavg", "otofl"]):
        mu = [100 * agg(have[(K, m)], "f1")[0] for K in Ks]; sd = [100 * agg(have[(K, m)], "f1")[1] for K in Ks]
        ax.bar(np.arange(3) + (i - 0.5) * (w + 0.02), mu, w, yerr=sd, color=COL[m], label=NAME[m],
               error_kw=dict(lw=0.8, capsize=2, ecolor=INK2))
    ax.set_xticks(range(3), [f"K = {K}" for K in Ks]); ax.set_ylabel("Macro-F1 (%)")
    lo = min(100 * agg(have[(K, m)], "f1")[0] for K in Ks for m in ["fedavg", "otofl"])
    ax.set_ylim(max(0, lo - 15), 100); ax.grid(axis="y"); ax.legend(loc="upper right", ncol=2)
    savefig(fig, "scalability")

# --- client heterogeneity map (S2, seed 0)
v = fl[("S2", "otofl")]
if v:
    r = [x for x in v if x["seed"] == 0][0]
    pc = r["per_client"]
    fig, ax = plt.subplots(figsize=(3.5, 2.2))
    Hs = [c["H"] for c in pc]; ns = [c["n_train"] for c in pc]
    ax.bar(range(len(pc)), Hs, color=["#2a78d6" if c["site"] == 0 else "#eb6834" for c in pc], width=0.7)
    for i, c in enumerate(pc):
        ax.text(i, c["H"] + 0.01, f"{c['n_train']}", ha="center", fontsize=6, color=INK2)
    ax.set_xticks(range(len(pc)), [f"C{i+1}" for i in range(len(pc))])
    ax.set_ylabel("Label skew $H_k$ (TV distance)"); ax.grid(axis="y")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#2a78d6", label="Chile clients"), Patch(color="#eb6834", label="Turkey clients")], loc="upper center", ncol=2, frameon=False); ax.set_ylim(0, 0.92)
    savefig(fig, "heterogeneity")

# --- per-site accuracy: local, federated, centralised (S1)
bars = [("Local (Chile only)", group("local", "site0"), "#6250d6"), ("Local (Turkey only)", group("local", "site1"), "#9e93e8"),
        ("FedAvg", fl[("S1", "fedavg")], COL["fedavg"]), ("FedProx", fl[("S1", "fedprox")], COL["fedprox"]),
        ("FedBN", fl[("S1", "fedbn")], COL["fedbn"]), ("AP-FedBN", fl[("S1", "otofl")], COL["otofl"]),
        ("Centralised", cen["fenet5"], COL["central"])]
bars = [b for b in bars if b[1]]
if bars:
    fig, ax = plt.subplots(figsize=(7.1, 2.3))
    w = 0.8 / len(bars)
    for i, (lab, v, c) in enumerate(bars):
        mu = [100 * agg(v, k)[0] for k in ("acc_chile", "acc_turkey")]; sd = [100 * agg(v, k)[1] for k in ("acc_chile", "acc_turkey")]
        ax.bar(np.arange(2) + (i - (len(bars) - 1) / 2) * w, mu, w * 0.92, yerr=sd, color=c, label=lab,
               error_kw=dict(lw=0.7, capsize=1.5, ecolor=INK2))
    ax.set_xticks(range(2), ["Site 1 test images (Chile)", "Site 2 test images (Turkey)"]); ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 105); ax.grid(axis="y"); ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.28))
    savefig(fig, "persite")

print(json.dumps({k: v for k, v in summary.items() if k != "significance"}, indent=0, default=lambda x: round(float(x), 4))[:6000])
print("significance", json.dumps(sig, default=float, indent=0))
