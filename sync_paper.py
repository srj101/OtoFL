"""Copy analysis tables/figures into the paper and build the significance table rows."""
import json, shutil, sys, os
S, P = sys.argv[1], sys.argv[2]
A = sys.argv[3] if len(sys.argv) > 3 else "ana"
for n in ["central", "fl", "ablation", "dp", "tuned", "perclass", "cm"]:
    shutil.copy(f"{S}/{A}/tables/{n}.tex", f"{P}/tables/{n}_rows.tex")
for d in [A, "post"]:
    for f in os.listdir(f"{S}/{d}/figures"):
        if f.endswith(".pdf") or f.endswith(".png"):
            shutil.copy(f"{S}/{d}/figures/{f}", f"{P}/figures/{f}")
NM = {"fedavg": "FedAvg", "fedprox": "FedProx", "scaffold": "SCAFFOLD", "fedbn": "FedBN", "otofl": "AP-FedBN", "local": "local only", "fedbn_ft": "FedBN+FT", "fedper": "FedPer"}
sig = json.load(open(f"{S}/{A}/summary.json"))["significance"]
order = [("fedbn", "fedavg"), ("fedbn", "fedprox"), ("fedbn", "scaffold"), ("fedprox", "fedavg"), ("otofl", "fedbn"),
         ("fedbn_ft", "fedbn"), ("fedper", "fedbn"), ("fedbn", "local"), ("fedper", "local"), ("fedbn_ft", "local"), ("otofl", "local"), ("fedavg", "local")]
sel = []
for st in ["S1", "S2", "S3"]:
    for a, v in order:
        sel += [s for s in sig if s["setting"] == st and s["a"] == a and s["vs"] == v]
# Holm-Bonferroni adjustment in two families: the six pre-specified primary comparisons
# (FedBN vs FedAvg and AP-FedBN vs FedBN in each scenario) and the remaining exploratory comparisons
PRIMARY = {("fedbn", "fedavg"), ("otofl", "fedbn")}
def holm(ps):
    m = len(ps); rank = sorted(range(m), key=lambda i: ps[i]); adj = [0.0] * m; run = 0.0
    for k, i in enumerate(rank):
        run = max(run, min(1.0, (m - k) * ps[i])); adj[i] = run
    return adj
prim = [i for i, s in enumerate(sel) if (s["a"], s["vs"]) in PRIMARY]
expl = [i for i in range(len(sel)) if i not in prim]
adj = [0.0] * len(sel)
for fam in (prim, expl):
    for i, a in zip(fam, holm([sel[i]["p"] for i in fam])):
        adj[i] = a
summ = {f'{sel[i]["setting"]}:{sel[i]["a"]}-{sel[i]["vs"]}': round(adj[i], 4) for i in range(len(sel))}
json.dump(summ, open(f"{S}/{A}/holm.json", "w"), indent=1)
fmt = lambda x: "$<$0.001" if x < 0.001 else f"{x:.3f}" if x < 0.01 else f"{x:.2f}"
fmtH = lambda x: "$<$0.01" if x < 0.01 else f"{x:.2f}"  # bootstrap resolution 1/B=1e-4 limits the smallest adjusted p
rows, last = [], None
for s, ph in zip(sel, adj):
    st, a, v = s["setting"], s["a"], s["vs"]
    pre = ("\\midrule " if last and last != st else "") + (st if last != st else "")
    rows.append(f"{pre} & {NM[a]} vs {NM[v]}{'$^\\dagger$' if (a, v) in PRIMARY else ''} & {100*s['diff']:+.1f} & [{100*s['lo']:.1f}, {100*s['hi']:.1f}] & {fmt(s['p'])} & {fmtH(ph)} \\\\")
    last = st
open(f"{P}/tables/sig_rows.tex", "w").write("\n".join(rows) + "\n")
print(len(rows), "sig rows")
