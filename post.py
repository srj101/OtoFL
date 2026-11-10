"""Post-hoc analyses on saved seed-0 models: dataset figure, corruption robustness, Grad-CAM.
Usage: python post.py WORKDIR OUTDIR
"""
import sys, os, json, copy
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(__file__))
import otofl as O
from sklearn.metrics import f1_score
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
W, OUT = sys.argv[1], sys.argv[2]
FIG = os.path.join(OUT, "figures"); os.makedirs(FIG, exist_ok=True)
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 0.6, "grid.color": GRID, "grid.linewidth": 0.5, "pdf.fonttype": 42,
                     "legend.frameon": False, "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
X, y, site, grp, classes = O.load(os.path.join(W, "cache"))
CLS = ["Normal", "Earwax", "AOM", "COM", "Myringosclerosis"]
tr, te = O.split(y, site, grp, 0)
summary = {}

def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".pdf"), bbox_inches="tight"); fig.savefig(os.path.join(FIG, name + ".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)

# ---------------------------------------------------------------- dataset examples
rng = np.random.default_rng(3)
fig, axes = plt.subplots(2, 5, figsize=(7.1, 3.1))
for si, sname in enumerate(["Chile", "Turkey"]):
    for c in range(5):
        ax = axes[si, c]; ax.axis("off")
        ids = np.where((site == si) & (y == c))[0]
        if len(ids):
            ax.imshow(X[rng.choice(ids)].permute(1, 2, 0).numpy())
            ax.set_title(f"{CLS[c]} (n={len(ids)})", fontsize=7, color=INK)
        else:
            ax.text(0.5, 0.5, "not present", ha="center", va="center", color=INK2, fontsize=7, transform=ax.transAxes)
            ax.set_title(f"{CLS[c]} (n=0)", fontsize=7, color=INK2)
    axes[si, 0].text(-0.12, 0.5, f"Site {si+1}: {sname}", rotation=90, va="center", ha="center", transform=axes[si, 0].transAxes, fontsize=8)
fig.tight_layout(pad=0.3)
save(fig, "dataset")

# ---------------------------------------------------------------- load models
def load_central(name):
    m = O.build(name, 5).to(O.DEV); m.load_state_dict(torch.load(os.path.join(W, f"models/central_{name}_0.pt"), map_location=O.DEV)); m.eval(); return m

def otofl_site_models(tag):
    ck = torch.load(os.path.join(W, f"models/otofl_{tag}_0.pt"), map_location=O.DEV, weights_only=False)
    ms = []
    for k in range(len(ck["norms"])):
        m = O.build("fenet5", 5).to(O.DEV); sd = copy.deepcopy(ck["global_"]); sd.update(ck["norms"][k]); m.load_state_dict(sd); m.eval(); ms.append(m)
    return ms, ck["csite"]

models = {}
for nm in ["fenet5", "mobilenet_v3_large", "efficientnet_b0"]:
    if os.path.exists(os.path.join(W, f"models/central_{nm}_0.pt")):
        models["Central " + {"fenet5": "Fenet5", "mobilenet_v3_large": "MobileNetV3-L", "efficientnet_b0": "EfficientNet-B0"}[nm]] = nm
have_otofl = os.path.exists(os.path.join(W, "models/otofl_S1_0.pt"))

@torch.no_grad()
def predict_with(fn_model, xb_u8, sites, corrupt):
    out = []
    for i in range(0, len(xb_u8), 64):
        x = xb_u8[i:i + 64].to(O.DEV).float() / 255
        x = F.interpolate(x, size=(O.RES, O.RES), mode="bilinear", antialias=True, align_corners=False)
        x = corrupt(x).clamp(0, 1)
        x = (x - O.MEAN.to(O.DEV)) / O.STD.to(O.DEV)
        m = fn_model(sites[i:i + 64])
        if isinstance(m, (list, dict)):
            p = torch.zeros(len(x), 5, device=O.DEV)
            for s in np.unique(sites[i:i + 64]):
                k = torch.as_tensor(sites[i:i + 64] == s, device=O.DEV)
                p[k] = F.softmax(m[int(s)](x[k]), 1)
        else:
            p = F.softmax(m(x), 1)
        out.append(p.cpu())
    return torch.cat(out).numpy()

def gblur(x, s):
    k = int(2 * round(3 * s) + 1); t = torch.arange(k, device=x.device) - k // 2
    g = torch.exp(-t.float() ** 2 / (2 * s * s)); g = g / g.sum()
    x = F.conv2d(F.pad(x, (k // 2, k // 2, 0, 0), mode="reflect"), g.view(1, 1, 1, k).repeat(3, 1, 1, 1), groups=3)
    return F.conv2d(F.pad(x, (0, 0, k // 2, k // 2), mode="reflect"), g.view(1, 1, k, 1).repeat(3, 1, 1, 1), groups=3)

gen = torch.Generator(device=O.DEV).manual_seed(0)
CORR = {
    "Gaussian noise": [lambda x, s=s: x + s * torch.randn(x.shape, device=x.device, generator=gen) for s in (0.04, 0.08, 0.12, 0.16)],
    "Defocus blur": [lambda x, s=s: gblur(x, s) for s in (1.0, 2.0, 3.0, 4.0)],
    "Low brightness": [lambda x, s=s: x * s for s in (0.8, 0.6, 0.45, 0.3)],
    "Low contrast": [lambda x, s=s: (x - x.mean((1, 2, 3), keepdim=True)) * s + x.mean((1, 2, 3), keepdim=True) for s in (0.7, 0.5, 0.35, 0.2)],
}
evals = dict(models)
if have_otofl:
    ms, csite = otofl_site_models("S1")
    site_to_model = {int(csite[k]): ms[k] for k in range(len(ms))}
    evals["AP-FedBN (S1)"] = "otofl"
COLR = {"AP-FedBN (S1)": "#2a78d6", "Central Fenet5": "#008300", "Central MobileNetV3-L": "#eb6834", "Central EfficientNet-B0": "#1baf7a"}
rob = {}
for lab, nm in evals.items():
    if nm == "otofl":
        fn = lambda s: site_to_model
    else:
        mm = load_central(nm); fn = lambda s, mm=mm: mm
    clean = predict_with(fn, X[te], site[te], lambda x: x)
    rob[lab] = {"clean": f1_score(y[te], clean.argmax(1), average="macro")}
    for cn, fs in CORR.items():
        rob[lab][cn] = [f1_score(y[te], predict_with(fn, X[te], site[te], f).argmax(1), average="macro") for f in fs]
summary["robustness"] = rob
fig, axes = plt.subplots(1, 4, figsize=(7.1, 2.0), sharey=True)
for ax, cn in zip(axes, CORR):
    for lab in rob:
        ax.plot(range(5), [100 * rob[lab]["clean"]] + [100 * v for v in rob[lab][cn]], marker="o", ms=3, lw=1.6 if "AP-FedBN" in lab else 1.1,
                color=COLR.get(lab, INK2), label=lab)
    ax.set_title(cn, loc="left"); ax.set_xticks(range(5), ["0", "1", "2", "3", "4"]); ax.set_xlabel("Severity"); ax.grid(axis="y")
axes[0].set_ylabel("Macro-F1 (%)")
h, l = axes[0].get_legend_handles_labels(); fig.legend(h, l, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.12))
save(fig, "robustness")

# ---------------------------------------------------------------- Grad-CAM on AP-FedBN (S1) site models
if have_otofl:
    feats = {}
    def gradcam(m, x, cls):
        blk = m.blocks[-1]
        h1 = blk.a3.register_forward_hook(lambda mod, i, o: feats.__setitem__("a", o))
        out = m(x); h1.remove()
        a = feats["a"]; g = torch.autograd.grad(out[0, cls], a)[0]
        cam = F.relu((g.mean((2, 3), keepdim=True) * a).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        return (cam / (cam.max() + 1e-8)).detach().cpu().numpy(), F.softmax(out, 1)[0].detach().cpu().numpy()
    fig, axes = plt.subplots(2, 5, figsize=(7.1, 3.2))
    rng = np.random.default_rng(1)
    for c in range(5):
        ids = [i for i in te if y[i] == c]
        # prefer a correctly classified example
        pick = None
        for i in rng.permutation(ids):
            m = site_to_model[int(site[i])]
            x = X[i:i + 1].to(O.DEV).float() / 255
            x = F.interpolate(x, size=(O.RES, O.RES), mode="bilinear", antialias=True, align_corners=False)
            xn = ((x - O.MEAN.to(O.DEV)) / O.STD.to(O.DEV)).requires_grad_(True)
            cam, p = gradcam(m, xn, c)
            if p.argmax() == c:
                pick = (x[0].permute(1, 2, 0).cpu().numpy(), cam, p[c], int(site[i])); break
        ax0, ax1 = axes[0, c], axes[1, c]
        ax0.axis("off"); ax1.axis("off")
        if pick:
            img, cam, pc, s = pick
            ax0.imshow(img); ax0.set_title(CLS[c], fontsize=7.5)
            ax1.imshow(img); ax1.imshow(cam, cmap="inferno", alpha=0.45)
            ax1.set_title(f"p = {pc:.2f} (site {s+1})", fontsize=6.5, color=INK2)
    fig.tight_layout(pad=0.3)
    save(fig, "gradcam")

json.dump(summary, open(os.path.join(OUT, "post_summary.json"), "w"), indent=1, default=float)
print(json.dumps(summary, indent=1, default=lambda v: round(float(v), 4)))
