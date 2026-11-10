"""OtoFL experiments: centralized, local, and federated training on public otoscopy data.

Usage: python otofl.py --cache DIR --out FILE.json --mode {central,fl} [options]
"""
import argparse, json, math, os, time, copy
import numpy as np
import warnings; warnings.filterwarnings("ignore")
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.model_selection import StratifiedGroupKFold

DEV = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

# ----------------------------------------------------------------- data

def load(cache):
    idx = json.load(open(os.path.join(cache, "index.json")))
    X = torch.load(os.path.join(cache, "images.pt"))
    recs = idx["records"]
    y = np.array([r["label"] for r in recs])
    site = np.array([0 if r["site"] == "chile" else 1 for r in recs])
    grp = np.array([r["group"] for r in recs])
    return X, y, site, grp, idx["classes"]


def split(y, site, grp, seed, test_frac=0.2):
    """Group-aware, stratified train/test split done separately per site."""
    tr, te = [], []
    for s in np.unique(site):
        ids = np.where(site == s)[0]
        k = int(round(1 / test_frac))
        sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
        a, b = next(iter(sgkf.split(ids, y[ids], grp[ids])))
        tr += list(ids[a]); te += list(ids[b])
    tr, te = np.array(tr), np.array(te)
    assert not (set(grp[tr]) & set(grp[te])), "group leakage"
    return tr, te


def partition(tr, te, y, site, n_clients, alpha, seed, n_cls):
    """Assign train/test images to clients. Each client belongs to one site.
    Within a site, class c is split across that site's clients with Dirichlet(alpha)
    shares (alpha=None -> IID). Train and test use the same shares, so every
    client's test set follows its own label distribution."""
    rng = np.random.default_rng(seed)
    n_site = [int((site[tr] == s).sum()) for s in (0, 1)]
    if n_clients == 2:
        per_site = [1, 1]
    else:
        k0 = int(round(n_clients * n_site[0] / sum(n_site)))
        k0 = min(max(k0, 1), n_clients - 1)
        per_site = [k0, n_clients - k0]
    clients_tr = [[] for _ in range(n_clients)]
    clients_te = [[] for _ in range(n_clients)]
    client_site = []
    off = 0
    for s in (0, 1):
        ks = list(range(off, off + per_site[s])); off += per_site[s]
        client_site += [s] * len(ks)
        for c in range(n_cls):
            if alpha is None:
                q = np.ones(len(ks)) / len(ks)
            else:
                q = rng.dirichlet(alpha * np.ones(len(ks)))
            for pool, out in ((tr, clients_tr), (te, clients_te)):
                ids = pool[(site[pool] == s) & (y[pool] == c)].copy()
                rng.shuffle(ids)
                cuts = (np.cumsum(q)[:-1] * len(ids)).astype(int)
                for k, part in zip(ks, np.split(ids, cuts)):
                    out[k] += list(part)
    # guarantee every client has some training data
    for k in range(n_clients):
        if len(clients_tr[k]) < 8:
            donor = max(range(n_clients), key=lambda j: len(clients_tr[j]) if client_site[j] == client_site[k] else -1)
            clients_tr[k] += clients_tr[donor][:8]; clients_tr[donor] = clients_tr[donor][8:]
    return [np.array(c, dtype=int) for c in clients_tr], [np.array(c, dtype=int) for c in clients_te], client_site


def augment(xb, gen):
    """GPU augmentation: flip, rotation/scale, brightness/contrast jitter. xb in [0,1]."""
    B = xb.shape[0]
    flip = (torch.rand(B, device=xb.device, generator=gen) < 0.5).float().view(B, 1, 1, 1)
    xb = flip * xb.flip(3) + (1 - flip) * xb
    ang = (torch.rand(B, device=xb.device, generator=gen) * 2 - 1) * math.radians(20)
    sc = 1 / (0.85 + 0.3 * torch.rand(B, device=xb.device, generator=gen))
    cos, sin = torch.cos(ang) * sc, torch.sin(ang) * sc
    theta = torch.stack([torch.stack([cos, -sin, torch.zeros_like(cos)], 1),
                         torch.stack([sin, cos, torch.zeros_like(cos)], 1)], 1)
    grid = F.affine_grid(theta, xb.shape, align_corners=False)
    xb = F.grid_sample(xb, grid, padding_mode="reflection", align_corners=False)
    br = (1 + (torch.rand(B, 1, 1, 1, device=xb.device, generator=gen) * 2 - 1) * 0.2)
    ct = (1 + (torch.rand(B, 1, 1, 1, device=xb.device, generator=gen) * 2 - 1) * 0.2)
    m = xb.mean(dim=(1, 2, 3), keepdim=True)
    xb = ((xb - m) * ct + m) * br
    return xb.clamp(0, 1)


RES = 160

def to_input(xb_uint8, train, gen=None):
    x = xb_uint8.to(DEV, non_blocking=True).float() / 255.0
    if x.shape[-1] != RES:
        x = F.interpolate(x, size=(RES, RES), mode="bilinear", antialias=True, align_corners=False)
    if train:
        x = augment(x, gen)
    return (x - MEAN.to(DEV)) / STD.to(DEV)

# ----------------------------------------------------------------- models

class FenetBlock(nn.Module):
    def __init__(self, cin, cout, norm):
        super().__init__()
        N = (lambda c: nn.BatchNorm2d(c)) if norm == "bn" else (lambda c: nn.GroupNorm(min(8, c // 4), c))
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1, bias=False); self.n1 = N(cout); self.a1 = nn.PReLU(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1, bias=False); self.n2 = N(cout); self.a2 = nn.PReLU(cout)
        self.fuse = nn.Conv2d(cin + cout, cout, 1, bias=False); self.n3 = N(cout); self.a3 = nn.PReLU(cout)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        h = self.a1(self.n1(self.c1(x)))
        h = self.a2(self.n2(self.c2(h)))
        h = self.a3(self.n3(self.fuse(torch.cat([x, h], 1))))
        return self.pool(h)


class Fenet5(nn.Module):
    """Five Fenet blocks (24-48-72-96-120 filters), GAP, dropout, linear head."""
    def __init__(self, n_cls, norm="bn", width=24, drop=0.2):
        super().__init__()
        ch = [3] + [width * (i + 1) for i in range(5)]
        self.blocks = nn.Sequential(*[FenetBlock(ch[i], ch[i + 1], norm) for i in range(5)])
        self.drop = nn.Dropout(drop)
        self.head = nn.Linear(ch[-1], n_cls)

    def forward(self, x):
        h = self.blocks(x)
        return self.head(self.drop(h.mean((2, 3))))


class FrozenMNV3(nn.Module):
    """ImageNet MobileNetV3-L; backbone frozen (BN in eval mode, ImageNet statistics),
    only the classifier MLP is trained. Safe for DP-SGD (no batch-dependent layers)."""
    def __init__(self, n_cls):
        super().__init__()
        import torchvision.models as tv
        m = tv.mobilenet_v3_large(weights="IMAGENET1K_V2")
        self.features, self.pool = m.features, m.avgpool
        for p in self.features.parameters():
            p.requires_grad_(False)
        self.head = nn.Sequential(nn.Linear(960, 256), nn.Hardswish(), nn.Dropout(0.2), nn.Linear(256, n_cls))

    def train(self, mode=True):
        super().train(mode)
        self.features.eval()
        return self

    def forward(self, x):
        return self.head(torch.flatten(self.pool(self.features(x)), 1))


def build(name, n_cls):
    import torchvision.models as tv
    if name == "fenet5":
        return Fenet5(n_cls, "bn")
    if name == "fenet5_gn":
        return Fenet5(n_cls, "gn")
    if name == "resnet50":
        m = tv.resnet50(weights="IMAGENET1K_V2"); m.fc = nn.Linear(m.fc.in_features, n_cls)
    elif name == "densenet121":
        m = tv.densenet121(weights="IMAGENET1K_V1"); m.classifier = nn.Linear(m.classifier.in_features, n_cls)
    elif name == "efficientnet_b0":
        m = tv.efficientnet_b0(weights="IMAGENET1K_V1"); m.classifier[1] = nn.Linear(m.classifier[1].in_features, n_cls)
    elif name == "mobilenet_v3_large":
        m = tv.mobilenet_v3_large(weights="IMAGENET1K_V2"); m.classifier[3] = nn.Linear(m.classifier[3].in_features, n_cls)
    elif name == "mnv3_frozen":
        m = FrozenMNV3(n_cls)
    elif name == "convnext_tiny":
        m = tv.convnext_tiny(weights="IMAGENET1K_V1"); m.classifier[2] = nn.Linear(m.classifier[2].in_features, n_cls)
    else:
        raise ValueError(name)
    return m


def is_norm_key(model):
    """Names of parameters/buffers that belong to normalisation layers."""
    keys = set()
    for mn, mod in model.named_modules():
        if isinstance(mod, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)) or type(mod).__name__ == "LayerNorm2d":
            for pn, _ in list(mod.named_parameters(recurse=False)) + list(mod.named_buffers(recurse=False)):
                keys.add(f"{mn}.{pn}" if mn else pn)
    return keys

# ----------------------------------------------------------------- train / eval

SCAFFOLD_MOM = False  # set by --scaffold_mom: give SCAFFOLD the same momentum/weight decay as the other methods

def run_epochs(model, X, y, ids, epochs, lr, bs, gen, prox=None, scaffold=None, wd=5e-4, prox_mask=None):
    """Local SGD (momentum 0.9). prox=(mu, global_params); scaffold=(c_global, c_local).
    prox_mask: per-parameter booleans; False = no proximal pull (parameters kept local)."""
    model.train()
    plain = scaffold is not None and not SCAFFOLD_MOM
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.0 if plain else 0.9, weight_decay=0.0 if plain else wd)  # original SCAFFOLD assumes plain SGD
    yt = torch.as_tensor(y)
    steps = 0
    for _ in range(epochs):
        perm = ids[torch.randperm(len(ids), generator=torch.Generator().manual_seed(int(torch.randint(0, 2**31, (1,), generator=gen_cpu(gen)).item()))).numpy()]
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]
            if len(b) < 2:
                continue
            xb = to_input(X[b], True, gen); yb = yt[b].to(DEV)
            loss = F.cross_entropy(model(xb), yb)
            opt.zero_grad(set_to_none=True); loss.backward()
            if prox is not None:
                mu, gp = prox
                for i, (p, g) in enumerate(zip(model.parameters(), gp)):
                    if p.grad is not None and (prox_mask is None or prox_mask[i]):
                        p.grad.add_(p.detach() - g, alpha=mu)
            if scaffold is not None:
                cg, cl = scaffold
                for p, a, b2 in zip(model.parameters(), cg, cl):
                    if p.grad is not None:
                        p.grad.add_(a - b2)
            opt.step(); steps += 1
    return steps


_GEN_CPU = {}
def gen_cpu(gen):
    if id(gen) not in _GEN_CPU:
        _GEN_CPU[id(gen)] = torch.Generator().manual_seed(int(gen.initial_seed()))
    return _GEN_CPU[id(gen)]


@torch.no_grad()
def predict(model, X, ids, bs=128):
    model.eval()
    out = []
    for i in range(0, len(ids), bs):
        out.append(F.softmax(model(to_input(X[ids[i:i + bs]], False)), 1).float().cpu())
    return torch.cat(out).numpy() if out else np.zeros((0, 1))

# ----------------------------------------------------------------- DP-SGD (per-sample clipping, Poisson sampling)

def dp_epochs(model, X, y, ids, steps, lr, q, clip, sigma, gen, prox=None):
    from torch.func import functional_call, vmap, grad
    model.train()
    params = {k: v.detach() for k, v in model.named_parameters() if v.requires_grad}
    buffers = {k: v.detach() for k, v in model.named_buffers()}
    buffers.update({k: v.detach() for k, v in model.named_parameters() if not v.requires_grad})
    yt = torch.as_tensor(y)

    def loss_fn(p, b, xi, yi):
        out = functional_call(model, (p, b), (xi.unsqueeze(0),))
        return F.cross_entropy(out, yi.unsqueeze(0))
    psg = vmap(grad(loss_fn), in_dims=(None, None, 0, 0), randomness="different")
    mom = {k: torch.zeros_like(v) for k, v in params.items()}
    rng = np.random.default_rng(int(gen.initial_seed()) + 7)
    expected = q * len(ids)
    for _ in range(steps):
        b = ids[rng.random(len(ids)) < q]
        acc = {k: torch.zeros_like(v) for k, v in params.items()}
        for i in range(0, len(b), 32):  # micro-batches for memory
            bb = b[i:i + 32]
            xb = to_input(X[bb], True, gen); yb = yt[bb].to(DEV)
            g = psg(params, buffers, xb, yb)
            norms = torch.sqrt(sum((gi.reshape(len(bb), -1) ** 2).sum(1) for gi in g.values()))
            scale = (clip / (norms + 1e-6)).clamp(max=1.0)
            for k in acc:
                acc[k] += (g[k] * scale.view(-1, *[1] * (g[k].dim() - 1))).sum(0)
        new = {}
        for k, v in params.items():
            noisy = (acc[k] + torch.randn_like(v) * sigma * clip) / expected
            if prox is not None:
                mu, gp = prox
                noisy = noisy + mu * (v - gp[k])
            noisy = noisy + 5e-4 * v
            mom[k] = 0.9 * mom[k] + noisy
            new[k] = v - lr * mom[k]
        params = new
    with torch.no_grad():
        for k, p in model.named_parameters():
            if k in params:
                p.copy_(params[k])

# ----------------------------------------------------------------- experiments

def leak_split(args, y, site):
    """Site-1 (Chile) splits that ignore duplicate groups, for the leakage experiment.
    official: the dataset's own Training/Testing folders; random: stratified image-level 80/20 split."""
    from sklearn.model_selection import StratifiedKFold
    paths = [r["path"] for r in json.load(open(os.path.join(args.cache, "index.json")))["records"]]
    ids = np.where(site == 0)[0]
    if args.split == "official":
        tr = np.array([i for i in ids if "/Training-validation/" in paths[i]]); te = np.array([i for i in ids if "/Testing/" in paths[i]])
    else:
        a, b = next(iter(StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed).split(ids, y[ids])))
        tr, te = ids[a], ids[b]
    assert len(tr) and len(te) and not set(tr) & set(te)
    return tr, te


def central(args, X, y, site, grp, classes):
    tr, te = split(y, site, grp, args.seed) if args.split == "group" else leak_split(args, y, site)
    if args.train_site is not None:
        tr = tr[site[tr] == args.train_site]
    torch.manual_seed(args.seed)
    gen = torch.Generator(device=DEV).manual_seed(args.seed)
    model = build(args.model, len(classes)).to(DEV)
    curve = []
    opt_lr = args.lr
    for ep in range(args.epochs):
        lr = 0.5 * opt_lr * (1 + math.cos(math.pi * ep / args.epochs))
        run_epochs(model, X, y, tr, 1, lr, args.bs, gen)
        if (ep + 1) % args.eval_every == 0 or ep == args.epochs - 1:
            p = predict(model, X, te)
            curve.append([ep + 1, float((p.argmax(1) == y[te]).mean())])
            print(f"ep {ep+1} acc {curve[-1][1]:.4f}", flush=True)
    p = predict(model, X, te)
    nparam = sum(t.numel() for t in model.parameters())
    if args.save_model:
        torch.save(model.state_dict(), args.save_model)
    return dict(test_ids=te.tolist(), probs=p.tolist(), labels=y[te].tolist(), sites=site[te].tolist(),
                curve=curve, n_params=nparam, n_train=len(tr))


def label_hist(y, ids, n_cls):
    h = np.bincount(y[ids], minlength=n_cls).astype(float)
    return h / max(h.sum(), 1)


def fl(args, X, y, site, grp, classes):
    n_cls = len(classes)
    tr, te = split(y, site, grp, args.seed)
    alpha = None if args.alpha <= 0 else args.alpha
    ctr, cte, csite = partition(tr, te, y, site, args.clients, alpha, args.seed, n_cls)
    cval = None
    if getattr(args, "val_frac", 0) > 0:
        # hold out whole duplicate groups from each client's TRAINING data for validation
        rng = np.random.default_rng(args.seed + 777)
        cval, new_tr = [], []
        for c in ctr:
            c = np.array(c); gs = np.unique(grp[c]); rng.shuffle(gs)
            nv = max(1, int(round(args.val_frac * len(gs)))) if len(gs) >= 5 else 0
            vg = set(gs[:nv].tolist())
            m = np.array([g in vg for g in grp[c]], dtype=bool)
            cval.append(c[m]); new_tr.append(c[~m])
        ctr = new_tr
    torch.manual_seed(args.seed)
    gen = torch.Generator(device=DEV).manual_seed(args.seed)
    gmodel = build(args.model, n_cls).to(DEV)
    if args.method == "local":
        # reference: every client trains alone on its own data, same schedule, no communication
        ms = [copy.deepcopy(gmodel) for _ in range(args.clients)]
        for r in range(args.rounds):
            lr = 0.5 * args.lr * (1 + math.cos(math.pi * r / args.rounds))
            for k in range(args.clients):
                run_epochs(ms[k], X, y, ctr[k], args.local_epochs, lr, args.bs, gen)
        probs, order, per_client = [], [], []
        for k in range(args.clients):
            p = predict(ms[k], X, cte[k]) if len(cte[k]) else np.zeros((0, n_cls))
            probs.append(p); order += list(cte[k])
            per_client.append(dict(n_test=len(cte[k]), acc=float((p.argmax(1) == y[cte[k]]).mean()) if len(cte[k]) else None,
                                   site=int(csite[k]), n_train=len(ctr[k]), H=0.0, mu=0.0))
        probs = np.concatenate(probs); order = np.array(order)
        return dict(test_ids=order.tolist(), probs=probs.tolist(), labels=y[order].tolist(), sites=site[order].tolist(),
                    curve=[], per_client=per_client, comm_MB=0.0, sigma=None,
                    n_params=int(sum(p.numel() for p in gmodel.parameters())), shared_params=0)
    names = [k for k, _ in gmodel.named_parameters()]
    norm_keys = is_norm_key(gmodel)
    local_norm = args.method in ("fedbn", "otofl", "otofl_fixedmu", "silobn", "fedper")
    if args.method == "silobn":  # SiloBN: keep only BN running statistics local, share affine parameters
        norm_keys = {k for k in norm_keys if "running_" in k or "num_batches" in k}
    if args.method == "fedper":  # FedPer-style personalisation: local normalisation layers AND local classifier head
        norm_keys = set(norm_keys) | {k for k in gmodel.state_dict() if k.startswith("head.")}
    prox_mask = [not (local_norm and n in norm_keys) for n in names] if getattr(args, "prox_skip_local", False) else None
    adaptive = args.method in ("otofl", "otofl_nobn")
    K = args.clients
    n_k = np.array([len(c) for c in ctr], dtype=float)
    # heterogeneity score: total-variation distance of each client's label histogram
    # from the federation-wide histogram (computed from counts only)
    p_glob = np.bincount(y[tr], minlength=n_cls) / len(tr)
    H = np.array([0.5 * np.abs(label_hist(y, c, n_cls) - p_glob).sum() for c in ctr])
    mu_k = np.full(K, args.mu)
    if adaptive:
        mu_k = args.mu * (1 + args.lam * H)
    elif args.method in ("fedavg", "fedbn", "scaffold", "silobn", "fedper"):
        mu_k[:] = 0.0
    # per-client state for local normalisation layers
    gstate = copy.deepcopy(gmodel.state_dict())
    client_norm = [{k: v.clone() for k, v in gstate.items() if k in norm_keys} for _ in range(K)]
    c_glob = [torch.zeros_like(p) for p in gmodel.parameters()] if args.method == "scaffold" else None
    c_loc = [[torch.zeros_like(p) for p in gmodel.parameters()] for _ in range(K)] if args.method == "scaffold" else None
    sigma = None
    if args.dp_eps > 0:
        from opacus.accountants.utils import get_noise_multiplier
        steps_tot = []
        for k in range(K):
            q = min(1.0, args.bs / len(ctr[k]))
            steps_tot.append((q, args.rounds * args.local_epochs * int(math.ceil(1 / q))))
        # calibrate sigma for the client with the most steps (worst case) -> all clients meet eps
        sigmas = [get_noise_multiplier(target_epsilon=args.dp_eps, target_delta=args.dp_delta,
                                       sample_rate=q, steps=s, accountant="rdp") for q, s in steps_tot]
        sigma = max(sigmas)
        print("DP sigma", sigma, flush=True)
    curve = []
    comm_bytes = 0
    shared_keys = [k for k in gstate if not (local_norm and k in norm_keys)]
    frozen = {k for k, v in gmodel.named_parameters() if not v.requires_grad}
    frozen |= {f"{mn}.{bn}" for mn, mod in gmodel.named_modules() for bn, _ in mod.named_buffers(recurse=False)
               if any(f"{mn}.".startswith(f + ".") or mn == f for f in ["features"]) and hasattr(gmodel, "pool")}
    shared_numel = sum(gstate[k].numel() for k in shared_keys if gstate[k].dtype.is_floating_point and k not in frozen)
    for r in range(args.rounds):
        lr = 0.5 * args.lr * (1 + math.cos(math.pi * r / args.rounds))
        sel = np.arange(K) if args.frac >= 1 else np.sort(np.random.default_rng(args.seed * 1000 + r).choice(K, max(1, int(round(args.frac * K))), replace=False))
        states, weights, dcs = [], [], []
        gparams = [p.detach().clone() for p in gmodel.parameters()]
        for k in sel:
            m = copy.deepcopy(gmodel)
            if local_norm:
                sd = m.state_dict(); sd.update(client_norm[k]); m.load_state_dict(sd)
            prox = (float(mu_k[k]), gparams) if mu_k[k] > 0 else None
            if args.dp_eps > 0:
                q = min(1.0, args.bs / len(ctr[k]))
                steps = args.local_epochs * int(math.ceil(1 / q))
                pd = {n: g for n, g in zip(names, gparams)} if prox else None
                dp_epochs(m, X, y, ctr[k], steps, lr, q, args.dp_clip, sigma, gen,
                          prox=(prox[0], pd) if prox else None)
            elif args.method == "scaffold":
                steps = run_epochs(m, X, y, ctr[k], args.local_epochs, lr, args.bs, gen, scaffold=(c_glob, c_loc[k]))
                newc = []
                for p, g0, cg, cl in zip(m.parameters(), gparams, c_glob, c_loc[k]):
                    newc.append(cl - cg + (g0 - p.detach()) / (max(steps, 1) * lr))
                dcs.append([a - b for a, b in zip(newc, c_loc[k])]); c_loc[k] = newc
            else:
                run_epochs(m, X, y, ctr[k], args.local_epochs, lr, args.bs, gen, prox=prox, prox_mask=prox_mask)
            sd = m.state_dict()
            if local_norm:
                client_norm[k] = {kk: sd[kk].clone() for kk in client_norm[k]}
            states.append(sd); weights.append(n_k[k])
            comm_bytes += (4 if args.method == "scaffold" else 2) * shared_numel * 4
        w = np.array(weights) / sum(weights)
        new = {}
        for kk in gstate:
            if local_norm and kk in norm_keys:
                new[kk] = gstate[kk]; continue
            if not states[0][kk].dtype.is_floating_point:
                new[kk] = states[0][kk]; continue
            new[kk] = sum(float(wi) * s[kk] for wi, s in zip(w, states))
        gmodel.load_state_dict(new); gstate = new
        if args.method == "scaffold":
            for i in range(len(c_glob)):
                c_glob[i] = c_glob[i] + sum(d[i] for d in dcs) / K
        if (r + 1) % args.eval_every == 0 or r == args.rounds - 1:
            probs, labs = evaluate_clients(gmodel, client_norm if local_norm else None, X, y, cte)
            curve.append([r + 1, float((probs.argmax(1) == labs).mean())])
            print(f"round {r+1} acc {curve[-1][1]:.4f}", flush=True)
    if cval is not None:
        # choose fine-tuning epochs and learning rate on the held-out validation groups only
        from sklearn.metrics import f1_score
        grid = [(0, 0.0)] + [(e, l) for e in (1, 3, 5) for l in (0.0005, 0.001, 0.002, 0.005)]
        scores = []
        for e, l in grid:
            pv, lv = [], []
            for k in range(K):
                if not len(cval[k]):
                    continue
                m = copy.deepcopy(gmodel)
                if local_norm:
                    sd = m.state_dict(); sd.update(client_norm[k]); m.load_state_dict(sd)
                if e > 0:
                    run_epochs(m, X, y, ctr[k], e, l, args.bs, gen)
                pv.append(predict(m, X, cval[k]).argmax(1)); lv.append(y[cval[k]])
            f = float(f1_score(np.concatenate(lv), np.concatenate(pv), average="macro"))
            scores.append(dict(epochs=e, lr=l, val_f1=f)); print("ft-val", e, l, f"{f:.4f}", flush=True)
        return dict(ft_val=scores, n_val=int(sum(len(v) for v in cval)))
    probs, labs = [], []
    per_client = []
    order = []
    for k in range(K):
        m = copy.deepcopy(gmodel)
        if local_norm:
            sd = m.state_dict(); sd.update(client_norm[k]); m.load_state_dict(sd)
        if args.finetune > 0:  # personalisation: brief local fine-tuning of the final federated model
            run_epochs(m, X, y, ctr[k], args.finetune, args.ft_lr, args.bs, gen)
        p = predict(m, X, cte[k]) if len(cte[k]) else np.zeros((0, n_cls))
        probs.append(p); order += list(cte[k])
        per_client.append(dict(n_test=len(cte[k]), acc=float((p.argmax(1) == y[cte[k]]).mean()) if len(cte[k]) else None,
                               site=int(csite[k]), n_train=int(n_k[k]), H=float(H[k]), mu=float(mu_k[k])))
    probs = np.concatenate(probs); order = np.array(order)
    if args.save_model:
        torch.save(dict(global_=gmodel.state_dict(), norms=client_norm, csite=csite), args.save_model)
    return dict(test_ids=order.tolist(), probs=probs.tolist(), labels=y[order].tolist(), sites=site[order].tolist(),
                curve=curve, per_client=per_client, comm_MB=comm_bytes / 2**20, sigma=sigma,
                n_params=int(sum(p.numel() for p in gmodel.parameters())), shared_params=int(shared_numel))


def evaluate_clients(gmodel, client_norm, X, y, cte):
    probs, labs = [], []
    for k in range(len(cte)):
        if not len(cte[k]):
            continue
        m = gmodel
        if client_norm is not None:
            m = copy.deepcopy(gmodel); sd = m.state_dict(); sd.update(client_norm[k]); m.load_state_dict(sd)
        probs.append(predict(m, X, cte[k])); labs.append(y[cte[k]])
    return np.concatenate(probs), np.concatenate(labs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--mode", default="fl"); ap.add_argument("--model", default="fenet5")
    ap.add_argument("--method", default="fedavg"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--clients", type=int, default=2); ap.add_argument("--alpha", type=float, default=0.0)
    ap.add_argument("--frac", type=float, default=1.0)
    ap.add_argument("--rounds", type=int, default=50); ap.add_argument("--local_epochs", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=100); ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--bs", type=int, default=32); ap.add_argument("--mu", type=float, default=0.01)
    ap.add_argument("--lam", type=float, default=9.0)
    ap.add_argument("--finetune", type=int, default=0); ap.add_argument("--ft_lr", type=float, default=0.005)
    ap.add_argument("--dp_eps", type=float, default=0.0); ap.add_argument("--dp_delta", type=float, default=1e-4)
    ap.add_argument("--dp_clip", type=float, default=1.0)
    ap.add_argument("--train_site", type=int, default=None)
    ap.add_argument("--scaffold_mom", action="store_true"); ap.add_argument("--prox_skip_local", action="store_true")
    ap.add_argument("--split", default="group", choices=["group", "official", "random"])
    ap.add_argument("--ft_select", action="store_true"); ap.add_argument("--val_frac", type=float, default=0.0)
    ap.add_argument("--eval_every", type=int, default=5); ap.add_argument("--save_model", default=None)
    args = ap.parse_args()
    global SCAFFOLD_MOM; SCAFFOLD_MOM = args.scaffold_mom
    X, y, site, grp, classes = load(args.cache)
    t0 = time.time()
    if args.ft_select:
        a2 = copy.copy(args); a2.val_frac = 0.2; a2.finetune = 0
        sel = fl(a2, X, y, site, grp, classes)
        best = max(sel["ft_val"], key=lambda d: d["val_f1"])  # first maximum = fewest epochs / lowest lr on ties
        args.finetune, args.ft_lr = int(best["epochs"]), float(best["lr"])
        print("chosen", args.finetune, args.ft_lr, flush=True)
        res = fl(args, X, y, site, grp, classes)
        res["ft_val"] = sel["ft_val"]; res["n_val"] = sel["n_val"]
    else:
        res = central(args, X, y, site, grp, classes) if args.mode == "central" else fl(args, X, y, site, grp, classes)
    res.update(vars(args)); res["time_s"] = time.time() - t0; res["classes"] = classes
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(res, open(args.out, "w"))
    print("done", args.out, f"{res['time_s']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
