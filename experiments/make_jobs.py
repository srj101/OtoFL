"""Rebuild experiments/jobs.txt from the arguments stored in every result JSON.

Usage: python experiments/make_jobs.py RES_DIR > experiments/jobs.txt
Each line is  NAME|ARGS  where NAME is the result path under RES_DIR (without .json).
"""
import glob, json, os, sys

RES = sys.argv[1]
DEFAULTS = dict(mode="fl", model="fenet5", method="fedavg", clients=2, alpha=0.0, frac=1.0, rounds=50,
                local_epochs=2, epochs=100, lr=0.02, bs=32, mu=0.01, lam=9.0, finetune=0, ft_lr=0.005,
                dp_eps=0.0, dp_delta=1e-4, dp_clip=1.0, train_site=None, split="group", val_frac=0.0)
FLAGS = ("scaffold_mom", "prox_skip_local", "ft_select")
SKIP = {"cache", "out", "eval_every", "save_model"}
for f in sorted(glob.glob(os.path.join(RES, "**", "*.json"), recursive=True)):
    r = json.load(open(f))
    if "method" not in r:
        continue
    args = []
    for k in ["mode", "model", "method", "seed"] + [k for k in DEFAULTS if k not in ("mode", "model", "method")]:
        if k in SKIP or k not in r:
            continue
        if k == "seed" or r[k] != DEFAULTS.get(k):
            if r[k] is not None:
                args.append(f"--{k} {r[k]}")
    args += [f"--{k}" for k in FLAGS if r.get(k)]
    name = os.path.relpath(f, RES)[:-5]
    print(f"{name}|{' '.join(args)}")
