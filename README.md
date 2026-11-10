# OtoFL: a federated learning benchmark for otoscopy

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE) ![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg) ![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)

Code for the paper

> Md. Salim Reza Joy, Nusrat Faruqi. **When does federated learning help otoscopy? A two-hospital benchmark of federated strategies and differential privacy.** Submitted to *Computer Methods and Programs in Biomedicine*, 2026.

OtoFL combines two public otoscopy datasets from hospitals in Chile and Turkey (1,142 images, five classes), groups near-duplicate video frames before splitting, and compares federated strategies (FedAvg, FedProx, SCAFFOLD, FedBN, AP-FedBN, SiloBN, FedPer, FedAvg with group normalisation) against local and centralised training, with optional sample-level differential privacy (DP-SGD). Every number, table and figure in the paper is produced by the scripts in this repository.

## Contents

| File | Purpose |
|---|---|
| `prep_data.py` | Builds the image index: harmonises labels to five classes, groups near-duplicate frames by perceptual hash (256-bit pHash, Hamming distance ≤ 10, within site) and caches the images. |
| `otofl.py` | All training: centralised, local and federated runs, the federated strategies, the Fenet5 backbone, fine-tuning and DP-SGD. Writes one JSON file per run with predictions and metrics. |
| `experiments/jobs.txt` | The exact arguments of every run in the paper (one line per run: `NAME\|ARGS`). |
| `experiments/run_jobs.py` | Runs the job list in parallel over the available GPUs; restartable. |
| `experiments/make_jobs.py` | Regenerates `jobs.txt` from the arguments stored in result files. |
| `analyze.py` | Aggregates all runs into metrics, paired bootstrap tests, LaTeX tables and figures. |
| `leak_analysis.py` | Duplicate leakage on the official, random and group-aware Site 1 splits. |
| `hash_sensitivity.py` | Sensitivity of duplicate detection to the Hamming threshold. |
| `embed.py`, `dupcheck.py`, `dup_effect.py` | Second, embedding-based duplicate detector (DINOv2 ViT-B/14, ResNet-50) and the accuracy on flagged vs. unflagged test images. |
| `extra_r3.py` | Longer-training comparisons restricted to unflagged test images, and per-class sensitivity at Site 2. |
| `post.py` | Dataset figure, corruption robustness and Grad-CAM on saved seed-0 models. |
| `sync_paper.py` | Copies generated tables and figures into the LaTeX manuscript and builds the significance table with Holm correction. |

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Use Python 3.10 or newer and install PyTorch for your platform first. **scikit-learn must be 1.7.2**: newer releases change the output of `StratifiedGroupKFold` for a fixed seed, which changes the splits. Training runs on CUDA, Apple MPS or CPU (one Fenet5 federated run takes 3–5 minutes on an NVIDIA T4).

## Data

Both datasets are public. Download them into one working directory, for example `work/`:

```bash
mkdir -p work/data/chile work/data/turkey && cd work
curl -L -o datos.rar   https://ndownloader.figshare.com/files/21793293   # Site 1: Ear Imagery Database (Chile)
curl -L -o eardrum.zip https://ndownloader.figshare.com/files/26200970   # Site 2: eardrum dataset (Turkey)
(cd data/chile && unrar x -o+ ../../datos.rar)
(cd data/turkey && unzip -oq ../../eardrum.zip)
cd ..
```

- Site 1: Viscaino et al., Ear Imagery Database, <https://doi.org/10.6084/m9.figshare.11886630>
- Site 2: Başaran et al., eardrum dataset (Özel Van Akdamar Hospital), <https://doi.org/10.6084/m9.figshare.13648166>

Please cite the original dataset papers when you use the data. Otitis externa and tympanostomy-tube images are excluded automatically.

## Reproducing the paper

All commands below use a working directory `work/` that holds `data/`, and they write `cache/`, `res2/` (one JSON per run), `logs/` and `ana2/` (tables and figures) inside it.

**1. Build the index and image cache**

```bash
python prep_data.py work/data work/cache
```

This should report 1,142 images, 207 near-duplicate pairs and 978 groups.

**2. Run the experiments**

```bash
python experiments/run_jobs.py work --workers 4            # all 316 runs
python experiments/run_jobs.py work --filter fl/S1/         # or a subset, e.g. scenario S1
```

A single run, for example FedBN in the two-hospital scenario (S1) with seed 0:

```bash
python otofl.py --cache work/cache --out work/res2/fl/S1/fedbn_s0.json \
    --method fedbn --clients 2 --alpha 0 --rounds 40 --local_epochs 1 --seed 0
```

Scenarios: S1 `--clients 2 --alpha 0`; S2 `--clients 10 --alpha 0.3` (Dirichlet label skew); S3 `--clients 10 --alpha 0`. Methods: `fedavg`, `fedprox`, `scaffold`, `fedbn`, `otofl` (AP-FedBN), `silobn`, `fedper`, `local`; `--mode central` trains on pooled data, or on one site with `--train_site 0|1`. DP runs add `--dp_eps EPS` (with `--model mnv3_frozen`).

**3. Duplicate checks (optional, needed for the leakage tables)**

```bash
python embed.py work/cache work/emb.npz
python dupcheck.py work/cache work/emb.npz work/ana2/dupcheck.json
python dup_effect.py work
python leak_analysis.py work
python hash_sensitivity.py work/cache/index.json work/ana2/hash_sensitivity.json
python extra_r3.py work
```

**4. Tables, figures and statistics**

```bash
OTOFL_INDEX=work/cache/index.json python analyze.py work/res2 work/ana2
```

Robustness and Grad-CAM (`python post.py work work/post`) use seed-0 models saved by adding `--save_model work/models/NAME.pt` to the corresponding runs.

`work/ana2/summary.json` holds every metric and comparison; `work/ana2/tables/` and `work/ana2/figures/` hold the LaTeX rows and PDF figures used in the paper.

## Evaluation protocol in brief

- Splits are stratified by class and keep each near-duplicate group on one side; 20% of each site is held out for testing. The seed changes the split, the client partition and the initialisation.
- The main strategies, fine-tuning, local training and the longer-training runs use eight seeds (0–7); centralised, tuned, ablation, scale, pretrained-backbone and DP runs use three (0–2).
- Comparisons use a two-level paired bootstrap (seeds, then near-duplicate groups within each site; 10,000 replicates) and Holm correction within a primary family of six pre-specified comparisons and an exploratory family. Three-seed comparisons are reported with seed-level paired t-intervals.
- No test image is used for model selection or early stopping; the model after the final round is reported.

## Citation

If you use this code, please cite the paper (see `CITATION.cff`).

## License

MIT (see `LICENSE`). The datasets are distributed by their owners under their own terms.
