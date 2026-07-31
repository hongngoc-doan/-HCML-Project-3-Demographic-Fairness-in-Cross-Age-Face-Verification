# Demographic Fairness in Cross-Age Face Verification

Does the age gap between two face photos amplify demographic bias in automated face
verification — and does face-image-quality (FIQA) explain away cross-age verification
failures? This repository implements the full, reproducible pipeline used to answer both
questions on two benchmarks, **CALFW** and **AgeDB-30** (6,000 pairs each, 12,000 total).

## Repository layout

```
.
├── src/                        # standalone, parametrized pipeline scripts
│   ├── preprocessing/
│   │   ├── build_manifest.py   # Step 0: raw pair folders -> manifest.csv
│   │   └── build_pairs.py      # Step 4: assembles the final pair-level table
│   ├── features/
│   │   ├── iresnet.py          # shared IResNet-34/50/100 backbone definitions
│   │   ├── generate_embeddings.py  # Step 1: face-recognition embeddings (IResNet-34)
│   │   ├── extract_features.py     # Step 2: FairFace race / gender / age prediction
│   │   └── crfiqa_scorer.py        # Step 3: CR-FIQA(S) / CR-FIQA(L) quality scoring
│   └── analysis/
│       └── analyze.py          # Steps 5-7: thresholds, fairness audits, plots
├── data/                      # raw benchmark images (not redistributed)
│   ├── calfw/                  # 6,000 pair{N}_{label}/ folders
│   └── agedb_30/               # 6,000 pair{N}_{label}/ folders
├── models/                    # pretrained checkpoints (not redistributed)
│   ├── 195520backbone.pth              # IResNet-34 face-recognition backbone
│   ├── 32572backbone.pth               # CR-FIQA(S), IResNet-50 backbone
│   ├── 181952backbone.pth              # CR-FIQA(L), IResNet-100 backbone
│   ├── res34_fair_align_multi_7_20190809.pt   # FairFace demographic model
├── outputs/                   # pipeline outputs
│   ├── manifest.csv                # output of build_manifest.py
│   ├── embeddings/, embeddings.csv/.npz/.parquet  # output of generate_embeddings.py
│   ├── image_features.csv          # output of extract_features.py
│   ├── fairface_predictions.csv    # resumable FairFace prediction cache
│   ├── crfiqa_scores_S.csv         # output of crfiqa_scorer.py --backbone S
│   ├── crfiqa_scores_L.csv         # output of crfiqa_scorer.py --backbone L
│   ├── pairs.csv                    # output of build_pairs.py (the final pair table, 31 columns)
│   └── analysis/                    # output of analyze.py: thresholds, regressions, 41 plots
└── requirements.txt
```

The CSVs and plots under `outputs/` are the actual artifacts this pipeline produced on
one full run and are committed as a reference/example — running the scripts yourself
regenerates them deterministically (Steps 5-7 have no randomness; `statsmodels` logit
fits are exact MLE given fixed data).

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Then populate `data/calfw`, `data/agedb_30`, and `models/` with the CALFW/AgeDB-30 pair
folders and the checkpoints listed above (all third-party assets, intentionally not
committed to this repository).

## Pipeline

Each script is standalone and independently parametrized via `argparse`; every argument
below has a sensible default matching the folder layout above, so most steps can be run
with no flags at all once `data/` and `models/` are populated. Only the flags worth
knowing about are shown here — run any script with `--help` for the full list.

```bash
# 1. Flatten the raw pair folders into one manifest CSV
python3 src/preprocessing/build_manifest.py \
    --calfw_dir data/calfw --agedb_dir data/agedb_30 \
    --out_csv outputs/manifest.csv

# 2. Face-recognition embeddings (512-d, IResNet-34), resumable per-image .npy cache
python3 src/features/generate_embeddings.py \
    --manifest outputs/manifest.csv \
    --weights models/195520backbone.pth \
    --emb_dir outputs/embeddings \
    --batch_size 64 --device cuda   # omit --device to auto-detect

# 3. FairFace demographic / age prediction (race, gender, 9-bin age)
python3 src/features/extract_features.py \
    --embeddings_base outputs/embeddings \
    --ff_weights models/res34_fair_align_multi_7_20190809.pt \
    --out_csv outputs/image_features.csv

# 4. CR-FIQA quality scoring — run ONCE PER BACKBONE (S and L are independent audits)
python3 src/features/crfiqa_scorer.py --backbone S --manifest outputs/manifest.csv
python3 src/features/crfiqa_scorer.py --backbone L --manifest outputs/manifest.csv

# 5. Assemble the final pair-level table (similarity, age gap, quality, demographics)
python3 src/preprocessing/build_pairs.py \
    --manifest outputs/manifest.csv \
    --features outputs/image_features.csv \
    --crfiqa_scores_s outputs/crfiqa_scores_S.csv \
    --crfiqa_scores_l outputs/crfiqa_scores_L.csv \
    --out_csv outputs/pairs.csv

# 6-8. Thresholds, fairness audits (logistic regression), and all plots
python3 src/analysis/analyze.py \
    --pairs_csv outputs/pairs.csv \
    --outdir outputs/analysis
```

Steps 1-4 are individually resumable: each caches partial progress to disk (per-image
`.npy` embeddings, an append-only FairFace prediction cache, append-only CR-FIQA score
files) and only (re-)computes what's missing, so an interrupted run can simply be
restarted with the same command.

### Pipeline diagram

```
data/calfw, data/agedb_30 (6,000 + 6,000 pairs)
    |
    v
build_manifest.py --------------------------> manifest.csv
    |                                              |
    v                                              |
generate_embeddings.py (IResNet-34)                |
  -> embeddings.csv/.npz/.parquet                  |
    |                                              |
    v                                              |
extract_features.py (FairFace)                     |
  -> image_features.csv                            |
    |                                              |
    |            crfiqa_scorer.py --backbone S <---+
    |              -> crfiqa_scores_S.csv           |
    |            crfiqa_scorer.py --backbone L <---+
    |              -> crfiqa_scores_L.csv
    v                    |        |
    +--------------------+--------+
                 |
                 v
        build_pairs.py (joins manifest + features + both quality files)
                 |
                 v
        outputs/pairs.csv (12,000 pairs, 31 columns)
                 |
                 v
        analyze.py (Steps 5-7)
                 |
        +--------+--------+
        |                 |
        v                 v
thresholds_summary.csv   interaction_audit_summary.csv
fnmr_by_*.csv            regression_*.txt
                 |
                 v
        outputs/analysis/plots/ (41 figures)
```

## Key findings (summary)

- Verification accuracy is high on both benchmarks (EER ≈ 6.1% CALFW / 3.9% AgeDB-30) and
  broadly race-invariant in ranking quality.
- Larger age gaps measurably increase failure rate in **CALFW** (age-gap main effect
  significant, p < 0.001 across both CR-FIQA backbones) but **not detectably in
  AgeDB-30** (p > 0.06 in both).
- No statistically robust demographic-group × age-gap **interaction** was found in either
  dataset — a single borderline term (CALFW, race, Middle Eastern, CR-FIQA(S), p ≈ 0.042)
  does not replicate under CR-FIQA(L) (p ≈ 0.10) and reads as underpowered given ~20
  interaction terms tested and the heavy White-skew of both benchmarks, not as evidence of
  no true effect.
- Image quality (CR-FIQA `quality_min`) is the single most consistent predictor of
  verification failure across all 8 independent regression audits (p < 1e-8 in every
  case), and quality itself degrades with age gap — a plausible mediating mechanism for
  the CALFW age-gap effect.

## Reproducibility notes

- Age gap and demographic group are **model-predicted** (FairFace), not ground-truth
  annotations — no such metadata survives in the distributed CALFW/AgeDB-30 archives.
- CR-FIQA(S) and CR-FIQA(L) are two independent quality analyses; they are never averaged,
  combined, or compared against one another anywhere in this pipeline.
- CALFW and AgeDB-30 are kept fully independent throughout `analyze.py` (separate ROC
  curves, thresholds, and regressions) and are never pooled.
