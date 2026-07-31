"""
extract_features.py
--------------------
Produces the per-image feature table (image_features.csv) used by
build_pairs.py. As of this rewrite, it does NOT compute face-recognition
embeddings itself -- it LOADS them from the file(s) already produced by
generate_embeddings.py (embeddings.parquet preferred, falling back to
.npz or .csv), and only runs the FairFace demographic/age model, which is
this script's sole remaining job.

Rationale for the split: embedding generation (generate_embeddings.py) is
the expensive, rarely-changing part of the pipeline; demographic/age
prediction is a separate concern that may be re-run or swapped out
independently (e.g. trying a different demographic classifier) without
ever re-touching the embeddings. Keeping them as two scripts joined by a
file, rather than one script doing both, makes that possible.

Workflow:
    python3 src/features/generate_embeddings.py    # produces embeddings.{csv,npz,parquet}
    python3 src/features/extract_features.py       # loads that file, adds FairFace, writes image_features.csv

Output columns in image_features.csv (one row per unique image):
    dataset, pair_id, filename, img_path, label, is_genuine,   <- carried over from embeddings file
    emb_norm,                                                   <- carried over from embeddings file
    pred_race, pred_race_conf, pred_gender, pred_gender_conf,
    pred_age_bin, pred_age_mid                                  <- computed here
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))

# ---- FairFace label maps (from the official FairFace repo, res34_fair_align_multi_7 model) ----
FAIRFACE_RACE = ["White", "Black", "Latino_Hispanic", "East Asian",
                  "Southeast Asian", "Indian", "Middle Eastern"]
FAIRFACE_GENDER = ["Male", "Female"]
# 9 age bins with representative midpoints used ONLY to derive an estimated numeric age
FAIRFACE_AGE_BINS = ["0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70+"]

FAIRFACE_AGE_MID = [1, 6, 15, 25, 35, 45, 55, 65, 75]


def build_fairface_model(weights_path, device):
    # Official FairFace checkpoint is a fine-tuned torchvision resnet34 with fc -> 18
    model = torchvision.models.resnet34(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 18)
    sd = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(sd, strict=True)
    model.eval().to(device)
    return model


# FairFace expects 224x224 ImageNet-normalized images
ff_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


@torch.no_grad()
def predict_demographics_batch(paths, ff_model, device):
    ff_batch = torch.stack([ff_transform(Image.open(p).convert("RGB")) for p in paths]).to(device)
    logits = ff_model(ff_batch).cpu().numpy()  # (B, 18)
    race_logits, gender_logits, age_logits = logits[:, 0:7], logits[:, 7:9], logits[:, 9:18]
    race_idx, gender_idx, age_idx = race_logits.argmax(1), gender_logits.argmax(1), age_logits.argmax(1)
    race_prob = F.softmax(torch.tensor(race_logits), dim=1).numpy()
    gender_prob = F.softmax(torch.tensor(gender_logits), dim=1).numpy()

    rows = []
    for i, p in enumerate(paths):
        rows.append({
            "img_path": p,
            "pred_race": FAIRFACE_RACE[race_idx[i]],
            "pred_race_conf": float(race_prob[i, race_idx[i]]),
            "pred_gender": FAIRFACE_GENDER[gender_idx[i]],
            "pred_gender_conf": float(gender_prob[i, gender_idx[i]]),
            "pred_age_bin": FAIRFACE_AGE_BINS[age_idx[i]],
            "pred_age_mid": FAIRFACE_AGE_MID[age_idx[i]],
        })
    return rows


def load_embeddings_file(base_path_no_ext: str) -> pd.DataFrame:
    """Load whichever embeddings export exists, preferring the fastest format.
    Only the metadata + emb_norm columns are needed here (not the full 512-d
    vectors), so we avoid materializing those columns from the CSV when a
    parquet/npz file is available."""
    parquet_path = base_path_no_ext + ".parquet"
    npz_path = base_path_no_ext + ".npz"
    csv_path = base_path_no_ext + ".csv"

    keep_cols = ["img_path", "dataset", "pair_id", "filename", "label", "is_genuine", "emb_norm"]

    if os.path.exists(parquet_path):
        print(f"Loading embeddings metadata from {parquet_path}")
        df = pd.read_parquet(parquet_path, columns=keep_cols)
    elif os.path.exists(npz_path):
        print(f"Loading embeddings metadata from {npz_path}")
        npz = np.load(npz_path, allow_pickle=True)
        df = pd.DataFrame({c: npz[c] for c in keep_cols})
    elif os.path.exists(csv_path):
        print(f"Loading embeddings metadata from {csv_path} (slowest option -- "
              f"consider generating .parquet/.npz too via generate_embeddings.py)")
        df = pd.read_csv(csv_path, usecols=keep_cols)
    else:
        raise FileNotFoundError(
            f"No embeddings file found at {base_path_no_ext}.{{parquet,npz,csv}} -- "
            f"run generate_embeddings.py first.")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings_base", default="outputs/embeddings",
                     help="Path (without extension) to the generate_embeddings.py output.")
    ap.add_argument("--ff_weights", default="models/res34_fair_align_multi_7_20190809.pt")
    ap.add_argument("--demo_cache", default="outputs/fairface_predictions.csv",
                     help="Resumable, FairFace-predictions-only cache (append-only).")
    ap.add_argument("--out_csv", default="outputs/image_features.csv",
                     help="Final merged output (embeddings metadata + FairFace); regenerated fresh each run.")
    ap.add_argument("--batch_size", type=int, default=16)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_grad_enabled(False)

    emb_meta = load_embeddings_file(args.embeddings_base)
    all_paths = emb_meta["img_path"].tolist()

    done_paths = set()
    if os.path.exists(args.demo_cache):
        done_paths = set(pd.read_csv(args.demo_cache, usecols=["img_path"])["img_path"])
    todo = [p for p in all_paths if p not in done_paths]
    print(f"{len(done_paths)} images already have demographic predictions, {len(todo)} remaining")

    if todo:
        ff_model = build_fairface_model(args.ff_weights, device)
        for i in range(0, len(todo), args.batch_size):
            batch_paths = todo[i:i + args.batch_size]
            try:
                rows = predict_demographics_batch(batch_paths, ff_model, device)
            except Exception as e:
                print(f"ERROR on batch starting {batch_paths[0]}: {e}", flush=True)
                continue
            new_df = pd.DataFrame(rows)
            header = not os.path.exists(args.demo_cache)
            new_df.to_csv(args.demo_cache, mode="a", header=header, index=False)
            print(f"  processed {i + len(batch_paths)}/{len(todo)}", flush=True)

    demo = pd.read_csv(args.demo_cache).drop_duplicates(subset="img_path", keep="last")
    merged = emb_meta.merge(demo, on="img_path", how="inner")
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    merged.to_csv(args.out_csv, index=False)
    print("Done. Final image_features.csv:", len(merged), "rows x", merged.shape[1], "cols ->", args.out_csv)


if __name__ == "__main__":
    main()
