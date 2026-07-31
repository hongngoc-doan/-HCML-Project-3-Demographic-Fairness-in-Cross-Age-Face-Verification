"""
build_manifest.py
------------------
Parses the pair-folder dataset structure (pairN_label/img1.png, img2.png)
for CALFW and AgeDB-30 into a flat manifest CSV.

Folder convention (verified against the actual uploaded zips):
    calfw/pair{N}_{0|1}/{imgA}.png, {imgB}.png      label: 1=genuine, 0=impostor
    agedb_30/pair{N}_{True|False}/{imgA}.png, {imgB}.png   label: True=genuine, False=impostor

Each pair folder contains exactly two images. The label is embedded in the
folder name and is the SAME for both images of the pair (there is no
separate per-image label).
"""
import os
import re
import argparse
import pandas as pd

PAIR_RE = re.compile(r"^pair(\d+)_(.+)$")

LABEL_MAP = {"1": 1, "0": 0, "true": 1, "false": 0, "True": 1, "False": 0}


def parse_dataset(root_dir: str, dataset_name: str) -> pd.DataFrame:
    rows = []
    for entry in sorted(os.listdir(root_dir)):
        full = os.path.join(root_dir, entry)
        if not os.path.isdir(full):
            continue
        m = PAIR_RE.match(entry)
        if not m:
            print(f"WARNING: skipping unrecognized folder name: {entry}")
            continue
        pair_id, label_raw = m.group(1), m.group(2)
        label = LABEL_MAP.get(label_raw)
        if label is None:
            raise ValueError(f"Unrecognized label token '{label_raw}' in folder {entry}")

        imgs = sorted(
            f for f in os.listdir(full)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        )
        if len(imgs) != 2:
            raise ValueError(f"Expected exactly 2 images in {full}, found {len(imgs)}: {imgs}")

        rows.append({
            "dataset": dataset_name,
            "pair_id": int(pair_id),
            "label": label,  # 1 = genuine (same identity), 0 = impostor (different identity)
            "img1_path": os.path.join(full, imgs[0]),
            "img2_path": os.path.join(full, imgs[1]),
        })
    df = pd.DataFrame(rows).sort_values(["dataset", "pair_id"]).reset_index(drop=True)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calfw_dir", default="data/calfw")
    ap.add_argument("--agedb_dir", default="data/agedb_30")
    ap.add_argument("--out_csv", default="outputs/manifest.csv")
    args = ap.parse_args()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)

    dfs = []
    if args.calfw_dir and os.path.isdir(args.calfw_dir):
        print(f"Parsing CALFW from {args.calfw_dir}...")
        dfs.append(parse_dataset(args.calfw_dir, "CALFW"))
    else:
        print(f"WARNING: CALFW directory not found at {args.calfw_dir}")
    
    if args.agedb_dir and os.path.isdir(args.agedb_dir):
        print(f"Parsing AgeDB-30 from {args.agedb_dir}...")
        dfs.append(parse_dataset(args.agedb_dir, "AgeDB30"))
    else:
        print(f"WARNING: AgeDB-30 directory not found at {args.agedb_dir}")

    if not dfs:
        raise ValueError("No datasets found. Check --calfw_dir and --agedb_dir paths.")

    manifest = pd.concat(dfs, ignore_index=True)
    manifest.to_csv(args.out_csv, index=False)

    print(f"\n✓ Manifest built: {args.out_csv}")
    print("\nDataset breakdown:")
    print(manifest.groupby(["dataset", "label"]).size())
    print(f"\nTotal pairs: {len(manifest)}")
    print(f"Total images: {len(manifest) * 2}")


if __name__ == "__main__":
    main()
