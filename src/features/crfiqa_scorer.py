"""
crfiqa_scorer.py
-----------------
Computes GENUINE CR-FIQA quality scores using the official pretrained
checkpoints (fdbtrs/CR-FIQA, CVPR 2023), now supplied by the user:

  32572backbone.pth  -> CR-FIQA(S), iresnet50 backbone
  181952backbone.pth -> CR-FIQA(L), iresnet100 backbone   (used by default:
                         larger model, and CR-FIQA(L) is the paper's
                         headline configuration)

Checkpoint structure (verified directly from the .pth state_dict):
  - Standard IResNet backbone keys (conv1, bn1, prelu, layer1..4, bn2, fc,
    features) IDENTICAL to iresnet.py's IResNet class.
  - PLUS one extra head: `qs` = nn.Linear(512, 1), applied to the 512-d
    face embedding ("features" output) to regress the quality score.
    This matches the CR-FIQA paper's design: the quality-regression head
    sits on top of the same embedding used for recognition, trained
    jointly with the classification margin loss so the quality score
    reflects "relative classifiability" (how well-separated the sample's
    embedding is from its class boundary vs. from other classes).

This is a SEPARATE network from the FR backbone already used for
verification embeddings (195520backbone.pth, iresnet34) -- CR-FIQA is
purpose-trained for quality estimation and is not expected to share an
architecture with the enrollment/verification backbone.

Usage:
    python3 src/features/crfiqa_scorer.py --backbone L    # or S
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from iresnet import iresnet50, iresnet100

CHECKPOINTS = {
    "S": {"path": "models/32572backbone.pth", "arch": iresnet50, "name": "CR-FIQA(S)"},
    "L": {"path": "models/181952backbone.pth", "arch": iresnet100, "name": "CR-FIQA(L)"},
}


class CRFIQAModel(nn.Module):
    """IResNet backbone + the CR-FIQA quality-regression head (`qs`)."""
    def __init__(self, arch_fn):
        super().__init__()
        self.backbone = arch_fn()
        self.qs = nn.Linear(512, 1)

    def forward(self, x):
        emb = self.backbone(x)          # (B, 512), CR-FIQA's own embedding space
        q = self.qs(emb)                # (B, 1), raw quality score
        return emb, q


def build_model(variant: str, device: str) -> CRFIQAModel:
    cfg = CHECKPOINTS[variant]
    model = CRFIQAModel(cfg["arch"])
    sd = torch.load(cfg["path"], map_location="cpu")
    backbone_sd = {k: v for k, v in sd.items() if k != "qs.weight" and k != "qs.bias"}
    qs_sd = {"weight": sd["qs.weight"], "bias": sd["qs.bias"]}
    missing, unexpected = model.backbone.load_state_dict(backbone_sd, strict=True)
    model.qs.load_state_dict(qs_sd, strict=True)
    model.eval().to(device)
    print(f"Loaded {cfg['name']} ({cfg['path']}) -- backbone strict load OK "
          f"(missing={missing}, unexpected={unexpected})")
    return model


# CR-FIQA (like the ArcFace-family models it's trained alongside) expects
# 112x112, BGR-ordered pixels normalized to roughly [-1, 1]. Since PIL loads
# RGB, we flip channels to match the official preprocessing.
#
# NOTE ON METHODOLOGY DOCUMENT: methodology_calfw.docx Step 4 says
# preprocessing is "same as embedding extraction" (i.e. RGB, no channel
# flip -- see Step 1). That is incomplete: the official CR-FIQA repo
# (fdbtrs/CR-FIQA) explicitly preprocesses its evaluation images to BGR
# before scoring (feature_extraction/extract_xqlfw.py: "saves the images
# in BGR format"), consistent with the wider ArcFace/InsightFace/MXNet
# lineage these checkpoints come from. This BGR flip is therefore kept as
# written -- it is the methodology document that should be updated to
# mention it, not this code that should be changed to drop it (dropping it
# would silently produce non-genuine CR-FIQA scores despite using the
# genuine checkpoint). See project summary for the source check.
crfiqa_transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


def to_bgr(img_tensor: torch.Tensor) -> torch.Tensor:
    return img_tensor[[2, 1, 0], :, :]


@torch.no_grad()
def score_batch(paths, model, device):
    imgs = []
    for p in paths:
        t = crfiqa_transform(Image.open(p).convert("RGB"))
        imgs.append(to_bgr(t))
    batch = torch.stack(imgs).to(device)
    _, q = model(batch)
    return q.squeeze(1).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="outputs/manifest.csv")
    ap.add_argument("--backbone", choices=["S", "L"], default="L")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--out_csv", default=None,
                     help="Defaults to crfiqa_scores_<BACKBONE>.csv. Methodology Step 4 runs both "
                          "backbones 'in parallel' (i.e. as two separate runs); the old shared default "
                          "of crfiqa_scores.csv meant a second run (e.g. --backbone L after --backbone S) "
                          "would see every image already present in the file from the FIRST run's rows "
                          "and score nothing, since resumability here keys on img_path alone. "
                          "Backbone-specific defaults avoid that collision and also match "
                          "build_pairs.py's --crfiqa_scores_s / --crfiqa_scores_l defaults.")
    args = ap.parse_args()
    if args.out_csv is None:
        args.out_csv = f"outputs/crfiqa_scores_{args.backbone}.csv"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(args.backbone, device)

    manifest = pd.read_csv(args.manifest)
    unique_paths = sorted(set(manifest["img1_path"]) | set(manifest["img2_path"]))

    done = set()
    if os.path.exists(args.out_csv):
        done = set(pd.read_csv(args.out_csv)["img_path"])
    todo = [p for p in unique_paths if p not in done]
    print(f"{len(done)} already scored, {len(todo)} remaining")

    for i in range(0, len(todo), args.batch_size):
        batch = todo[i:i + args.batch_size]
        try:
            scores = score_batch(batch, model, device)
        except Exception as e:
            print(f"ERROR on batch starting {batch[0]}: {e}", flush=True)
            continue
        rows = [{"img_path": p, "crfiqa_raw": float(s), "crfiqa_backbone": args.backbone}
                for p, s in zip(batch, scores)]
        header = not os.path.exists(args.out_csv)
        pd.DataFrame(rows).to_csv(args.out_csv, mode="a", header=header, index=False)
        print(f"  scored {i + len(batch)}/{len(todo)}", flush=True)

    print("Done.")


if __name__ == "__main__":
    main()
