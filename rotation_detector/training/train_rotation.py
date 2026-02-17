"""
Option A: Pretrained CXR feature extractor + Logistic Regression rotation classifier.

What you get:
- train_rotation.py: reads rotation_labels.csv (path,label), extracts embeddings using a pretrained CXR model,
  trains a logistic regression classifier, and saves:
    - rotation_encoder.pt (feature extractor weights reference)
    - rotation_clf.joblib (sklearn classifier)
    - rotation_config.json (preprocess + threshold defaults)
- inference helpers you can import into your project.

Assumptions:
- You have a CSV like:
    path,label
    cxr/img001.dcm,1
    cxr/img002.dcm,0
- Paths can be .dcm or .png/.jpg; DICOMs are auto-windowed with a simple robust scaling.

Install:
  pip install torch torchvision torchaudio
  pip install torchxrayvision scikit-learn joblib pandas pydicom pillow tqdm

Run:
  python train_rotation.py --csv rotation_labels.csv --out_dir rotation_rotclf --device mps
  # or --device cpu

Then in your pipeline:
  from rotation_infer import load_rotation_model, predict_rotation
  rot = predict_rotation(dicom_path_or_pil_or_np)
"""

# ---------------------------- train_rotation.py ----------------------------

import os
import json
import argparse
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
from sklearn import dummy
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision.transforms as T
import torchxrayvision as xrv

from PIL import Image, ImageOps
import pydicom

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
import joblib

def dicom_to_pil_gray(path: str) -> Image.Image:
    ds = pydicom.dcmread(path)
    arr = ds.pixel_array.astype(np.float32)

    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    arr = arr * slope + intercept

    lo, hi = np.percentile(arr, 1.0), np.percentile(arr, 99.0)
    arr = np.clip(arr, lo, hi) if hi > lo else np.zeros_like(arr)
    arr = (arr - lo) / (hi - lo) if hi > lo else arr
    img_u8 = (arr * 255.0).astype(np.uint8)

    img = Image.fromarray(img_u8, mode="L")
    if getattr(ds, "PhotometricInterpretation", "MONOCHROME2") == "MONOCHROME1":
        img = ImageOps.invert(img)
    return img

# --------- DICOM -> PIL (robust, simple) ----------
def dicom_to_pil_rgb(path: str) -> Image.Image:
    ds = pydicom.dcmread(path)
    arr = ds.pixel_array.astype(np.float32)

    # Apply rescale if present
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    arr = arr * slope + intercept

    # Robust percentile window
    lo, hi = np.percentile(arr, 1.0), np.percentile(arr, 99.0)
    if hi <= lo:
        img_u8 = np.zeros(arr.shape, dtype=np.uint8)
    else:
        arr = np.clip(arr, lo, hi)
        arr = (arr - lo) / (hi - lo)
        img_u8 = (arr * 255.0).astype(np.uint8)

    img = Image.fromarray(img_u8, mode="L")

    # Handle inversion
    if getattr(ds, "PhotometricInterpretation", "MONOCHROME2") == "MONOCHROME1":
        img = ImageOps.invert(img)

    return img.convert("RGB")

def dicom_to_uint8_2d(path: str) -> np.ndarray:
    ds = pydicom.dcmread(path)
    arr = ds.pixel_array.astype(np.float32)

    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    arr = arr * slope + intercept

    lo, hi = np.percentile(arr, 1.0), np.percentile(arr, 99.0)
    if hi <= lo:
        u8 = np.zeros(arr.shape, dtype=np.uint8)
    else:
        arr = np.clip(arr, lo, hi)
        u8 = ((arr - lo) / (hi - lo) * 255.0).astype(np.uint8)

    # handle MONOCHROME1 inversion
    if getattr(ds, "PhotometricInterpretation", "MONOCHROME2") == "MONOCHROME1":
        u8 = 255 - u8

    return u8


def load_image_any(path: str) -> Image.Image:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".dcm":
        return dicom_to_pil_gray(path)
    return Image.open(path).convert("RGB")

def xrv_preprocess_from_uint8(img_u8_2d: np.ndarray) -> torch.Tensor:
    """
    img_u8_2d: HxW uint8 (0..255)
    returns: torch tensor shape [1, 224, 224] float32
    """
    img = xrv.datasets.normalize(img_u8_2d, 255)  # important: uses XRV expected scaling :contentReference[oaicite:1]{index=1}
    if len(img.shape) > 2:
        img = img[:, :, 0]
    img = img[None, :, :]  # add channel -> [1,H,W]
    img = xrv.datasets.XRayCenterCrop()(img)      # :contentReference[oaicite:2]{index=2}
    img = xrv.datasets.XRayResizer(224)(img)      # keep encoder input stable
    return torch.from_numpy(img).float()

# --------- Preprocess: resize with padding to 224x224 ----------
def resize_pad(img: Image.Image, size: int = 224) -> Image.Image:
    img = img.convert("L") 

    w, h = img.size
    scale = size / float(max(w, h))
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    img = img.resize((new_w, new_h), resample=Image.BICUBIC)

    canvas = Image.new("L", (size, size), 0)  # <-- L canvas, not RGB
    left = (size - new_w) // 2
    top = (size - new_h) // 2
    canvas.paste(img, (left, top))
    return canvas


@dataclass
class RotationModelArtifacts:
    encoder_name: str
    img_size: int = 224


def get_device(device_str: str) -> torch.device:
    if device_str == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if device_str == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# --------- Encoder: TorchXRayVision pretrained DenseNet (CXR-specific) ----------
def load_encoder(encoder_name: str = "densenet121-res224-all") -> Tuple[nn.Module, int]:
    """
    Uses torchxrayvision. Encoder outputs a feature embedding.
    Common options:
      - densenet121-res224-all
      - densenet121-res224-chex
      - densenet121-res224-rsna
    """

    # pretrained weights
    model = xrv.models.get_model(encoder_name)
    model.eval()

    # We'll use the logits before classifier? In xrv, model is already a classifier.
    # Easiest: take penultimate features via model.features2() if available.
    # If not available, we will forward and use the returned logits as embeddings (still works).
    embed_dim = None

    # Try common feature hooks
    if hasattr(model, "features"):
        # DenseNet has .features; we can do a forward hook on global pooling output
        # But simplest robust approach: use xrv's feature extractor helper if present.
        pass

    # We'll implement a safe wrapper that returns pooled features for DenseNet variants.
    class DenseNetEmbedding(nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base

        def forward(self, x):
            # For xrv DenseNet, base.features exists and base.classifier exists
            if hasattr(self.base, "features"):
                feats = self.base.features(x)
                out = torch.relu(feats)
                # global average pool
                out = torch.nn.functional.adaptive_avg_pool2d(out, (1, 1))
                out = torch.flatten(out, 1)
                return out
            # fallback: use logits
            return self.base(x)

    embedder = DenseNetEmbedding(model).eval()

    # infer embed dim
    with torch.no_grad():
        dummy = torch.zeros(1, 1, 224, 224)  # was (1,3,224,224)
        emb = embedder(dummy)
        embed_dim = int(emb.shape[1])

    return embedder, embed_dim


def build_transform(img_size: int = 224) -> T.Compose:
    return T.Compose([
        T.Lambda(lambda im: resize_pad(im.convert("L"), img_size)),  # <-- force grayscale
        T.ToTensor(),                                               # -> [1, H, W]
        T.Normalize(mean=[0.5], std=[0.25]),                         # <-- 1-channel normalize
    ])


def extract_embeddings(
    df: pd.DataFrame,
    encoder: nn.Module,
    transform: T.Compose,
    device: torch.device,
    batch_size: int = 16,
) -> Tuple[np.ndarray, np.ndarray]:
    paths = df["path"].astype(str).tolist()
    labels = df["label"].astype(int).to_numpy()

    encoder.to(device)
    encoder.eval()

    embs: list[np.ndarray] = []

    # simple batching
    batch_imgs = []
    batch_idx = []

    def flush():
        nonlocal batch_imgs, batch_idx
        if not batch_imgs:
            return
        x = torch.stack(batch_imgs, dim=0).to(device)
        with torch.no_grad():
            e = encoder(x).detach().cpu().numpy()
        embs.append(e)
        batch_imgs, batch_idx[:] = [], []

    for i, p in enumerate(tqdm(paths, desc="Embedding")):
        # img = load_image_any(p)
        # x = transform(img)
        # batch_imgs.append(x)
        img_u8 = dicom_to_uint8_2d(p)  # or for PNG/JPG: read grayscale uint8
        x = xrv_preprocess_from_uint8(img_u8)  # [1,224,224]
        batch_imgs.append(x)
        batch_idx.append(i)
        if len(batch_imgs) >= batch_size:
            flush()

    flush()

    X = np.concatenate(embs, axis=0)
    y = labels
    return X, y


def cross_validate_logreg(X: np.ndarray, y: np.ndarray, n_splits: int = 5) -> Dict[str, Any]:
    # For tiny data, reduce splits if needed
    unique, counts = np.unique(y, return_counts=True)
    min_class = int(counts.min()) if len(counts) > 0 else 0
    n_splits = min(n_splits, min_class) if min_class >= 2 else 2
    n_splits = max(2, n_splits)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    aucs, accs = [], []
    cms = []

    for tr, te in skf.split(X, y):
        clf = LogisticRegression(
            C=0.5,               # stronger regularization helps small data
            max_iter=2000,
            class_weight="balanced",
            solver="liblinear",
        )
        clf.fit(X[tr], y[tr])
        probs = clf.predict_proba(X[te])[:, 1]
        preds = (probs >= 0.5).astype(int)

        # AUC requires both classes in test split; handle edge cases
        if len(np.unique(y[te])) == 2:
            aucs.append(roc_auc_score(y[te], probs))
        accs.append(accuracy_score(y[te], preds))
        cms.append(confusion_matrix(y[te], preds).tolist())

    out = {
        "n_splits": n_splits,
        "auc_mean": float(np.mean(aucs)) if aucs else None,
        "acc_mean": float(np.mean(accs)) if accs else None,
        "confusion_matrices": cms,
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="CSV with columns: path,label")
    ap.add_argument("--out_dir", default="rotation_rotclf", help="Output dir for artifacts")
    ap.add_argument("--encoder", default="densenet121-res224-all", help="TorchXRayVision model name")
    ap.add_argument("--device", default="mps", choices=["cpu", "mps", "cuda"])
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--threshold", type=float, default=0.7, help="Default rotation threshold (tune later)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.csv)
    if "path" not in df.columns or "label" not in df.columns:
        raise ValueError("CSV must have columns: path,label")

    device = get_device(args.device)

    # 1) Load encoder
    encoder, embed_dim = load_encoder(args.encoder)

    # 2) Extract embeddings
    transform = build_transform(img_size=224)
    X, y = extract_embeddings(df, encoder, transform, device, batch_size=args.batch_size)

    np.save(os.path.join(args.out_dir, "features.npy"), X)
    np.save(os.path.join(args.out_dir, "labels.npy"), y)

    # 3) CV report
    report = cross_validate_logreg(X, y, n_splits=5)

    # 4) Train final classifier on all data
    clf = LogisticRegression(
        C=0.5,
        max_iter=2000,
        class_weight="balanced",
        solver="liblinear",
    )
    clf.fit(X, y)
    joblib.dump(clf, os.path.join(args.out_dir, "rotation_clf.joblib"))

    # Save config
    cfg = {
        "encoder_name": args.encoder,
        "embed_dim": embed_dim,
        "img_size": 224,
        "normalize": {"mean": [0.5, 0.5, 0.5], "std": [0.25, 0.25, 0.25]},
        "default_threshold": float(args.threshold),
        "cv_report": report,
    }
    with open(os.path.join(args.out_dir, "rotation_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    print("\n✅ Saved artifacts to:", args.out_dir)
    print("   - features.npy / labels.npy")
    print("   - rotation_clf.joblib")
    print("   - rotation_config.json")
    print("\n📊 CV summary:", report)


if __name__ == "__main__":
    main()
