# ---------------------------- rotation_infer.py ----------------------------
# Save the below in a separate file (rotation_infer.py) or paste into your project.

"""
Usage:
  from rotation_infer import load_rotation_model, predict_rotation

  rm = load_rotation_model("rotation_rotclf")
  out = predict_rotation(rm, "cxr/overexposed.dcm")
  print(out)
"""

# python eval_classifier.py --csv ./labels/rotation_labels.csv --model_dir ./training/rotation_rotclf

import os as _os
import json as _json
import numpy as _np
import joblib as _joblib
import torch as _torch
import torchvision.transforms as _T
from PIL import Image as _Image
from training.train_rotation import dicom_to_uint8_2d, resize_pad, xrv_preprocess_from_uint8
from training.train_rotation import load_encoder, load_image_any

# NOTE: reuse dicom_to_pil_rgb/load_image_any/resize_pad from above, or duplicate them here.
# For brevity, import them if in same module, otherwise copy those defs into this file.

def load_rotation_model(artifact_dir: str, device: str = "cpu"):
    with open(_os.path.join(artifact_dir, "rotation_config.json"), "r") as f:
        cfg = _json.load(f)

    enc_name = cfg["encoder_name"]
    img_size = int(cfg["img_size"])
    mean = cfg["normalize"]["mean"]
    std = cfg["normalize"]["std"]

    dev = _torch.device(device)
    if device == "mps" and _torch.backends.mps.is_available():
        dev = _torch.device("mps")
    elif device == "cuda" and _torch.cuda.is_available():
        dev = _torch.device("cuda")
    else:
        dev = _torch.device("cpu")

    encoder, _ = load_encoder(enc_name)
    encoder.to(dev).eval()

    transform = _T.Compose([
        _T.Lambda(lambda im: resize_pad(im.convert("L"), img_size)),  # force grayscale
        _T.ToTensor(),                                                # -> [1,H,W]
        _T.Normalize(mean=[0.5], std=[0.25]),                          # 1-channel normalize
    ])

    clf = _joblib.load(_os.path.join(artifact_dir, "rotation_clf.joblib"))
    return {"cfg": cfg, "device": dev, "encoder": encoder, "transform": transform, "clf": clf}


def predict_rotation(rm: dict[str, any], path_or_img, threshold: float = None) -> dict[str, any]:
    cfg = rm["cfg"]
    dev = rm["device"]
    encoder = rm["encoder"]
    transform = rm["transform"]
    clf = rm["clf"]

    if threshold is None:
        threshold = float(cfg.get("default_threshold", 0.7))

    # Load image
    # if isinstance(path_or_img, str):
    #     img = load_image_any(path_or_img)
    # elif isinstance(path_or_img, _Image.Image):
    #     img = path_or_img.convert("RGB")
    # else:
    #     # assume numpy array
    #     img = _Image.fromarray(_np.asarray(path_or_img)).convert("RGB")

    # x = transform(img).unsqueeze(0).to(dev)

    # with _torch.no_grad():
    #     emb = encoder(x).detach().cpu().numpy()
    img_u8 = dicom_to_uint8_2d(path_or_img)
    x = xrv_preprocess_from_uint8(img_u8).unsqueeze(0).to(dev)  # [1,1,224,224]
    with _torch.no_grad():
        emb = encoder(x).detach().cpu().numpy()
    p_rot = float(clf.predict_proba(emb)[:, 1][0])
    rotated = bool(p_rot >= threshold)

    return {
        "p_rotated": p_rot,
        "rotation_detected": rotated,
        "threshold": float(threshold),
    }