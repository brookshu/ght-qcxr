import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# ---------------- CONFIG ----------------
CVAT_XML = Path("cvat_export/annotations/annotations.xml")   # <-- your CVAT XML path
IMAGES_DIR = Path("cvat_export/images")          # <-- folder with the PNGs referenced by <image name="...">
OUT_DIR = Path("dataset_yolo")

VAL_RATIO = 0.2
SEED = 42

# Labels in your CVAT XML
LEFT_LABEL = "left_clavicle"
RIGHT_LABEL = "right_clavicle"
SPINE_LABEL = "spinous_process"

# Output: one YOLO "class" for the landmark instance
CLASS_ID = 0
CLASS_NAME = "cxr_landmarks"

# bbox padding around keypoints as fraction of image size
PAD_FRAC = 0.02
# ----------------------------------------


def parse_point_str(s: str):
    # "x,y" -> (float(x), float(y))
    x_str, y_str = s.split(",")
    return float(x_str), float(y_str)


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def build_image_index(root: Path):
    """Index all png/jpg/jpeg under IMAGES_DIR by basename (fast + robust)."""
    idx = {}
    exts = {".png", ".jpg", ".jpeg"}
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            idx.setdefault(p.name, p)
    return idx


def img_size(p: Path):
    with Image.open(p) as im:
        return im.size  # (W,H)


def choose_spine_point(spine_pts, target_y):
    """
    spine_pts: list[(x,y)]
    target_y: float
    choose spine point with y closest to target_y
    """
    return min(spine_pts, key=lambda pt: abs(pt[1] - target_y))


def bbox_from_pts(pts, W, H, pad_frac=PAD_FRAC):
    """
    pts: list of (x,y) in pixels; may include None
    returns normalized (xc, yc, w, h) in [0,1]
    """
    xs = [p[0] for p in pts if p is not None]
    ys = [p[1] for p in pts if p is not None]
    if not xs:
        return None

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    pad_x = pad_frac * W
    pad_y = pad_frac * H

    x_min = max(0.0, x_min - pad_x)
    y_min = max(0.0, y_min - pad_y)
    x_max = min(float(W), x_max + pad_x)
    y_max = min(float(H), y_max + pad_y)

    bw = max(1.0, x_max - x_min)
    bh = max(1.0, y_max - y_min)

    xc = (x_min + x_max) / 2.0
    yc = (y_min + y_max) / 2.0

    return (clamp01(xc / W), clamp01(yc / H), clamp01(bw / W), clamp01(bh / H))


def yolo_kpt_triplet(pt, W, H, visible: bool):
    """
    Return [x_norm, y_norm, v] with v in {0,1}
    If pt is None => [0,0,0]
    """
    if pt is None or not visible:
        return [0.0, 0.0, 0.0]
    x, y = pt
    return [clamp01(x / W), clamp01(y / H), 1.0]


def main():
    random.seed(SEED)

    if not CVAT_XML.exists():
        raise FileNotFoundError(f"CVAT XML not found: {CVAT_XML.resolve()}")
    if not IMAGES_DIR.exists():
        raise FileNotFoundError(f"Images dir not found: {IMAGES_DIR.resolve()}")

    img_index = build_image_index(IMAGES_DIR)
    if not img_index:
        raise RuntimeError(f"No images found under {IMAGES_DIR.resolve()}")

    # Parse XML
    tree = ET.parse(CVAT_XML)
    root = tree.getroot()

    images = root.findall("image")
    print(f"Found {len(images)} <image> entries in XML.")

    # Output dirs
    img_tr = OUT_DIR / "images" / "train"
    img_va = OUT_DIR / "images" / "val"
    lab_tr = OUT_DIR / "labels" / "train"
    lab_va = OUT_DIR / "labels" / "val"
    for p in [img_tr, img_va, lab_tr, lab_va]:
        p.mkdir(parents=True, exist_ok=True)

    # Split ids
    ids = list(range(len(images)))
    random.shuffle(ids)
    n_val = int(len(ids) * VAL_RATIO)
    val_ids = set(ids[:n_val])

    copied = 0
    labeled = 0
    skipped_no_clav = 0
    skipped_no_spine = 0
    skipped_missing_img = 0

    for idx, img_el in enumerate(tqdm(images, desc="Converting")):
        name = img_el.get("name")
        if not name:
            continue

        # Locate image
        src = img_index.get(Path(name).name)
        if src is None:
            # try exact relative path
            candidate = IMAGES_DIR / name
            if candidate.exists():
                src = candidate
            else:
                skipped_missing_img += 1
                continue

        W = int(img_el.get("width", "0"))
        H = int(img_el.get("height", "0"))
        if W == 0 or H == 0:
            # fallback to reading the actual image
            W, H = img_size(src)

        # Collect points
        left_pts = []
        right_pts = []
        spine_pts = []

        for p in img_el.findall("points"):
            label = p.get("label", "")
            pt_str = p.get("points", "")
            if not pt_str:
                continue
            # CVAT "points" for a single point is "x,y"
            x, y = parse_point_str(pt_str)

            if label == LEFT_LABEL:
                left_pts.append((x, y))
            elif label == RIGHT_LABEL:
                right_pts.append((x, y))
            elif label == SPINE_LABEL:
                spine_pts.append((x, y))

        # Choose ONE clavicle point each:
        # if multiple exist (rare), take the one with smallest x-distance to midline? we'll just take first.
        left = left_pts[0] if left_pts else None
        right = right_pts[0] if right_pts else None

        # Must have at least one clavicle to define "clavicle height"
        if left is None and right is None:
            skipped_no_clav += 1
            continue

        # Must have spine candidates to pick from
        if not spine_pts:
            skipped_no_spine += 1
            continue

        # Define target y-level (clavicle height)
        if left is not None and right is not None:
            target_y = 0.5 * (left[1] + right[1])
        elif left is not None:
            target_y = left[1]
        else:
            target_y = right[1]

        spine = choose_spine_point(spine_pts, target_y)

        # Auto bbox from the three points
        bbox = bbox_from_pts([left, right, spine], W, H)
        if bbox is None:
            continue
        xc, yc, bw, bh = bbox

        # Keypoints (3): left, right, spine
        # We always mark chosen points visible=1 if present; if missing, visibility=0
        kpts = []
        kpts += yolo_kpt_triplet(left, W, H, visible=(left is not None))
        kpts += yolo_kpt_triplet(right, W, H, visible=(right is not None))
        kpts += yolo_kpt_triplet(spine, W, H, visible=True)

        line = " ".join(
            [str(CLASS_ID), f"{xc:.6f}", f"{yc:.6f}", f"{bw:.6f}", f"{bh:.6f}"] +
            [f"{v:.6f}" for v in kpts]
        )

        # Copy image + write label
        is_val = idx in val_ids
        dst_img = (img_va if is_val else img_tr) / Path(name).name
        dst_lab = (lab_va if is_val else lab_tr) / (Path(name).stem + ".txt")

        shutil.copy2(src, dst_img)
        dst_lab.write_text(line + "\n", encoding="utf-8")

        copied += 1
        labeled += 1

    # dataset.yaml
    yaml_path = OUT_DIR / "dataset.yaml"
    yaml_text = f"""path: {OUT_DIR.resolve()}
train: images/train
val: images/val
names:
  0: {CLASS_NAME}
"""
    yaml_path.write_text(yaml_text, encoding="utf-8")

    print("\nDone.")
    print(f"Images copied: {copied}")
    print(f"Label files written: {labeled}")
    print(f"Skipped (missing image file): {skipped_missing_img}")
    print(f"Skipped (no clavicle points): {skipped_no_clav}")
    print(f"Skipped (no spinous_process points): {skipped_no_spine}")
    print(f"YAML: {yaml_path.resolve()}")
    print("\nTrain with:")
    print(f"  yolo pose train model=yolo11s-pose.pt data={yaml_path} imgsz=1024")


if __name__ == "__main__":
    main()