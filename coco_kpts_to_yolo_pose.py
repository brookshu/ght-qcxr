import os, json, random, shutil
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# --------- CONFIG ----------
COCO_JSON = "cvat_export/annotations/person_keypoints_default.json"
IMAGES_DIR = "cvat_export/images"
OUT_DIR = "dataset_yolo"
VAL_RATIO = 0.2
SEED = 42

# bbox padding (as fraction of image size) around keypoints
PAD_FRAC = 0.02  # 2% padding
# ---------------------------

random.seed(SEED)

def mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def img_size(p: Path):
    with Image.open(p) as im:
        return im.size  # (W,H)

def coco_kpts_to_yolo_triplets(kpts, W, H):
    # COCO: [x,y,v]*K where v: 0=not labeled, 1=occluded, 2=visible
    out = []
    for i in range(0, len(kpts), 3):
        x, y, v = kpts[i], kpts[i+1], kpts[i+2]
        vis = 1.0 if v and v > 0 else 0.0
        xn = min(max(float(x)/W, 0.0), 1.0)
        yn = min(max(float(y)/H, 0.0), 1.0)
        out.extend([xn, yn, vis])
    return out

def bbox_from_kpts(kpts, W, H):
    xs, ys = [], []
    for i in range(0, len(kpts), 3):
        x, y, v = kpts[i], kpts[i+1], kpts[i+2]
        if v and v > 0:
            xs.append(x); ys.append(y)
    if not xs:
        return None  # no labeled keypoints
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    pad_x = PAD_FRAC * W
    pad_y = PAD_FRAC * H
    x_min = max(0.0, x_min - pad_x)
    y_min = max(0.0, y_min - pad_y)
    x_max = min(float(W), x_max + pad_x)
    y_max = min(float(H), y_max + pad_y)
    bw = max(1.0, x_max - x_min)
    bh = max(1.0, y_max - y_min)
    xc = (x_min + x_max) / 2.0
    yc = (y_min + y_max) / 2.0
    return (xc/W, yc/H, bw/W, bh/H)

def main():
    coco = json.load(open(COCO_JSON, "r", encoding="utf-8"))
    images = coco["images"]
    anns = coco["annotations"]
    cats = sorted(coco["categories"], key=lambda c: c["id"])
    cat_id_to_cls = {c["id"]: i for i, c in enumerate(cats)}
    names = [c["name"] for c in cats]

    imgid_to_fname = {im["id"]: im["file_name"] for im in images}
    ann_by_img = {}
    for a in anns:
        ann_by_img.setdefault(a["image_id"], []).append(a)

    out = Path(OUT_DIR)
    img_tr, img_va = out/"images/train", out/"images/val"
    lab_tr, lab_va = out/"labels/train", out/"labels/val"
    for p in [img_tr, img_va, lab_tr, lab_va]:
        mkdir(p)

    labeled_imgs = [im for im in images if im["id"] in ann_by_img]
    random.shuffle(labeled_imgs)
    n_val = int(len(labeled_imgs) * VAL_RATIO)
    val_ids = set(im["id"] for im in labeled_imgs[:n_val])

    for im in tqdm(labeled_imgs, desc="Converting"):
        img_id = im["id"]
        fname = Path(imgid_to_fname[img_id]).name
        src_img = Path(IMAGES_DIR) / imgid_to_fname[img_id]
        if not src_img.exists():
            # sometimes COCO file_name is just basename
            src_img = Path(IMAGES_DIR) / fname
        if not src_img.exists():
            raise FileNotFoundError(f"Missing image: {imgid_to_fname[img_id]}")

        W, H = img_size(src_img)
        is_val = img_id in val_ids
        dst_img = (img_va if is_val else img_tr) / fname
        shutil.copy2(src_img, dst_img)

        label_lines = []
        for a in ann_by_img[img_id]:
            cls = cat_id_to_cls[a["category_id"]]
            kpts = a.get("keypoints", [])
            if not kpts:
                continue

            bbox = bbox_from_kpts(kpts, W, H)
            if bbox is None:
                continue
            xc, yc, bw, bh = bbox
            yolo_kpts = coco_kpts_to_yolo_triplets(kpts, W, H)

            parts = [str(cls), f"{xc:.6f}", f"{yc:.6f}", f"{bw:.6f}", f"{bh:.6f}"]
            parts += [f"{v:.6f}" for v in yolo_kpts]
            label_lines.append(" ".join(parts))

        dst_lab = (lab_va if is_val else lab_tr) / (Path(fname).stem + ".txt")
        dst_lab.write_text("\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8")

    # dataset.yaml
    yaml = out / "dataset.yaml"
    text = f"""path: {out.resolve()}
train: images/train
val: images/val
names:
"""
    for i, n in enumerate(names):
        text += f"  {i}: {n}\n"
    yaml.write_text(text, encoding="utf-8")

    print("Done.")
    print("Train with:")
    print(f"  yolo pose train model=yolo11s-pose.pt data={yaml} imgsz=1024")

if __name__ == "__main__":
    main()