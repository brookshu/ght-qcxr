import json, random, shutil
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# --------- CONFIG ----------
COCO_JSON = Path("cvat_export/annotations/person_keypoints_default.json")
EXPORT_ROOT = Path("cvat_export")   # where your unzipped export lives
OUT_DIR = Path("dataset_yolo")
VAL_RATIO = 0.2
SEED = 42
PAD_FRAC = 0.02  # bbox padding around keypoints
# ---------------------------

random.seed(SEED)

def mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def img_size(p: Path):
    with Image.open(p) as im:
        return im.size  # (W, H)

def build_image_index(root: Path):
    """Index all png/jpg/jpeg under export root by basename."""
    idx = {}
    exts = {".png", ".jpg", ".jpeg"}
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            idx.setdefault(p.name, p)
    return idx

def coco_kpts_to_yolo_triplets(kpts, W, H):
    # COCO: [x,y,v]*K where v: 0=not labeled, 1=occluded, 2=visible
    out = []
    for i in range(0, len(kpts), 3):
        x, y, v = kpts[i], kpts[i+1], kpts[i+2]
        vis = 1.0 if v and v > 0 else 0.0
        xn = min(max(float(x) / W, 0.0), 1.0)
        yn = min(max(float(y) / H, 0.0), 1.0)
        out.extend([xn, yn, vis])
    return out

def bbox_from_kpts(kpts, W, H):
    xs, ys = [], []
    for i in range(0, len(kpts), 3):
        x, y, v = kpts[i], kpts[i+1], kpts[i+2]
        if v and v > 0:
            xs.append(x); ys.append(y)
    if not xs:
        return None
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
    return (xc / W, yc / H, bw / W, bh / H)

def main():
    print("COCO_JSON:", COCO_JSON.resolve())
    print("EXPORT_ROOT:", EXPORT_ROOT.resolve())

    if not COCO_JSON.exists():
        print("\n❌ COCO JSON not found.")
        print("Expected:", COCO_JSON.resolve())
        return

    coco = json.loads(COCO_JSON.read_text(encoding="utf-8"))
    images = coco.get("images", [])
    anns = coco.get("annotations", [])
    cats = sorted(coco.get("categories", []), key=lambda c: c.get("id", 0))

    print(f"\nCOCO summary: images={len(images)} annotations={len(anns)} categories={len(cats)}")

    if len(images) == 0:
        print("\n❌ No images in COCO JSON. Export likely failed or wrong format.")
        return
    if len(anns) == 0:
        print("\n❌ No annotations in COCO JSON. You probably exported without annotations, or nothing is labeled.")
        return

    # Check keypoints presence
    kp_anns = [a for a in anns if isinstance(a.get("keypoints", None), list) and len(a.get("keypoints", [])) > 0]
    print(f"Annotations with keypoints: {len(kp_anns)} / {len(anns)}")
    if len(kp_anns) == 0:
        print("\n❌ No keypoints found in annotations.")
        print("Make sure you exported **COCO Keypoints**, not COCO Instances/Detection.")
        return

    # Show sample file names
    print("\nFirst 5 COCO image file_name entries:")
    for im in images[:5]:
        print("  -", im.get("file_name"))

    # Index actual files
    img_index = build_image_index(EXPORT_ROOT)
    print(f"\nFound {len(img_index)} image files under EXPORT_ROOT (recursively).")
    if len(img_index) == 0:
        print("\n❌ No images were found inside your export folder.")
        print("Your CVAT export likely produced only annotations/manifest without images.")
        return

    # Determine how many COCO file_names can be resolved
    resolvable = 0
    not_found_examples = []
    for im in images[:200]:  # sample first 200
        fn = im.get("file_name", "")
        base = Path(fn).name
        if (EXPORT_ROOT / fn).exists() or base in img_index:
            resolvable += 1
        else:
            if len(not_found_examples) < 5:
                not_found_examples.append(fn)

    print(f"Resolvable COCO images (sample up to 200): {resolvable} / {min(200, len(images))}")
    if resolvable == 0:
        print("\n❌ None of the COCO file_name paths match actual files.")
        print("Examples of missing file_name entries:")
        for ex in not_found_examples:
            print("  -", ex)
        print("\nThis means your images are somewhere else, or names changed.")
        return

    # category mapping
    cat_id_to_cls = {c["id"]: i for i, c in enumerate(cats)}
    names = [c["name"] for c in cats] if cats else ["object"]

    # group annotations by image_id
    ann_by_img = {}
    for a in kp_anns:
        ann_by_img.setdefault(a["image_id"], []).append(a)

    labeled_imgs = [im for im in images if im["id"] in ann_by_img]
    print(f"\nImages with at least 1 keypoint-annotation: {len(labeled_imgs)} / {len(images)}")

    if len(labeled_imgs) == 0:
        print("\n❌ No images have keypoint annotations linked to them.")
        print("This usually happens if points were created but not saved as annotations tied to images/instances.")
        return

    # output dirs
    img_tr, img_va = OUT_DIR / "images/train", OUT_DIR / "images/val"
    lab_tr, lab_va = OUT_DIR / "labels/train", OUT_DIR / "labels/val"
    for p in [img_tr, img_va, lab_tr, lab_va]:
        mkdir(p)

    random.shuffle(labeled_imgs)
    n_val = int(len(labeled_imgs) * VAL_RATIO)
    val_ids = set(im["id"] for im in labeled_imgs[:n_val])

    copied = 0
    written = 0
    skipped_missing_img = 0
    skipped_no_bbox = 0

    for im in tqdm(labeled_imgs, desc="Converting"):
        img_id = im["id"]
        fn = im.get("file_name", "")
        base = Path(fn).name

        src = (EXPORT_ROOT / fn)
        if not src.exists():
            src = img_index.get(base, None)
        if src is None or not src.exists():
            skipped_missing_img += 1
            continue

        W, H = img_size(src)
        is_val = img_id in val_ids
        dst_img = (img_va if is_val else img_tr) / base
        shutil.copy2(src, dst_img)
        copied += 1

        lines = []
        for a in ann_by_img[img_id]:
            cls = cat_id_to_cls.get(a["category_id"], 0)
            kpts = a.get("keypoints", [])
            bbox = bbox_from_kpts(kpts, W, H)
            if bbox is None:
                skipped_no_bbox += 1
                continue
            xc, yc, bw, bh = bbox
            yolo_kpts = coco_kpts_to_yolo_triplets(kpts, W, H)

            parts = [str(cls), f"{xc:.6f}", f"{yc:.6f}", f"{bw:.6f}", f"{bh:.6f}"]
            parts += [f"{v:.6f}" for v in yolo_kpts]
            lines.append(" ".join(parts))

        dst_lab = (lab_va if is_val else lab_tr) / (Path(base).stem + ".txt")
        dst_lab.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        written += 1

    # dataset.yaml
    yaml_path = OUT_DIR / "dataset.yaml"
    text = f"""path: {OUT_DIR.resolve()}
train: images/train
val: images/val
names:
"""
    for i, n in enumerate(names):
        text += f"  {i}: {n}\n"
    yaml_path.write_text(text, encoding="utf-8")

    print("\n✅ Conversion finished")
    print("Images copied:", copied)
    print("Label files written:", written)
    print("Skipped (missing image):", skipped_missing_img)
    print("Skipped (no labeled kpts -> bbox):", skipped_no_bbox)
    print("YAML:", yaml_path.resolve())

    if copied == 0:
        print("\n❌ Still copied 0 images. The printed diagnostics above will tell you why.")
    else:
        print("\nTrain with:")
        print(f"  yolo pose train model=yolo11s-pose.pt data={yaml_path} imgsz=1024")

if __name__ == "__main__":
    main()