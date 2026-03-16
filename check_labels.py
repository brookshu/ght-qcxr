from pathlib import Path

root = Path("/Users/brookshu/Documents/GitHub/ght-qcxr/ght-qcxr/dataset_yolo")

for split in ["train", "val"]:
    img_dir = root / "images" / split
    lbl_dir = root / "labels" / split

    img_stems = {p.stem for p in img_dir.glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}}
    lbl_stems = {p.stem for p in lbl_dir.glob("*.txt")}

    print(f"\n--- {split} ---")
    print("images:", len(img_stems))
    print("labels:", len(lbl_stems))
    print("missing labels:", len(img_stems - lbl_stems))
    print("missing images:", len(lbl_stems - img_stems))
    print("example missing labels:", list(sorted(img_stems - lbl_stems))[:10])
    print("example missing images:", list(sorted(lbl_stems - img_stems))[:10])