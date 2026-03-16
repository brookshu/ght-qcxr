from pathlib import Path
import cv2

IMG_DIR = Path("dataset_yolo/images/train")
LBL_DIR = Path("dataset_yolo/labels/train")

SAVE_DIR = Path("label_visualization")
SAVE_DIR.mkdir(exist_ok=True)

for img_path in IMG_DIR.glob("*.png"):

    label_path = LBL_DIR / (img_path.stem + ".txt")

    if not label_path.exists():
        continue

    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]

    with open(label_path) as f:
        for line in f.readlines():
            vals = list(map(float, line.strip().split()))

            cls = int(vals[0])
            xc, yc, bw, bh = vals[1:5]

            # bbox
            x1 = int((xc - bw/2) * w)
            y1 = int((yc - bh/2) * h)
            x2 = int((xc + bw/2) * w)
            y2 = int((yc + bh/2) * h)

            cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)

            # keypoints
            kpts = vals[5:]
            for i in range(0, len(kpts), 3):
                x = int(kpts[i] * w)
                y = int(kpts[i+1] * h)
                v = kpts[i+2]

                if v > 0:
                    cv2.circle(img, (x,y), 6, (0,0,255), -1)

    cv2.imwrite(str(SAVE_DIR / img_path.name), img)

print("Saved visualizations to", SAVE_DIR)