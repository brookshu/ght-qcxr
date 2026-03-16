import os
import csv
import cv2
import numpy as np
from ultralytics import YOLO
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

# -----------------------------
# USER SETTINGS
# -----------------------------
MODEL_PATH = "rib_detector/runs/segment/train4/weights/best.pt"
SOURCE_PATH = "all_png_output/"   # can be a single image or a folder
#SOURCE_PATH = "poor_insp/"
OUTPUT_DIR = "inspiration_results"

CONF = 0.25
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# Geometry filters
MIN_Y_FRAC = 0.08     # ignore very top of image
MAX_Y_FRAC = 0.82     # ignore very low ribs near diaphragm/abdomen
MIN_X_FRAC = 0.08     # ignore extreme left
MAX_X_FRAC = 0.92     # ignore extreme right

# Local patch radius around each sampling point
PATCH_RADIUS = 3

# Offset distance for sampling just above and below the rib
SAMPLE_OFFSET = 80

# Segmentation mask threshold
MASK_THRESH = 0.0

# Adequate inspiration threshold
ADEQUATE_RIB_COUNT = 10


# -----------------------------
# HELPERS
# -----------------------------
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def list_images(source_path):
    if os.path.isfile(source_path):
        return [source_path]

    image_paths = []
    for fname in sorted(os.listdir(source_path)):
        ext = os.path.splitext(fname)[1].lower()
        if ext in IMG_EXTS:
            image_paths.append(os.path.join(source_path, fname))
    return image_paths


def normalize_gray(gray):
    """Normalize grayscale image to 0-255 for more stable relative comparison."""
    gray = gray.astype(np.float32)
    mn, mx = gray.min(), gray.max()
    if mx <= mn:
        return np.zeros_like(gray, dtype=np.uint8)
    gray = (gray - mn) / (mx - mn) * 255.0
    return gray.astype(np.uint8)


def resize_mask_to_image(mask, image_shape):
    """Resize model mask to image size if needed."""
    h, w = image_shape[:2]
    if mask.shape[0] == h and mask.shape[1] == w:
        return mask
    resized = cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
    return resized


def get_ordered_rib_points(mask):
    """
    Convert a rib mask to an ordered set of representative points along x.
    We use the median y at each x column to get a rough centerline.
    """
    ys, xs = np.where(mask > 0)
    if len(xs) < 5: ## reject tiny masks with too few points
        return None 

    unique_xs = np.unique(xs)
    points = []

    for x in unique_xs:
        y_vals = ys[xs == x]
        if len(y_vals) == 0:
            continue
        y_med = int(np.median(y_vals))
        points.append((int(x), y_med))

    if len(points) < 10:
        return None

    points.sort(key=lambda p: p[0])
    return points


def pick_fifth_points(points):
    """
    Pick approximate left-fifth, middle-fifth, right-fifth positions
    along the rib centerline.
    """
    n = len(points)
    if n < 5:
        return None

    idxs = [
        int(0.20 * (n - 1)),
        int(0.50 * (n - 1)),
        int(0.80 * (n - 1)),
    ]
    return [points[i] for i in idxs]


def local_tangent(points, idx):
    """
    Estimate local tangent direction around point idx using neighboring points.
    Returns normalized tangent vector (tx, ty).
    """
    n = len(points)
    i0 = max(0, idx - 3)
    i1 = min(n - 1, idx + 3)

    x0, y0 = points[i0]
    x1, y1 = points[i1]

    dx = x1 - x0
    dy = y1 - y0
    norm = np.hypot(dx, dy)

    if norm < 1e-6:
        return (1.0, 0.0)

    return (dx / norm, dy / norm)


def get_patch_values(gray, cx, cy, r, exclude_mask=None):
    """
    Get pixel values from a square patch centered at (cx, cy).
    Optionally exclude pixels where exclude_mask > 0.
    """
    h, w = gray.shape[:2]
    x1 = max(0, cx - r)
    x2 = min(w, cx + r + 1)
    y1 = max(0, cy - r)
    y2 = min(h, cy + r + 1)

    patch = gray[y1:y2, x1:x2]
    if patch.size == 0:
        return np.array([])

    if exclude_mask is None:
        return patch.reshape(-1)

    mask_patch = exclude_mask[y1:y2, x1:x2]
    vals = patch[mask_patch == 0]
    return vals.reshape(-1)


def sample_background_for_point(gray, rib_mask, point, tangent, offset=SAMPLE_OFFSET, patch_radius=PATCH_RADIUS):
    """
    Sample background intensity slightly above and below the rib, along the normal direction.
    Excludes the rib mask itself from the patch.
    """
    x, y = point
    tx, ty = tangent

    # Normal vector
    nx, ny = -ty, tx

    # Two offset sample points
    p1x = int(round(x + offset * nx))
    p1y = int(round(y + offset * ny))
    p2x = int(round(x - offset * nx))
    p2y = int(round(y - offset * ny))

    vals1 = get_patch_values(gray, p1x, p1y, patch_radius, exclude_mask=rib_mask)
    vals2 = get_patch_values(gray, p2x, p2y, patch_radius, exclude_mask=rib_mask)

    vals = np.concatenate([vals1, vals2]) if (len(vals1) + len(vals2)) > 0 else np.array([])

    if len(vals) == 0:
        return None

    return float(np.median(vals))


def rib_geometry_ok(points, image_shape):
    """
    Reject ribs too close to image borders or too low/high.
    Uses rib centroid.
    """
    h, w = image_shape[:2]
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    cx = float(np.median(xs))
    cy = float(np.median(ys))

    if not (MIN_X_FRAC * w <= cx <= MAX_X_FRAC * w):
        return False
    if not (MIN_Y_FRAC * h <= cy <= MAX_Y_FRAC * h):
        return False

    return True


def compute_rib_score(gray, rib_mask):
    """
    For one rib mask:
    - extract ordered centerline points
    - sample background at left / middle / right
    - return per-rib score info
    """
    points = get_ordered_rib_points(rib_mask)
    if points is None:
        return None

    # if not rib_geometry_ok(points, gray.shape):
    #     return None

    chosen_points = pick_fifth_points(points)
    if chosen_points is None:
        return None

    # Find approximate indices of chosen points in full point list
    point_to_idx = {p: i for i, p in enumerate(points)}

    sample_scores = []
    sample_meta = []

    for pt in chosen_points:
        idx = point_to_idx.get(pt, None)
        if idx is None:
            continue

        tangent = local_tangent(points, idx)
        bg_score = sample_background_for_point(gray, rib_mask, pt, tangent)

        if bg_score is not None:
            sample_scores.append(bg_score)
            sample_meta.append((pt[0], pt[1], bg_score))

    if len(sample_scores) < 2:
        return None

    rib_score = float(np.median(sample_scores))

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    cx = int(np.median(xs))
    cy = int(np.median(ys))

    return {
        "centerline_points": points,
        "sample_points": sample_meta,   # (x, y, score)
        "sample_scores": sample_scores,
        "rib_score": rib_score,
        "centroid": (cx, cy),
    }


def classify_ribs_relative(rib_infos):
    """
    Classify ribs using relative brightness within the same image.
    Darker surrounding background -> more likely within lung field.

    Rule:
    - compute reference = median rib_score across ribs in image
    - accept rib if at least 2 of its 3 local samples are <= reference
    """
    if len(rib_infos) == 0:
        return [], None

    all_scores = [r["rib_score"] for r in rib_infos]
    #reference = float(np.median(all_scores))
    reference = float(np.percentile(all_scores, 99))

    accepted = []
    for rib in rib_infos:
        darker_count = sum(score <= reference for score in rib["sample_scores"])
        rib["reference_score"] = reference
        rib["accepted"] = darker_count >= 2
        rib["darker_count"] = darker_count
        accepted.append(rib)

    return accepted, reference


def draw_results(image_bgr, rib_infos, rib_count, adequate, save_path):
    vis = image_bgr.copy()

    for rib in rib_infos:
        cx, cy = rib["centroid"]
        accepted = rib.get("accepted", False)

        color = (0, 255, 0) if accepted else (0, 0, 255)

        # Draw centroid
        cv2.circle(vis, (cx, cy), 8, color, -1)

        # Draw centerline lightly
        pts = rib["centerline_points"]
        for i in range(1, len(pts)):
            cv2.line(vis, pts[i - 1], pts[i], color, 1)

        # Draw sample points
        for sx, sy, score in rib["sample_points"]:
            cv2.circle(vis, (int(sx), int(sy)), 5, (255, 255, 0), -1)
            cv2.putText(
                vis,
                f"{int(round(score))}",
                (int(sx) + 4, int(sy) - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 0),
                1,
                cv2.LINE_AA,
            )

    label = f"Accepted ribs: {rib_count} | Adequate: {adequate}"
    cv2.putText(
        vis,
        label,
        (25, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0) if adequate else (0, 0, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.imwrite(save_path, vis)


def process_image(model, image_path, output_dir):
    image_name = os.path.basename(image_path)

    bgr = cv2.imread(image_path)
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if bgr is None or gray is None:
        print(f"Could not read {image_path}")
        return None

    gray = normalize_gray(gray)
    h, w = gray.shape[:2]

    results = model.predict(
        source=image_path,
        conf=CONF,
        save=False,
        verbose=False,
    )

    result = results[0]

    rib_infos = []

    if result.masks is not None and result.masks.data is not None:
        print(f"  Found {len(result.masks.data)} rib candidates.")
        masks = result.masks.data.cpu().numpy()

        for mask in masks:
            mask = resize_mask_to_image(mask, gray.shape)
            rib_mask = (mask > MASK_THRESH).astype(np.uint8)

            if rib_mask.sum() == 0:
                continue

            info = compute_rib_score(gray, rib_mask)
            if info is not None:
                rib_infos.append(info)

    rib_infos, reference = classify_ribs_relative(rib_infos)

    accepted_ribs = [r for r in rib_infos if r.get("accepted", False)]
    rib_count = len(accepted_ribs)
    adequate = rib_count >= ADEQUATE_RIB_COUNT

    save_path = os.path.join(output_dir, image_name)
    draw_results(bgr, rib_infos, rib_count, adequate, save_path)

    return {
        "image_name": image_name,
        "num_predicted_ribs_used": len(rib_infos),
        "accepted_rib_count": rib_count,
        "adequate_inspiration": int(adequate),
        "reference_score": "" if reference is None else round(reference, 2),
    }


# -----------------------------
# MAIN
# -----------------------------
def main():
    ensure_dir(OUTPUT_DIR)

    model = YOLO(MODEL_PATH)
    image_paths = list_images(SOURCE_PATH)

    if len(image_paths) == 0:
        print("No images found.")
        return

    csv_path = os.path.join(OUTPUT_DIR, "inspiration_results.csv")
    rows = []

    for i, image_path in enumerate(image_paths, 1):
        print(f"[{i}/{len(image_paths)}] Processing {os.path.basename(image_path)}")
        row = process_image(model, image_path, OUTPUT_DIR)
        if row is not None:
            rows.append(row)
    adequate = 0
    inadequate = 0
    inadequate_images = []
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image_name",
                "num_predicted_ribs_used",
                "accepted_rib_count",
                "adequate_inspiration",
                "reference_score",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
        for r in rows:
            if r["adequate_inspiration"]:
                adequate += 1
            else:
                inadequate += 1
                inadequate_images.append(r["image_name"])

    y_true = [1 for r in rows]
    y_pred = [r["adequate_inspiration"] for r in rows]
    print(y_true)
    print(y_pred)
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)

    plt.plot(fpr, tpr)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.show()

    print(f"\nDone. Results saved to: {OUTPUT_DIR}")
    print(f"CSV: {csv_path}")
    print(f"Adequate: {adequate}")
    print(f"Inadequate: {inadequate}")
    print(f"Inadequate Images: {inadequate_images}")


if __name__ == "__main__":
    main()