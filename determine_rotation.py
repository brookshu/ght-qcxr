from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
import os
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

# ============================================================
# CONFIG
# ============================================================

# Better clavicle detector
CLAVICLE_MODEL_PATH = (
    "/Users/brookshu/Documents/GitHub/ght-qcxr/ght-qcxr/clavicle_bbox_detector/runs/detect/train13-clavicle-bboxes/weights/best.pt"
)

# Second model: despite the old naming, treat it as a bbox detector too
SPINE_MODEL_PATH = (
    "/Users/brookshu/Documents/GitHub/ght-qcxr/ght-qcxr/keypoint_detector/runs/detect/train14/weights/best.pt"
)

IMAGE_PATH = (
    "/Users/brookshu/Documents/GitHub/ght-qcxr/ght-qcxr/clavicle_bbox_detector/dataset_yolo/images/val/100000AD.png"
)
FOLDER_PATH = (
    "/Users/brookshu/Documents/GitHub/ght-qcxr/ght-qcxr/all_png_output"
)
LABELS_FOLDER_PATH = (
    "/Users/brookshu/Documents/GitHub/ght-qcxr/ght-qcxr/rotation_detector/labels/rotation_labels.csv"
)

OUTPUT_PATH = "rotation_hybrid_bbox_debug.png"

# ------------------------------------------------------------
# CLASS IDS
# ------------------------------------------------------------
# Clavicle detector classes
CLAV_LEFT_ID = 0
CLAV_RIGHT_ID = 1
# if the clavicle model also predicts spine, we ignore it here

# Spine model classes
# CHANGE THIS if your second model uses a different class index for spine
SPINE_ID = 2

# ------------------------------------------------------------
# THRESHOLDS
# ------------------------------------------------------------
CLAVICLE_CONF = 0.10
SPINE_CONF = 0.10

# Tune this on your labeled set
ROTATION_SCORE_THRESH = 0.05 #0.16

# [[TN FP]
#  [FN TP]]

# ============================================================
# HELPERS
# ============================================================

def box_center_xyxy(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def filter_boxes_by_class(boxes_xyxy, confs, class_ids, target_class, conf_thresh):
    out = []
    for box, conf, cls in zip(boxes_xyxy, confs, class_ids):
        if int(cls) == target_class and float(conf) >= conf_thresh:
            out.append({
                "box": box,
                "conf": float(conf),
                "center": box_center_xyxy(box),
            })
    return out


def pick_best_detection(dets):
    if not dets:
        return None
    return max(dets, key=lambda d: d["conf"])


def estimate_spine_x_at_y(spine_dets, target_y):
    """
    Estimate spine x-position at the clavicle y-level.

    If multiple spine detections exist, fit x = a*y + b.
    If only one spine detection exists, use its center x.
    """
    if len(spine_dets) == 0:
        return None

    if len(spine_dets) == 1:
        return float(spine_dets[0]["center"][0])

    xs = np.array([d["center"][0] for d in spine_dets], dtype=float)
    ys = np.array([d["center"][1] for d in spine_dets], dtype=float)

    coeffs = np.polyfit(ys, xs, deg=1)  # x = a*y + b
    a, b = coeffs
    return float(a * target_y + b)


def compute_rotation(left_det, right_det, spine_dets):
    if left_det is None or right_det is None:
        return {
            "ok": False,
            "reason": "Missing left or right clavicle detection."
        }

    if len(spine_dets) == 0:
        return {
            "ok": False,
            "reason": "No spine detections."
        }

    left_x, left_y = left_det["center"]
    right_x, right_y = right_det["center"]

    clavicle_y = (left_y + right_y) / 2.0
    spine_x = estimate_spine_x_at_y(spine_dets, clavicle_y)

    if spine_x is None:
        return {
            "ok": False,
            "reason": "Could not estimate spine x at clavicle level."
        }

    d_left = abs(left_x - spine_x)
    d_right = abs(right_x - spine_x)

    if (d_left + d_right) <= 1e-8:
        return {
            "ok": False,
            "reason": "Degenerate left/right distances."
        }

    rotation_score = abs(d_left - d_right) / (d_left + d_right)
    rotated = rotation_score > ROTATION_SCORE_THRESH

    direction_hint = "none"
    if d_left > d_right:
        direction_hint = "left clavicle farther from spine midline"
    elif d_right > d_left:
        direction_hint = "right clavicle farther from spine midline"

    return {
        "ok": True,
        "left_center": (left_x, left_y),
        "right_center": (right_x, right_y),
        "spine_x_at_clavicle_y": spine_x,
        "clavicle_y": clavicle_y,
        "d_left": d_left,
        "d_right": d_right,
        "rotation_score": rotation_score,
        "rotated": rotated,
        "direction_hint": direction_hint,
    }


def draw_results(image, left_det, right_det, spine_dets, rotation_result, save_path):
    vis = image.copy()

    def draw_box(det, label, thickness=2):
        x1, y1, x2, y2 = map(int, det["box"])
        cx, cy = map(int, det["center"])
        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 255, 255), thickness)
        cv2.circle(vis, (cx, cy), 4, (255, 255, 255), -1)
        cv2.putText(
            vis,
            f"{label} {det['conf']:.2f}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    if left_det is not None:
        draw_box(left_det, "L clavicle", thickness=2)

    if right_det is not None:
        draw_box(right_det, "R clavicle", thickness=2)

    for i, det in enumerate(spine_dets):
        draw_box(det, f"Spine{i}", thickness=1)

    if rotation_result.get("ok"):
        spine_x = int(rotation_result["spine_x_at_clavicle_y"])
        clavicle_y = int(rotation_result["clavicle_y"])
        h, w = vis.shape[:2]

        cv2.line(vis, (spine_x, 0), (spine_x, h - 1), (255, 255, 255), 1)
        cv2.line(vis, (0, clavicle_y), (w - 1, clavicle_y), (180, 180, 180), 1)

        cv2.putText(
            vis,
            f"rotation_score={rotation_result['rotation_score']:.3f}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            vis,
            f"rotated={rotation_result['rotated']}",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            vis,
            rotation_result["direction_hint"],
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    cv2.imwrite(str(save_path), vis)


def run_detector(model_path, image_path, conf):
    model = YOLO(model_path)
    results = model.predict(
        source=image_path,
        conf=conf,
        save=False,
        verbose=False
    )

    result = results[0]

    if result.boxes is None or len(result.boxes) == 0:
        return [], [], []

    boxes_xyxy = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    class_ids = result.boxes.cls.cpu().numpy()
    return boxes_xyxy, confs, class_ids


# ============================================================
# MAIN
# ============================================================

def get_results(image_path=IMAGE_PATH):
    image = cv2.imread(image_path)
    if image is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    # --------------------------------------------------------
    # 1) Clavicle detector
    # --------------------------------------------------------
    c_boxes, c_confs, c_classes = run_detector(
        CLAVICLE_MODEL_PATH,
        image_path,
        CLAVICLE_CONF
    )

    left_dets = filter_boxes_by_class(
        c_boxes, c_confs, c_classes, CLAV_LEFT_ID, CLAVICLE_CONF
    )
    right_dets = filter_boxes_by_class(
        c_boxes, c_confs, c_classes, CLAV_RIGHT_ID, CLAVICLE_CONF
    )

    left_best = pick_best_detection(left_dets)
    right_best = pick_best_detection(right_dets)

    # --------------------------------------------------------
    # 2) Spine detector
    # --------------------------------------------------------
    s_boxes, s_confs, s_classes = run_detector(
        SPINE_MODEL_PATH,
        image_path,
        SPINE_CONF
    )

    spine_dets = filter_boxes_by_class(
        s_boxes, s_confs, s_classes, SPINE_ID, SPINE_CONF
    )

    # --------------------------------------------------------
    # 3) Compute rotation
    # --------------------------------------------------------
    rotation_result = compute_rotation(left_best, right_best, spine_dets)

    # --------------------------------------------------------
    # 4) Print summary
    # --------------------------------------------------------
    # print("\n=== HYBRID BBOX SUMMARY ===")
    # print(f"Image: {image_path}")
    # print(f"Left clavicle detections:  {len(left_dets)}")
    # print(f"Right clavicle detections: {len(right_dets)}")
    # print(f"Spine detections:          {len(spine_dets)}")

    # if left_best is not None:
    #     print(f"Best left clavicle center:  {left_best['center']} conf={left_best['conf']:.3f}")
    # if right_best is not None:
    #     print(f"Best right clavicle center: {right_best['center']} conf={right_best['conf']:.3f}")

    # for i, det in enumerate(spine_dets):
    #     print(f"Spine {i}: center={det['center']} conf={det['conf']:.3f}")

    # print("\n=== ROTATION RESULT ===")
    # for k, v in rotation_result.items():
    #     print(f"{k}: {v}")
    #     if k == "rotated":
    #         print(f"Rotation decision: {'ROTATED' if v else 'NOT ROTATED'}")
    #         print()
    #         print()
    #         print()
    # print(f"Rotation score: {rotation_result.get('rotated', 'No')}")

    # --------------------------------------------------------
    # 5) Save visualization
    # --------------------------------------------------------
    # draw_results(image, left_best, right_best, spine_dets, rotation_result, OUTPUT_PATH)
    # print(f"\nSaved debug image to: {Path(OUTPUT_PATH).resolve()}")
    return rotation_result.get('rotated', False)

def main():
    labels = pd.read_csv(LABELS_FOLDER_PATH)
    label_map = dict(zip(labels["path"], labels["label"]))
    y_true = []
    y_pred = []

    
    for i in os.listdir(FOLDER_PATH):
        if i.endswith(".png") or i.endswith(".jpg") or i.endswith(".jpeg"):
            img = os.path.join(FOLDER_PATH, i)
            label_path = os.path.join("cxr/", i.replace(".png", ".dcm"))
            label = label_map.get(label_path, "Unknown")
            #print(label_path, label_path in label_map)
            OUTPUT_PATH = f"debug_{i}"
            true = True if label==1 else False
            pred = get_results(image_path=img)
            y_true.append(true)
            y_pred.append(pred)
            #print(f"\n{img}:", pred, f"label={true}")
            #print(get_results(image_path=img))

    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    tn, fp, fn, tp = cm.ravel()
    #print(tn, fp, fn, tp)
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)

    plt.plot(fpr, tpr)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.show()

if __name__ == "__main__":
    main()