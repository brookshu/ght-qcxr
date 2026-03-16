import os
import xml.etree.ElementTree as ET
from pathlib import Path

'''
converts cvat xml file with point annotations into YOLO format text files, one per image.
does not cut out any points 
'''



# -----------------------------
# CONFIG
# -----------------------------
XML_PATH = "cvat_export/annotations/annotations.xml"
OUTPUT_LABELS_DIR = "labels"

# map CVAT label names -> YOLO class ids
CLASS_MAP = {
    "left_clavicle": 0,
    "right_clavicle": 1,
    "spinous_process": 2,
}

# size of the box created around each point, in pixels
# increase a little if needed
BOX_SIZE = 12

# set True if you want to skip points whose labels are not in CLASS_MAP
SKIP_UNKNOWN_LABELS = True


# -----------------------------
# HELPERS
# -----------------------------
def clamp(val, low, high):
    return max(low, min(val, high))


def point_to_yolo_bbox(x, y, img_w, img_h, box_size):
    half = box_size / 2.0

    x1 = clamp(x - half, 0, img_w)
    y1 = clamp(y - half, 0, img_h)
    x2 = clamp(x + half, 0, img_w)
    y2 = clamp(y + half, 0, img_h)

    bw = x2 - x1
    bh = y2 - y1
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    # normalize for YOLO
    cx /= img_w
    cy /= img_h
    bw /= img_w
    bh /= img_h

    return cx, cy, bw, bh


def parse_points_string(points_str):
    """
    CVAT point string format is usually:
    'x,y' for one point
    or 'x1,y1;x2,y2;...' if multiple are packed together
    """
    pts = []
    for pair in points_str.strip().split(";"):
        if not pair:
            continue
        x_str, y_str = pair.split(",")
        pts.append((float(x_str), float(y_str)))
    return pts


# -----------------------------
# MAIN
# -----------------------------
def convert_cvat_points_xml_to_yolo():
    os.makedirs(OUTPUT_LABELS_DIR, exist_ok=True)

    tree = ET.parse(XML_PATH)
    root = tree.getroot()

    total_images = 0
    total_labels_written = 0
    total_points_seen = 0
    total_points_skipped = 0

    for image in root.findall(".//image"):
        total_images += 1

        image_name = image.attrib["name"]
        img_w = float(image.attrib["width"])
        img_h = float(image.attrib["height"])

        stem = Path(image_name).stem
        out_path = Path(OUTPUT_LABELS_DIR) / f"{stem}.txt"

        lines = []

        # collect ALL point annotations
        for points_tag in image.findall("points"):
            label_name = points_tag.attrib.get("label", "").strip()

            if label_name not in CLASS_MAP:
                if SKIP_UNKNOWN_LABELS:
                    total_points_skipped += 1
                    continue
                else:
                    raise ValueError(
                        f"Label '{label_name}' not found in CLASS_MAP for image '{image_name}'"
                    )

            class_id = CLASS_MAP[label_name]
            pts = parse_points_string(points_tag.attrib["points"])

            for x, y in pts:
                total_points_seen += 1
                cx, cy, bw, bh = point_to_yolo_bbox(x, y, img_w, img_h, BOX_SIZE)
                lines.append(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                total_labels_written += 1

        # also support points nested inside skeletons, if present
        for skeleton_tag in image.findall("skeleton"):
            for points_tag in skeleton_tag.findall("points"):
                label_name = points_tag.attrib.get("label", "").strip()

                if label_name not in CLASS_MAP:
                    if SKIP_UNKNOWN_LABELS:
                        total_points_skipped += 1
                        continue
                    else:
                        raise ValueError(
                            f"Label '{label_name}' not found in CLASS_MAP for image '{image_name}'"
                        )

                class_id = CLASS_MAP[label_name]
                pts = parse_points_string(points_tag.attrib["points"])

                for x, y in pts:
                    total_points_seen += 1
                    cx, cy, bw, bh = point_to_yolo_bbox(x, y, img_w, img_h, BOX_SIZE)
                    lines.append(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                    total_labels_written += 1

        with open(out_path, "w") as f:
            f.write("\n".join(lines))

    print("Done.")
    print(f"Images processed: {total_images}")
    print(f"Points seen: {total_points_seen}")
    print(f"Labels written: {total_labels_written}")
    print(f"Points skipped: {total_points_skipped}")


if __name__ == "__main__":
    convert_cvat_points_xml_to_yolo()