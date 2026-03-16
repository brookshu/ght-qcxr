import xml.etree.ElementTree as ET
import os

xml_path = "annotations-2.xml"
output_dir = "labels"

os.makedirs(output_dir, exist_ok=True)

tree = ET.parse(xml_path)
root = tree.getroot()

class_map = {
    "Posterior RIbs": 0
}

for image in root.findall("image"):

    name = image.get("name")
    width = float(image.get("width"))
    height = float(image.get("height"))

    label_file = os.path.join(output_dir, name.replace(".png", ".txt"))

    with open(label_file, "w") as f:

        for poly in image.findall("polyline"):

            label = poly.get("label")
            cls = class_map[label]

            points = poly.get("points").split(";")

            coords = []

            for p in points:
                x, y = map(float, p.split(","))

                x /= width
                y /= height

                coords.append(f"{x} {y}")

            line = str(cls) + " " + " ".join(coords)
            f.write(line + "\n")