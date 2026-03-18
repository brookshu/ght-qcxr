import os
import pydicom
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from matplotlib.colors import ListedColormap

# -----------------------------
# 1) Extract projection from DICOMs
# -----------------------------
def extract_projection(dicom_folder):
    records = []
    total_files = 0
    dicom_read = 0
    ap_pa_found = 0

    for root, dirs, files in os.walk(dicom_folder):
        folder_name = os.path.basename(root).rstrip("^")

        for file in files:
            total_files += 1
            filepath = os.path.join(root, file)

            try:
                ds = pydicom.dcmread(filepath, stop_before_pixels=True, force=True)

                if "SOPClassUID" not in ds:
                    continue

                dicom_read += 1

                view_position = getattr(ds, "ViewPosition", None)

                if view_position is None:
                    continue

                view_position = str(view_position).strip().upper()

                if view_position in ["AP", "PA"]:
                    records.append({
                        "Record ID": folder_name,
                        "Projection": view_position
                    })
                    ap_pa_found += 1

            except Exception:
                continue

    df = pd.DataFrame(records)

    print("\nProjection Extraction Summary")
    print("-----------------------------")
    print(f"Total files scanned: {total_files}")
    print(f"DICOMs read: {dicom_read}")
    print(f"AP/PA projections found: {ap_pa_found}")

    return df


# -----------------------------
# 2) Paths
# -----------------------------
dicom_folder = "/Users/ilanadeutsch/Desktop/ght-qcxr/Tygerberg Data/TrenDx_CXRs_Renamed_new"
output_png = "/Users/ilanadeutsch/Desktop/ght-qcxr/projection_confusion_matrix.png"
output_roc_png = "/Users/ilanadeutsch/Desktop/ght-qcxr/projection_roc_curve.png"


# -----------------------------
# 3) Load projection data
# -----------------------------
df = extract_projection(dicom_folder)

df["Record ID"] = df["Record ID"].astype(str).str.strip().str.upper()
df = df.drop_duplicates(subset=["Record ID"]).copy()

# -----------------------------
# 4) Define truth and prediction
# -----------------------------
# Self check: both come from the same header field
# Positive class = PA
df["y_true"] = (df["Projection"] == "PA").astype(int)
df["y_pred"] = (df["Projection"] == "PA").astype(int)

# -----------------------------
# 5) Metrics
# -----------------------------
cm = confusion_matrix(df["y_true"], df["y_pred"], labels=[0, 1])
tn, fp, fn, tp = cm.ravel()

f1 = f1_score(df["y_true"], df["y_pred"], zero_division=0)
accuracy = accuracy_score(df["y_true"], df["y_pred"])
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
auc = 1.0

print("\nProjection Classification Results")
print("---------------------------------")
print(f"TN: {tn}")
print(f"FP: {fp}")
print(f"FN: {fn}")
print(f"TP: {tp}")
print(f"\nAccuracy: {accuracy:.3f}")
print(f"F1 Score: {f1:.3f}")
print(f"Sensitivity: {sensitivity:.3f}")
print(f"Specificity: {specificity:.3f}")
print(f"AUC: {auc:.3f}")

# -----------------------------
# 6) Plot confusion matrix
# -----------------------------
color_mask = np.array([
    [0, 1],
    [1, 0]
])

cmap = ListedColormap(["#B9E3B2", "#F3B0B0"])

fig, ax = plt.subplots(figsize=(8.5, 10))

ax.imshow(color_mask, cmap=cmap)

for i in range(2):
    for j in range(2):
        ax.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center",
            fontsize=34,
            fontweight="bold"
        )

ax.set_xticks([0, 1])
ax.set_yticks([0, 1])

ax.set_xticklabels(
    ["AP\nprojection", "PA\nprojection"],
    fontsize=17,
    fontweight="semibold"
)

ax.set_yticklabels(
    ["AP\nprojection", "PA\nprojection"],
    fontsize=17,
    fontweight="semibold"
)

ax.set_xlabel("Predicted", fontsize=24, fontweight="bold", labelpad=20)
ax.set_ylabel("Ground Truth", fontsize=24, fontweight="bold", labelpad=20)

ax.set_title(
    "Projection Classification\nRead from DICOM Header",
    fontsize=20,
    fontweight="bold",
    pad=25
)

ax.set_xticks(np.arange(-0.5, 2, 1), minor=True)
ax.set_yticks(np.arange(-0.5, 2, 1), minor=True)

ax.grid(which="minor", color="black", linewidth=2)
ax.tick_params(which="minor", bottom=False, left=False)

metrics_text = (
    f"Accuracy: {accuracy:.3f}\n"
    f"F1 Score: {f1:.3f}\n"
    f"Sensitivity: {sensitivity:.3f}\n"
    f"Specificity: {specificity:.3f}"
)

fig.text(
    0.5,
    0.05,
    metrics_text,
    ha="center",
    fontsize=20,
    fontweight="bold",
    linespacing=1.7
)

plt.tight_layout(rect=[0.05, 0.12, 0.98, 0.95])
plt.savefig(output_png, dpi=300)
plt.close()

print(f"\nSaved confusion matrix plot to: {output_png}")

# -----------------------------
# 7) Plot perfect ROC curve
# -----------------------------
# Since this is a self check using the same header field for truth and prediction,
# the ROC is perfect by definition.
fpr = np.array([0.0, 0.0, 1.0])
tpr = np.array([0.0, 1.0, 1.0])

fig, ax = plt.subplots(figsize=(8, 8))

ax.plot(fpr, tpr, color="#4E2A84", linewidth=3, label=f"AUC = {auc:.3f}")
ax.plot([0, 1], [0, 1], linestyle=":", linewidth=2, color="gray")

ax.set_xlim(0, 1)
ax.set_ylim(0, 1.02)

ax.set_xlabel("False Positive Rate", fontsize=16, fontweight="bold")
ax.set_ylabel("True Positive Rate", fontsize=16, fontweight="bold")
ax.set_title(
    "ROC Curve for Projection Classification",
    fontsize=18,
    fontweight="bold",
    pad=20
)

ax.grid(alpha=0.25)
ax.legend(fontsize=14, loc="lower right")

roc_metrics_text = (
    f"Sensitivity: {sensitivity:.3f}\n"
    f"Specificity: {specificity:.3f}\n"
    f"F1 Score: {f1:.3f}"
)

fig.text(
    0.5,
    0.005,
    roc_metrics_text,
    ha="center",
    va="bottom",
    fontsize=15,
    fontweight="bold",
    linespacing=1.6,
    color="black"
)

plt.tight_layout(rect=[0.05, 0.16, 0.98, 0.97])
plt.savefig(output_roc_png, dpi=300, bbox_inches="tight", facecolor="white")
plt.close()

print(f"Saved ROC plot to: {output_roc_png}")