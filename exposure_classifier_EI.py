import os
import pydicom
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, roc_curve, roc_auc_score
from matplotlib.colors import ListedColormap

TOLERANCE_PCT = 0.20

# Northwestern inspired colors
PURPLE = "#4E2A84"
LIGHT_PURPLE = "#B6ACD1"
GRAY = "#D8D6D0"
DARK_GRAY = "#716C6B"

# -----------------------------
# Extract EI + Target EI
# -----------------------------
def extract_ei_target(dicom_folder):
    records = []

    for root, dirs, files in os.walk(dicom_folder):
        folder_name = os.path.basename(root).rstrip("^")

        for file in files:
            filepath = os.path.join(root, file)

            try:
                ds = pydicom.dcmread(filepath, stop_before_pixels=True, force=True)

                ei_tag = ds.get((0x0018, 0x1411))
                target_ei_tag = ds.get((0x0018, 0x1412))

                if ei_tag is not None and target_ei_tag is not None:
                    records.append({
                        "Record ID": folder_name,
                        "EI": float(ei_tag.value),
                        "Target_EI": float(target_ei_tag.value)
                    })

            except Exception:
                continue

    return pd.DataFrame(records)


# -----------------------------
# Paths
# -----------------------------
dicom_folder = "/Users/ilanadeutsch/Desktop/ght-qcxr/Tygerberg Data/TrenDx_CXRs_Renamed_new"
spreadsheet_path = "/Users/ilanadeutsch/Desktop/ght-qcxr/Tygerberg Data/TrENDx-CXRQuality_ForMartin.xlsx"
output_cm_png = "/Users/ilanadeutsch/Desktop/ght-qcxr/confusion_matrix_targetEI.png"
output_roc_png = "/Users/ilanadeutsch/Desktop/ght-qcxr/roc_curve_targetEI.png"


# -----------------------------
# Load Data
# -----------------------------
df_ei = extract_ei_target(dicom_folder)
df_labels = pd.read_excel(spreadsheet_path)

df_ei["Record ID"] = df_ei["Record ID"].astype(str).str.strip().str.upper()
df_labels["Record ID"] = df_labels["Record ID"].astype(str).str.strip().str.upper()

df = pd.merge(df_ei, df_labels, on="Record ID", how="inner")

# -----------------------------
# Ground Truth
# -----------------------------
under_col = "If X-ray is sub-optimal or unreadable, please give reason. (choice=Under-exposed)"
over_col = "If X-ray is sub-optimal or unreadable, please give reason. (choice=Over-exposed)"

df["y_true"] = (
    (df[under_col] == "Checked") |
    (df[over_col] == "Checked")
).astype(int)

# -----------------------------
# Continuous score for ROC
# -----------------------------
df["exposure_score"] = np.abs(df["EI"] - df["Target_EI"]) / df["Target_EI"]

# -----------------------------
# Prediction using Target EI ± tolerance
# -----------------------------
low = df["Target_EI"] * (1 - TOLERANCE_PCT)
high = df["Target_EI"] * (1 + TOLERANCE_PCT)

df["y_pred"] = ((df["EI"] < low) | (df["EI"] > high)).astype(int)

# -----------------------------
# Metrics at chosen threshold
# -----------------------------
cm = confusion_matrix(df["y_true"], df["y_pred"], labels=[0, 1])
tn, fp, fn, tp = cm.ravel()

f1 = f1_score(df["y_true"], df["y_pred"], zero_division=0)
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan

# -----------------------------
# ROC + AUC
# -----------------------------
fpr, tpr, thresholds = roc_curve(df["y_true"], df["exposure_score"])
auc = roc_auc_score(df["y_true"], df["exposure_score"])

print("\nTarget EI ±20% Evaluation")
print("--------------------------")
print(f"TN: {tn}")
print(f"FP: {fp}")
print(f"FN: {fn}")
print(f"TP: {tp}")

print(f"\nF1 Score: {f1:.3f}")
print(f"Sensitivity: {sensitivity:.3f}")
print(f"Specificity: {specificity:.3f}")
print(f"AUC: {auc:.3f}")

# -----------------------------
# Confusion Matrix Plot
# -----------------------------
color_mask = np.array([
    [0, 1],
    [1, 0]
])

cmap = ListedColormap([LIGHT_PURPLE, GRAY])

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
            fontweight="bold",
            color=PURPLE
        )

ax.set_xticks([0, 1])
ax.set_yticks([0, 1])

ax.set_xticklabels(
    ["Normal\nexposure", "Sub-optimal\nexposure"],
    fontsize=17,
    fontweight="semibold",
    color=PURPLE
)

ax.set_yticklabels(
    ["Normal\nexposure", "Sub-optimal\nexposure"],
    fontsize=17,
    fontweight="semibold",
    color=PURPLE
)

ax.set_xlabel("Predicted", fontsize=24, fontweight="bold", labelpad=20, color=PURPLE)
ax.set_ylabel("Ground Truth", fontsize=24, fontweight="bold", labelpad=20, color=PURPLE)

title = f"Exposure Classification\nTarget EI ±{int(TOLERANCE_PCT * 100)}%"
ax.set_title(title, fontsize=20, fontweight="bold", pad=25, color=PURPLE)

ax.set_xticks(np.arange(-0.5, 2, 1), minor=True)
ax.set_yticks(np.arange(-0.5, 2, 1), minor=True)
ax.grid(which="minor", color=PURPLE, linewidth=2)
ax.tick_params(which="minor", bottom=False, left=False)

metrics_text = (
    f"F1 Score: {f1:.3f}\n"
    f"Sensitivity: {sensitivity:.3f}\n"
    f"Specificity: {specificity:.3f}\n"
    f"AUC: {auc:.3f}"
)

fig.text(
    0.5,
    0.05,
    metrics_text,
    ha="center",
    fontsize=20,
    fontweight="bold",
    linespacing=1.7,
    color=PURPLE
)

plt.tight_layout(rect=[0.05, 0.12, 0.98, 0.95])
plt.savefig(output_cm_png, dpi=300, bbox_inches="tight")
plt.close()

print(f"\nSaved confusion matrix plot to: {output_cm_png}")

# -----------------------------
# ROC Curve Plot
# -----------------------------
fig, ax = plt.subplots(figsize=(8, 8))

# ROC line only in purple
ax.plot(fpr, tpr, color="#4E2A84", linewidth=3, label=f"AUC = {auc:.3f}")

# Random classifier reference
ax.plot([0, 1], [0, 1], linestyle=":", linewidth=2, color="gray")

# Axes back to standard ROC labels
ax.set_xlabel("False Positive Rate", fontsize=16, fontweight="bold")
ax.set_ylabel("True Positive Rate", fontsize=16, fontweight="bold")

ax.set_title(
    "ROC Curve for Exposure Classification",
    fontsize=18,
    fontweight="bold",
    pad=20
)

ax.grid(alpha=0.25)
ax.legend(fontsize=14, loc="lower right")

# Put only threshold-based metrics below
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