import os
import pydicom
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, roc_curve, roc_auc_score
from matplotlib.colors import ListedColormap

# -----------------------------
# 1) Extract EI from DICOMs
# -----------------------------
def extract_ei(dicom_folder):
    records = []
    total_files = 0
    read_files = 0

    for root, dirs, files in os.walk(dicom_folder):
        folder_name = os.path.basename(root).rstrip("^")

        for file in files:
            total_files += 1
            filepath = os.path.join(root, file)

            try:
                ds = pydicom.dcmread(filepath, stop_before_pixels=True, force=True)
                ei = getattr(ds, "ExposureIndex", np.nan)

                records.append({
                    "Record ID": folder_name,
                    "EI": ei
                })
                read_files += 1

            except Exception:
                continue

    df = pd.DataFrame(records)

    print(f"Total files scanned: {total_files}")
    print(f"Successfully read headers: {read_files}")

    return df


# -----------------------------
# 2) Paths
# -----------------------------
dicom_folder = "/Users/ilanadeutsch/Desktop/ght-qcxr/Tygerberg Data/TrenDx_CXRs_Renamed_new"
spreadsheet_path = "/Users/ilanadeutsch/Desktop/ght-qcxr/Tygerberg Data/TrENDx-CXRQuality_ForMartin.xlsx"
output_png = "/Users/ilanadeutsch/Desktop/ght-qcxr/confusion_matrix_best_thresholds.png"
output_roc_png = "/Users/ilanadeutsch/Desktop/ght-qcxr/roc_curve_best_thresholds.png"


# -----------------------------
# 3) Load + merge
# -----------------------------
df_ei = extract_ei(dicom_folder)
df_labels = pd.read_excel(spreadsheet_path)

df_ei["Record ID"] = df_ei["Record ID"].astype(str).str.strip().str.upper()
df_labels["Record ID"] = df_labels["Record ID"].astype(str).str.strip().str.upper()

df_merged = pd.merge(df_ei, df_labels, on="Record ID", how="inner")
df_merged = df_merged.dropna(subset=["EI"]).copy()

under_col = "If X-ray is sub-optimal or unreadable, please give reason. (choice=Under-exposed)"
over_col = "If X-ray is sub-optimal or unreadable, please give reason. (choice=Over-exposed)"

# -----------------------------
# 4) Define ground truth
# Positive class = sub-optimal exposure
# Here: under OR over exposed
# -----------------------------
df_merged["y_true"] = (
    (df_merged[under_col] == "Checked") |
    (df_merged[over_col] == "Checked")
).astype(int)

# -----------------------------
# 5) Search for best lower/upper thresholds
# Objective = maximize F1 score
# -----------------------------
ei_values = np.sort(df_merged["EI"].unique())

best_f1 = -1
best_lower = None
best_upper = None
best_cm = None
best_pred = None

for lower in ei_values:
    for upper in ei_values:
        if lower >= upper:
            continue

        y_pred = ((df_merged["EI"] < lower) | (df_merged["EI"] > upper)).astype(int)

        cm = confusion_matrix(df_merged["y_true"], y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        f1 = f1_score(df_merged["y_true"], y_pred, zero_division=0)

        if (
            f1 > best_f1 or
            (
                np.isclose(f1, best_f1) and best_cm is not None and (
                    tp > best_cm.ravel()[3] or
                    (tp == best_cm.ravel()[3] and fn < best_cm.ravel()[2])
                )
            )
        ):
            best_f1 = f1
            best_lower = lower
            best_upper = upper
            best_cm = cm
            best_pred = y_pred.copy()

# store best predictions
df_merged["y_pred"] = best_pred

tn, fp, fn, tp = best_cm.ravel()
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan

# -----------------------------
# 6) Build continuous score for ROC
# 0 inside normal interval, positive outside
# -----------------------------
df_merged["roc_score"] = np.where(
    df_merged["EI"] < best_lower,
    best_lower - df_merged["EI"],
    np.where(
        df_merged["EI"] > best_upper,
        df_merged["EI"] - best_upper,
        0
    )
)

# -----------------------------
# 7) ROC + AUC
# -----------------------------
auc = np.nan
fpr, tpr = None, None

if df_merged["y_true"].nunique() < 2:
    print("\nROC could not be computed because only one class is present.")
else:
    fpr, tpr, thresholds = roc_curve(df_merged["y_true"], df_merged["roc_score"])
    auc = roc_auc_score(df_merged["y_true"], df_merged["roc_score"])

# -----------------------------
# 8) Print results to terminal
# -----------------------------
print("\nBest thresholds found:")
print(f"Lower EI threshold: {best_lower}")
print(f"Upper EI threshold: {best_upper}")

print("\nBest confusion matrix:")
print(best_cm)

print("\nCounts:")
print(f"TN: {tn}")
print(f"FP: {fp}")
print(f"FN: {fn}")
print(f"TP: {tp}")

print("\nMetrics:")
print(f"F1 Score: {best_f1:.4f}")
print(f"Sensitivity: {sensitivity:.4f}")
print(f"Specificity: {specificity:.4f}")
if not np.isnan(auc):
    print(f"AUC: {auc:.4f}")

# -----------------------------
# 9) Build red/green confusion matrix image
# -----------------------------
color_mask = np.array([
    [0, 1],
    [1, 0]
])

cmap = ListedColormap(["#B9E3B2", "#F3B0B0"])

# -----------------------------
# 10) Plot confusion matrix
# -----------------------------
fig, ax = plt.subplots(figsize=(8.5, 10))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

ax.imshow(color_mask, cmap=cmap, vmin=0, vmax=1)

for i in range(2):
    for j in range(2):
        value = best_cm[i, j]
        ax.text(
            j, i, f"{value}",
            ha="center",
            va="center",
            fontsize=34,
            fontweight="bold",
            color="black"
        )

ax.set_xticks([0, 1])
ax.set_yticks([0, 1])

ax.set_xticklabels(
    ["Normal\nexposure", "Sub-optimal\nexposure"],
    fontsize=17,
    fontweight="semibold"
)
ax.set_yticklabels(
    ["Normal\nexposure", "Sub-optimal\nexposure"],
    fontsize=17,
    fontweight="semibold"
)

ax.set_xlabel("Predicted", fontsize=24, fontweight="bold", labelpad=22)
ax.set_ylabel("Ground Truth", fontsize=24, fontweight="bold", labelpad=28)

title_label = (
    "Sub-optimal Exposure Classification\n"
    f"[EI < {best_lower:.1f} or EI > {best_upper:.1f}]"
)
ax.set_title(title_label, fontsize=22, fontweight="bold", pad=28)

ax.set_xticks(np.arange(-0.5, 2, 1), minor=True)
ax.set_yticks(np.arange(-0.5, 2, 1), minor=True)
ax.grid(which="minor", color="black", linestyle="-", linewidth=2.2)
ax.tick_params(which="minor", bottom=False, left=False)

ax.tick_params(axis="both", which="major", length=0)
ax.tick_params(axis="x", pad=14)
ax.tick_params(axis="y", pad=18)

metrics_text = (
    f"F1 Score: {best_f1:.3f}\n"
    f"Sensitivity: {sensitivity:.3f}\n"
    f"Specificity: {specificity:.3f}"
)

fig.text(
    0.5, 0.09,
    metrics_text,
    ha="center",
    va="center",
    fontsize=20,
    fontweight="semibold",
    linespacing=1.8
)

plt.tight_layout(rect=[0.06, 0.14, 0.98, 0.93])
plt.savefig(output_png, dpi=300, bbox_inches="tight", facecolor="white")
plt.close()

print(f"\nSaved PNG to: {output_png}")

# -----------------------------
# 11) Plot ROC curve
# -----------------------------
if fpr is not None and tpr is not None:
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.plot(fpr, tpr, color="#4E2A84", linewidth=3, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle=":", linewidth=2, color="gray")

    ax.set_xlabel("False Positive Rate", fontsize=16, fontweight="bold")
    ax.set_ylabel("True Positive Rate", fontsize=16, fontweight="bold")
    ax.set_title("ROC Curve for Exposure Classification", fontsize=18, fontweight="bold", pad=20)

    ax.grid(alpha=0.25)
    ax.legend(fontsize=14, loc="lower right")

    roc_metrics_text = (
        f"Sensitivity: {sensitivity:.3f}\n"
        f"Specificity: {specificity:.3f}\n"
        f"F1 Score: {best_f1:.3f}"
    )

    fig.text(
        0.5,
        0.01,
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

    print(f"Saved ROC PNG to: {output_roc_png}")