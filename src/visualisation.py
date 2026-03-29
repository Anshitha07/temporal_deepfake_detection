import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

# =========================
CSV_PATH = "results/cross_dataset_predictions.csv"
# =========================

df = pd.read_csv(CSV_PATH)

# Fix #7: use the ground_truth column if available, else fall back to filename prefix
if "ground_truth" in df.columns:
    labels = df["ground_truth"].values
else:
    def get_label(name):
        name = name.lower()
        if name.startswith("real"):
            return 0
        else:
            return 1

    df["ground_truth"] = df["video"].apply(get_label)
    labels = df["ground_truth"].values

preds = df["prediction"].values
probs = df["confidence"].values

accuracy = (labels == preds).mean() * 100
print(f"\nAccuracy: {accuracy:.2f}%")

os.makedirs("results", exist_ok=True)

# =========================
# Confusion Matrix
# =========================

cm = confusion_matrix(labels, preds)

plt.figure(figsize=(5,5))
sns.heatmap(cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Real","Fake"],
            yticklabels=["Real","Fake"])

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.savefig("results/confusion_matrix.png")
plt.show()

# =========================
# ROC Curve
# =========================

fpr, tpr, _ = roc_curve(labels, probs)
roc_auc = auc(fpr, tpr)

plt.figure()

plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0,1],[0,1],'k--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()

plt.savefig("results/roc_curve.png")
plt.show()

# =========================
# Confidence Histogram
# =========================

plt.figure()

plt.hist(probs, bins=20)

plt.xlabel("Fake Probability")
plt.ylabel("Count")
plt.title("Prediction Confidence Distribution")

plt.savefig("results/confidence_histogram.png")
plt.show()

print("\nPlots saved to results/")