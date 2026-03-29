import csv
import os


def log_results_csv(csv_path, model_name, metrics):
    file_exists = os.path.isfile(csv_path)

    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow([
                "Model",
                "Accuracy",
                "ROC_AUC",
                "Precision",
                "Recall",
                "F1"
            ])

        writer.writerow([
            model_name,
            metrics["accuracy"],
            metrics["roc_auc"],
            metrics["precision"],
            metrics["recall"],
            metrics["f1"]
        ])
