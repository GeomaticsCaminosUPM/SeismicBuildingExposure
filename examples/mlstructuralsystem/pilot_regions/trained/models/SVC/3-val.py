import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
import mlflow
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(1, str(Path(__file__).resolve().parents[2]))

import config

import SeismicBuildingExposure.mlstructuralsystem.dataset as dataset 

MODEL_TYPE = "SVC"
THIS_DIR   = Path(__file__).parent.resolve()
MODEL_PATH = THIS_DIR / f"model_final_{MODEL_TYPE}.pkl"


def main():
    model = joblib.load(MODEL_PATH)
    print(f"Loaded model from: {MODEL_PATH}")

    X_val, y_val = dataset.load(split="val",cfg=config)
    y_pred = model.predict(X_val)

    print("\n=== VAL SET EVALUATION ===\n")
    report = classification_report(y_val, y_pred)
    print(report)

    macro_f1 = f1_score(y_val, y_pred, average="macro")
    acc      = accuracy_score(y_val, y_pred)
    print(f"Macro-F1: {macro_f1:.4f}")
    print(f"Accuracy: {acc:.4f}")

    cm = confusion_matrix(y_val, y_pred)
    print("Confusion matrix:\n", cm)

    cm_path     = THIS_DIR / "confusion_matrix_val.csv"
    report_path = THIS_DIR / "classification_report_val.txt"

    pd.DataFrame(cm).to_csv(cm_path, index=False)
    print(f"Confusion matrix saved at: {cm_path}")

    with open(report_path, "w") as f:
        f.write(report)
    print(f"Classification report saved at: {report_path}")

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.EXPERIMENT_FINAL)

    with mlflow.start_run(
        run_name=f"val_{MODEL_TYPE}",
        tags={"model_type": MODEL_TYPE, "stage": "val"}
    ):
        mlflow.log_param("model_type", MODEL_TYPE)
        mlflow.log_metric("val_macro_f1", macro_f1)
        mlflow.log_metric("val_accuracy", acc)
        mlflow.log_artifact(str(cm_path))
        mlflow.log_text(report, "classification_report_val.txt")

    print("Results logged to MLflow.")

if __name__ == "__main__":
    main()