import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
import mlflow
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from SeismicBuildingExposure.SeismicBuildingExposure.MLstructural_system.examples.pilot_regions.config import (
    LABEL, MLFLOW_TRACKING_URI, EXPERIMENT_FINAL, load_data
)

MODEL_TYPE = "LogisticRegression"
THIS_DIR   = Path(__file__).parent.resolve()
MODEL_PATH = THIS_DIR / f"model_final_{MODEL_TYPE}.pkl"


def main():
    model = joblib.load(MODEL_PATH)
    print(f"Loaded model from: {MODEL_PATH}")

    X_test, y_test = load_data(split="test")
    y_pred = model.predict(X_test)

    print("\n=== TEST SET EVALUATION ===\n")
    report = classification_report(y_test, y_pred)
    print(report)

    macro_f1 = f1_score(y_test, y_pred, average="macro")
    acc      = accuracy_score(y_test, y_pred)
    print(f"Macro-F1: {macro_f1:.4f}")
    print(f"Accuracy: {acc:.4f}")

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion matrix:\n", cm)

    cm_path     = THIS_DIR / "confusion_matrix_test.csv"
    report_path = THIS_DIR / "classification_report_test.txt"

    pd.DataFrame(cm).to_csv(cm_path, index=False)
    print(f"Confusion matrix saved at: {cm_path}")

    with open(report_path, "w") as f:
        f.write(report)
    print(f"Classification report saved at: {report_path}")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_FINAL)

    with mlflow.start_run(
        run_name=f"test_{MODEL_TYPE}",
        tags={"model_type": MODEL_TYPE, "stage": "test"}
    ):
        mlflow.log_param("model_type", MODEL_TYPE)
        mlflow.log_metric("test_macro_f1", macro_f1)
        mlflow.log_metric("test_accuracy", acc)
        mlflow.log_artifact(str(cm_path))
        mlflow.log_text(report, "classification_report_test.txt")

    print("Results logged to MLflow.")

if __name__ == "__main__":
    main()