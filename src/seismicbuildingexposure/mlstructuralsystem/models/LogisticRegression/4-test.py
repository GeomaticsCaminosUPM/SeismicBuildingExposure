import pandas as pd
import joblib
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(1, str(Path(__file__).resolve().parents[2]))

import config
import seismicbuildingexposure.mlstructuralsystem.dataset as dataset

MODEL_TYPE = "LogisticRegression"
THIS_DIR   = Path(__file__).parent.resolve()
MODEL_PATH = THIS_DIR / f"model_final_{MODEL_TYPE}.pkl"


def main():
    # Load trained model
    model = joblib.load(MODEL_PATH)
    print(f"Loaded model from: {MODEL_PATH}")

    # Load test data (no labels)
    X_test = dataset.load(split="test", cfg=config)

    # Ensure DataFrame format
    if not isinstance(X_test, pd.DataFrame):
        X_test = pd.DataFrame(X_test)

    print(f"Test data shape: {X_test.shape}")

    # Predict
    y_pred = model.predict(X_test)

    # Safety check
    if len(X_test) != len(y_pred):
        raise ValueError("Mismatch between test rows and predictions")

    # Add predictions
    X_test["predicted_label"] = y_pred

    # Save results
    dataset.save_test(X_test, model_name=MODEL_TYPE, cfg=config)


if __name__ == "__main__":
    main()