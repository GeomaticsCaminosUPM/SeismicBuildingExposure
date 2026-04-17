import pandas as pd
import joblib
from pathlib import Path
import sys

# Add project paths
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(1, str(Path(__file__).resolve().parents[2]))

import config
import seismicbuildingexposure.mlstructuralsystem.dataset as dataset

MODEL_TYPE = "SVC"
THIS_DIR   = Path(__file__).parent.resolve()
MODEL_PATH = THIS_DIR / f"model_final_{MODEL_TYPE}.pkl"


def main():
    # Load trained model
    model = joblib.load(MODEL_PATH)
    print(f"Loaded model from: {MODEL_PATH}")

    # Load test data (no labels)
    X_test = dataset.load(split="test", cfg=config)
    print(f"Test data shape: {X_test.shape}")

    # Ensure DataFrame format
    if not isinstance(X_test, pd.DataFrame):
        X_test = pd.DataFrame(X_test)

    # Predict
    y_pred = model.predict(X_test)

    # Sanity check
    if len(X_test) != len(y_pred):
        raise ValueError("Mismatch between input rows and predictions")

    # Add predictions
    X_test["predicted_label"] = y_pred

    # Optional: add probabilities if available
    if hasattr(model, "predict_proba"):
        try:
            probs = model.predict_proba(X_test)
            prob_df = pd.DataFrame(
                probs,
                columns=[f"prob_class_{i}" for i in range(probs.shape[1])]
            )
            X_test = pd.concat([X_test.reset_index(drop=True), prob_df], axis=1)
        except Exception:
            print("Warning: predict_proba not available for this SVC model")

    dataset.save_test(X_test,cfg=config)



if __name__ == "__main__":
    main()