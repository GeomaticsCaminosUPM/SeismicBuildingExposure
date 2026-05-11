import pandas as pd
import joblib
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(1, str(Path(__file__).resolve().parents[2]))

import config
import SeismicBuildingExposure.mlstructuralsystem.dataset as dataset

MODEL_TYPE = "XGBoost"
THIS_DIR   = Path(__file__).parent.resolve()
MODEL_PATH = THIS_DIR / f"model_final_{MODEL_TYPE}.pkl"

def main():
    # Load trained model
    model = joblib.load(MODEL_PATH)
    print(f"Loaded model from: {MODEL_PATH}")

    # Load test data WITHOUT labels
    # Make sure your dataset.load supports this!
    X_test = dataset.load(split="test", cfg=config)

    # If your loader returns (X, y), then do:
    # X_test, _ = dataset.load(split="test", cfg=config)

    print(f"Test data shape: {X_test.shape}")

    # Predict
    y_pred = model.predict(X_test)

    # Convert to DataFrame (if needed)
    if not isinstance(X_test, pd.DataFrame):
        X_test = pd.DataFrame(X_test)

    # Add predictions
    X_test["predicted_label"] = y_pred

    # Save results
    dataset.save_test(X_test, model_name=MODEL_TYPE, cfg=config)

if __name__ == "__main__":
    main()