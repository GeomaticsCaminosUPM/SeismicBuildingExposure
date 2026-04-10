import pandas as pd
import numpy as np
import mlflow
import shutil
from sklearn.model_selection import StratifiedKFold, ParameterGrid
from sklearn.metrics import f1_score, accuracy_score
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
import joblib
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from SeismicBuildingExposure.SeismicBuildingExposure.MLstructural_system.examples.pilot_regions.config import (
    LABEL, RANDOM_STATE, N_SPLITS, MLFLOW_TRACKING_URI, EXPERIMENT_CV, load_data
)

MODEL_TYPE = "RandomForest"
THIS_DIR   = Path(__file__).parent.resolve()
MODELS_DIR = THIS_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)


def get_class_weights(y):
    counts = y.value_counts()
    return (counts.max() / counts).to_dict()

def build_rf(params, class_weight=None):
    correct_types = {}
    for key, value in params.items():
        if isinstance(value, str):
            try:
                if '.' not in value and value.lstrip('-').isdigit():
                    correct_types[key] = int(value)
                else:
                    correct_types[key] = float(value)
            except Exception:
                correct_types[key] = value
        else:
            correct_types[key] = value
    return RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight=class_weight,
        **correct_types
    )

def train_fold(X_train, y_train, X_val, y_val, model, use_smote):
    if use_smote:
        sm = SMOTE(random_state=RANDOM_STATE)
        X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
        X_train_res = pd.DataFrame(X_train_res, columns=X_train.columns)
    else:
        X_train_res, y_train_res = X_train, y_train
    model.fit(X_train_res, y_train_res)
    y_pred = model.predict(X_val)
    f1  = f1_score(y_val, y_pred, average="macro")
    acc = accuracy_score(y_val, y_pred)
    return model, y_pred, f1, acc

def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_CV)
    X, y = load_data(split="train")

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    EXPERIMENT_CONFIGS = {
        "baseline":     {"use_smote": False, "use_class_weight": False},
        "class_weight": {"use_smote": False, "use_class_weight": True},
        "smote":        {"use_smote": True,  "use_class_weight": False},
    }
    RF_GRID = {
        "max_depth":        [6, 12],
        "min_samples_split": [2, 8],
    }

    for config_name, cfg in EXPERIMENT_CONFIGS.items():
        for grid_params in ParameterGrid(RF_GRID):
            run_name = f"{MODEL_TYPE}_{config_name}_{grid_params}"
            with mlflow.start_run(run_name=run_name):
                mlflow.log_param("model_type", MODEL_TYPE)
                mlflow.log_params(grid_params)
                mlflow.log_param("strategy", config_name)

                f1_fold  = []
                acc_fold = []

                for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), 1):
                    X_train, X_val = X.iloc[tr_idx], X.iloc[val_idx]
                    y_train, y_val = y.iloc[tr_idx], y.iloc[val_idx]

                    if cfg["use_class_weight"]:
                        model = build_rf(grid_params, class_weight=get_class_weights(y_train))
                    else:
                        model = build_rf(grid_params)

                    model_trained, _, f1, acc = train_fold(
                        X_train, y_train, X_val, y_val,
                        model, use_smote=cfg["use_smote"]
                    )

                    fold_path = MODELS_DIR / f"rf_{config_name}_fold{fold}.pkl"
                    joblib.dump(model_trained, fold_path)
                    mlflow.log_artifact(str(fold_path), artifact_path=f"fold_{fold}_model")
                    mlflow.log_metric(f"f1_macro_fold_{fold}", f1)
                    mlflow.log_metric(f"accuracy_fold_{fold}", acc)
                    f1_fold.append(f1)
                    acc_fold.append(acc)

                mlflow.log_metric("mean_f1_macro", np.mean(f1_fold))
                mlflow.log_metric("std_f1_macro",  np.std(f1_fold))
                mlflow.log_metric("mean_acc",       np.mean(acc_fold))

                for fold in range(1, N_SPLITS + 1):
                    pth = MODELS_DIR / f"rf_{config_name}_fold{fold}.pkl"
                    if pth.exists():
                        pth.unlink()

    try:
        shutil.rmtree(MODELS_DIR)
        print(f"Removed {MODELS_DIR}")
    except Exception as e:
        print(f"Could not remove {MODELS_DIR}: {e}")

    print("Training complete.")

if __name__ == "__main__":
    main()