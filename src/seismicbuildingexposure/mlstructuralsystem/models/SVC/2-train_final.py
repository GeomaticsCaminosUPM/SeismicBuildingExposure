import pandas as pd
import mlflow
import mlflow.sklearn
import joblib
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from SeismicBuildingExposure.SeismicBuildingExposure.MLstructural_system.examples.pilot_regions.config import (
    LABEL, RANDOM_STATE, MLFLOW_TRACKING_URI, EXPERIMENT_CV, load_data
)

MODEL_TYPE       = "SVC"
THIS_DIR         = Path(__file__).parent.resolve()
MODEL_FINAL_PATH = THIS_DIR / f"model_final_{MODEL_TYPE}.pkl"


def get_class_weights(y):
    counts = y.value_counts()
    return (counts.max() / counts).to_dict()

def safe_cast_param(param):
    if isinstance(param, str):
        try:
            if '.' not in param and param.lstrip('-').isdigit():
                return int(param)
            return float(param)
        except ValueError:
            return param
    return param

def get_best_run_and_params(experiment_name):
    runs = mlflow.search_runs(
        experiment_names=[experiment_name],
        filter_string="params.model_type = 'SVC' and attributes.status = 'FINISHED'",
        output_format="pandas"
    )
    if len(runs) == 0:
        raise RuntimeError(
            f"No completed SVC runs found in '{experiment_name}'. "
            "Check that 1-train_cv.py ran successfully."
        )

    metric_col = next((c for c in runs.columns if c.endswith("mean_f1_macro")), None)
    if metric_col is None:
        raise RuntimeError("No mean_f1_macro metric found. Check metric name in CV script.")

    best_run = runs.sort_values(metric_col, ascending=False).iloc[0]

    all_params = {k.replace("params.", ""): best_run[k] for k in best_run.index if k.startswith("params.")}
    svc_keys   = {"kernel", "C", "degree"}
    best_params = {k: safe_cast_param(v) for k, v in all_params.items() if k in svc_keys and v is not None}

    strategy         = best_run.get("params.strategy", "")
    use_smote        = strategy == "smote"
    use_class_weight = strategy == "class_weight"

    print(f"Best run id:               {best_run['run_id']}")
    print(f"Best strategy:             {strategy}")
    print(f"Best SVC params:           {best_params}")
    print(f"use_smote:                 {use_smote}")
    print(f"use_class_weight:          {use_class_weight}")

    return best_params, use_smote, use_class_weight, best_run["run_id"]

def build_svc(params, class_weight=None):
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
    return SVC(
        probability=True,
        random_state=RANDOM_STATE,
        class_weight=class_weight,
        **correct_types
    )

def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_CV)

    X, y = load_data(split="train")

    best_params, USE_SMOTE, USE_CLASS_WEIGHT, parent_run_id = get_best_run_and_params(EXPERIMENT_CV)

    if USE_SMOTE:
        sm = SMOTE(random_state=RANDOM_STATE)
        X_res, y_res = sm.fit_resample(X, y)
        X_res = pd.DataFrame(X_res, columns=X.columns)
    else:
        X_res, y_res = X, y

    class_weight = get_class_weights(y_res) if USE_CLASS_WEIGHT else None

    model = build_svc(best_params, class_weight=class_weight)
    model.fit(X_res, y_res)

    joblib.dump(model, MODEL_FINAL_PATH)
    print(f"Model saved at: {MODEL_FINAL_PATH}")

    with mlflow.start_run(
        run_name=f"final_model_{MODEL_TYPE}",
        tags={"model_type": MODEL_TYPE, "stage": "final"}
    ):
        mlflow.sklearn.log_model(model, name=f"model_final_{MODEL_TYPE}")
        mlflow.log_params(best_params)
        mlflow.log_param("use_smote", USE_SMOTE)
        mlflow.log_param("use_class_weight", USE_CLASS_WEIGHT)
        mlflow.log_param("parent_cv_run", parent_run_id)
        mlflow.log_param("model_type", MODEL_TYPE)
        mlflow.set_tag("parent_cv_run", parent_run_id)

    print("Final model trained and registered in MLflow.")

if __name__ == "__main__":
    main()