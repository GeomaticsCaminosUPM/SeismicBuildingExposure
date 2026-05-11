import pandas as pd
import mlflow
import mlflow.xgboost
import joblib
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(1, str(Path(__file__).resolve().parents[2]))

import config

import SeismicBuildingExposure.mlstructuralsystem.dataset as dataset 

MODEL_TYPE       = "XGBoost"
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
        filter_string="params.model_type = 'XGBoost' and attributes.status = 'FINISHED'",
        output_format="pandas"
    )
    if len(runs) == 0:
        raise RuntimeError(
            f"No completed XGBoost runs found in '{experiment_name}'. "
            "Check that 1-train_cv.py ran successfully."
        )

    metric_col = next((c for c in runs.columns if c.endswith("mean_f1_macro")), None)
    if metric_col is None:
        raise RuntimeError("No mean_f1_macro metric found. Check metric name in CV script.")

    best_run = runs.sort_values(metric_col, ascending=False).iloc[0]

    all_params = {k.replace("params.", ""): best_run[k] for k in best_run.index if k.startswith("params.")}
    xgb_keys   = {"max_depth", "learning_rate", "subsample", "colsample_bytree"}
    best_params = {k: safe_cast_param(v) for k, v in all_params.items() if k in xgb_keys and v is not None}

    strategy         = best_run.get("params.strategy", "")
    use_smote        = strategy == "smote"
    use_class_weight = strategy == "class_weight"

    print(f"Best run id:               {best_run['run_id']}")
    print(f"Best strategy:             {strategy}")
    print(f"Best XGBoost params:       {best_params}")
    print(f"use_smote:                 {use_smote}")
    print(f"use_class_weight:          {use_class_weight}")

    return best_params, use_smote, use_class_weight, best_run["run_id"]

def main():
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.EXPERIMENT_CV)

    X, y = dataset.load(split="train",cfg=config)
    N_CLASSES = y.nunique()

    best_params, USE_SMOTE, USE_CLASS_WEIGHT, parent_run_id = get_best_run_and_params(config.EXPERIMENT_CV)

    if USE_SMOTE:
        sm = SMOTE(random_state=config.RANDOM_STATE)
        X_res, y_res = sm.fit_resample(X, y)
        X_res = pd.DataFrame(X_res, columns=X.columns)
    else:
        X_res, y_res = X, y

    fit_kwargs = {}
    if USE_CLASS_WEIGHT:
        fit_kwargs["sample_weight"] = y_res.map(get_class_weights(y_res))

    model = XGBClassifier(
        objective="multi:softprob",
        num_class=N_CLASSES,
        eval_metric="mlogloss",
        n_estimators=2000,
        early_stopping_rounds=None,  # sin val set en entrenamiento final
        n_jobs=-1,
        random_state=config.RANDOM_STATE,
        **best_params
    )
    model.fit(X_res, y_res, **fit_kwargs)

    joblib.dump(model, MODEL_FINAL_PATH)
    print(f"Model saved at: {MODEL_FINAL_PATH}")

    with mlflow.start_run(
        run_name=f"final_model_{MODEL_TYPE}",
        tags={"model_type": MODEL_TYPE, "stage": "final"}
    ):
        mlflow.xgboost.log_model(model, name=f"model_final_{MODEL_TYPE}")
        mlflow.log_params(best_params)
        mlflow.log_param("use_smote", USE_SMOTE)
        mlflow.log_param("use_class_weight", USE_CLASS_WEIGHT)
        mlflow.log_param("parent_cv_run", parent_run_id)
        mlflow.log_param("model_type", MODEL_TYPE)
        mlflow.set_tag("parent_cv_run", parent_run_id)

    print("Final model trained and registered in MLflow.")

if __name__ == "__main__":
    main()