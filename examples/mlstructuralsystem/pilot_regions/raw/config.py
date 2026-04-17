import os
import pandas as pd
from typing import Literal
from pathlib import Path

# ── Project root ──────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.resolve()

# ── Dataset paths ──────────────────────────────────────────────────────────────────
TRAIN_PATH = PROJECT_ROOT / "dataset" / "raw_data" # File or folder
VAL_PATH  = None # File or folder or None
VAL_SIZE = 0.2 # If VAL_PATH is None then split train dataset into train/val with val_size

TEST_PATH = "VAL" # "VAL" - Use the same dataset as VAL as test so that you can see the output dataframe

CLEANED_DATASET_PATH = PROJECT_ROOT / "dataset" / "cleaned"
PREPROCESSED_OUTPUT_PATH = PROJECT_ROOT / "dataset"
TEST_OUTPUT_PATH = PROJECT_ROOT / "dataset"

# ── Reproducibility ────────────────────────────────────────────────────────
RANDOM_STATE = 42
N_SPLITS     = 5

# ============================================================
# Dataset footprint code
# ============================================================

POSITION_BUFFER = 0.15
POSITION_MIN_RADIUS = 0.5
POSITION_MIN_ANGULAR_ACC = 2.133
POSITION_MIN_CONFINEMENT = 1
POSITION_MIN_ANGLE = 0.78
POSITION_MIN_FORCE = 0.1666

GNDT_MIN_LENGTH = 0
GNDT_MIN_AREA = 0

FSI_ECCENTRICITY_COL = "EC8_eccentricity_ratio"
FSI_ECCENTRICITY_VAL = 0.3

FSI_SETBACK_COL = "GNDT_setback_ratio"
FSI_SETBACK_VAL = 0.3

FSI_SLENDERNESS_COL = "slenderness_inertia"
FSI_SLENDERNESS_VAL = 4 

# ============================================================
# Dataset config
# ============================================================

# ── Columns ─────────────────────────────────────────────────────────────────

STRATIFY_COLUMN = "city"
MIN_AREA = 4

# Dictionary for categorical features and all the valid values
CATEGORICAL_FEATURES = {
    "relative_position":["confined","corner","isolated","lateral","torque"],
    "fsi":["eccentricity","regular","setbacks","slenderness"],
    "roof":["asphalt","bright_concrete","dark_shadow","metallic"],
}

# Dictionary for ordinal features and their explicit order.
# This ensures that features with a natural order (e.g., low, medium, high)
# are encoded in a way that the model can understand their relationship.
ORDINAL_FEATURES = {
    'code_quality': ['pre_code', 'low_code', 'medium_code', 'high_code'],
}


FEATURES = [
    "height",
    "area",
    "perimeter",
    "code_quality",
    "EC8_eccentricity_ratio",
    "EC8_radius_ratio",
    "EC8_compactness",
    "CR_eccentricity_ratio",
    "ASCE7_parallelity_angle",
    "ASCE7_setback_ratio",
    "ASCE7_hole_ratio",
    "NTC_setback_ratio",
    "NTC_hole_ratio",
    "GNDT_main_shape_slenderness",
    "GNDT_setback_ratio",
    "GNDT_eccentricity_ratio",
    "GNDT_setback_slenderness",
    "slenderness_elevation",
    "slenderness_inertia",
    "slenderness_bbox",
    "slenderness_circunscribed",
    "inertia_vs_circle",
    "regularity_boolean",
    "fsi",
    "contact_force",
    "contact_confinement_ratio",
    "contact_angular_acc",
    "contact_angle",
    "relative_position",
    "roof",
]

# The name of the target variable
LABEL = "simplified_structural_system"
LABEL_VALUES = ["M","CR","W","ADO"]

# ── MLflow ─────────────────────────────────────────────────────────────────
MLFLOW_DB_PATH      = PROJECT_ROOT / "mlflow.db"
MLFLOW_TRACKING_URI = os.environ.get(
    "MLFLOW_TRACKING_URI",
    f"sqlite:///{MLFLOW_DB_PATH}"
)
EXPERIMENT_CV    = "multiclass_model_selection"
EXPERIMENT_FINAL = "final_evaluation"

# ============================================================
# Model config
# ============================================================

# Models to run (in order)
MODELS = [
    "LogisticRegression",
    "RandomForest",
    "XGBoost",
    "CatBoost",
    "SVC",
    # "RRAE",
    # "AutoGluon",  # Uncomment if you want to run AutoGluon
]

# Model-specific configurations
MODEL_CONFIG = {
    "LogisticRegression": {
        "scripts": ["1-train_cv.py", "2-train_final.py", "3-val.py"],
        "estimated_time_minutes": 0.5
    },
    "RandomForest": {
        "scripts": ["1-train_cv.py", "2-train_final.py", "3-val.py"],
        "estimated_time_minutes": 1.5
    },
    "XGBoost": {
        "scripts": ["1-train_cv.py", "2-train_final.py", "3-val.py"],
        "estimated_time_minutes": 2
    },
    "CatBoost": {
        "scripts": ["1-train_cv.py", "2-train_final.py", "3-val.py"],
        "estimated_time_minutes": 1.1
    },
    "SVC": {
        "scripts": ["1-train_cv.py", "2-train_final.py", "3-val.py"],
        "estimated_time_minutes": 8
    },
    "RRAE": {
        "scripts": ["1-train_cv.py", "2-train_final.py", "3-val.py"],
        "estimated_time_minutes": 100  
    },
    # "AutoGluon": {
    #     "scripts": ["1-train.py", "2-val.py"],  # Different workflow
    #     "estimated_time_minutes": 120
    # }
}

# Skip models that are already completed
SKIP_COMPLETED = True

# Continue from specific model (None to start from beginning)
START_FROM_MODEL = None  # e.g., "XGBoost" to skip previous models
