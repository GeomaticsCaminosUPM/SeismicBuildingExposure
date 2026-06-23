"""
Public API for structural-system dataset preprocessing.

This module is a thin facade over `dataset_utils/`, kept so that existing
call sites (`main.py`, `1-train_cv.py`, `2-train_final.py`, `3-val.py`,
`test.py`, etc.) can keep doing:

    import SeismicBuildingExposure.mlstructuralsystem.dataset as dataset
    dataset.preprocess(config)
    X, y = dataset.load(split="train", cfg=config)
    dataset.save_test(df, model_name="XGBoost", cfg=config)

without any changes, while the implementation is split by responsibility:

- dataset_utils.geometry_features  -> seismic shape/position feature engineering
- dataset_utils.validation         -> label/column/category validation & cleaning
- dataset_utils.splitting          -> stratified train/val split
- dataset_utils.encoding           -> sklearn ColumnTransformer + LabelEncoder
- dataset_utils.io                 -> raw file reading & on-disk caching
- dataset_utils.visualization      -> plotting helpers (not used by the pipeline)
- dataset_utils.pipeline           -> orchestration: preprocess / load / save_test
"""

# Orchestration entry points (the ones every external script actually calls)
from .dataset_utils.pipeline import preprocess, load, save_test

# Re-exported for backwards compatibility, in case any script or notebook
# imports these directly from `dataset` instead of from their new home.
from .dataset_utils.geometry_features import add_position_features, add_irregularity_features
from .dataset_utils.validation import check_features
from .dataset_utils.splitting import create_train_test_split
from .dataset_utils.encoding import (
    identify_feature_types,
    create_preprocessor,
    encode,
    get_feature_names,
    get_label_encoder,
    scale_data,
)
from .dataset_utils.io import (
    read_geofile,
    read_tabular,
    is_geofile,
    is_tabular,
    process_file,
    read_mixed,
)
from .dataset_utils.visualization import plot_2D_cluster, plot_confusion_matrix

__all__ = [
    "preprocess", "load", "save_test",
    "add_position_features", "add_irregularity_features",
    "check_features",
    "create_train_test_split",
    "identify_feature_types", "create_preprocessor", "encode",
    "get_feature_names", "get_label_encoder", "scale_data",
    "read_geofile", "read_tabular", "is_geofile", "is_tabular",
    "process_file", "read_mixed",
    "plot_2D_cluster", "plot_confusion_matrix",
]