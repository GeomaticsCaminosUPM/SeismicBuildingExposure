"""
Feature encoding and scaling pipeline built on scikit-learn primitives.

Builds a `ColumnTransformer` that scales numeric features, ordinally encodes
ordinal features (respecting the explicit order from `config.ORDINAL_FEATURES`),
and one-hot encodes categorical features. Also handles label encoding with a
fixed class order driven by `config.LABEL_VALUES`.
"""

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer


def identify_feature_types(data_df: pd.DataFrame, cfg) -> dict:
    """
    Automatically identify numeric, ordinal, and categorical features.
    
    Args:
        data_df: Input dataframe
    
    Returns:
        Dictionary with keys 'numeric', 'ordinal', 'categorical' containing column lists
    """
    processed_df = data_df.copy()
    all_cols = list(
        set(cfg.FEATURES+[cfg.LABEL]).intersection(set(processed_df.columns))
    )
    processed_df = processed_df[all_cols]

    # Identify ordinal features from mappings
    ordinal_features = [col for col in cfg.ORDINAL_FEATURES.keys() if col in all_cols]
    
    # Identify numeric features (float and int types, excluding ordinal)
    numeric_features = [col for col in all_cols 
                       if col not in ordinal_features 
                       and data_df[col].dtype in ['int64', 'int32', 'float64', 'float32']]
    
    # Identify categorical features (object, category types, excluding ordinal)
    categorical_features = [col for col in all_cols 
                           if col not in ordinal_features 
                           and col not in numeric_features
                           and data_df[col].dtype == 'category']
    
    feature_types = {
        'numeric': numeric_features,
        'ordinal': ordinal_features,
        'categorical': categorical_features
    }
    
    print("Feature type identification:")
    print(f"  - Numeric features ({len(numeric_features)}): {numeric_features}")
    print(f"  - Ordinal features ({len(ordinal_features)}): {ordinal_features}")
    print(f"  - Categorical features ({len(categorical_features)}): {categorical_features}")
    print("\n", 50*"#", "\n")

    return feature_types


def create_preprocessor(data_df: pd.DataFrame, cfg, scale_numeric: bool = True) -> ColumnTransformer:
    """
    Create a sklearn ColumnTransformer for preprocessing all feature types.
    
    Args:
        data_df: Input dataframe to determine feature types and ordinal categories
        scale_numeric: Whether to scale numeric features (default: True)
    
    Returns:
        Fitted ColumnTransformer ready to transform data
    """

    data_df = data_df[cfg.FEATURES]
    
    feature_types = identify_feature_types(data_df, cfg)
    
    transformers = []
    
    # Numeric features: scale if requested, otherwise pass through
    if feature_types['numeric']:
        if scale_numeric:
            numeric_transformer = StandardScaler()
        else:
            numeric_transformer = 'passthrough'
        transformers.append(('num', numeric_transformer, feature_types['numeric']))
    
    # Ordinal features: encode with explicit ordering
    if feature_types['ordinal']:
        # Create ordinal encoder with categories from mappings
        ordinal_categories = [cfg.ORDINAL_FEATURES[col] for col in feature_types['ordinal']]
        ordinal_transformer = OrdinalEncoder(
            categories=ordinal_categories,
            handle_unknown='use_encoded_value',
            unknown_value=-1
        )
        transformers.append(('ord', ordinal_transformer, feature_types['ordinal']))
    
    # Categorical features: one-hot encode
    if feature_types['categorical']:
        categorical_transformer = OneHotEncoder(
            drop=None,  # TODO: Drop first category to avoid multicollinearity for LINEAR models
            sparse_output=False,
            handle_unknown='ignore'
        )
        transformers.append(('cat', categorical_transformer, feature_types['categorical']))
    
    # Create the column transformer
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder='drop'  # Drop any columns not specified
    )
    
    return preprocessor

def encode(
        data_df: pd.DataFrame,
        cfg,
        scale_numeric: bool = True,
        preprocessor: ColumnTransformer|None = None,
        label_encoder: LabelEncoder|None = None,
        fit: bool = True,
        is_test: bool = False,
    ) -> tuple:
    """
    Apply full preprocessing pipeline to a dataset.
    
    Args:
        data_df: Raw dataframe to process
        scale_numeric: Whether to scale numeric features
        preprocessor: Pre-fitted preprocessor (if None, creates new one)
        label_encoder: Pre-fitted label encoder (if None, creates new one)
        fit: Whether to fit the preprocessor (True for train, False for val)
    
    Returns:
        Tuple of (X, y, preprocessor, label_encoder) where:
        - X: Processed features (numpy array)
        - y: Encoded labels (numpy array)
        - preprocessor: The fitted preprocessor
        - label_encoder: The fitted label encoder
    """
    processed_df = data_df.copy()
    all_cols = list(
        set(cfg.FEATURES+[cfg.LABEL]).intersection(set(processed_df.columns))
    )
    processed_df = processed_df[all_cols]
    # Create preprocessor and label encoder if not provided
    if preprocessor is None:
        preprocessor = create_preprocessor(processed_df, cfg, scale_numeric=scale_numeric)
    
    if (not is_test) and label_encoder is None:
        label_encoder = get_label_encoder(processed_df, cfg)
    
    # Prepare features (X) and labels (y)
    if is_test:
        X_df = processed_df.copy()
    else:
        X_df = processed_df.drop(columns=[cfg.LABEL])
        y = label_encoder.transform(processed_df[cfg.LABEL])
    # Transform features
    if fit:
        X = preprocessor.fit_transform(X_df)
    else:
        X = preprocessor.transform(X_df)
    
    if is_test:
        return X, preprocessor
    else:
        return X, y, preprocessor, label_encoder


def get_feature_names(preprocessor: ColumnTransformer, data_df: pd.DataFrame, cfg) -> list:
    """
    Extract feature names after preprocessing.
    
    Args:
        preprocessor: Fitted ColumnTransformer
        data_df: Original dataframe (to get feature type information)
    
    Returns:
        List of feature names after transformation
    """
    feature_types = identify_feature_types(data_df, cfg)
    feature_names = []
    
    for name, transformer, columns in preprocessor.transformers_:
        if name == 'num':
            feature_names.extend(columns)
        elif name == 'ord':
            feature_names.extend(columns)
        elif name == 'cat':
            if hasattr(transformer, 'get_feature_names_out'):
                cat_features = transformer.get_feature_names_out(columns)
                feature_names.extend(cat_features)
            else:
                feature_names.extend(columns)
    
    return feature_names

def get_label_encoder(data_df: pd.DataFrame, cfg) -> LabelEncoder:
    """
    Create and fit a LabelEncoder using a fixed label order.
    
    Args:
        data_df: DataFrame containing the label column
        cfg: config with LABEL column name
    
    Returns:
        Fitted LabelEncoder
    """
    label_encoder = LabelEncoder()
    
    if hasattr(cfg, "LABEL_VALUES"):
        # enforce fixed order
        label_encoder.fit(cfg.LABEL_VALUES)
    else:
        label_encoder.fit(data_df[cfg.LABEL])

    print(
        "Label encoding:",
        dict(zip(label_encoder.classes_,
                 label_encoder.transform(label_encoder.classes_)))
    )
    print("\n", 50 * "#", "\n")
    
    return label_encoder


def scale_data(data_df: pd.DataFrame | np.ndarray) -> pd.DataFrame | np.ndarray:
    """
    Scales numeric features in a DataFrame or NumPy array using StandardScaler.

    If a DataFrame is provided, it automatically detects and scales only the
    numeric (float) columns. If a NumPy array is provided, it scales the entire array.

    Args:
        data_df (pd.DataFrame or np.ndarray): The input data containing numeric features.

    Returns:
        pd.DataFrame or np.ndarray: The scaled data, returned in the same format as the input.

    Raises:
        ValueError: If the input data is not a pandas DataFrame or a NumPy array.
    """
    # Handle NumPy array input
    if isinstance(data_df, np.ndarray):
        scaler = StandardScaler()
        return scaler.fit_transform(data_df)

    # Handle pandas DataFrame input
    elif isinstance(data_df, pd.DataFrame):
        # Identify numeric columns to be scaled (typically float types)
        numeric_cols = data_df.select_dtypes(include=['float', 'int']).columns
        if len(numeric_cols) == 0:
            return data_df  # Return as-is if no numeric columns
        scaler = StandardScaler()
        data_scaled = data_df.copy()
        # Fit and transform only the numeric columns, preserving the rest
        data_scaled[numeric_cols] = scaler.fit_transform(data_df[numeric_cols])
        return data_scaled

    # Handle unsupported types
    else:
        raise ValueError("Unsupported data type for scaling. Must be DataFrame or ndarray.")