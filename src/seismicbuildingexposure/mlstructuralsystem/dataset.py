"""
Data preprocessing utilities for structural system prediction.

This module provides a suite of functions designed to prepare building data for a
machine learning model that predicts structural systems. It handles the entire
preprocessing pipeline from loading and cleaning raw data to feature engineering
and data splitting.

Includes:
- Data loading and cleaning with ordinal mapping enforcement for categorical features.
- A stratified, group-aware train-test split to ensure representative subsets.
- Scaling of numeric data using StandardScaler.
- Advanced feature generation using AutoGluon's feature engineering capabilities.
- Helper functions for visualizing model inputs and outputs.
"""

# === Imports ===
import os
from pathlib import Path

import pandas as pd
import numpy as np
import geopandas as gpd
import pyarrow.parquet as pq
from shapely import wkt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

from autogluon.features.generators import AutoMLPipelineFeatureGenerator
from autogluon.tabular import TabularDataset

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import distinctipy
from typing import Literal

from .. import footprint

def add_position_features(gdf, cfg):
    gdf = gdf.copy()

    if any(gdf.geometry.type == 'MultiPolygon'):
        print("There are multiplart geometries. Exploding geometries.")

    if any(gdf.geometry.type.str.contains('Polygon') == False):
        print("Some geometries are not Polygon")
        
    if "id" not in gdf.columns:
        gdf["id"] = gdf.index

    gdf = gdf.explode().reset_index(drop=True)
    gdf = gdf.to_crs(gdf.estimate_utm_crs())

    if "height" in gdf.columns:
        height_column = "height"
    else:
        height_column = None

    forces = footprint.position.contact_forces_df(
        gdf,
        height_column=height_column,
        buffer=cfg.POSITION_BUFFER,
        min_radius=cfg.POSITION_MIN_RADIUS
    )

    if any(field.startswith("contact_") for field in cfg.FEATURES):
        gdf["contact_force"] = forces["force"]
        gdf["contact_confinement_ratio"] = forces["confinement_ratio"]
        gdf["contact_angular_acc"] = forces["angular_acc"]
        gdf["contact_angle"] = forces["angle"]

    gdf['relative_position'] = footprint.position.relative_position(
        forces,
        min_angular_acc=cfg.POSITION_MIN_ANGULAR_ACC,
        min_confinement=cfg.POSITION_MIN_CONFINEMENT,
        min_angle=cfg.POSITION_MIN_ANGLE,
        min_force=cfg.POSITION_MIN_FORCE
    )

    gdf = gdf.to_crs(4326)
    return gdf 


def add_irregularity_features(gdf, cfg):
    def _needs_prefix(features, cfg, prefix, do_fsi=False):
        """
        Returns True if:
        - any requested field starts with `prefix`, OR
        - FSI dependency columns reference that prefix
        """
        if any(f.startswith(prefix) for f in features):
            return True

        if not do_fsi:
            return False

        ecc = getattr(cfg, "FSI_ECCENTRICITY_COL", "")
        setb = getattr(cfg, "FSI_SETBACK_COL", "")
        slen = getattr(cfg, "FSI_SLENDERNESS_COL", "")

        return (
            ecc.startswith(prefix)
            or setb.startswith(prefix)
            or slen.startswith(prefix)
        )

    gdf = gdf.copy()

    # -------------------------
    # Geometry checks
    # -------------------------
    geom_types = gdf.geometry.geom_type

    if (geom_types == "MultiPolygon").any():
        print("There are multipart geometries. Exploding geometries.")

    if (~geom_types.str.contains("Polygon")).any():
        print("Some geometries are not Polygon")

    # -------------------------
    # ID column
    # -------------------------
    if "id" not in gdf.columns:
        gdf["id"] = gdf.index

    # -------------------------
    # Normalize geometry
    # -------------------------
    gdf = gdf.explode(ignore_index=True)

    try:
        gdf = gdf.to_crs(gdf.estimate_utm_crs())
    except Exception:
        pass

    height_col = "height" if "height" in gdf.columns else None
    features = set(cfg.FEATURES)

    # -------------------------
    # FSI dependency flag
    # -------------------------
    do_fsi = "fsi" in features

    # -------------------------
    # Feature requirements
    # -------------------------
    needs_ec8 = _needs_prefix(features, cfg, "EC8_", do_fsi)
    needs_cr = _needs_prefix(features, cfg, "CR_", do_fsi)
    needs_ntc = _needs_prefix(features, cfg, "NTC_", do_fsi)
    needs_asce7 = _needs_prefix(features, cfg, "ASCE7_", do_fsi)
    needs_gndt = _needs_prefix(features, cfg, "GNDT_", do_fsi)

    # -------------------------
    # EC8
    # -------------------------
    if needs_ec8:
        ec8 = footprint.shape.eurocode_8_df(gdf)
        gdf["EC8_eccentricity_ratio"] = ec8["eccentricity_ratio"]
        gdf["EC8_radius_ratio"] = ec8["radius_ratio"]
        gdf["EC8_compactness"] = ec8["compactness"]
        gdf["EC8_direction_eccentricity"] = ec8["angle_eccentricity"]

    # -------------------------
    # Costa Rica
    # -------------------------
    if needs_cr:
        cr = footprint.shape.codigo_sismico_costa_rica_df(gdf)
        gdf["CR_eccentricity_ratio"] = cr["eccentricity_ratio"]
        gdf["CR_direction_eccentricity"] = cr["angle"]

    # -------------------------
    # NTC Mexico
    # -------------------------
    if needs_ntc:
        mx = footprint.shape.NTC_mexico_df(gdf)
        gdf["NTC_setback_ratio"] = mx["setback_ratio"]
        gdf["NTC_hole_ratio"] = mx["hole_ratio"]

    # -------------------------
    # ASCE 7
    # -------------------------
    if needs_asce7:
        asce = footprint.shape.asce_7_df(gdf)
        gdf["ASCE7_setback_ratio"] = asce["setback_ratio"]
        gdf["ASCE7_hole_ratio"] = asce["hole_ratio"]
        gdf["ASCE7_parallelity_angle"] = asce["parallelity_angle"]

    # -------------------------
    # GNDT Italy
    # -------------------------
    if needs_gndt:
        gndt = footprint.shape.gndt_italy_df(
            gdf,
            min_length=cfg.GNDT_MIN_LENGTH,
            min_area=cfg.GNDT_MIN_AREA
        )
        gdf["GNDT_main_shape_slenderness"] = gndt["beta_1_main_shape_slenderness"]
        gdf["GNDT_setback_ratio"] = gndt["beta_2_setback_ratio"]
        gdf["GNDT_eccentricity_ratio"] = gndt["beta_4_eccentricity_ratio"]
        gdf["GNDT_setback_slenderness"] = gndt["beta_6_setback_slenderness"]

    # -------------------------
    # Slenderness metrics
    # -------------------------
    if ("slenderness_elevation" in features) or ("slenderness_elevation" == cfg.FSI_SLENDERNESS_COL):
        if height_col is None:
            raise ValueError("Column 'height' is needed")

        a = footprint.shape.get_a(gdf)
        gdf["slenderness_elevation"] = gdf[height_col] / a

    if ("slenderness_inertia" in features) or ("slenderness_inertia" == cfg.FSI_SLENDERNESS_COL):
        val, ang = footprint.shape.inertia_slenderness(gdf, return_direction=True)
        gdf["slenderness_inertia"] = val
        gdf["inertia_direction"] = ang

    if ("slenderness_bbox" in features) or ("slenderness_bbox" == cfg.FSI_SLENDERNESS_COL):
        val, ang = footprint.shape.min_bbox_slenderness(gdf, return_direction=True)
        gdf["slenderness_bbox"] = val
        gdf["bbox_direction"] = ang

    if ("slenderness_circunscribed" in features) or ("slenderness_circunscribed" == cfg.FSI_SLENDERNESS_COL):
        val, ang = footprint.shape.circunscribed_slenderness(gdf, return_direction=True)
        gdf["slenderness_circunscribed"] = val
        gdf["circunscribed_direction"] = ang

    if ("inertia_vs_circle" in features) or ("inertia_vs_circle" == cfg.FSI_SLENDERNESS_COL):
        gdf["inertia_vs_circle"], _ = footprint.shape.inertia_circle(gdf)

    # -------------------------
    # FSI classification
    # -------------------------
    if do_fsi:
        gdf["fsi"] = "regular"

        gdf.loc[gdf[cfg.FSI_ECCENTRICITY_COL] > cfg.FSI_ECCENTRICITY_VAL, "fsi"] = "eccentricity"
        gdf.loc[gdf[cfg.FSI_SETBACK_COL] > cfg.FSI_SETBACK_VAL, "fsi"] = "setbacks"
        gdf.loc[gdf[cfg.FSI_SLENDERNESS_COL] > cfg.FSI_SLENDERNESS_VAL, "fsi"] = "slenderness"

        if "regularity_boolean" in features:
            gdf["regularity_boolean"] = gdf["fsi"].eq("regular")

    return gdf


def check_features(data: gpd.GeoDataFrame | pd.DataFrame, cfg) -> gpd.GeoDataFrame | pd.DataFrame:
    """
    Standardize target labels and enforce ordinal category ordering in a GeoDataFrame or DataFrame.

    This function takes a GeoDataFrame or DataFrame, verifies the presence of the target label column,
    applies predefined label replacements for consistency, and enforces an explicit ordering on
    specified ordinal categorical columns. It performs safety checks to detect mismatches between
    the expected ordinal categories and those actually present in the data, updating the mappings
    or raising exceptions as necessary.

    Args:
        data (gpd.GeoDataFrame | pd.DataFrame): Input spatial data or tabular data frame.

    Returns:
        gpd.GeoDataFrame | pd.DataFrame: The input data with cleaned labels and ordered categorical
        columns as defined in the ordinal feature mappings.

    Raises:
        Exception: If the label column is missing or if any ordinal feature contains categories
                   present in the data but not defined in the ordinal mappings.
    """
    if "geometry" in data.columns and not isinstance(data,gpd.GeoDataFrame):
        # Convert WKT strings to shapely geometries
        data["geometry"] = data["geometry"].apply(wkt.loads)

        # Convert to GeoDataFrame
        data = gpd.GeoDataFrame(data, geometry="geometry", crs="EPSG:4326")

    if isinstance(data, gpd.GeoDataFrame) and data.geometry.name != "geometry":
        data = data.rename_geometry("geometry")
        
    if cfg.LABEL not in data.columns:
        raise Exception(f"Label column '{cfg.LABEL}' not found in dataset columns: {list(data.columns)}")

    print("Dataset label counts")
    print(data[cfg.LABEL].value_counts())

    if cfg.CATEGORICAL_FEATURES is not None:
        for col, categories in cfg.CATEGORICAL_FEATURES.items():
            if col in data.columns:
                actual_values = set(data[col].dropna().unique())
                defined_set = set(categories)

                mismatch = actual_values - defined_set
                if mismatch:
                    raise ValueError(
                        f"Column '{col}' has mismatch between valid values and dataset values.\n"
                        f"Valid values: {defined_set}\n"
                        f"Dataset values: {actual_values}\n"
                        f"Mismatch: {mismatch}"
                    )

                # Convert to categorical (no order)
                data[col] = pd.Categorical(
                    data[col],
                    categories=categories,
                    ordered=False
                )

    if cfg.ORDINAL_FEATURES is not None:
        for col, defined_order in cfg.ORDINAL_FEATURES.items():
            if col in data.columns:
                actual_values = set(data[col].dropna().unique())
                defined_set = set(defined_order)
                if actual_values != defined_set:
                    print(f"  - WARNING: Column '{col}' has mismatch between defined order and actual data.")

                    missing_from_data = defined_set - actual_values
                    if missing_from_data:
                        print(f"    - Categories defined but NOT in data: {missing_from_data}. Updating mapping.")
                        # Update mapping to only include categories present in the data
                        defined_order = [val for val in defined_order if val in actual_values]
                        cfg.ORDINAL_FEATURES[col] = defined_order

                    missing_from_definition = actual_values - defined_set
                    if missing_from_definition:
                        raise Exception(f"    - Categories in data but NOT in definition: {missing_from_definition}")
                else:
                    print(f"  - Column '{col}': Mapping is consistent with data.")

                data[col] = pd.Categorical(data[col], categories=defined_order, ordered=True)

    if "area" in cfg.FEATURES and "area" not in data.columns:
        if isinstance(data, gpd.GeoDataFrame):
            print("Adding 'area' column from footprint geometries. Transforming crs to utm.")
            data.geometry = data.geometry.to_crs(data.estimate_utm_crs())
            """TODO: Geographic area for very large or multi city datasets"""
            data["area"] = data.geometry.area

    if "perimeter" in cfg.FEATURES and "perimeter" not in data.columns:
        if isinstance(data, gpd.GeoDataFrame):
            print("Adding 'perimeter' column from footprint geometries. Transforming crs to utm.")
            data.geometry = data.geometry.to_crs(data.estimate_utm_crs())
            """TODO: Geographic perimeter for very large or multi city datasets"""
            data["perimeter"] = data.geometry.boundary.length 

    if isinstance(data, gpd.GeoDataFrame): 
        add_position_features(data,cfg)
        if "relative_position" in cfg.FEATURES and "relative_position" not in data.columns:
            add_position_features(data,cfg)

    if isinstance(data, gpd.GeoDataFrame): 
        data.geometry = data.geometry.to_crs(4326)      
            
    for col in data.columns:
        if (col in cfg.FEATURES and 
            col not in cfg.ORDINAL_FEATURES.keys() and 
            col not in cfg.CATEGORICAL_FEATURES.keys()
        ):
            data[col] = pd.to_numeric(data[col])

    if hasattr(cfg, "STRATIFY_COLUMN"): 
        if cfg.STRATIFY_COLUMN is not None:
            if cfg.STRATIFY_COLUMN not in data.columns:
                raise Exception(f"Mandatory stratify column {cfg.STRATIFY_COLUMN} not in dataset columns {data.columns}.")

    actual_columns = set(data.columns)
    defined_columns = set(cfg.FEATURES)

    missing_columns = defined_columns - actual_columns
    extra_columns = actual_columns - defined_columns - {"id","source","geometry"}

    # Print extra columns (informational)
    if extra_columns:
        print("Extra columns with no use for model (in dataset but not in FEATURES):", extra_columns)

    # Raise error only for missing columns
    if missing_columns:
        raise ValueError(
            "Column mismatch detected: dataset is missing required FEATURES.\n"
            f"Defined columns (features): {defined_columns}\n"
            f"Dataset columns: {actual_columns}\n"
            f"Missing columns (in FEATURES but not in dataset): {missing_columns}"
        )
    
    return data


def create_train_test_split(df: pd.DataFrame,
                            cfg,
                            stratify_col: str|None = None,
                            test_size: float = 0.3
                        ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Performs a stratified train-test split based on the interaction of target and group columns.

    This method ensures that the distribution of the target variable within each group (e.g., city)
    is preserved across the training and testing sets. It handles the edge case where a
    stratum might have only one sample by assigning it directly to the training set.

    Args:
        df (pd.DataFrame): The input dataframe to split.
        target_col (str): The column name to stratify on (usually the label).
        group_col (str): A secondary column for group-aware stratification (e.g., 'city').
        test_size (float): The proportion of the dataset to allocate to the test set.
        random_state (int): The seed for the random number generator for reproducibility.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the training and testing dataframes.
    """
    if test_size == 0:
        raise Exception("Invalid value 0 for test_size")
    elif test_size >= 1:
        raise Exception(f"Invalid value {test_size} for test_size")
    
    df_copy = df.copy()
    # Create a temporary stratification column by combining the target and group columns.
    # This allows stratification on the joint distribution.
    strata_col_name = "_strata"
    if stratify_col is not None:
        strata_col_name = "_strata"
        df_copy[strata_col_name] = df_copy[cfg.LABEL].astype(str) + "_" + df_copy[stratify_col].astype(str)
    else:
        df_copy[strata_col_name] = df_copy[cfg.LABEL].astype(str).copy()

    # Identify strata that contain only one sample, as train_test_split cannot handle them.
    strata_counts = df_copy[strata_col_name].value_counts()
    single_sample_strata = strata_counts[strata_counts < 2].index

    if not single_sample_strata.empty:
        print(f"Warning: Found {len(single_sample_strata)} strata with only 1 sample. "
              f"These will be placed in the training set.")
        # Isolate the single-sample rows and the rows that can be split
        single_sample_df = df_copy[df_copy[strata_col_name].isin(single_sample_strata)]
        splittable_df = df_copy[~df_copy[strata_col_name].isin(single_sample_strata)]
    else:
        # If no single-sample strata, all data is splittable
        single_sample_df = pd.DataFrame()
        splittable_df = df_copy

    # Perform the stratified split on the splittable portion of the data
    train_df, test_df = train_test_split(
        splittable_df,
        test_size=test_size,
        random_state=cfg.RANDOM_STATE,
        stratify=splittable_df[strata_col_name]
    )

    # Add the single-sample rows back into the training set
    if not single_sample_df.empty:
        train_df = pd.concat([train_df, single_sample_df])

    # Remove the temporary stratification column before returning the dataframes
    return train_df.drop(columns=[strata_col_name]), test_df.drop(columns=[strata_col_name])


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

def encode(data_df: pd.DataFrame,
           cfg,
                      scale_numeric: bool = True,
                      preprocessor: ColumnTransformer|None = None,
                      label_encoder: LabelEncoder|None = None,
                      fit: bool = True) -> tuple:
    """
    Apply full preprocessing pipeline to a dataset.
    
    Args:
        data_df: Raw dataframe to process
        scale_numeric: Whether to scale numeric features
        preprocessor: Pre-fitted preprocessor (if None, creates new one)
        label_encoder: Pre-fitted label encoder (if None, creates new one)
        fit: Whether to fit the preprocessor (True for train, False for test)
    
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
    
    if label_encoder is None:
        label_encoder = get_label_encoder(processed_df, cfg)
    
    # Prepare features (X) and labels (y)
    X_df = processed_df.drop(columns=[cfg.LABEL])
    y = label_encoder.transform(processed_df[cfg.LABEL])
    
    # Transform features
    if fit:
        X = preprocessor.fit_transform(X_df)
    else:
        X = preprocessor.transform(X_df)
    
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
    Create and fit a LabelEncoder for the target column.
    
    Args:
        data_df: Dataframe containing the label column
    
    Returns:
        Fitted LabelEncoder
    """
    label_encoder = LabelEncoder()
    label_encoder.fit(data_df[cfg.LABEL])
    
    print(f"Label encoding: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")
    print("\n", 50*"#", "\n")
    
    return label_encoder


###############################

def get_feature_generator(data_df: pd.DataFrame, cfg) -> AutoMLPipelineFeatureGenerator:
    """
    Creates and fits an AutoGluon feature generator on the provided data.

    This generator is configured to automatically handle numeric and categorical
    features, creating new features as needed. It is fitted on the entire dataset
    (without the label) to learn the data's properties for consistent transformation
    across train and test sets.

    Args:
        data_df (pd.DataFrame): The input dataframe to fit the generator on.

    Returns:
        AutoMLPipelineFeatureGenerator: The fitted feature generator object.
    """
    # Initialize AutoGluon's feature generator with a specific configuration
    feature_generator = AutoMLPipelineFeatureGenerator(
        enable_numeric_features=True,
        enable_categorical_features=True,
        enable_datetime_features=False,
        enable_text_special_features=False,
        enable_text_ngram_features=False,
        enable_vision_features=False
    )

    # Convert DataFrame to AutoGluon's TabularDataset for optimized handling
    data = TabularDataset(data_df)

    X_cols = list(
        set(cfg.FEATURES).intersection(set(data.columns))
    )
    X = data[X_cols]

    # Fit the feature generator on the feature data
    feature_generator.fit(X)

    return feature_generator


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


def encode_autogluon(data_df: pd.DataFrame, cfg,
                       scale: bool = True,
                       feature_generator: AutoMLPipelineFeatureGenerator|None = None,
                       label_encoder: LabelEncoder|None = None) -> tuple:
    """
    Applies a full preprocessing pipeline to a dataset.

    This function orchestrates feature generation using AutoGluon, label encoding,
    and optional scaling. If a feature generator or label encoder is not provided,

    it will create and fit new ones based on the input data.

    Args:
        data_df (pd.DataFrame): The raw dataframe to be processed.
        scale (bool): If True, applies StandardScaler to the generated features.
        feature_generator (AutoMLPipelineFeatureGenerator, optional): A pre-fitted
            feature generator. If None, one will be created. Defaults to None.
        label_encoder (LabelEncoder, optional): A pre-fitted label encoder.
            If None, one will be created. Defaults to None.

    Returns:
        tuple: A tuple containing (X, y), where X are the processed features
               (pd.DataFrame) and y are the encoded labels (np.ndarray).
    """
    # Create a copy to avoid modifying the original dataframe
    processed_df = data_df.copy()

    # If a generator or encoder is not provided, create them from the data
    if feature_generator is None:
        feature_generator = get_feature_generator(processed_df, cfg)
    if label_encoder is None:
        label_encoder = get_label_encoder(processed_df, cfg)

    all_columns = list(
        set(cfg.FEATURES+[cfg.LABEL]).intersection(set(processed_df.columns))
    )
    processed_df = processed_df[all_columns]

    # Convert to TabularDataset for AutoGluon compatibility
    data = TabularDataset(processed_df)

    # Generate features using the fitted generator
    X = feature_generator.transform(data.drop(columns=cfg.LABEL))
    # Encode the labels
    y = label_encoder.transform(data[cfg.LABEL])

    # Apply scaling to the generated features if requested
    if scale:
        X = scale_data(X)

    return X, y


def plot_2D_cluster(
        X_2D: np.ndarray,
        y: np.ndarray,
        label_encoder: LabelEncoder,
        title: str = "2D visualization of processed features",
        xlabel: str = "Component 1",
        ylabel: str = "Component 2"
    ) -> None:
    """
    Creates a 2D scatter plot to visualize clustered or dimensionally-reduced data.

    Each point is colored according to its class label, which is useful for assessing
    class separability after dimensionality reduction techniques like PCA or UMAP.

    Args:
        X_2D (np.ndarray): Data with two dimensions, shape (n_samples, 2).
        y (np.ndarray): Integer-encoded labels for each sample.
        label_encoder (LabelEncoder): The fitted encoder used to map integer labels
                                      back to class names for the legend.
        title (str, optional): The title of the plot.
        xlabel (str, optional): The label for the x-axis.
        ylabel (str, optional): The label for the y-axis.
    """
    plt.figure(figsize=(10, 8))
    # Define a color palette for the classes

    colors = colors = distinctipy.get_colors(len(label_encoder.classes_))

    # Plot points for each class separately to create a legend
    for i, class_name in enumerate(label_encoder.classes_):
        # Find the indices for the current class
        idx = (y == i)
        plt.scatter(X_2D[idx, 0], X_2D[idx, 1], color=colors[i], label=f'{class_name}', alpha=0.7, s=20)

    plt.legend()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(y_pred: np.ndarray, y_true: np.ndarray, normalize: bool = False) -> None:
    """
    Plots a confusion matrix for evaluating classification performance.

    Uses a seaborn heatmap for a clear visual representation. The matrix can be
    normalized to show recall rates for each class.

    Args:
        y_pred (np.ndarray): The predicted labels from a model.
        y_true (np.ndarray): The ground truth labels.
        normalize (bool): If True, the confusion matrix is normalized by the number
                          of true instances for each class (row-wise normalization).
                          Defaults to False.
    """
    # Automatically infer the sorted list of class names from the data
    class_names = np.unique(np.concatenate([y_true, y_pred]))

    # Compute the confusion matrix using scikit-learn
    cm = confusion_matrix(y_true, y_pred, labels=class_names)

    # Format for printing (raw counts or normalized floats)
    fmt = 'd'
    title = 'Confusion Matrix (Counts)'

    if normalize:
        # Normalize each row (true class) so it sums to 1
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title = 'Confusion Matrix'

    # Plot the heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title(title)
    plt.tight_layout()
    plt.show()

def read_geofile(file, cfg):
    file_path = Path(file)

    # Load based on file type
    if file_path.suffix.lower() in [".parquet", ".geoparquet"]:
        gdf = gpd.read_parquet(file_path)
    else:
        gdf = gpd.read_file(file_path)

    # Ensure CRS is WGS84
    gdf = gdf.to_crs(epsg=4326)

    # Add source column
    gdf["source"] = str(file_path)

    # Select only relevant columns that exist
    all_columns = list(
        set(["id"] + cfg.FEATURES + [cfg.LABEL, "source", "geometry"])
        .intersection(gdf.columns)
    )

    gdf = check_features(gdf[all_columns], cfg)

    return gdf

def read_tabular(file, cfg):
    file_path = Path(file)
    suffix = file_path.suffix.lower()

    # Load based on file type
    if suffix == ".csv":
        df = pd.read_csv(file_path)
    elif suffix in [".xlsx", ".xls"]:
        df = pd.read_excel(file_path)
    elif suffix == ".parquet":
        df = pd.read_parquet(file_path)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    # Add source column
    df["source"] = str(file_path)

    # Keep only relevant columns that exist
    all_columns = list(
        set(["id"] + cfg.FEATURES + [cfg.LABEL, "source"])
        .intersection(df.columns)
    )

    df = check_features(df[all_columns], cfg)

    return df

def is_geofile(file_path: Path) -> bool:
    file_path = Path(file_path)
    # Heuristic: GeoParquet usually contains geometry metadata
    if file_path.suffix.lower() == ".parquet":
        try:
            parquet_file = pq.ParquetFile(file_path)
            schema = parquet_file.schema_arrow

            # Check if a geometry column exists
            return "geometry" in schema.names
        except Exception:
            return False
        
    return file_path.suffix.lower() in [".geojson", ".gpkg", ".shp"]


def is_tabular(file_path: Path) -> bool:
    file_path = Path(file_path)
    return file_path.suffix.lower() in [".csv", ".xlsx", ".xls", ".parquet"]


def process_file(file_path, cfg, geofile=False):
    file_path = Path(file_path)
    if is_geofile(file_path):
        df_i = read_geofile(file_path, cfg)
        geofile = True
    elif is_tabular(file_path):
        df_i = read_tabular(file_path, cfg)
        if geofile:
            raise Exception(f"Combining geometry and tabular files is not supported. File {file_path} is tabular.")
            # Ensure geometry column exists if mixing with geodata
            df_i["geometry"] = None
    else:
        raise ValueError(f"Unsupported file type: {file_path}")
    
    return df_i, geofile
    
def read_mixed(path, cfg):
    path = Path(path)
    geofile = False

    # Resolve output path
    output_folder = Path(cfg.PREPROCESSED_DATASET_PATH)
    if not output_folder.is_absolute():
        output_folder = Path(cfg.PROJECT_ROOT) / output_folder

    os.makedirs(output_folder, exist_ok=True)

    # Case 1: folder
    if path.is_dir():
        dfs = []
        for file_path in path.iterdir():
            if file_path.is_file():
                filename_wo_ext = os.path.splitext(file_path)[0]
                output_path = os.path.join(output_folder, filename_wo_ext)
                if os.path.isfile(output_path + ".gpkg"):
                    file_path = output_path + ".gpkg"
                elif os.path.isfile(output_path + ".csv"):
                    file_path = output_path + ".csv"

                df_i, geofile = process_file(file_path, cfg, geofile=geofile)
                if geofile:
                    df_i.to_file(output_path + ".gpkg")
                else:
                    df_i.to_csv(output_path + ".csv", index=False)

                dfs.append(df_i)

        df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    # Case 2: single file
    else:
        df, geofile = process_file(path, cfg, geofile=geofile)

    return df, geofile

def preprocess(cfg):
    train_path = cfg.TRAIN_PATH 
    if hasattr(cfg, "STRATIFY_COLUMN"):
        stratify_column = cfg.STRATIFY_COLUMN
    else:
        stratify_column = None

    if hasattr(cfg, "TEST_PATH"):
        test_path = cfg.TEST_PATH
    else:
        test_path = None 

    if hasattr(cfg, "TEST_SIZE"):
        test_size = cfg.TEST_SIZE
    else:
        test_size = 0 


    # Resolve output path
    output_path = Path(cfg.TRAIN_TEST_OUTPUT_PATH)
    if not output_path.is_absolute():
        output_path = Path(cfg.PROJECT_ROOT) / output_path

    os.makedirs(output_path, exist_ok=True)

    # Load data
    df, train_geofile = read_mixed(train_path, cfg)

    if test_path is None:
        train_df, test_df = create_train_test_split(
            df,
            cfg,
            stratify_col=stratify_column,
            test_size=test_size
        )
        test_geofile = train_geofile
    else:
        train_df = df
        test_df, test_geofile = read_mixed(test_path, cfg)

    # Save train
    if train_geofile:
        train_df.to_file(output_path / "train_raw.gpkg")
        train_df = pd.DataFrame(train_df.drop(columns="geometry"))
    else:
        train_df.to_csv(output_path / "train_raw.csv", index=False)

    # Save test
    if test_geofile:
        test_df.to_file(output_path / "test_raw.gpkg")
        test_df = pd.DataFrame(test_df.drop(columns="geometry"))
    else:
        test_df.to_csv(output_path / "test_raw.csv", index=False)

    print(f"Train/test splits saved in {output_path}")

    # Preprocessing
    X_train, y_train, preprocessor, label_encoder = encode(
        train_df,
        cfg,
        scale_numeric=True,
        fit=True
    )

    X_test, y_test, _, _ = encode(
        test_df,
        cfg,
        preprocessor=preprocessor,
        label_encoder=label_encoder,
        fit=False
    )

    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")

    feature_names = get_feature_names(preprocessor, train_df, cfg)
    print(f"Feature names: {feature_names}")
    print(f"Num. features: {len(feature_names)}")

    # Build preprocessed DataFrames
    train_preprocessed = pd.DataFrame(X_train, columns=feature_names)
    train_preprocessed[cfg.LABEL] = y_train

    test_preprocessed = pd.DataFrame(X_test, columns=feature_names)
    test_preprocessed[cfg.LABEL] = y_test

    # Save outputs (corrected paths)
    train_output_path = output_path / "train_preprocessed.csv"
    test_output_path = output_path / "test_preprocessed.csv"

    train_preprocessed.to_csv(train_output_path, index=False)
    test_preprocessed.to_csv(test_output_path, index=False)

    print(f"Preprocessed datasets saved in {output_path}")

def load(split: Literal["train", "test"], cfg):
    df = None
    if split=="train":
        path = Path(cfg.TRAIN_TEST_OUTPUT_PATH) / "train_preprocessed.csv"
        if os.path.isfile(path):
            df = pd.read_csv(path)
        else:
            preprocess(cfg)
            df = pd.read_csv(path)
    elif split=="test":
        path = Path(cfg.TRAIN_TEST_OUTPUT_PATH) / "test_preprocessed.csv"
        if os.path.isfile(path):
            df = pd.read_csv(path)
        else:
            preprocess(cfg)
            df = pd.read_csv(path)
    else:
        raise Exception("Argument split must be 'train' or 'test'.")
    X = df.drop(columns=[cfg.LABEL])
    y = df[cfg.LABEL]
    return X, y