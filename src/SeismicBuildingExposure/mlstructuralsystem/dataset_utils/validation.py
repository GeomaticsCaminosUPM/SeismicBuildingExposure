"""
Dataset validation and cleaning.

This module standardizes target labels, enforces categorical/ordinal
consistency against `config.py` definitions, removes invalid geometries
or missing values, and triggers geometry-derived feature generation.
"""

import pandas as pd
import geopandas as gpd
from shapely import wkt

from .geometry_features import add_irregularity_features, add_position_features


def check_features(data: gpd.GeoDataFrame | pd.DataFrame, cfg, test=False) -> gpd.GeoDataFrame | pd.DataFrame:
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
    data = data.copy()
    if test:
        if cfg.LABEL in data.columns:
            test = False 

    if "geometry" in data.columns and not isinstance(data,gpd.GeoDataFrame):
        # Convert WKT strings to shapely geometries
        data["geometry"] = data["geometry"].apply(wkt.loads)

        # Convert to GeoDataFrame
        data = gpd.GeoDataFrame(data, geometry="geometry", crs="EPSG:4326")

    if isinstance(data, gpd.GeoDataFrame) and data.geometry.name != "geometry":
        data = data.rename_geometry("geometry")

    if isinstance(data, gpd.GeoDataFrame):
        if len(data.explode(index_parts=False).reset_index(drop=True)) != len(data):
            print("\n" + "="*60)
            print("⚠️ WARNING: There are multiplart geometries. Exploding geometries.")
            print("="*60 + "\n")
            data = data.explode(index_parts=False).reset_index(drop=True)

        mask = data.geometry.type.str.contains("Polygon", na=False)

        n_removed = (~mask).sum()

        if n_removed > 0:
            print("\n" + "="*60)
            print(f"⚠️ WARNING: Removed {n_removed} rows with non-Polygon or invalid geometries")
            print("="*60 + "\n")

            data = data[mask]
        
    if (not test) and (cfg.LABEL not in data.columns):
        raise Exception(f"Label column '{cfg.LABEL}' not found in dataset columns: {list(data.columns)}")
    
    if not test:
        mask = data[cfg.LABEL].isin(cfg.LABEL_VALUES) & data[cfg.LABEL].notna()
        n_removed = (~mask).sum()
        invalid_values = (
            data.loc[~mask, cfg.LABEL]
            .astype("object")
            .dropna()
            .unique()
            .tolist()
        )
        data = data[mask]
        if n_removed > 0:
            print("\n" + "="*60)
            print(f"⚠️ WARNING: Removed {n_removed} rows due to missing or invalid values in column '{cfg.LABEL}'. Invalid values {invalid_values}.")
            print("="*60 + "\n")
            
            data = data[data[cfg.LABEL].notna()]

        # Convert to categorical (no order)
        data[cfg.LABEL] = pd.Categorical(
            data[cfg.LABEL],
            categories=cfg.LABEL_VALUES,
            ordered=False
        )

    for col in ["id",*cfg.FEATURES]:
        if col in data.columns:
            n_removed = data[col].isna().sum()
            
            if n_removed > 0:
                print("\n" + "="*60)
                print(f"⚠️ WARNING: Removed {n_removed} rows due to missing values in column '{col}'")
                print("="*60 + "\n")
                
                data = data[data[col].notna()]

    data = data.reset_index(drop=True)

    if not test:
        print("Dataset label counts")
        label_counts = data[cfg.LABEL].value_counts()
        print(label_counts)

        # Find labels with fewer than 5 samples
        rare_labels = label_counts[label_counts < 5].index.tolist()

        if rare_labels:
            rare_counts = label_counts[label_counts < 5].to_dict()

            raise ValueError(
                "\n"
                + "=" * 60
                + "\n"
                + "❌ ERROR: Some labels have fewer than 5 instances.\n\n"
                + f"Rare labels detected: {rare_counts}\n\n"
                + "These labels should be removed from config.py LABEL_VALUES before training.\n"
                + "=" * 60
            )

    if cfg.CATEGORICAL_FEATURES is not None:
        for col, categories in cfg.CATEGORICAL_FEATURES.items():
            if col in data.columns:
                actual_values = set(data[col].dropna().unique())
                defined_set = set(categories)

                mismatch = actual_values - defined_set
                if mismatch:
                    raise ValueError(
                        "\n"
                        + "=" * 60
                        + "\n"
                        + f"❌ ERROR: Column '{col}' has mismatch between "
                        "defined categorical values and dataset values.\n\n"
                        + f"Defined values : {sorted(defined_set)}\n"
                        + f"Dataset values : {sorted(actual_values)}\n"
                        + f"Mismatch       : {sorted(mismatch)}\n\n"
                        + "Please update CATEGORICAL_FEATURES in config.py "
                        "or clean the dataset values.\n"
                        + "=" * 60
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
                    # print(f"  - WARNING: Column '{col}' has mismatch between defined order and actual data.")

                    # missing_from_data = defined_set - actual_values
                    # if missing_from_data:
                    #     print(f"    - Categories defined but NOT in data: {missing_from_data}. Updating mapping.")
                    #     # Update mapping to only include categories present in the data
                    #     defined_order = [val for val in defined_order if val in actual_values]
                    #     cfg.ORDINAL_FEATURES[col] = defined_order

                    missing_from_definition = actual_values - defined_set
                    if missing_from_definition:
                        raise ValueError(
                            "\n"
                            + "=" * 60
                            + "\n"
                            + f"❌ ERROR: Column '{col}' has categories in the dataset "
                            "that are NOT defined in ORDINAL_FEATURES.\n\n"
                            + f"Missing categories: {sorted(missing_from_definition)}\n\n"
                            + "Please update ORDINAL_FEATURES in config.py "
                            "to include these categories.\n"
                            + "=" * 60
                        )
                else:
                    print(f"  - Column '{col}': Mapping is consistent with data.")

                data[col] = pd.Categorical(data[col], categories=defined_order, ordered=True)

    if "area" in cfg.FEATURES and "area" not in data.columns:
        if isinstance(data, gpd.GeoDataFrame):
            print("Adding 'area' column from footprint geometries. Transforming crs to utm.")
            data.geometry = data.geometry.to_crs(data.estimate_utm_crs())
            """TODO: Geographic area for very large or multi city datasets"""
            data["area"] = data.geometry.area

    if hasattr(cfg, "MIN_AREA"): 
        if cfg.MIN_AREA is not None and cfg.MIN_AREA > 0:
            if "area" in data.columns:
                deleted_cols = sum(data["area"] < cfg.MIN_AREA)
                if deleted_cols > 0:
                    print("\n" + "="*60)
                    print(f"⚠️ WARNING: Removed {deleted_cols} rows with polygon geometries of less than {cfg.MIN_AREA} m2 area")
                    print("="*60 + "\n")
                    data = data[data["area"] > cfg.MIN_AREA]

            elif isinstance(data, gpd.GeoDataFrame):
                print("Computing 'area' from footprint geometries. Transforming crs to utm.")
                data.geometry = data.geometry.to_crs(data.estimate_utm_crs())
                """TODO: Geographic area for very large or multi city datasets"""
                deleted_cols = sum(data.geometry.area < cfg.MIN_AREA)
                if deleted_cols > 0:
                    print("\n" + "="*60)
                    print(f"⚠️ WARNING: Removed {deleted_cols} rows with polygon geometries of less than {cfg.MIN_AREA} m2 area")
                    print("="*60 + "\n")
                    data = data[data.geometry.area > cfg.MIN_AREA]

            data = data.reset_index(drop=True)

    if "perimeter" in cfg.FEATURES and "perimeter" not in data.columns:
        if isinstance(data, gpd.GeoDataFrame):
            print("Adding 'perimeter' column from footprint geometries. Transforming crs to utm.")
            data.geometry = data.geometry.to_crs(data.estimate_utm_crs())
            """TODO: Geographic perimeter for very large or multi city datasets"""
            data["perimeter"] = data.geometry.boundary.length 

    if isinstance(data, gpd.GeoDataFrame): 
        data = add_irregularity_features(data,cfg)
        data = add_position_features(data,cfg)

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
                raise Exception(f"Mandatory stratify column {cfg.STRATIFY_COLUMN} not in dataset columns {list(data.columns)}.")

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
            "Column mismatch detected: dataset is missing required FEATURES.\n\n"
            f"Required columns (features): {defined_columns}\n\n"
            f"Dataset columns: {actual_columns}\n\n"
            f"Missing columns (in FEATURES but not in dataset): {missing_columns}"
        )
    
    return data