"""
End-to-end dataset pipeline orchestration.

`preprocess` loads raw train/val/test sources (or their cached cleaned
versions), validates and feature-engineers them, fits the encoding pipeline
on train, applies it to val/test, and writes the final model-ready CSVs.

`load` is the thin reader used by training/validation/test scripts to pull
those model-ready CSVs (running `preprocess` first if they don't exist yet).

`save_test` writes model predictions back onto the cleaned test dataset.
"""

import os
from pathlib import Path
from typing import Literal

import pandas as pd
import geopandas as gpd

from .io import read_mixed
from .encoding import encode, get_feature_names
from .splitting import create_train_test_split


def preprocess(cfg):
    train_path = Path(cfg.TRAIN_PATH)
    if hasattr(cfg, "STRATIFY_COLUMN"):
        stratify_column = cfg.STRATIFY_COLUMN
    else:
        stratify_column = "source"

    if hasattr(cfg, "VAL_PATH"):
        val_path = cfg.VAL_PATH
    else:
        val_path = None 

    if hasattr(cfg, "VAL_SIZE"):
        val_size = cfg.VAL_SIZE
    else:
        val_size = 0 


    # Resolve output path
    output_path = Path(cfg.PREPROCESSED_OUTPUT_PATH)
    if not output_path.is_absolute():
        output_path = Path(cfg.PROJECT_ROOT) / output_path

    os.makedirs(output_path, exist_ok=True)

    if (output_path / "train_cleaned.gpkg").exists() and (output_path / "val_cleaned.gpkg").exists():
        train_df = gpd.read_file(output_path / "train_cleaned.gpkg")
        val_df = gpd.read_file(output_path / "val_cleaned.gpkg")

    elif (output_path / "train_cleaned.csv").exists() and (output_path / "val_cleaned.csv").exists():
        train_df = pd.read_csv(output_path / "train_cleaned.csv")
        val_df = pd.read_csv(output_path / "val_cleaned.csv")
    else:
        # Load data
        df, train_geofile = read_mixed(train_path, cfg, test=False)

        if val_path is None:
            train_df, val_df = create_train_test_split(
                df,
                cfg,
                stratify_col=stratify_column,
                val_size=val_size
            )
            val_geofile = train_geofile
        else:
            train_df = df
            val_df, val_geofile = read_mixed(val_path, cfg, test=False)

        all_columns = list(
            set(["id"] + cfg.FEATURES + [cfg.LABEL, stratify_column, "source", "geometry"])
            .intersection(set(train_df.columns))
        )
        train_df = train_df[all_columns].copy()
            
        all_columns = list(
            set(["id"] + cfg.FEATURES + [cfg.LABEL, stratify_column, "source", "geometry"])
            .intersection(set(val_df.columns))
        )
        val_df = val_df[all_columns].copy()

        # Save train
        if train_geofile:
            train_df.to_file(output_path / "train_cleaned.gpkg")
            train_df = pd.DataFrame(train_df.drop(columns="geometry"))
        else:
            train_df.to_csv(output_path / "train_cleaned.csv", index=False)

        # Save val
        if val_geofile:
            val_df.to_file(output_path / "val_cleaned.gpkg")
            val_df = pd.DataFrame(val_df.drop(columns="geometry"))
        else:
            val_df.to_csv(output_path / "val_cleaned.csv", index=False)

        print(f"Train/val splits saved in {output_path}")

    if hasattr(cfg, "TEST_PATH"):
        test_path = cfg.TEST_PATH
    else:
        test_path = None 
        
    if test_path is not None:
        if (output_path / "test_cleaned.gpkg").exists():
            test_df = gpd.read_file(output_path / "test_cleaned.gpkg")
        elif (output_path / "test_cleaned.csv").exists():
            test_df = pd.read_csv(output_path / "test_cleaned.csv")
        else:
            test_geofile = False
            if test_path == "VAL":
                if (output_path / "val_cleaned.gpkg").exists():
                    test_geofile = True
                    test_df = gpd.read_file(output_path / "val_cleaned.gpkg")
                elif (output_path / "val_cleaned.csv").exists():
                    test_geofile = False
                    test_df = pd.read_csv(output_path / "val_cleaned.csv")
                    
                test_df = test_df.copy().drop(columns=[cfg.LABEL])
            else:
                test_path = Path(test_path)
                if not test_path.is_absolute():
                    test_path = Path(cfg.PROJECT_ROOT) / test_path
            
                # Load data
                test_df, test_geofile = read_mixed(test_path, cfg, test=True)
                all_columns = list(
                    set(["id"] + cfg.FEATURES + [cfg.LABEL, stratify_column, "source", "geometry"])
                    .intersection(set(test_df.columns))
                )
                test_df = test_df[all_columns].copy()

            # Save test
            if test_geofile:
                test_df.to_file(output_path / "test_cleaned.gpkg")
                test_df = pd.DataFrame(test_df.drop(columns="geometry"))
            else:
                test_df.to_csv(output_path / "test_cleaned.csv", index=False)

            print(f"Preprocessed test dataset saved in {output_path}")

    # Preprocessing
    X_train, y_train, preprocessor, label_encoder = encode(
        train_df,
        cfg,
        scale_numeric=True,
        fit=True
    )

    X_val, y_val, _, _ = encode(
        val_df,
        cfg,
        preprocessor=preprocessor,
        label_encoder=label_encoder,
        fit=False
    )

    print(f"Training set shape: {X_train.shape}")
    print(f"Val set shape: {X_val.shape}")

    feature_names = get_feature_names(preprocessor, train_df, cfg)
    print(f"Feature names: {feature_names}")
    print(f"Num. features: {len(feature_names)}")

    # Build cleaned DataFrames
    train_preprocessed = pd.DataFrame(X_train, columns=feature_names)
    train_preprocessed[cfg.LABEL] = list(y_train)
    train_preprocessed["id"] = list(train_df["id"])
    train_preprocessed["id"] = train_preprocessed["id"].astype(int)

    val_preprocessed = pd.DataFrame(X_val, columns=feature_names)
    val_preprocessed[cfg.LABEL] = list(y_val)
    val_preprocessed["id"] = list(val_df["id"])
    val_preprocessed["id"] = val_preprocessed["id"].astype(int)

    # Save outputs
    train_output_path = output_path / "train.csv"
    val_output_path = output_path / "val.csv"

    train_preprocessed.to_csv(train_output_path, index=False)
    val_preprocessed.to_csv(val_output_path, index=False)

    print(f"Preprocessed train/val datasets saved in {output_path}")

    if test_path is not None:
        X_test, _ = encode(
            test_df,
            cfg,
            preprocessor=preprocessor,
            label_encoder=label_encoder,
            fit=False,
            is_test=True
        )

        print(f"Test set shape: {X_test.shape}")
        test_preprocessed = pd.DataFrame(X_test, columns=feature_names)
        test_preprocessed["id"] = list(test_df["id"])
        test_preprocessed["id"] = test_preprocessed["id"].astype(int)

        test_preprocessed_output_path = output_path / "test.csv"
        test_preprocessed.to_csv(test_preprocessed_output_path, index=False)
        print(f"Preprocessed test dataset saved in {output_path}")

def load(split: Literal["train", "val", "test"], cfg):
    df = None
    if split=="train":
        path = Path(cfg.PREPROCESSED_OUTPUT_PATH) / "train.csv"
        if path.is_file():
            df = pd.read_csv(path)
        else:
            preprocess(cfg)
            df = pd.read_csv(path)
    elif split=="val":
        path = Path(cfg.PREPROCESSED_OUTPUT_PATH) / "val.csv"
        if path.is_file():
            df = pd.read_csv(path)
        else:
            preprocess(cfg)
            df = pd.read_csv(path)
    elif split=="test":
        path = Path(cfg.PREPROCESSED_OUTPUT_PATH) / "test.csv"
        if path.is_file():
            df = pd.read_csv(path)
        else:
            preprocess(cfg)
            df = pd.read_csv(path)
    else:
        raise Exception("Argument split must be 'train' or 'val' or 'test'.")
    
    if split == "test":
        X = df.drop(columns=[col for col in [cfg.LABEL, "id"] if col in df.columns])
        return X
    else:
        X = df.drop(columns=[cfg.LABEL, "id"])
        y = df[cfg.LABEL]
        return X, y
    
def save_test(df,model_name, cfg):
    input_path = Path(cfg.PREPROCESSED_OUTPUT_PATH) 
    input_test_file = input_path / "test.csv"
    if hasattr(cfg, "LABEL_VALUES"):
        df["predicted_label"] = df["predicted_label"].map(
            dict(enumerate(cfg.LABEL_VALUES))
        )

    df = df.rename(columns={"predicted_label":cfg.LABEL})
    input_df = pd.read_csv(input_test_file)

    # Add id col
    if "id" in input_df.columns:
        df.loc[:, "id"] = input_df.loc[:, "id"].values

    is_geofile = False
    if (input_path / "test_cleaned.gpkg").exists():
        test_df = gpd.read_file(input_path / "test_cleaned.gpkg")
        is_geofile = True
    elif (input_path / "test_cleaned.csv").exists():
        test_df = pd.read_csv(input_path / "test_cleaned.csv")
        is_geofile = False

    if "id" in df.columns:
        test_df = test_df.merge(df[["id",cfg.LABEL]],on="id", how="left")
    else:
        test_df[:,cfg.LABEL] = df[:,cfg.LABEL].values

    if hasattr(cfg, "TEST_OUTPUT_PATH"):
        test_output_path = Path(cfg.TEST_OUTPUT_PATH)
    else:
        test_output_path = Path(cfg.PREPROCESSED_OUTPUT_PATH)

    if not test_output_path.is_absolute():
        test_output_path = Path(cfg.PROJECT_ROOT) / test_output_path
    
    os.makedirs(test_output_path, exist_ok=True)
    if is_geofile:
        test_output_path = test_output_path / f"test_output_{model_name}.gpkg"
        test_df.to_file(test_output_path)
    else:
        test_output_path = test_output_path / f"test_output_{model_name}.csv"
        test_df.to_csv(test_output_path)

    print(f"Test output saved as {test_output_path}")