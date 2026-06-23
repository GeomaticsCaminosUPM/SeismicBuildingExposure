"""
File I/O for raw dataset ingestion.

Detects whether an input file is geospatial (geojson/gpkg/shp/geoparquet)
or plain tabular (csv/xlsx/parquet), reads it accordingly, runs it through
`check_features`, and caches the cleaned result on disk so re-runs can skip
re-reading the raw source.
"""

import os
from pathlib import Path

import pandas as pd
import geopandas as gpd
import pyarrow.parquet as pq

from .validation import check_features


def read_geofile(file, cfg, test=False):
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

    # # Select only relevant columns that exist
    # all_columns = list(
    #     set(["id"] + cfg.FEATURES + [cfg.LABEL, "source", "geometry"])
    #     .intersection(gdf.columns)
    # )

    # gdf = check_features(gdf[all_columns], cfg, test=test)

    gdf = check_features(gdf, cfg, test=test)

    return gdf

def read_tabular(file, cfg, test=False):
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

    # # Keep only relevant columns that exist
    # all_columns = list(
    #     set(["id"] + cfg.FEATURES + [cfg.LABEL, "source"])
    #     .intersection(df.columns)
    # )

    # df = check_features(df[all_columns], cfg, test=test)

    df = check_features(df, cfg, test=test)

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


def process_file(file_path, cfg, geofile=False, test=False):
    file_path = Path(file_path)
    if is_geofile(file_path):
        df_i = read_geofile(file_path, cfg, test=test)
        geofile = True
    elif is_tabular(file_path):
        df_i = read_tabular(file_path, cfg, test=test)
        if geofile:
            raise Exception(f"Combining geometry and tabular files is not supported. File {file_path} is tabular.")
            # Ensure geometry column exists if mixing with geodata
            df_i["geometry"] = None
    else:
        raise ValueError(f"Unsupported file type: {file_path}")
    
    return df_i, geofile
    
def read_mixed(path, cfg, test=False):
    path = Path(path)
    geofile = False

    # Resolve output path
    output_folder = Path(cfg.CLEANED_DATASET_PATH)
    if not output_folder.is_absolute():
        output_folder = Path(cfg.PROJECT_ROOT) / output_folder

    os.makedirs(output_folder, exist_ok=True)

    # Case 1: folder
    if path.is_dir():
        dfs = []
        for file_path in path.iterdir():
            filename_wo_ext = file_path.stem  # removes extension
            output_path = Path(output_folder) / filename_wo_ext
            if (output_path.with_suffix(".gpkg")).is_file():
                file_path = output_path.with_suffix(".gpkg")

            elif (output_path.with_suffix(".csv")).is_file():
                file_path = output_path.with_suffix(".csv")

            if file_path.is_file():
                df_i, geofile = process_file(file_path, cfg, geofile=geofile, test=test)
                if geofile:
                    df_i.to_file(output_path.with_suffix(".gpkg"))
                else:
                    df_i.to_csv(output_path.with_suffix(".csv"), index=False)

                dfs.append(df_i)

        df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    # Case 2: single file
    else:
        filename_wo_ext = path.stem  # removes extension
        output_path = Path(output_folder) / filename_wo_ext
        if (output_path.with_suffix(".gpkg")).is_file():
            file_path = output_path.with_suffix(".gpkg")

        elif (output_path.with_suffix(".csv")).is_file():
            file_path = output_path.with_suffix(".csv")
        else:
            file_path = path 

        if file_path.is_file():
            df, geofile = process_file(file_path, cfg, geofile=geofile, test=test)
            if geofile:
                df.to_file(output_path.with_suffix(".gpkg"))
            else:
                df.to_csv(output_path.with_suffix(".csv"), index=False)

    return df, geofile