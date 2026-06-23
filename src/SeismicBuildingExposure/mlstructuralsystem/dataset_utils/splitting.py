"""
Stratified, group-aware train/val splitting.
"""

import pandas as pd
from sklearn.model_selection import train_test_split


def create_train_test_split(df: pd.DataFrame,
                            cfg,
                            stratify_col: str|None = None,
                            val_size: float = 0.3
                        ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Performs a stratified train-val split based on the interaction of target and group columns.

    This method ensures that the distribution of the target variable within each group (e.g., city)
    is preserved across the training and valing sets. It handles the edge case where a
    stratum might have only one sample by assigning it directly to the training set.

    Args:
        df (pd.DataFrame): The input dataframe to split.
        target_col (str): The column name to stratify on (usually the label).
        group_col (str): A secondary column for group-aware stratification (e.g., 'city').
        val_size (float): The proportion of the dataset to allocate to the val set.
        random_state (int): The seed for the random number generator for reproducibility.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the training and valing dataframes.
    """
    if val_size == 0:
        raise Exception("Invalid value 0 for val_size")
    elif val_size >= 1:
        raise Exception(f"Invalid value {val_size} for val_size")
    
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
    train_df, val_df = train_test_split(
        splittable_df,
        test_size=val_size,
        random_state=cfg.RANDOM_STATE,
        stratify=splittable_df[strata_col_name]
    )

    # Add the single-sample rows back into the training set
    if not single_sample_df.empty:
        train_df = pd.concat([train_df, single_sample_df])

    # Remove the temporary stratification column before returning the dataframes
    return train_df.drop(columns=[strata_col_name]), val_df.drop(columns=[strata_col_name])