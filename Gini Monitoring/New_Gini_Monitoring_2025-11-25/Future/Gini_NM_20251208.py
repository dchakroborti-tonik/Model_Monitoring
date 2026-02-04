# %% [markdown]
# # Define Library

import io
import os
import pickle
import tempfile
import time
import uuid
from datetime import datetime, timedelta
from typing import Union

import duckdb as dd
import gcsfs
import joblib
import matplotlib.pyplot as plt
import numpy as np

# %%
# %% [markdown]
# # Jupyter Notebook Loading Header
#
# This is a custom loading header for Jupyter Notebooks in Visual Studio Code.
# It includes common imports and settings to get you started quickly.
# %% [markdown]
## Import Libraries
import pandas as pd
import seaborn as sns
from google.cloud import bigquery, storage
from sklearn.metrics import roc_auc_score

path = r"C:\Users\Dwaipayan\AppData\Roaming\gcloud\legacy_credentials\dchakroborti@tonikbank.com\adc.json"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = path
client = bigquery.Client(project="prj-prod-dataplatform")
os.environ["GOOGLE_CLOUD_PROJECT"] = "prj-prod-dataplatform"

# %% [markdown]
## Configure Settings
# Set options or configurations as needed
pd.set_option("display.max_columns", None)
pd.set_option("Display.max_rows", 100)

# %% [markdown]
# # Function

# %% [markdown]
# ## calculate_gini


# %%
def calculate_gini(pd_scores, bad_indicators):
    """
    Calculate Gini coefficient from scores and binary indicators

    Parameters:
    pd_scores: array-like of scores/probabilities
    bad_indicators: array-like of binary outcomes (0/1)

    Returns:
    float: Gini coefficient
    """
    # Convert inputs to numpy arrays and ensure they're numeric
    pd_scores = np.array(pd_scores, dtype=float)
    bad_indicators = np.array(bad_indicators, dtype=int)

    # Check for valid input data
    if len(pd_scores) == 0 or len(bad_indicators) == 0:
        return np.nan

    # Check if we have both good and bad cases (needed for ROC AUC)
    if len(np.unique(bad_indicators)) < 2:
        return np.nan

    # Calculate AUC using sklearn
    try:
        auc = roc_auc_score(bad_indicators, pd_scores)
        # Calculate Gini from AUC
        gini = 2 * auc - 1
        return gini
    except ValueError:
        return np.nan


# %% [markdown]
# ## calculate_periodic_gini_prod_ver_trench


# %%
def calculate_periodic_gini_prod_ver_trench(
    df,
    score_column,
    label_column,
    namecolumn,
    model_version_column=None,
    trench_column=None,
    loan_type_column=None,
    loan_product_type_column=None,
):
    """
    Calculate periodic Gini coefficients at multiple levels:
    - Overall
    - By model version
    - By all combinations of model version, trench category, loan type, and loan product type

    Parameters:
    df: DataFrame with disbursement dates and score/label columns
    score_column: name of the score column
    label_column: name of the label column
    namecolumn: name for the bad rate label
    model_version_column: (optional) name of column for model version (e.g., 'modelVersionId')
    trench_column: (optional) name of column for trench category (e.g., 'trenchCategory')
    loan_type_column: (optional) name of loan type column (e.g., 'loan_type')
    loan_product_type_column: (optional) name of loan product type column (e.g., 'loan_product_type')
    """
    # Input validation
    required_columns = ["disbursementdate", score_column, label_column]
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"Missing required columns. Need: {required_columns}")

    optional_columns = {
        "model_version": model_version_column,
        "trench": trench_column,
        "loan_type": loan_type_column,
        "loan_product_type": loan_product_type_column,
    }

    for col_name, col in optional_columns.items():
        if col and col not in df.columns:
            raise ValueError(
                f"{col_name.replace('_', ' ').title()} column '{col}' not found in dataframe"
            )

    # Create a copy to avoid modifying original dataframe
    df = df.copy()

    # Ensure date is datetime type
    df["disbursementdate"] = pd.to_datetime(df["disbursementdate"])

    # Ensure score and label columns are numeric
    df[score_column] = pd.to_numeric(df[score_column], errors="coerce")
    df[label_column] = pd.to_numeric(df[label_column], errors="coerce")

    # Drop rows with invalid values
    df = df.dropna(subset=[score_column, label_column])

    # Define list of datasets to process
    datasets_to_process = [("Overall", df, {})]

    # Add model version level segmentation
    if model_version_column:
        for model_version in sorted(df[model_version_column].dropna().unique()):
            model_df = df[df[model_version_column] == model_version]
            metadata = {model_version_column: model_version}
            datasets_to_process.append(
                (f"ModelVersion_{model_version}", model_df, metadata)
            )

    # Add all combinations of model_version, trench, loan_type, loan_product_type
    segment_columns = []
    if model_version_column:
        segment_columns.append(("ModelVersion", model_version_column))
    if trench_column:
        segment_columns.append(("Trench", trench_column))
    if loan_type_column:
        segment_columns.append(("LoanType", loan_type_column))
    if loan_product_type_column:
        segment_columns.append(("ProductType", loan_product_type_column))

    # Generate all combinations (only if we have multiple segment columns)
    if len(segment_columns) > 1:
        # Create combinations by iterating through all segment dimensions
        def generate_combinations(
            df, segment_columns, index=0, current_filter=None, current_name=""
        ):
            if current_filter is None:
                current_filter = {}

            if index >= len(segment_columns):
                # Apply all filters and create segment
                filtered_df = df
                for col, val in current_filter.items():
                    filtered_df = filtered_df[filtered_df[col] == val]

                if len(filtered_df) > 0:
                    yield (current_name.strip("_"), filtered_df, current_filter.copy())
                return

            seg_name, seg_col = segment_columns[index]
            for seg_value in sorted(df[seg_col].dropna().unique()):
                new_filter = current_filter.copy()
                new_filter[seg_col] = seg_value
                new_name = current_name + f"{seg_name}_{seg_value}_"

                yield from generate_combinations(
                    df, segment_columns, index + 1, new_filter, new_name
                )

        for combo_name, combo_df, combo_metadata in generate_combinations(
            df, segment_columns
        ):
            datasets_to_process.append((combo_name, combo_df, combo_metadata))

    all_results = []

    # Process each dataset
    for dataset_name, dataset_df, metadata in datasets_to_process:
        # Calculate weekly Gini
        dataset_df_copy = dataset_df.copy()
        dataset_df_copy["week"] = dataset_df_copy["disbursementdate"].dt.to_period("W")
        weekly_gini = (
            dataset_df_copy.groupby("week")
            .apply(
                lambda x: (
                    calculate_gini(x[score_column], x[label_column])
                    if len(x) >= 10
                    else np.nan
                )
            )
            .reset_index(name="gini")
        )
        weekly_gini["period"] = "Week"
        weekly_gini["start_date"] = weekly_gini["week"].apply(
            lambda x: x.to_timestamp()
        )
        weekly_gini["end_date"] = weekly_gini["start_date"] + timedelta(days=6)
        weekly_gini = weekly_gini[["start_date", "end_date", "gini", "period"]]

        # Calculate monthly Gini
        dataset_df_copy = dataset_df.copy()
        dataset_df_copy["month"] = dataset_df_copy["disbursementdate"].dt.to_period("M")
        monthly_gini = (
            dataset_df_copy.groupby("month")
            .apply(
                lambda x: (
                    calculate_gini(x[score_column], x[label_column])
                    if len(x) >= 20
                    else np.nan
                )
            )
            .reset_index(name="gini")
        )
        monthly_gini["period"] = "Month"
        monthly_gini["start_date"] = monthly_gini["month"].apply(
            lambda x: x.to_timestamp()
        )
        monthly_gini["end_date"] = (
            monthly_gini["start_date"] + pd.DateOffset(months=1) - pd.Timedelta(days=1)
        )
        monthly_gini = monthly_gini[["start_date", "end_date", "gini", "period"]]

        # Combine results for this dataset
        gini_results = pd.concat([weekly_gini, monthly_gini], ignore_index=True)
        gini_results = gini_results.sort_values(by="start_date").reset_index(drop=True)

        # Add metadata columns
        gini_results["Model_Name"] = score_column
        gini_results["bad_rate"] = namecolumn
        gini_results["segment_type"] = dataset_name

        # Add individual segment components
        if model_version_column and model_version_column in metadata:
            gini_results["model_version"] = metadata[model_version_column]
        else:
            gini_results["model_version"] = None

        if trench_column and trench_column in metadata:
            gini_results["trench_category"] = metadata[trench_column]
        else:
            gini_results["trench_category"] = None

        if loan_type_column and loan_type_column in metadata:
            gini_results["loan_type"] = metadata[loan_type_column]
        else:
            gini_results["loan_type"] = None

        if loan_product_type_column and loan_product_type_column in metadata:
            gini_results["loan_product_type"] = metadata[loan_product_type_column]
        else:
            gini_results["loan_product_type"] = None

        gini_results.rename(
            columns={"gini": f"{score_column}_{namecolumn}_gini"}, inplace=True
        )

        all_results.append(gini_results)

    # Combine all results
    final_results = pd.concat(all_results, ignore_index=True)

    return final_results


# Usage:
# gini_results = calculate_periodic_gini_prod_ver_trench(
#     df_concat,
#     'Alpha_cic_sil_score',
#     'deffpd0',
#     'FPD0',
#     model_version_column='modelVersionId',
#     trench_column='trenchCategory',
#     loan_type_column='loan_type',
#     loan_product_type_column='loan_product_type'
# )

# %%
# def calculate_periodic_gini_prod_ver_trench_md(df, score_column, label_column, namecolumn,
#                                         model_version_column=None, trench_column=None,
#                                         loan_type_column=None, loan_product_type_column=None,
#                                         account_id_column=None):
#     """
#     Calculate periodic Gini coefficients at multiple levels:
#     - Overall
#     - By model version
#     - By all combinations of model version, trench category, loan type, and loan product type

#     Parameters:
#     df: DataFrame with disbursement dates and score/label columns
#     score_column: name of the score column
#     label_column: name of the label column
#     namecolumn: name for the bad rate label
#     model_version_column: (optional) name of column for model version (e.g., 'modelVersionId')
#     trench_column: (optional) name of column for trench category (e.g., 'trenchCategory')
#     loan_type_column: (optional) name of loan type column (e.g., 'loan_type')
#     loan_product_type_column: (optional) name of loan product type column (e.g., 'loan_product_type')
#     account_id_column: (optional) name of column for distinct account IDs (e.g., 'digitalLoanAccountId')
#     """
#     # Input validation
#     required_columns = ['disbursementdate', score_column, label_column]
#     if not all(col in df.columns for col in required_columns):
#         raise ValueError(f"Missing required columns. Need: {required_columns}")

#     optional_columns = {
#         'model_version': model_version_column,
#         'trench': trench_column,
#         'loan_type': loan_type_column,
#         'loan_product_type': loan_product_type_column,
#         'account_id': account_id_column
#     }

#     for col_name, col in optional_columns.items():
#         if col and col not in df.columns:
#             raise ValueError(f"{col_name.replace('_', ' ').title()} column '{col}' not found in dataframe")

#     # Create a copy to avoid modifying original dataframe
#     df = df.copy()

#     # Ensure date is datetime type
#     df['disbursementdate'] = pd.to_datetime(df['disbursementdate'])

#     # Ensure score and label columns are numeric
#     df[score_column] = pd.to_numeric(df[score_column], errors='coerce')
#     df[label_column] = pd.to_numeric(df[label_column], errors='coerce')

#     # Drop rows with invalid values
#     df = df.dropna(subset=[score_column, label_column])

#     # Define list of datasets to process
#     datasets_to_process = [('Overall', df, {})]

#     # Add model version level segmentation
#     if model_version_column:
#         for model_version in sorted(df[model_version_column].dropna().unique()):
#             model_df = df[df[model_version_column] == model_version]
#             metadata = {model_version_column: model_version}
#             datasets_to_process.append((f'ModelVersion_{model_version}', model_df, metadata))

#     # Add all combinations of model_version, trench, loan_type, loan_product_type
#     segment_columns = []
#     if model_version_column:
#         segment_columns.append(('ModelVersion', model_version_column))
#     if trench_column:
#         segment_columns.append(('Trench', trench_column))
#     if loan_type_column:
#         segment_columns.append(('LoanType', loan_type_column))
#     if loan_product_type_column:
#         segment_columns.append(('ProductType', loan_product_type_column))

#     # Generate all combinations (only if we have multiple segment columns)
#     if len(segment_columns) > 1:
#         # Create combinations by iterating through all segment dimensions
#         def generate_combinations(df, segment_columns, index=0, current_filter=None, current_name=''):
#             if current_filter is None:
#                 current_filter = {}

#             if index >= len(segment_columns):
#                 # Apply all filters and create segment
#                 filtered_df = df
#                 for col, val in current_filter.items():
#                     filtered_df = filtered_df[filtered_df[col] == val]

#                 if len(filtered_df) > 0:
#                     yield (current_name.strip('_'), filtered_df, current_filter.copy())
#                 return

#             seg_name, seg_col = segment_columns[index]
#             for seg_value in sorted(df[seg_col].dropna().unique()):
#                 new_filter = current_filter.copy()
#                 new_filter[seg_col] = seg_value
#                 new_name = current_name + f'{seg_name}_{seg_value}_'

#                 yield from generate_combinations(df, segment_columns, index + 1, new_filter, new_name)

#         for combo_name, combo_df, combo_metadata in generate_combinations(df, segment_columns):
#             datasets_to_process.append((combo_name, combo_df, combo_metadata))

#     all_results = []

#     # Process each dataset
#     for dataset_name, dataset_df, metadata in datasets_to_process:
#         # Calculate weekly Gini
#         dataset_df_copy = dataset_df.copy()
#         dataset_df_copy['week'] = dataset_df_copy['disbursementdate'].dt.to_period('W')
#         weekly_gini = dataset_df_copy.groupby('week').apply(
#             lambda x: calculate_gini(x[score_column], x[label_column])
#             if len(x) >= 10 else np.nan
#         ).reset_index(name='gini')
#         weekly_gini['period'] = 'Week'
#         weekly_gini['start_date'] = weekly_gini['week'].apply(lambda x: x.to_timestamp())
#         weekly_gini['end_date'] = weekly_gini['start_date'] + timedelta(days=6)

#         # Add distinct account count for weekly
#         if account_id_column:
#             weekly_gini['distinct_accounts'] = dataset_df_copy.groupby('week')[account_id_column].nunique().values

#         weekly_gini = weekly_gini[['start_date', 'end_date', 'gini', 'period']]

#         # Calculate monthly Gini
#         dataset_df_copy = dataset_df.copy()
#         dataset_df_copy['month'] = dataset_df_copy['disbursementdate'].dt.to_period('M')
#         monthly_gini = dataset_df_copy.groupby('month').apply(
#             lambda x: calculate_gini(x[score_column], x[label_column])
#             if len(x) >= 20 else np.nan
#         ).reset_index(name='gini')
#         monthly_gini['period'] = 'Month'
#         monthly_gini['start_date'] = monthly_gini['month'].apply(lambda x: x.to_timestamp())
#         monthly_gini['end_date'] = monthly_gini['start_date'] + pd.DateOffset(months=1) - pd.Timedelta(days=1)

#         # Add distinct account count for monthly
#         if account_id_column:
#             monthly_gini['distinct_accounts'] = dataset_df_copy.groupby('month')[account_id_column].nunique().values

#         monthly_gini = monthly_gini[['start_date', 'end_date', 'gini', 'period']]

#         # Combine results for this dataset
#         gini_results = pd.concat([weekly_gini, monthly_gini], ignore_index=True)
#         gini_results = gini_results.sort_values(by='start_date').reset_index(drop=True)

#         # Add metadata columns
#         gini_results['Model_Name'] = score_column
#         gini_results['bad_rate'] = namecolumn
#         gini_results['segment_type'] = dataset_name

#         # Add individual segment components
#         if model_version_column and model_version_column in metadata:
#             gini_results['model_version'] = metadata[model_version_column]
#         else:
#             gini_results['model_version'] = None

#         if trench_column and trench_column in metadata:
#             gini_results['trench_category'] = metadata[trench_column]
#         else:
#             gini_results['trench_category'] = None

#         if loan_type_column and loan_type_column in metadata:
#             gini_results['loan_type'] = metadata[loan_type_column]
#         else:
#             gini_results['loan_type'] = None

#         if loan_product_type_column and loan_product_type_column in metadata:
#             gini_results['loan_product_type'] = metadata[loan_product_type_column]
#         else:
#             gini_results['loan_product_type'] = None

#         gini_results.rename(columns={'gini': f'{score_column}_{namecolumn}_gini'}, inplace=True)

#         all_results.append(gini_results)

#     # Combine all results
#     final_results = pd.concat(all_results, ignore_index=True)

#     return final_results


# # Usage:
# # gini_results = calculate_periodic_gini_prod_ver_trench(
# #     df_concat,
# #     'Alpha_cic_sil_score',
# #     'deffpd0',
# #     'FPD0',
# #     model_version_column='modelVersionId',
# #     trench_column='trenchCategory',
# #     loan_type_column='loan_type',
# #     loan_product_type_column='loan_product_type',
# #     account_id_column='digitalLoanAccountId'
# # )

from datetime import timedelta
from itertools import combinations

import numpy as np

# %%
import pandas as pd


def calculate_gini(scores, labels):
    """Calculate Gini coefficient"""
    n = len(scores)
    if n < 2:
        return np.nan

    sorted_indices = np.argsort(scores)
    sorted_labels = labels.iloc[sorted_indices].values
    cumsum_labels = np.cumsum(sorted_labels)

    gini = 1 - 2 * np.sum(cumsum_labels) / (n * np.sum(sorted_labels))
    return gini


def calculate_periodic_gini_prod_ver_trench_md(
    df,
    score_column,
    label_column,
    namecolumn,
    model_version_column=None,
    trench_column=None,
    loan_type_column=None,
    loan_product_type_column=None,
    account_id_column=None,
):
    """
    Calculate periodic Gini coefficients at multiple levels:
    - Overall (entire dataset)
    - By each individual segment (model version, trench, loan type, loan product type)
    - By all possible combinations of segments

    Parameters:
    df: DataFrame with disbursement dates and score/label columns
    score_column: name of the score column
    label_column: name of the label column
    namecolumn: name for the bad rate label
    model_version_column: (optional) name of column for model version
    trench_column: (optional) name of column for trench category
    loan_type_column: (optional) name of loan type column
    loan_product_type_column: (optional) name of loan product type column
    account_id_column: (optional) name of column for distinct account IDs
    """
    # Input validation
    required_columns = ["disbursementdate", score_column, label_column]
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"Missing required columns. Need: {required_columns}")

    optional_columns = {
        "model_version": model_version_column,
        "trench": trench_column,
        "loan_type": loan_type_column,
        "loan_product_type": loan_product_type_column,
        "account_id": account_id_column,
    }

    for col_name, col in optional_columns.items():
        if col and col not in df.columns:
            raise ValueError(
                f"{col_name.replace('_', ' ').title()} column '{col}' not found in dataframe"
            )

    # Create a copy to avoid modifying original dataframe
    df = df.copy()

    # Ensure date is datetime type
    df["disbursementdate"] = pd.to_datetime(df["disbursementdate"])

    # Ensure score and label columns are numeric
    df[score_column] = pd.to_numeric(df[score_column], errors="coerce")
    df[label_column] = pd.to_numeric(df[label_column], errors="coerce")

    # Drop rows with invalid values
    df = df.dropna(subset=[score_column, label_column])

    # Define list of datasets to process
    datasets_to_process = [("Overall", df, {})]

    # Create list of available segment columns
    segment_columns = []
    if model_version_column:
        segment_columns.append(("ModelVersion", model_version_column))
    if trench_column:
        segment_columns.append(("Trench", trench_column))
    if loan_type_column:
        segment_columns.append(("LoanType", loan_type_column))
    if loan_product_type_column:
        segment_columns.append(("ProductType", loan_product_type_column))

    # Generate all possible combinations of segment columns (from single to all)
    for r in range(1, len(segment_columns) + 1):
        for combo in combinations(segment_columns, r):
            # Create combinations by iterating through all segment dimensions
            def generate_combinations(
                df, segment_columns, index=0, current_filter=None, current_name=""
            ):
                if current_filter is None:
                    current_filter = {}

                if index >= len(segment_columns):
                    # Apply all filters and create segment
                    filtered_df = df
                    for col, val in current_filter.items():
                        filtered_df = filtered_df[filtered_df[col] == val]

                    if len(filtered_df) > 0:
                        yield (
                            current_name.strip("_"),
                            filtered_df,
                            current_filter.copy(),
                        )
                    return

                seg_name, seg_col = segment_columns[index]
                for seg_value in sorted(df[seg_col].dropna().unique()):
                    new_filter = current_filter.copy()
                    new_filter[seg_col] = seg_value
                    new_name = current_name + f"{seg_name}_{seg_value}_"

                    yield from generate_combinations(
                        df, segment_columns, index + 1, new_filter, new_name
                    )

            for combo_name, combo_df, combo_metadata in generate_combinations(
                df, list(combo)
            ):
                datasets_to_process.append((combo_name, combo_df, combo_metadata))

    all_results = []

    # Process each dataset
    for dataset_name, dataset_df, metadata in datasets_to_process:
        # Calculate weekly Gini
        dataset_df_copy = dataset_df.copy()
        dataset_df_copy["week"] = dataset_df_copy["disbursementdate"].dt.to_period("W")
        weekly_gini = (
            dataset_df_copy.groupby("week")
            .apply(
                lambda x: (
                    calculate_gini(x[score_column], x[label_column])
                    if len(x) >= 10
                    else np.nan
                )
            )
            .reset_index(name="gini")
        )
        weekly_gini["period"] = "Week"
        weekly_gini["start_date"] = weekly_gini["week"].apply(
            lambda x: x.to_timestamp()
        )
        weekly_gini["end_date"] = weekly_gini["start_date"] + timedelta(days=6)

        # Add distinct account count for weekly
        if account_id_column:
            weekly_account_counts = (
                dataset_df_copy.groupby("week")[account_id_column]
                .nunique()
                .reset_index()
            )
            weekly_account_counts.columns = ["week", "distinct_accounts"]
            weekly_gini = weekly_gini.merge(
                weekly_account_counts, on="week", how="left"
            )

        weekly_gini = weekly_gini[
            ["start_date", "end_date", "gini", "period"]
            + (["distinct_accounts"] if account_id_column else [])
        ]

        # Calculate monthly Gini
        dataset_df_copy = dataset_df.copy()
        dataset_df_copy["month"] = dataset_df_copy["disbursementdate"].dt.to_period("M")
        monthly_gini = (
            dataset_df_copy.groupby("month")
            .apply(
                lambda x: (
                    calculate_gini(x[score_column], x[label_column])
                    if len(x) >= 20
                    else np.nan
                )
            )
            .reset_index(name="gini")
        )
        monthly_gini["period"] = "Month"
        monthly_gini["start_date"] = monthly_gini["month"].apply(
            lambda x: x.to_timestamp()
        )
        monthly_gini["end_date"] = (
            monthly_gini["start_date"] + pd.DateOffset(months=1) - pd.Timedelta(days=1)
        )

        # Add distinct account count for monthly
        if account_id_column:
            monthly_account_counts = (
                dataset_df_copy.groupby("month")[account_id_column]
                .nunique()
                .reset_index()
            )
            monthly_account_counts.columns = ["month", "distinct_accounts"]
            monthly_gini = monthly_gini.merge(
                monthly_account_counts, on="month", how="left"
            )

        monthly_gini = monthly_gini[
            ["start_date", "end_date", "gini", "period"]
            + (["distinct_accounts"] if account_id_column else [])
        ]

        # Combine results for this dataset
        gini_results = pd.concat([weekly_gini, monthly_gini], ignore_index=True)
        gini_results = gini_results.sort_values(by="start_date").reset_index(drop=True)

        # Add metadata columns
        gini_results["Model_Name"] = score_column
        gini_results["bad_rate"] = namecolumn
        gini_results["segment_type"] = dataset_name

        # Add individual segment components
        if model_version_column and model_version_column in metadata:
            gini_results["model_version"] = metadata[model_version_column]
        else:
            gini_results["model_version"] = None

        if trench_column and trench_column in metadata:
            gini_results["trench_category"] = metadata[trench_column]
        else:
            gini_results["trench_category"] = None

        if loan_type_column and loan_type_column in metadata:
            gini_results["loan_type"] = metadata[loan_type_column]
        else:
            gini_results["loan_type"] = None

        if loan_product_type_column and loan_product_type_column in metadata:
            gini_results["loan_product_type"] = metadata[loan_product_type_column]
        else:
            gini_results["loan_product_type"] = None

        gini_results.rename(
            columns={"gini": f"{score_column}_{namecolumn}_gini"}, inplace=True
        )

        all_results.append(gini_results)

    # Combine all results
    final_results = pd.concat(all_results, ignore_index=True)

    return final_results


# Usage:
# gini_results = calculate_periodic_gini_prod_ver_trench_md(
#     df_concat,
#     'Alpha_cic_sil_score',
#     'deffpd0',
#     'FPD0',
#     model_version_column='modelVersionId',
#     trench_column='trenchCategory',
#     loan_type_column='loan_type',
#     loan_product_type_column='loan_product_type',
#     account_id_column='digitalLoanAccountId'
# )

from datetime import timedelta
from itertools import combinations

import numpy as np

# %%
import pandas as pd


def calculate_gini(scores, labels):
    """Calculate Gini coefficient"""
    n = len(scores)
    if n < 2:
        return np.nan

    sorted_indices = np.argsort(scores)
    sorted_labels = labels.iloc[sorted_indices].values
    cumsum_labels = np.cumsum(sorted_labels)

    gini = 1 - 2 * np.sum(cumsum_labels) / (n * np.sum(sorted_labels))
    return gini


def calculate_periodic_gini_prod_ver_trench_md_col(
    df,
    score_column,
    label_column,
    namecolumn,
    model_version_column=None,
    trench_column=None,
    loan_type_column=None,
    loan_product_type_column=None,
    account_id_column=None,
):
    """
    Calculate periodic Gini coefficients at multiple levels (wide format):
    - Overall (entire dataset)
    - By each individual segment (model version, trench, loan type, loan product type)
    - By all possible combinations of segments

    Returns wide format with one row per period and separate columns for each segment_type's Gini.

    Parameters:
    df: DataFrame with disbursement dates and score/label columns
    score_column: name of the score column
    label_column: name of the label column
    namecolumn: name for the bad rate label
    model_version_column: (optional) name of column for model version
    trench_column: (optional) name of column for trench category
    loan_type_column: (optional) name of loan type column
    loan_product_type_column: (optional) name of loan product type column
    account_id_column: (optional) name of column for distinct account IDs
    """
    # Input validation
    required_columns = ["disbursementdate", score_column, label_column]
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"Missing required columns. Need: {required_columns}")

    optional_columns = {
        "model_version": model_version_column,
        "trench": trench_column,
        "loan_type": loan_type_column,
        "loan_product_type": loan_product_type_column,
        "account_id": account_id_column,
    }

    for col_name, col in optional_columns.items():
        if col and col not in df.columns:
            raise ValueError(
                f"{col_name.replace('_', ' ').title()} column '{col}' not found in dataframe"
            )

    # Create a copy to avoid modifying original dataframe
    df = df.copy()

    # Ensure date is datetime type
    df["disbursementdate"] = pd.to_datetime(df["disbursementdate"])

    # Ensure score and label columns are numeric
    df[score_column] = pd.to_numeric(df[score_column], errors="coerce")
    df[label_column] = pd.to_numeric(df[label_column], errors="coerce")

    # Drop rows with invalid values
    df = df.dropna(subset=[score_column, label_column])

    # Define list of datasets to process
    datasets_to_process = [("Overall", df, {})]

    # Create list of available segment columns
    segment_columns = []
    if model_version_column:
        segment_columns.append(("ModelVersion", model_version_column))
    if trench_column:
        segment_columns.append(("Trench", trench_column))
    if loan_type_column:
        segment_columns.append(("LoanType", loan_type_column))
    if loan_product_type_column:
        segment_columns.append(("ProductType", loan_product_type_column))

    # Generate all possible combinations of segment columns (from single to all)
    for r in range(1, len(segment_columns) + 1):
        for combo in combinations(segment_columns, r):
            # Create combinations by iterating through all segment dimensions
            def generate_combinations(
                df, segment_columns, index=0, current_filter=None, current_name=""
            ):
                if current_filter is None:
                    current_filter = {}

                if index >= len(segment_columns):
                    # Apply all filters and create segment
                    filtered_df = df
                    for col, val in current_filter.items():
                        filtered_df = filtered_df[filtered_df[col] == val]

                    if len(filtered_df) > 0:
                        yield (
                            current_name.strip("_"),
                            filtered_df,
                            current_filter.copy(),
                        )
                    return

                seg_name, seg_col = segment_columns[index]
                for seg_value in sorted(df[seg_col].dropna().unique()):
                    new_filter = current_filter.copy()
                    new_filter[seg_col] = seg_value
                    new_name = current_name + f"{seg_name}_{seg_value}_"

                    yield from generate_combinations(
                        df, segment_columns, index + 1, new_filter, new_name
                    )

            for combo_name, combo_df, combo_metadata in generate_combinations(
                df, list(combo)
            ):
                datasets_to_process.append((combo_name, combo_df, combo_metadata))

    all_results = []

    # Process each dataset
    for dataset_name, dataset_df, metadata in datasets_to_process:
        # Calculate weekly Gini
        dataset_df_copy = dataset_df.copy()
        dataset_df_copy["week"] = dataset_df_copy["disbursementdate"].dt.to_period("W")
        weekly_gini = (
            dataset_df_copy.groupby("week")
            .apply(
                lambda x: (
                    calculate_gini(x[score_column], x[label_column])
                    if len(x) >= 10
                    else np.nan
                )
            )
            .reset_index(name="gini")
        )
        weekly_gini["period"] = "Week"
        weekly_gini["start_date"] = weekly_gini["week"].apply(
            lambda x: x.to_timestamp()
        )
        weekly_gini["end_date"] = weekly_gini["start_date"] + timedelta(days=6)
        weekly_gini = weekly_gini[["start_date", "end_date", "gini", "period"]]

        # Calculate monthly Gini
        dataset_df_copy = dataset_df.copy()
        dataset_df_copy["month"] = dataset_df_copy["disbursementdate"].dt.to_period("M")
        monthly_gini = (
            dataset_df_copy.groupby("month")
            .apply(
                lambda x: (
                    calculate_gini(x[score_column], x[label_column])
                    if len(x) >= 20
                    else np.nan
                )
            )
            .reset_index(name="gini")
        )
        monthly_gini["period"] = "Month"
        monthly_gini["start_date"] = monthly_gini["month"].apply(
            lambda x: x.to_timestamp()
        )
        monthly_gini["end_date"] = (
            monthly_gini["start_date"] + pd.DateOffset(months=1) - pd.Timedelta(days=1)
        )
        monthly_gini = monthly_gini[["start_date", "end_date", "gini", "period"]]

        # Combine results for this dataset
        gini_results = pd.concat([weekly_gini, monthly_gini], ignore_index=True)
        gini_results = gini_results.sort_values(by="start_date").reset_index(drop=True)

        # Create column name for this segment's Gini
        gini_column_name = f"{score_column}_{namecolumn}_gini_{dataset_name}"
        gini_results[gini_column_name] = gini_results["gini"]
        gini_results = gini_results.drop("gini", axis=1)

        all_results.append(gini_results)

    # Merge all results on start_date, end_date, and period
    final_results = all_results[0]
    for result_df in all_results[1:]:
        final_results = final_results.merge(
            result_df, on=["start_date", "end_date", "period"], how="outer"
        )

    # Add metadata columns (these will be the same for each row)
    final_results["Model_Name"] = score_column
    final_results["bad_rate"] = namecolumn

    if model_version_column:
        final_results["model_version"] = None
    else:
        final_results["model_version"] = None

    if trench_column:
        final_results["trench_category"] = None
    else:
        final_results["trench_category"] = None

    if loan_type_column:
        final_results["loan_type"] = None
    else:
        final_results["loan_type"] = None

    if loan_product_type_column:
        final_results["loan_product_type"] = None
    else:
        final_results["loan_product_type"] = None

    # Reorder columns: metadata first, then all Gini columns
    metadata_cols = [
        "start_date",
        "end_date",
        "period",
        "Model_Name",
        "bad_rate",
        "model_version",
        "trench_category",
        "loan_type",
        "loan_product_type",
    ]
    gini_cols = [
        col
        for col in final_results.columns
        if col.startswith(f"{score_column}_{namecolumn}_gini_")
    ]

    final_results = final_results[metadata_cols + gini_cols]

    return final_results


# Usage:
# gini_results = calculate_periodic_gini_prod_ver_trench_md(
#     df_concat,
#     'Alpha_cic_sil_score',
#     'deffpd0',
#     'FPD0',
#     model_version_column='modelVersionId',
#     trench_column='trenchCategory',
#     loan_type_column='loan_type',
#     loan_product_type_column='loan_product_type',
#     account_id_column='digitalLoanAccountId'
# )

from datetime import timedelta
from itertools import combinations

import numpy as np

# %%
import pandas as pd


def calculate_gini(scores, labels):
    """
    Calculate Gini coefficient with proper handling of edge cases

    Returns np.nan when:
    - Fewer than 2 observations
    - No positive labels (sum of labels = 0)
    """
    n = len(scores)
    if n < 2:
        return np.nan

    label_sum = np.sum(labels)

    # Handle case where no positive labels exist (all zeros)
    # This prevents division by zero warning
    if label_sum == 0:
        return np.nan

    sorted_indices = np.argsort(scores)
    sorted_labels = labels.iloc[sorted_indices].values
    cumsum_labels = np.cumsum(sorted_labels)

    gini = 1 - 2 * np.sum(cumsum_labels) / (n * label_sum)
    return gini


def calculate_periodic_gini_prod_ver_trench_dimfact(
    df,
    score_column,
    label_column,
    namecolumn,
    model_version_column=None,
    trench_column=None,
    loan_type_column=None,
    loan_product_type_column=None,
    account_id_column=None,
):
    """
    Calculate periodic Gini coefficients and return Power BI-friendly long format
    with fact and dimension tables.

    Returns:
    - fact_table: Long format with one row per segment per period
    - dimension_table: Unique segment combinations for filtering

    Parameters:
    df: DataFrame with disbursement dates and score/label columns
    score_column: name of the score column
    label_column: name of the label column
    namecolumn: name for the bad rate label
    model_version_column: (optional) name of column for model version
    trench_column: (optional) name of column for trench category
    loan_type_column: (optional) name of loan type column
    loan_product_type_column: (optional) name of loan product type column
    account_id_column: (optional) name of column for distinct account IDs
    """
    # Input validation
    required_columns = ["disbursementdate", score_column, label_column]
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"Missing required columns. Need: {required_columns}")

    optional_columns = {
        "model_version": model_version_column,
        "trench": trench_column,
        "loan_type": loan_type_column,
        "loan_product_type": loan_product_type_column,
        "account_id": account_id_column,
    }

    for col_name, col in optional_columns.items():
        if col and col not in df.columns:
            raise ValueError(
                f"{col_name.replace('_', ' ').title()} column '{col}' not found in dataframe"
            )

    # Create a copy to avoid modifying original dataframe
    df = df.copy()

    # Ensure date is datetime type
    df["disbursementdate"] = pd.to_datetime(df["disbursementdate"])

    # Ensure score and label columns are numeric
    df[score_column] = pd.to_numeric(df[score_column], errors="coerce")
    df[label_column] = pd.to_numeric(df[label_column], errors="coerce")

    # Drop rows with invalid values
    df = df.dropna(subset=[score_column, label_column])

    # Define list of datasets to process
    datasets_to_process = [("Overall", df, {})]

    # Create list of available segment columns
    segment_columns = []
    if model_version_column:
        segment_columns.append(("ModelVersion", model_version_column))
    if trench_column:
        segment_columns.append(("Trench", trench_column))
    if loan_type_column:
        segment_columns.append(("LoanType", loan_type_column))
    if loan_product_type_column:
        segment_columns.append(("ProductType", loan_product_type_column))

    # Generate all possible combinations of segment columns
    for r in range(1, len(segment_columns) + 1):
        for combo in combinations(segment_columns, r):

            def generate_combinations(
                df, segment_columns, index=0, current_filter=None, current_name=""
            ):
                if current_filter is None:
                    current_filter = {}

                if index >= len(segment_columns):
                    filtered_df = df
                    for col, val in current_filter.items():
                        filtered_df = filtered_df[filtered_df[col] == val]

                    if len(filtered_df) > 0:
                        yield (
                            current_name.strip("_"),
                            filtered_df,
                            current_filter.copy(),
                        )
                    return

                seg_name, seg_col = segment_columns[index]
                for seg_value in sorted(df[seg_col].dropna().unique()):
                    new_filter = current_filter.copy()
                    new_filter[seg_col] = seg_value
                    new_name = current_name + f"{seg_name}_{seg_value}_"

                    yield from generate_combinations(
                        df, segment_columns, index + 1, new_filter, new_name
                    )

            for combo_name, combo_df, combo_metadata in generate_combinations(
                df, list(combo)
            ):
                datasets_to_process.append((combo_name, combo_df, combo_metadata))

    all_results = []

    # Process each dataset
    for dataset_name, dataset_df, metadata in datasets_to_process:
        # Calculate weekly Gini
        dataset_df_copy = dataset_df.copy()
        dataset_df_copy["week"] = dataset_df_copy["disbursementdate"].dt.to_period("W")
        weekly_gini = (
            dataset_df_copy.groupby("week")
            .apply(
                lambda x: (
                    calculate_gini(x[score_column], x[label_column])
                    if len(x) >= 10
                    else np.nan
                )
            )
            .reset_index(name="gini_value")
        )
        weekly_gini["period"] = "Week"
        weekly_gini["start_date"] = weekly_gini["week"].apply(
            lambda x: x.to_timestamp()
        )
        weekly_gini["end_date"] = weekly_gini["start_date"] + timedelta(days=6)

        # Add distinct account count for weekly
        if account_id_column:
            weekly_account_counts = (
                dataset_df_copy.groupby("week")[account_id_column]
                .nunique()
                .reset_index()
            )
            weekly_account_counts.columns = ["week", "distinct_accounts"]
            weekly_gini = weekly_gini.merge(
                weekly_account_counts, on="week", how="left"
            )
        else:
            weekly_gini["distinct_accounts"] = None

        weekly_gini = weekly_gini[
            ["start_date", "end_date", "gini_value", "period", "distinct_accounts"]
        ]

        # Calculate monthly Gini
        dataset_df_copy = dataset_df.copy()
        dataset_df_copy["month"] = dataset_df_copy["disbursementdate"].dt.to_period("M")
        monthly_gini = (
            dataset_df_copy.groupby("month")
            .apply(
                lambda x: (
                    calculate_gini(x[score_column], x[label_column])
                    if len(x) >= 20
                    else np.nan
                )
            )
            .reset_index(name="gini_value")
        )
        monthly_gini["period"] = "Month"
        monthly_gini["start_date"] = monthly_gini["month"].apply(
            lambda x: x.to_timestamp()
        )
        monthly_gini["end_date"] = (
            monthly_gini["start_date"] + pd.DateOffset(months=1) - pd.Timedelta(days=1)
        )

        # Add distinct account count for monthly
        if account_id_column:
            monthly_account_counts = (
                dataset_df_copy.groupby("month")[account_id_column]
                .nunique()
                .reset_index()
            )
            monthly_account_counts.columns = ["month", "distinct_accounts"]
            monthly_gini = monthly_gini.merge(
                monthly_account_counts, on="month", how="left"
            )
        else:
            monthly_gini["distinct_accounts"] = None

        monthly_gini = monthly_gini[
            ["start_date", "end_date", "gini_value", "period", "distinct_accounts"]
        ]

        # Combine results for this dataset
        gini_results = pd.concat([weekly_gini, monthly_gini], ignore_index=True)
        gini_results = gini_results.sort_values(by="start_date").reset_index(drop=True)

        # Add metadata columns
        gini_results["Model_Name"] = score_column
        gini_results["bad_rate"] = namecolumn
        gini_results["segment_type"] = dataset_name

        # Add individual segment components
        gini_results["model_version"] = (
            metadata.get(model_version_column, None) if model_version_column else None
        )
        gini_results["trench_category"] = (
            metadata.get(trench_column, None) if trench_column else None
        )
        gini_results["loan_type"] = (
            metadata.get(loan_type_column, None) if loan_type_column else None
        )
        gini_results["loan_product_type"] = (
            metadata.get(loan_product_type_column, None)
            if loan_product_type_column
            else None
        )

        all_results.append(gini_results)

    # Combine all results
    fact_table = pd.concat(all_results, ignore_index=True)

    # Create dimension table (unique segment combinations for filtering)
    dimension_table = (
        fact_table[
            [
                "Model_Name",
                "bad_rate",
                "segment_type",
                "model_version",
                "trench_category",
                "loan_type",
                "loan_product_type",
            ]
        ]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    dimension_table["segment_id"] = range(len(dimension_table))

    # Add segment_id to fact table
    fact_table = fact_table.merge(
        dimension_table[
            [
                "segment_id",
                "Model_Name",
                "bad_rate",
                "segment_type",
                "model_version",
                "trench_category",
                "loan_type",
                "loan_product_type",
            ]
        ],
        on=[
            "Model_Name",
            "bad_rate",
            "segment_type",
            "model_version",
            "trench_category",
            "loan_type",
            "loan_product_type",
        ],
        how="left",
    )

    # Reorder columns in fact table
    fact_table = fact_table[
        [
            "segment_id",
            "start_date",
            "end_date",
            "period",
            "gini_value",
            "distinct_accounts",
            "Model_Name",
            "bad_rate",
            "segment_type",
            "model_version",
            "trench_category",
            "loan_type",
            "loan_product_type",
        ]
    ]

    return fact_table, dimension_table


# Usage:
# fact_table, dimension_table = calculate_periodic_gini_prod_ver_trench_dimfact(
#     df_concat,
#     'Alpha_cic_sil_score',
#     'deffpd0',
#     'FPD0',
#     model_version_column='modelVersionId',
#     trench_column='trenchCategory',
#     loan_type_column='loan_type',
#     loan_product_type_column='loan_product_type',
#     account_id_column='digitalLoanAccountId'
# )
#
# # In Power BI:
# # 1. Import fact_table and dimension_table
# # 2. Create relationship: dimension_table[segment_id] -> fact_table[segment_id]
# # 3. Use dimension table columns as filters
# # 4. Create DAX measures:
# #    - Gini Measure = AVERAGE(fact_table[gini_value])
# #    - Account Count = SUM(fact_table[distinct_accounts])
# # 5. Use start_date, end_date, period for time-based analysis

# %% [markdown]
# # update_tables

# %%


def update_tables(
    fact_table: pd.DataFrame,
    dimension_table: pd.DataFrame,
    model_name: str,
    product: str,
) -> tuple:
    """
    Updates fact_table and dimension_table:
    - Sets 'Model_display_name' to the given model_name
    - Replaces NaN values in specified columns with 'Overall'

    Returns:
        Updated fact_table and dimension_table as a tuple
    """
    # Columns where missing values should be replaced
    cols_to_replace = [
        "model_version",
        "trench_category",
        "loan_type",
        "loan_product_type",
    ]

    # Update fact_table
    fact_table["Model_display_name"] = model_name
    fact_table["Product_Category"] = product
    fact_table[cols_to_replace] = fact_table[cols_to_replace].fillna("Overall")

    # Update dimension_table
    dimension_table["Model_display_name"] = model_name
    dimension_table["Product_Category"] = product
    dimension_table[cols_to_replace] = dimension_table[cols_to_replace].fillna(
        "Overall"
    )

    return fact_table, dimension_table


# %% [markdown]
# # SIL

# %% [markdown]
# ## cic_model_sil

# %% [markdown]
# ### FPD0

# %% [markdown]
# ### Test

# %%
sq = """ 
with modelname as 
  (SELECT
    customerId,digitalLoanAccountId,prediction Alpha_cic_sil_score,start_time,end_time,modelDisplayName,modelVersionId, 
    case when trenchCategory is null then 'ALL'
         when trenchCategory = '' then 'ALL'
         else trenchCategory end as trenchCategory,
    REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature
  FROM prj-prod-dataplatform.audit_balance.ml_model_run_details
  WHERE modelDisplayName in ('Alpha - CIC-SIL-Model', 'cic_model_sil', 'Sil-Alpha-CIC-SIL-Model')
  ),
  deliquency as
(select loanAccountNumber,
case when obs_min_inst_def0 >= 1 and min_inst_def0 = 1 then 1 else 0 end deffpd0,
case when obs_min_inst_def10 >=1 and min_inst_def10 =1 then 1 else 0 end deffpd10,
case when obs_min_inst_def30 >=1 and min_inst_def30 =1 then 1 else 0 end deffpd30,
case when obs_min_inst_def30 >=2 and min_inst_def30 in (1,2) then 1 else 0 end deffspd30,
case when obs_min_inst_def30 >=3 and min_inst_def30 in (1,2,3) then 1 else 0 end deffstpd30,
case when obs_min_inst_def0 >= 1 then 1 else 0 end flg_mature_fpd0,
case when obs_min_inst_def10 >=1 then 1 else 0 end flg_mature_fpd10,
case when obs_min_inst_def30 >=1 then 1 else 0 end flg_mature_fpd30,
case when obs_min_inst_def30 >=2 then 1 else 0 end flg_mature_fspd_30,
case when obs_min_inst_def30 >=3 then 1 else 0 end flg_mature_fstpd_30
from prj-prod-dataplatform.risk_credit_mis.loan_deliquency_data),
base as 
  (select distinct r.customerId,
  r.digitalLoanAccountId,
  loanmaster.loanAccountNumber,
  r.modelDisplayName,
  r.Alpha_cic_sil_score,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month, 
  'Test' Data_selection,  
  deffpd0,
  flg_mature_fpd0,
  loanmaster.new_loan_type,
  modelVersionId, trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile' 
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall' 
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
  from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  inner join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
   left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
  where loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and r.Alpha_cic_sil_score is not null
  and flg_mature_fpd0 = 1
  )
  select *  from base
  ;
"""
dfd = client.query(sq).to_dataframe()
dfd = dfd.drop_duplicates(keep="first")
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()


# %%
df1 = dfd.copy()

# %% [markdown]
# ### Train

# %%
sq = """ 
  with modelname as 
  (SELECT
    customerId,digitalLoanAccountId,prediction Alpha_cic_sil_score,start_time,end_time,modelDisplayName,modelVersionId, 
    case when trenchCategory is null then 'ALL' 
        when trenchCategory = ''then 'ALL' 
        else trenchCategory end as trenchCategory,
    REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature
  FROM prj-prod-dataplatform.dap_ds_poweruser_playground.ml_training_model_run_details
  WHERE modelDisplayName in ('Alpha - CIC-SIL-Model', 'cic_model_sil', 'Sil-Alpha-CIC-SIL-Model')
  ),
  deliquency as
(select loanAccountNumber,
case when obs_min_inst_def0 >= 1 and min_inst_def0 = 1 then 1 else 0 end deffpd0,
case when obs_min_inst_def10 >=1 and min_inst_def10 =1 then 1 else 0 end deffpd10,
case when obs_min_inst_def30 >=1 and min_inst_def30 =1 then 1 else 0 end deffpd30,
case when obs_min_inst_def30 >=2 and min_inst_def30 in (1,2) then 1 else 0 end deffspd30,
case when obs_min_inst_def30 >=3 and min_inst_def30 in (1,2,3) then 1 else 0 end deffstpd30,
case when obs_min_inst_def0 >= 1 then 1 else 0 end flg_mature_fpd0,
case when obs_min_inst_def10 >=1 then 1 else 0 end flg_mature_fpd10,
case when obs_min_inst_def30 >=1 then 1 else 0 end flg_mature_fpd30,
case when obs_min_inst_def30 >=2 then 1 else 0 end flg_mature_fspd_30,
case when obs_min_inst_def30 >=3 then 1 else 0 end flg_mature_fstpd_30
from prj-prod-dataplatform.risk_credit_mis.loan_deliquency_data),
base as
(select distinct r.customerId,
  r.digitalLoanAccountId,
  loanmaster.loanAccountNumber,
  r.modelDisplayName,
  r.Alpha_cic_sil_score,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month, 
  'Train' Data_selection,  
    deffpd0,
  flg_mature_fpd0,
  loanmaster.new_loan_type,
  modelVersionId,
    trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile' 
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall' 
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
    from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  inner join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
  left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
  where loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and r.Alpha_cic_sil_score is not null
  and flg_mature_fpd0 = 1
  )
  select * from base;
  """
dfd = client.query(sq).to_dataframe()
dfd = dfd.drop_duplicates(keep="first")
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()

# %%
df2 = dfd.copy()

# %% [markdown]
# ### Concat

# %%
# df_concat = pd.concat([df1, df2], ignore_index=True)
# print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")

# df_concat = (df_concat
#              .sort_values(by=['digitalLoanAccountId', 'Data_selection'],
#                           key=lambda s: s.map({'Train': 0, 'Test': 1}))
#              .drop_duplicates(subset=['digitalLoanAccountId'], keep='first')
#              .reset_index(drop=True))
# print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")
# df_concat.head()


# 1) Get all IDs present in Train
train_ids = set(df2["digitalLoanAccountId"])

# 2) Keep only Test rows whose ID is NOT in Train
df1_no_dupes = df1[~df1["digitalLoanAccountId"].isin(train_ids)]

# 3) Concatenate
df_concat = pd.concat([df1_no_dupes, df2], ignore_index=True)

print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")
df_concat.head()


# %%
df_concat.info()

# %%
df_concat.groupby(
    ["Data_selection", "new_loan_type", "modelVersionId", "loan_product_type"]
)["digitalLoanAccountId"].nunique()

# %%
df_concat["new_loan_type"].unique()

# %%
df_concat["Alpha_cic_sil_score"] = pd.to_numeric(
    df_concat["Alpha_cic_sil_score"], errors="coerce"
)

# %%
# df_concat.to_csv(r"D:\OneDrive - Tonik Financial Pte Ltd\MyStuff\Data Engineering\Model_Monitoring\Gini Monitoring\New_Gini_Monitoring_2025-11-25\Future\sample.csv", index = False)

# %%
fact_table, dimension_table = calculate_periodic_gini_prod_ver_trench_dimfact(
    df_concat,
    "Alpha_cic_sil_score",
    "deffpd0",
    "FPD0",
    model_version_column="modelVersionId",
    trench_column="trenchCategory",
    loan_type_column="new_loan_type",
    loan_product_type_column="loan_product_type",
    account_id_column="digitalLoanAccountId",
)

# %%
fact_table, dimension_table = update_tables(
    fact_table, dimension_table, "cic_model_sil", "SIL"
)

# %%
# print(fact_table.columns)
# fact_table['segment_type'].value_counts(dropna=False)
# fact_table['Model_display_name'] = 'cic_model_sil'

# %%
# # Columns where missing values should be replaced
# cols_to_replace = ['model_version', 'trench_category', 'loan_type', 'loan_product_type']

# # Replace NaN values with 'Overall'
# fact_table[cols_to_replace] = fact_table[cols_to_replace].fillna('Overall')

# fact_table.head()


# %%
# dimension_table['Model_display_name'] = 'cic_model_sil'

# %%
# # Columns where missing values should be replaced
# cols_to_replace = ['model_version', 'trench_category', 'loan_type', 'loan_product_type']

# # Replace NaN values with 'Overall'
# dimension_table[cols_to_replace] = dimension_table[cols_to_replace].fillna('Overall')

# dimension_table.head()

# %%
# Upload to BigQuery
table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_table"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_TRUNCATE",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(fact_table, table_id, job_config=job_config)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_table"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_TRUNCATE",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(dimension_table, table_id, job_config=job_config)
job.result()  # Wait for the job to complete

# %% [markdown]
# ### FPD10

# %% [markdown]
# ### Test

# %%
sq = """ 
with modelname as 
  (SELECT
    customerId,digitalLoanAccountId,prediction Alpha_cic_sil_score,start_time,end_time,modelDisplayName,modelVersionId, 
    case when trenchCategory is null then 'ALL'
         when trenchCategory = '' then 'ALL'
         else trenchCategory end as trenchCategory,
    REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature
  FROM prj-prod-dataplatform.audit_balance.ml_model_run_details
  WHERE modelDisplayName in ('Alpha - CIC-SIL-Model', 'cic_model_sil', 'Sil-Alpha-CIC-SIL-Model')
  ),
  deliquency as
(select loanAccountNumber,
case when obs_min_inst_def0 >= 1 and min_inst_def0 = 1 then 1 else 0 end deffpd0,
case when obs_min_inst_def10 >=1 and min_inst_def10 =1 then 1 else 0 end deffpd10,
case when obs_min_inst_def30 >=1 and min_inst_def30 =1 then 1 else 0 end deffpd30,
case when obs_min_inst_def30 >=2 and min_inst_def30 in (1,2) then 1 else 0 end deffspd30,
case when obs_min_inst_def30 >=3 and min_inst_def30 in (1,2,3) then 1 else 0 end deffstpd30,
case when obs_min_inst_def0 >= 1 then 1 else 0 end flg_mature_fpd0,
case when obs_min_inst_def10 >=1 then 1 else 0 end flg_mature_fpd10,
case when obs_min_inst_def30 >=1 then 1 else 0 end flg_mature_fpd30,
case when obs_min_inst_def30 >=2 then 1 else 0 end flg_mature_fspd_30,
case when obs_min_inst_def30 >=3 then 1 else 0 end flg_mature_fstpd_30
from prj-prod-dataplatform.risk_credit_mis.loan_deliquency_data),
base as 
  (select distinct r.customerId,
  r.digitalLoanAccountId,
  loanmaster.loanAccountNumber,
  r.modelDisplayName,
  r.Alpha_cic_sil_score,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month, 
  'Test' Data_selection,  
  del.deffpd10,
  del.flg_mature_fpd10,
  loanmaster.new_loan_type,
  modelVersionId, trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile' 
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall' 
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
  from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  inner join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
   left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
  where loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and r.Alpha_cic_sil_score is not null
  and del.flg_mature_fpd10 = 1
  )
  select *  from base
  ;
"""
dfd = client.query(sq).to_dataframe()
dfd = dfd.drop_duplicates(keep="first")
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()


# %%
df1 = dfd.copy()

# %% [markdown]
# ### Train

# %%
sq = """ 
  with modelname as 
  (SELECT
    customerId,digitalLoanAccountId,prediction Alpha_cic_sil_score,start_time,end_time,modelDisplayName,modelVersionId, 
    case when trenchCategory is null then 'ALL' 
        when trenchCategory = ''then 'ALL' 
        else trenchCategory end as trenchCategory,
    REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature
  FROM prj-prod-dataplatform.dap_ds_poweruser_playground.ml_training_model_run_details
  WHERE modelDisplayName in ('Alpha - CIC-SIL-Model', 'cic_model_sil', 'Sil-Alpha-CIC-SIL-Model')
  ),
  deliquency as
(select loanAccountNumber,
case when obs_min_inst_def0 >= 1 and min_inst_def0 = 1 then 1 else 0 end deffpd0,
case when obs_min_inst_def10 >=1 and min_inst_def10 =1 then 1 else 0 end deffpd10,
case when obs_min_inst_def30 >=1 and min_inst_def30 =1 then 1 else 0 end deffpd30,
case when obs_min_inst_def30 >=2 and min_inst_def30 in (1,2) then 1 else 0 end deffspd30,
case when obs_min_inst_def30 >=3 and min_inst_def30 in (1,2,3) then 1 else 0 end deffstpd30,
case when obs_min_inst_def0 >= 1 then 1 else 0 end flg_mature_fpd0,
case when obs_min_inst_def10 >=1 then 1 else 0 end flg_mature_fpd10,
case when obs_min_inst_def30 >=1 then 1 else 0 end flg_mature_fpd30,
case when obs_min_inst_def30 >=2 then 1 else 0 end flg_mature_fspd_30,
case when obs_min_inst_def30 >=3 then 1 else 0 end flg_mature_fstpd_30
from prj-prod-dataplatform.risk_credit_mis.loan_deliquency_data),
base as
(select distinct r.customerId,
  r.digitalLoanAccountId,
  loanmaster.loanAccountNumber,
  r.modelDisplayName,
  r.Alpha_cic_sil_score,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month, 
  'Train' Data_selection,  
    del.deffpd10,
  del.flg_mature_fpd10,
  loanmaster.new_loan_type,
  modelVersionId,
    trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile' 
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall' 
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
    from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  inner join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
  left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
  where loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and r.Alpha_cic_sil_score is not null
  and del.flg_mature_fpd10 = 1
  )
  select * from base;
  """
dfd = client.query(sq).to_dataframe()
dfd = dfd.drop_duplicates(keep="first")
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()

# %%
df2 = dfd.copy()

# %% [markdown]
# ### Concat

# %%
# df_concat = pd.concat([df1, df2], ignore_index=True)
# print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")

# df_concat = (df_concat
#              .sort_values(by=['digitalLoanAccountId', 'Data_selection'],
#                           key=lambda s: s.map({'Train': 0, 'Test': 1}))
#              .drop_duplicates(subset=['digitalLoanAccountId'], keep='first')
#              .reset_index(drop=True))
# print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")
# df_concat.head()


# 1) Get all IDs present in Train
train_ids = set(df2["digitalLoanAccountId"])

# 2) Keep only Test rows whose ID is NOT in Train
df1_no_dupes = df1[~df1["digitalLoanAccountId"].isin(train_ids)]

# 3) Concatenate
df_concat = pd.concat([df1_no_dupes, df2], ignore_index=True)

print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")
df_concat.head()


# %%
df_concat.groupby(
    ["Data_selection", "new_loan_type", "modelVersionId", "loan_product_type"]
)["digitalLoanAccountId"].nunique()

# %%
df_concat["Alpha_cic_sil_score"] = pd.to_numeric(
    df_concat["Alpha_cic_sil_score"], errors="coerce"
)

# %%
fact_table, dimension_table = calculate_periodic_gini_prod_ver_trench_dimfact(
    df_concat,
    "Alpha_cic_sil_score",
    "deffpd10",
    "FPD10",
    model_version_column="modelVersionId",
    trench_column="trenchCategory",
    loan_type_column="new_loan_type",
    loan_product_type_column="loan_product_type",
    account_id_column="digitalLoanAccountId",
)

# %%
fact_table, dimension_table = update_tables(
    fact_table, dimension_table, model_name="cic_model_sil", product="SIL"
)

# %%
# Upload to BigQuery
table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_table"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(fact_table, table_id, job_config=job_config)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_table"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(dimension_table, table_id, job_config=job_config)
job.result()  # Wait for the job to complete

# %%


# %% [markdown]
# ### FPD30

# %% [markdown]
# ### Test

# %%
sq = """ 
with modelname as 
  (SELECT
    customerId,digitalLoanAccountId,prediction Alpha_cic_sil_score,start_time,end_time,modelDisplayName,modelVersionId, 
    case when trenchCategory is null then 'ALL'
         when trenchCategory = '' then 'ALL'
         else trenchCategory end as trenchCategory,
    REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature
  FROM prj-prod-dataplatform.audit_balance.ml_model_run_details
  WHERE modelDisplayName in ('Alpha - CIC-SIL-Model', 'cic_model_sil', 'Sil-Alpha-CIC-SIL-Model')
  ),
  deliquency as
(select loanAccountNumber,
case when obs_min_inst_def0 >= 1 and min_inst_def0 = 1 then 1 else 0 end deffpd0,
case when obs_min_inst_def10 >=1 and min_inst_def10 =1 then 1 else 0 end deffpd10,
case when obs_min_inst_def30 >=1 and min_inst_def30 =1 then 1 else 0 end deffpd30,
case when obs_min_inst_def30 >=2 and min_inst_def30 in (1,2) then 1 else 0 end deffspd30,
case when obs_min_inst_def30 >=3 and min_inst_def30 in (1,2,3) then 1 else 0 end deffstpd30,
case when obs_min_inst_def0 >= 1 then 1 else 0 end flg_mature_fpd0,
case when obs_min_inst_def10 >=1 then 1 else 0 end flg_mature_fpd10,
case when obs_min_inst_def30 >=1 then 1 else 0 end flg_mature_fpd30,
case when obs_min_inst_def30 >=2 then 1 else 0 end flg_mature_fspd_30,
case when obs_min_inst_def30 >=3 then 1 else 0 end flg_mature_fstpd_30
from prj-prod-dataplatform.risk_credit_mis.loan_deliquency_data),
base as 
  (select distinct r.customerId,
  r.digitalLoanAccountId,
  loanmaster.loanAccountNumber,
  r.modelDisplayName,
  r.Alpha_cic_sil_score,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month, 
  'Test' Data_selection,  
  del.deffpd30,
  del.flg_mature_fpd30,
  loanmaster.new_loan_type,
  modelVersionId, trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile' 
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall' 
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
  from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  inner join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
   left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
  where loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and r.Alpha_cic_sil_score is not null
  and del.flg_mature_fpd30 = 1
  )
  select *  from base
  ;
"""
dfd = client.query(sq).to_dataframe()
dfd = dfd.drop_duplicates(keep="first")
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()


# %%
df1 = dfd.copy()

# %% [markdown]
# ### Train

# %%
sq = """ 
  with modelname as 
  (SELECT
    customerId,digitalLoanAccountId,prediction Alpha_cic_sil_score,start_time,end_time,modelDisplayName,modelVersionId, 
    case when trenchCategory is null then 'ALL' 
        when trenchCategory = ''then 'ALL' 
        else trenchCategory end as trenchCategory,
    REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature
  FROM prj-prod-dataplatform.dap_ds_poweruser_playground.ml_training_model_run_details
  WHERE modelDisplayName in ('Alpha - CIC-SIL-Model', 'cic_model_sil', 'Sil-Alpha-CIC-SIL-Model')
  ),
  deliquency as
(select loanAccountNumber,
case when obs_min_inst_def0 >= 1 and min_inst_def0 = 1 then 1 else 0 end deffpd0,
case when obs_min_inst_def10 >=1 and min_inst_def10 =1 then 1 else 0 end deffpd10,
case when obs_min_inst_def30 >=1 and min_inst_def30 =1 then 1 else 0 end deffpd30,
case when obs_min_inst_def30 >=2 and min_inst_def30 in (1,2) then 1 else 0 end deffspd30,
case when obs_min_inst_def30 >=3 and min_inst_def30 in (1,2,3) then 1 else 0 end deffstpd30,
case when obs_min_inst_def0 >= 1 then 1 else 0 end flg_mature_fpd0,
case when obs_min_inst_def10 >=1 then 1 else 0 end flg_mature_fpd10,
case when obs_min_inst_def30 >=1 then 1 else 0 end flg_mature_fpd30,
case when obs_min_inst_def30 >=2 then 1 else 0 end flg_mature_fspd_30,
case when obs_min_inst_def30 >=3 then 1 else 0 end flg_mature_fstpd_30
from prj-prod-dataplatform.risk_credit_mis.loan_deliquency_data),
base as
(select distinct r.customerId,
  r.digitalLoanAccountId,
  loanmaster.loanAccountNumber,
  r.modelDisplayName,
  r.Alpha_cic_sil_score,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month, 
  'Train' Data_selection,  
    del.deffpd30,
  del.flg_mature_fpd30,
  loanmaster.new_loan_type,
  modelVersionId,
    trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile' 
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall' 
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
    from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  inner join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
  left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
  where loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and r.Alpha_cic_sil_score is not null
  and del.flg_mature_fpd30 = 1
  )
  select * from base;
  """
dfd = client.query(sq).to_dataframe()
dfd = dfd.drop_duplicates(keep="first")
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()

# %%
df2 = dfd.copy()

# %% [markdown]
# ### Concat

# %%
# df_concat = pd.concat([df1, df2], ignore_index=True)
# print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")

# df_concat = (df_concat
#              .sort_values(by=['digitalLoanAccountId', 'Data_selection'],
#                           key=lambda s: s.map({'Train': 0, 'Test': 1}))
#              .drop_duplicates(subset=['digitalLoanAccountId'], keep='first')
#              .reset_index(drop=True))
# print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")
# df_concat.head()


# 1) Get all IDs present in Train
train_ids = set(df2["digitalLoanAccountId"])

# 2) Keep only Test rows whose ID is NOT in Train
df1_no_dupes = df1[~df1["digitalLoanAccountId"].isin(train_ids)]

# 3) Concatenate
df_concat = pd.concat([df1_no_dupes, df2], ignore_index=True)

print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")
df_concat.head()


# %%
df_concat.groupby(
    ["Data_selection", "new_loan_type", "modelVersionId", "loan_product_type"]
)["digitalLoanAccountId"].nunique()

# %%
df_concat["Alpha_cic_sil_score"] = pd.to_numeric(
    df_concat["Alpha_cic_sil_score"], errors="coerce"
)

# %%
fact_table, dimension_table = calculate_periodic_gini_prod_ver_trench_dimfact(
    df_concat,
    "Alpha_cic_sil_score",
    "deffpd30",
    "FPD30",
    model_version_column="modelVersionId",
    trench_column="trenchCategory",
    loan_type_column="new_loan_type",
    loan_product_type_column="loan_product_type",
    account_id_column="digitalLoanAccountId",
)

# %%
fact_table, dimension_table = update_tables(
    fact_table, dimension_table, model_name="cic_model_sil", product="SIL"
)

# %%
# Upload to BigQuery
table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_table"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(fact_table, table_id, job_config=job_config)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_table"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(dimension_table, table_id, job_config=job_config)
job.result()  # Wait for the job to complete

# %%


# %% [markdown]
# ### FSPD30

# %% [markdown]
# ### Test

# %%
sq = """ 
with modelname as 
  (SELECT
    customerId,digitalLoanAccountId,prediction Alpha_cic_sil_score,start_time,end_time,modelDisplayName,modelVersionId, 
    case when trenchCategory is null then 'ALL'
         when trenchCategory = '' then 'ALL'
         else trenchCategory end as trenchCategory,
    REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature
  FROM prj-prod-dataplatform.audit_balance.ml_model_run_details
  WHERE modelDisplayName in ('Alpha - CIC-SIL-Model', 'cic_model_sil', 'Sil-Alpha-CIC-SIL-Model')
  ),
  deliquency as
(select loanAccountNumber,
case when obs_min_inst_def0 >= 1 and min_inst_def0 = 1 then 1 else 0 end deffpd0,
case when obs_min_inst_def10 >=1 and min_inst_def10 =1 then 1 else 0 end deffpd10,
case when obs_min_inst_def30 >=1 and min_inst_def30 =1 then 1 else 0 end deffpd30,
case when obs_min_inst_def30 >=2 and min_inst_def30 in (1,2) then 1 else 0 end deffspd30,
case when obs_min_inst_def30 >=3 and min_inst_def30 in (1,2,3) then 1 else 0 end deffstpd30,
case when obs_min_inst_def0 >= 1 then 1 else 0 end flg_mature_fpd0,
case when obs_min_inst_def10 >=1 then 1 else 0 end flg_mature_fpd10,
case when obs_min_inst_def30 >=1 then 1 else 0 end flg_mature_fpd30,
case when obs_min_inst_def30 >=2 then 1 else 0 end flg_mature_fspd_30,
case when obs_min_inst_def30 >=3 then 1 else 0 end flg_mature_fstpd_30
from prj-prod-dataplatform.risk_credit_mis.loan_deliquency_data),
base as 
  (select distinct r.customerId,
  r.digitalLoanAccountId,
  loanmaster.loanAccountNumber,
  r.modelDisplayName,
  r.Alpha_cic_sil_score,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month, 
  'Test' Data_selection,  
  del.deffspd30,
  del.flg_mature_fspd_30,
  loanmaster.new_loan_type,
  modelVersionId, trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile' 
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall' 
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
  from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  inner join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
   left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
  where loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and r.Alpha_cic_sil_score is not null
  and del.flg_mature_fspd_30 = 1
  )
  select *  from base
  ;
"""
dfd = client.query(sq).to_dataframe()
dfd = dfd.drop_duplicates(keep="first")
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()


# %%
df1 = dfd.copy()

# %% [markdown]
# ### Train

# %%
sq = """ 
  with modelname as 
  (SELECT
    customerId,digitalLoanAccountId,prediction Alpha_cic_sil_score,start_time,end_time,modelDisplayName,modelVersionId, 
    case when trenchCategory is null then 'ALL' 
        when trenchCategory = ''then 'ALL' 
        else trenchCategory end as trenchCategory,
    REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature
  FROM prj-prod-dataplatform.dap_ds_poweruser_playground.ml_training_model_run_details
  WHERE modelDisplayName in ('Alpha - CIC-SIL-Model', 'cic_model_sil', 'Sil-Alpha-CIC-SIL-Model')
  ),
  deliquency as
(select loanAccountNumber,
case when obs_min_inst_def0 >= 1 and min_inst_def0 = 1 then 1 else 0 end deffpd0,
case when obs_min_inst_def10 >=1 and min_inst_def10 =1 then 1 else 0 end deffpd10,
case when obs_min_inst_def30 >=1 and min_inst_def30 =1 then 1 else 0 end deffpd30,
case when obs_min_inst_def30 >=2 and min_inst_def30 in (1,2) then 1 else 0 end deffspd30,
case when obs_min_inst_def30 >=3 and min_inst_def30 in (1,2,3) then 1 else 0 end deffstpd30,
case when obs_min_inst_def0 >= 1 then 1 else 0 end flg_mature_fpd0,
case when obs_min_inst_def10 >=1 then 1 else 0 end flg_mature_fpd10,
case when obs_min_inst_def30 >=1 then 1 else 0 end flg_mature_fpd30,
case when obs_min_inst_def30 >=2 then 1 else 0 end flg_mature_fspd_30,
case when obs_min_inst_def30 >=3 then 1 else 0 end flg_mature_fstpd_30
from prj-prod-dataplatform.risk_credit_mis.loan_deliquency_data),
base as
(select distinct r.customerId,
  r.digitalLoanAccountId,
  loanmaster.loanAccountNumber,
  r.modelDisplayName,
  r.Alpha_cic_sil_score,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month, 
  'Train' Data_selection,  
    del.deffspd30,
  del.flg_mature_fspd_30,
  loanmaster.new_loan_type,
  modelVersionId,
    trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile' 
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall' 
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
    from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  inner join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
  left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
  where loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and r.Alpha_cic_sil_score is not null
  and del.flg_mature_fspd_30 = 1
  )
  select * from base;
  """
dfd = client.query(sq).to_dataframe()
dfd = dfd.drop_duplicates(keep="first")
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()

# %%
df2 = dfd.copy()

# %% [markdown]
# ### Concat

# %%
# df_concat = pd.concat([df1, df2], ignore_index=True)
# print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")

# df_concat = (df_concat
#              .sort_values(by=['digitalLoanAccountId', 'Data_selection'],
#                           key=lambda s: s.map({'Train': 0, 'Test': 1}))
#              .drop_duplicates(subset=['digitalLoanAccountId'], keep='first')
#              .reset_index(drop=True))
# print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")
# df_concat.head()


# 1) Get all IDs present in Train
train_ids = set(df2["digitalLoanAccountId"])

# 2) Keep only Test rows whose ID is NOT in Train
df1_no_dupes = df1[~df1["digitalLoanAccountId"].isin(train_ids)]

# 3) Concatenate
df_concat = pd.concat([df1_no_dupes, df2], ignore_index=True)

print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")
df_concat.head()


# %%
df_concat.groupby(
    ["Data_selection", "new_loan_type", "modelVersionId", "loan_product_type"]
)["digitalLoanAccountId"].nunique()

# %%
df_concat["Alpha_cic_sil_score"] = pd.to_numeric(
    df_concat["Alpha_cic_sil_score"], errors="coerce"
)

# %%
fact_table, dimension_table = calculate_periodic_gini_prod_ver_trench_dimfact(
    df_concat,
    "Alpha_cic_sil_score",
    "deffspd30",
    "FSPD30",
    model_version_column="modelVersionId",
    trench_column="trenchCategory",
    loan_type_column="new_loan_type",
    loan_product_type_column="loan_product_type",
    account_id_column="digitalLoanAccountId",
)

# %%
fact_table, dimension_table = update_tables(
    fact_table, dimension_table, model_name="cic_model_sil", product="SIL"
)

# %%
# Upload to BigQuery
table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_table"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(fact_table, table_id, job_config=job_config)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_table"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(dimension_table, table_id, job_config=job_config)
job.result()  # Wait for the job to complete

# %%


# %% [markdown]
# ### FSTPD30

# %% [markdown]
# ### Test

# %%
sq = """ 
with modelname as 
  (SELECT
    customerId,digitalLoanAccountId,prediction Alpha_cic_sil_score,start_time,end_time,modelDisplayName,modelVersionId, 
    case when trenchCategory is null then 'ALL'
         when trenchCategory = '' then 'ALL'
         else trenchCategory end as trenchCategory,
    REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature
  FROM prj-prod-dataplatform.audit_balance.ml_model_run_details
  WHERE modelDisplayName in ('Alpha - CIC-SIL-Model', 'cic_model_sil', 'Sil-Alpha-CIC-SIL-Model')
  ),
  deliquency as
(select loanAccountNumber,
case when obs_min_inst_def0 >= 1 and min_inst_def0 = 1 then 1 else 0 end deffpd0,
case when obs_min_inst_def10 >=1 and min_inst_def10 =1 then 1 else 0 end deffpd10,
case when obs_min_inst_def30 >=1 and min_inst_def30 =1 then 1 else 0 end deffpd30,
case when obs_min_inst_def30 >=2 and min_inst_def30 in (1,2) then 1 else 0 end deffspd30,
case when obs_min_inst_def30 >=3 and min_inst_def30 in (1,2,3) then 1 else 0 end deffstpd30,
case when obs_min_inst_def0 >= 1 then 1 else 0 end flg_mature_fpd0,
case when obs_min_inst_def10 >=1 then 1 else 0 end flg_mature_fpd10,
case when obs_min_inst_def30 >=1 then 1 else 0 end flg_mature_fpd30,
case when obs_min_inst_def30 >=2 then 1 else 0 end flg_mature_fspd_30,
case when obs_min_inst_def30 >=3 then 1 else 0 end flg_mature_fstpd_30
from prj-prod-dataplatform.risk_credit_mis.loan_deliquency_data),
base as 
  (select distinct r.customerId,
  r.digitalLoanAccountId,
  loanmaster.loanAccountNumber,
  r.modelDisplayName,
  r.Alpha_cic_sil_score,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month, 
  'Test' Data_selection,  
  del.deffstpd30,
  del.flg_mature_fstpd_30,
  loanmaster.new_loan_type,
  modelVersionId, trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile' 
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall' 
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
  from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  inner join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
   left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
  where loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and r.Alpha_cic_sil_score is not null
  and del.flg_mature_fstpd_30 = 1
  )
  select *  from base
  ;
"""
dfd = client.query(sq).to_dataframe()
dfd = dfd.drop_duplicates(keep="first")
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()


# %%
df1 = dfd.copy()

# %% [markdown]
# ### Train

# %%
sq = """ 
  with modelname as 
  (SELECT
    customerId,digitalLoanAccountId,prediction Alpha_cic_sil_score,start_time,end_time,modelDisplayName,modelVersionId, 
    case when trenchCategory is null then 'ALL' 
        when trenchCategory = ''then 'ALL' 
        else trenchCategory end as trenchCategory,
    REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature
  FROM prj-prod-dataplatform.dap_ds_poweruser_playground.ml_training_model_run_details
  WHERE modelDisplayName in ('Alpha - CIC-SIL-Model', 'cic_model_sil', 'Sil-Alpha-CIC-SIL-Model')
  ),
  deliquency as
(select loanAccountNumber,
case when obs_min_inst_def0 >= 1 and min_inst_def0 = 1 then 1 else 0 end deffpd0,
case when obs_min_inst_def10 >=1 and min_inst_def10 =1 then 1 else 0 end deffpd10,
case when obs_min_inst_def30 >=1 and min_inst_def30 =1 then 1 else 0 end deffpd30,
case when obs_min_inst_def30 >=2 and min_inst_def30 in (1,2) then 1 else 0 end deffspd30,
case when obs_min_inst_def30 >=3 and min_inst_def30 in (1,2,3) then 1 else 0 end deffstpd30,
case when obs_min_inst_def0 >= 1 then 1 else 0 end flg_mature_fpd0,
case when obs_min_inst_def10 >=1 then 1 else 0 end flg_mature_fpd10,
case when obs_min_inst_def30 >=1 then 1 else 0 end flg_mature_fpd30,
case when obs_min_inst_def30 >=2 then 1 else 0 end flg_mature_fspd_30,
case when obs_min_inst_def30 >=3 then 1 else 0 end flg_mature_fstpd_30
from prj-prod-dataplatform.risk_credit_mis.loan_deliquency_data),
base as
(select distinct r.customerId,
  r.digitalLoanAccountId,
  loanmaster.loanAccountNumber,
  r.modelDisplayName,
  r.Alpha_cic_sil_score,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month, 
  'Train' Data_selection,  
    del.deffstpd30,
  del.flg_mature_fstpd_30,
  loanmaster.new_loan_type,
  modelVersionId,
    trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile' 
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall' 
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
    from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  inner join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
  left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
  where loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and r.Alpha_cic_sil_score is not null
  and del.flg_mature_fstpd_30 = 1
  )
  select * from base;
  """
dfd = client.query(sq).to_dataframe()
dfd = dfd.drop_duplicates(keep="first")
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()

# %%
df2 = dfd.copy()

# %% [markdown]
# ### Concat

# %%
# df_concat = pd.concat([df1, df2], ignore_index=True)
# print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")

# df_concat = (df_concat
#              .sort_values(by=['digitalLoanAccountId', 'Data_selection'],
#                           key=lambda s: s.map({'Train': 0, 'Test': 1}))
#              .drop_duplicates(subset=['digitalLoanAccountId'], keep='first')
#              .reset_index(drop=True))
# print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")
# df_concat.head()


# 1) Get all IDs present in Train
train_ids = set(df2["digitalLoanAccountId"])

# 2) Keep only Test rows whose ID is NOT in Train
df1_no_dupes = df1[~df1["digitalLoanAccountId"].isin(train_ids)]

# 3) Concatenate
df_concat = pd.concat([df1_no_dupes, df2], ignore_index=True)

print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")
df_concat.head()


# %%
df_concat["Alpha_cic_sil_score"] = pd.to_numeric(
    df_concat["Alpha_cic_sil_score"], errors="coerce"
)

# %%
df_concat.groupby(
    ["Data_selection", "new_loan_type", "modelVersionId", "loan_product_type"]
)["digitalLoanAccountId"].nunique()

# %%
fact_table, dimension_table = calculate_periodic_gini_prod_ver_trench_dimfact(
    df_concat,
    "Alpha_cic_sil_score",
    "deffstpd30",
    "FSTPD30",
    model_version_column="modelVersionId",
    trench_column="trenchCategory",
    loan_type_column="new_loan_type",
    loan_product_type_column="loan_product_type",
    account_id_column="digitalLoanAccountId",
)

# %%
fact_table, dimension_table = update_tables(
    fact_table, dimension_table, model_name="cic_model_sil", product="SIL"
)
print(f"The shape of the fact table is:\t {fact_table.shape}")
print(f"The shape of the dimension table is:\t {dimension_table.shape}")

# %%
# Upload to BigQuery
table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_table"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(fact_table, table_id, job_config=job_config)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_table"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(dimension_table, table_id, job_config=job_config)
job.result()  # Wait for the job to complete

# %% [markdown]
# ## alpha_stack_model_sil

# %% [markdown]
# ### FPD0

# %% [markdown]
# ### Test

# %%
sq = """ 
with modelname as 
  (SELECT
    customerId,digitalLoanAccountId,prediction Sil_Alpha_Stack_score,start_time,end_time,modelDisplayName,modelVersionId, 
    case when trenchCategory is null then 'ALL'
         when trenchCategory = '' then 'ALL'
         else trenchCategory end as trenchCategory,
    REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature
  FROM prj-prod-dataplatform.audit_balance.ml_model_run_details
  WHERE modelDisplayName in  ('Alpha - StackingModel', 'alpha_stack_model_sil')
  ),
  deliquency as
(select loanAccountNumber,
case when obs_min_inst_def0 >= 1 and min_inst_def0 = 1 then 1 else 0 end deffpd0,
case when obs_min_inst_def10 >=1 and min_inst_def10 =1 then 1 else 0 end deffpd10,
case when obs_min_inst_def30 >=1 and min_inst_def30 =1 then 1 else 0 end deffpd30,
case when obs_min_inst_def30 >=2 and min_inst_def30 in (1,2) then 1 else 0 end deffspd30,
case when obs_min_inst_def30 >=3 and min_inst_def30 in (1,2,3) then 1 else 0 end deffstpd30,
case when obs_min_inst_def0 >= 1 then 1 else 0 end flg_mature_fpd0,
case when obs_min_inst_def10 >=1 then 1 else 0 end flg_mature_fpd10,
case when obs_min_inst_def30 >=1 then 1 else 0 end flg_mature_fpd30,
case when obs_min_inst_def30 >=2 then 1 else 0 end flg_mature_fspd_30,
case when obs_min_inst_def30 >=3 then 1 else 0 end flg_mature_fstpd_30
from prj-prod-dataplatform.risk_credit_mis.loan_deliquency_data),
base as 
  (select distinct r.customerId,
  r.digitalLoanAccountId,
  loanmaster.loanAccountNumber,
  r.modelDisplayName,
  r.Sil_Alpha_Stack_score,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month, 
  'Test' Data_selection,  
  deffpd0,
  flg_mature_fpd0,
  loanmaster.new_loan_type,
  modelVersionId, trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile' 
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall' 
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
  from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  inner join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
   left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
  where loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and r.Sil_Alpha_Stack_score is not null
  and flg_mature_fpd0 = 1
  )
  select *  from base
  ;
"""
dfd = client.query(sq).to_dataframe()
dfd = dfd.drop_duplicates(keep="first")
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()


# %%
df1 = dfd.copy()

# %% [markdown]
# ### Train

# %%
sq = """ 
  with modelname as 
  (SELECT
    customerId,digitalLoanAccountId,prediction Sil_Alpha_Stack_score,start_time,end_time,modelDisplayName,modelVersionId, 
    case when trenchCategory is null then 'ALL' 
        when trenchCategory = ''then 'ALL' 
        else trenchCategory end as trenchCategory,
    REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature
  FROM prj-prod-dataplatform.dap_ds_poweruser_playground.ml_training_model_run_details
  WHERE modelDisplayName in  ('Alpha - StackingModel', 'alpha_stack_model_sil')
  ),
  deliquency as
(select loanAccountNumber,
case when obs_min_inst_def0 >= 1 and min_inst_def0 = 1 then 1 else 0 end deffpd0,
case when obs_min_inst_def10 >=1 and min_inst_def10 =1 then 1 else 0 end deffpd10,
case when obs_min_inst_def30 >=1 and min_inst_def30 =1 then 1 else 0 end deffpd30,
case when obs_min_inst_def30 >=2 and min_inst_def30 in (1,2) then 1 else 0 end deffspd30,
case when obs_min_inst_def30 >=3 and min_inst_def30 in (1,2,3) then 1 else 0 end deffstpd30,
case when obs_min_inst_def0 >= 1 then 1 else 0 end flg_mature_fpd0,
case when obs_min_inst_def10 >=1 then 1 else 0 end flg_mature_fpd10,
case when obs_min_inst_def30 >=1 then 1 else 0 end flg_mature_fpd30,
case when obs_min_inst_def30 >=2 then 1 else 0 end flg_mature_fspd_30,
case when obs_min_inst_def30 >=3 then 1 else 0 end flg_mature_fstpd_30
from prj-prod-dataplatform.risk_credit_mis.loan_deliquency_data),
base as
(select distinct r.customerId,
  r.digitalLoanAccountId,
  loanmaster.loanAccountNumber,
  r.modelDisplayName,
  r.Sil_Alpha_Stack_score,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month, 
  'Train' Data_selection,  
    deffpd0,
  flg_mature_fpd0,
  loanmaster.new_loan_type,
  modelVersionId,
    trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile' 
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall' 
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
    from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  inner join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
  left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
  where loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and r.Sil_Alpha_Stack_score is not null
  and flg_mature_fpd0 = 1
  )
  select * from base;
  """
dfd = client.query(sq).to_dataframe()
dfd = dfd.drop_duplicates(keep="first")
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()

# %%
df2 = dfd.copy()

# %% [markdown]
# ### Concat

# %%
# df_concat = pd.concat([df1, df2], ignore_index=True)
# print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")

# df_concat = (df_concat
#              .sort_values(by=['digitalLoanAccountId', 'Data_selection'],
#                           key=lambda s: s.map({'Train': 0, 'Test': 1}))
#              .drop_duplicates(subset=['digitalLoanAccountId'], keep='first')
#              .reset_index(drop=True))
# print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")
# df_concat.head()

# 1) Get all IDs present in Train
train_ids = set(df2["digitalLoanAccountId"])

# 2) Keep only Test rows whose ID is NOT in Train
df1_no_dupes = df1[~df1["digitalLoanAccountId"].isin(train_ids)]

# 3) Concatenate
df_concat = pd.concat([df1_no_dupes, df2], ignore_index=True)

print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")
df_concat.head()


# %%
df_concat.info()

# %%
df_concat.groupby(
    ["Data_selection", "new_loan_type", "modelVersionId", "loan_product_type"]
)["digitalLoanAccountId"].nunique()

# %%
df_concat["new_loan_type"].unique()

# %%
df_concat["Sil_Alpha_Stack_score"] = pd.to_numeric(
    df_concat["Sil_Alpha_Stack_score"], errors="coerce"
)

# %%
fact_table, dimension_table = calculate_periodic_gini_prod_ver_trench_dimfact(
    df_concat,
    "Sil_Alpha_Stack_score",
    "deffpd0",
    "FPD0",
    model_version_column="modelVersionId",
    trench_column="trenchCategory",
    loan_type_column="new_loan_type",
    loan_product_type_column="loan_product_type",
    account_id_column="digitalLoanAccountId",
)

# %%
fact_table, dimension_table = update_tables(
    fact_table, dimension_table, model_name="alpha_stack_model_sil", product="SIL"
)
print(f"The shape of the fact table is:\t {fact_table.shape}")
print(f"The shape of the dimension table is:\t {dimension_table.shape}")

# %%
# Upload to BigQuery
table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_table"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(fact_table, table_id, job_config=job_config)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_table"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(dimension_table, table_id, job_config=job_config)
job.result()  # Wait for the job to complete

# %% [markdown]
# ### FPD10

# %% [markdown]
# ### Test

# %%
sq = """ 
with modelname as 
  (SELECT
    customerId,digitalLoanAccountId,prediction Sil_Alpha_Stack_score,start_time,end_time,modelDisplayName,modelVersionId, 
    case when trenchCategory is null then 'ALL'
         when trenchCategory = '' then 'ALL'
         else trenchCategory end as trenchCategory,
    REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature
  FROM prj-prod-dataplatform.audit_balance.ml_model_run_details
  WHERE modelDisplayName in  ('Alpha - StackingModel', 'alpha_stack_model_sil')
  ),
  deliquency as
(select loanAccountNumber,
case when obs_min_inst_def0 >= 1 and min_inst_def0 = 1 then 1 else 0 end deffpd0,
case when obs_min_inst_def10 >=1 and min_inst_def10 =1 then 1 else 0 end deffpd10,
case when obs_min_inst_def30 >=1 and min_inst_def30 =1 then 1 else 0 end deffpd30,
case when obs_min_inst_def30 >=2 and min_inst_def30 in (1,2) then 1 else 0 end deffspd30,
case when obs_min_inst_def30 >=3 and min_inst_def30 in (1,2,3) then 1 else 0 end deffstpd30,
case when obs_min_inst_def0 >= 1 then 1 else 0 end flg_mature_fpd0,
case when obs_min_inst_def10 >=1 then 1 else 0 end flg_mature_fpd10,
case when obs_min_inst_def30 >=1 then 1 else 0 end flg_mature_fpd30,
case when obs_min_inst_def30 >=2 then 1 else 0 end flg_mature_fspd_30,
case when obs_min_inst_def30 >=3 then 1 else 0 end flg_mature_fstpd_30
from prj-prod-dataplatform.risk_credit_mis.loan_deliquency_data),
base as 
  (select distinct r.customerId,
  r.digitalLoanAccountId,
  loanmaster.loanAccountNumber,
  r.modelDisplayName,
  r.Sil_Alpha_Stack_score,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month, 
  'Test' Data_selection,  
  del.deffpd10,
  del.flg_mature_fpd10,
  loanmaster.new_loan_type,
  modelVersionId, trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile' 
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall' 
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
  from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  inner join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
   left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
  where loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and r.Sil_Alpha_Stack_score is not null
  and del.flg_mature_fpd10 = 1
  )
  select *  from base
  ;
"""
dfd = client.query(sq).to_dataframe()
dfd = dfd.drop_duplicates(keep="first")
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()


# %%
df1 = dfd.copy()

# %% [markdown]
# ### Train

# %%
sq = """ 
  with modelname as 
  (SELECT
    customerId,digitalLoanAccountId,prediction Sil_Alpha_Stack_score,start_time,end_time,modelDisplayName,modelVersionId, 
    case when trenchCategory is null then 'ALL' 
        when trenchCategory = ''then 'ALL' 
        else trenchCategory end as trenchCategory,
    REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature
  FROM prj-prod-dataplatform.dap_ds_poweruser_playground.ml_training_model_run_details
  WHERE modelDisplayName in  ('Alpha - StackingModel', 'alpha_stack_model_sil')
  ),
  deliquency as
(select loanAccountNumber,
case when obs_min_inst_def0 >= 1 and min_inst_def0 = 1 then 1 else 0 end deffpd0,
case when obs_min_inst_def10 >=1 and min_inst_def10 =1 then 1 else 0 end deffpd10,
case when obs_min_inst_def30 >=1 and min_inst_def30 =1 then 1 else 0 end deffpd30,
case when obs_min_inst_def30 >=2 and min_inst_def30 in (1,2) then 1 else 0 end deffspd30,
case when obs_min_inst_def30 >=3 and min_inst_def30 in (1,2,3) then 1 else 0 end deffstpd30,
case when obs_min_inst_def0 >= 1 then 1 else 0 end flg_mature_fpd0,
case when obs_min_inst_def10 >=1 then 1 else 0 end flg_mature_fpd10,
case when obs_min_inst_def30 >=1 then 1 else 0 end flg_mature_fpd30,
case when obs_min_inst_def30 >=2 then 1 else 0 end flg_mature_fspd_30,
case when obs_min_inst_def30 >=3 then 1 else 0 end flg_mature_fstpd_30
from prj-prod-dataplatform.risk_credit_mis.loan_deliquency_data),
base as
(select distinct r.customerId,
  r.digitalLoanAccountId,
  loanmaster.loanAccountNumber,
  r.modelDisplayName,
  r.Sil_Alpha_Stack_score,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month, 
  'Train' Data_selection,  
    del.deffpd10,
  del.flg_mature_fpd10,
  loanmaster.new_loan_type,
  modelVersionId,
    trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile' 
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall' 
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
    from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  inner join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
  left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
  where loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and r.Sil_Alpha_Stack_score is not null
  and del.flg_mature_fpd10 = 1
  )
  select * from base;
  """
dfd = client.query(sq).to_dataframe()
dfd = dfd.drop_duplicates(keep="first")
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()

# %%
df2 = dfd.copy()

# %% [markdown]
# ### Concat

# %%
# df_concat = pd.concat([df1, df2], ignore_index=True)
# print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")

# df_concat = (df_concat
#              .sort_values(by=['digitalLoanAccountId', 'Data_selection'],
#                           key=lambda s: s.map({'Train': 0, 'Test': 1}))
#              .drop_duplicates(subset=['digitalLoanAccountId'], keep='first')
#              .reset_index(drop=True))
# print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")
# df_concat.head()


# 1) Get all IDs present in Train
train_ids = set(df2["digitalLoanAccountId"])

# 2) Keep only Test rows whose ID is NOT in Train
df1_no_dupes = df1[~df1["digitalLoanAccountId"].isin(train_ids)]

# 3) Concatenate
df_concat = pd.concat([df1_no_dupes, df2], ignore_index=True)

print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")
df_concat.head()


# %%
df_concat.groupby(
    ["Data_selection", "new_loan_type", "modelVersionId", "loan_product_type"]
)["digitalLoanAccountId"].nunique()

# %%
df_concat["Sil_Alpha_Stack_score"] = pd.to_numeric(
    df_concat["Sil_Alpha_Stack_score"], errors="coerce"
)

# %%
fact_table, dimension_table = calculate_periodic_gini_prod_ver_trench_dimfact(
    df_concat,
    "Sil_Alpha_Stack_score",
    "deffpd10",
    "FPD10",
    model_version_column="modelVersionId",
    trench_column="trenchCategory",
    loan_type_column="new_loan_type",
    loan_product_type_column="loan_product_type",
    account_id_column="digitalLoanAccountId",
)

# %%
fact_table, dimension_table = update_tables(
    fact_table, dimension_table, model_name="alpha_stack_model_sil", product="SIL"
)

# %%
# Upload to BigQuery
table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_table"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(fact_table, table_id, job_config=job_config)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_table"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(dimension_table, table_id, job_config=job_config)
job.result()  # Wait for the job to complete

# %%


# %% [markdown]
# ### FPD30

# %% [markdown]
# ### Test

# %%
sq = """ 
with modelname as 
  (SELECT
    customerId,digitalLoanAccountId,prediction Sil_Alpha_Stack_score,start_time,end_time,modelDisplayName,modelVersionId, 
    case when trenchCategory is null then 'ALL'
         when trenchCategory = '' then 'ALL'
         else trenchCategory end as trenchCategory,
    REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature
  FROM prj-prod-dataplatform.audit_balance.ml_model_run_details
  WHERE modelDisplayName in  ('Alpha - StackingModel', 'alpha_stack_model_sil')
  ),
  deliquency as
(select loanAccountNumber,
case when obs_min_inst_def0 >= 1 and min_inst_def0 = 1 then 1 else 0 end deffpd0,
case when obs_min_inst_def10 >=1 and min_inst_def10 =1 then 1 else 0 end deffpd10,
case when obs_min_inst_def30 >=1 and min_inst_def30 =1 then 1 else 0 end deffpd30,
case when obs_min_inst_def30 >=2 and min_inst_def30 in (1,2) then 1 else 0 end deffspd30,
case when obs_min_inst_def30 >=3 and min_inst_def30 in (1,2,3) then 1 else 0 end deffstpd30,
case when obs_min_inst_def0 >= 1 then 1 else 0 end flg_mature_fpd0,
case when obs_min_inst_def10 >=1 then 1 else 0 end flg_mature_fpd10,
case when obs_min_inst_def30 >=1 then 1 else 0 end flg_mature_fpd30,
case when obs_min_inst_def30 >=2 then 1 else 0 end flg_mature_fspd_30,
case when obs_min_inst_def30 >=3 then 1 else 0 end flg_mature_fstpd_30
from prj-prod-dataplatform.risk_credit_mis.loan_deliquency_data),
base as 
  (select distinct r.customerId,
  r.digitalLoanAccountId,
  loanmaster.loanAccountNumber,
  r.modelDisplayName,
  r.Sil_Alpha_Stack_score,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month, 
  'Test' Data_selection,  
  del.deffpd30,
  del.flg_mature_fpd30,
  loanmaster.new_loan_type,
  modelVersionId, trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile' 
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall' 
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
  from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  inner join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
   left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
  where loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and r.Sil_Alpha_Stack_score is not null
  and del.flg_mature_fpd30 = 1
  )
  select *  from base
  ;
"""
dfd = client.query(sq).to_dataframe()
dfd = dfd.drop_duplicates(keep="first")
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()


# %%
df1 = dfd.copy()

# %% [markdown]
# ### Train

# %%
sq = """ 
  with modelname as 
  (SELECT
    customerId,digitalLoanAccountId,prediction Sil_Alpha_Stack_score,start_time,end_time,modelDisplayName,modelVersionId, 
    case when trenchCategory is null then 'ALL' 
        when trenchCategory = ''then 'ALL' 
        else trenchCategory end as trenchCategory,
    REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature
  FROM prj-prod-dataplatform.dap_ds_poweruser_playground.ml_training_model_run_details
  WHERE modelDisplayName in  ('Alpha - StackingModel', 'alpha_stack_model_sil')
  ),
  deliquency as
(select loanAccountNumber,
case when obs_min_inst_def0 >= 1 and min_inst_def0 = 1 then 1 else 0 end deffpd0,
case when obs_min_inst_def10 >=1 and min_inst_def10 =1 then 1 else 0 end deffpd10,
case when obs_min_inst_def30 >=1 and min_inst_def30 =1 then 1 else 0 end deffpd30,
case when obs_min_inst_def30 >=2 and min_inst_def30 in (1,2) then 1 else 0 end deffspd30,
case when obs_min_inst_def30 >=3 and min_inst_def30 in (1,2,3) then 1 else 0 end deffstpd30,
case when obs_min_inst_def0 >= 1 then 1 else 0 end flg_mature_fpd0,
case when obs_min_inst_def10 >=1 then 1 else 0 end flg_mature_fpd10,
case when obs_min_inst_def30 >=1 then 1 else 0 end flg_mature_fpd30,
case when obs_min_inst_def30 >=2 then 1 else 0 end flg_mature_fspd_30,
case when obs_min_inst_def30 >=3 then 1 else 0 end flg_mature_fstpd_30
from prj-prod-dataplatform.risk_credit_mis.loan_deliquency_data),
base as
(select distinct r.customerId,
  r.digitalLoanAccountId,
  loanmaster.loanAccountNumber,
  r.modelDisplayName,
  r.Sil_Alpha_Stack_score,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month, 
  'Train' Data_selection,  
    del.deffpd30,
  del.flg_mature_fpd30,
  loanmaster.new_loan_type,
  modelVersionId,
    trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile' 
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall' 
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
    from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  inner join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
  left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
  where loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and r.Sil_Alpha_Stack_score is not null
  and del.flg_mature_fpd30 = 1
  )
  select * from base;
  """
dfd = client.query(sq).to_dataframe()
dfd = dfd.drop_duplicates(keep="first")
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()

# %%
df2 = dfd.copy()

# %% [markdown]
# ### Concat

# %%
# df_concat = pd.concat([df1, df2], ignore_index=True)
# print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")

# df_concat = (df_concat
#              .sort_values(by=['digitalLoanAccountId', 'Data_selection'],
#                           key=lambda s: s.map({'Train': 0, 'Test': 1}))
#              .drop_duplicates(subset=['digitalLoanAccountId'], keep='first')
#              .reset_index(drop=True))
# print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")
# df_concat.head()


# 1) Get all IDs present in Train
train_ids = set(df2["digitalLoanAccountId"])

# 2) Keep only Test rows whose ID is NOT in Train
df1_no_dupes = df1[~df1["digitalLoanAccountId"].isin(train_ids)]

# 3) Concatenate
df_concat = pd.concat([df1_no_dupes, df2], ignore_index=True)

print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")
df_concat.head()


# %%
df_concat.groupby(
    ["Data_selection", "new_loan_type", "modelVersionId", "loan_product_type"]
)["digitalLoanAccountId"].nunique()

# %%
fact_table, dimension_table = calculate_periodic_gini_prod_ver_trench_dimfact(
    df_concat,
    "Sil_Alpha_Stack_score",
    "deffpd30",
    "FPD30",
    model_version_column="modelVersionId",
    trench_column="trenchCategory",
    loan_type_column="new_loan_type",
    loan_product_type_column="loan_product_type",
    account_id_column="digitalLoanAccountId",
)

# %%
fact_table, dimension_table = update_tables(
    fact_table, dimension_table, model_name="alpha_stack_model_sil", product="SIL"
)

# %%
# Upload to BigQuery
table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_table"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(fact_table, table_id, job_config=job_config)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_table"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(dimension_table, table_id, job_config=job_config)
job.result()  # Wait for the job to complete

# %%


# %% [markdown]
# ### FSPD30

# %% [markdown]
# ### Test

# %%
sq = """ 
with modelname as 
  (SELECT
    customerId,digitalLoanAccountId,prediction Sil_Alpha_Stack_score,start_time,end_time,modelDisplayName,modelVersionId, 
    case when trenchCategory is null then 'ALL'
         when trenchCategory = '' then 'ALL'
         else trenchCategory end as trenchCategory,
    REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature
  FROM prj-prod-dataplatform.audit_balance.ml_model_run_details
  WHERE modelDisplayName in  ('Alpha - StackingModel', 'alpha_stack_model_sil')
  ),
  deliquency as
(select loanAccountNumber,
case when obs_min_inst_def0 >= 1 and min_inst_def0 = 1 then 1 else 0 end deffpd0,
case when obs_min_inst_def10 >=1 and min_inst_def10 =1 then 1 else 0 end deffpd10,
case when obs_min_inst_def30 >=1 and min_inst_def30 =1 then 1 else 0 end deffpd30,
case when obs_min_inst_def30 >=2 and min_inst_def30 in (1,2) then 1 else 0 end deffspd30,
case when obs_min_inst_def30 >=3 and min_inst_def30 in (1,2,3) then 1 else 0 end deffstpd30,
case when obs_min_inst_def0 >= 1 then 1 else 0 end flg_mature_fpd0,
case when obs_min_inst_def10 >=1 then 1 else 0 end flg_mature_fpd10,
case when obs_min_inst_def30 >=1 then 1 else 0 end flg_mature_fpd30,
case when obs_min_inst_def30 >=2 then 1 else 0 end flg_mature_fspd_30,
case when obs_min_inst_def30 >=3 then 1 else 0 end flg_mature_fstpd_30
from prj-prod-dataplatform.risk_credit_mis.loan_deliquency_data),
base as 
  (select distinct r.customerId,
  r.digitalLoanAccountId,
  loanmaster.loanAccountNumber,
  r.modelDisplayName,
  r.Sil_Alpha_Stack_score,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month, 
  'Test' Data_selection,  
  del.deffspd30,
  del.flg_mature_fspd_30,
  loanmaster.new_loan_type,
  modelVersionId, trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile' 
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall' 
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
  from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  inner join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
   left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
  where loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and r.Sil_Alpha_Stack_score is not null
  and del.flg_mature_fspd_30 = 1
  )
  select *  from base
  ;
"""
dfd = client.query(sq).to_dataframe()
dfd = dfd.drop_duplicates(keep="first")
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()


# %%
df1 = dfd.copy()

# %% [markdown]
# ### Train

# %%
sq = """ 
  with modelname as 
  (SELECT
    customerId,digitalLoanAccountId,prediction Sil_Alpha_Stack_score,start_time,end_time,modelDisplayName,modelVersionId, 
    case when trenchCategory is null then 'ALL' 
        when trenchCategory = ''then 'ALL' 
        else trenchCategory end as trenchCategory,
    REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature
  FROM prj-prod-dataplatform.dap_ds_poweruser_playground.ml_training_model_run_details
  WHERE modelDisplayName in  ('Alpha - StackingModel', 'alpha_stack_model_sil')
  ),
  deliquency as
(select loanAccountNumber,
case when obs_min_inst_def0 >= 1 and min_inst_def0 = 1 then 1 else 0 end deffpd0,
case when obs_min_inst_def10 >=1 and min_inst_def10 =1 then 1 else 0 end deffpd10,
case when obs_min_inst_def30 >=1 and min_inst_def30 =1 then 1 else 0 end deffpd30,
case when obs_min_inst_def30 >=2 and min_inst_def30 in (1,2) then 1 else 0 end deffspd30,
case when obs_min_inst_def30 >=3 and min_inst_def30 in (1,2,3) then 1 else 0 end deffstpd30,
case when obs_min_inst_def0 >= 1 then 1 else 0 end flg_mature_fpd0,
case when obs_min_inst_def10 >=1 then 1 else 0 end flg_mature_fpd10,
case when obs_min_inst_def30 >=1 then 1 else 0 end flg_mature_fpd30,
case when obs_min_inst_def30 >=2 then 1 else 0 end flg_mature_fspd_30,
case when obs_min_inst_def30 >=3 then 1 else 0 end flg_mature_fstpd_30
from prj-prod-dataplatform.risk_credit_mis.loan_deliquency_data),
base as
(select distinct r.customerId,
  r.digitalLoanAccountId,
  loanmaster.loanAccountNumber,
  r.modelDisplayName,
  r.Sil_Alpha_Stack_score,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month, 
  'Train' Data_selection,  
    del.deffspd30,
  del.flg_mature_fspd_30,
  loanmaster.new_loan_type,
  modelVersionId,
    trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile' 
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall' 
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
    from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  inner join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
  left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
  where loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and r.Sil_Alpha_Stack_score is not null
  and del.flg_mature_fspd_30 = 1
  )
  select * from base;
  """
dfd = client.query(sq).to_dataframe()
dfd = dfd.drop_duplicates(keep="first")
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()

# %%
df2 = dfd.copy()

# %% [markdown]
# ### Concat

# %%
# df_concat = pd.concat([df1, df2], ignore_index=True)
# print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")

# df_concat = (df_concat
#              .sort_values(by=['digitalLoanAccountId', 'Data_selection'],
#                           key=lambda s: s.map({'Train': 0, 'Test': 1}))
#              .drop_duplicates(subset=['digitalLoanAccountId'], keep='first')
#              .reset_index(drop=True))
# print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")
# df_concat.head()


# 1) Get all IDs present in Train
train_ids = set(df2["digitalLoanAccountId"])

# 2) Keep only Test rows whose ID is NOT in Train
df1_no_dupes = df1[~df1["digitalLoanAccountId"].isin(train_ids)]

# 3) Concatenate
df_concat = pd.concat([df1_no_dupes, df2], ignore_index=True)

print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")
df_concat.head()


# %%
df_concat["Sil_Alpha_Stack_score"] = pd.to_numeric(
    df_concat["Sil_Alpha_Stack_score"], errors="coerce"
)

# %%
df_concat.groupby(
    ["Data_selection", "new_loan_type", "modelVersionId", "loan_product_type"]
)["digitalLoanAccountId"].nunique()

# %%
fact_table, dimension_table = calculate_periodic_gini_prod_ver_trench_dimfact(
    df_concat,
    "Sil_Alpha_Stack_score",
    "deffspd30",
    "FSPD30",
    model_version_column="modelVersionId",
    trench_column="trenchCategory",
    loan_type_column="new_loan_type",
    loan_product_type_column="loan_product_type",
    account_id_column="digitalLoanAccountId",
)

# %%
fact_table, dimension_table = update_tables(
    fact_table, dimension_table, model_name="alpha_stack_model_sil", product="SIL"
)

# %%
# Upload to BigQuery
table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_table"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(fact_table, table_id, job_config=job_config)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_table"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(dimension_table, table_id, job_config=job_config)
job.result()  # Wait for the job to complete

# %%


# %% [markdown]
# ### FSTPD30

# %% [markdown]
# ### Test

# %%
sq = """ 
with modelname as 
  (SELECT
    customerId,digitalLoanAccountId,prediction Sil_Alpha_Stack_score,start_time,end_time,modelDisplayName,modelVersionId, 
    case when trenchCategory is null then 'ALL'
         when trenchCategory = '' then 'ALL'
         else trenchCategory end as trenchCategory,
    REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature
  FROM prj-prod-dataplatform.audit_balance.ml_model_run_details
  WHERE modelDisplayName in  ('Alpha - StackingModel', 'alpha_stack_model_sil')
  ),
  deliquency as
(select loanAccountNumber,
case when obs_min_inst_def0 >= 1 and min_inst_def0 = 1 then 1 else 0 end deffpd0,
case when obs_min_inst_def10 >=1 and min_inst_def10 =1 then 1 else 0 end deffpd10,
case when obs_min_inst_def30 >=1 and min_inst_def30 =1 then 1 else 0 end deffpd30,
case when obs_min_inst_def30 >=2 and min_inst_def30 in (1,2) then 1 else 0 end deffspd30,
case when obs_min_inst_def30 >=3 and min_inst_def30 in (1,2,3) then 1 else 0 end deffstpd30,
case when obs_min_inst_def0 >= 1 then 1 else 0 end flg_mature_fpd0,
case when obs_min_inst_def10 >=1 then 1 else 0 end flg_mature_fpd10,
case when obs_min_inst_def30 >=1 then 1 else 0 end flg_mature_fpd30,
case when obs_min_inst_def30 >=2 then 1 else 0 end flg_mature_fspd_30,
case when obs_min_inst_def30 >=3 then 1 else 0 end flg_mature_fstpd_30
from prj-prod-dataplatform.risk_credit_mis.loan_deliquency_data),
base as 
  (select distinct r.customerId,
  r.digitalLoanAccountId,
  loanmaster.loanAccountNumber,
  r.modelDisplayName,
  r.Sil_Alpha_Stack_score,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month, 
  'Test' Data_selection,  
  del.deffstpd30,
  del.flg_mature_fstpd_30,
  loanmaster.new_loan_type,
  modelVersionId, trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile' 
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall' 
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
  from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  inner join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
   left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
  where loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and r.Sil_Alpha_Stack_score is not null
  and del.flg_mature_fstpd_30 = 1
  )
  select *  from base
  ;
"""
dfd = client.query(sq).to_dataframe()
dfd = dfd.drop_duplicates(keep="first")
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()


# %%
df1 = dfd.copy()

# %% [markdown]
# ### Train

# %%
sq = """ 
  with modelname as 
  (SELECT
    customerId,digitalLoanAccountId,prediction Sil_Alpha_Stack_score,start_time,end_time,modelDisplayName,modelVersionId, 
    case when trenchCategory is null then 'ALL' 
        when trenchCategory = ''then 'ALL' 
        else trenchCategory end as trenchCategory,
    REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature
  FROM prj-prod-dataplatform.dap_ds_poweruser_playground.ml_training_model_run_details
  WHERE modelDisplayName in  ('Alpha - StackingModel', 'alpha_stack_model_sil')
  ),
  deliquency as
(select loanAccountNumber,
case when obs_min_inst_def0 >= 1 and min_inst_def0 = 1 then 1 else 0 end deffpd0,
case when obs_min_inst_def10 >=1 and min_inst_def10 =1 then 1 else 0 end deffpd10,
case when obs_min_inst_def30 >=1 and min_inst_def30 =1 then 1 else 0 end deffpd30,
case when obs_min_inst_def30 >=2 and min_inst_def30 in (1,2) then 1 else 0 end deffspd30,
case when obs_min_inst_def30 >=3 and min_inst_def30 in (1,2,3) then 1 else 0 end deffstpd30,
case when obs_min_inst_def0 >= 1 then 1 else 0 end flg_mature_fpd0,
case when obs_min_inst_def10 >=1 then 1 else 0 end flg_mature_fpd10,
case when obs_min_inst_def30 >=1 then 1 else 0 end flg_mature_fpd30,
case when obs_min_inst_def30 >=2 then 1 else 0 end flg_mature_fspd_30,
case when obs_min_inst_def30 >=3 then 1 else 0 end flg_mature_fstpd_30
from prj-prod-dataplatform.risk_credit_mis.loan_deliquency_data),
base as
(select distinct r.customerId,
  r.digitalLoanAccountId,
  loanmaster.loanAccountNumber,
  r.modelDisplayName,
  r.Sil_Alpha_Stack_score,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month, 
  'Train' Data_selection,  
    del.deffstpd30,
  del.flg_mature_fstpd_30,
  loanmaster.new_loan_type,
  modelVersionId,
    trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile' 
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall' 
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
    from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  inner join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
  left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
  where loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and r.Sil_Alpha_Stack_score is not null
  and del.flg_mature_fstpd_30 = 1
  )
  select * from base;
  """
dfd = client.query(sq).to_dataframe()
dfd = dfd.drop_duplicates(keep="first")
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()

# %%
df2 = dfd.copy()

# %% [markdown]
# ### Concat

# %%
# df_concat = pd.concat([df1, df2], ignore_index=True)
# print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")

# df_concat = (df_concat
#              .sort_values(by=['digitalLoanAccountId', 'Data_selection'],
#                           key=lambda s: s.map({'Train': 0, 'Test': 1}))
#              .drop_duplicates(subset=['digitalLoanAccountId'], keep='first')
#              .reset_index(drop=True))
# print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")
# df_concat.head()


# 1) Get all IDs present in Train
train_ids = set(df2["digitalLoanAccountId"])

# 2) Keep only Test rows whose ID is NOT in Train
df1_no_dupes = df1[~df1["digitalLoanAccountId"].isin(train_ids)]

# 3) Concatenate
df_concat = pd.concat([df1_no_dupes, df2], ignore_index=True)

print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")
df_concat.head()


# %%
df_concat["Sil_Alpha_Stack_score"] = pd.to_numeric(
    df_concat["Sil_Alpha_Stack_score"], errors="coerce"
)

# %%
df_concat.groupby(
    ["Data_selection", "new_loan_type", "modelVersionId", "loan_product_type"]
)["digitalLoanAccountId"].nunique()

# %%
fact_table, dimension_table = calculate_periodic_gini_prod_ver_trench_dimfact(
    df_concat,
    "Sil_Alpha_Stack_score",
    "deffstpd30",
    "FSTPD30",
    model_version_column="modelVersionId",
    trench_column="trenchCategory",
    loan_type_column="new_loan_type",
    loan_product_type_column="loan_product_type",
    account_id_column="digitalLoanAccountId",
)

# %%
fact_table, dimension_table = update_tables(
    fact_table, dimension_table, model_name="alpha_stack_model_sil", product="SIL"
)
print(f"The shape of the fact table is:\t {fact_table.shape}")
print(f"The shape of the dimension table is:\t {dimension_table.shape}")

# %%
# Upload to BigQuery
table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_table"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(fact_table, table_id, job_config=job_config)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_table"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(dimension_table, table_id, job_config=job_config)
job.result()  # Wait for the job to complete

# %% [markdown]
# ## Beta SIL STACK Score Model

# %%


# %% [markdown]
# ### FPD0

# %% [markdown]
# ### Test

# %%
sq = """ 
WITH cleaned AS (
  SELECT
    customerId,digitalLoanAccountId,prediction,start_time,end_time,modelDisplayName,modelVersionId,
            case when trenchCategory is null then 'ALL'
         when trenchCategory = '' then 'ALL'
         else trenchCategory end as trenchCategory,
    REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature
  FROM `prj-prod-dataplatform.audit_balance.ml_model_run_details`
  WHERE modelDisplayName in ('Beta - StackScoreModel', 'beta_stack_model_sil')
  ), 
  modelname as (
  select customerId,
  digitalLoanAccountId,
  start_time,
  prediction sil_beta_stack_score,
  modelDisplayName,
  modelVersionId,
  trenchCategory,
  from cleaned
  ),
  deliquency as
(select loanAccountNumber,
case when obs_min_inst_def0 >= 1 and min_inst_def0 = 1 then 1 else 0 end deffpd0,
case when obs_min_inst_def10 >=1 and min_inst_def10 =1 then 1 else 0 end deffpd10,
case when obs_min_inst_def30 >=1 and min_inst_def30 =1 then 1 else 0 end deffpd30,
case when obs_min_inst_def30 >=2 and min_inst_def30 in (1,2) then 1 else 0 end deffspd30,
case when obs_min_inst_def30 >=3 and min_inst_def30 in (1,2,3) then 1 else 0 end deffstpd30,
case when obs_min_inst_def0 >= 1 then 1 else 0 end flg_mature_fpd0,
case when obs_min_inst_def10 >=1 then 1 else 0 end flg_mature_fpd10,
case when obs_min_inst_def30 >=1 then 1 else 0 end flg_mature_fpd30,
case when obs_min_inst_def30 >=2 then 1 else 0 end flg_mature_fspd_30,
case when obs_min_inst_def30 >=3 then 1 else 0 end flg_mature_fstpd_30
from prj-prod-dataplatform.risk_credit_mis.loan_deliquency_data),
base as 
  (select distinct r.customerId,
  r.digitalLoanAccountId,
  loanmaster.loanAccountNumber,
  r.modelDisplayName,
  r.sil_beta_stack_score,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month, 
  'Test' Data_selection,  
  deffpd0,
  flg_mature_fpd0,
  loanmaster.new_loan_type,
  modelVersionId, trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile' 
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall' 
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
  from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  inner join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
   left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
  where loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and r.sil_beta_stack_score is not null
  and flg_mature_fpd0 = 1
  )
  select *  from base
  ;
"""
dfd = client.query(sq).to_dataframe()
dfd = dfd.drop_duplicates(keep="first")
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()


# %%
df1 = dfd.copy()

# %% [markdown]
# ### Train

# %%
sq = """ 
WITH cleaned AS (
  SELECT
    customerId,digitalLoanAccountId,prediction,start_time,end_time,modelDisplayName,modelVersionId,
            case when trenchCategory is null then 'ALL'
         when trenchCategory = '' then 'ALL'
         else trenchCategory end as trenchCategory,
    REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature
  FROM prj-prod-dataplatform.dap_ds_poweruser_playground.ml_training_model_run_details
  WHERE modelDisplayName in ('Beta - StackScoreModel', 'beta_stack_model_sil')
  ), 
  modelname as (
  select customerId,
  digitalLoanAccountId,
  start_time,
  prediction sil_beta_stack_score,
  modelDisplayName,
  modelVersionId,
  trenchCategory,
  from cleaned
  ),
  deliquency as
(select loanAccountNumber,
case when obs_min_inst_def0 >= 1 and min_inst_def0 = 1 then 1 else 0 end deffpd0,
case when obs_min_inst_def10 >=1 and min_inst_def10 =1 then 1 else 0 end deffpd10,
case when obs_min_inst_def30 >=1 and min_inst_def30 =1 then 1 else 0 end deffpd30,
case when obs_min_inst_def30 >=2 and min_inst_def30 in (1,2) then 1 else 0 end deffspd30,
case when obs_min_inst_def30 >=3 and min_inst_def30 in (1,2,3) then 1 else 0 end deffstpd30,
case when obs_min_inst_def0 >= 1 then 1 else 0 end flg_mature_fpd0,
case when obs_min_inst_def10 >=1 then 1 else 0 end flg_mature_fpd10,
case when obs_min_inst_def30 >=1 then 1 else 0 end flg_mature_fpd30,
case when obs_min_inst_def30 >=2 then 1 else 0 end flg_mature_fspd_30,
case when obs_min_inst_def30 >=3 then 1 else 0 end flg_mature_fstpd_30
from prj-prod-dataplatform.risk_credit_mis.loan_deliquency_data),
base as
(select distinct r.customerId,
  r.digitalLoanAccountId,
  loanmaster.loanAccountNumber,
  r.modelDisplayName,
  r.sil_beta_stack_score,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month, 
  'Train' Data_selection,  
    deffpd0,
  flg_mature_fpd0,
  loanmaster.new_loan_type,
  modelVersionId,
    trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile' 
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall' 
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
    from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  inner join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
  left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
  where loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and r.sil_beta_stack_score is not null
  and flg_mature_fpd0 = 1
  )
  select * from base;
  """
dfd = client.query(sq).to_dataframe()
dfd = dfd.drop_duplicates(keep="first")
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()

# %%
df2 = dfd.copy()

# %% [markdown]
# ### Concat

# %%
# df_concat = pd.concat([df1, df2], ignore_index=True)
# print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")

# df_concat = (df_concat
#              .sort_values(by=['digitalLoanAccountId', 'Data_selection'],
#                           key=lambda s: s.map({'Train': 0, 'Test': 1}))
#              .drop_duplicates(subset=['digitalLoanAccountId'], keep='first')
#              .reset_index(drop=True))
# print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")
# df_concat.head()

# 1) Get all IDs present in Train
train_ids = set(df2["digitalLoanAccountId"])

# 2) Keep only Test rows whose ID is NOT in Train
df1_no_dupes = df1[~df1["digitalLoanAccountId"].isin(train_ids)]

# 3) Concatenate
df_concat = pd.concat([df1_no_dupes, df2], ignore_index=True)

print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")
df_concat.head()


# %%
df_concat.columns

# %%
df_concat.groupby(
    ["Data_selection", "new_loan_type", "modelVersionId", "loan_product_type"]
)["digitalLoanAccountId"].nunique()

# %%
df_concat["new_loan_type"].unique()

# %%
df_concat["sil_beta_stack_score"] = pd.to_numeric(
    df_concat["sil_beta_stack_score"], errors="coerce"
)

# %%
fact_table, dimension_table = calculate_periodic_gini_prod_ver_trench_dimfact(
    df_concat,
    "sil_beta_stack_score",
    "deffpd0",
    "FPD0",
    model_version_column="modelVersionId",
    trench_column="trenchCategory",
    loan_type_column="new_loan_type",
    loan_product_type_column="loan_product_type",
    account_id_column="digitalLoanAccountId",
)

# %%
fact_table, dimension_table = update_tables(
    fact_table, dimension_table, model_name="beta_stack_model_sil", product="SIL"
)
print(f"The shape of the fact table is:\t {fact_table.shape}")
print(f"The shape of the dimension table is:\t {dimension_table.shape}")

# %%
# Upload to BigQuery
table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_table"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(fact_table, table_id, job_config=job_config)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_table"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(dimension_table, table_id, job_config=job_config)
job.result()  # Wait for the job to complete

# %% [markdown]
# ### FPD10

# %% [markdown]
# ### Test

# %%
sq = """ 
WITH cleaned AS (
  SELECT
    customerId,digitalLoanAccountId,prediction,start_time,end_time,modelDisplayName,modelVersionId,
            case when trenchCategory is null then 'ALL'
         when trenchCategory = '' then 'ALL'
         else trenchCategory end as trenchCategory,
    REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature
  FROM `prj-prod-dataplatform.audit_balance.ml_model_run_details`
  WHERE modelDisplayName in ('Beta - StackScoreModel', 'beta_stack_model_sil')
  ), 
  modelname as (
  select customerId,
  digitalLoanAccountId,
  start_time,
  prediction sil_beta_stack_score,
  modelDisplayName,
  modelVersionId,
  trenchCategory,
  from cleaned
  ),
  deliquency as
(select loanAccountNumber,
case when obs_min_inst_def0 >= 1 and min_inst_def0 = 1 then 1 else 0 end deffpd0,
case when obs_min_inst_def10 >=1 and min_inst_def10 =1 then 1 else 0 end deffpd10,
case when obs_min_inst_def30 >=1 and min_inst_def30 =1 then 1 else 0 end deffpd30,
case when obs_min_inst_def30 >=2 and min_inst_def30 in (1,2) then 1 else 0 end deffspd30,
case when obs_min_inst_def30 >=3 and min_inst_def30 in (1,2,3) then 1 else 0 end deffstpd30,
case when obs_min_inst_def0 >= 1 then 1 else 0 end flg_mature_fpd0,
case when obs_min_inst_def10 >=1 then 1 else 0 end flg_mature_fpd10,
case when obs_min_inst_def30 >=1 then 1 else 0 end flg_mature_fpd30,
case when obs_min_inst_def30 >=2 then 1 else 0 end flg_mature_fspd_30,
case when obs_min_inst_def30 >=3 then 1 else 0 end flg_mature_fstpd_30
from prj-prod-dataplatform.risk_credit_mis.loan_deliquency_data),
base as 
  (select distinct r.customerId,
  r.digitalLoanAccountId,
  loanmaster.loanAccountNumber,
  r.modelDisplayName,
  r.sil_beta_stack_score,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month, 
  'Test' Data_selection,  
  del.deffpd10,
  del.flg_mature_fpd10,
  loanmaster.new_loan_type,
  modelVersionId, trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile' 
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall' 
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
  from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  inner join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
   left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
  where loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and r.sil_beta_stack_score is not null
  and del.flg_mature_fpd10 = 1
  )
  select *  from base
  ;
"""
dfd = client.query(sq).to_dataframe()
dfd = dfd.drop_duplicates(keep="first")
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()


# %%
df1 = dfd.copy()

# %% [markdown]
# ### Train

# %%
sq = """ 
WITH cleaned AS (
  SELECT
    customerId,digitalLoanAccountId,prediction,start_time,end_time,modelDisplayName,modelVersionId,
            case when trenchCategory is null then 'ALL'
         when trenchCategory = '' then 'ALL'
         else trenchCategory end as trenchCategory,
    REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature
  FROM prj-prod-dataplatform.dap_ds_poweruser_playground.ml_training_model_run_details
  WHERE modelDisplayName in ('Beta - StackScoreModel', 'beta_stack_model_sil')
  ), 
  modelname as (
  select customerId,
  digitalLoanAccountId,
  start_time,
  prediction sil_beta_stack_score,
  modelDisplayName,
  modelVersionId,
  trenchCategory,
  from cleaned
  ),
  deliquency as
(select loanAccountNumber,
case when obs_min_inst_def0 >= 1 and min_inst_def0 = 1 then 1 else 0 end deffpd0,
case when obs_min_inst_def10 >=1 and min_inst_def10 =1 then 1 else 0 end deffpd10,
case when obs_min_inst_def30 >=1 and min_inst_def30 =1 then 1 else 0 end deffpd30,
case when obs_min_inst_def30 >=2 and min_inst_def30 in (1,2) then 1 else 0 end deffspd30,
case when obs_min_inst_def30 >=3 and min_inst_def30 in (1,2,3) then 1 else 0 end deffstpd30,
case when obs_min_inst_def0 >= 1 then 1 else 0 end flg_mature_fpd0,
case when obs_min_inst_def10 >=1 then 1 else 0 end flg_mature_fpd10,
case when obs_min_inst_def30 >=1 then 1 else 0 end flg_mature_fpd30,
case when obs_min_inst_def30 >=2 then 1 else 0 end flg_mature_fspd_30,
case when obs_min_inst_def30 >=3 then 1 else 0 end flg_mature_fstpd_30
from prj-prod-dataplatform.risk_credit_mis.loan_deliquency_data),
base as
(select distinct r.customerId,
  r.digitalLoanAccountId,
  loanmaster.loanAccountNumber,
  r.modelDisplayName,
  r.sil_beta_stack_score,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month, 
  'Train' Data_selection,  
    del.deffpd10,
  del.flg_mature_fpd10,
  loanmaster.new_loan_type,
  modelVersionId,
    trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile' 
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall' 
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
    from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  inner join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
  left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
  where loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and r.sil_beta_stack_score is not null
  and del.flg_mature_fpd10 = 1
  )
  select * from base;
  """
dfd = client.query(sq).to_dataframe()
dfd = dfd.drop_duplicates(keep="first")
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()

# %%
df2 = dfd.copy()

# %% [markdown]
# ### Concat

# %%
# df_concat = pd.concat([df1, df2], ignore_index=True)
# print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")

# df_concat = (df_concat
#              .sort_values(by=['digitalLoanAccountId', 'Data_selection'],
#                           key=lambda s: s.map({'Train': 0, 'Test': 1}))
#              .drop_duplicates(subset=['digitalLoanAccountId'], keep='first')
#              .reset_index(drop=True))
# print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")
# df_concat.head()

# 1) Get all IDs present in Train
train_ids = set(df2["digitalLoanAccountId"])

# 2) Keep only Test rows whose ID is NOT in Train
df1_no_dupes = df1[~df1["digitalLoanAccountId"].isin(train_ids)]

# 3) Concatenate
df_concat = pd.concat([df1_no_dupes, df2], ignore_index=True)

print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")
df_concat.head()


# %%
df_concat.groupby(
    ["Data_selection", "new_loan_type", "modelVersionId", "loan_product_type"]
)["digitalLoanAccountId"].nunique()

# %%
fact_table, dimension_table = calculate_periodic_gini_prod_ver_trench_dimfact(
    df_concat,
    "sil_beta_stack_score",
    "deffpd10",
    "FPD10",
    model_version_column="modelVersionId",
    trench_column="trenchCategory",
    loan_type_column="new_loan_type",
    loan_product_type_column="loan_product_type",
    account_id_column="digitalLoanAccountId",
)

# %%
fact_table, dimension_table = update_tables(
    fact_table, dimension_table, model_name="beta_stack_model_sil", product="SIL"
)

# %%
df_concat["sil_beta_stack_score"] = pd.to_numeric(
    df_concat["sil_beta_stack_score"], errors="coerce"
)

# %%
# Upload to BigQuery
table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_table"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(fact_table, table_id, job_config=job_config)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_table"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(dimension_table, table_id, job_config=job_config)
job.result()  # Wait for the job to complete

# %%


# %% [markdown]
# ### FPD30

# %% [markdown]
# ### Test

# %%
sq = """ 
WITH cleaned AS (
  SELECT
    customerId,digitalLoanAccountId,prediction,start_time,end_time,modelDisplayName,modelVersionId,
            case when trenchCategory is null then 'ALL'
         when trenchCategory = '' then 'ALL'
         else trenchCategory end as trenchCategory,
    REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature
  FROM `prj-prod-dataplatform.audit_balance.ml_model_run_details`
  WHERE modelDisplayName in ('Beta - StackScoreModel', 'beta_stack_model_sil')
  ), 
  modelname as (
  select customerId,
  digitalLoanAccountId,
  start_time,
  prediction sil_beta_stack_score,
  modelDisplayName,
  modelVersionId,
  trenchCategory,
  from cleaned
  ),
  deliquency as
(select loanAccountNumber,
case when obs_min_inst_def0 >= 1 and min_inst_def0 = 1 then 1 else 0 end deffpd0,
case when obs_min_inst_def10 >=1 and min_inst_def10 =1 then 1 else 0 end deffpd10,
case when obs_min_inst_def30 >=1 and min_inst_def30 =1 then 1 else 0 end deffpd30,
case when obs_min_inst_def30 >=2 and min_inst_def30 in (1,2) then 1 else 0 end deffspd30,
case when obs_min_inst_def30 >=3 and min_inst_def30 in (1,2,3) then 1 else 0 end deffstpd30,
case when obs_min_inst_def0 >= 1 then 1 else 0 end flg_mature_fpd0,
case when obs_min_inst_def10 >=1 then 1 else 0 end flg_mature_fpd10,
case when obs_min_inst_def30 >=1 then 1 else 0 end flg_mature_fpd30,
case when obs_min_inst_def30 >=2 then 1 else 0 end flg_mature_fspd_30,
case when obs_min_inst_def30 >=3 then 1 else 0 end flg_mature_fstpd_30
from prj-prod-dataplatform.risk_credit_mis.loan_deliquency_data),
base as 
  (select distinct r.customerId,
  r.digitalLoanAccountId,
  loanmaster.loanAccountNumber,
  r.modelDisplayName,
  r.sil_beta_stack_score,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month, 
  'Test' Data_selection,  
  del.deffpd30,
  del.flg_mature_fpd30,
  loanmaster.new_loan_type,
  modelVersionId, trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile' 
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall' 
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
  from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  inner join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
   left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
  where loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and r.sil_beta_stack_score is not null
  and del.flg_mature_fpd30 = 1
  )
  select *  from base
  ;
"""
dfd = client.query(sq).to_dataframe()
dfd = dfd.drop_duplicates(keep="first")
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()


# %%
df1 = dfd.copy()

# %% [markdown]
# ### Train

# %%
sq = """ 
WITH cleaned AS (
  SELECT
    customerId,digitalLoanAccountId,prediction,start_time,end_time,modelDisplayName,modelVersionId,
            case when trenchCategory is null then 'ALL'
         when trenchCategory = '' then 'ALL'
         else trenchCategory end as trenchCategory,
    REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature
  FROM prj-prod-dataplatform.dap_ds_poweruser_playground.ml_training_model_run_details
  WHERE modelDisplayName in ('Beta - StackScoreModel', 'beta_stack_model_sil')
  ), 
  modelname as (
  select customerId,
  digitalLoanAccountId,
  start_time,
  prediction sil_beta_stack_score,
  modelDisplayName,
  modelVersionId,
  trenchCategory,
  from cleaned
  ),
  deliquency as
(select loanAccountNumber,
case when obs_min_inst_def0 >= 1 and min_inst_def0 = 1 then 1 else 0 end deffpd0,
case when obs_min_inst_def10 >=1 and min_inst_def10 =1 then 1 else 0 end deffpd10,
case when obs_min_inst_def30 >=1 and min_inst_def30 =1 then 1 else 0 end deffpd30,
case when obs_min_inst_def30 >=2 and min_inst_def30 in (1,2) then 1 else 0 end deffspd30,
case when obs_min_inst_def30 >=3 and min_inst_def30 in (1,2,3) then 1 else 0 end deffstpd30,
case when obs_min_inst_def0 >= 1 then 1 else 0 end flg_mature_fpd0,
case when obs_min_inst_def10 >=1 then 1 else 0 end flg_mature_fpd10,
case when obs_min_inst_def30 >=1 then 1 else 0 end flg_mature_fpd30,
case when obs_min_inst_def30 >=2 then 1 else 0 end flg_mature_fspd_30,
case when obs_min_inst_def30 >=3 then 1 else 0 end flg_mature_fstpd_30
from prj-prod-dataplatform.risk_credit_mis.loan_deliquency_data),
base as
(select distinct r.customerId,
  r.digitalLoanAccountId,
  loanmaster.loanAccountNumber,
  r.modelDisplayName,
  r.sil_beta_stack_score,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month, 
  'Train' Data_selection,  
    del.deffpd30,
  del.flg_mature_fpd30,
  loanmaster.new_loan_type,
  modelVersionId,
    trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile' 
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall' 
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
    from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  inner join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
  left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
  where loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and r.sil_beta_stack_score is not null
  and del.flg_mature_fpd30 = 1
  )
  select * from base;
  """
dfd = client.query(sq).to_dataframe()
dfd = dfd.drop_duplicates(keep="first")
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()

# %%
df2 = dfd.copy()

# %% [markdown]
# ### Concat

# %%
# df_concat = pd.concat([df1, df2], ignore_index=True)
# print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")

# df_concat = (df_concat
#              .sort_values(by=['digitalLoanAccountId', 'Data_selection'],
#                           key=lambda s: s.map({'Train': 0, 'Test': 1}))
#              .drop_duplicates(subset=['digitalLoanAccountId'], keep='first')
#              .reset_index(drop=True))
# print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")
# df_concat.head()


# 1) Get all IDs present in Train
train_ids = set(df2["digitalLoanAccountId"])

# 2) Keep only Test rows whose ID is NOT in Train
df1_no_dupes = df1[~df1["digitalLoanAccountId"].isin(train_ids)]

# 3) Concatenate
df_concat = pd.concat([df1_no_dupes, df2], ignore_index=True)

print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")
df_concat.head()


# %%
df_concat.groupby(
    ["Data_selection", "new_loan_type", "modelVersionId", "loan_product_type"]
)["digitalLoanAccountId"].nunique()

# %%
df_concat["sil_beta_stack_score"] = pd.to_numeric(
    df_concat["sil_beta_stack_score"], errors="coerce"
)

# %%
fact_table, dimension_table = calculate_periodic_gini_prod_ver_trench_dimfact(
    df_concat,
    "sil_beta_stack_score",
    "deffpd30",
    "FPD30",
    model_version_column="modelVersionId",
    trench_column="trenchCategory",
    loan_type_column="new_loan_type",
    loan_product_type_column="loan_product_type",
    account_id_column="digitalLoanAccountId",
)

# %%
fact_table, dimension_table = update_tables(
    fact_table, dimension_table, model_name="beta_stack_model_sil", product="SIL"
)

# %%
# Upload to BigQuery
table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_table"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(fact_table, table_id, job_config=job_config)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_table"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(dimension_table, table_id, job_config=job_config)
job.result()  # Wait for the job to complete

# %%


# %% [markdown]
# ### FSPD30

# %% [markdown]
# ### Test

# %%
sq = """ 
WITH cleaned AS (
  SELECT
    customerId,digitalLoanAccountId,prediction,start_time,end_time,modelDisplayName,modelVersionId,
            case when trenchCategory is null then 'ALL'
         when trenchCategory = '' then 'ALL'
         else trenchCategory end as trenchCategory,
    REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature
  FROM `prj-prod-dataplatform.audit_balance.ml_model_run_details`
  WHERE modelDisplayName in ('Beta - StackScoreModel', 'beta_stack_model_sil')
  ), 
  modelname as (
  select customerId,
  digitalLoanAccountId,
  start_time,
  prediction sil_beta_stack_score,
  modelDisplayName,
  modelVersionId,
  trenchCategory,
  from cleaned
  ),
  deliquency as
(select loanAccountNumber,
case when obs_min_inst_def0 >= 1 and min_inst_def0 = 1 then 1 else 0 end deffpd0,
case when obs_min_inst_def10 >=1 and min_inst_def10 =1 then 1 else 0 end deffpd10,
case when obs_min_inst_def30 >=1 and min_inst_def30 =1 then 1 else 0 end deffpd30,
case when obs_min_inst_def30 >=2 and min_inst_def30 in (1,2) then 1 else 0 end deffspd30,
case when obs_min_inst_def30 >=3 and min_inst_def30 in (1,2,3) then 1 else 0 end deffstpd30,
case when obs_min_inst_def0 >= 1 then 1 else 0 end flg_mature_fpd0,
case when obs_min_inst_def10 >=1 then 1 else 0 end flg_mature_fpd10,
case when obs_min_inst_def30 >=1 then 1 else 0 end flg_mature_fpd30,
case when obs_min_inst_def30 >=2 then 1 else 0 end flg_mature_fspd_30,
case when obs_min_inst_def30 >=3 then 1 else 0 end flg_mature_fstpd_30
from prj-prod-dataplatform.risk_credit_mis.loan_deliquency_data),
base as 
  (select distinct r.customerId,
  r.digitalLoanAccountId,
  loanmaster.loanAccountNumber,
  r.modelDisplayName,
  r.sil_beta_stack_score,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month, 
  'Test' Data_selection,  
  del.deffspd30,
  del.flg_mature_fspd_30,
  loanmaster.new_loan_type,
  modelVersionId, trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile' 
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall' 
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
  from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  inner join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
   left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
  where loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and r.sil_beta_stack_score is not null
  and del.flg_mature_fspd_30 = 1
  )
  select *  from base
  ;
"""
dfd = client.query(sq).to_dataframe()
dfd = dfd.drop_duplicates(keep="first")
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()


# %%
df1 = dfd.copy()

# %% [markdown]
# ### Train

# %%
sq = """ 
WITH cleaned AS (
  SELECT
    customerId,digitalLoanAccountId,prediction,start_time,end_time,modelDisplayName,modelVersionId,
            case when trenchCategory is null then 'ALL'
         when trenchCategory = '' then 'ALL'
         else trenchCategory end as trenchCategory,
    REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature
  FROM prj-prod-dataplatform.dap_ds_poweruser_playground.ml_training_model_run_details
  WHERE modelDisplayName in ('Beta - StackScoreModel', 'beta_stack_model_sil')
  ), 
  modelname as (
  select customerId,
  digitalLoanAccountId,
  start_time,
  prediction sil_beta_stack_score,
  modelDisplayName,
  modelVersionId,
  trenchCategory,
  from cleaned
  ),
  deliquency as
(select loanAccountNumber,
case when obs_min_inst_def0 >= 1 and min_inst_def0 = 1 then 1 else 0 end deffpd0,
case when obs_min_inst_def10 >=1 and min_inst_def10 =1 then 1 else 0 end deffpd10,
case when obs_min_inst_def30 >=1 and min_inst_def30 =1 then 1 else 0 end deffpd30,
case when obs_min_inst_def30 >=2 and min_inst_def30 in (1,2) then 1 else 0 end deffspd30,
case when obs_min_inst_def30 >=3 and min_inst_def30 in (1,2,3) then 1 else 0 end deffstpd30,
case when obs_min_inst_def0 >= 1 then 1 else 0 end flg_mature_fpd0,
case when obs_min_inst_def10 >=1 then 1 else 0 end flg_mature_fpd10,
case when obs_min_inst_def30 >=1 then 1 else 0 end flg_mature_fpd30,
case when obs_min_inst_def30 >=2 then 1 else 0 end flg_mature_fspd_30,
case when obs_min_inst_def30 >=3 then 1 else 0 end flg_mature_fstpd_30
from prj-prod-dataplatform.risk_credit_mis.loan_deliquency_data),
base as
(select distinct r.customerId,
  r.digitalLoanAccountId,
  loanmaster.loanAccountNumber,
  r.modelDisplayName,
  r.sil_beta_stack_score,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month, 
  'Train' Data_selection,  
    del.deffspd30,
  del.flg_mature_fspd_30,
  loanmaster.new_loan_type,
  modelVersionId,
    trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile' 
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall' 
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
    from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  inner join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
  left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
  where loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and r.sil_beta_stack_score is not null
  and del.flg_mature_fspd_30 = 1
  )
  select * from base;
  """
dfd = client.query(sq).to_dataframe()
dfd = dfd.drop_duplicates(keep="first")
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()

# %%
df2 = dfd.copy()

# %% [markdown]
# ### Concat

# %%
# df_concat = pd.concat([df1, df2], ignore_index=True)
# print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")

# df_concat = (df_concat
#              .sort_values(by=['digitalLoanAccountId', 'Data_selection'],
#                           key=lambda s: s.map({'Train': 0, 'Test': 1}))
#              .drop_duplicates(subset=['digitalLoanAccountId'], keep='first')
#              .reset_index(drop=True))
# print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")
# df_concat.head()


# 1) Get all IDs present in Train
train_ids = set(df2["digitalLoanAccountId"])

# 2) Keep only Test rows whose ID is NOT in Train
df1_no_dupes = df1[~df1["digitalLoanAccountId"].isin(train_ids)]

# 3) Concatenate
df_concat = pd.concat([df1_no_dupes, df2], ignore_index=True)

print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")
df_concat.head()


# %%
df_concat["sil_beta_stack_score"] = pd.to_numeric(
    df_concat["sil_beta_stack_score"], errors="coerce"
)

# %%
fact_table, dimension_table = calculate_periodic_gini_prod_ver_trench_dimfact(
    df_concat,
    "sil_beta_stack_score",
    "deffspd30",
    "FSPD30",
    model_version_column="modelVersionId",
    trench_column="trenchCategory",
    loan_type_column="new_loan_type",
    loan_product_type_column="loan_product_type",
    account_id_column="digitalLoanAccountId",
)

# %%
fact_table, dimension_table = update_tables(
    fact_table, dimension_table, model_name="beta_stack_model_sil", product="SIL"
)

# %%
# Upload to BigQuery
table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_table"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(fact_table, table_id, job_config=job_config)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_table"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(dimension_table, table_id, job_config=job_config)
job.result()  # Wait for the job to complete

# %%


# %% [markdown]
# ### FSTPD30

# %% [markdown]
# ### Test

# %%
sq = """ 
WITH cleaned AS (
  SELECT
    customerId,digitalLoanAccountId,prediction,start_time,end_time,modelDisplayName,modelVersionId,
            case when trenchCategory is null then 'ALL'
         when trenchCategory = '' then 'ALL'
         else trenchCategory end as trenchCategory,
    REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature
  FROM `prj-prod-dataplatform.audit_balance.ml_model_run_details`
  WHERE modelDisplayName in ('Beta - StackScoreModel', 'beta_stack_model_sil')
  ), 
  modelname as (
  select customerId,
  digitalLoanAccountId,
  start_time,
  prediction sil_beta_stack_score,
  modelDisplayName,
  modelVersionId,
  trenchCategory,
  from cleaned
  ),
  deliquency as
(select loanAccountNumber,
case when obs_min_inst_def0 >= 1 and min_inst_def0 = 1 then 1 else 0 end deffpd0,
case when obs_min_inst_def10 >=1 and min_inst_def10 =1 then 1 else 0 end deffpd10,
case when obs_min_inst_def30 >=1 and min_inst_def30 =1 then 1 else 0 end deffpd30,
case when obs_min_inst_def30 >=2 and min_inst_def30 in (1,2) then 1 else 0 end deffspd30,
case when obs_min_inst_def30 >=3 and min_inst_def30 in (1,2,3) then 1 else 0 end deffstpd30,
case when obs_min_inst_def0 >= 1 then 1 else 0 end flg_mature_fpd0,
case when obs_min_inst_def10 >=1 then 1 else 0 end flg_mature_fpd10,
case when obs_min_inst_def30 >=1 then 1 else 0 end flg_mature_fpd30,
case when obs_min_inst_def30 >=2 then 1 else 0 end flg_mature_fspd_30,
case when obs_min_inst_def30 >=3 then 1 else 0 end flg_mature_fstpd_30
from prj-prod-dataplatform.risk_credit_mis.loan_deliquency_data),
base as 
  (select distinct r.customerId,
  r.digitalLoanAccountId,
  loanmaster.loanAccountNumber,
  r.modelDisplayName,
  r.sil_beta_stack_score,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month, 
  'Test' Data_selection,  
  del.deffstpd30,
  del.flg_mature_fstpd_30,
  loanmaster.new_loan_type,
  modelVersionId, trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile' 
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall' 
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
  from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  inner join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
   left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
  where loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and r.sil_beta_stack_score is not null
  and del.flg_mature_fstpd_30 = 1
  )
  select *  from base
  ;
"""
dfd = client.query(sq).to_dataframe()
dfd = dfd.drop_duplicates(keep="first")
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()


# %%
df1 = dfd.copy()

# %% [markdown]
# ### Train

# %%
sq = """ 
WITH cleaned AS (
  SELECT
    customerId,digitalLoanAccountId,prediction,start_time,end_time,modelDisplayName,modelVersionId,
            case when trenchCategory is null then 'ALL'
         when trenchCategory = '' then 'ALL'
         else trenchCategory end as trenchCategory,
    REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature
  FROM prj-prod-dataplatform.dap_ds_poweruser_playground.ml_training_model_run_details
  WHERE modelDisplayName in ('Beta - StackScoreModel', 'beta_stack_model_sil')
  ), 
  modelname as (
  select customerId,
  digitalLoanAccountId,
  start_time,
  prediction sil_beta_stack_score,
  modelDisplayName,
  modelVersionId,
  trenchCategory,
  from cleaned
  ),
  deliquency as
(select loanAccountNumber,
case when obs_min_inst_def0 >= 1 and min_inst_def0 = 1 then 1 else 0 end deffpd0,
case when obs_min_inst_def10 >=1 and min_inst_def10 =1 then 1 else 0 end deffpd10,
case when obs_min_inst_def30 >=1 and min_inst_def30 =1 then 1 else 0 end deffpd30,
case when obs_min_inst_def30 >=2 and min_inst_def30 in (1,2) then 1 else 0 end deffspd30,
case when obs_min_inst_def30 >=3 and min_inst_def30 in (1,2,3) then 1 else 0 end deffstpd30,
case when obs_min_inst_def0 >= 1 then 1 else 0 end flg_mature_fpd0,
case when obs_min_inst_def10 >=1 then 1 else 0 end flg_mature_fpd10,
case when obs_min_inst_def30 >=1 then 1 else 0 end flg_mature_fpd30,
case when obs_min_inst_def30 >=2 then 1 else 0 end flg_mature_fspd_30,
case when obs_min_inst_def30 >=3 then 1 else 0 end flg_mature_fstpd_30
from prj-prod-dataplatform.risk_credit_mis.loan_deliquency_data),
base as
(select distinct r.customerId,
  r.digitalLoanAccountId,
  loanmaster.loanAccountNumber,
  r.modelDisplayName,
  r.sil_beta_stack_score,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month, 
  'Train' Data_selection,  
    del.deffstpd30,
  del.flg_mature_fstpd_30,
  loanmaster.new_loan_type,
  modelVersionId,
    trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile' 
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall' 
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
    from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  inner join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
  left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
  where loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and r.sil_beta_stack_score is not null
  and del.flg_mature_fstpd_30 = 1
  )
  select * from base;
  """
dfd = client.query(sq).to_dataframe()
dfd = dfd.drop_duplicates(keep="first")
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()

# %%
df2 = dfd.copy()

# %% [markdown]
# ### Concat

# %%
# df_concat = pd.concat([df1, df2], ignore_index=True)
# print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")

# df_concat = (df_concat
#              .sort_values(by=['digitalLoanAccountId', 'Data_selection'],
#                           key=lambda s: s.map({'Train': 0, 'Test': 1}))
#              .drop_duplicates(subset=['digitalLoanAccountId'], keep='first')
#              .reset_index(drop=True))
# print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")
# df_concat.head()


# 1) Get all IDs present in Train
train_ids = set(df2["digitalLoanAccountId"])

# 2) Keep only Test rows whose ID is NOT in Train
df1_no_dupes = df1[~df1["digitalLoanAccountId"].isin(train_ids)]

# 3) Concatenate
df_concat = pd.concat([df1_no_dupes, df2], ignore_index=True)

print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")
df_concat.head()


# %%
df_concat["sil_beta_stack_score"] = pd.to_numeric(
    df_concat["sil_beta_stack_score"], errors="coerce"
)

# %%
fact_table, dimension_table = calculate_periodic_gini_prod_ver_trench_dimfact(
    df_concat,
    "sil_beta_stack_score",
    "deffstpd30",
    "FSTPD30",
    model_version_column="modelVersionId",
    trench_column="trenchCategory",
    loan_type_column="new_loan_type",
    loan_product_type_column="loan_product_type",
    account_id_column="digitalLoanAccountId",
)

# %%
fact_table, dimension_table = update_tables(
    fact_table, dimension_table, model_name="beta_stack_model_sil", product="SIL"
)
print(f"The shape of the fact table is:\t {fact_table.shape}")
print(f"The shape of the dimension table is:\t {dimension_table.shape}")

# %%
# Upload to BigQuery
table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_table"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(fact_table, table_id, job_config=job_config)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_table"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(dimension_table, table_id, job_config=job_config)
job.result()  # Wait for the job to complete

# %%


# %% [markdown]
# ## Beta Sil App Score

# %% [markdown]
# ### FPD0

# %% [markdown]
# ### Test

# %%
sq = """ 
WITH cleaned AS (
  SELECT
    customerId,digitalLoanAccountId,prediction,start_time,end_time,modelDisplayName,modelVersionId,
    case when trenchCategory is null then 'ALL'
 when trenchCategory = '' then 'ALL'
 else trenchCategory end as trenchCategory,

    REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature,
    REPLACE(REPLACE(prediction, "'", '"'), "None", "null") AS prediction_clean
  FROM `prj-prod-dataplatform.audit_balance.ml_model_run_details`
  WHERE modelDisplayName in ('Beta - AppsScoreModel', 'apps_score_model_sil')
   ), 
  modelname as (
  select customerId,
  digitalLoanAccountId,
  start_time,
  safe_cast(JSON_VALUE(prediction_clean, "$.combined_score") AS float64) as sil_beta_app_score,
  case when modelDisplayName = 'Beta - AppsScoreModel' then 'apps_score_model_sil'
       else modelDisplayName end as modelDisplayName,
  modelVersionId,
  trenchCategory,
  from cleaned
  ),
  deliquency as
(select loanAccountNumber,
case when obs_min_inst_def0 >= 1 and min_inst_def0 = 1 then 1 else 0 end deffpd0,
case when obs_min_inst_def10 >=1 and min_inst_def10 =1 then 1 else 0 end deffpd10,
case when obs_min_inst_def30 >=1 and min_inst_def30 =1 then 1 else 0 end deffpd30,
case when obs_min_inst_def30 >=2 and min_inst_def30 in (1,2) then 1 else 0 end deffspd30,
case when obs_min_inst_def30 >=3 and min_inst_def30 in (1,2,3) then 1 else 0 end deffstpd30,
case when obs_min_inst_def0 >= 1 then 1 else 0 end flg_mature_fpd0,
case when obs_min_inst_def10 >=1 then 1 else 0 end flg_mature_fpd10,
case when obs_min_inst_def30 >=1 then 1 else 0 end flg_mature_fpd30,
case when obs_min_inst_def30 >=2 then 1 else 0 end flg_mature_fspd_30,
case when obs_min_inst_def30 >=3 then 1 else 0 end flg_mature_fstpd_30
from prj-prod-dataplatform.risk_credit_mis.loan_deliquency_data),
base as 
  (select distinct r.customerId,
  r.digitalLoanAccountId,
  loanmaster.loanAccountNumber,
  r.modelDisplayName,
  r.sil_beta_app_score,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month, 
  'Test' Data_selection,  
  deffpd0,
  flg_mature_fpd0,
  loanmaster.new_loan_type,
  modelVersionId, trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile' 
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall' 
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
  from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  inner join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
   left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
  where loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and r.sil_beta_app_score is not null
  and flg_mature_fpd0 = 1
  )
  select *  from base
  ;
"""
dfd = client.query(sq).to_dataframe()
dfd = dfd.drop_duplicates(keep="first")
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()


# %%
df1 = dfd.copy()

# %% [markdown]
# ### Train

# %%
sq = """ 
WITH cleaned AS (
  SELECT
    customerId,digitalLoanAccountId,prediction,start_time,end_time,modelDisplayName,modelVersionId,
        case when trenchCategory is null then 'ALL'
 when trenchCategory = '' then 'ALL'
 else trenchCategory end as trenchCategory,
    REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature,
    REPLACE(REPLACE(cast(prediction as string), "'", '"'), "None", "null") AS prediction_clean
  FROM prj-prod-dataplatform.dap_ds_poweruser_playground.ml_training_model_run_details
  WHERE modelDisplayName in ('Beta - AppsScoreModel', 'apps_score_model_sil')
    ),
  modelname as (
  select customerId,
  digitalLoanAccountId,
  start_time,
  coalesce(prediction, safe_cast(JSON_VALUE(prediction_clean, "$.combined_score") AS float64)) as sil_beta_app_score, 
  case when modelDisplayName = 'Beta - AppsScoreModel' then 'apps_score_model_sil'
       else modelDisplayName end as modelDisplayName,
  modelVersionId,
  trenchCategory
  from cleaned
  ),
  deliquency as
(select loanAccountNumber,
case when obs_min_inst_def0 >= 1 and min_inst_def0 = 1 then 1 else 0 end deffpd0,
case when obs_min_inst_def10 >=1 and min_inst_def10 =1 then 1 else 0 end deffpd10,
case when obs_min_inst_def30 >=1 and min_inst_def30 =1 then 1 else 0 end deffpd30,
case when obs_min_inst_def30 >=2 and min_inst_def30 in (1,2) then 1 else 0 end deffspd30,
case when obs_min_inst_def30 >=3 and min_inst_def30 in (1,2,3) then 1 else 0 end deffstpd30,
case when obs_min_inst_def0 >= 1 then 1 else 0 end flg_mature_fpd0,
case when obs_min_inst_def10 >=1 then 1 else 0 end flg_mature_fpd10,
case when obs_min_inst_def30 >=1 then 1 else 0 end flg_mature_fpd30,
case when obs_min_inst_def30 >=2 then 1 else 0 end flg_mature_fspd_30,
case when obs_min_inst_def30 >=3 then 1 else 0 end flg_mature_fstpd_30
from prj-prod-dataplatform.risk_credit_mis.loan_deliquency_data),
base as
(select distinct r.customerId,
  r.digitalLoanAccountId,
  loanmaster.loanAccountNumber,
  r.modelDisplayName,
  r.sil_beta_app_score,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month, 
  'Train' Data_selection,  
    deffpd0,
  flg_mature_fpd0,
  loanmaster.new_loan_type,
  modelVersionId,
    trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile' 
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall' 
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
    from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  inner join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
  left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
  where loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and r.sil_beta_app_score is not null
  and flg_mature_fpd0 = 1
  )
  select * from base;
  """
dfd = client.query(sq).to_dataframe()
dfd = dfd.drop_duplicates(keep="first")
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()

# %%
df2 = dfd.copy()

# %% [markdown]
# ### Concat

# %%
# df_concat = pd.concat([df1, df2], ignore_index=True)
# print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")

# df_concat = (df_concat
#              .sort_values(by=['digitalLoanAccountId', 'Data_selection'],
#                           key=lambda s: s.map({'Train': 0, 'Test': 1}))
#              .drop_duplicates(subset=['digitalLoanAccountId'], keep='first')
#              .reset_index(drop=True))
# print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")
# df_concat.head()


# 1) Get all IDs present in Train
train_ids = set(df2["digitalLoanAccountId"])

# 2) Keep only Test rows whose ID is NOT in Train
df1_no_dupes = df1[~df1["digitalLoanAccountId"].isin(train_ids)]

# 3) Concatenate
df_concat = pd.concat([df1_no_dupes, df2], ignore_index=True)

print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")
df_concat.head()


# %%
df_concat["sil_beta_app_score"] = pd.to_numeric(
    df_concat["sil_beta_app_score"], errors="coerce"
)

# %%
fact_table, dimension_table = calculate_periodic_gini_prod_ver_trench_dimfact(
    df_concat,
    "sil_beta_app_score",
    "deffpd0",
    "FPD0",
    model_version_column="modelVersionId",
    trench_column="trenchCategory",
    loan_type_column="new_loan_type",
    loan_product_type_column="loan_product_type",
    account_id_column="digitalLoanAccountId",
)

# %%
fact_table, dimension_table = update_tables(
    fact_table, dimension_table, model_name="apps_score_model_sil", product="SIL"
)
print(f"The shape of the fact table is:\t {fact_table.shape}")
print(f"The shape of the dimension table is:\t {dimension_table.shape}")

# %%
# Upload to BigQuery
table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_table"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(fact_table, table_id, job_config=job_config)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_table"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(dimension_table, table_id, job_config=job_config)
job.result()  # Wait for the job to complete

# %% [markdown]
# ### FPD10

# %% [markdown]
# ### Test

# %%
sq = """ 
WITH cleaned AS (
  SELECT
    customerId,digitalLoanAccountId,prediction,start_time,end_time,modelDisplayName,modelVersionId,
    case when trenchCategory is null then 'ALL'
 when trenchCategory = '' then 'ALL'
 else trenchCategory end as trenchCategory,

    REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature,
    REPLACE(REPLACE(prediction, "'", '"'), "None", "null") AS prediction_clean
  FROM `prj-prod-dataplatform.audit_balance.ml_model_run_details`
  WHERE modelDisplayName in ('Beta - AppsScoreModel', 'apps_score_model_sil')
   ), 
  modelname as (
  select customerId,
  digitalLoanAccountId,
  start_time,
  safe_cast(JSON_VALUE(prediction_clean, "$.combined_score") AS float64) as sil_beta_app_score,
  case when modelDisplayName = 'Beta - AppsScoreModel' then 'apps_score_model_sil'
       else modelDisplayName end as modelDisplayName,
  modelVersionId,
  trenchCategory,
  from cleaned
  ),
  deliquency as
(select loanAccountNumber,
case when obs_min_inst_def0 >= 1 and min_inst_def0 = 1 then 1 else 0 end deffpd0,
case when obs_min_inst_def10 >=1 and min_inst_def10 =1 then 1 else 0 end deffpd10,
case when obs_min_inst_def30 >=1 and min_inst_def30 =1 then 1 else 0 end deffpd30,
case when obs_min_inst_def30 >=2 and min_inst_def30 in (1,2) then 1 else 0 end deffspd30,
case when obs_min_inst_def30 >=3 and min_inst_def30 in (1,2,3) then 1 else 0 end deffstpd30,
case when obs_min_inst_def0 >= 1 then 1 else 0 end flg_mature_fpd0,
case when obs_min_inst_def10 >=1 then 1 else 0 end flg_mature_fpd10,
case when obs_min_inst_def30 >=1 then 1 else 0 end flg_mature_fpd30,
case when obs_min_inst_def30 >=2 then 1 else 0 end flg_mature_fspd_30,
case when obs_min_inst_def30 >=3 then 1 else 0 end flg_mature_fstpd_30
from prj-prod-dataplatform.risk_credit_mis.loan_deliquency_data),
base as 
  (select distinct r.customerId,
  r.digitalLoanAccountId,
  loanmaster.loanAccountNumber,
  r.modelDisplayName,
  r.sil_beta_app_score,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month, 
  'Test' Data_selection,  
  del.deffpd10,
  del.flg_mature_fpd10,
  loanmaster.new_loan_type,
  modelVersionId, trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile' 
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall' 
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
  from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  inner join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
   left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
  where loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and r.sil_beta_app_score is not null
  and del.flg_mature_fpd10 = 1
  )
  select *  from base
  ;
"""
dfd = client.query(sq).to_dataframe()
dfd = dfd.drop_duplicates(keep="first")
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()


# %%
df1 = dfd.copy()

# %% [markdown]
# ### Train

# %%
sq = """ 
WITH cleaned AS (
  SELECT
    customerId,digitalLoanAccountId,prediction,start_time,end_time,modelDisplayName,modelVersionId,
        case when trenchCategory is null then 'ALL'
 when trenchCategory = '' then 'ALL'
 else trenchCategory end as trenchCategory,
    REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature,
    REPLACE(REPLACE(cast(prediction as string), "'", '"'), "None", "null") AS prediction_clean
  FROM prj-prod-dataplatform.dap_ds_poweruser_playground.ml_training_model_run_details
  WHERE modelDisplayName in ('Beta - AppsScoreModel', 'apps_score_model_sil')
    ),
  modelname as (
  select customerId,
  digitalLoanAccountId,
  start_time,
  coalesce(prediction, safe_cast(JSON_VALUE(prediction_clean, "$.combined_score") AS float64)) as sil_beta_app_score, 
  case when modelDisplayName = 'Beta - AppsScoreModel' then 'apps_score_model_sil'
       else modelDisplayName end as modelDisplayName,
  modelVersionId,
  trenchCategory
  from cleaned
  ),
  deliquency as
(select loanAccountNumber,
case when obs_min_inst_def0 >= 1 and min_inst_def0 = 1 then 1 else 0 end deffpd0,
case when obs_min_inst_def10 >=1 and min_inst_def10 =1 then 1 else 0 end deffpd10,
case when obs_min_inst_def30 >=1 and min_inst_def30 =1 then 1 else 0 end deffpd30,
case when obs_min_inst_def30 >=2 and min_inst_def30 in (1,2) then 1 else 0 end deffspd30,
case when obs_min_inst_def30 >=3 and min_inst_def30 in (1,2,3) then 1 else 0 end deffstpd30,
case when obs_min_inst_def0 >= 1 then 1 else 0 end flg_mature_fpd0,
case when obs_min_inst_def10 >=1 then 1 else 0 end flg_mature_fpd10,
case when obs_min_inst_def30 >=1 then 1 else 0 end flg_mature_fpd30,
case when obs_min_inst_def30 >=2 then 1 else 0 end flg_mature_fspd_30,
case when obs_min_inst_def30 >=3 then 1 else 0 end flg_mature_fstpd_30
from prj-prod-dataplatform.risk_credit_mis.loan_deliquency_data),
base as
(select distinct r.customerId,
  r.digitalLoanAccountId,
  loanmaster.loanAccountNumber,
  r.modelDisplayName,
  r.sil_beta_app_score,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month, 
  'Train' Data_selection,  
    del.deffpd10,
  del.flg_mature_fpd10,
  loanmaster.new_loan_type,
  modelVersionId,
    trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile' 
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall' 
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
    from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  inner join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
  left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
  where loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and r.sil_beta_app_score is not null
  and del.flg_mature_fpd10 = 1
  )
  select * from base;
  """
dfd = client.query(sq).to_dataframe()
dfd = dfd.drop_duplicates(keep="first")
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()

# %%
df2 = dfd.copy()

# %% [markdown]
# ### Concat

# %%
# df_concat = pd.concat([df1, df2], ignore_index=True)
# print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")

# df_concat = (df_concat
#              .sort_values(by=['digitalLoanAccountId', 'Data_selection'],
#                           key=lambda s: s.map({'Train': 0, 'Test': 1}))
#              .drop_duplicates(subset=['digitalLoanAccountId'], keep='first')
#              .reset_index(drop=True))
# print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")
# df_concat.head()


# 1) Get all IDs present in Train
train_ids = set(df2["digitalLoanAccountId"])

# 2) Keep only Test rows whose ID is NOT in Train
df1_no_dupes = df1[~df1["digitalLoanAccountId"].isin(train_ids)]

# 3) Concatenate
df_concat = pd.concat([df1_no_dupes, df2], ignore_index=True)

print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")
df_concat.head()


# %%
df_concat["sil_beta_app_score"] = pd.to_numeric(
    df_concat["sil_beta_app_score"], errors="coerce"
)

# %%
fact_table, dimension_table = calculate_periodic_gini_prod_ver_trench_dimfact(
    df_concat,
    "sil_beta_app_score",
    "deffpd10",
    "FPD10",
    model_version_column="modelVersionId",
    trench_column="trenchCategory",
    loan_type_column="new_loan_type",
    loan_product_type_column="loan_product_type",
    account_id_column="digitalLoanAccountId",
)

# %%
fact_table, dimension_table = update_tables(
    fact_table, dimension_table, model_name="apps_score_model_sil", product="SIL"
)

# %%


# %%
# Upload to BigQuery
table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_table"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(fact_table, table_id, job_config=job_config)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_table"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(dimension_table, table_id, job_config=job_config)
job.result()  # Wait for the job to complete

# %%


# %% [markdown]
# ### FPD30

# %% [markdown]
# ### Test

# %%
sq = """ 
WITH cleaned AS (
  SELECT
    customerId,digitalLoanAccountId,prediction,start_time,end_time,modelDisplayName,modelVersionId,
    case when trenchCategory is null then 'ALL'
 when trenchCategory = '' then 'ALL'
 else trenchCategory end as trenchCategory,

    REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature,
    REPLACE(REPLACE(prediction, "'", '"'), "None", "null") AS prediction_clean
  FROM `prj-prod-dataplatform.audit_balance.ml_model_run_details`
  WHERE modelDisplayName in ('Beta - AppsScoreModel', 'apps_score_model_sil')
   ), 
  modelname as (
  select customerId,
  digitalLoanAccountId,
  start_time,
  safe_cast(JSON_VALUE(prediction_clean, "$.combined_score") AS float64) as sil_beta_app_score,
  case when modelDisplayName = 'Beta - AppsScoreModel' then 'apps_score_model_sil'
       else modelDisplayName end as modelDisplayName,
  modelVersionId,
  trenchCategory,
  from cleaned
  ),
  deliquency as
(select loanAccountNumber,
case when obs_min_inst_def0 >= 1 and min_inst_def0 = 1 then 1 else 0 end deffpd0,
case when obs_min_inst_def10 >=1 and min_inst_def10 =1 then 1 else 0 end deffpd10,
case when obs_min_inst_def30 >=1 and min_inst_def30 =1 then 1 else 0 end deffpd30,
case when obs_min_inst_def30 >=2 and min_inst_def30 in (1,2) then 1 else 0 end deffspd30,
case when obs_min_inst_def30 >=3 and min_inst_def30 in (1,2,3) then 1 else 0 end deffstpd30,
case when obs_min_inst_def0 >= 1 then 1 else 0 end flg_mature_fpd0,
case when obs_min_inst_def10 >=1 then 1 else 0 end flg_mature_fpd10,
case when obs_min_inst_def30 >=1 then 1 else 0 end flg_mature_fpd30,
case when obs_min_inst_def30 >=2 then 1 else 0 end flg_mature_fspd_30,
case when obs_min_inst_def30 >=3 then 1 else 0 end flg_mature_fstpd_30
from prj-prod-dataplatform.risk_credit_mis.loan_deliquency_data),
base as 
  (select distinct r.customerId,
  r.digitalLoanAccountId,
  loanmaster.loanAccountNumber,
  r.modelDisplayName,
  r.sil_beta_app_score,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month, 
  'Test' Data_selection,  
  del.deffpd30,
  del.flg_mature_fpd30,
  loanmaster.new_loan_type,
  modelVersionId, trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile' 
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall' 
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
  from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  inner join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
   left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
  where loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and r.sil_beta_app_score is not null
  and del.flg_mature_fpd30 = 1
  )
  select *  from base
  ;
"""
dfd = client.query(sq).to_dataframe()
dfd = dfd.drop_duplicates(keep="first")
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()


# %%
df1 = dfd.copy()

# %% [markdown]
# ### Train

# %%
sq = """ 
WITH cleaned AS (
  SELECT
    customerId,digitalLoanAccountId,prediction,start_time,end_time,modelDisplayName,modelVersionId,
        case when trenchCategory is null then 'ALL'
 when trenchCategory = '' then 'ALL'
 else trenchCategory end as trenchCategory,
    REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature,
    REPLACE(REPLACE(cast(prediction as string), "'", '"'), "None", "null") AS prediction_clean
  FROM prj-prod-dataplatform.dap_ds_poweruser_playground.ml_training_model_run_details
  WHERE modelDisplayName in ('Beta - AppsScoreModel', 'apps_score_model_sil')
    ),
  modelname as (
  select customerId,
  digitalLoanAccountId,
  start_time,
  coalesce(prediction, safe_cast(JSON_VALUE(prediction_clean, "$.combined_score") AS float64)) as sil_beta_app_score, 
  case when modelDisplayName = 'Beta - AppsScoreModel' then 'apps_score_model_sil'
       else modelDisplayName end as modelDisplayName,
  modelVersionId,
  trenchCategory
  from cleaned
  ),
  deliquency as
(select loanAccountNumber,
case when obs_min_inst_def0 >= 1 and min_inst_def0 = 1 then 1 else 0 end deffpd0,
case when obs_min_inst_def10 >=1 and min_inst_def10 =1 then 1 else 0 end deffpd10,
case when obs_min_inst_def30 >=1 and min_inst_def30 =1 then 1 else 0 end deffpd30,
case when obs_min_inst_def30 >=2 and min_inst_def30 in (1,2) then 1 else 0 end deffspd30,
case when obs_min_inst_def30 >=3 and min_inst_def30 in (1,2,3) then 1 else 0 end deffstpd30,
case when obs_min_inst_def0 >= 1 then 1 else 0 end flg_mature_fpd0,
case when obs_min_inst_def10 >=1 then 1 else 0 end flg_mature_fpd10,
case when obs_min_inst_def30 >=1 then 1 else 0 end flg_mature_fpd30,
case when obs_min_inst_def30 >=2 then 1 else 0 end flg_mature_fspd_30,
case when obs_min_inst_def30 >=3 then 1 else 0 end flg_mature_fstpd_30
from prj-prod-dataplatform.risk_credit_mis.loan_deliquency_data),
base as
(select distinct r.customerId,
  r.digitalLoanAccountId,
  loanmaster.loanAccountNumber,
  r.modelDisplayName,
  r.sil_beta_app_score,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month, 
  'Train' Data_selection,  
    del.deffpd30,
  del.flg_mature_fpd30,
  loanmaster.new_loan_type,
  modelVersionId,
    trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile' 
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall' 
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
    from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  inner join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
  left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
  where loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and r.sil_beta_app_score is not null
  and del.flg_mature_fpd30 = 1
  )
  select * from base;
  """
dfd = client.query(sq).to_dataframe()
dfd = dfd.drop_duplicates(keep="first")
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()

# %%
df2 = dfd.copy()

# %% [markdown]
# ### Concat

# %%
# df_concat = pd.concat([df1, df2], ignore_index=True)
# print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")

# df_concat = (df_concat
#              .sort_values(by=['digitalLoanAccountId', 'Data_selection'],
#                           key=lambda s: s.map({'Train': 0, 'Test': 1}))
#              .drop_duplicates(subset=['digitalLoanAccountId'], keep='first')
#              .reset_index(drop=True))
# print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")
# df_concat.head()


# 1) Get all IDs present in Train
train_ids = set(df2["digitalLoanAccountId"])

# 2) Keep only Test rows whose ID is NOT in Train
df1_no_dupes = df1[~df1["digitalLoanAccountId"].isin(train_ids)]

# 3) Concatenate
df_concat = pd.concat([df1_no_dupes, df2], ignore_index=True)

print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")
df_concat.head()


# %%
df_concat["sil_beta_app_score"] = pd.to_numeric(
    df_concat["sil_beta_app_score"], errors="coerce"
)

# %%
fact_table, dimension_table = calculate_periodic_gini_prod_ver_trench_dimfact(
    df_concat,
    "sil_beta_app_score",
    "deffpd30",
    "FPD30",
    model_version_column="modelVersionId",
    trench_column="trenchCategory",
    loan_type_column="new_loan_type",
    loan_product_type_column="loan_product_type",
    account_id_column="digitalLoanAccountId",
)

# %%
fact_table, dimension_table = update_tables(
    fact_table, dimension_table, model_name="apps_score_model_sil", product="SIL"
)

# %%
# Upload to BigQuery
table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_table"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(fact_table, table_id, job_config=job_config)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_table"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(dimension_table, table_id, job_config=job_config)
job.result()  # Wait for the job to complete

# %%


# %% [markdown]
# ### FSPD30

# %% [markdown]
# ### Test

# %%
sq = """ 
WITH cleaned AS (
  SELECT
    customerId,digitalLoanAccountId,prediction,start_time,end_time,modelDisplayName,modelVersionId,
    case when trenchCategory is null then 'ALL'
 when trenchCategory = '' then 'ALL'
 else trenchCategory end as trenchCategory,

    REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature,
    REPLACE(REPLACE(prediction, "'", '"'), "None", "null") AS prediction_clean
  FROM `prj-prod-dataplatform.audit_balance.ml_model_run_details`
  WHERE modelDisplayName in ('Beta - AppsScoreModel', 'apps_score_model_sil')
   ), 
  modelname as (
  select customerId,
  digitalLoanAccountId,
  start_time,
  safe_cast(JSON_VALUE(prediction_clean, "$.combined_score") AS float64) as sil_beta_app_score,
  case when modelDisplayName = 'Beta - AppsScoreModel' then 'apps_score_model_sil'
       else modelDisplayName end as modelDisplayName,
  modelVersionId,
  trenchCategory,
  from cleaned
  ),
  deliquency as
(select loanAccountNumber,
case when obs_min_inst_def0 >= 1 and min_inst_def0 = 1 then 1 else 0 end deffpd0,
case when obs_min_inst_def10 >=1 and min_inst_def10 =1 then 1 else 0 end deffpd10,
case when obs_min_inst_def30 >=1 and min_inst_def30 =1 then 1 else 0 end deffpd30,
case when obs_min_inst_def30 >=2 and min_inst_def30 in (1,2) then 1 else 0 end deffspd30,
case when obs_min_inst_def30 >=3 and min_inst_def30 in (1,2,3) then 1 else 0 end deffstpd30,
case when obs_min_inst_def0 >= 1 then 1 else 0 end flg_mature_fpd0,
case when obs_min_inst_def10 >=1 then 1 else 0 end flg_mature_fpd10,
case when obs_min_inst_def30 >=1 then 1 else 0 end flg_mature_fpd30,
case when obs_min_inst_def30 >=2 then 1 else 0 end flg_mature_fspd_30,
case when obs_min_inst_def30 >=3 then 1 else 0 end flg_mature_fstpd_30
from prj-prod-dataplatform.risk_credit_mis.loan_deliquency_data),
base as 
  (select distinct r.customerId,
  r.digitalLoanAccountId,
  loanmaster.loanAccountNumber,
  r.modelDisplayName,
  r.sil_beta_app_score,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month, 
  'Test' Data_selection,  
  del.deffspd30,
  del.flg_mature_fspd_30,
  loanmaster.new_loan_type,
  modelVersionId, trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile' 
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall' 
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
  from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  inner join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
   left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
  where loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and r.sil_beta_app_score is not null
  and del.flg_mature_fspd_30 = 1
  )
  select *  from base
  ;
"""
dfd = client.query(sq).to_dataframe()
dfd = dfd.drop_duplicates(keep="first")
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()


# %%
df1 = dfd.copy()

# %% [markdown]
# ### Train

# %%
sq = """ 
WITH cleaned AS (
  SELECT
    customerId,digitalLoanAccountId,prediction,start_time,end_time,modelDisplayName,modelVersionId,
        case when trenchCategory is null then 'ALL'
 when trenchCategory = '' then 'ALL'
 else trenchCategory end as trenchCategory,
    REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature,
    REPLACE(REPLACE(cast(prediction as string), "'", '"'), "None", "null") AS prediction_clean
  FROM prj-prod-dataplatform.dap_ds_poweruser_playground.ml_training_model_run_details
  WHERE modelDisplayName in ('Beta - AppsScoreModel', 'apps_score_model_sil')
    ),
  modelname as (
  select customerId,
  digitalLoanAccountId,
  start_time,
  coalesce(prediction, safe_cast(JSON_VALUE(prediction_clean, "$.combined_score") AS float64)) as sil_beta_app_score, 
  case when modelDisplayName = 'Beta - AppsScoreModel' then 'apps_score_model_sil'
       else modelDisplayName end as modelDisplayName,
  modelVersionId,
  trenchCategory
  from cleaned
  ),
  deliquency as
(select loanAccountNumber,
case when obs_min_inst_def0 >= 1 and min_inst_def0 = 1 then 1 else 0 end deffpd0,
case when obs_min_inst_def10 >=1 and min_inst_def10 =1 then 1 else 0 end deffpd10,
case when obs_min_inst_def30 >=1 and min_inst_def30 =1 then 1 else 0 end deffpd30,
case when obs_min_inst_def30 >=2 and min_inst_def30 in (1,2) then 1 else 0 end deffspd30,
case when obs_min_inst_def30 >=3 and min_inst_def30 in (1,2,3) then 1 else 0 end deffstpd30,
case when obs_min_inst_def0 >= 1 then 1 else 0 end flg_mature_fpd0,
case when obs_min_inst_def10 >=1 then 1 else 0 end flg_mature_fpd10,
case when obs_min_inst_def30 >=1 then 1 else 0 end flg_mature_fpd30,
case when obs_min_inst_def30 >=2 then 1 else 0 end flg_mature_fspd_30,
case when obs_min_inst_def30 >=3 then 1 else 0 end flg_mature_fstpd_30
from prj-prod-dataplatform.risk_credit_mis.loan_deliquency_data),
base as
(select distinct r.customerId,
  r.digitalLoanAccountId,
  loanmaster.loanAccountNumber,
  r.modelDisplayName,
  r.sil_beta_app_score,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month, 
  'Train' Data_selection,  
    del.deffspd30,
  del.flg_mature_fspd_30,
  loanmaster.new_loan_type,
  modelVersionId,
    trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile' 
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall' 
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
    from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  inner join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
  left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
  where loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and r.sil_beta_app_score is not null
  and del.flg_mature_fspd_30 = 1
  )
  select * from base;
  """
dfd = client.query(sq).to_dataframe()
dfd = dfd.drop_duplicates(keep="first")
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()

# %%
df2 = dfd.copy()

# %% [markdown]
# ### Concat

# %%
# df_concat = pd.concat([df1, df2], ignore_index=True)
# print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")

# df_concat = (df_concat
#              .sort_values(by=['digitalLoanAccountId', 'Data_selection'],
#                           key=lambda s: s.map({'Train': 0, 'Test': 1}))
#              .drop_duplicates(subset=['digitalLoanAccountId'], keep='first')
#              .reset_index(drop=True))
# print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")
# df_concat.head()


# 1) Get all IDs present in Train
train_ids = set(df2["digitalLoanAccountId"])

# 2) Keep only Test rows whose ID is NOT in Train
df1_no_dupes = df1[~df1["digitalLoanAccountId"].isin(train_ids)]

# 3) Concatenate
df_concat = pd.concat([df1_no_dupes, df2], ignore_index=True)

print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")
df_concat.head()


# %%
df_concat["sil_beta_app_score"] = pd.to_numeric(
    df_concat["sil_beta_app_score"], errors="coerce"
)

# %%
fact_table, dimension_table = calculate_periodic_gini_prod_ver_trench_dimfact(
    df_concat,
    "sil_beta_app_score",
    "deffspd30",
    "FSPD30",
    model_version_column="modelVersionId",
    trench_column="trenchCategory",
    loan_type_column="new_loan_type",
    loan_product_type_column="loan_product_type",
    account_id_column="digitalLoanAccountId",
)

# %%
fact_table, dimension_table = update_tables(
    fact_table, dimension_table, model_name="apps_score_model_sil", product="SIL"
)

# %%
# Upload to BigQuery
table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_table"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(fact_table, table_id, job_config=job_config)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_table"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(dimension_table, table_id, job_config=job_config)
job.result()  # Wait for the job to complete

# %%


# %% [markdown]
# ### FSTPD30

# %% [markdown]
# ### Test

# %%
sq = """ 
WITH cleaned AS (
  SELECT
    customerId,digitalLoanAccountId,prediction,start_time,end_time,modelDisplayName,modelVersionId,
    case when trenchCategory is null then 'ALL'
 when trenchCategory = '' then 'ALL'
 else trenchCategory end as trenchCategory,

    REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature,
    REPLACE(REPLACE(prediction, "'", '"'), "None", "null") AS prediction_clean
  FROM `prj-prod-dataplatform.audit_balance.ml_model_run_details`
  WHERE modelDisplayName in ('Beta - AppsScoreModel', 'apps_score_model_sil')
   ), 
  modelname as (
  select customerId,
  digitalLoanAccountId,
  start_time,
  safe_cast(JSON_VALUE(prediction_clean, "$.combined_score") AS float64) as sil_beta_app_score,
  case when modelDisplayName = 'Beta - AppsScoreModel' then 'apps_score_model_sil'
       else modelDisplayName end as modelDisplayName,
  modelVersionId,
  trenchCategory,
  from cleaned
  ),
  deliquency as
(select loanAccountNumber,
case when obs_min_inst_def0 >= 1 and min_inst_def0 = 1 then 1 else 0 end deffpd0,
case when obs_min_inst_def10 >=1 and min_inst_def10 =1 then 1 else 0 end deffpd10,
case when obs_min_inst_def30 >=1 and min_inst_def30 =1 then 1 else 0 end deffpd30,
case when obs_min_inst_def30 >=2 and min_inst_def30 in (1,2) then 1 else 0 end deffspd30,
case when obs_min_inst_def30 >=3 and min_inst_def30 in (1,2,3) then 1 else 0 end deffstpd30,
case when obs_min_inst_def0 >= 1 then 1 else 0 end flg_mature_fpd0,
case when obs_min_inst_def10 >=1 then 1 else 0 end flg_mature_fpd10,
case when obs_min_inst_def30 >=1 then 1 else 0 end flg_mature_fpd30,
case when obs_min_inst_def30 >=2 then 1 else 0 end flg_mature_fspd_30,
case when obs_min_inst_def30 >=3 then 1 else 0 end flg_mature_fstpd_30
from prj-prod-dataplatform.risk_credit_mis.loan_deliquency_data),
base as 
  (select distinct r.customerId,
  r.digitalLoanAccountId,
  loanmaster.loanAccountNumber,
  r.modelDisplayName,
  r.sil_beta_app_score,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month, 
  'Test' Data_selection,  
  del.deffstpd30,
  del.flg_mature_fstpd_30,
  loanmaster.new_loan_type,
  modelVersionId, trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile' 
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall' 
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
  from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  inner join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
   left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
  where loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and r.sil_beta_app_score is not null
  and del.flg_mature_fstpd_30 = 1
  )
  select *  from base
  ;
"""
dfd = client.query(sq).to_dataframe()
dfd = dfd.drop_duplicates(keep="first")
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()


# %%
df1 = dfd.copy()

# %% [markdown]
# ### Train

# %%
sq = """ 
WITH cleaned AS (
  SELECT
    customerId,digitalLoanAccountId,prediction,start_time,end_time,modelDisplayName,modelVersionId,
        case when trenchCategory is null then 'ALL'
 when trenchCategory = '' then 'ALL'
 else trenchCategory end as trenchCategory,
    REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature,
    REPLACE(REPLACE(cast(prediction as string), "'", '"'), "None", "null") AS prediction_clean
  FROM prj-prod-dataplatform.dap_ds_poweruser_playground.ml_training_model_run_details
  WHERE modelDisplayName in ('Beta - AppsScoreModel', 'apps_score_model_sil')
    ),
  modelname as (
  select customerId,
  digitalLoanAccountId,
  start_time,
  coalesce(prediction, safe_cast(JSON_VALUE(prediction_clean, "$.combined_score") AS float64)) as sil_beta_app_score, 
  case when modelDisplayName = 'Beta - AppsScoreModel' then 'apps_score_model_sil'
       else modelDisplayName end as modelDisplayName,
  modelVersionId,
  trenchCategory
  from cleaned
  ),
  deliquency as
(select loanAccountNumber,
case when obs_min_inst_def0 >= 1 and min_inst_def0 = 1 then 1 else 0 end deffpd0,
case when obs_min_inst_def10 >=1 and min_inst_def10 =1 then 1 else 0 end deffpd10,
case when obs_min_inst_def30 >=1 and min_inst_def30 =1 then 1 else 0 end deffpd30,
case when obs_min_inst_def30 >=2 and min_inst_def30 in (1,2) then 1 else 0 end deffspd30,
case when obs_min_inst_def30 >=3 and min_inst_def30 in (1,2,3) then 1 else 0 end deffstpd30,
case when obs_min_inst_def0 >= 1 then 1 else 0 end flg_mature_fpd0,
case when obs_min_inst_def10 >=1 then 1 else 0 end flg_mature_fpd10,
case when obs_min_inst_def30 >=1 then 1 else 0 end flg_mature_fpd30,
case when obs_min_inst_def30 >=2 then 1 else 0 end flg_mature_fspd_30,
case when obs_min_inst_def30 >=3 then 1 else 0 end flg_mature_fstpd_30
from prj-prod-dataplatform.risk_credit_mis.loan_deliquency_data),
base as
(select distinct r.customerId,
  r.digitalLoanAccountId,
  loanmaster.loanAccountNumber,
  r.modelDisplayName,
  r.sil_beta_app_score,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month, 
  'Train' Data_selection,  
    del.deffstpd30,
  del.flg_mature_fstpd_30,
  loanmaster.new_loan_type,
  modelVersionId,
    trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile' 
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall' 
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
    from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  inner join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
  left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
  where loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and r.sil_beta_app_score is not null
  and del.flg_mature_fstpd_30 = 1
  )
  select * from base;
  """
dfd = client.query(sq).to_dataframe()
dfd = dfd.drop_duplicates(keep="first")
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()

# %%
df2 = dfd.copy()

# %% [markdown]
# ### Concat

# %%
# df_concat = pd.concat([df1, df2], ignore_index=True)
# print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")

# df_concat = (df_concat
#              .sort_values(by=['digitalLoanAccountId', 'Data_selection'],
#                           key=lambda s: s.map({'Train': 0, 'Test': 1}))
#              .drop_duplicates(subset=['digitalLoanAccountId'], keep='first')
#              .reset_index(drop=True))
# print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")
# df_concat.head()


# 1) Get all IDs present in Train
train_ids = set(df2["digitalLoanAccountId"])

# 2) Keep only Test rows whose ID is NOT in Train
df1_no_dupes = df1[~df1["digitalLoanAccountId"].isin(train_ids)]

# 3) Concatenate
df_concat = pd.concat([df1_no_dupes, df2], ignore_index=True)

print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")
df_concat.head()


# %%
df_concat["sil_beta_app_score"] = pd.to_numeric(
    df_concat["sil_beta_app_score"], errors="coerce"
)

# %%
fact_table, dimension_table = calculate_periodic_gini_prod_ver_trench_dimfact(
    df_concat,
    "sil_beta_app_score",
    "deffstpd30",
    "FSTPD30",
    model_version_column="modelVersionId",
    trench_column="trenchCategory",
    loan_type_column="new_loan_type",
    loan_product_type_column="loan_product_type",
    account_id_column="digitalLoanAccountId",
)

# %%
fact_table, dimension_table = update_tables(
    fact_table, dimension_table, model_name="apps_score_model_sil", product="SIL"
)
print(f"The shape of the fact table is:\t {fact_table.shape}")
print(f"The shape of the dimension table is:\t {dimension_table.shape}")

# %%
# Upload to BigQuery
table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_table"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(fact_table, table_id, job_config=job_config)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_table"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(dimension_table, table_id, job_config=job_config)
job.result()  # Wait for the job to complete

# %%


# %% [markdown]
# # Beta SIL Demo Score

# %%


# %% [markdown]
# ### FPD0

# %% [markdown]
# ### Test

# %%
sq = """ 
WITH cleaned AS (
  SELECT
  customerId,digitalLoanAccountId,prediction,start_time,end_time,modelDisplayName,modelVersionId,
  case when trenchCategory is null then 'ALL'
 when trenchCategory = '' then 'ALL'
 else trenchCategory end as trenchCategory,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature_cleaned
  FROM `prj-prod-dataplatform.audit_balance.ml_model_run_details`
  WHERE modelDisplayName in ('Beta - DemoScoreModel', 'beta_demo_model_sil')
  ), 
  modelname as (
  select customerId,
  digitalLoanAccountId,
  start_time,
  prediction sil_beta_demo_score,
  modelDisplayName,
  modelVersionId, trenchCategory,
  from cleaned
  ),
  deliquency as
(select loanAccountNumber,
case when obs_min_inst_def0 >= 1 and min_inst_def0 = 1 then 1 else 0 end deffpd0,
case when obs_min_inst_def10 >=1 and min_inst_def10 =1 then 1 else 0 end deffpd10,
case when obs_min_inst_def30 >=1 and min_inst_def30 =1 then 1 else 0 end deffpd30,
case when obs_min_inst_def30 >=2 and min_inst_def30 in (1,2) then 1 else 0 end deffspd30,
case when obs_min_inst_def30 >=3 and min_inst_def30 in (1,2,3) then 1 else 0 end deffstpd30,
case when obs_min_inst_def0 >= 1 then 1 else 0 end flg_mature_fpd0,
case when obs_min_inst_def10 >=1 then 1 else 0 end flg_mature_fpd10,
case when obs_min_inst_def30 >=1 then 1 else 0 end flg_mature_fpd30,
case when obs_min_inst_def30 >=2 then 1 else 0 end flg_mature_fspd_30,
case when obs_min_inst_def30 >=3 then 1 else 0 end flg_mature_fstpd_30
from prj-prod-dataplatform.risk_credit_mis.loan_deliquency_data),
base as 
  (select distinct r.customerId,
  r.digitalLoanAccountId,
  loanmaster.loanAccountNumber,
  r.modelDisplayName,
  r.sil_beta_demo_score,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month, 
  'Test' Data_selection,  
  deffpd0,
  flg_mature_fpd0,
  loanmaster.new_loan_type,
  modelVersionId, trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile' 
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall' 
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
  from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  inner join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
   left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
  where loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and r.sil_beta_demo_score is not null
  and flg_mature_fpd0 = 1
  )
  select *  from base
  ;
"""
dfd = client.query(sq).to_dataframe()
dfd = dfd.drop_duplicates(keep="first")
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()


# %%
df1 = dfd.copy()

# %% [markdown]
# ### Train

# %%
sq = """ 
WITH cleaned AS (
  SELECT
  customerId,digitalLoanAccountId,prediction,start_time,end_time,modelDisplayName,modelVersionId,
  case when trenchCategory is null then 'ALL'
 when trenchCategory = '' then 'ALL'
 else trenchCategory end as trenchCategory,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature_cleaned
  FROM prj-prod-dataplatform.dap_ds_poweruser_playground.ml_training_model_run_details
  WHERE modelDisplayName in ('Beta - DemoScoreModel', 'beta_demo_model_sil')
  ), 
  modelname as (
  select customerId,
  digitalLoanAccountId,
  start_time,
  prediction sil_beta_demo_score,
  case when modelDisplayName = 'Beta - DemoScoreModel' then 'beta_demo_model_sil' else modelDisplayName end as modelDisplayName,
  modelVersionId, trenchCategory,
  from cleaned
  ),
  deliquency as
(select loanAccountNumber,
case when obs_min_inst_def0 >= 1 and min_inst_def0 = 1 then 1 else 0 end deffpd0,
case when obs_min_inst_def10 >=1 and min_inst_def10 =1 then 1 else 0 end deffpd10,
case when obs_min_inst_def30 >=1 and min_inst_def30 =1 then 1 else 0 end deffpd30,
case when obs_min_inst_def30 >=2 and min_inst_def30 in (1,2) then 1 else 0 end deffspd30,
case when obs_min_inst_def30 >=3 and min_inst_def30 in (1,2,3) then 1 else 0 end deffstpd30,
case when obs_min_inst_def0 >= 1 then 1 else 0 end flg_mature_fpd0,
case when obs_min_inst_def10 >=1 then 1 else 0 end flg_mature_fpd10,
case when obs_min_inst_def30 >=1 then 1 else 0 end flg_mature_fpd30,
case when obs_min_inst_def30 >=2 then 1 else 0 end flg_mature_fspd_30,
case when obs_min_inst_def30 >=3 then 1 else 0 end flg_mature_fstpd_30
from prj-prod-dataplatform.risk_credit_mis.loan_deliquency_data),
base as
(select distinct r.customerId,
  r.digitalLoanAccountId,
  loanmaster.loanAccountNumber,
  r.modelDisplayName,
  r.sil_beta_demo_score,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month, 
  'Train' Data_selection,  
    deffpd0,
  flg_mature_fpd0,
  loanmaster.new_loan_type,
  modelVersionId,
    trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile' 
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall' 
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
    from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  inner join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
  left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
  where loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and r.sil_beta_demo_score is not null
  and flg_mature_fpd0 = 1
  )
  select * from base;
  """
dfd = client.query(sq).to_dataframe()
dfd = dfd.drop_duplicates(keep="first")
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()

# %%
df2 = dfd.copy()

# %% [markdown]
# ### Concat

# %%
# df_concat = pd.concat([df1, df2], ignore_index=True)
# print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")

# df_concat = (df_concat
#              .sort_values(by=['digitalLoanAccountId', 'Data_selection'],
#                           key=lambda s: s.map({'Train': 0, 'Test': 1}))
#              .drop_duplicates(subset=['digitalLoanAccountId'], keep='first')
#              .reset_index(drop=True))
# print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")
# df_concat.head()


# 1) Get all IDs present in Train
train_ids = set(df2["digitalLoanAccountId"])

# 2) Keep only Test rows whose ID is NOT in Train
df1_no_dupes = df1[~df1["digitalLoanAccountId"].isin(train_ids)]

# 3) Concatenate
df_concat = pd.concat([df1_no_dupes, df2], ignore_index=True)

print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")
df_concat.head()


# %%
df_concat["sil_beta_demo_score"] = pd.to_numeric(
    df_concat["sil_beta_demo_score"], errors="coerce"
)

# %%
fact_table, dimension_table = calculate_periodic_gini_prod_ver_trench_dimfact(
    df_concat,
    "sil_beta_demo_score",
    "deffpd0",
    "FPD0",
    model_version_column="modelVersionId",
    trench_column="trenchCategory",
    loan_type_column="new_loan_type",
    loan_product_type_column="loan_product_type",
    account_id_column="digitalLoanAccountId",
)

# %%
fact_table, dimension_table = update_tables(
    fact_table, dimension_table, model_name="beta_demo_model_sil", product="SIL"
)
print(f"The shape of the fact table is:\t {fact_table.shape}")
print(f"The shape of the dimension table is:\t {dimension_table.shape}")

# %%
# Upload to BigQuery
table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_table"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(fact_table, table_id, job_config=job_config)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_table"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(dimension_table, table_id, job_config=job_config)
job.result()  # Wait for the job to complete

# %% [markdown]
# ### FPD10

# %% [markdown]
# ### Test

# %%
sq = """ 
WITH cleaned AS (
  SELECT
  customerId,digitalLoanAccountId,prediction,start_time,end_time,modelDisplayName,modelVersionId,
  case when trenchCategory is null then 'ALL'
 when trenchCategory = '' then 'ALL'
 else trenchCategory end as trenchCategory,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature_cleaned
  FROM `prj-prod-dataplatform.audit_balance.ml_model_run_details`
  WHERE modelDisplayName in ('Beta - DemoScoreModel', 'beta_demo_model_sil')
  ), 
  modelname as (
  select customerId,
  digitalLoanAccountId,
  start_time,
  prediction sil_beta_demo_score,
  modelDisplayName,
  modelVersionId, trenchCategory,
  from cleaned
  ),
  deliquency as
(select loanAccountNumber,
case when obs_min_inst_def0 >= 1 and min_inst_def0 = 1 then 1 else 0 end deffpd0,
case when obs_min_inst_def10 >=1 and min_inst_def10 =1 then 1 else 0 end deffpd10,
case when obs_min_inst_def30 >=1 and min_inst_def30 =1 then 1 else 0 end deffpd30,
case when obs_min_inst_def30 >=2 and min_inst_def30 in (1,2) then 1 else 0 end deffspd30,
case when obs_min_inst_def30 >=3 and min_inst_def30 in (1,2,3) then 1 else 0 end deffstpd30,
case when obs_min_inst_def0 >= 1 then 1 else 0 end flg_mature_fpd0,
case when obs_min_inst_def10 >=1 then 1 else 0 end flg_mature_fpd10,
case when obs_min_inst_def30 >=1 then 1 else 0 end flg_mature_fpd30,
case when obs_min_inst_def30 >=2 then 1 else 0 end flg_mature_fspd_30,
case when obs_min_inst_def30 >=3 then 1 else 0 end flg_mature_fstpd_30
from prj-prod-dataplatform.risk_credit_mis.loan_deliquency_data),
base as 
  (select distinct r.customerId,
  r.digitalLoanAccountId,
  loanmaster.loanAccountNumber,
  r.modelDisplayName,
  r.sil_beta_demo_score,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month, 
  'Test' Data_selection,  
  del.deffpd10,
  del.flg_mature_fpd10,
  loanmaster.new_loan_type,
  modelVersionId, trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile' 
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall' 
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
  from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  inner join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
   left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
  where loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and r.sil_beta_demo_score is not null
  and del.flg_mature_fpd10 = 1
  )
  select *  from base
  ;
"""
dfd = client.query(sq).to_dataframe()
dfd = dfd.drop_duplicates(keep="first")
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()


# %%
df1 = dfd.copy()

# %% [markdown]
# ### Train

# %%
sq = """ 
WITH cleaned AS (
  SELECT
  customerId,digitalLoanAccountId,prediction,start_time,end_time,modelDisplayName,modelVersionId,
  case when trenchCategory is null then 'ALL'
 when trenchCategory = '' then 'ALL'
 else trenchCategory end as trenchCategory,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature_cleaned
  FROM prj-prod-dataplatform.dap_ds_poweruser_playground.ml_training_model_run_details
  WHERE modelDisplayName in ('Beta - DemoScoreModel', 'beta_demo_model_sil')
  ), 
  modelname as (
  select customerId,
  digitalLoanAccountId,
  start_time,
  prediction sil_beta_demo_score,
  case when modelDisplayName = 'Beta - DemoScoreModel' then 'beta_demo_model_sil' else modelDisplayName end as modelDisplayName,
  modelVersionId, trenchCategory,
  from cleaned
  ),
  deliquency as
(select loanAccountNumber,
case when obs_min_inst_def0 >= 1 and min_inst_def0 = 1 then 1 else 0 end deffpd0,
case when obs_min_inst_def10 >=1 and min_inst_def10 =1 then 1 else 0 end deffpd10,
case when obs_min_inst_def30 >=1 and min_inst_def30 =1 then 1 else 0 end deffpd30,
case when obs_min_inst_def30 >=2 and min_inst_def30 in (1,2) then 1 else 0 end deffspd30,
case when obs_min_inst_def30 >=3 and min_inst_def30 in (1,2,3) then 1 else 0 end deffstpd30,
case when obs_min_inst_def0 >= 1 then 1 else 0 end flg_mature_fpd0,
case when obs_min_inst_def10 >=1 then 1 else 0 end flg_mature_fpd10,
case when obs_min_inst_def30 >=1 then 1 else 0 end flg_mature_fpd30,
case when obs_min_inst_def30 >=2 then 1 else 0 end flg_mature_fspd_30,
case when obs_min_inst_def30 >=3 then 1 else 0 end flg_mature_fstpd_30
from prj-prod-dataplatform.risk_credit_mis.loan_deliquency_data),
base as
(select distinct r.customerId,
  r.digitalLoanAccountId,
  loanmaster.loanAccountNumber,
  r.modelDisplayName,
  r.sil_beta_demo_score,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month, 
  'Train' Data_selection,  
    del.deffpd10,
  del.flg_mature_fpd10,
  loanmaster.new_loan_type,
  modelVersionId,
    trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile' 
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall' 
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
    from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  inner join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
  left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
  where loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and r.sil_beta_demo_score is not null
  and del.flg_mature_fpd10 = 1
  )
  select * from base;
  """
dfd = client.query(sq).to_dataframe()
dfd = dfd.drop_duplicates(keep="first")
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()

# %%
df2 = dfd.copy()

# %% [markdown]
# ### Concat

# %%
# df_concat = pd.concat([df1, df2], ignore_index=True)
# print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")

# df_concat = (df_concat
#              .sort_values(by=['digitalLoanAccountId', 'Data_selection'],
#                           key=lambda s: s.map({'Train': 0, 'Test': 1}))
#              .drop_duplicates(subset=['digitalLoanAccountId'], keep='first')
#              .reset_index(drop=True))
# print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")
# df_concat.head()


# 1) Get all IDs present in Train
train_ids = set(df2["digitalLoanAccountId"])

# 2) Keep only Test rows whose ID is NOT in Train
df1_no_dupes = df1[~df1["digitalLoanAccountId"].isin(train_ids)]

# 3) Concatenate
df_concat = pd.concat([df1_no_dupes, df2], ignore_index=True)

print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")
df_concat.head()


# %%
df_concat["sil_beta_demo_score"] = pd.to_numeric(
    df_concat["sil_beta_demo_score"], errors="coerce"
)

# %%
fact_table, dimension_table = calculate_periodic_gini_prod_ver_trench_dimfact(
    df_concat,
    "sil_beta_demo_score",
    "deffpd10",
    "FPD10",
    model_version_column="modelVersionId",
    trench_column="trenchCategory",
    loan_type_column="new_loan_type",
    loan_product_type_column="loan_product_type",
    account_id_column="digitalLoanAccountId",
)

# %%
fact_table, dimension_table = update_tables(
    fact_table, dimension_table, model_name="beta_demo_model_sil", product="SIL"
)

# %%
# Upload to BigQuery
table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_table"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(fact_table, table_id, job_config=job_config)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_table"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(dimension_table, table_id, job_config=job_config)
job.result()  # Wait for the job to complete

# %%


# %% [markdown]
# ### FPD30

# %% [markdown]
# ### Test

# %%
sq = """ 
WITH cleaned AS (
  SELECT
  customerId,digitalLoanAccountId,prediction,start_time,end_time,modelDisplayName,modelVersionId,
  case when trenchCategory is null then 'ALL'
 when trenchCategory = '' then 'ALL'
 else trenchCategory end as trenchCategory,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature_cleaned
  FROM `prj-prod-dataplatform.audit_balance.ml_model_run_details`
  WHERE modelDisplayName in ('Beta - DemoScoreModel', 'beta_demo_model_sil')
  ), 
  modelname as (
  select customerId,
  digitalLoanAccountId,
  start_time,
  prediction sil_beta_demo_score,
  modelDisplayName,
  modelVersionId, trenchCategory,
  from cleaned
  ),
  deliquency as
(select loanAccountNumber,
case when obs_min_inst_def0 >= 1 and min_inst_def0 = 1 then 1 else 0 end deffpd0,
case when obs_min_inst_def10 >=1 and min_inst_def10 =1 then 1 else 0 end deffpd10,
case when obs_min_inst_def30 >=1 and min_inst_def30 =1 then 1 else 0 end deffpd30,
case when obs_min_inst_def30 >=2 and min_inst_def30 in (1,2) then 1 else 0 end deffspd30,
case when obs_min_inst_def30 >=3 and min_inst_def30 in (1,2,3) then 1 else 0 end deffstpd30,
case when obs_min_inst_def0 >= 1 then 1 else 0 end flg_mature_fpd0,
case when obs_min_inst_def10 >=1 then 1 else 0 end flg_mature_fpd10,
case when obs_min_inst_def30 >=1 then 1 else 0 end flg_mature_fpd30,
case when obs_min_inst_def30 >=2 then 1 else 0 end flg_mature_fspd_30,
case when obs_min_inst_def30 >=3 then 1 else 0 end flg_mature_fstpd_30
from prj-prod-dataplatform.risk_credit_mis.loan_deliquency_data),
base as 
  (select distinct r.customerId,
  r.digitalLoanAccountId,
  loanmaster.loanAccountNumber,
  r.modelDisplayName,
  r.sil_beta_demo_score,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month, 
  'Test' Data_selection,  
  del.deffpd30,
  del.flg_mature_fpd30,
  loanmaster.new_loan_type,
  modelVersionId, trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile' 
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall' 
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
  from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  inner join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
   left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
  where loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and r.sil_beta_demo_score is not null
  and del.flg_mature_fpd30 = 1
  )
  select *  from base
  ;
"""
dfd = client.query(sq).to_dataframe()
dfd = dfd.drop_duplicates(keep="first")
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()


# %%
df1 = dfd.copy()

# %% [markdown]
# ### Train

# %%
sq = """ 
WITH cleaned AS (
  SELECT
  customerId,digitalLoanAccountId,prediction,start_time,end_time,modelDisplayName,modelVersionId,
  case when trenchCategory is null then 'ALL'
 when trenchCategory = '' then 'ALL'
 else trenchCategory end as trenchCategory,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature_cleaned
  FROM prj-prod-dataplatform.dap_ds_poweruser_playground.ml_training_model_run_details
  WHERE modelDisplayName in ('Beta - DemoScoreModel', 'beta_demo_model_sil')
  ), 
  modelname as (
  select customerId,
  digitalLoanAccountId,
  start_time,
  prediction sil_beta_demo_score,
  case when modelDisplayName = 'Beta - DemoScoreModel' then 'beta_demo_model_sil' else modelDisplayName end as modelDisplayName,
  modelVersionId, trenchCategory,
  from cleaned
  ),
  deliquency as
(select loanAccountNumber,
case when obs_min_inst_def0 >= 1 and min_inst_def0 = 1 then 1 else 0 end deffpd0,
case when obs_min_inst_def10 >=1 and min_inst_def10 =1 then 1 else 0 end deffpd10,
case when obs_min_inst_def30 >=1 and min_inst_def30 =1 then 1 else 0 end deffpd30,
case when obs_min_inst_def30 >=2 and min_inst_def30 in (1,2) then 1 else 0 end deffspd30,
case when obs_min_inst_def30 >=3 and min_inst_def30 in (1,2,3) then 1 else 0 end deffstpd30,
case when obs_min_inst_def0 >= 1 then 1 else 0 end flg_mature_fpd0,
case when obs_min_inst_def10 >=1 then 1 else 0 end flg_mature_fpd10,
case when obs_min_inst_def30 >=1 then 1 else 0 end flg_mature_fpd30,
case when obs_min_inst_def30 >=2 then 1 else 0 end flg_mature_fspd_30,
case when obs_min_inst_def30 >=3 then 1 else 0 end flg_mature_fstpd_30
from prj-prod-dataplatform.risk_credit_mis.loan_deliquency_data),
base as
(select distinct r.customerId,
  r.digitalLoanAccountId,
  loanmaster.loanAccountNumber,
  r.modelDisplayName,
  r.sil_beta_demo_score,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month, 
  'Train' Data_selection,  
    del.deffpd30,
  del.flg_mature_fpd30,
  loanmaster.new_loan_type,
  modelVersionId,
    trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile' 
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall' 
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
    from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  inner join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
  left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
  where loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and r.sil_beta_demo_score is not null
  and del.flg_mature_fpd30 = 1
  )
  select * from base;
  """
dfd = client.query(sq).to_dataframe()
dfd = dfd.drop_duplicates(keep="first")
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()

# %%
df2 = dfd.copy()

# %% [markdown]
# ### Concat

# %%
# df_concat = pd.concat([df1, df2], ignore_index=True)
# print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")

# df_concat = (df_concat
#              .sort_values(by=['digitalLoanAccountId', 'Data_selection'],
#                           key=lambda s: s.map({'Train': 0, 'Test': 1}))
#              .drop_duplicates(subset=['digitalLoanAccountId'], keep='first')
#              .reset_index(drop=True))
# print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")
# df_concat.head()


# 1) Get all IDs present in Train
train_ids = set(df2["digitalLoanAccountId"])

# 2) Keep only Test rows whose ID is NOT in Train
df1_no_dupes = df1[~df1["digitalLoanAccountId"].isin(train_ids)]

# 3) Concatenate
df_concat = pd.concat([df1_no_dupes, df2], ignore_index=True)

print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")
df_concat.head()


# %%
df_concat["sil_beta_demo_score"] = pd.to_numeric(
    df_concat["sil_beta_demo_score"], errors="coerce"
)

# %%
fact_table, dimension_table = calculate_periodic_gini_prod_ver_trench_dimfact(
    df_concat,
    "sil_beta_demo_score",
    "deffpd30",
    "FPD30",
    model_version_column="modelVersionId",
    trench_column="trenchCategory",
    loan_type_column="new_loan_type",
    loan_product_type_column="loan_product_type",
    account_id_column="digitalLoanAccountId",
)

# %%
fact_table, dimension_table = update_tables(
    fact_table, dimension_table, model_name="beta_demo_model_sil", product="SIL"
)

# %%
# Upload to BigQuery
table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_table"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(fact_table, table_id, job_config=job_config)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_table"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(dimension_table, table_id, job_config=job_config)
job.result()  # Wait for the job to complete

# %%


# %% [markdown]
# ### FSPD30

# %% [markdown]
# ### Test

# %%
sq = """ 
WITH cleaned AS (
  SELECT
  customerId,digitalLoanAccountId,prediction,start_time,end_time,modelDisplayName,modelVersionId,
  case when trenchCategory is null then 'ALL'
 when trenchCategory = '' then 'ALL'
 else trenchCategory end as trenchCategory,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature_cleaned
  FROM `prj-prod-dataplatform.audit_balance.ml_model_run_details`
  WHERE modelDisplayName in ('Beta - DemoScoreModel', 'beta_demo_model_sil')
  ), 
  modelname as (
  select customerId,
  digitalLoanAccountId,
  start_time,
  prediction sil_beta_demo_score,
  modelDisplayName,
  modelVersionId, trenchCategory,
  from cleaned
  ),
  deliquency as
(select loanAccountNumber,
case when obs_min_inst_def0 >= 1 and min_inst_def0 = 1 then 1 else 0 end deffpd0,
case when obs_min_inst_def10 >=1 and min_inst_def10 =1 then 1 else 0 end deffpd10,
case when obs_min_inst_def30 >=1 and min_inst_def30 =1 then 1 else 0 end deffpd30,
case when obs_min_inst_def30 >=2 and min_inst_def30 in (1,2) then 1 else 0 end deffspd30,
case when obs_min_inst_def30 >=3 and min_inst_def30 in (1,2,3) then 1 else 0 end deffstpd30,
case when obs_min_inst_def0 >= 1 then 1 else 0 end flg_mature_fpd0,
case when obs_min_inst_def10 >=1 then 1 else 0 end flg_mature_fpd10,
case when obs_min_inst_def30 >=1 then 1 else 0 end flg_mature_fpd30,
case when obs_min_inst_def30 >=2 then 1 else 0 end flg_mature_fspd_30,
case when obs_min_inst_def30 >=3 then 1 else 0 end flg_mature_fstpd_30
from prj-prod-dataplatform.risk_credit_mis.loan_deliquency_data),
base as 
  (select distinct r.customerId,
  r.digitalLoanAccountId,
  loanmaster.loanAccountNumber,
  r.modelDisplayName,
  r.sil_beta_demo_score,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month, 
  'Test' Data_selection,  
  del.deffspd30,
  del.flg_mature_fspd_30,
  loanmaster.new_loan_type,
  modelVersionId, trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile' 
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall' 
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
  from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  inner join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
   left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
  where loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and r.sil_beta_demo_score is not null
  and del.flg_mature_fspd_30 = 1
  )
  select *  from base
  ;
"""
dfd = client.query(sq).to_dataframe()
dfd = dfd.drop_duplicates(keep="first")
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()


# %%
df1 = dfd.copy()

# %% [markdown]
# ### Train

# %%
sq = """ 
WITH cleaned AS (
  SELECT
  customerId,digitalLoanAccountId,prediction,start_time,end_time,modelDisplayName,modelVersionId,
  case when trenchCategory is null then 'ALL'
 when trenchCategory = '' then 'ALL'
 else trenchCategory end as trenchCategory,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature_cleaned
  FROM prj-prod-dataplatform.dap_ds_poweruser_playground.ml_training_model_run_details
  WHERE modelDisplayName in ('Beta - DemoScoreModel', 'beta_demo_model_sil')
  ), 
  modelname as (
  select customerId,
  digitalLoanAccountId,
  start_time,
  prediction sil_beta_demo_score,
  case when modelDisplayName = 'Beta - DemoScoreModel' then 'beta_demo_model_sil' else modelDisplayName end as modelDisplayName,
  modelVersionId, trenchCategory,
  from cleaned
  ),
  deliquency as
(select loanAccountNumber,
case when obs_min_inst_def0 >= 1 and min_inst_def0 = 1 then 1 else 0 end deffpd0,
case when obs_min_inst_def10 >=1 and min_inst_def10 =1 then 1 else 0 end deffpd10,
case when obs_min_inst_def30 >=1 and min_inst_def30 =1 then 1 else 0 end deffpd30,
case when obs_min_inst_def30 >=2 and min_inst_def30 in (1,2) then 1 else 0 end deffspd30,
case when obs_min_inst_def30 >=3 and min_inst_def30 in (1,2,3) then 1 else 0 end deffstpd30,
case when obs_min_inst_def0 >= 1 then 1 else 0 end flg_mature_fpd0,
case when obs_min_inst_def10 >=1 then 1 else 0 end flg_mature_fpd10,
case when obs_min_inst_def30 >=1 then 1 else 0 end flg_mature_fpd30,
case when obs_min_inst_def30 >=2 then 1 else 0 end flg_mature_fspd_30,
case when obs_min_inst_def30 >=3 then 1 else 0 end flg_mature_fstpd_30
from prj-prod-dataplatform.risk_credit_mis.loan_deliquency_data),
base as
(select distinct r.customerId,
  r.digitalLoanAccountId,
  loanmaster.loanAccountNumber,
  r.modelDisplayName,
  r.sil_beta_demo_score,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month, 
  'Train' Data_selection,  
    del.deffspd30,
  del.flg_mature_fspd_30,
  loanmaster.new_loan_type,
  modelVersionId,
    trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile' 
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall' 
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
    from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  inner join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
  left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
  where loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and r.sil_beta_demo_score is not null
  and del.flg_mature_fspd_30 = 1
  )
  select * from base;
  """
dfd = client.query(sq).to_dataframe()
dfd = dfd.drop_duplicates(keep="first")
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()

# %%
df2 = dfd.copy()

# %% [markdown]
# ### Concat

# %%
# df_concat = pd.concat([df1, df2], ignore_index=True)
# print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")

# df_concat = (df_concat
#              .sort_values(by=['digitalLoanAccountId', 'Data_selection'],
#                           key=lambda s: s.map({'Train': 0, 'Test': 1}))
#              .drop_duplicates(subset=['digitalLoanAccountId'], keep='first')
#              .reset_index(drop=True))
# print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")
# df_concat.head()

# 1) Get all IDs present in Train
train_ids = set(df2["digitalLoanAccountId"])

# 2) Keep only Test rows whose ID is NOT in Train
df1_no_dupes = df1[~df1["digitalLoanAccountId"].isin(train_ids)]

# 3) Concatenate
df_concat = pd.concat([df1_no_dupes, df2], ignore_index=True)

print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")
df_concat.head()


# %%
df_concat["sil_beta_demo_score"] = pd.to_numeric(
    df_concat["sil_beta_demo_score"], errors="coerce"
)

# %%
fact_table, dimension_table = calculate_periodic_gini_prod_ver_trench_dimfact(
    df_concat,
    "sil_beta_demo_score",
    "deffspd30",
    "FSPD30",
    model_version_column="modelVersionId",
    trench_column="trenchCategory",
    loan_type_column="new_loan_type",
    loan_product_type_column="loan_product_type",
    account_id_column="digitalLoanAccountId",
)

# %%
fact_table, dimension_table = update_tables(
    fact_table, dimension_table, model_name="beta_demo_model_sil", product="SIL"
)

# %%
# Upload to BigQuery
table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_table"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(fact_table, table_id, job_config=job_config)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_table"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(dimension_table, table_id, job_config=job_config)
job.result()  # Wait for the job to complete

# %%


# %% [markdown]
# ### FSTPD30

# %% [markdown]
# ### Test

# %%
sq = """ 
WITH cleaned AS (
  SELECT
  customerId,digitalLoanAccountId,prediction,start_time,end_time,modelDisplayName,modelVersionId,
  case when trenchCategory is null then 'ALL'
 when trenchCategory = '' then 'ALL'
 else trenchCategory end as trenchCategory,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature_cleaned
  FROM `prj-prod-dataplatform.audit_balance.ml_model_run_details`
  WHERE modelDisplayName in ('Beta - DemoScoreModel', 'beta_demo_model_sil')
  ), 
  modelname as (
  select customerId,
  digitalLoanAccountId,
  start_time,
  prediction sil_beta_demo_score,
  modelDisplayName,
  modelVersionId, trenchCategory,
  from cleaned
  ),
  deliquency as
(select loanAccountNumber,
case when obs_min_inst_def0 >= 1 and min_inst_def0 = 1 then 1 else 0 end deffpd0,
case when obs_min_inst_def10 >=1 and min_inst_def10 =1 then 1 else 0 end deffpd10,
case when obs_min_inst_def30 >=1 and min_inst_def30 =1 then 1 else 0 end deffpd30,
case when obs_min_inst_def30 >=2 and min_inst_def30 in (1,2) then 1 else 0 end deffspd30,
case when obs_min_inst_def30 >=3 and min_inst_def30 in (1,2,3) then 1 else 0 end deffstpd30,
case when obs_min_inst_def0 >= 1 then 1 else 0 end flg_mature_fpd0,
case when obs_min_inst_def10 >=1 then 1 else 0 end flg_mature_fpd10,
case when obs_min_inst_def30 >=1 then 1 else 0 end flg_mature_fpd30,
case when obs_min_inst_def30 >=2 then 1 else 0 end flg_mature_fspd_30,
case when obs_min_inst_def30 >=3 then 1 else 0 end flg_mature_fstpd_30
from prj-prod-dataplatform.risk_credit_mis.loan_deliquency_data),
base as 
  (select distinct r.customerId,
  r.digitalLoanAccountId,
  loanmaster.loanAccountNumber,
  r.modelDisplayName,
  r.sil_beta_demo_score,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month, 
  'Test' Data_selection,  
  del.deffstpd30,
  del.flg_mature_fstpd_30,
  loanmaster.new_loan_type,
  modelVersionId, trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile' 
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall' 
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
  from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  inner join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
   left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
  where loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and r.sil_beta_demo_score is not null
  and del.flg_mature_fstpd_30 = 1
  )
  select *  from base
  ;
"""
dfd = client.query(sq).to_dataframe()
dfd = dfd.drop_duplicates(keep="first")
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()


# %%
df1 = dfd.copy()

# %% [markdown]
# ### Train

# %%
sq = """ 
WITH cleaned AS (
  SELECT
  customerId,digitalLoanAccountId,prediction,start_time,end_time,modelDisplayName,modelVersionId,
  case when trenchCategory is null then 'ALL'
 when trenchCategory = '' then 'ALL'
 else trenchCategory end as trenchCategory,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature_cleaned
  FROM prj-prod-dataplatform.dap_ds_poweruser_playground.ml_training_model_run_details
  WHERE modelDisplayName in ('Beta - DemoScoreModel', 'beta_demo_model_sil')
  ), 
  modelname as (
  select customerId,
  digitalLoanAccountId,
  start_time,
  prediction sil_beta_demo_score,
  case when modelDisplayName = 'Beta - DemoScoreModel' then 'beta_demo_model_sil' else modelDisplayName end as modelDisplayName,
  modelVersionId, trenchCategory,
  from cleaned
  ),
  deliquency as
(select loanAccountNumber,
case when obs_min_inst_def0 >= 1 and min_inst_def0 = 1 then 1 else 0 end deffpd0,
case when obs_min_inst_def10 >=1 and min_inst_def10 =1 then 1 else 0 end deffpd10,
case when obs_min_inst_def30 >=1 and min_inst_def30 =1 then 1 else 0 end deffpd30,
case when obs_min_inst_def30 >=2 and min_inst_def30 in (1,2) then 1 else 0 end deffspd30,
case when obs_min_inst_def30 >=3 and min_inst_def30 in (1,2,3) then 1 else 0 end deffstpd30,
case when obs_min_inst_def0 >= 1 then 1 else 0 end flg_mature_fpd0,
case when obs_min_inst_def10 >=1 then 1 else 0 end flg_mature_fpd10,
case when obs_min_inst_def30 >=1 then 1 else 0 end flg_mature_fpd30,
case when obs_min_inst_def30 >=2 then 1 else 0 end flg_mature_fspd_30,
case when obs_min_inst_def30 >=3 then 1 else 0 end flg_mature_fstpd_30
from prj-prod-dataplatform.risk_credit_mis.loan_deliquency_data),
base as
(select distinct r.customerId,
  r.digitalLoanAccountId,
  loanmaster.loanAccountNumber,
  r.modelDisplayName,
  r.sil_beta_demo_score,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month, 
  'Train' Data_selection,  
    del.deffstpd30,
  del.flg_mature_fstpd_30,
  loanmaster.new_loan_type,
  modelVersionId,
    trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile' 
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall' 
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
    from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  inner join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
  left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
  where loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and r.sil_beta_demo_score is not null
  and del.flg_mature_fstpd_30 = 1
  )
  select * from base;
  """
dfd = client.query(sq).to_dataframe()
dfd = dfd.drop_duplicates(keep="first")
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()

# %%
df2 = dfd.copy()

# %% [markdown]
# ### Concat

# %%
# df_concat = pd.concat([df1, df2], ignore_index=True)
# print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")

# df_concat = (df_concat
#              .sort_values(by=['digitalLoanAccountId', 'Data_selection'],
#                           key=lambda s: s.map({'Train': 0, 'Test': 1}))
#              .drop_duplicates(subset=['digitalLoanAccountId'], keep='first')
#              .reset_index(drop=True))
# print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")
# df_concat.head()

# 1) Get all IDs present in Train
train_ids = set(df2["digitalLoanAccountId"])

# 2) Keep only Test rows whose ID is NOT in Train
df1_no_dupes = df1[~df1["digitalLoanAccountId"].isin(train_ids)]

# 3) Concatenate
df_concat = pd.concat([df1_no_dupes, df2], ignore_index=True)

print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")
df_concat.head()


# %%
df_concat["sil_beta_demo_score"] = pd.to_numeric(
    df_concat["sil_beta_demo_score"], errors="coerce"
)

# %%
fact_table, dimension_table = calculate_periodic_gini_prod_ver_trench_dimfact(
    df_concat,
    "sil_beta_demo_score",
    "deffstpd30",
    "FSTPD30",
    model_version_column="modelVersionId",
    trench_column="trenchCategory",
    loan_type_column="new_loan_type",
    loan_product_type_column="loan_product_type",
    account_id_column="digitalLoanAccountId",
)

# %%
fact_table, dimension_table = update_tables(
    fact_table, dimension_table, model_name="beta_demo_model_sil", product="SIL"
)
print(f"The shape of the fact table is:\t {fact_table.shape}")
print(f"The shape of the dimension table is:\t {dimension_table.shape}")

# %%
# Upload to BigQuery
table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_table"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(fact_table, table_id, job_config=job_config)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_table"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(dimension_table, table_id, job_config=job_config)
job.result()  # Wait for the job to complete

# %%


# %% [markdown]
# # CASH

# %% [markdown]
# # Alpha-Cash-CIC-Model

# %% [markdown]
# ### FPD0

# %% [markdown]
# ### Test

# %%
sq = r"""
WITH parsed as (
select customerId, digitalLoanAccountId,modelDisplayName,modelVersionId,start_time,end_time,prediction,trenchCategory,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeatures,
FROM `prj-prod-dataplatform.audit_balance.ml_model_run_details`
where modelDisplayName in ('Alpha-Cash-CIC-Model','Alpha Cash CIC Model','cic_model_cash')
),
model_run as (
select customerId,digitalLoanAccountId,modelName, publish_time,requestPayload as requestPayload_clean
--REPLACE(REPLACE(requestPayload, "'", '"'), "None", "null") AS requestPayload_clean
from `prj-prod-dataplatform.audit_balance.ml_request_details` 
WHERE modelName = 'Alpha-Cash-Model-response'
QUALIFY ROW_NUMBER() OVER (PARTITION BY customerId, digitalLoanAccountId,modelName ORDER BY publish_time DESC ) = 1)
, 
  modelname as (
  select p.customerId,
  p.digitalLoanAccountId,
  p.start_time,
  p.prediction aCicScore,
  case when p.modelDisplayName like 'Alpha%' then 'cic_model_cash' else p.modelDisplayName end as modelDisplayName,
  p.modelVersionId,
  coalesce (p.trenchCategory, REGEXP_EXTRACT(m.requestPayload_clean, r"trenchCategory[:=]['\"]?([^'\"]+)['\"]?")) trenchCategory
  from parsed p
  left join model_run m on m.digitalLoanAccountId = p.digitalLoanAccountId
  ),
  deliquency as
(select loanAccountNumber,
case when obs_min_inst_def0 >= 1 and min_inst_def0 = 1 then 1 else 0 end deffpd0,
case when obs_min_inst_def10 >=1 and min_inst_def10 =1 then 1 else 0 end deffpd10,
case when obs_min_inst_def30 >=1 and min_inst_def30 =1 then 1 else 0 end deffpd30,
case when obs_min_inst_def30 >=2 and min_inst_def30 in (1,2) then 1 else 0 end deffspd30,
case when obs_min_inst_def30 >=3 and min_inst_def30 in (1,2,3) then 1 else 0 end deffstpd30,
case when obs_min_inst_def0 >= 1 then 1 else 0 end flg_mature_fpd0,
case when obs_min_inst_def10 >=1 then 1 else 0 end flg_mature_fpd10,
case when obs_min_inst_def30 >=1 then 1 else 0 end flg_mature_fpd30,
case when obs_min_inst_def30 >=2 then 1 else 0 end flg_mature_fspd_30,
case when obs_min_inst_def30 >=3 then 1 else 0 end flg_mature_fstpd_30
from prj-prod-dataplatform.risk_credit_mis.loan_deliquency_data),
base as 
  (select distinct r.customerId,
  r.digitalLoanAccountId,
  loanmaster.loanAccountNumber,
  r.modelDisplayName,
  r.aCicScore,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month, 
  'Test' Data_selection,  
  deffpd0,
  flg_mature_fpd0,
  loanmaster.new_loan_type,
  modelVersionId, trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile' 
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall' 
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
  from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  inner join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
   left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
  where loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and r.aCicScore is not null
  and flg_mature_fpd0 = 1
  )
  select *  from base
  ;
"""
dfd = client.query(sq).to_dataframe()
dfd = dfd.drop_duplicates(keep="first")
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()


# %%
df1 = dfd.copy()

# %% [markdown]
# ### Train

# %%
sq = """ 
WITH parsed as (
select customerId, digitalLoanAccountId,modelDisplayName,modelVersionId,start_time,end_time,prediction,trenchCategory,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeatures,
FROM prj-prod-dataplatform.dap_ds_poweruser_playground.ml_training_model_run_details
where modelDisplayName in ('Alpha-Cash-CIC-Model','Alpha Cash CIC Model','cic_model_cash')
and modelVersionId = 'v1'
),
  modelname as (
  select customerId,
  digitalLoanAccountId,
  start_time,
  prediction aCicScore,
  case when modelDisplayName like 'Alpha%' then 'cic_model_cash' else modelDisplayName end as modelDisplayName,
  modelVersionId,
  trenchCategory
  from parsed
  ), 
  deliquency as
(select loanAccountNumber,
case when obs_min_inst_def0 >= 1 and min_inst_def0 = 1 then 1 else 0 end deffpd0,
case when obs_min_inst_def10 >=1 and min_inst_def10 =1 then 1 else 0 end deffpd10,
case when obs_min_inst_def30 >=1 and min_inst_def30 =1 then 1 else 0 end deffpd30,
case when obs_min_inst_def30 >=2 and min_inst_def30 in (1,2) then 1 else 0 end deffspd30,
case when obs_min_inst_def30 >=3 and min_inst_def30 in (1,2,3) then 1 else 0 end deffstpd30,
case when obs_min_inst_def0 >= 1 then 1 else 0 end flg_mature_fpd0,
case when obs_min_inst_def10 >=1 then 1 else 0 end flg_mature_fpd10,
case when obs_min_inst_def30 >=1 then 1 else 0 end flg_mature_fpd30,
case when obs_min_inst_def30 >=2 then 1 else 0 end flg_mature_fspd_30,
case when obs_min_inst_def30 >=3 then 1 else 0 end flg_mature_fstpd_30
from prj-prod-dataplatform.risk_credit_mis.loan_deliquency_data),
base as
(select distinct r.customerId,
  r.digitalLoanAccountId,
  loanmaster.loanAccountNumber,
  r.modelDisplayName,
  r.aCicScore,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month, 
  'Train' Data_selection,  
    deffpd0,
  flg_mature_fpd0,
  loanmaster.new_loan_type,
  modelVersionId,
    trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile' 
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall' 
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
    from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  inner join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
  left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
  where loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and r.aCicScore is not null
  and flg_mature_fpd0 = 1
  )
  select * from base;
  """
dfd = client.query(sq).to_dataframe()
dfd = dfd.drop_duplicates(keep="first")
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()

# %%
df2 = dfd.copy()

# %% [markdown]
# ### Concat

# %%
# df_concat = pd.concat([df1, df2], ignore_index=True)
# print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")

# df_concat = (df_concat
#              .sort_values(by=['digitalLoanAccountId', 'Data_selection'],
#                           key=lambda s: s.map({'Train': 0, 'Test': 1}))
#              .drop_duplicates(subset=['digitalLoanAccountId'], keep='first')
#              .reset_index(drop=True))
# print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")
# df_concat.head()

# 1) Get all IDs present in Train
train_ids = set(df2["digitalLoanAccountId"])

# 2) Keep only Test rows whose ID is NOT in Train
df1_no_dupes = df1[~df1["digitalLoanAccountId"].isin(train_ids)]

# 3) Concatenate
df_concat = pd.concat([df1_no_dupes, df2], ignore_index=True)

print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")
df_concat.head()


# %%
df_concat["aCicScore"] = pd.to_numeric(df_concat["aCicScore"], errors="coerce")

# %%
fact_table, dimension_table = calculate_periodic_gini_prod_ver_trench_dimfact(
    df_concat,
    "aCicScore",
    "deffpd0",
    "FPD0",
    model_version_column="modelVersionId",
    trench_column="trenchCategory",
    loan_type_column="new_loan_type",
    loan_product_type_column="loan_product_type",
    account_id_column="digitalLoanAccountId",
)

# %%
fact_table, dimension_table = update_tables(
    fact_table, dimension_table, model_name="cic_model_cash", product="CASH"
)
print(f"The shape of the fact table is:\t {fact_table.shape}")
print(f"The shape of the dimension table is:\t {dimension_table.shape}")

# %%
# Upload to BigQuery
table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_table"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(fact_table, table_id, job_config=job_config)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_table"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(dimension_table, table_id, job_config=job_config)
job.result()  # Wait for the job to complete

# %% [markdown]
# ### FPD10

# %% [markdown]
# ### Test

# %%
sq = r"""
WITH parsed as (
select customerId, digitalLoanAccountId,modelDisplayName,modelVersionId,start_time,end_time,prediction,trenchCategory,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeatures,
FROM `prj-prod-dataplatform.audit_balance.ml_model_run_details`
where modelDisplayName in ('Alpha-Cash-CIC-Model','Alpha Cash CIC Model','cic_model_cash')
),
model_run as (
select customerId,digitalLoanAccountId,modelName, publish_time,requestPayload as requestPayload_clean
--REPLACE(REPLACE(requestPayload, "'", '"'), "None", "null") AS requestPayload_clean
from `prj-prod-dataplatform.audit_balance.ml_request_details` 
WHERE modelName = 'Alpha-Cash-Model-response'
QUALIFY ROW_NUMBER() OVER (PARTITION BY customerId, digitalLoanAccountId,modelName ORDER BY publish_time DESC ) = 1)
, 
  modelname as (
  select p.customerId,
  p.digitalLoanAccountId,
  p.start_time,
  p.prediction aCicScore,
  case when p.modelDisplayName like 'Alpha%' then 'cic_model_cash' else p.modelDisplayName end as modelDisplayName,
  p.modelVersionId,
  coalesce (p.trenchCategory, REGEXP_EXTRACT(m.requestPayload_clean, r"trenchCategory[:=]['\"]?([^'\"]+)['\"]?")) trenchCategory
  from parsed p
  left join model_run m on m.digitalLoanAccountId = p.digitalLoanAccountId
  ),
  deliquency as
(select loanAccountNumber,
case when obs_min_inst_def0 >= 1 and min_inst_def0 = 1 then 1 else 0 end deffpd0,
case when obs_min_inst_def10 >=1 and min_inst_def10 =1 then 1 else 0 end deffpd10,
case when obs_min_inst_def30 >=1 and min_inst_def30 =1 then 1 else 0 end deffpd30,
case when obs_min_inst_def30 >=2 and min_inst_def30 in (1,2) then 1 else 0 end deffspd30,
case when obs_min_inst_def30 >=3 and min_inst_def30 in (1,2,3) then 1 else 0 end deffstpd30,
case when obs_min_inst_def0 >= 1 then 1 else 0 end flg_mature_fpd0,
case when obs_min_inst_def10 >=1 then 1 else 0 end flg_mature_fpd10,
case when obs_min_inst_def30 >=1 then 1 else 0 end flg_mature_fpd30,
case when obs_min_inst_def30 >=2 then 1 else 0 end flg_mature_fspd_30,
case when obs_min_inst_def30 >=3 then 1 else 0 end flg_mature_fstpd_30
from prj-prod-dataplatform.risk_credit_mis.loan_deliquency_data),
base as 
  (select distinct r.customerId,
  r.digitalLoanAccountId,
  loanmaster.loanAccountNumber,
  r.modelDisplayName,
  r.aCicScore,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month, 
  'Test' Data_selection,  
  del.deffpd10,
  del.flg_mature_fpd10,
  loanmaster.new_loan_type,
  modelVersionId, trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile' 
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall' 
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
  from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  inner join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
   left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
  where loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and r.aCicScore is not null
  and del.flg_mature_fpd10 = 1
  )
  select *  from base
  ;
"""
dfd = client.query(sq).to_dataframe()
dfd = dfd.drop_duplicates(keep="first")
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()


# %%
df1 = dfd.copy()

# %% [markdown]
# ### Train

# %%
sq = """ 
WITH parsed as (
select customerId, digitalLoanAccountId,modelDisplayName,modelVersionId,start_time,end_time,prediction,trenchCategory,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeatures,
FROM prj-prod-dataplatform.dap_ds_poweruser_playground.ml_training_model_run_details
where modelDisplayName in ('Alpha-Cash-CIC-Model','Alpha Cash CIC Model','cic_model_cash')
and modelVersionId = 'v1'
),
  modelname as (
  select customerId,
  digitalLoanAccountId,
  start_time,
  prediction aCicScore,
  case when modelDisplayName like 'Alpha%' then 'cic_model_cash' else modelDisplayName end as modelDisplayName,
  modelVersionId,
  trenchCategory
  from parsed
  ), 
  deliquency as
(select loanAccountNumber,
case when obs_min_inst_def0 >= 1 and min_inst_def0 = 1 then 1 else 0 end deffpd0,
case when obs_min_inst_def10 >=1 and min_inst_def10 =1 then 1 else 0 end deffpd10,
case when obs_min_inst_def30 >=1 and min_inst_def30 =1 then 1 else 0 end deffpd30,
case when obs_min_inst_def30 >=2 and min_inst_def30 in (1,2) then 1 else 0 end deffspd30,
case when obs_min_inst_def30 >=3 and min_inst_def30 in (1,2,3) then 1 else 0 end deffstpd30,
case when obs_min_inst_def0 >= 1 then 1 else 0 end flg_mature_fpd0,
case when obs_min_inst_def10 >=1 then 1 else 0 end flg_mature_fpd10,
case when obs_min_inst_def30 >=1 then 1 else 0 end flg_mature_fpd30,
case when obs_min_inst_def30 >=2 then 1 else 0 end flg_mature_fspd_30,
case when obs_min_inst_def30 >=3 then 1 else 0 end flg_mature_fstpd_30
from prj-prod-dataplatform.risk_credit_mis.loan_deliquency_data),
base as
(select distinct r.customerId,
  r.digitalLoanAccountId,
  loanmaster.loanAccountNumber,
  r.modelDisplayName,
  r.aCicScore,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month, 
  'Train' Data_selection,  
    del.deffpd10,
  del.flg_mature_fpd10,
  loanmaster.new_loan_type,
  modelVersionId,
    trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile' 
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall' 
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
    from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  inner join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
  left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
  where loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and r.aCicScore is not null
  and del.flg_mature_fpd10 = 1
  )
  select * from base;
  """
dfd = client.query(sq).to_dataframe()
dfd = dfd.drop_duplicates(keep="first")
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()

# %%
df2 = dfd.copy()

# %% [markdown]
# ### Concat

# %%
# df_concat = pd.concat([df1, df2], ignore_index=True)
# print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")

# df_concat = (df_concat
#              .sort_values(by=['digitalLoanAccountId', 'Data_selection'],
#                           key=lambda s: s.map({'Train': 0, 'Test': 1}))
#              .drop_duplicates(subset=['digitalLoanAccountId'], keep='first')
#              .reset_index(drop=True))
# print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")
# df_concat.head()

# 1) Get all IDs present in Train
train_ids = set(df2["digitalLoanAccountId"])

# 2) Keep only Test rows whose ID is NOT in Train
df1_no_dupes = df1[~df1["digitalLoanAccountId"].isin(train_ids)]

# 3) Concatenate
df_concat = pd.concat([df1_no_dupes, df2], ignore_index=True)

print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")
df_concat.head()


# %%
df_concat["aCicScore"] = pd.to_numeric(df_concat["aCicScore"], errors="coerce")

# %%
fact_table, dimension_table = calculate_periodic_gini_prod_ver_trench_dimfact(
    df_concat,
    "aCicScore",
    "deffpd10",
    "FPD10",
    model_version_column="modelVersionId",
    trench_column="trenchCategory",
    loan_type_column="new_loan_type",
    loan_product_type_column="loan_product_type",
    account_id_column="digitalLoanAccountId",
)

# %%
fact_table, dimension_table = update_tables(
    fact_table, dimension_table, model_name="cic_model_cash", product="CASH"
)

# %%
# Upload to BigQuery
table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_table"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(fact_table, table_id, job_config=job_config)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_table"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(dimension_table, table_id, job_config=job_config)
job.result()  # Wait for the job to complete

# %%


# %% [markdown]
# ### FPD30

# %% [markdown]
# ### Test

# %%
sq = r"""
WITH parsed as (
select customerId, digitalLoanAccountId,modelDisplayName,modelVersionId,start_time,end_time,prediction,trenchCategory,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeatures,
FROM `prj-prod-dataplatform.audit_balance.ml_model_run_details`
where modelDisplayName in ('Alpha-Cash-CIC-Model','Alpha Cash CIC Model','cic_model_cash')
),
model_run as (
select customerId,digitalLoanAccountId,modelName, publish_time,requestPayload as requestPayload_clean
--REPLACE(REPLACE(requestPayload, "'", '"'), "None", "null") AS requestPayload_clean
from `prj-prod-dataplatform.audit_balance.ml_request_details` 
WHERE modelName = 'Alpha-Cash-Model-response'
QUALIFY ROW_NUMBER() OVER (PARTITION BY customerId, digitalLoanAccountId,modelName ORDER BY publish_time DESC ) = 1)
, 
  modelname as (
  select p.customerId,
  p.digitalLoanAccountId,
  p.start_time,
  p.prediction aCicScore,
  case when p.modelDisplayName like 'Alpha%' then 'cic_model_cash' else p.modelDisplayName end as modelDisplayName,
  p.modelVersionId,
  coalesce (p.trenchCategory, REGEXP_EXTRACT(m.requestPayload_clean, r"trenchCategory[:=]['\"]?([^'\"]+)['\"]?")) trenchCategory
  from parsed p
  left join model_run m on m.digitalLoanAccountId = p.digitalLoanAccountId
  ),
  deliquency as
(select loanAccountNumber,
case when obs_min_inst_def0 >= 1 and min_inst_def0 = 1 then 1 else 0 end deffpd0,
case when obs_min_inst_def10 >=1 and min_inst_def10 =1 then 1 else 0 end deffpd10,
case when obs_min_inst_def30 >=1 and min_inst_def30 =1 then 1 else 0 end deffpd30,
case when obs_min_inst_def30 >=2 and min_inst_def30 in (1,2) then 1 else 0 end deffspd30,
case when obs_min_inst_def30 >=3 and min_inst_def30 in (1,2,3) then 1 else 0 end deffstpd30,
case when obs_min_inst_def0 >= 1 then 1 else 0 end flg_mature_fpd0,
case when obs_min_inst_def10 >=1 then 1 else 0 end flg_mature_fpd10,
case when obs_min_inst_def30 >=1 then 1 else 0 end flg_mature_fpd30,
case when obs_min_inst_def30 >=2 then 1 else 0 end flg_mature_fspd_30,
case when obs_min_inst_def30 >=3 then 1 else 0 end flg_mature_fstpd_30
from prj-prod-dataplatform.risk_credit_mis.loan_deliquency_data),
base as 
  (select distinct r.customerId,
  r.digitalLoanAccountId,
  loanmaster.loanAccountNumber,
  r.modelDisplayName,
  r.aCicScore,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month, 
  'Test' Data_selection,  
  del.deffpd30,
  del.flg_mature_fpd30,
  loanmaster.new_loan_type,
  modelVersionId, trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile' 
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall' 
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
  from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  inner join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
   left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
  where loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and r.aCicScore is not null
  and del.flg_mature_fpd30 = 1
  )
  select *  from base
  ;
"""
dfd = client.query(sq).to_dataframe()
dfd = dfd.drop_duplicates(keep="first")
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()


# %%
df1 = dfd.copy()

# %% [markdown]
# ### Train

# %%
sq = """ 
WITH parsed as (
select customerId, digitalLoanAccountId,modelDisplayName,modelVersionId,start_time,end_time,prediction,trenchCategory,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeatures,
FROM prj-prod-dataplatform.dap_ds_poweruser_playground.ml_training_model_run_details
where modelDisplayName in ('Alpha-Cash-CIC-Model','Alpha Cash CIC Model','cic_model_cash')
and modelVersionId = 'v1'
),
  modelname as (
  select customerId,
  digitalLoanAccountId,
  start_time,
  prediction aCicScore,
  case when modelDisplayName like 'Alpha%' then 'cic_model_cash' else modelDisplayName end as modelDisplayName,
  modelVersionId,
  trenchCategory
  from parsed
  ), 
  deliquency as
(select loanAccountNumber,
case when obs_min_inst_def0 >= 1 and min_inst_def0 = 1 then 1 else 0 end deffpd0,
case when obs_min_inst_def10 >=1 and min_inst_def10 =1 then 1 else 0 end deffpd10,
case when obs_min_inst_def30 >=1 and min_inst_def30 =1 then 1 else 0 end deffpd30,
case when obs_min_inst_def30 >=2 and min_inst_def30 in (1,2) then 1 else 0 end deffspd30,
case when obs_min_inst_def30 >=3 and min_inst_def30 in (1,2,3) then 1 else 0 end deffstpd30,
case when obs_min_inst_def0 >= 1 then 1 else 0 end flg_mature_fpd0,
case when obs_min_inst_def10 >=1 then 1 else 0 end flg_mature_fpd10,
case when obs_min_inst_def30 >=1 then 1 else 0 end flg_mature_fpd30,
case when obs_min_inst_def30 >=2 then 1 else 0 end flg_mature_fspd_30,
case when obs_min_inst_def30 >=3 then 1 else 0 end flg_mature_fstpd_30
from prj-prod-dataplatform.risk_credit_mis.loan_deliquency_data),
base as
(select distinct r.customerId,
  r.digitalLoanAccountId,
  loanmaster.loanAccountNumber,
  r.modelDisplayName,
  r.aCicScore,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month, 
  'Train' Data_selection,  
    del.deffpd30,
  del.flg_mature_fpd30,
  loanmaster.new_loan_type,
  modelVersionId,
    trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile' 
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall' 
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
    from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  inner join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
  left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
  where loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and r.aCicScore is not null
  and del.flg_mature_fpd30 = 1
  )
  select * from base;
  """
dfd = client.query(sq).to_dataframe()
dfd = dfd.drop_duplicates(keep="first")
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()

# %%
df2 = dfd.copy()

# %% [markdown]
# ### Concat

# %%
# df_concat = pd.concat([df1, df2], ignore_index=True)
# print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")

# df_concat = (df_concat
#              .sort_values(by=['digitalLoanAccountId', 'Data_selection'],
#                           key=lambda s: s.map({'Train': 0, 'Test': 1}))
#              .drop_duplicates(subset=['digitalLoanAccountId'], keep='first')
#              .reset_index(drop=True))
# print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")
# df_concat.head()


# 1) Get all IDs present in Train
train_ids = set(df2["digitalLoanAccountId"])

# 2) Keep only Test rows whose ID is NOT in Train
df1_no_dupes = df1[~df1["digitalLoanAccountId"].isin(train_ids)]

# 3) Concatenate
df_concat = pd.concat([df1_no_dupes, df2], ignore_index=True)

print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")
df_concat.head()


# %%
df_concat["aCicScore"] = pd.to_numeric(df_concat["aCicScore"], errors="coerce")

# %%
fact_table, dimension_table = calculate_periodic_gini_prod_ver_trench_dimfact(
    df_concat,
    "aCicScore",
    "deffpd30",
    "FPD30",
    model_version_column="modelVersionId",
    trench_column="trenchCategory",
    loan_type_column="new_loan_type",
    loan_product_type_column="loan_product_type",
    account_id_column="digitalLoanAccountId",
)

# %%
fact_table, dimension_table = update_tables(
    fact_table, dimension_table, model_name="cic_model_cash", product="CASH"
)

# %%
# Upload to BigQuery
table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_table"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(fact_table, table_id, job_config=job_config)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_table"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(dimension_table, table_id, job_config=job_config)
job.result()  # Wait for the job to complete

# %%


# %% [markdown]
# ### FSPD30

# %% [markdown]
# ### Test

# %%
sq = r"""
WITH parsed as (
select customerId, digitalLoanAccountId,modelDisplayName,modelVersionId,start_time,end_time,prediction,trenchCategory,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeatures,
FROM `prj-prod-dataplatform.audit_balance.ml_model_run_details`
where modelDisplayName in ('Alpha-Cash-CIC-Model','Alpha Cash CIC Model','cic_model_cash')
),
model_run as (
select customerId,digitalLoanAccountId,modelName, publish_time,requestPayload as requestPayload_clean
--REPLACE(REPLACE(requestPayload, "'", '"'), "None", "null") AS requestPayload_clean
from `prj-prod-dataplatform.audit_balance.ml_request_details` 
WHERE modelName = 'Alpha-Cash-Model-response'
QUALIFY ROW_NUMBER() OVER (PARTITION BY customerId, digitalLoanAccountId,modelName ORDER BY publish_time DESC ) = 1)
, 
  modelname as (
  select p.customerId,
  p.digitalLoanAccountId,
  p.start_time,
  p.prediction aCicScore,
  case when p.modelDisplayName like 'Alpha%' then 'cic_model_cash' else p.modelDisplayName end as modelDisplayName,
  p.modelVersionId,
  coalesce (p.trenchCategory, REGEXP_EXTRACT(m.requestPayload_clean, r"trenchCategory[:=]['\"]?([^'\"]+)['\"]?")) trenchCategory
  from parsed p
  left join model_run m on m.digitalLoanAccountId = p.digitalLoanAccountId
  ),
  deliquency as
(select loanAccountNumber,
case when obs_min_inst_def0 >= 1 and min_inst_def0 = 1 then 1 else 0 end deffpd0,
case when obs_min_inst_def10 >=1 and min_inst_def10 =1 then 1 else 0 end deffpd10,
case when obs_min_inst_def30 >=1 and min_inst_def30 =1 then 1 else 0 end deffpd30,
case when obs_min_inst_def30 >=2 and min_inst_def30 in (1,2) then 1 else 0 end deffspd30,
case when obs_min_inst_def30 >=3 and min_inst_def30 in (1,2,3) then 1 else 0 end deffstpd30,
case when obs_min_inst_def0 >= 1 then 1 else 0 end flg_mature_fpd0,
case when obs_min_inst_def10 >=1 then 1 else 0 end flg_mature_fpd10,
case when obs_min_inst_def30 >=1 then 1 else 0 end flg_mature_fpd30,
case when obs_min_inst_def30 >=2 then 1 else 0 end flg_mature_fspd_30,
case when obs_min_inst_def30 >=3 then 1 else 0 end flg_mature_fstpd_30
from prj-prod-dataplatform.risk_credit_mis.loan_deliquency_data),
base as 
  (select distinct r.customerId,
  r.digitalLoanAccountId,
  loanmaster.loanAccountNumber,
  r.modelDisplayName,
  r.aCicScore,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month, 
  'Test' Data_selection,  
  del.deffspd30,
  del.flg_mature_fspd_30,
  loanmaster.new_loan_type,
  modelVersionId, trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile' 
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall' 
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
  from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  inner join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
   left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
  where loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and r.aCicScore is not null
  and del.flg_mature_fspd_30 = 1
  )
  select *  from base
  ;
"""
dfd = client.query(sq).to_dataframe()
dfd = dfd.drop_duplicates(keep="first")
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()


# %%
df1 = dfd.copy()

# %% [markdown]
# ### Train

# %%
sq = """ 
WITH parsed as (
select customerId, digitalLoanAccountId,modelDisplayName,modelVersionId,start_time,end_time,prediction,trenchCategory,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeatures,
FROM prj-prod-dataplatform.dap_ds_poweruser_playground.ml_training_model_run_details
where modelDisplayName in ('Alpha-Cash-CIC-Model','Alpha Cash CIC Model','cic_model_cash')
and modelVersionId = 'v1'
),
  modelname as (
  select customerId,
  digitalLoanAccountId,
  start_time,
  prediction aCicScore,
  case when modelDisplayName like 'Alpha%' then 'cic_model_cash' else modelDisplayName end as modelDisplayName,
  modelVersionId,
  trenchCategory
  from parsed
  ), 
  deliquency as
(select loanAccountNumber,
case when obs_min_inst_def0 >= 1 and min_inst_def0 = 1 then 1 else 0 end deffpd0,
case when obs_min_inst_def10 >=1 and min_inst_def10 =1 then 1 else 0 end deffpd10,
case when obs_min_inst_def30 >=1 and min_inst_def30 =1 then 1 else 0 end deffpd30,
case when obs_min_inst_def30 >=2 and min_inst_def30 in (1,2) then 1 else 0 end deffspd30,
case when obs_min_inst_def30 >=3 and min_inst_def30 in (1,2,3) then 1 else 0 end deffstpd30,
case when obs_min_inst_def0 >= 1 then 1 else 0 end flg_mature_fpd0,
case when obs_min_inst_def10 >=1 then 1 else 0 end flg_mature_fpd10,
case when obs_min_inst_def30 >=1 then 1 else 0 end flg_mature_fpd30,
case when obs_min_inst_def30 >=2 then 1 else 0 end flg_mature_fspd_30,
case when obs_min_inst_def30 >=3 then 1 else 0 end flg_mature_fstpd_30
from prj-prod-dataplatform.risk_credit_mis.loan_deliquency_data),
base as
(select distinct r.customerId,
  r.digitalLoanAccountId,
  loanmaster.loanAccountNumber,
  r.modelDisplayName,
  r.aCicScore,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month, 
  'Train' Data_selection,  
    del.deffspd30,
  del.flg_mature_fspd_30,
  loanmaster.new_loan_type,
  modelVersionId,
    trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile' 
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall' 
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
    from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  inner join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
  left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
  where loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and r.aCicScore is not null
  and del.flg_mature_fspd_30 = 1
  )
  select * from base;
  """
dfd = client.query(sq).to_dataframe()
dfd = dfd.drop_duplicates(keep="first")
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()

# %%
df2 = dfd.copy()

# %% [markdown]
# ### Concat

# %%
# df_concat = pd.concat([df1, df2], ignore_index=True)
# print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")

# df_concat = (df_concat
#              .sort_values(by=['digitalLoanAccountId', 'Data_selection'],
#                           key=lambda s: s.map({'Train': 0, 'Test': 1}))
#              .drop_duplicates(subset=['digitalLoanAccountId'], keep='first')
#              .reset_index(drop=True))
# print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")
# df_concat.head()


# 1) Get all IDs present in Train
train_ids = set(df2["digitalLoanAccountId"])

# 2) Keep only Test rows whose ID is NOT in Train
df1_no_dupes = df1[~df1["digitalLoanAccountId"].isin(train_ids)]

# 3) Concatenate
df_concat = pd.concat([df1_no_dupes, df2], ignore_index=True)

print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")
df_concat.head()


# %%
df_concat.groupby(
    ["Data_selection", "new_loan_type", "modelVersionId", "loan_product_type"]
)["digitalLoanAccountId"].nunique()

# %%
df_concat["aCicScore"] = pd.to_numeric(df_concat["aCicScore"], errors="coerce")

# %%
fact_table, dimension_table = calculate_periodic_gini_prod_ver_trench_dimfact(
    df_concat,
    "aCicScore",
    "deffspd30",
    "FSPD30",
    model_version_column="modelVersionId",
    trench_column="trenchCategory",
    loan_type_column="new_loan_type",
    loan_product_type_column="loan_product_type",
    account_id_column="digitalLoanAccountId",
)

# %%
fact_table, dimension_table = update_tables(
    fact_table, dimension_table, model_name="cic_model_cash", product="CASH"
)

# %%
# Upload to BigQuery
table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_table"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(fact_table, table_id, job_config=job_config)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_table"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(dimension_table, table_id, job_config=job_config)
job.result()  # Wait for the job to complete

# %%


# %% [markdown]
# ### FSTPD30

# %% [markdown]
# ### Test

# %%
sq = r"""
WITH parsed as (
select customerId, digitalLoanAccountId,modelDisplayName,modelVersionId,start_time,end_time,prediction,trenchCategory,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeatures,
FROM `prj-prod-dataplatform.audit_balance.ml_model_run_details`
where modelDisplayName in ('Alpha-Cash-CIC-Model','Alpha Cash CIC Model','cic_model_cash')
),
model_run as (
select customerId,digitalLoanAccountId,modelName, publish_time,requestPayload as requestPayload_clean
--REPLACE(REPLACE(requestPayload, "'", '"'), "None", "null") AS requestPayload_clean
from `prj-prod-dataplatform.audit_balance.ml_request_details` 
WHERE modelName = 'Alpha-Cash-Model-response'
QUALIFY ROW_NUMBER() OVER (PARTITION BY customerId, digitalLoanAccountId,modelName ORDER BY publish_time DESC ) = 1)
, 
  modelname as (
  select p.customerId,
  p.digitalLoanAccountId,
  p.start_time,
  p.prediction aCicScore,
  case when p.modelDisplayName like 'Alpha%' then 'cic_model_cash' else p.modelDisplayName end as modelDisplayName,
  p.modelVersionId,
  coalesce (p.trenchCategory, REGEXP_EXTRACT(m.requestPayload_clean, r"trenchCategory[:=]['\"]?([^'\"]+)['\"]?")) trenchCategory
  from parsed p
  left join model_run m on m.digitalLoanAccountId = p.digitalLoanAccountId
  ),
  deliquency as
(select loanAccountNumber,
case when obs_min_inst_def0 >= 1 and min_inst_def0 = 1 then 1 else 0 end deffpd0,
case when obs_min_inst_def10 >=1 and min_inst_def10 =1 then 1 else 0 end deffpd10,
case when obs_min_inst_def30 >=1 and min_inst_def30 =1 then 1 else 0 end deffpd30,
case when obs_min_inst_def30 >=2 and min_inst_def30 in (1,2) then 1 else 0 end deffspd30,
case when obs_min_inst_def30 >=3 and min_inst_def30 in (1,2,3) then 1 else 0 end deffstpd30,
case when obs_min_inst_def0 >= 1 then 1 else 0 end flg_mature_fpd0,
case when obs_min_inst_def10 >=1 then 1 else 0 end flg_mature_fpd10,
case when obs_min_inst_def30 >=1 then 1 else 0 end flg_mature_fpd30,
case when obs_min_inst_def30 >=2 then 1 else 0 end flg_mature_fspd_30,
case when obs_min_inst_def30 >=3 then 1 else 0 end flg_mature_fstpd_30
from prj-prod-dataplatform.risk_credit_mis.loan_deliquency_data),
base as 
  (select distinct r.customerId,
  r.digitalLoanAccountId,
  loanmaster.loanAccountNumber,
  r.modelDisplayName,
  r.aCicScore,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month, 
  'Test' Data_selection,  
  del.deffstpd30,
  del.flg_mature_fstpd_30,
  loanmaster.new_loan_type,
  modelVersionId, trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile' 
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall' 
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
  from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  inner join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
   left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
  where loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and r.aCicScore is not null
  and del.flg_mature_fstpd_30 = 1
  )
  select *  from base
  ;
"""
dfd = client.query(sq).to_dataframe()
dfd = dfd.drop_duplicates(keep="first")
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()


# %%
df1 = dfd.copy()

# %% [markdown]
# ### Train

# %%
sq = """ 
WITH parsed as (
select customerId, digitalLoanAccountId,modelDisplayName,modelVersionId,start_time,end_time,prediction,trenchCategory,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeatures,
FROM prj-prod-dataplatform.dap_ds_poweruser_playground.ml_training_model_run_details
where modelDisplayName in ('Alpha-Cash-CIC-Model','Alpha Cash CIC Model','cic_model_cash')
and modelVersionId = 'v1'
),
  modelname as (
  select customerId,
  digitalLoanAccountId,
  start_time,
  prediction aCicScore,
  case when modelDisplayName like 'Alpha%' then 'cic_model_cash' else modelDisplayName end as modelDisplayName,
  modelVersionId,
  trenchCategory
  from parsed
  ), 
  deliquency as
(select loanAccountNumber,
case when obs_min_inst_def0 >= 1 and min_inst_def0 = 1 then 1 else 0 end deffpd0,
case when obs_min_inst_def10 >=1 and min_inst_def10 =1 then 1 else 0 end deffpd10,
case when obs_min_inst_def30 >=1 and min_inst_def30 =1 then 1 else 0 end deffpd30,
case when obs_min_inst_def30 >=2 and min_inst_def30 in (1,2) then 1 else 0 end deffspd30,
case when obs_min_inst_def30 >=3 and min_inst_def30 in (1,2,3) then 1 else 0 end deffstpd30,
case when obs_min_inst_def0 >= 1 then 1 else 0 end flg_mature_fpd0,
case when obs_min_inst_def10 >=1 then 1 else 0 end flg_mature_fpd10,
case when obs_min_inst_def30 >=1 then 1 else 0 end flg_mature_fpd30,
case when obs_min_inst_def30 >=2 then 1 else 0 end flg_mature_fspd_30,
case when obs_min_inst_def30 >=3 then 1 else 0 end flg_mature_fstpd_30
from prj-prod-dataplatform.risk_credit_mis.loan_deliquency_data),
base as
(select distinct r.customerId,
  r.digitalLoanAccountId,
  loanmaster.loanAccountNumber,
  r.modelDisplayName,
  r.aCicScore,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month, 
  'Train' Data_selection,  
    del.deffstpd30,
  del.flg_mature_fstpd_30,
  loanmaster.new_loan_type,
  modelVersionId,
    trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile' 
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall' 
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
    from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  inner join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
  left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
  where loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and r.aCicScore is not null
  and del.flg_mature_fstpd_30 = 1
  )
  select * from base;
  """
dfd = client.query(sq).to_dataframe()
dfd = dfd.drop_duplicates(keep="first")
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()

# %%
df2 = dfd.copy()

# %% [markdown]
# ### Concat

# %%
# df_concat = pd.concat([df1, df2], ignore_index=True)
# print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")

# df_concat = (df_concat
#              .sort_values(by=['digitalLoanAccountId', 'Data_selection'],
#                           key=lambda s: s.map({'Train': 0, 'Test': 1}))
#              .drop_duplicates(subset=['digitalLoanAccountId'], keep='first')
#              .reset_index(drop=True))
# print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")
# df_concat.head()


# 1) Get all IDs present in Train
train_ids = set(df2["digitalLoanAccountId"])

# 2) Keep only Test rows whose ID is NOT in Train
df1_no_dupes = df1[~df1["digitalLoanAccountId"].isin(train_ids)]

# 3) Concatenate
df_concat = pd.concat([df1_no_dupes, df2], ignore_index=True)

print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")
df_concat.head()


# %%
df_concat["aCicScore"] = pd.to_numeric(df_concat["aCicScore"], errors="coerce")

# %%
fact_table, dimension_table = calculate_periodic_gini_prod_ver_trench_dimfact(
    df_concat,
    "aCicScore",
    "deffstpd30",
    "FSTPD30",
    model_version_column="modelVersionId",
    trench_column="trenchCategory",
    loan_type_column="new_loan_type",
    loan_product_type_column="loan_product_type",
    account_id_column="digitalLoanAccountId",
)

# %%
fact_table, dimension_table = update_tables(
    fact_table, dimension_table, model_name="cic_model_cash", product="CASH"
)
print(f"The shape of the fact table is:\t {fact_table.shape}")
print(f"The shape of the dimension table is:\t {dimension_table.shape}")

# %%
# Upload to BigQuery
table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_table"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(fact_table, table_id, job_config=job_config)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_table"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(dimension_table, table_id, job_config=job_config)
job.result()  # Wait for the job to complete

# %% [markdown]
# # Alpha-Cash-Stack-Model

# %% [markdown]
# ### FPD0

# %% [markdown]
# ### Test

# %%
sq = r""" 
WITH parsed as (
select customerId, digitalLoanAccountId,modelDisplayName,modelVersionId,start_time,end_time,prediction,trenchCategory,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeatures,
FROM `prj-prod-dataplatform.audit_balance.ml_model_run_details`
where modelDisplayName in ('Alpha-Cash-Stack-Model', 'alpha_stack_model_cash')
),
model_run as (
select customerId,digitalLoanAccountId,modelName, publish_time,requestPayload as requestPayload_clean
from `prj-prod-dataplatform.audit_balance.ml_request_details` 
WHERE modelName = 'Alpha-Cash-Model-response'
QUALIFY ROW_NUMBER() OVER (PARTITION BY customerId, digitalLoanAccountId,modelName ORDER BY publish_time DESC ) = 1)
, 
  modelname as (
  select p.customerId,
  p.digitalLoanAccountId,
  p.start_time,
  p.prediction aStackScore,
  case when p.modelDisplayName like 'Alpha%' then 'alpha_stack_model_cash' else p.modelDisplayName end as modelDisplayName,
  p.modelVersionId,
  coalesce (p.trenchCategory, REGEXP_EXTRACT(m.requestPayload_clean, r"trenchCategory[:=]['\"]?([^'\"]+)['\"]?")) trenchCategory
  from parsed p
  left join model_run m on m.digitalLoanAccountId = p.digitalLoanAccountId
  ),
  deliquency as
(select loanAccountNumber,
case when obs_min_inst_def0 >= 1 and min_inst_def0 = 1 then 1 else 0 end deffpd0,
case when obs_min_inst_def10 >=1 and min_inst_def10 =1 then 1 else 0 end deffpd10,
case when obs_min_inst_def30 >=1 and min_inst_def30 =1 then 1 else 0 end deffpd30,
case when obs_min_inst_def30 >=2 and min_inst_def30 in (1,2) then 1 else 0 end deffspd30,
case when obs_min_inst_def30 >=3 and min_inst_def30 in (1,2,3) then 1 else 0 end deffstpd30,
case when obs_min_inst_def0 >= 1 then 1 else 0 end flg_mature_fpd0,
case when obs_min_inst_def10 >=1 then 1 else 0 end flg_mature_fpd10,
case when obs_min_inst_def30 >=1 then 1 else 0 end flg_mature_fpd30,
case when obs_min_inst_def30 >=2 then 1 else 0 end flg_mature_fspd_30,
case when obs_min_inst_def30 >=3 then 1 else 0 end flg_mature_fstpd_30
from prj-prod-dataplatform.risk_credit_mis.loan_deliquency_data),
base as 
  (select distinct r.customerId,
  r.digitalLoanAccountId,
  loanmaster.loanAccountNumber,
  r.modelDisplayName,
  r.aStackScore,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month, 
  'Test' Data_selection,  
  deffpd0,
  flg_mature_fpd0,
  loanmaster.new_loan_type,
  modelVersionId, trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile' 
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall' 
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
  from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  inner join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
   left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
  where loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and r.aStackScore is not null
  and flg_mature_fpd0 = 1
  )
  select *  from base
  ;
"""
dfd = client.query(sq).to_dataframe()
dfd = dfd.drop_duplicates(keep="first")
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()


# %%
df1 = dfd.copy()

# %% [markdown]
# ### Train

# %%
sq = """ 
WITH parsed as (
select customerId, digitalLoanAccountId,modelDisplayName,modelVersionId,start_time,end_time,prediction,trenchCategory,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeatures,
FROM prj-prod-dataplatform.dap_ds_poweruser_playground.ml_training_model_run_details
where modelDisplayName in ('Alpha-Cash-Stack-Model', 'alpha_stack_model_cash')
),
  modelname as (
  select customerId,
  digitalLoanAccountId,
  start_time,
  prediction aStackScore,
  case when p.modelDisplayName like 'Alpha%' then 'alpha_stack_model_cash' else p.modelDisplayName end as modelDisplayName,
  p.modelVersionId,
  trenchCategory
  from parsed p
  ) ,
  deliquency as
(select loanAccountNumber,
case when obs_min_inst_def0 >= 1 and min_inst_def0 = 1 then 1 else 0 end deffpd0,
case when obs_min_inst_def10 >=1 and min_inst_def10 =1 then 1 else 0 end deffpd10,
case when obs_min_inst_def30 >=1 and min_inst_def30 =1 then 1 else 0 end deffpd30,
case when obs_min_inst_def30 >=2 and min_inst_def30 in (1,2) then 1 else 0 end deffspd30,
case when obs_min_inst_def30 >=3 and min_inst_def30 in (1,2,3) then 1 else 0 end deffstpd30,
case when obs_min_inst_def0 >= 1 then 1 else 0 end flg_mature_fpd0,
case when obs_min_inst_def10 >=1 then 1 else 0 end flg_mature_fpd10,
case when obs_min_inst_def30 >=1 then 1 else 0 end flg_mature_fpd30,
case when obs_min_inst_def30 >=2 then 1 else 0 end flg_mature_fspd_30,
case when obs_min_inst_def30 >=3 then 1 else 0 end flg_mature_fstpd_30
from prj-prod-dataplatform.risk_credit_mis.loan_deliquency_data),
base as
(select distinct r.customerId,
  r.digitalLoanAccountId,
  loanmaster.loanAccountNumber,
  r.modelDisplayName,
  r.aStackScore,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month, 
  'Train' Data_selection,  
    deffpd0,
  flg_mature_fpd0,
  loanmaster.new_loan_type,
  modelVersionId,
    trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile' 
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall' 
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
    from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  inner join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
  left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
  where loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and r.aStackScore is not null
  and flg_mature_fpd0 = 1
  )
  select * from base;
  """
dfd = client.query(sq).to_dataframe()
dfd = dfd.drop_duplicates(keep="first")
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()

# %%
df2 = dfd.copy()

# %% [markdown]
# ### Concat

# %%
# df_concat = pd.concat([df1, df2], ignore_index=True)
# print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")

# df_concat = (df_concat
#              .sort_values(by=['digitalLoanAccountId', 'Data_selection'],
#                           key=lambda s: s.map({'Train': 0, 'Test': 1}))
#              .drop_duplicates(subset=['digitalLoanAccountId'], keep='first')
#              .reset_index(drop=True))
# print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")
# df_concat.head()


# 1) Get all IDs present in Train
train_ids = set(df2["digitalLoanAccountId"])

# 2) Keep only Test rows whose ID is NOT in Train
df1_no_dupes = df1[~df1["digitalLoanAccountId"].isin(train_ids)]

# 3) Concatenate
df_concat = pd.concat([df1_no_dupes, df2], ignore_index=True)

print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")
df_concat.head()


# %%
df_concat["aStackScore"] = pd.to_numeric(df_concat["aStackScore"], errors="coerce")

# %%
fact_table, dimension_table = calculate_periodic_gini_prod_ver_trench_dimfact(
    df_concat,
    "aStackScore",
    "deffpd0",
    "FPD0",
    model_version_column="modelVersionId",
    trench_column="trenchCategory",
    loan_type_column="new_loan_type",
    loan_product_type_column="loan_product_type",
    account_id_column="digitalLoanAccountId",
)

# %%
fact_table, dimension_table = update_tables(
    fact_table, dimension_table, model_name="alpha_stack_model_cash", product="CASH"
)
print(f"The shape of the fact table is:\t {fact_table.shape}")
print(f"The shape of the dimension table is:\t {dimension_table.shape}")

# %%
# Upload to BigQuery
table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_table"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(fact_table, table_id, job_config=job_config)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_table"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(dimension_table, table_id, job_config=job_config)
job.result()  # Wait for the job to complete

# %% [markdown]
# ### FPD10

# %% [markdown]
# ### Test

# %%
sq = r""" 
WITH parsed as (
select customerId, digitalLoanAccountId,modelDisplayName,modelVersionId,start_time,end_time,prediction,trenchCategory,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeatures,
FROM `prj-prod-dataplatform.audit_balance.ml_model_run_details`
where modelDisplayName in ('Alpha-Cash-Stack-Model', 'alpha_stack_model_cash')
),
model_run as (
select customerId,digitalLoanAccountId,modelName, publish_time,requestPayload as requestPayload_clean
from `prj-prod-dataplatform.audit_balance.ml_request_details` 
WHERE modelName = 'Alpha-Cash-Model-response'
QUALIFY ROW_NUMBER() OVER (PARTITION BY customerId, digitalLoanAccountId,modelName ORDER BY publish_time DESC ) = 1)
, 
  modelname as (
  select p.customerId,
  p.digitalLoanAccountId,
  p.start_time,
  p.prediction aStackScore,
  case when p.modelDisplayName like 'Alpha%' then 'alpha_stack_model_cash' else p.modelDisplayName end as modelDisplayName,
  p.modelVersionId,
  coalesce (p.trenchCategory, REGEXP_EXTRACT(m.requestPayload_clean, r"trenchCategory[:=]['\"]?([^'\"]+)['\"]?")) trenchCategory
  from parsed p
  left join model_run m on m.digitalLoanAccountId = p.digitalLoanAccountId
  ),
  deliquency as
(select loanAccountNumber,
case when obs_min_inst_def0 >= 1 and min_inst_def0 = 1 then 1 else 0 end deffpd0,
case when obs_min_inst_def10 >=1 and min_inst_def10 =1 then 1 else 0 end deffpd10,
case when obs_min_inst_def30 >=1 and min_inst_def30 =1 then 1 else 0 end deffpd30,
case when obs_min_inst_def30 >=2 and min_inst_def30 in (1,2) then 1 else 0 end deffspd30,
case when obs_min_inst_def30 >=3 and min_inst_def30 in (1,2,3) then 1 else 0 end deffstpd30,
case when obs_min_inst_def0 >= 1 then 1 else 0 end flg_mature_fpd0,
case when obs_min_inst_def10 >=1 then 1 else 0 end flg_mature_fpd10,
case when obs_min_inst_def30 >=1 then 1 else 0 end flg_mature_fpd30,
case when obs_min_inst_def30 >=2 then 1 else 0 end flg_mature_fspd_30,
case when obs_min_inst_def30 >=3 then 1 else 0 end flg_mature_fstpd_30
from prj-prod-dataplatform.risk_credit_mis.loan_deliquency_data),
base as 
  (select distinct r.customerId,
  r.digitalLoanAccountId,
  loanmaster.loanAccountNumber,
  r.modelDisplayName,
  r.aStackScore,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month, 
  'Test' Data_selection,  
  del.deffpd10,
  del.flg_mature_fpd10,
  loanmaster.new_loan_type,
  modelVersionId, trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile' 
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall' 
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
  from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  inner join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
   left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
  where loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and r.aStackScore is not null
  and del.flg_mature_fpd10 = 1
  )
  select *  from base
  ;
"""
dfd = client.query(sq).to_dataframe()
dfd = dfd.drop_duplicates(keep="first")
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()


# %%
df1 = dfd.copy()

# %% [markdown]
# ### Train

# %%
sq = """ 
WITH parsed as (
select customerId, digitalLoanAccountId,modelDisplayName,modelVersionId,start_time,end_time,prediction,trenchCategory,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeatures,
FROM prj-prod-dataplatform.dap_ds_poweruser_playground.ml_training_model_run_details
where modelDisplayName in ('Alpha-Cash-Stack-Model', 'alpha_stack_model_cash')
),
  modelname as (
  select customerId,
  digitalLoanAccountId,
  start_time,
  prediction aStackScore,
  case when p.modelDisplayName like 'Alpha%' then 'alpha_stack_model_cash' else p.modelDisplayName end as modelDisplayName,
  p.modelVersionId,
  trenchCategory
  from parsed p
  ),
  deliquency as
(select loanAccountNumber,
case when obs_min_inst_def0 >= 1 and min_inst_def0 = 1 then 1 else 0 end deffpd0,
case when obs_min_inst_def10 >=1 and min_inst_def10 =1 then 1 else 0 end deffpd10,
case when obs_min_inst_def30 >=1 and min_inst_def30 =1 then 1 else 0 end deffpd30,
case when obs_min_inst_def30 >=2 and min_inst_def30 in (1,2) then 1 else 0 end deffspd30,
case when obs_min_inst_def30 >=3 and min_inst_def30 in (1,2,3) then 1 else 0 end deffstpd30,
case when obs_min_inst_def0 >= 1 then 1 else 0 end flg_mature_fpd0,
case when obs_min_inst_def10 >=1 then 1 else 0 end flg_mature_fpd10,
case when obs_min_inst_def30 >=1 then 1 else 0 end flg_mature_fpd30,
case when obs_min_inst_def30 >=2 then 1 else 0 end flg_mature_fspd_30,
case when obs_min_inst_def30 >=3 then 1 else 0 end flg_mature_fstpd_30
from prj-prod-dataplatform.risk_credit_mis.loan_deliquency_data),
base as
(select distinct r.customerId,
  r.digitalLoanAccountId,
  loanmaster.loanAccountNumber,
  r.modelDisplayName,
  r.aStackScore,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month, 
  'Train' Data_selection,  
    del.deffpd10,
  del.flg_mature_fpd10,
  loanmaster.new_loan_type,
  modelVersionId,
    trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile' 
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall' 
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
    from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  inner join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
  left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
  where loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and r.aStackScore is not null
  and del.flg_mature_fpd10 = 1
  )
  select * from base;
  """
dfd = client.query(sq).to_dataframe()
dfd = dfd.drop_duplicates(keep="first")
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()

# %%
df2 = dfd.copy()

# %% [markdown]
# ### Concat

# %%
# df_concat = pd.concat([df1, df2], ignore_index=True)
# print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")

# df_concat = (df_concat
#              .sort_values(by=['digitalLoanAccountId', 'Data_selection'],
#                           key=lambda s: s.map({'Train': 0, 'Test': 1}))
#              .drop_duplicates(subset=['digitalLoanAccountId'], keep='first')
#              .reset_index(drop=True))
# print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")
# df_concat.head()


# 1) Get all IDs present in Train
train_ids = set(df2["digitalLoanAccountId"])

# 2) Keep only Test rows whose ID is NOT in Train
df1_no_dupes = df1[~df1["digitalLoanAccountId"].isin(train_ids)]

# 3) Concatenate
df_concat = pd.concat([df1_no_dupes, df2], ignore_index=True)

print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")
df_concat.head()


# %%
df_concat["aStackScore"] = pd.to_numeric(df_concat["aStackScore"], errors="coerce")

# %%
fact_table, dimension_table = calculate_periodic_gini_prod_ver_trench_dimfact(
    df_concat,
    "aStackScore",
    "deffpd10",
    "FPD10",
    model_version_column="modelVersionId",
    trench_column="trenchCategory",
    loan_type_column="new_loan_type",
    loan_product_type_column="loan_product_type",
    account_id_column="digitalLoanAccountId",
)

# %%
fact_table, dimension_table = update_tables(
    fact_table, dimension_table, model_name="alpha_stack_model_cash", product="CASH"
)

# %%
# Upload to BigQuery
table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_table"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(fact_table, table_id, job_config=job_config)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_table"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(dimension_table, table_id, job_config=job_config)
job.result()  # Wait for the job to complete

# %%


# %% [markdown]
# ### FPD30

# %% [markdown]
# ### Test

# %%
sq = r""" 
WITH parsed as (
select customerId, digitalLoanAccountId,modelDisplayName,modelVersionId,start_time,end_time,prediction,trenchCategory,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeatures,
FROM `prj-prod-dataplatform.audit_balance.ml_model_run_details`
where modelDisplayName in ('Alpha-Cash-Stack-Model', 'alpha_stack_model_cash')
and modelVersionId = 'v1'
),
model_run as (
select customerId,digitalLoanAccountId,modelName, publish_time,requestPayload as requestPayload_clean
from `prj-prod-dataplatform.audit_balance.ml_request_details` 
WHERE modelName = 'Alpha-Cash-Model-response'
QUALIFY ROW_NUMBER() OVER (PARTITION BY customerId, digitalLoanAccountId,modelName ORDER BY publish_time DESC ) = 1)
, 
  modelname as (
  select p.customerId,
  p.digitalLoanAccountId,
  p.start_time,
  p.prediction aStackScore,
  case when p.modelDisplayName like 'Alpha%' then 'alpha_stack_model_cash' else p.modelDisplayName end as modelDisplayName,
  p.modelVersionId,
  coalesce (p.trenchCategory, REGEXP_EXTRACT(m.requestPayload_clean, r"trenchCategory[:=]['\"]?([^'\"]+)['\"]?")) trenchCategory
  from parsed p
  left join model_run m on m.digitalLoanAccountId = p.digitalLoanAccountId
  ),
  deliquency as
(select loanAccountNumber,
case when obs_min_inst_def0 >= 1 and min_inst_def0 = 1 then 1 else 0 end deffpd0,
case when obs_min_inst_def10 >=1 and min_inst_def10 =1 then 1 else 0 end deffpd10,
case when obs_min_inst_def30 >=1 and min_inst_def30 =1 then 1 else 0 end deffpd30,
case when obs_min_inst_def30 >=2 and min_inst_def30 in (1,2) then 1 else 0 end deffspd30,
case when obs_min_inst_def30 >=3 and min_inst_def30 in (1,2,3) then 1 else 0 end deffstpd30,
case when obs_min_inst_def0 >= 1 then 1 else 0 end flg_mature_fpd0,
case when obs_min_inst_def10 >=1 then 1 else 0 end flg_mature_fpd10,
case when obs_min_inst_def30 >=1 then 1 else 0 end flg_mature_fpd30,
case when obs_min_inst_def30 >=2 then 1 else 0 end flg_mature_fspd_30,
case when obs_min_inst_def30 >=3 then 1 else 0 end flg_mature_fstpd_30
from prj-prod-dataplatform.risk_credit_mis.loan_deliquency_data),
base as 
  (select distinct r.customerId,
  r.digitalLoanAccountId,
  loanmaster.loanAccountNumber,
  r.modelDisplayName,
  r.aStackScore,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month, 
  'Test' Data_selection,  
  del.deffpd30,
  del.flg_mature_fpd30,
  loanmaster.new_loan_type,
  modelVersionId, trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile' 
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall' 
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
  from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  inner join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
   left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
  where loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and r.aStackScore is not null
  and del.flg_mature_fpd30 = 1
  )
  select *  from base
  ;
"""
dfd = client.query(sq).to_dataframe()
dfd = dfd.drop_duplicates(keep="first")
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()


# %%
df1 = dfd.copy()

# %% [markdown]
# ### Train

# %%
sq = """ 
WITH parsed as (
select customerId, digitalLoanAccountId,modelDisplayName,modelVersionId,start_time,end_time,prediction,trenchCategory,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeatures,
FROM prj-prod-dataplatform.dap_ds_poweruser_playground.ml_training_model_run_details
where modelDisplayName in ('Alpha-Cash-Stack-Model', 'alpha_stack_model_cash')
and modelVersionId = 'v1'
),
  modelname as (
  select customerId,
  digitalLoanAccountId,
  start_time,
  prediction aStackScore,
  case when p.modelDisplayName like 'Alpha%' then 'alpha_stack_model_cash' else p.modelDisplayName end as modelDisplayName,
  p.modelVersionId,
  trenchCategory
  from parsed p
  ) ,
  deliquency as
(select loanAccountNumber,
case when obs_min_inst_def0 >= 1 and min_inst_def0 = 1 then 1 else 0 end deffpd0,
case when obs_min_inst_def10 >=1 and min_inst_def10 =1 then 1 else 0 end deffpd10,
case when obs_min_inst_def30 >=1 and min_inst_def30 =1 then 1 else 0 end deffpd30,
case when obs_min_inst_def30 >=2 and min_inst_def30 in (1,2) then 1 else 0 end deffspd30,
case when obs_min_inst_def30 >=3 and min_inst_def30 in (1,2,3) then 1 else 0 end deffstpd30,
case when obs_min_inst_def0 >= 1 then 1 else 0 end flg_mature_fpd0,
case when obs_min_inst_def10 >=1 then 1 else 0 end flg_mature_fpd10,
case when obs_min_inst_def30 >=1 then 1 else 0 end flg_mature_fpd30,
case when obs_min_inst_def30 >=2 then 1 else 0 end flg_mature_fspd_30,
case when obs_min_inst_def30 >=3 then 1 else 0 end flg_mature_fstpd_30
from prj-prod-dataplatform.risk_credit_mis.loan_deliquency_data),
base as
(select distinct r.customerId,
  r.digitalLoanAccountId,
  loanmaster.loanAccountNumber,
  r.modelDisplayName,
  r.aStackScore,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month, 
  'Train' Data_selection,  
    del.deffpd30,
  del.flg_mature_fpd30,
  loanmaster.new_loan_type,
  modelVersionId,
    trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile' 
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall' 
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
    from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  inner join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
  left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
  where loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and r.aStackScore is not null
  and del.flg_mature_fpd30 = 1
  )
  select * from base;
  """
dfd = client.query(sq).to_dataframe()
dfd = dfd.drop_duplicates(keep="first")
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()

# %%
df2 = dfd.copy()

# %% [markdown]
# ### Concat

# %%
# df_concat = pd.concat([df1, df2], ignore_index=True)
# print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")

# df_concat = (df_concat
#              .sort_values(by=['digitalLoanAccountId', 'Data_selection'],
#                           key=lambda s: s.map({'Train': 0, 'Test': 1}))
#              .drop_duplicates(subset=['digitalLoanAccountId'], keep='first')
#              .reset_index(drop=True))
# print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")
# df_concat.head()


# 1) Get all IDs present in Train
train_ids = set(df2["digitalLoanAccountId"])

# 2) Keep only Test rows whose ID is NOT in Train
df1_no_dupes = df1[~df1["digitalLoanAccountId"].isin(train_ids)]

# 3) Concatenate
df_concat = pd.concat([df1_no_dupes, df2], ignore_index=True)

print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")
df_concat.head()


# %%
df_concat["aStackScore"] = pd.to_numeric(df_concat["aStackScore"], errors="coerce")

# %%
fact_table, dimension_table = calculate_periodic_gini_prod_ver_trench_dimfact(
    df_concat,
    "aStackScore",
    "deffpd30",
    "FPD30",
    model_version_column="modelVersionId",
    trench_column="trenchCategory",
    loan_type_column="new_loan_type",
    loan_product_type_column="loan_product_type",
    account_id_column="digitalLoanAccountId",
)

# %%
fact_table, dimension_table = update_tables(
    fact_table, dimension_table, model_name="alpha_stack_model_cash", product="CASH"
)

# %%
# Upload to BigQuery
table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_table"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(fact_table, table_id, job_config=job_config)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_table"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(dimension_table, table_id, job_config=job_config)
job.result()  # Wait for the job to complete

# %%


# %% [markdown]
# ### FSPD30

# %% [markdown]
# ### Test

# %%
sq = r""" 
WITH parsed as (
select customerId, digitalLoanAccountId,modelDisplayName,modelVersionId,start_time,end_time,prediction,trenchCategory,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeatures,
FROM `prj-prod-dataplatform.audit_balance.ml_model_run_details`
where modelDisplayName in ('Alpha-Cash-Stack-Model', 'alpha_stack_model_cash')
and modelVersionId = 'v1'
),
model_run as (
select customerId,digitalLoanAccountId,modelName, publish_time,requestPayload as requestPayload_clean
from `prj-prod-dataplatform.audit_balance.ml_request_details` 
WHERE modelName = 'Alpha-Cash-Model-response'
QUALIFY ROW_NUMBER() OVER (PARTITION BY customerId, digitalLoanAccountId,modelName ORDER BY publish_time DESC ) = 1)
, 
  modelname as (
  select p.customerId,
  p.digitalLoanAccountId,
  p.start_time,
  p.prediction aStackScore,
  case when p.modelDisplayName like 'Alpha%' then 'alpha_stack_model_cash' else p.modelDisplayName end as modelDisplayName,
  p.modelVersionId,
  coalesce (p.trenchCategory, REGEXP_EXTRACT(m.requestPayload_clean, r"trenchCategory[:=]['\"]?([^'\"]+)['\"]?")) trenchCategory
  from parsed p
  left join model_run m on m.digitalLoanAccountId = p.digitalLoanAccountId
  ),
  deliquency as
(select loanAccountNumber,
case when obs_min_inst_def0 >= 1 and min_inst_def0 = 1 then 1 else 0 end deffpd0,
case when obs_min_inst_def10 >=1 and min_inst_def10 =1 then 1 else 0 end deffpd10,
case when obs_min_inst_def30 >=1 and min_inst_def30 =1 then 1 else 0 end deffpd30,
case when obs_min_inst_def30 >=2 and min_inst_def30 in (1,2) then 1 else 0 end deffspd30,
case when obs_min_inst_def30 >=3 and min_inst_def30 in (1,2,3) then 1 else 0 end deffstpd30,
case when obs_min_inst_def0 >= 1 then 1 else 0 end flg_mature_fpd0,
case when obs_min_inst_def10 >=1 then 1 else 0 end flg_mature_fpd10,
case when obs_min_inst_def30 >=1 then 1 else 0 end flg_mature_fpd30,
case when obs_min_inst_def30 >=2 then 1 else 0 end flg_mature_fspd_30,
case when obs_min_inst_def30 >=3 then 1 else 0 end flg_mature_fstpd_30
from prj-prod-dataplatform.risk_credit_mis.loan_deliquency_data),
base as 
  (select distinct r.customerId,
  r.digitalLoanAccountId,
  loanmaster.loanAccountNumber,
  r.modelDisplayName,
  r.aStackScore,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month, 
  'Test' Data_selection,  
  del.deffspd30,
  del.flg_mature_fspd_30,
  loanmaster.new_loan_type,
  modelVersionId, trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile' 
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall' 
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
  from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  inner join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
   left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
  where loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and r.aStackScore is not null
  and del.flg_mature_fspd_30 = 1
  )
  select *  from base
  ;
"""
dfd = client.query(sq).to_dataframe()
dfd = dfd.drop_duplicates(keep="first")
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()


# %%
df1 = dfd.copy()

# %% [markdown]
# ### Train

# %%
sq = """ 
WITH parsed as (
select customerId, digitalLoanAccountId,modelDisplayName,modelVersionId,start_time,end_time,prediction,trenchCategory,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeatures,
FROM prj-prod-dataplatform.dap_ds_poweruser_playground.ml_training_model_run_details
where modelDisplayName in ('Alpha-Cash-Stack-Model', 'alpha_stack_model_cash')
and modelVersionId = 'v1'
),
  modelname as (
  select customerId,
  digitalLoanAccountId,
  start_time,
  prediction aStackScore,
  case when p.modelDisplayName like 'Alpha%' then 'alpha_stack_model_cash' else p.modelDisplayName end as modelDisplayName,
  p.modelVersionId,
  trenchCategory
  from parsed p
  ) 
  , 
  deliquency as
(select loanAccountNumber,
case when obs_min_inst_def0 >= 1 and min_inst_def0 = 1 then 1 else 0 end deffpd0,
case when obs_min_inst_def10 >=1 and min_inst_def10 =1 then 1 else 0 end deffpd10,
case when obs_min_inst_def30 >=1 and min_inst_def30 =1 then 1 else 0 end deffpd30,
case when obs_min_inst_def30 >=2 and min_inst_def30 in (1,2) then 1 else 0 end deffspd30,
case when obs_min_inst_def30 >=3 and min_inst_def30 in (1,2,3) then 1 else 0 end deffstpd30,
case when obs_min_inst_def0 >= 1 then 1 else 0 end flg_mature_fpd0,
case when obs_min_inst_def10 >=1 then 1 else 0 end flg_mature_fpd10,
case when obs_min_inst_def30 >=1 then 1 else 0 end flg_mature_fpd30,
case when obs_min_inst_def30 >=2 then 1 else 0 end flg_mature_fspd_30,
case when obs_min_inst_def30 >=3 then 1 else 0 end flg_mature_fstpd_30
from prj-prod-dataplatform.risk_credit_mis.loan_deliquency_data),
base as
(select distinct r.customerId,
  r.digitalLoanAccountId,
  loanmaster.loanAccountNumber,
  r.modelDisplayName,
  r.aStackScore,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month, 
  'Train' Data_selection,  
    del.deffspd30,
  del.flg_mature_fspd_30,
  loanmaster.new_loan_type,
  modelVersionId,
    trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile' 
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall' 
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
    from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  inner join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
  left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
  where loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and r.aStackScore is not null
  and del.flg_mature_fspd_30 = 1
  )
  select * from base;
  """
dfd = client.query(sq).to_dataframe()
dfd = dfd.drop_duplicates(keep="first")
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()

# %%
df2 = dfd.copy()

# %% [markdown]
# ### Concat

# %%
# df_concat = pd.concat([df1, df2], ignore_index=True)
# print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")

# df_concat = (df_concat
#              .sort_values(by=['digitalLoanAccountId', 'Data_selection'],
#                           key=lambda s: s.map({'Train': 0, 'Test': 1}))
#              .drop_duplicates(subset=['digitalLoanAccountId'], keep='first')
#              .reset_index(drop=True))
# print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")
# df_concat.head()


# 1) Get all IDs present in Train
train_ids = set(df2["digitalLoanAccountId"])

# 2) Keep only Test rows whose ID is NOT in Train
df1_no_dupes = df1[~df1["digitalLoanAccountId"].isin(train_ids)]

# 3) Concatenate
df_concat = pd.concat([df1_no_dupes, df2], ignore_index=True)

print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")
df_concat.head()


# %%
df_concat["aStackScore"] = pd.to_numeric(df_concat["aStackScore"], errors="coerce")

# %%
fact_table, dimension_table = calculate_periodic_gini_prod_ver_trench_dimfact(
    df_concat,
    "aStackScore",
    "deffspd30",
    "FSPD30",
    model_version_column="modelVersionId",
    trench_column="trenchCategory",
    loan_type_column="new_loan_type",
    loan_product_type_column="loan_product_type",
    account_id_column="digitalLoanAccountId",
)

# %%
fact_table, dimension_table = update_tables(
    fact_table, dimension_table, model_name="alpha_stack_model_cash", product="CASH"
)

# %%
# Upload to BigQuery
table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_table"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(fact_table, table_id, job_config=job_config)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_table"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(dimension_table, table_id, job_config=job_config)
job.result()  # Wait for the job to complete

# %%


# %% [markdown]
# ### FSTPD30

# %% [markdown]
# ### Test

# %%
sq = r""" 
WITH parsed as (
select customerId, digitalLoanAccountId,modelDisplayName,modelVersionId,start_time,end_time,prediction,trenchCategory,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeatures,
FROM `prj-prod-dataplatform.audit_balance.ml_model_run_details`
where modelDisplayName in ('Alpha-Cash-Stack-Model', 'alpha_stack_model_cash')
and modelVersionId = 'v1'
),
model_run as (
select customerId,digitalLoanAccountId,modelName, publish_time,requestPayload as requestPayload_clean
from `prj-prod-dataplatform.audit_balance.ml_request_details` 
WHERE modelName = 'Alpha-Cash-Model-response'
QUALIFY ROW_NUMBER() OVER (PARTITION BY customerId, digitalLoanAccountId,modelName ORDER BY publish_time DESC ) = 1)
, 
  modelname as (
  select p.customerId,
  p.digitalLoanAccountId,
  p.start_time,
  p.prediction aStackScore,
  case when p.modelDisplayName like 'Alpha%' then 'alpha_stack_model_cash' else p.modelDisplayName end as modelDisplayName,
  p.modelVersionId,
  coalesce (p.trenchCategory, REGEXP_EXTRACT(m.requestPayload_clean, r"trenchCategory[:=]['\"]?([^'\"]+)['\"]?")) trenchCategory
  from parsed p
  left join model_run m on m.digitalLoanAccountId = p.digitalLoanAccountId
  ),
  deliquency as
(select loanAccountNumber,
case when obs_min_inst_def0 >= 1 and min_inst_def0 = 1 then 1 else 0 end deffpd0,
case when obs_min_inst_def10 >=1 and min_inst_def10 =1 then 1 else 0 end deffpd10,
case when obs_min_inst_def30 >=1 and min_inst_def30 =1 then 1 else 0 end deffpd30,
case when obs_min_inst_def30 >=2 and min_inst_def30 in (1,2) then 1 else 0 end deffspd30,
case when obs_min_inst_def30 >=3 and min_inst_def30 in (1,2,3) then 1 else 0 end deffstpd30,
case when obs_min_inst_def0 >= 1 then 1 else 0 end flg_mature_fpd0,
case when obs_min_inst_def10 >=1 then 1 else 0 end flg_mature_fpd10,
case when obs_min_inst_def30 >=1 then 1 else 0 end flg_mature_fpd30,
case when obs_min_inst_def30 >=2 then 1 else 0 end flg_mature_fspd_30,
case when obs_min_inst_def30 >=3 then 1 else 0 end flg_mature_fstpd_30
from prj-prod-dataplatform.risk_credit_mis.loan_deliquency_data),
base as 
  (select distinct r.customerId,
  r.digitalLoanAccountId,
  loanmaster.loanAccountNumber,
  r.modelDisplayName,
  r.aStackScore,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month, 
  'Test' Data_selection,  
  del.deffstpd30,
  del.flg_mature_fstpd_30,
  loanmaster.new_loan_type,
  modelVersionId, trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile' 
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall' 
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
  from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  inner join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
   left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
  where loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and r.aStackScore is not null
  and del.flg_mature_fstpd_30 = 1
  )
  select *  from base
  ;
"""
dfd = client.query(sq).to_dataframe()
dfd = dfd.drop_duplicates(keep="first")
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()


# %%
df1 = dfd.copy()

# %% [markdown]
# ### Train

# %%
sq = """ 
WITH parsed as (
select customerId, digitalLoanAccountId,modelDisplayName,modelVersionId,start_time,end_time,prediction,trenchCategory,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeatures,
FROM prj-prod-dataplatform.dap_ds_poweruser_playground.ml_training_model_run_details
where modelDisplayName in ('Alpha-Cash-Stack-Model', 'alpha_stack_model_cash')
and modelVersionId = 'v1'
),
  modelname as (
  select customerId,
  digitalLoanAccountId,
  start_time,
  prediction aStackScore,
  case when p.modelDisplayName like 'Alpha%' then 'alpha_stack_model_cash' else p.modelDisplayName end as modelDisplayName,
  p.modelVersionId,
  trenchCategory
  from parsed p
  ) , 
  deliquency as
(select loanAccountNumber,
case when obs_min_inst_def0 >= 1 and min_inst_def0 = 1 then 1 else 0 end deffpd0,
case when obs_min_inst_def10 >=1 and min_inst_def10 =1 then 1 else 0 end deffpd10,
case when obs_min_inst_def30 >=1 and min_inst_def30 =1 then 1 else 0 end deffpd30,
case when obs_min_inst_def30 >=2 and min_inst_def30 in (1,2) then 1 else 0 end deffspd30,
case when obs_min_inst_def30 >=3 and min_inst_def30 in (1,2,3) then 1 else 0 end deffstpd30,
case when obs_min_inst_def0 >= 1 then 1 else 0 end flg_mature_fpd0,
case when obs_min_inst_def10 >=1 then 1 else 0 end flg_mature_fpd10,
case when obs_min_inst_def30 >=1 then 1 else 0 end flg_mature_fpd30,
case when obs_min_inst_def30 >=2 then 1 else 0 end flg_mature_fspd_30,
case when obs_min_inst_def30 >=3 then 1 else 0 end flg_mature_fstpd_30
from prj-prod-dataplatform.risk_credit_mis.loan_deliquency_data),
base as
(select distinct r.customerId,
  r.digitalLoanAccountId,
  loanmaster.loanAccountNumber,
  r.modelDisplayName,
  r.aStackScore,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month, 
  'Train' Data_selection,  
    del.deffstpd30,
  del.flg_mature_fstpd_30,
  loanmaster.new_loan_type,
  modelVersionId,
    trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile' 
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall' 
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
    from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  inner join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
  left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
  where loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and r.aStackScore is not null
  and del.flg_mature_fstpd_30 = 1
  )
  select * from base;
  """
dfd = client.query(sq).to_dataframe()
dfd = dfd.drop_duplicates(keep="first")
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()

# %%
df2 = dfd.copy()

# %% [markdown]
# ### Concat

# %%
# df_concat = pd.concat([df1, df2], ignore_index=True)
# print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")

# df_concat = (df_concat
#              .sort_values(by=['digitalLoanAccountId', 'Data_selection'],
#                           key=lambda s: s.map({'Train': 0, 'Test': 1}))
#              .drop_duplicates(subset=['digitalLoanAccountId'], keep='first')
#              .reset_index(drop=True))
# print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")
# df_concat.head()


# 1) Get all IDs present in Train
train_ids = set(df2["digitalLoanAccountId"])

# 2) Keep only Test rows whose ID is NOT in Train
df1_no_dupes = df1[~df1["digitalLoanAccountId"].isin(train_ids)]

# 3) Concatenate
df_concat = pd.concat([df1_no_dupes, df2], ignore_index=True)

print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")
df_concat.head()


# %%
df_concat["aStackScore"] = pd.to_numeric(df_concat["aStackScore"], errors="coerce")

# %%
fact_table, dimension_table = calculate_periodic_gini_prod_ver_trench_dimfact(
    df_concat,
    "aStackScore",
    "deffstpd30",
    "FSTPD30",
    model_version_column="modelVersionId",
    trench_column="trenchCategory",
    loan_type_column="new_loan_type",
    loan_product_type_column="loan_product_type",
    account_id_column="digitalLoanAccountId",
)

# %%
fact_table, dimension_table = update_tables(
    fact_table, dimension_table, model_name="alpha_stack_model_cash", product="CASH"
)
print(f"The shape of the fact table is:\t {fact_table.shape}")
print(f"The shape of the dimension table is:\t {dimension_table.shape}")

# %%
# Upload to BigQuery
table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_table"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(fact_table, table_id, job_config=job_config)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_table"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(dimension_table, table_id, job_config=job_config)
job.result()  # Wait for the job to complete

# %% [markdown]
# # Beta-Cash-Demo-Model

# %% [markdown]
# ### FPD0

# %% [markdown]
# ### Test

# %%
sq = """ 
WITH parsed as (
select customerId, digitalLoanAccountId,modelDisplayName,modelVersionId,start_time,end_time,prediction,trenchCategory,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeatures,
FROM `prj-prod-dataplatform.audit_balance.ml_model_run_details`
where modelDisplayName in ('Beta-Cash-Demo-Model', 'beta_demo_model_cash')
),
latest_request as (
select * from parsed
QUALIFY ROW_NUMBER() OVER (PARTITION BY customerId, digitalLoanAccountId,modelDisplayName ORDER BY start_time DESC ) = 1),
model_run as (
select customerId,digitalLoanAccountId,modelName, publish_time,
REPLACE(REPLACE(requestPayload, "'", '"'), "None", "null") AS requestPayload_clean
from `prj-prod-dataplatform.audit_balance.ml_request_details` 
WHERE modelName = 'Beta-Cash-Model-response'
QUALIFY ROW_NUMBER() OVER (PARTITION BY customerId, digitalLoanAccountId,modelName ORDER BY publish_time DESC ) = 1)
, 
  modelname as (
  select p.customerId,
  p.digitalLoanAccountId,
  start_time,
  prediction Beta_Cash_Demo_Score,
  coalesce(p.trenchCategory, JSON_VALUE(m.requestPayload_clean, "$.predictions.trenchCategory")) AS trenchCategory,
  case when p.modelDisplayName like 'Beta-Cash-Demo-Model' then 'beta_demo_model_cash' else p.modelDisplayName end as modelDisplayName,
  p.modelVersionId
  from latest_request p 
  left join model_run m on m.digitalLoanAccountId = p.digitalLoanAccountId
  ), 
  deliquency as
(select loanAccountNumber,
case when obs_min_inst_def0 >= 1 and min_inst_def0 = 1 then 1 else 0 end deffpd0,
case when obs_min_inst_def10 >=1 and min_inst_def10 =1 then 1 else 0 end deffpd10,
case when obs_min_inst_def30 >=1 and min_inst_def30 =1 then 1 else 0 end deffpd30,
case when obs_min_inst_def30 >=2 and min_inst_def30 in (1,2) then 1 else 0 end deffspd30,
case when obs_min_inst_def30 >=3 and min_inst_def30 in (1,2,3) then 1 else 0 end deffstpd30,
case when obs_min_inst_def0 >= 1 then 1 else 0 end flg_mature_fpd0,
case when obs_min_inst_def10 >=1 then 1 else 0 end flg_mature_fpd10,
case when obs_min_inst_def30 >=1 then 1 else 0 end flg_mature_fpd30,
case when obs_min_inst_def30 >=2 then 1 else 0 end flg_mature_fspd_30,
case when obs_min_inst_def30 >=3 then 1 else 0 end flg_mature_fstpd_30
from prj-prod-dataplatform.risk_credit_mis.loan_deliquency_data),
base as 
  (select distinct r.customerId,
  r.digitalLoanAccountId,
  loanmaster.loanAccountNumber,
  r.modelDisplayName,
  r.Beta_Cash_Demo_Score,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month, 
  'Test' Data_selection,  
  deffpd0,
  flg_mature_fpd0,
  loanmaster.new_loan_type,
  modelVersionId, trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile' 
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall' 
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
  from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  inner join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
   left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
  where loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and r.Beta_Cash_Demo_Score is not null
  and flg_mature_fpd0 = 1
  )
  select *  from base
  ;
"""
dfd = client.query(sq).to_dataframe()
dfd = dfd.drop_duplicates(keep="first")
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()


# %%
df1 = dfd.copy()

# %% [markdown]
# ### Train

# %%
sq = """ 
WITH parsed as (
select customerId, digitalLoanAccountId,modelDisplayName,modelVersionId,start_time,end_time,prediction,trenchCategory,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeatures,
FROM prj-prod-dataplatform.dap_ds_poweruser_playground.ml_training_model_run_details
where modelDisplayName in ('Beta-Cash-Demo-Model', 'beta_demo_model_cash')
),
  modelname as (
  select customerId,
  digitalLoanAccountId,
  start_time,
  prediction Beta_Cash_Demo_Score,
  case when modelDisplayName like 'Beta-Cash-Demo-Model' then 'beta_demo_model_cash' else modelDisplayName end as modelDisplayName,
  modelVersionId,
  trenchCategory
  from parsed
  ) 
  ,
  deliquency as
(select loanAccountNumber,
case when obs_min_inst_def0 >= 1 and min_inst_def0 = 1 then 1 else 0 end deffpd0,
case when obs_min_inst_def10 >=1 and min_inst_def10 =1 then 1 else 0 end deffpd10,
case when obs_min_inst_def30 >=1 and min_inst_def30 =1 then 1 else 0 end deffpd30,
case when obs_min_inst_def30 >=2 and min_inst_def30 in (1,2) then 1 else 0 end deffspd30,
case when obs_min_inst_def30 >=3 and min_inst_def30 in (1,2,3) then 1 else 0 end deffstpd30,
case when obs_min_inst_def0 >= 1 then 1 else 0 end flg_mature_fpd0,
case when obs_min_inst_def10 >=1 then 1 else 0 end flg_mature_fpd10,
case when obs_min_inst_def30 >=1 then 1 else 0 end flg_mature_fpd30,
case when obs_min_inst_def30 >=2 then 1 else 0 end flg_mature_fspd_30,
case when obs_min_inst_def30 >=3 then 1 else 0 end flg_mature_fstpd_30
from prj-prod-dataplatform.risk_credit_mis.loan_deliquency_data),
base as
(select distinct r.customerId,
  r.digitalLoanAccountId,
  loanmaster.loanAccountNumber,
  r.modelDisplayName,
  r.Beta_Cash_Demo_Score,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month, 
  'Train' Data_selection,  
    deffpd0,
  flg_mature_fpd0,
  loanmaster.new_loan_type,
  modelVersionId,
    trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile' 
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall' 
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
    from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  inner join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
  left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
  where loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and r.Beta_Cash_Demo_Score is not null
  and flg_mature_fpd0 = 1
  )
  select * from base;
  """
dfd = client.query(sq).to_dataframe()
dfd = dfd.drop_duplicates(keep="first")
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()

# %%
df2 = dfd.copy()

# %% [markdown]
# ### Concat

# %%
# df_concat = pd.concat([df1, df2], ignore_index=True)
# print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")

# df_concat = (df_concat
#              .sort_values(by=['digitalLoanAccountId', 'Data_selection'],
#                           key=lambda s: s.map({'Train': 0, 'Test': 1}))
#              .drop_duplicates(subset=['digitalLoanAccountId'], keep='first')
#              .reset_index(drop=True))
# print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")
# df_concat.head()


# 1) Get all IDs present in Train
train_ids = set(df2["digitalLoanAccountId"])

# 2) Keep only Test rows whose ID is NOT in Train
df1_no_dupes = df1[~df1["digitalLoanAccountId"].isin(train_ids)]

# 3) Concatenate
df_concat = pd.concat([df1_no_dupes, df2], ignore_index=True)

print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")
df_concat.head()


# %%
df_concat["Beta_Cash_Demo_Score"] = pd.to_numeric(
    df_concat["Beta_Cash_Demo_Score"], errors="coerce"
)

# %%
fact_table, dimension_table = calculate_periodic_gini_prod_ver_trench_dimfact(
    df_concat,
    "Beta_Cash_Demo_Score",
    "deffpd0",
    "FPD0",
    model_version_column="modelVersionId",
    trench_column="trenchCategory",
    loan_type_column="new_loan_type",
    loan_product_type_column="loan_product_type",
    account_id_column="digitalLoanAccountId",
)

# %%
fact_table, dimension_table = update_tables(
    fact_table, dimension_table, model_name="beta_demo_model_cash", product="CASH"
)
print(f"The shape of the fact table is:\t {fact_table.shape}")
print(f"The shape of the dimension table is:\t {dimension_table.shape}")

# %%
# Upload to BigQuery
table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_table"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(fact_table, table_id, job_config=job_config)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_table"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(dimension_table, table_id, job_config=job_config)
job.result()  # Wait for the job to complete

# %% [markdown]
# ### FPD10

# %% [markdown]
# ### Test

# %%
sq = """ 
WITH parsed as (
select customerId, digitalLoanAccountId,modelDisplayName,modelVersionId,start_time,end_time,prediction,trenchCategory,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeatures,
FROM `prj-prod-dataplatform.audit_balance.ml_model_run_details`
where modelDisplayName in ('Beta-Cash-Demo-Model', 'beta_demo_model_cash')
),
latest_request as (
select * from parsed
QUALIFY ROW_NUMBER() OVER (PARTITION BY customerId, digitalLoanAccountId,modelDisplayName ORDER BY start_time DESC ) = 1),
model_run as (
select customerId,digitalLoanAccountId,modelName, publish_time,
REPLACE(REPLACE(requestPayload, "'", '"'), "None", "null") AS requestPayload_clean
from `prj-prod-dataplatform.audit_balance.ml_request_details` 
WHERE modelName = 'Beta-Cash-Model-response'
QUALIFY ROW_NUMBER() OVER (PARTITION BY customerId, digitalLoanAccountId,modelName ORDER BY publish_time DESC ) = 1)
, 
  modelname as (
  select p.customerId,
  p.digitalLoanAccountId,
  start_time,
  prediction Beta_Cash_Demo_Score,
  coalesce(p.trenchCategory, JSON_VALUE(m.requestPayload_clean, "$.predictions.trenchCategory")) AS trenchCategory,
  case when p.modelDisplayName like 'Beta-Cash-Demo-Model' then 'beta_demo_model_cash' else p.modelDisplayName end as modelDisplayName,
  p.modelVersionId
  from latest_request p 
  left join model_run m on m.digitalLoanAccountId = p.digitalLoanAccountId
  ), 
  deliquency as
(select loanAccountNumber,
case when obs_min_inst_def0 >= 1 and min_inst_def0 = 1 then 1 else 0 end deffpd0,
case when obs_min_inst_def10 >=1 and min_inst_def10 =1 then 1 else 0 end deffpd10,
case when obs_min_inst_def30 >=1 and min_inst_def30 =1 then 1 else 0 end deffpd30,
case when obs_min_inst_def30 >=2 and min_inst_def30 in (1,2) then 1 else 0 end deffspd30,
case when obs_min_inst_def30 >=3 and min_inst_def30 in (1,2,3) then 1 else 0 end deffstpd30,
case when obs_min_inst_def0 >= 1 then 1 else 0 end flg_mature_fpd0,
case when obs_min_inst_def10 >=1 then 1 else 0 end flg_mature_fpd10,
case when obs_min_inst_def30 >=1 then 1 else 0 end flg_mature_fpd30,
case when obs_min_inst_def30 >=2 then 1 else 0 end flg_mature_fspd_30,
case when obs_min_inst_def30 >=3 then 1 else 0 end flg_mature_fstpd_30
from prj-prod-dataplatform.risk_credit_mis.loan_deliquency_data),
base as 
  (select distinct r.customerId,
  r.digitalLoanAccountId,
  loanmaster.loanAccountNumber,
  r.modelDisplayName,
  r.Beta_Cash_Demo_Score,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month, 
  'Test' Data_selection,  
  del.deffpd10,
  del.flg_mature_fpd10,
  loanmaster.new_loan_type,
  modelVersionId, trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile' 
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall' 
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
  from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  inner join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
   left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
  where loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and r.Beta_Cash_Demo_Score is not null
  and del.flg_mature_fpd10 = 1
  )
  select *  from base
  ;
"""
dfd = client.query(sq).to_dataframe()
dfd = dfd.drop_duplicates(keep="first")
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()


# %%
df1 = dfd.copy()

# %% [markdown]
# ### Train

# %%
sq = """ 
WITH parsed as (
select customerId, digitalLoanAccountId,modelDisplayName,modelVersionId,start_time,end_time,prediction,trenchCategory,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeatures,
FROM prj-prod-dataplatform.dap_ds_poweruser_playground.ml_training_model_run_details
where modelDisplayName in ('Beta-Cash-Demo-Model', 'beta_demo_model_cash')
),
  modelname as (
  select customerId,
  digitalLoanAccountId,
  start_time,
  prediction Beta_Cash_Demo_Score,
  case when modelDisplayName like 'Beta-Cash-Demo-Model' then 'beta_demo_model_cash' else modelDisplayName end as modelDisplayName,
  modelVersionId,
  trenchCategory
  from parsed
  )
  ,
  deliquency as
(select loanAccountNumber,
case when obs_min_inst_def0 >= 1 and min_inst_def0 = 1 then 1 else 0 end deffpd0,
case when obs_min_inst_def10 >=1 and min_inst_def10 =1 then 1 else 0 end deffpd10,
case when obs_min_inst_def30 >=1 and min_inst_def30 =1 then 1 else 0 end deffpd30,
case when obs_min_inst_def30 >=2 and min_inst_def30 in (1,2) then 1 else 0 end deffspd30,
case when obs_min_inst_def30 >=3 and min_inst_def30 in (1,2,3) then 1 else 0 end deffstpd30,
case when obs_min_inst_def0 >= 1 then 1 else 0 end flg_mature_fpd0,
case when obs_min_inst_def10 >=1 then 1 else 0 end flg_mature_fpd10,
case when obs_min_inst_def30 >=1 then 1 else 0 end flg_mature_fpd30,
case when obs_min_inst_def30 >=2 then 1 else 0 end flg_mature_fspd_30,
case when obs_min_inst_def30 >=3 then 1 else 0 end flg_mature_fstpd_30
from prj-prod-dataplatform.risk_credit_mis.loan_deliquency_data),
base as
(select distinct r.customerId,
  r.digitalLoanAccountId,
  loanmaster.loanAccountNumber,
  r.modelDisplayName,
  r.Beta_Cash_Demo_Score,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month, 
  'Train' Data_selection,  
    del.deffpd10,
  del.flg_mature_fpd10,
  loanmaster.new_loan_type,
  modelVersionId,
    trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile' 
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall' 
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
    from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  inner join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
  left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
  where loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and r.Beta_Cash_Demo_Score is not null
  and del.flg_mature_fpd10 = 1
  )
  select * from base;
  """
dfd = client.query(sq).to_dataframe()
dfd = dfd.drop_duplicates(keep="first")
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()

# %%
df2 = dfd.copy()

# %% [markdown]
# ### Concat

# %%
# df_concat = pd.concat([df1, df2], ignore_index=True)
# print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")

# df_concat = (df_concat
#              .sort_values(by=['digitalLoanAccountId', 'Data_selection'],
#                           key=lambda s: s.map({'Train': 0, 'Test': 1}))
#              .drop_duplicates(subset=['digitalLoanAccountId'], keep='first')
#              .reset_index(drop=True))
# print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")
# df_concat.head()


# 1) Get all IDs present in Train
train_ids = set(df2["digitalLoanAccountId"])

# 2) Keep only Test rows whose ID is NOT in Train
df1_no_dupes = df1[~df1["digitalLoanAccountId"].isin(train_ids)]

# 3) Concatenate
df_concat = pd.concat([df1_no_dupes, df2], ignore_index=True)

print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")
df_concat.head()


# %%
df_concat["Beta_Cash_Demo_Score"] = pd.to_numeric(
    df_concat["Beta_Cash_Demo_Score"], errors="coerce"
)

# %%
fact_table, dimension_table = calculate_periodic_gini_prod_ver_trench_dimfact(
    df_concat,
    "Beta_Cash_Demo_Score",
    "deffpd10",
    "FPD10",
    model_version_column="modelVersionId",
    trench_column="trenchCategory",
    loan_type_column="new_loan_type",
    loan_product_type_column="loan_product_type",
    account_id_column="digitalLoanAccountId",
)

# %%
fact_table, dimension_table = update_tables(
    fact_table, dimension_table, model_name="beta_demo_model_cash", product="CASH"
)

# %%
# Upload to BigQuery
table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_table"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(fact_table, table_id, job_config=job_config)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_table"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(dimension_table, table_id, job_config=job_config)
job.result()  # Wait for the job to complete

# %%


# %% [markdown]
# ### FPD30

# %% [markdown]
# ### Test

# %%
sq = """ 
WITH parsed as (
select customerId, digitalLoanAccountId,modelDisplayName,modelVersionId,start_time,end_time,prediction,trenchCategory,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeatures,
FROM `prj-prod-dataplatform.audit_balance.ml_model_run_details`
where modelDisplayName in ('Beta-Cash-Demo-Model', 'beta_demo_model_cash')
),
latest_request as (
select * from parsed
QUALIFY ROW_NUMBER() OVER (PARTITION BY customerId, digitalLoanAccountId,modelDisplayName ORDER BY start_time DESC ) = 1),
model_run as (
select customerId,digitalLoanAccountId,modelName, publish_time,
REPLACE(REPLACE(requestPayload, "'", '"'), "None", "null") AS requestPayload_clean
from `prj-prod-dataplatform.audit_balance.ml_request_details` 
WHERE modelName = 'Beta-Cash-Model-response'
QUALIFY ROW_NUMBER() OVER (PARTITION BY customerId, digitalLoanAccountId,modelName ORDER BY publish_time DESC ) = 1)
, 
  modelname as (
  select p.customerId,
  p.digitalLoanAccountId,
  start_time,
  prediction Beta_Cash_Demo_Score,
  coalesce(p.trenchCategory, JSON_VALUE(m.requestPayload_clean, "$.predictions.trenchCategory")) AS trenchCategory,
  case when p.modelDisplayName like 'Beta-Cash-Demo-Model' then 'beta_demo_model_cash' else p.modelDisplayName end as modelDisplayName,
  p.modelVersionId
  from latest_request p 
  left join model_run m on m.digitalLoanAccountId = p.digitalLoanAccountId
  ), 
  deliquency as
(select loanAccountNumber,
case when obs_min_inst_def0 >= 1 and min_inst_def0 = 1 then 1 else 0 end deffpd0,
case when obs_min_inst_def10 >=1 and min_inst_def10 =1 then 1 else 0 end deffpd10,
case when obs_min_inst_def30 >=1 and min_inst_def30 =1 then 1 else 0 end deffpd30,
case when obs_min_inst_def30 >=2 and min_inst_def30 in (1,2) then 1 else 0 end deffspd30,
case when obs_min_inst_def30 >=3 and min_inst_def30 in (1,2,3) then 1 else 0 end deffstpd30,
case when obs_min_inst_def0 >= 1 then 1 else 0 end flg_mature_fpd0,
case when obs_min_inst_def10 >=1 then 1 else 0 end flg_mature_fpd10,
case when obs_min_inst_def30 >=1 then 1 else 0 end flg_mature_fpd30,
case when obs_min_inst_def30 >=2 then 1 else 0 end flg_mature_fspd_30,
case when obs_min_inst_def30 >=3 then 1 else 0 end flg_mature_fstpd_30
from prj-prod-dataplatform.risk_credit_mis.loan_deliquency_data),
base as 
  (select distinct r.customerId,
  r.digitalLoanAccountId,
  loanmaster.loanAccountNumber,
  r.modelDisplayName,
  r.Beta_Cash_Demo_Score,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month, 
  'Test' Data_selection,  
  del.deffpd30,
  del.flg_mature_fpd30,
  loanmaster.new_loan_type,
  modelVersionId, trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile' 
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall' 
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
  from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  inner join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
   left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
  where loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and r.Beta_Cash_Demo_Score is not null
  and del.flg_mature_fpd30 = 1
  )
  select *  from base
  ;
"""
dfd = client.query(sq).to_dataframe()
dfd = dfd.drop_duplicates(keep="first")
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()


# %%
df1 = dfd.copy()

# %% [markdown]
# ### Train

# %%
sq = """ 
WITH parsed as (
select customerId, digitalLoanAccountId,modelDisplayName,modelVersionId,start_time,end_time,prediction,trenchCategory,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeatures,
FROM prj-prod-dataplatform.dap_ds_poweruser_playground.ml_training_model_run_details
where modelDisplayName in ('Beta-Cash-Demo-Model', 'beta_demo_model_cash')
),
  modelname as (
  select customerId,
  digitalLoanAccountId,
  start_time,
  prediction Beta_Cash_Demo_Score,
  case when modelDisplayName like 'Beta-Cash-Demo-Model' then 'beta_demo_model_cash' else modelDisplayName end as modelDisplayName,
  modelVersionId,
  trenchCategory
  from parsed
  ) ,
  deliquency as
(select loanAccountNumber,
case when obs_min_inst_def0 >= 1 and min_inst_def0 = 1 then 1 else 0 end deffpd0,
case when obs_min_inst_def10 >=1 and min_inst_def10 =1 then 1 else 0 end deffpd10,
case when obs_min_inst_def30 >=1 and min_inst_def30 =1 then 1 else 0 end deffpd30,
case when obs_min_inst_def30 >=2 and min_inst_def30 in (1,2) then 1 else 0 end deffspd30,
case when obs_min_inst_def30 >=3 and min_inst_def30 in (1,2,3) then 1 else 0 end deffstpd30,
case when obs_min_inst_def0 >= 1 then 1 else 0 end flg_mature_fpd0,
case when obs_min_inst_def10 >=1 then 1 else 0 end flg_mature_fpd10,
case when obs_min_inst_def30 >=1 then 1 else 0 end flg_mature_fpd30,
case when obs_min_inst_def30 >=2 then 1 else 0 end flg_mature_fspd_30,
case when obs_min_inst_def30 >=3 then 1 else 0 end flg_mature_fstpd_30
from prj-prod-dataplatform.risk_credit_mis.loan_deliquency_data),
base as
(select distinct r.customerId,
  r.digitalLoanAccountId,
  loanmaster.loanAccountNumber,
  r.modelDisplayName,
  r.Beta_Cash_Demo_Score,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month, 
  'Train' Data_selection,  
    del.deffpd30,
  del.flg_mature_fpd30,
  loanmaster.new_loan_type,
  modelVersionId,
    trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile' 
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall' 
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
    from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  inner join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
  left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
  where loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and r.Beta_Cash_Demo_Score is not null
  and del.flg_mature_fpd30 = 1
  )
  select * from base;
  """
dfd = client.query(sq).to_dataframe()
dfd = dfd.drop_duplicates(keep="first")
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()

# %%
df2 = dfd.copy()

# %% [markdown]
# ### Concat

# %%
# df_concat = pd.concat([df1, df2], ignore_index=True)
# print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")

# df_concat = (df_concat
#              .sort_values(by=['digitalLoanAccountId', 'Data_selection'],
#                           key=lambda s: s.map({'Train': 0, 'Test': 1}))
#              .drop_duplicates(subset=['digitalLoanAccountId'], keep='first')
#              .reset_index(drop=True))
# print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")
# df_concat.head()


# 1) Get all IDs present in Train
train_ids = set(df2["digitalLoanAccountId"])

# 2) Keep only Test rows whose ID is NOT in Train
df1_no_dupes = df1[~df1["digitalLoanAccountId"].isin(train_ids)]

# 3) Concatenate
df_concat = pd.concat([df1_no_dupes, df2], ignore_index=True)

print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")
df_concat.head()


# %%
df_concat["Beta_Cash_Demo_Score"] = pd.to_numeric(
    df_concat["Beta_Cash_Demo_Score"], errors="coerce"
)

# %%
fact_table, dimension_table = calculate_periodic_gini_prod_ver_trench_dimfact(
    df_concat,
    "Beta_Cash_Demo_Score",
    "deffpd30",
    "FPD30",
    model_version_column="modelVersionId",
    trench_column="trenchCategory",
    loan_type_column="new_loan_type",
    loan_product_type_column="loan_product_type",
    account_id_column="digitalLoanAccountId",
)

# %%
fact_table, dimension_table = update_tables(
    fact_table, dimension_table, model_name="beta_demo_model_cash", product="CASH"
)

# %%
# Upload to BigQuery
table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_table"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(fact_table, table_id, job_config=job_config)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_table"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(dimension_table, table_id, job_config=job_config)
job.result()  # Wait for the job to complete

# %%


# %% [markdown]
# ### FSPD30

# %% [markdown]
# ### Test

# %%
sq = """ 
WITH parsed as (
select customerId, digitalLoanAccountId,modelDisplayName,modelVersionId,start_time,end_time,prediction,trenchCategory,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeatures,
FROM `prj-prod-dataplatform.audit_balance.ml_model_run_details`
where modelDisplayName in ('Beta-Cash-Demo-Model', 'beta_demo_model_cash')
),
latest_request as (
select * from parsed
QUALIFY ROW_NUMBER() OVER (PARTITION BY customerId, digitalLoanAccountId,modelDisplayName ORDER BY start_time DESC ) = 1),
model_run as (
select customerId,digitalLoanAccountId,modelName, publish_time,
REPLACE(REPLACE(requestPayload, "'", '"'), "None", "null") AS requestPayload_clean
from `prj-prod-dataplatform.audit_balance.ml_request_details` 
WHERE modelName = 'Beta-Cash-Model-response'
QUALIFY ROW_NUMBER() OVER (PARTITION BY customerId, digitalLoanAccountId,modelName ORDER BY publish_time DESC ) = 1)
, 
  modelname as (
  select p.customerId,
  p.digitalLoanAccountId,
  start_time,
  prediction Beta_Cash_Demo_Score,
  coalesce(p.trenchCategory, JSON_VALUE(m.requestPayload_clean, "$.predictions.trenchCategory")) AS trenchCategory,
  case when p.modelDisplayName like 'Beta-Cash-Demo-Model' then 'beta_demo_model_cash' else p.modelDisplayName end as modelDisplayName,
  p.modelVersionId
  from latest_request p 
  left join model_run m on m.digitalLoanAccountId = p.digitalLoanAccountId
  ), 
  deliquency as
(select loanAccountNumber,
case when obs_min_inst_def0 >= 1 and min_inst_def0 = 1 then 1 else 0 end deffpd0,
case when obs_min_inst_def10 >=1 and min_inst_def10 =1 then 1 else 0 end deffpd10,
case when obs_min_inst_def30 >=1 and min_inst_def30 =1 then 1 else 0 end deffpd30,
case when obs_min_inst_def30 >=2 and min_inst_def30 in (1,2) then 1 else 0 end deffspd30,
case when obs_min_inst_def30 >=3 and min_inst_def30 in (1,2,3) then 1 else 0 end deffstpd30,
case when obs_min_inst_def0 >= 1 then 1 else 0 end flg_mature_fpd0,
case when obs_min_inst_def10 >=1 then 1 else 0 end flg_mature_fpd10,
case when obs_min_inst_def30 >=1 then 1 else 0 end flg_mature_fpd30,
case when obs_min_inst_def30 >=2 then 1 else 0 end flg_mature_fspd_30,
case when obs_min_inst_def30 >=3 then 1 else 0 end flg_mature_fstpd_30
from prj-prod-dataplatform.risk_credit_mis.loan_deliquency_data),
base as 
  (select distinct r.customerId,
  r.digitalLoanAccountId,
  loanmaster.loanAccountNumber,
  r.modelDisplayName,
  r.Beta_Cash_Demo_Score,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month, 
  'Test' Data_selection,  
  del.deffspd30,
  del.flg_mature_fspd_30,
  loanmaster.new_loan_type,
  modelVersionId, trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile' 
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall' 
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
  from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  inner join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
   left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
  where loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and r.Beta_Cash_Demo_Score is not null
  and del.flg_mature_fspd_30 = 1
  )
  select *  from base
  ;
"""
dfd = client.query(sq).to_dataframe()
dfd = dfd.drop_duplicates(keep="first")
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()


# %%
df1 = dfd.copy()

# %% [markdown]
# ### Train

# %%
sq = """ 
WITH parsed as (
select customerId, digitalLoanAccountId,modelDisplayName,modelVersionId,start_time,end_time,prediction,trenchCategory,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeatures,
FROM prj-prod-dataplatform.dap_ds_poweruser_playground.ml_training_model_run_details
where modelDisplayName in ('Beta-Cash-Demo-Model', 'beta_demo_model_cash')
),
  modelname as (
  select customerId,
  digitalLoanAccountId,
  start_time,
  prediction Beta_Cash_Demo_Score,
  case when modelDisplayName like 'Beta-Cash-Demo-Model' then 'beta_demo_model_cash' else modelDisplayName end as modelDisplayName,
  modelVersionId,
  trenchCategory
  from parsed
  )
  , 
  deliquency as
(select loanAccountNumber,
case when obs_min_inst_def0 >= 1 and min_inst_def0 = 1 then 1 else 0 end deffpd0,
case when obs_min_inst_def10 >=1 and min_inst_def10 =1 then 1 else 0 end deffpd10,
case when obs_min_inst_def30 >=1 and min_inst_def30 =1 then 1 else 0 end deffpd30,
case when obs_min_inst_def30 >=2 and min_inst_def30 in (1,2) then 1 else 0 end deffspd30,
case when obs_min_inst_def30 >=3 and min_inst_def30 in (1,2,3) then 1 else 0 end deffstpd30,
case when obs_min_inst_def0 >= 1 then 1 else 0 end flg_mature_fpd0,
case when obs_min_inst_def10 >=1 then 1 else 0 end flg_mature_fpd10,
case when obs_min_inst_def30 >=1 then 1 else 0 end flg_mature_fpd30,
case when obs_min_inst_def30 >=2 then 1 else 0 end flg_mature_fspd_30,
case when obs_min_inst_def30 >=3 then 1 else 0 end flg_mature_fstpd_30
from prj-prod-dataplatform.risk_credit_mis.loan_deliquency_data),
base as
(select distinct r.customerId,
  r.digitalLoanAccountId,
  loanmaster.loanAccountNumber,
  r.modelDisplayName,
  r.Beta_Cash_Demo_Score,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month, 
  'Train' Data_selection,  
    del.deffspd30,
  del.flg_mature_fspd_30,
  loanmaster.new_loan_type,
  modelVersionId,
    trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile' 
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall' 
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
    from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  inner join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
  left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
  where loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and r.Beta_Cash_Demo_Score is not null
  and del.flg_mature_fspd_30 = 1
  )
  select * from base;
  """
dfd = client.query(sq).to_dataframe()
dfd = dfd.drop_duplicates(keep="first")
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()

# %%
df2 = dfd.copy()

# %% [markdown]
# ### Concat

# %%
# df_concat = pd.concat([df1, df2], ignore_index=True)
# print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")

# df_concat = (df_concat
#              .sort_values(by=['digitalLoanAccountId', 'Data_selection'],
#                           key=lambda s: s.map({'Train': 0, 'Test': 1}))
#              .drop_duplicates(subset=['digitalLoanAccountId'], keep='first')
#              .reset_index(drop=True))
# print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")
# df_concat.head()


# 1) Get all IDs present in Train
train_ids = set(df2["digitalLoanAccountId"])

# 2) Keep only Test rows whose ID is NOT in Train
df1_no_dupes = df1[~df1["digitalLoanAccountId"].isin(train_ids)]

# 3) Concatenate
df_concat = pd.concat([df1_no_dupes, df2], ignore_index=True)

print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")
df_concat.head()


# %%
df_concat["Beta_Cash_Demo_Score"] = pd.to_numeric(
    df_concat["Beta_Cash_Demo_Score"], errors="coerce"
)

# %%
fact_table, dimension_table = calculate_periodic_gini_prod_ver_trench_dimfact(
    df_concat,
    "Beta_Cash_Demo_Score",
    "deffspd30",
    "FSPD30",
    model_version_column="modelVersionId",
    trench_column="trenchCategory",
    loan_type_column="new_loan_type",
    loan_product_type_column="loan_product_type",
    account_id_column="digitalLoanAccountId",
)

# %%
fact_table, dimension_table = update_tables(
    fact_table, dimension_table, model_name="beta_demo_model_cash", product="CASH"
)

# %%
# Upload to BigQuery
table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_table"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(fact_table, table_id, job_config=job_config)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_table"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(dimension_table, table_id, job_config=job_config)
job.result()  # Wait for the job to complete

# %%


# %% [markdown]
# ### FSTPD30

# %% [markdown]
# ### Test

# %%
sq = """ 
WITH parsed as (
select customerId, digitalLoanAccountId,modelDisplayName,modelVersionId,start_time,end_time,prediction,trenchCategory,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeatures,
FROM `prj-prod-dataplatform.audit_balance.ml_model_run_details`
where modelDisplayName in ('Beta-Cash-Demo-Model', 'beta_demo_model_cash')
),
latest_request as (
select * from parsed
QUALIFY ROW_NUMBER() OVER (PARTITION BY customerId, digitalLoanAccountId,modelDisplayName ORDER BY start_time DESC ) = 1),
model_run as (
select customerId,digitalLoanAccountId,modelName, publish_time,
REPLACE(REPLACE(requestPayload, "'", '"'), "None", "null") AS requestPayload_clean
from `prj-prod-dataplatform.audit_balance.ml_request_details` 
WHERE modelName = 'Beta-Cash-Model-response'
QUALIFY ROW_NUMBER() OVER (PARTITION BY customerId, digitalLoanAccountId,modelName ORDER BY publish_time DESC ) = 1)
, 
  modelname as (
  select p.customerId,
  p.digitalLoanAccountId,
  start_time,
  prediction Beta_Cash_Demo_Score,
  coalesce(p.trenchCategory, JSON_VALUE(m.requestPayload_clean, "$.predictions.trenchCategory")) AS trenchCategory,
  case when p.modelDisplayName like 'Beta-Cash-Demo-Model' then 'beta_demo_model_cash' else p.modelDisplayName end as modelDisplayName,
  p.modelVersionId
  from latest_request p 
  left join model_run m on m.digitalLoanAccountId = p.digitalLoanAccountId
  ),
  deliquency as
(select loanAccountNumber,
case when obs_min_inst_def0 >= 1 and min_inst_def0 = 1 then 1 else 0 end deffpd0,
case when obs_min_inst_def10 >=1 and min_inst_def10 =1 then 1 else 0 end deffpd10,
case when obs_min_inst_def30 >=1 and min_inst_def30 =1 then 1 else 0 end deffpd30,
case when obs_min_inst_def30 >=2 and min_inst_def30 in (1,2) then 1 else 0 end deffspd30,
case when obs_min_inst_def30 >=3 and min_inst_def30 in (1,2,3) then 1 else 0 end deffstpd30,
case when obs_min_inst_def0 >= 1 then 1 else 0 end flg_mature_fpd0,
case when obs_min_inst_def10 >=1 then 1 else 0 end flg_mature_fpd10,
case when obs_min_inst_def30 >=1 then 1 else 0 end flg_mature_fpd30,
case when obs_min_inst_def30 >=2 then 1 else 0 end flg_mature_fspd_30,
case when obs_min_inst_def30 >=3 then 1 else 0 end flg_mature_fstpd_30
from prj-prod-dataplatform.risk_credit_mis.loan_deliquency_data),
base as 
  (select distinct r.customerId,
  r.digitalLoanAccountId,
  loanmaster.loanAccountNumber,
  r.modelDisplayName,
  r.Beta_Cash_Demo_Score,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month, 
  'Test' Data_selection,  
  del.deffstpd30,
  del.flg_mature_fstpd_30,
  loanmaster.new_loan_type,
  modelVersionId, trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile' 
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall' 
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
  from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  inner join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
   left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
  where loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and r.Beta_Cash_Demo_Score is not null
  and del.flg_mature_fstpd_30 = 1
  )
  select *  from base
  ;
"""
dfd = client.query(sq).to_dataframe()
dfd = dfd.drop_duplicates(keep="first")
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()


# %%
df1 = dfd.copy()

# %% [markdown]
# ### Train

# %%
sq = """ 
WITH parsed as (
select customerId, digitalLoanAccountId,modelDisplayName,modelVersionId,start_time,end_time,prediction,trenchCategory,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeatures,
FROM prj-prod-dataplatform.dap_ds_poweruser_playground.ml_training_model_run_details
where modelDisplayName in ('Beta-Cash-Demo-Model', 'beta_demo_model_cash')
),
  modelname as (
  select customerId,
  digitalLoanAccountId,
  start_time,
  prediction Beta_Cash_Demo_Score,
  case when modelDisplayName like 'Beta-Cash-Demo-Model' then 'beta_demo_model_cash' else modelDisplayName end as modelDisplayName,
  modelVersionId,
  trenchCategory
  from parsed
  ), 
  deliquency as
(select loanAccountNumber,
case when obs_min_inst_def0 >= 1 and min_inst_def0 = 1 then 1 else 0 end deffpd0,
case when obs_min_inst_def10 >=1 and min_inst_def10 =1 then 1 else 0 end deffpd10,
case when obs_min_inst_def30 >=1 and min_inst_def30 =1 then 1 else 0 end deffpd30,
case when obs_min_inst_def30 >=2 and min_inst_def30 in (1,2) then 1 else 0 end deffspd30,
case when obs_min_inst_def30 >=3 and min_inst_def30 in (1,2,3) then 1 else 0 end deffstpd30,
case when obs_min_inst_def0 >= 1 then 1 else 0 end flg_mature_fpd0,
case when obs_min_inst_def10 >=1 then 1 else 0 end flg_mature_fpd10,
case when obs_min_inst_def30 >=1 then 1 else 0 end flg_mature_fpd30,
case when obs_min_inst_def30 >=2 then 1 else 0 end flg_mature_fspd_30,
case when obs_min_inst_def30 >=3 then 1 else 0 end flg_mature_fstpd_30
from prj-prod-dataplatform.risk_credit_mis.loan_deliquency_data),
base as
(select distinct r.customerId,
  r.digitalLoanAccountId,
  loanmaster.loanAccountNumber,
  r.modelDisplayName,
  r.Beta_Cash_Demo_Score,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month, 
  'Train' Data_selection,  
    del.deffstpd30,
  del.flg_mature_fstpd_30,
  loanmaster.new_loan_type,
  modelVersionId,
    trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile' 
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall' 
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
    from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  inner join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
  left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
  where loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and r.Beta_Cash_Demo_Score is not null
  and del.flg_mature_fstpd_30 = 1
  )
  select * from base;
  """
dfd = client.query(sq).to_dataframe()
dfd = dfd.drop_duplicates(keep="first")
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()

# %%
df2 = dfd.copy()

# %% [markdown]
# ### Concat

# %%
# df_concat = pd.concat([df1, df2], ignore_index=True)
# print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")

# df_concat = (df_concat
#              .sort_values(by=['digitalLoanAccountId', 'Data_selection'],
#                           key=lambda s: s.map({'Train': 0, 'Test': 1}))
#              .drop_duplicates(subset=['digitalLoanAccountId'], keep='first')
#              .reset_index(drop=True))
# print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")
# df_concat.head()


# 1) Get all IDs present in Train
train_ids = set(df2["digitalLoanAccountId"])

# 2) Keep only Test rows whose ID is NOT in Train
df1_no_dupes = df1[~df1["digitalLoanAccountId"].isin(train_ids)]

# 3) Concatenate
df_concat = pd.concat([df1_no_dupes, df2], ignore_index=True)

print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")
df_concat.head()


# %%
df_concat["Beta_Cash_Demo_Score"] = pd.to_numeric(
    df_concat["Beta_Cash_Demo_Score"], errors="coerce"
)

# %%
fact_table, dimension_table = calculate_periodic_gini_prod_ver_trench_dimfact(
    df_concat,
    "Beta_Cash_Demo_Score",
    "deffstpd30",
    "FSTPD30",
    model_version_column="modelVersionId",
    trench_column="trenchCategory",
    loan_type_column="new_loan_type",
    loan_product_type_column="loan_product_type",
    account_id_column="digitalLoanAccountId",
)

# %%
fact_table, dimension_table = update_tables(
    fact_table, dimension_table, model_name="beta_demo_model_cash", product="CASH"
)
print(f"The shape of the fact table is:\t {fact_table.shape}")
print(f"The shape of the dimension table is:\t {dimension_table.shape}")

# %%
# Upload to BigQuery
table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_table"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(fact_table, table_id, job_config=job_config)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_table"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(dimension_table, table_id, job_config=job_config)
job.result()  # Wait for the job to complete

# %% [markdown]
# # Beta-Cash-AppScore-Model

# %% [markdown]
# ### FPD0

# %% [markdown]
# ### Test

# %%

sq = """
WITH parsed as (
select customerId, digitalLoanAccountId,modelDisplayName,modelVersionId,start_time,end_time,prediction,trenchCategory,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeatures,
REPLACE(REPLACE(prediction, "'", '"'), "None", "null") AS prediction_clean
FROM `prj-prod-dataplatform.audit_balance.ml_model_run_details`
where modelDisplayName in  ('Beta-Cash-AppScore-Model', 'apps_score_cash')
),
latest_request as (
select * from parsed
QUALIFY ROW_NUMBER() OVER (PARTITION BY customerId, digitalLoanAccountId,modelDisplayName ORDER BY start_time DESC ) = 1),

model_run as (
select customerId,digitalLoanAccountId,modelName, publish_time,
REPLACE(REPLACE(requestPayload, "'", '"'), "None", "null") AS requestPayload_clean
from `prj-prod-dataplatform.audit_balance.ml_request_details` 
WHERE modelName = 'Beta-Cash-Model-response'
QUALIFY ROW_NUMBER() OVER (PARTITION BY customerId, digitalLoanAccountId,modelName ORDER BY publish_time DESC ) = 1)
, 
  modelname as (
  select p.customerId,
  p.digitalLoanAccountId,
  start_time,
  coalesce(SAFE_CAST(JSON_VALUE(p.prediction_clean, "$.combined_score") AS Float64)) AS beta_cash_app_score,
  case when modelDisplayName like 'Beta-Cash-AppScore-Model' then 'apps_score_cash' else modelDisplayName end as modelDisplayName,
  p.modelVersionId,
  coalesce(p.trenchCategory, JSON_VALUE(m.requestPayload_clean, "$.predictions.trenchCategory"))  trenchCategory
  from latest_request p
  left join model_run m on p.digitalLoanAccountId = m.digitalLoanAccountId
  ), 
  deliquency as
(select loanAccountNumber,
case when obs_min_inst_def0 >= 1 and min_inst_def0 = 1 then 1 else 0 end deffpd0,
case when obs_min_inst_def10 >=1 and min_inst_def10 =1 then 1 else 0 end deffpd10,
case when obs_min_inst_def30 >=1 and min_inst_def30 =1 then 1 else 0 end deffpd30,
case when obs_min_inst_def30 >=2 and min_inst_def30 in (1,2) then 1 else 0 end deffspd30,
case when obs_min_inst_def30 >=3 and min_inst_def30 in (1,2,3) then 1 else 0 end deffstpd30,
case when obs_min_inst_def0 >= 1 then 1 else 0 end flg_mature_fpd0,
case when obs_min_inst_def10 >=1 then 1 else 0 end flg_mature_fpd10,
case when obs_min_inst_def30 >=1 then 1 else 0 end flg_mature_fpd30,
case when obs_min_inst_def30 >=2 then 1 else 0 end flg_mature_fspd_30,
case when obs_min_inst_def30 >=3 then 1 else 0 end flg_mature_fstpd_30
from prj-prod-dataplatform.risk_credit_mis.loan_deliquency_data),
base as 
  (select  distinct r.customerId,
  r.digitalLoanAccountId,
  loanmaster.loanAccountNumber,
  r.modelDisplayName,
  r.beta_cash_app_score,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month, 
  'Test' Data_selection,  
  deffpd0,
  flg_mature_fpd0,
  loanmaster.new_loan_type,
  modelVersionId, trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile' 
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall' 
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
  from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  inner join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
   left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
  where loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and r.beta_cash_app_score is not null
  and flg_mature_fpd0 = 1
  )
  select *  from base
  ;
"""
dfd = client.query(sq).to_dataframe()
dfd = dfd.drop_duplicates(keep="first")
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()


# %%
df1 = dfd.copy()

# %% [markdown]
# ### Train

# %%
sq = """WITH parsed as (
select customerId, digitalLoanAccountId,modelDisplayName,modelVersionId,start_time,end_time,prediction,trenchCategory,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeatures,
FROM prj-prod-dataplatform.dap_ds_poweruser_playground.ml_training_model_run_details
where modelDisplayName in  ('Beta-Cash-AppScore-Model', 'apps_score_cash')
),
  modelname as (
  select customerId,
  digitalLoanAccountId,
  start_time,
  prediction beta_cash_app_score,
  case when modelDisplayName like 'Beta-Cash-AppScore-Model' then 'apps_score_cash' else modelDisplayName end as modelDisplayName,
  modelVersionId,
  trenchCategory
  from parsed
  )
  ,
  deliquency as
(select loanAccountNumber,
case when obs_min_inst_def0 >= 1 and min_inst_def0 = 1 then 1 else 0 end deffpd0,
case when obs_min_inst_def10 >=1 and min_inst_def10 =1 then 1 else 0 end deffpd10,
case when obs_min_inst_def30 >=1 and min_inst_def30 =1 then 1 else 0 end deffpd30,
case when obs_min_inst_def30 >=2 and min_inst_def30 in (1,2) then 1 else 0 end deffspd30,
case when obs_min_inst_def30 >=3 and min_inst_def30 in (1,2,3) then 1 else 0 end deffstpd30,
case when obs_min_inst_def0 >= 1 then 1 else 0 end flg_mature_fpd0,
case when obs_min_inst_def10 >=1 then 1 else 0 end flg_mature_fpd10,
case when obs_min_inst_def30 >=1 then 1 else 0 end flg_mature_fpd30,
case when obs_min_inst_def30 >=2 then 1 else 0 end flg_mature_fspd_30,
case when obs_min_inst_def30 >=3 then 1 else 0 end flg_mature_fstpd_30
from prj-prod-dataplatform.risk_credit_mis.loan_deliquency_data),
base as
(select distinct r.customerId,
  r.digitalLoanAccountId,
  loanmaster.loanAccountNumber,
  r.modelDisplayName,
  r.beta_cash_app_score,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month, 
  'Train' Data_selection,  
    deffpd0,
  flg_mature_fpd0,
  loanmaster.new_loan_type,
  modelVersionId,
    trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile' 
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall' 
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
    from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  inner join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
  left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
  where loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and r.beta_cash_app_score is not null
  and flg_mature_fpd0 = 1
  )
  select * from base;
  """
dfd = client.query(sq).to_dataframe()
dfd = dfd.drop_duplicates(keep="first")
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()

# %%
df2 = dfd.copy()

# %% [markdown]
# ### Concat

# %%
# df_concat = pd.concat([df1, df2], ignore_index=True)
# print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")

# df_concat = (df_concat
#              .sort_values(by=['digitalLoanAccountId', 'Data_selection'],
#                           key=lambda s: s.map({'Train': 0, 'Test': 1}))
#              .drop_duplicates(subset=['digitalLoanAccountId'], keep='first')
#              .reset_index(drop=True))
# print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")
# df_concat.head()


# 1) Get all IDs present in Train
train_ids = set(df2["digitalLoanAccountId"])

# 2) Keep only Test rows whose ID is NOT in Train
df1_no_dupes = df1[~df1["digitalLoanAccountId"].isin(train_ids)]

# 3) Concatenate
df_concat = pd.concat([df1_no_dupes, df2], ignore_index=True)

print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")
df_concat.head()


# %%
df_concat["beta_cash_app_score"] = pd.to_numeric(
    df_concat["beta_cash_app_score"], errors="coerce"
)

# %%
fact_table, dimension_table = calculate_periodic_gini_prod_ver_trench_dimfact(
    df_concat,
    "beta_cash_app_score",
    "deffpd0",
    "FPD0",
    model_version_column="modelVersionId",
    trench_column="trenchCategory",
    loan_type_column="new_loan_type",
    loan_product_type_column="loan_product_type",
    account_id_column="digitalLoanAccountId",
)

# %%
fact_table, dimension_table = update_tables(
    fact_table, dimension_table, model_name="apps_score_cash", product="CASH"
)
print(f"The shape of the fact table is:\t {fact_table.shape}")
print(f"The shape of the dimension table is:\t {dimension_table.shape}")

# %%
# Upload to BigQuery
table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_table"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(fact_table, table_id, job_config=job_config)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_table"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(dimension_table, table_id, job_config=job_config)
job.result()  # Wait for the job to complete

# %% [markdown]
# ### FPD10

# %% [markdown]
# ### Test

# %%
sq = """
WITH parsed as (
select customerId, digitalLoanAccountId,modelDisplayName,modelVersionId,start_time,end_time,prediction,trenchCategory,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeatures,
REPLACE(REPLACE(prediction, "'", '"'), "None", "null") AS prediction_clean
FROM `prj-prod-dataplatform.audit_balance.ml_model_run_details`
where modelDisplayName in  ('Beta-Cash-AppScore-Model', 'apps_score_cash')
),
latest_request as (
select * from parsed
QUALIFY ROW_NUMBER() OVER (PARTITION BY customerId, digitalLoanAccountId,modelDisplayName ORDER BY start_time DESC ) = 1),

model_run as (
select customerId,digitalLoanAccountId,modelName, publish_time,
REPLACE(REPLACE(requestPayload, "'", '"'), "None", "null") AS requestPayload_clean
from `prj-prod-dataplatform.audit_balance.ml_request_details` 
WHERE modelName = 'Beta-Cash-Model-response'
QUALIFY ROW_NUMBER() OVER (PARTITION BY customerId, digitalLoanAccountId,modelName ORDER BY publish_time DESC ) = 1)
, 
  modelname as (
  select p.customerId,
  p.digitalLoanAccountId,
  start_time,
  coalesce(SAFE_CAST(JSON_VALUE(p.prediction_clean, "$.combined_score") AS Float64)) AS beta_cash_app_score,
  case when modelDisplayName like 'Beta-Cash-AppScore-Model' then 'apps_score_cash' else modelDisplayName end as modelDisplayName,
  p.modelVersionId,
  coalesce(p.trenchCategory, JSON_VALUE(m.requestPayload_clean, "$.predictions.trenchCategory"))  trenchCategory
  from latest_request p
  left join model_run m on p.digitalLoanAccountId = m.digitalLoanAccountId
  ),
  deliquency as
(select loanAccountNumber,
case when obs_min_inst_def0 >= 1 and min_inst_def0 = 1 then 1 else 0 end deffpd0,
case when obs_min_inst_def10 >=1 and min_inst_def10 =1 then 1 else 0 end deffpd10,
case when obs_min_inst_def30 >=1 and min_inst_def30 =1 then 1 else 0 end deffpd30,
case when obs_min_inst_def30 >=2 and min_inst_def30 in (1,2) then 1 else 0 end deffspd30,
case when obs_min_inst_def30 >=3 and min_inst_def30 in (1,2,3) then 1 else 0 end deffstpd30,
case when obs_min_inst_def0 >= 1 then 1 else 0 end flg_mature_fpd0,
case when obs_min_inst_def10 >=1 then 1 else 0 end flg_mature_fpd10,
case when obs_min_inst_def30 >=1 then 1 else 0 end flg_mature_fpd30,
case when obs_min_inst_def30 >=2 then 1 else 0 end flg_mature_fspd_30,
case when obs_min_inst_def30 >=3 then 1 else 0 end flg_mature_fstpd_30
from prj-prod-dataplatform.risk_credit_mis.loan_deliquency_data),
base as 
  (select distinct r.customerId,
  r.digitalLoanAccountId,
  loanmaster.loanAccountNumber,
  r.modelDisplayName,
  r.beta_cash_app_score,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month, 
  'Test' Data_selection,  
  del.deffpd10,
  del.flg_mature_fpd10,
  loanmaster.new_loan_type,
  modelVersionId, trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile' 
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall' 
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
  from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  inner join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
   left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
  where loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and r.beta_cash_app_score is not null
  and del.flg_mature_fpd10 = 1
  )
  select *  from base
  ;
"""
dfd = client.query(sq).to_dataframe()
dfd = dfd.drop_duplicates(keep="first")
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()


# %%
df1 = dfd.copy()

# %% [markdown]
# ### Train

# %%
sq = """WITH parsed as (
select customerId, digitalLoanAccountId,modelDisplayName,modelVersionId,start_time,end_time,prediction,trenchCategory,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeatures,
FROM prj-prod-dataplatform.dap_ds_poweruser_playground.ml_training_model_run_details
where modelDisplayName in  ('Beta-Cash-AppScore-Model', 'apps_score_cash')
),
  modelname as (
  select customerId,
  digitalLoanAccountId,
  start_time,
  prediction beta_cash_app_score,
  case when modelDisplayName like 'Beta-Cash-AppScore-Model' then 'apps_score_cash' else modelDisplayName end as modelDisplayName,
  modelVersionId,
  trenchCategory
  from parsed
  )
  ,
  deliquency as
(select loanAccountNumber,
case when obs_min_inst_def0 >= 1 and min_inst_def0 = 1 then 1 else 0 end deffpd0,
case when obs_min_inst_def10 >=1 and min_inst_def10 =1 then 1 else 0 end deffpd10,
case when obs_min_inst_def30 >=1 and min_inst_def30 =1 then 1 else 0 end deffpd30,
case when obs_min_inst_def30 >=2 and min_inst_def30 in (1,2) then 1 else 0 end deffspd30,
case when obs_min_inst_def30 >=3 and min_inst_def30 in (1,2,3) then 1 else 0 end deffstpd30,
case when obs_min_inst_def0 >= 1 then 1 else 0 end flg_mature_fpd0,
case when obs_min_inst_def10 >=1 then 1 else 0 end flg_mature_fpd10,
case when obs_min_inst_def30 >=1 then 1 else 0 end flg_mature_fpd30,
case when obs_min_inst_def30 >=2 then 1 else 0 end flg_mature_fspd_30,
case when obs_min_inst_def30 >=3 then 1 else 0 end flg_mature_fstpd_30
from prj-prod-dataplatform.risk_credit_mis.loan_deliquency_data),
base as
(select distinct r.customerId,
  r.digitalLoanAccountId,
  loanmaster.loanAccountNumber,
  r.modelDisplayName,
  r.beta_cash_app_score,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month, 
  'Train' Data_selection,  
    del.deffpd10,
  del.flg_mature_fpd10,
  loanmaster.new_loan_type,
  modelVersionId,
    trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile' 
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall' 
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
    from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  inner join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
  left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
  where loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and r.beta_cash_app_score is not null
  and del.flg_mature_fpd10 = 1
  )
  select * from base;
  """
dfd = client.query(sq).to_dataframe()
dfd = dfd.drop_duplicates(keep="first")
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()

# %%
df2 = dfd.copy()

# %% [markdown]
# ### Concat

# %%
# df_concat = pd.concat([df1, df2], ignore_index=True)
# print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")

# df_concat = (df_concat
#              .sort_values(by=['digitalLoanAccountId', 'Data_selection'],
#                           key=lambda s: s.map({'Train': 0, 'Test': 1}))
#              .drop_duplicates(subset=['digitalLoanAccountId'], keep='first')
#              .reset_index(drop=True))
# print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")
# df_concat.head()


# 1) Get all IDs present in Train
train_ids = set(df2["digitalLoanAccountId"])

# 2) Keep only Test rows whose ID is NOT in Train
df1_no_dupes = df1[~df1["digitalLoanAccountId"].isin(train_ids)]

# 3) Concatenate
df_concat = pd.concat([df1_no_dupes, df2], ignore_index=True)

print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")
df_concat.head()


# %%
df_concat["beta_cash_app_score"] = pd.to_numeric(
    df_concat["beta_cash_app_score"], errors="coerce"
)

# %%
fact_table, dimension_table = calculate_periodic_gini_prod_ver_trench_dimfact(
    df_concat,
    "beta_cash_app_score",
    "deffpd10",
    "FPD10",
    model_version_column="modelVersionId",
    trench_column="trenchCategory",
    loan_type_column="new_loan_type",
    loan_product_type_column="loan_product_type",
    account_id_column="digitalLoanAccountId",
)

# %%
fact_table, dimension_table = update_tables(
    fact_table, dimension_table, model_name="apps_score_cash", product="CASH"
)

# %%
# Upload to BigQuery
table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_table"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(fact_table, table_id, job_config=job_config)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_table"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(dimension_table, table_id, job_config=job_config)
job.result()  # Wait for the job to complete

# %%


# %% [markdown]
# ### FPD30

# %% [markdown]
# ### Test

# %%
sq = """
WITH parsed as (
select customerId, digitalLoanAccountId,modelDisplayName,modelVersionId,start_time,end_time,prediction,trenchCategory,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeatures,
REPLACE(REPLACE(prediction, "'", '"'), "None", "null") AS prediction_clean
FROM `prj-prod-dataplatform.audit_balance.ml_model_run_details`
where modelDisplayName in  ('Beta-Cash-AppScore-Model', 'apps_score_cash')
),
latest_request as (
select * from parsed
QUALIFY ROW_NUMBER() OVER (PARTITION BY customerId, digitalLoanAccountId,modelDisplayName ORDER BY start_time DESC ) = 1),

model_run as (
select customerId,digitalLoanAccountId,modelName, publish_time,
REPLACE(REPLACE(requestPayload, "'", '"'), "None", "null") AS requestPayload_clean
from `prj-prod-dataplatform.audit_balance.ml_request_details` 
WHERE modelName = 'Beta-Cash-Model-response'
QUALIFY ROW_NUMBER() OVER (PARTITION BY customerId, digitalLoanAccountId,modelName ORDER BY publish_time DESC ) = 1)
, 
  modelname as (
  select p.customerId,
  p.digitalLoanAccountId,
  start_time,
  coalesce(SAFE_CAST(JSON_VALUE(p.prediction_clean, "$.combined_score") AS Float64)) AS beta_cash_app_score,
  case when modelDisplayName like 'Beta-Cash-AppScore-Model' then 'apps_score_cash' else modelDisplayName end as modelDisplayName,
  p.modelVersionId,
  coalesce(p.trenchCategory, JSON_VALUE(m.requestPayload_clean, "$.predictions.trenchCategory"))  trenchCategory
  from latest_request p
  left join model_run m on p.digitalLoanAccountId = m.digitalLoanAccountId
  ), 
  deliquency as
(select loanAccountNumber,
case when obs_min_inst_def0 >= 1 and min_inst_def0 = 1 then 1 else 0 end deffpd0,
case when obs_min_inst_def10 >=1 and min_inst_def10 =1 then 1 else 0 end deffpd10,
case when obs_min_inst_def30 >=1 and min_inst_def30 =1 then 1 else 0 end deffpd30,
case when obs_min_inst_def30 >=2 and min_inst_def30 in (1,2) then 1 else 0 end deffspd30,
case when obs_min_inst_def30 >=3 and min_inst_def30 in (1,2,3) then 1 else 0 end deffstpd30,
case when obs_min_inst_def0 >= 1 then 1 else 0 end flg_mature_fpd0,
case when obs_min_inst_def10 >=1 then 1 else 0 end flg_mature_fpd10,
case when obs_min_inst_def30 >=1 then 1 else 0 end flg_mature_fpd30,
case when obs_min_inst_def30 >=2 then 1 else 0 end flg_mature_fspd_30,
case when obs_min_inst_def30 >=3 then 1 else 0 end flg_mature_fstpd_30
from prj-prod-dataplatform.risk_credit_mis.loan_deliquency_data),
base as 
  (select distinct r.customerId,
  r.digitalLoanAccountId,
  loanmaster.loanAccountNumber,
  r.modelDisplayName,
  r.beta_cash_app_score,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month, 
  'Test' Data_selection,  
  del.deffpd30,
  del.flg_mature_fpd30,
  loanmaster.new_loan_type,
  modelVersionId, trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile' 
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall' 
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
  from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  inner join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
   left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
  where loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and r.beta_cash_app_score is not null
  and del.flg_mature_fpd30 = 1
  )
  select *  from base
  ;
"""
dfd = client.query(sq).to_dataframe()
dfd = dfd.drop_duplicates(keep="first")
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()


# %%
df1 = dfd.copy()

# %% [markdown]
# ### Train

# %%
sq = """WITH parsed as (
select customerId, digitalLoanAccountId,modelDisplayName,modelVersionId,start_time,end_time,prediction,trenchCategory,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeatures,
FROM prj-prod-dataplatform.dap_ds_poweruser_playground.ml_training_model_run_details
where modelDisplayName in  ('Beta-Cash-AppScore-Model', 'apps_score_cash')
),
  modelname as (
  select customerId,
  digitalLoanAccountId,
  start_time,
  prediction beta_cash_app_score,
  case when modelDisplayName like 'Beta-Cash-AppScore-Model' then 'apps_score_cash' else modelDisplayName end as modelDisplayName,
  modelVersionId,
  trenchCategory
  from parsed
  ),
  deliquency as
(select loanAccountNumber,
case when obs_min_inst_def0 >= 1 and min_inst_def0 = 1 then 1 else 0 end deffpd0,
case when obs_min_inst_def10 >=1 and min_inst_def10 =1 then 1 else 0 end deffpd10,
case when obs_min_inst_def30 >=1 and min_inst_def30 =1 then 1 else 0 end deffpd30,
case when obs_min_inst_def30 >=2 and min_inst_def30 in (1,2) then 1 else 0 end deffspd30,
case when obs_min_inst_def30 >=3 and min_inst_def30 in (1,2,3) then 1 else 0 end deffstpd30,
case when obs_min_inst_def0 >= 1 then 1 else 0 end flg_mature_fpd0,
case when obs_min_inst_def10 >=1 then 1 else 0 end flg_mature_fpd10,
case when obs_min_inst_def30 >=1 then 1 else 0 end flg_mature_fpd30,
case when obs_min_inst_def30 >=2 then 1 else 0 end flg_mature_fspd_30,
case when obs_min_inst_def30 >=3 then 1 else 0 end flg_mature_fstpd_30
from prj-prod-dataplatform.risk_credit_mis.loan_deliquency_data),
base as
(select distinct r.customerId,
  r.digitalLoanAccountId,
  loanmaster.loanAccountNumber,
  r.modelDisplayName,
  r.beta_cash_app_score,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month, 
  'Train' Data_selection,  
    del.deffpd30,
  del.flg_mature_fpd30,
  loanmaster.new_loan_type,
  modelVersionId,
    trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile' 
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall' 
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
    from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  inner join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
  left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
  where loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and r.beta_cash_app_score is not null
  and del.flg_mature_fpd30 = 1
  )
  select * from base;
  """
dfd = client.query(sq).to_dataframe()
dfd = dfd.drop_duplicates(keep="first")
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()

# %%
df2 = dfd.copy()

# %% [markdown]
# ### Concat

# %%
# df_concat = pd.concat([df1, df2], ignore_index=True)
# print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")

# df_concat = (df_concat
#              .sort_values(by=['digitalLoanAccountId', 'Data_selection'],
#                           key=lambda s: s.map({'Train': 0, 'Test': 1}))
#              .drop_duplicates(subset=['digitalLoanAccountId'], keep='first')
#              .reset_index(drop=True))
# print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")
# df_concat.head()


# 1) Get all IDs present in Train
train_ids = set(df2["digitalLoanAccountId"])

# 2) Keep only Test rows whose ID is NOT in Train
df1_no_dupes = df1[~df1["digitalLoanAccountId"].isin(train_ids)]

# 3) Concatenate
df_concat = pd.concat([df1_no_dupes, df2], ignore_index=True)

print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")
df_concat.head()


# %%
df_concat["beta_cash_app_score"] = pd.to_numeric(
    df_concat["beta_cash_app_score"], errors="coerce"
)

# %%
fact_table, dimension_table = calculate_periodic_gini_prod_ver_trench_dimfact(
    df_concat,
    "beta_cash_app_score",
    "deffpd30",
    "FPD30",
    model_version_column="modelVersionId",
    trench_column="trenchCategory",
    loan_type_column="new_loan_type",
    loan_product_type_column="loan_product_type",
    account_id_column="digitalLoanAccountId",
)

# %%
fact_table, dimension_table = update_tables(
    fact_table, dimension_table, model_name="apps_score_cash", product="CASH"
)

# %%
# Upload to BigQuery
table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_table"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(fact_table, table_id, job_config=job_config)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_table"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(dimension_table, table_id, job_config=job_config)
job.result()  # Wait for the job to complete

# %%


# %% [markdown]
# ### FSPD30

# %% [markdown]
# ### Test

# %%
sq = """
WITH parsed as (
select customerId, digitalLoanAccountId,modelDisplayName,modelVersionId,start_time,end_time,prediction,trenchCategory,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeatures,
REPLACE(REPLACE(prediction, "'", '"'), "None", "null") AS prediction_clean
FROM `prj-prod-dataplatform.audit_balance.ml_model_run_details`
where modelDisplayName in  ('Beta-Cash-AppScore-Model', 'apps_score_cash')
),
latest_request as (
select * from parsed
QUALIFY ROW_NUMBER() OVER (PARTITION BY customerId, digitalLoanAccountId,modelDisplayName ORDER BY start_time DESC ) = 1),

model_run as (
select customerId,digitalLoanAccountId,modelName, publish_time,
REPLACE(REPLACE(requestPayload, "'", '"'), "None", "null") AS requestPayload_clean
from `prj-prod-dataplatform.audit_balance.ml_request_details` 
WHERE modelName = 'Beta-Cash-Model-response'
QUALIFY ROW_NUMBER() OVER (PARTITION BY customerId, digitalLoanAccountId,modelName ORDER BY publish_time DESC ) = 1)
, 
  modelname as (
  select p.customerId,
  p.digitalLoanAccountId,
  start_time,
  coalesce(SAFE_CAST(JSON_VALUE(p.prediction_clean, "$.combined_score") AS Float64)) AS beta_cash_app_score,
  case when modelDisplayName like 'Beta-Cash-AppScore-Model' then 'apps_score_cash' else modelDisplayName end as modelDisplayName,
  p.modelVersionId,
  coalesce(p.trenchCategory, JSON_VALUE(m.requestPayload_clean, "$.predictions.trenchCategory"))  trenchCategory
  from latest_request p
  left join model_run m on p.digitalLoanAccountId = m.digitalLoanAccountId
  ),
  deliquency as
(select loanAccountNumber,
case when obs_min_inst_def0 >= 1 and min_inst_def0 = 1 then 1 else 0 end deffpd0,
case when obs_min_inst_def10 >=1 and min_inst_def10 =1 then 1 else 0 end deffpd10,
case when obs_min_inst_def30 >=1 and min_inst_def30 =1 then 1 else 0 end deffpd30,
case when obs_min_inst_def30 >=2 and min_inst_def30 in (1,2) then 1 else 0 end deffspd30,
case when obs_min_inst_def30 >=3 and min_inst_def30 in (1,2,3) then 1 else 0 end deffstpd30,
case when obs_min_inst_def0 >= 1 then 1 else 0 end flg_mature_fpd0,
case when obs_min_inst_def10 >=1 then 1 else 0 end flg_mature_fpd10,
case when obs_min_inst_def30 >=1 then 1 else 0 end flg_mature_fpd30,
case when obs_min_inst_def30 >=2 then 1 else 0 end flg_mature_fspd_30,
case when obs_min_inst_def30 >=3 then 1 else 0 end flg_mature_fstpd_30
from prj-prod-dataplatform.risk_credit_mis.loan_deliquency_data),
base as 
  (select  distinct r.customerId,
  r.digitalLoanAccountId,
  loanmaster.loanAccountNumber,
  r.modelDisplayName,
  r.beta_cash_app_score,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month, 
  'Test' Data_selection,  
  del.deffspd30,
  del.flg_mature_fspd_30,
  loanmaster.new_loan_type,
  modelVersionId, trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile' 
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall' 
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
  from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  inner join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
   left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
  where loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and r.beta_cash_app_score is not null
  and del.flg_mature_fspd_30 = 1
  )
  select *  from base
  ;
"""
dfd = client.query(sq).to_dataframe()
dfd = dfd.drop_duplicates(keep="first")
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()


# %%
df1 = dfd.copy()

# %% [markdown]
# ### Train

# %%
sq = """WITH parsed as (
select customerId, digitalLoanAccountId,modelDisplayName,modelVersionId,start_time,end_time,prediction,trenchCategory,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeatures,
FROM prj-prod-dataplatform.dap_ds_poweruser_playground.ml_training_model_run_details
where modelDisplayName in  ('Beta-Cash-AppScore-Model', 'apps_score_cash')
),
  modelname as (
  select customerId,
  digitalLoanAccountId,
  start_time,
  prediction beta_cash_app_score,
  case when modelDisplayName like 'Beta-Cash-AppScore-Model' then 'apps_score_cash' else modelDisplayName end as modelDisplayName,
  modelVersionId,
  trenchCategory
  from parsed
  )
  , 
  deliquency as
(select loanAccountNumber,
case when obs_min_inst_def0 >= 1 and min_inst_def0 = 1 then 1 else 0 end deffpd0,
case when obs_min_inst_def10 >=1 and min_inst_def10 =1 then 1 else 0 end deffpd10,
case when obs_min_inst_def30 >=1 and min_inst_def30 =1 then 1 else 0 end deffpd30,
case when obs_min_inst_def30 >=2 and min_inst_def30 in (1,2) then 1 else 0 end deffspd30,
case when obs_min_inst_def30 >=3 and min_inst_def30 in (1,2,3) then 1 else 0 end deffstpd30,
case when obs_min_inst_def0 >= 1 then 1 else 0 end flg_mature_fpd0,
case when obs_min_inst_def10 >=1 then 1 else 0 end flg_mature_fpd10,
case when obs_min_inst_def30 >=1 then 1 else 0 end flg_mature_fpd30,
case when obs_min_inst_def30 >=2 then 1 else 0 end flg_mature_fspd_30,
case when obs_min_inst_def30 >=3 then 1 else 0 end flg_mature_fstpd_30
from prj-prod-dataplatform.risk_credit_mis.loan_deliquency_data),
base as
(select distinct r.customerId,
  r.digitalLoanAccountId,
  loanmaster.loanAccountNumber,
  r.modelDisplayName,
  r.beta_cash_app_score,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month, 
  'Train' Data_selection,  
    del.deffspd30,
  del.flg_mature_fspd_30,
  loanmaster.new_loan_type,
  modelVersionId,
    trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile' 
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall' 
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
    from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  inner join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
  left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
  where loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and r.beta_cash_app_score is not null
  and del.flg_mature_fspd_30 = 1
  )
  select * from base;
  """
dfd = client.query(sq).to_dataframe()
dfd = dfd.drop_duplicates(keep="first")
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()

# %%
df2 = dfd.copy()

# %% [markdown]
# ### Concat

# %%
# df_concat = pd.concat([df1, df2], ignore_index=True)
# print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")

# df_concat = (df_concat
#              .sort_values(by=['digitalLoanAccountId', 'Data_selection'],
#                           key=lambda s: s.map({'Train': 0, 'Test': 1}))
#              .drop_duplicates(subset=['digitalLoanAccountId'], keep='first')
#              .reset_index(drop=True))
# print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")
# df_concat.head()


# 1) Get all IDs present in Train
train_ids = set(df2["digitalLoanAccountId"])

# 2) Keep only Test rows whose ID is NOT in Train
df1_no_dupes = df1[~df1["digitalLoanAccountId"].isin(train_ids)]

# 3) Concatenate
df_concat = pd.concat([df1_no_dupes, df2], ignore_index=True)

print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")
df_concat.head()


# %%
df_concat["beta_cash_app_score"] = pd.to_numeric(
    df_concat["beta_cash_app_score"], errors="coerce"
)

# %%
fact_table, dimension_table = calculate_periodic_gini_prod_ver_trench_dimfact(
    df_concat,
    "beta_cash_app_score",
    "deffspd30",
    "FSPD30",
    model_version_column="modelVersionId",
    trench_column="trenchCategory",
    loan_type_column="new_loan_type",
    loan_product_type_column="loan_product_type",
    account_id_column="digitalLoanAccountId",
)

# %%
fact_table, dimension_table = update_tables(
    fact_table, dimension_table, model_name="apps_score_cash", product="CASH"
)

# %%
# Upload to BigQuery
table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_table"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(fact_table, table_id, job_config=job_config)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_table"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(dimension_table, table_id, job_config=job_config)
job.result()  # Wait for the job to complete

# %%


# %% [markdown]
# ### FSTPD30

# %% [markdown]
# ### Test

# %%
sq = """
WITH parsed as (
select customerId, digitalLoanAccountId,modelDisplayName,modelVersionId,start_time,end_time,prediction,trenchCategory,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeatures,
REPLACE(REPLACE(prediction, "'", '"'), "None", "null") AS prediction_clean
FROM `prj-prod-dataplatform.audit_balance.ml_model_run_details`
where modelDisplayName in  ('Beta-Cash-AppScore-Model', 'apps_score_cash')
),
latest_request as (
select * from parsed
QUALIFY ROW_NUMBER() OVER (PARTITION BY customerId, digitalLoanAccountId,modelDisplayName ORDER BY start_time DESC ) = 1),

model_run as (
select customerId,digitalLoanAccountId,modelName, publish_time,
REPLACE(REPLACE(requestPayload, "'", '"'), "None", "null") AS requestPayload_clean
from `prj-prod-dataplatform.audit_balance.ml_request_details` 
WHERE modelName = 'Beta-Cash-Model-response'
QUALIFY ROW_NUMBER() OVER (PARTITION BY customerId, digitalLoanAccountId,modelName ORDER BY publish_time DESC ) = 1)
, 
  modelname as (
  select p.customerId,
  p.digitalLoanAccountId,
  start_time,
  coalesce(SAFE_CAST(JSON_VALUE(p.prediction_clean, "$.combined_score") AS Float64)) AS beta_cash_app_score,
  case when modelDisplayName like 'Beta-Cash-AppScore-Model' then 'apps_score_cash' else modelDisplayName end as modelDisplayName,
  p.modelVersionId,
  coalesce(p.trenchCategory, JSON_VALUE(m.requestPayload_clean, "$.predictions.trenchCategory"))  trenchCategory
  from latest_request p
  left join model_run m on p.digitalLoanAccountId = m.digitalLoanAccountId
  ),
  deliquency as
(select loanAccountNumber,
case when obs_min_inst_def0 >= 1 and min_inst_def0 = 1 then 1 else 0 end deffpd0,
case when obs_min_inst_def10 >=1 and min_inst_def10 =1 then 1 else 0 end deffpd10,
case when obs_min_inst_def30 >=1 and min_inst_def30 =1 then 1 else 0 end deffpd30,
case when obs_min_inst_def30 >=2 and min_inst_def30 in (1,2) then 1 else 0 end deffspd30,
case when obs_min_inst_def30 >=3 and min_inst_def30 in (1,2,3) then 1 else 0 end deffstpd30,
case when obs_min_inst_def0 >= 1 then 1 else 0 end flg_mature_fpd0,
case when obs_min_inst_def10 >=1 then 1 else 0 end flg_mature_fpd10,
case when obs_min_inst_def30 >=1 then 1 else 0 end flg_mature_fpd30,
case when obs_min_inst_def30 >=2 then 1 else 0 end flg_mature_fspd_30,
case when obs_min_inst_def30 >=3 then 1 else 0 end flg_mature_fstpd_30
from prj-prod-dataplatform.risk_credit_mis.loan_deliquency_data),
base as 
  (select distinct r.customerId,
  r.digitalLoanAccountId,
  loanmaster.loanAccountNumber,
  r.modelDisplayName,
  r.beta_cash_app_score,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month, 
  'Test' Data_selection,  
  del.deffstpd30,
  del.flg_mature_fstpd_30,
  loanmaster.new_loan_type,
  modelVersionId, trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile' 
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall' 
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
  from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  inner join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
   left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
  where loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and r.beta_cash_app_score is not null
  and del.flg_mature_fstpd_30 = 1
  )
  select *  from base
  ;
"""
dfd = client.query(sq).to_dataframe()
dfd = dfd.drop_duplicates(keep="first")
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()


# %%
df1 = dfd.copy()

# %% [markdown]
# ### Train

# %%
sq = """WITH parsed as (
select customerId, digitalLoanAccountId,modelDisplayName,modelVersionId,start_time,end_time,prediction,trenchCategory,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeatures,
FROM prj-prod-dataplatform.dap_ds_poweruser_playground.ml_training_model_run_details
where modelDisplayName in  ('Beta-Cash-AppScore-Model', 'apps_score_cash')
),
  modelname as (
  select customerId,
  digitalLoanAccountId,
  start_time,
  prediction beta_cash_app_score,
  case when modelDisplayName like 'Beta-Cash-AppScore-Model' then 'apps_score_cash' else modelDisplayName end as modelDisplayName,
  modelVersionId,
  trenchCategory
  from parsed
  ),
  deliquency as
(select loanAccountNumber,
case when obs_min_inst_def0 >= 1 and min_inst_def0 = 1 then 1 else 0 end deffpd0,
case when obs_min_inst_def10 >=1 and min_inst_def10 =1 then 1 else 0 end deffpd10,
case when obs_min_inst_def30 >=1 and min_inst_def30 =1 then 1 else 0 end deffpd30,
case when obs_min_inst_def30 >=2 and min_inst_def30 in (1,2) then 1 else 0 end deffspd30,
case when obs_min_inst_def30 >=3 and min_inst_def30 in (1,2,3) then 1 else 0 end deffstpd30,
case when obs_min_inst_def0 >= 1 then 1 else 0 end flg_mature_fpd0,
case when obs_min_inst_def10 >=1 then 1 else 0 end flg_mature_fpd10,
case when obs_min_inst_def30 >=1 then 1 else 0 end flg_mature_fpd30,
case when obs_min_inst_def30 >=2 then 1 else 0 end flg_mature_fspd_30,
case when obs_min_inst_def30 >=3 then 1 else 0 end flg_mature_fstpd_30
from prj-prod-dataplatform.risk_credit_mis.loan_deliquency_data),
base as
(select distinct r.customerId,
  r.digitalLoanAccountId,
  loanmaster.loanAccountNumber,
  r.modelDisplayName,
  r.beta_cash_app_score,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month, 
  'Train' Data_selection,  
    del.deffstpd30,
  del.flg_mature_fstpd_30,
  loanmaster.new_loan_type,
  modelVersionId,
    trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile' 
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall' 
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
    from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  inner join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
  left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
  where loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and r.beta_cash_app_score is not null
  and del.flg_mature_fstpd_30 = 1
  )
  select * from base;
  """
dfd = client.query(sq).to_dataframe()
dfd = dfd.drop_duplicates(keep="first")
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()

# %%
df2 = dfd.copy()

# %% [markdown]
# ### Concat

# %%
# df_concat = pd.concat([df1, df2], ignore_index=True)
# print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")

# df_concat = (df_concat
#              .sort_values(by=['digitalLoanAccountId', 'Data_selection'],
#                           key=lambda s: s.map({'Train': 0, 'Test': 1}))
#              .drop_duplicates(subset=['digitalLoanAccountId'], keep='first')
#              .reset_index(drop=True))
# print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")
# df_concat.head()


# 1) Get all IDs present in Train
train_ids = set(df2["digitalLoanAccountId"])

# 2) Keep only Test rows whose ID is NOT in Train
df1_no_dupes = df1[~df1["digitalLoanAccountId"].isin(train_ids)]

# 3) Concatenate
df_concat = pd.concat([df1_no_dupes, df2], ignore_index=True)

print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")
df_concat.head()


# %%
df_concat["beta_cash_app_score"] = pd.to_numeric(
    df_concat["beta_cash_app_score"], errors="coerce"
)

# %%
fact_table, dimension_table = calculate_periodic_gini_prod_ver_trench_dimfact(
    df_concat,
    "beta_cash_app_score",
    "deffstpd30",
    "FSTPD30",
    model_version_column="modelVersionId",
    trench_column="trenchCategory",
    loan_type_column="new_loan_type",
    loan_product_type_column="loan_product_type",
    account_id_column="digitalLoanAccountId",
)

# %%
fact_table, dimension_table = update_tables(
    fact_table, dimension_table, model_name="apps_score_cash", product="CASH"
)
print(f"The shape of the fact table is:\t {fact_table.shape}")
print(f"The shape of the dimension table is:\t {dimension_table.shape}")

# %%
# Upload to BigQuery
table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_table"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(fact_table, table_id, job_config=job_config)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_table"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(dimension_table, table_id, job_config=job_config)
job.result()  # Wait for the job to complete

# %% [markdown]
# # Beta-Cash-Stack-Model

# %% [markdown]
# ### FPD0

# %% [markdown]
# ### Test

# %%

sq = """
WITH parsed as (
select customerId, digitalLoanAccountId,modelDisplayName,modelVersionId,start_time,end_time,prediction,trenchCategory,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeatures,
REPLACE(REPLACE(prediction, "'", '"'), "None", "null") AS prediction_clean
FROM `prj-prod-dataplatform.audit_balance.ml_model_run_details`
where modelDisplayName in  ('Beta-Cash-Stack-Model', 'beta_stack_model_cash')
and modelVersionId = 'v1'
),
latest_request as (
select * from parsed
QUALIFY ROW_NUMBER() OVER (PARTITION BY customerId, digitalLoanAccountId,modelDisplayName ORDER BY start_time DESC ) = 1),

model_run as (
select customerId,digitalLoanAccountId,modelName, publish_time,
REPLACE(REPLACE(requestPayload, "'", '"'), "None", "null") AS requestPayload_clean
from `prj-prod-dataplatform.audit_balance.ml_request_details` 
WHERE modelName = 'Beta-Cash-Model-response'
QUALIFY ROW_NUMBER() OVER (PARTITION BY customerId, digitalLoanAccountId,modelName ORDER BY publish_time DESC ) = 1)
, 
  modelname as (
  select p.customerId,
  p.digitalLoanAccountId,
  start_time,
  prediction AS Beta_cash_stack_score,
  coalesce(p.trenchCategory, JSON_VALUE(m.requestPayload_clean, "$.predictions.trenchCategory")) AS trenchCategory,
  case when modelDisplayName like 'Beta-Cash-Stack-Model' then 'beta_stack_model_cash' else p.modelDisplayName end as modelDisplayName,
  p.modelVersionId
  from latest_request p
  left join model_run m on m.digitalLoanAccountId = p.digitalLoanAccountId
  ),
  deliquency as
(select loanAccountNumber,
case when obs_min_inst_def0 >= 1 and min_inst_def0 = 1 then 1 else 0 end deffpd0,
case when obs_min_inst_def10 >=1 and min_inst_def10 =1 then 1 else 0 end deffpd10,
case when obs_min_inst_def30 >=1 and min_inst_def30 =1 then 1 else 0 end deffpd30,
case when obs_min_inst_def30 >=2 and min_inst_def30 in (1,2) then 1 else 0 end deffspd30,
case when obs_min_inst_def30 >=3 and min_inst_def30 in (1,2,3) then 1 else 0 end deffstpd30,
case when obs_min_inst_def0 >= 1 then 1 else 0 end flg_mature_fpd0,
case when obs_min_inst_def10 >=1 then 1 else 0 end flg_mature_fpd10,
case when obs_min_inst_def30 >=1 then 1 else 0 end flg_mature_fpd30,
case when obs_min_inst_def30 >=2 then 1 else 0 end flg_mature_fspd_30,
case when obs_min_inst_def30 >=3 then 1 else 0 end flg_mature_fstpd_30
from prj-prod-dataplatform.risk_credit_mis.loan_deliquency_data),
base as 
  (select 
  distinct
  r.customerId,
  r.digitalLoanAccountId,
  loanmaster.loanAccountNumber,
  r.modelDisplayName,
  r.Beta_cash_stack_score,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month, 
  'Test' Data_selection,  
  deffpd0,
  flg_mature_fpd0,
  loanmaster.new_loan_type,
  modelVersionId, trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile' 
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall' 
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
  from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  inner join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
   left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
  where loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and r.Beta_cash_stack_score is not null
  and flg_mature_fpd0 = 1
  )
  select *  from base
  ;
"""
dfd = client.query(sq).to_dataframe()
dfd = dfd.drop_duplicates(keep="first")
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()


# %%
df1 = dfd.copy()

# %% [markdown]
# ### Train

# %%
sq = """
WITH parsed as (
select customerId, digitalLoanAccountId,modelDisplayName,modelVersionId,start_time,end_time,prediction,trenchCategory,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeatures,
FROM prj-prod-dataplatform.dap_ds_poweruser_playground.ml_training_model_run_details
where modelDisplayName in  ('Beta-Cash-Stack-Model', 'beta_stack_model_cash')
),
  modelname as (
  select customerId,
  digitalLoanAccountId,
  start_time,
  prediction Beta_cash_stack_score,
    case when modelDisplayName like 'Beta-Cash-Stack-Model' then 'beta_stack_model_cash' else modelDisplayName end as modelDisplayName,
    modelVersionId,
  trenchCategory
  from parsed
  )
,
  deliquency as
(select loanAccountNumber,
case when obs_min_inst_def0 >= 1 and min_inst_def0 = 1 then 1 else 0 end deffpd0,
case when obs_min_inst_def10 >=1 and min_inst_def10 =1 then 1 else 0 end deffpd10,
case when obs_min_inst_def30 >=1 and min_inst_def30 =1 then 1 else 0 end deffpd30,
case when obs_min_inst_def30 >=2 and min_inst_def30 in (1,2) then 1 else 0 end deffspd30,
case when obs_min_inst_def30 >=3 and min_inst_def30 in (1,2,3) then 1 else 0 end deffstpd30,
case when obs_min_inst_def0 >= 1 then 1 else 0 end flg_mature_fpd0,
case when obs_min_inst_def10 >=1 then 1 else 0 end flg_mature_fpd10,
case when obs_min_inst_def30 >=1 then 1 else 0 end flg_mature_fpd30,
case when obs_min_inst_def30 >=2 then 1 else 0 end flg_mature_fspd_30,
case when obs_min_inst_def30 >=3 then 1 else 0 end flg_mature_fstpd_30
from prj-prod-dataplatform.risk_credit_mis.loan_deliquency_data),
base as
(select 
distinct
r.customerId,
  r.digitalLoanAccountId,
  loanmaster.loanAccountNumber,
  r.modelDisplayName,
  r.Beta_cash_stack_score,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month, 
  'Train' Data_selection,  
    deffpd0,
  flg_mature_fpd0,
  loanmaster.new_loan_type,
  modelVersionId,
    trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile' 
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall' 
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
    from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  inner join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
  left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
  where loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and r.Beta_cash_stack_score is not null
  and flg_mature_fpd0 = 1
  )
  select * from base;
  """
dfd = client.query(sq).to_dataframe()
dfd = dfd.drop_duplicates(keep="first")
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()

# %%
df2 = dfd.copy()

# %% [markdown]
# ### Concat

# %%
# df_concat = pd.concat([df1, df2], ignore_index=True)
# print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")

# df_concat = (df_concat
#              .sort_values(by=['digitalLoanAccountId', 'Data_selection'],
#                           key=lambda s: s.map({'Train': 0, 'Test': 1}))
#              .drop_duplicates(subset=['digitalLoanAccountId'], keep='first')
#              .reset_index(drop=True))
# print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")
# df_concat.head()


# 1) Get all IDs present in Train
train_ids = set(df2["digitalLoanAccountId"])

# 2) Keep only Test rows whose ID is NOT in Train
df1_no_dupes = df1[~df1["digitalLoanAccountId"].isin(train_ids)]

# 3) Concatenate
df_concat = pd.concat([df1_no_dupes, df2], ignore_index=True)

print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")
df_concat.head()


# %%
df_concat["Beta_cash_stack_score"] = pd.to_numeric(
    df_concat["Beta_cash_stack_score"], errors="coerce"
)

# %%
fact_table, dimension_table = calculate_periodic_gini_prod_ver_trench_dimfact(
    df_concat,
    "Beta_cash_stack_score",
    "deffpd0",
    "FPD0",
    model_version_column="modelVersionId",
    trench_column="trenchCategory",
    loan_type_column="new_loan_type",
    loan_product_type_column="loan_product_type",
    account_id_column="digitalLoanAccountId",
)

# %%
fact_table, dimension_table = update_tables(
    fact_table, dimension_table, model_name="beta_stack_model_cash", product="CASH"
)
print(f"The shape of the fact table is:\t {fact_table.shape}")
print(f"The shape of the dimension table is:\t {dimension_table.shape}")

# %%
# Upload to BigQuery
table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_table"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(fact_table, table_id, job_config=job_config)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_table"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(dimension_table, table_id, job_config=job_config)
job.result()  # Wait for the job to complete

# %% [markdown]
# ### FPD10

# %% [markdown]
# ### Test

# %%
sq = """
WITH parsed as (
select customerId, digitalLoanAccountId,modelDisplayName,modelVersionId,start_time,end_time,prediction,trenchCategory,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeatures,
REPLACE(REPLACE(prediction, "'", '"'), "None", "null") AS prediction_clean
FROM `prj-prod-dataplatform.audit_balance.ml_model_run_details`
where modelDisplayName in  ('Beta-Cash-Stack-Model', 'beta_stack_model_cash')
and modelVersionId = 'v1'
),
latest_request as (
select * from parsed
QUALIFY ROW_NUMBER() OVER (PARTITION BY customerId, digitalLoanAccountId,modelDisplayName ORDER BY start_time DESC ) = 1),

model_run as (
select customerId,digitalLoanAccountId,modelName, publish_time,
REPLACE(REPLACE(requestPayload, "'", '"'), "None", "null") AS requestPayload_clean
from `prj-prod-dataplatform.audit_balance.ml_request_details` 
WHERE modelName = 'Beta-Cash-Model-response'
QUALIFY ROW_NUMBER() OVER (PARTITION BY customerId, digitalLoanAccountId,modelName ORDER BY publish_time DESC ) = 1)
, 
  modelname as (
  select p.customerId,
  p.digitalLoanAccountId,
  start_time,
  prediction AS Beta_cash_stack_score,
  coalesce(p.trenchCategory, JSON_VALUE(m.requestPayload_clean, "$.predictions.trenchCategory")) AS trenchCategory,
  case when modelDisplayName like 'Beta-Cash-Stack-Model' then 'beta_stack_model_cash' else p.modelDisplayName end as modelDisplayName,
  p.modelVersionId
  from latest_request p
  left join model_run m on m.digitalLoanAccountId = p.digitalLoanAccountId
  ),
  deliquency as
(select loanAccountNumber,
case when obs_min_inst_def0 >= 1 and min_inst_def0 = 1 then 1 else 0 end deffpd0,
case when obs_min_inst_def10 >=1 and min_inst_def10 =1 then 1 else 0 end deffpd10,
case when obs_min_inst_def30 >=1 and min_inst_def30 =1 then 1 else 0 end deffpd30,
case when obs_min_inst_def30 >=2 and min_inst_def30 in (1,2) then 1 else 0 end deffspd30,
case when obs_min_inst_def30 >=3 and min_inst_def30 in (1,2,3) then 1 else 0 end deffstpd30,
case when obs_min_inst_def0 >= 1 then 1 else 0 end flg_mature_fpd0,
case when obs_min_inst_def10 >=1 then 1 else 0 end flg_mature_fpd10,
case when obs_min_inst_def30 >=1 then 1 else 0 end flg_mature_fpd30,
case when obs_min_inst_def30 >=2 then 1 else 0 end flg_mature_fspd_30,
case when obs_min_inst_def30 >=3 then 1 else 0 end flg_mature_fstpd_30
from prj-prod-dataplatform.risk_credit_mis.loan_deliquency_data),
base as 
  (select distinct r.customerId,
  r.digitalLoanAccountId,
  loanmaster.loanAccountNumber,
  r.modelDisplayName,
  r.Beta_cash_stack_score,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month, 
  'Test' Data_selection,  
  del.deffpd10,
  del.flg_mature_fpd10,
  loanmaster.new_loan_type,
  modelVersionId, trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile' 
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall' 
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
  from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  inner join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
   left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
  where loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and r.Beta_cash_stack_score is not null
  and del.flg_mature_fpd10 = 1
  )
  select *  from base
  ;
"""
dfd = client.query(sq).to_dataframe()
dfd = dfd.drop_duplicates(keep="first")
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()


# %%
df1 = dfd.copy()

# %% [markdown]
# ### Train

# %%
sq = """
WITH parsed as (
select customerId, digitalLoanAccountId,modelDisplayName,modelVersionId,start_time,end_time,prediction,trenchCategory,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeatures,
FROM prj-prod-dataplatform.dap_ds_poweruser_playground.ml_training_model_run_details
where modelDisplayName in  ('Beta-Cash-Stack-Model', 'beta_stack_model_cash')
),
  modelname as (
  select customerId,
  digitalLoanAccountId,
  start_time,
  prediction Beta_cash_stack_score,
    case when modelDisplayName like 'Beta-Cash-Stack-Model' then 'beta_stack_model_cash' else modelDisplayName end as modelDisplayName,
    modelVersionId,
  trenchCategory
  from parsed
  )
,
  deliquency as
(select loanAccountNumber,
case when obs_min_inst_def0 >= 1 and min_inst_def0 = 1 then 1 else 0 end deffpd0,
case when obs_min_inst_def10 >=1 and min_inst_def10 =1 then 1 else 0 end deffpd10,
case when obs_min_inst_def30 >=1 and min_inst_def30 =1 then 1 else 0 end deffpd30,
case when obs_min_inst_def30 >=2 and min_inst_def30 in (1,2) then 1 else 0 end deffspd30,
case when obs_min_inst_def30 >=3 and min_inst_def30 in (1,2,3) then 1 else 0 end deffstpd30,
case when obs_min_inst_def0 >= 1 then 1 else 0 end flg_mature_fpd0,
case when obs_min_inst_def10 >=1 then 1 else 0 end flg_mature_fpd10,
case when obs_min_inst_def30 >=1 then 1 else 0 end flg_mature_fpd30,
case when obs_min_inst_def30 >=2 then 1 else 0 end flg_mature_fspd_30,
case when obs_min_inst_def30 >=3 then 1 else 0 end flg_mature_fstpd_30
from prj-prod-dataplatform.risk_credit_mis.loan_deliquency_data),
base as
(select distinct r.customerId,
  r.digitalLoanAccountId,
  loanmaster.loanAccountNumber,
  r.modelDisplayName,
  r.Beta_cash_stack_score,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month, 
  'Train' Data_selection,  
    del.deffpd10,
  del.flg_mature_fpd10,
  loanmaster.new_loan_type,
  modelVersionId,
    trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile' 
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall' 
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
    from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  inner join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
  left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
  where loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and r.Beta_cash_stack_score is not null
  and del.flg_mature_fpd10 = 1
  )
  select * from base;
  """
dfd = client.query(sq).to_dataframe()
dfd = dfd.drop_duplicates(keep="first")
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()

# %%
df2 = dfd.copy()

# %% [markdown]
# ### Concat

# %%
# df_concat = pd.concat([df1, df2], ignore_index=True)
# print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")

# df_concat = (df_concat
#              .sort_values(by=['digitalLoanAccountId', 'Data_selection'],
#                           key=lambda s: s.map({'Train': 0, 'Test': 1}))
#              .drop_duplicates(subset=['digitalLoanAccountId'], keep='first')
#              .reset_index(drop=True))
# print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")
# df_concat.head()


# 1) Get all IDs present in Train
train_ids = set(df2["digitalLoanAccountId"])

# 2) Keep only Test rows whose ID is NOT in Train
df1_no_dupes = df1[~df1["digitalLoanAccountId"].isin(train_ids)]

# 3) Concatenate
df_concat = pd.concat([df1_no_dupes, df2], ignore_index=True)

print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")
df_concat.head()


# %%
df_concat["Beta_cash_stack_score"] = pd.to_numeric(
    df_concat["Beta_cash_stack_score"], errors="coerce"
)

# %%
fact_table, dimension_table = calculate_periodic_gini_prod_ver_trench_dimfact(
    df_concat,
    "Beta_cash_stack_score",
    "deffpd10",
    "FPD10",
    model_version_column="modelVersionId",
    trench_column="trenchCategory",
    loan_type_column="new_loan_type",
    loan_product_type_column="loan_product_type",
    account_id_column="digitalLoanAccountId",
)

# %%
fact_table, dimension_table = update_tables(
    fact_table, dimension_table, model_name="beta_stack_model_cash", product="CASH"
)

# %%
# Upload to BigQuery
table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_table"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(fact_table, table_id, job_config=job_config)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_table"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(dimension_table, table_id, job_config=job_config)
job.result()  # Wait for the job to complete

# %%


# %% [markdown]
# ### FPD30

# %% [markdown]
# ### Test

# %%
sq = """
WITH parsed as (
select customerId, digitalLoanAccountId,modelDisplayName,modelVersionId,start_time,end_time,prediction,trenchCategory,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeatures,
REPLACE(REPLACE(prediction, "'", '"'), "None", "null") AS prediction_clean
FROM `prj-prod-dataplatform.audit_balance.ml_model_run_details`
where modelDisplayName in  ('Beta-Cash-Stack-Model', 'beta_stack_model_cash')
and modelVersionId = 'v1'
),
latest_request as (
select * from parsed
QUALIFY ROW_NUMBER() OVER (PARTITION BY customerId, digitalLoanAccountId,modelDisplayName ORDER BY start_time DESC ) = 1),

model_run as (
select customerId,digitalLoanAccountId,modelName, publish_time,
REPLACE(REPLACE(requestPayload, "'", '"'), "None", "null") AS requestPayload_clean
from `prj-prod-dataplatform.audit_balance.ml_request_details` 
WHERE modelName = 'Beta-Cash-Model-response'
QUALIFY ROW_NUMBER() OVER (PARTITION BY customerId, digitalLoanAccountId,modelName ORDER BY publish_time DESC ) = 1)
, 
  modelname as (
  select p.customerId,
  p.digitalLoanAccountId,
  start_time,
  prediction AS Beta_cash_stack_score,
  coalesce(p.trenchCategory, JSON_VALUE(m.requestPayload_clean, "$.predictions.trenchCategory")) AS trenchCategory,
  case when modelDisplayName like 'Beta-Cash-Stack-Model' then 'beta_stack_model_cash' else p.modelDisplayName end as modelDisplayName,
  p.modelVersionId
  from latest_request p
  left join model_run m on m.digitalLoanAccountId = p.digitalLoanAccountId
  ),
  deliquency as
(select loanAccountNumber,
case when obs_min_inst_def0 >= 1 and min_inst_def0 = 1 then 1 else 0 end deffpd0,
case when obs_min_inst_def10 >=1 and min_inst_def10 =1 then 1 else 0 end deffpd10,
case when obs_min_inst_def30 >=1 and min_inst_def30 =1 then 1 else 0 end deffpd30,
case when obs_min_inst_def30 >=2 and min_inst_def30 in (1,2) then 1 else 0 end deffspd30,
case when obs_min_inst_def30 >=3 and min_inst_def30 in (1,2,3) then 1 else 0 end deffstpd30,
case when obs_min_inst_def0 >= 1 then 1 else 0 end flg_mature_fpd0,
case when obs_min_inst_def10 >=1 then 1 else 0 end flg_mature_fpd10,
case when obs_min_inst_def30 >=1 then 1 else 0 end flg_mature_fpd30,
case when obs_min_inst_def30 >=2 then 1 else 0 end flg_mature_fspd_30,
case when obs_min_inst_def30 >=3 then 1 else 0 end flg_mature_fstpd_30
from prj-prod-dataplatform.risk_credit_mis.loan_deliquency_data),
base as 
  (select distinct r.customerId,
  r.digitalLoanAccountId,
  loanmaster.loanAccountNumber,
  r.modelDisplayName,
  r.Beta_cash_stack_score,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month, 
  'Test' Data_selection,  
  del.deffpd30,
  del.flg_mature_fpd30,
  loanmaster.new_loan_type,
  modelVersionId, trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile' 
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall' 
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
  from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  inner join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
   left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
  where loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and r.Beta_cash_stack_score is not null
  and del.flg_mature_fpd30 = 1
  )
  select *  from base
  ;
"""
dfd = client.query(sq).to_dataframe()
dfd = dfd.drop_duplicates(keep="first")
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()


# %%
df1 = dfd.copy()

# %% [markdown]
# ### Train

# %%
sq = """
WITH parsed as (
select customerId, digitalLoanAccountId,modelDisplayName,modelVersionId,start_time,end_time,prediction,trenchCategory,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeatures,
FROM prj-prod-dataplatform.dap_ds_poweruser_playground.ml_training_model_run_details
where modelDisplayName in  ('Beta-Cash-Stack-Model', 'beta_stack_model_cash')
),
  modelname as (
  select customerId,
  digitalLoanAccountId,
  start_time,
  prediction Beta_cash_stack_score,
    case when modelDisplayName like 'Beta-Cash-Stack-Model' then 'beta_stack_model_cash' else modelDisplayName end as modelDisplayName,
    modelVersionId,
  trenchCategory
  from parsed
  )
,
  deliquency as
(select loanAccountNumber,
case when obs_min_inst_def0 >= 1 and min_inst_def0 = 1 then 1 else 0 end deffpd0,
case when obs_min_inst_def10 >=1 and min_inst_def10 =1 then 1 else 0 end deffpd10,
case when obs_min_inst_def30 >=1 and min_inst_def30 =1 then 1 else 0 end deffpd30,
case when obs_min_inst_def30 >=2 and min_inst_def30 in (1,2) then 1 else 0 end deffspd30,
case when obs_min_inst_def30 >=3 and min_inst_def30 in (1,2,3) then 1 else 0 end deffstpd30,
case when obs_min_inst_def0 >= 1 then 1 else 0 end flg_mature_fpd0,
case when obs_min_inst_def10 >=1 then 1 else 0 end flg_mature_fpd10,
case when obs_min_inst_def30 >=1 then 1 else 0 end flg_mature_fpd30,
case when obs_min_inst_def30 >=2 then 1 else 0 end flg_mature_fspd_30,
case when obs_min_inst_def30 >=3 then 1 else 0 end flg_mature_fstpd_30
from prj-prod-dataplatform.risk_credit_mis.loan_deliquency_data),
base as
(select distinct r.customerId,
  r.digitalLoanAccountId,
  loanmaster.loanAccountNumber,
  r.modelDisplayName,
  r.Beta_cash_stack_score,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month, 
  'Train' Data_selection,  
    del.deffpd30,
  del.flg_mature_fpd30,
  loanmaster.new_loan_type,
  modelVersionId,
    trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile' 
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall' 
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
    from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  inner join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
  left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
  where loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and r.Beta_cash_stack_score is not null
  and del.flg_mature_fpd30 = 1
  )
  select * from base;
  """
dfd = client.query(sq).to_dataframe()
dfd = dfd.drop_duplicates(keep="first")
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()

# %%
df2 = dfd.copy()

# %% [markdown]
# ### Concat

# %%
# df_concat = pd.concat([df1, df2], ignore_index=True)
# print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")

# df_concat = (df_concat
#              .sort_values(by=['digitalLoanAccountId', 'Data_selection'],
#                           key=lambda s: s.map({'Train': 0, 'Test': 1}))
#              .drop_duplicates(subset=['digitalLoanAccountId'], keep='first')
#              .reset_index(drop=True))
# print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")
# df_concat.head()


# 1) Get all IDs present in Train
train_ids = set(df2["digitalLoanAccountId"])

# 2) Keep only Test rows whose ID is NOT in Train
df1_no_dupes = df1[~df1["digitalLoanAccountId"].isin(train_ids)]

# 3) Concatenate
df_concat = pd.concat([df1_no_dupes, df2], ignore_index=True)

print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")
df_concat.head()


# %%
df_concat["Beta_cash_stack_score"] = pd.to_numeric(
    df_concat["Beta_cash_stack_score"], errors="coerce"
)

# %%
fact_table, dimension_table = calculate_periodic_gini_prod_ver_trench_dimfact(
    df_concat,
    "Beta_cash_stack_score",
    "deffpd30",
    "FPD30",
    model_version_column="modelVersionId",
    trench_column="trenchCategory",
    loan_type_column="new_loan_type",
    loan_product_type_column="loan_product_type",
    account_id_column="digitalLoanAccountId",
)

# %%
fact_table, dimension_table = update_tables(
    fact_table, dimension_table, model_name="beta_stack_model_cash", product="CASH"
)

# %%
# Upload to BigQuery
table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_table"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(fact_table, table_id, job_config=job_config)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_table"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(dimension_table, table_id, job_config=job_config)
job.result()  # Wait for the job to complete

# %%


# %% [markdown]
# ### FSPD30

# %% [markdown]
# ### Test

# %%
sq = """
WITH parsed as (
select customerId, digitalLoanAccountId,modelDisplayName,modelVersionId,start_time,end_time,prediction,trenchCategory,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeatures,
REPLACE(REPLACE(prediction, "'", '"'), "None", "null") AS prediction_clean
FROM `prj-prod-dataplatform.audit_balance.ml_model_run_details`
where modelDisplayName in  ('Beta-Cash-Stack-Model', 'beta_stack_model_cash')
and modelVersionId = 'v1'
),
latest_request as (
select * from parsed
QUALIFY ROW_NUMBER() OVER (PARTITION BY customerId, digitalLoanAccountId,modelDisplayName ORDER BY start_time DESC ) = 1),

model_run as (
select customerId,digitalLoanAccountId,modelName, publish_time,
REPLACE(REPLACE(requestPayload, "'", '"'), "None", "null") AS requestPayload_clean
from `prj-prod-dataplatform.audit_balance.ml_request_details` 
WHERE modelName = 'Beta-Cash-Model-response'
QUALIFY ROW_NUMBER() OVER (PARTITION BY customerId, digitalLoanAccountId,modelName ORDER BY publish_time DESC ) = 1)
, 
  modelname as (
  select p.customerId,
  p.digitalLoanAccountId,
  start_time,
  prediction AS Beta_cash_stack_score,
  coalesce(p.trenchCategory, JSON_VALUE(m.requestPayload_clean, "$.predictions.trenchCategory")) AS trenchCategory,
  case when modelDisplayName like 'Beta-Cash-Stack-Model' then 'beta_stack_model_cash' else p.modelDisplayName end as modelDisplayName,
  p.modelVersionId
  from latest_request p
  left join model_run m on m.digitalLoanAccountId = p.digitalLoanAccountId
  ),
  deliquency as
(select loanAccountNumber,
case when obs_min_inst_def0 >= 1 and min_inst_def0 = 1 then 1 else 0 end deffpd0,
case when obs_min_inst_def10 >=1 and min_inst_def10 =1 then 1 else 0 end deffpd10,
case when obs_min_inst_def30 >=1 and min_inst_def30 =1 then 1 else 0 end deffpd30,
case when obs_min_inst_def30 >=2 and min_inst_def30 in (1,2) then 1 else 0 end deffspd30,
case when obs_min_inst_def30 >=3 and min_inst_def30 in (1,2,3) then 1 else 0 end deffstpd30,
case when obs_min_inst_def0 >= 1 then 1 else 0 end flg_mature_fpd0,
case when obs_min_inst_def10 >=1 then 1 else 0 end flg_mature_fpd10,
case when obs_min_inst_def30 >=1 then 1 else 0 end flg_mature_fpd30,
case when obs_min_inst_def30 >=2 then 1 else 0 end flg_mature_fspd_30,
case when obs_min_inst_def30 >=3 then 1 else 0 end flg_mature_fstpd_30
from prj-prod-dataplatform.risk_credit_mis.loan_deliquency_data),
base as 
  (select distinct r.customerId,
  r.digitalLoanAccountId,
  loanmaster.loanAccountNumber,
  r.modelDisplayName,
  r.Beta_cash_stack_score,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month, 
  'Test' Data_selection,  
  del.deffspd30,
  del.flg_mature_fspd_30,
  loanmaster.new_loan_type,
  modelVersionId, trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile' 
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall' 
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
  from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  inner join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
   left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
  where loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and r.Beta_cash_stack_score is not null
  and del.flg_mature_fspd_30 = 1
  )
  select *  from base
  ;
"""
dfd = client.query(sq).to_dataframe()
dfd = dfd.drop_duplicates(keep="first")
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()


# %%
df1 = dfd.copy()

# %% [markdown]
# ### Train

# %%
sq = """
WITH parsed as (
select customerId, digitalLoanAccountId,modelDisplayName,modelVersionId,start_time,end_time,prediction,trenchCategory,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeatures,
FROM prj-prod-dataplatform.dap_ds_poweruser_playground.ml_training_model_run_details
where modelDisplayName in  ('Beta-Cash-Stack-Model', 'beta_stack_model_cash')
),
  modelname as (
  select customerId,
  digitalLoanAccountId,
  start_time,
  prediction Beta_cash_stack_score,
    case when modelDisplayName like 'Beta-Cash-Stack-Model' then 'beta_stack_model_cash' else modelDisplayName end as modelDisplayName,
    modelVersionId,
  trenchCategory
  from parsed
  )
,
  deliquency as
(select loanAccountNumber,
case when obs_min_inst_def0 >= 1 and min_inst_def0 = 1 then 1 else 0 end deffpd0,
case when obs_min_inst_def10 >=1 and min_inst_def10 =1 then 1 else 0 end deffpd10,
case when obs_min_inst_def30 >=1 and min_inst_def30 =1 then 1 else 0 end deffpd30,
case when obs_min_inst_def30 >=2 and min_inst_def30 in (1,2) then 1 else 0 end deffspd30,
case when obs_min_inst_def30 >=3 and min_inst_def30 in (1,2,3) then 1 else 0 end deffstpd30,
case when obs_min_inst_def0 >= 1 then 1 else 0 end flg_mature_fpd0,
case when obs_min_inst_def10 >=1 then 1 else 0 end flg_mature_fpd10,
case when obs_min_inst_def30 >=1 then 1 else 0 end flg_mature_fpd30,
case when obs_min_inst_def30 >=2 then 1 else 0 end flg_mature_fspd_30,
case when obs_min_inst_def30 >=3 then 1 else 0 end flg_mature_fstpd_30
from prj-prod-dataplatform.risk_credit_mis.loan_deliquency_data),
base as
(select distinct r.customerId,
  r.digitalLoanAccountId,
  loanmaster.loanAccountNumber,
  r.modelDisplayName,
  r.Beta_cash_stack_score,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month, 
  'Train' Data_selection,  
    del.deffspd30,
  del.flg_mature_fspd_30,
  loanmaster.new_loan_type,
  modelVersionId,
    trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile' 
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall' 
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
    from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  inner join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
  left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
  where loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and r.Beta_cash_stack_score is not null
  and del.flg_mature_fspd_30 = 1
  )
  select * from base;
  """
dfd = client.query(sq).to_dataframe()
dfd = dfd.drop_duplicates(keep="first")
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()

# %%
df2 = dfd.copy()

# %% [markdown]
# ### Concat

# %%
# df_concat = pd.concat([df1, df2], ignore_index=True)
# print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")

# df_concat = (df_concat
#              .sort_values(by=['digitalLoanAccountId', 'Data_selection'],
#                           key=lambda s: s.map({'Train': 0, 'Test': 1}))
#              .drop_duplicates(subset=['digitalLoanAccountId'], keep='first')
#              .reset_index(drop=True))
# print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")
# df_concat.head()


# 1) Get all IDs present in Train
train_ids = set(df2["digitalLoanAccountId"])

# 2) Keep only Test rows whose ID is NOT in Train
df1_no_dupes = df1[~df1["digitalLoanAccountId"].isin(train_ids)]

# 3) Concatenate
df_concat = pd.concat([df1_no_dupes, df2], ignore_index=True)

print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")
df_concat.head()


# %%
df_concat["Beta_cash_stack_score"] = pd.to_numeric(
    df_concat["Beta_cash_stack_score"], errors="coerce"
)

# %%
fact_table, dimension_table = calculate_periodic_gini_prod_ver_trench_dimfact(
    df_concat,
    "Beta_cash_stack_score",
    "deffspd30",
    "FSPD30",
    model_version_column="modelVersionId",
    trench_column="trenchCategory",
    loan_type_column="new_loan_type",
    loan_product_type_column="loan_product_type",
    account_id_column="digitalLoanAccountId",
)

# %%
fact_table, dimension_table = update_tables(
    fact_table, dimension_table, model_name="beta_stack_model_cash", product="CASH"
)

# %%
# Upload to BigQuery
table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_table"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(fact_table, table_id, job_config=job_config)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_table"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(dimension_table, table_id, job_config=job_config)
job.result()  # Wait for the job to complete

# %%


# %% [markdown]
# ### FSTPD30

# %% [markdown]
# ### Test

# %%
sq = """
WITH parsed as (
select customerId, digitalLoanAccountId,modelDisplayName,modelVersionId,start_time,end_time,prediction,trenchCategory,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeatures,
REPLACE(REPLACE(prediction, "'", '"'), "None", "null") AS prediction_clean
FROM `prj-prod-dataplatform.audit_balance.ml_model_run_details`
where modelDisplayName in  ('Beta-Cash-Stack-Model', 'beta_stack_model_cash')
and modelVersionId = 'v1'
),
latest_request as (
select * from parsed
QUALIFY ROW_NUMBER() OVER (PARTITION BY customerId, digitalLoanAccountId,modelDisplayName ORDER BY start_time DESC ) = 1),

model_run as (
select customerId,digitalLoanAccountId,modelName, publish_time,
REPLACE(REPLACE(requestPayload, "'", '"'), "None", "null") AS requestPayload_clean
from `prj-prod-dataplatform.audit_balance.ml_request_details` 
WHERE modelName = 'Beta-Cash-Model-response'
QUALIFY ROW_NUMBER() OVER (PARTITION BY customerId, digitalLoanAccountId,modelName ORDER BY publish_time DESC ) = 1)
, 
  modelname as (
  select p.customerId,
  p.digitalLoanAccountId,
  start_time,
  prediction AS Beta_cash_stack_score,
  coalesce(p.trenchCategory, JSON_VALUE(m.requestPayload_clean, "$.predictions.trenchCategory")) AS trenchCategory,
  case when modelDisplayName like 'Beta-Cash-Stack-Model' then 'beta_stack_model_cash' else p.modelDisplayName end as modelDisplayName,
  p.modelVersionId
  from latest_request p
  left join model_run m on m.digitalLoanAccountId = p.digitalLoanAccountId
  ),
  deliquency as
(select loanAccountNumber,
case when obs_min_inst_def0 >= 1 and min_inst_def0 = 1 then 1 else 0 end deffpd0,
case when obs_min_inst_def10 >=1 and min_inst_def10 =1 then 1 else 0 end deffpd10,
case when obs_min_inst_def30 >=1 and min_inst_def30 =1 then 1 else 0 end deffpd30,
case when obs_min_inst_def30 >=2 and min_inst_def30 in (1,2) then 1 else 0 end deffspd30,
case when obs_min_inst_def30 >=3 and min_inst_def30 in (1,2,3) then 1 else 0 end deffstpd30,
case when obs_min_inst_def0 >= 1 then 1 else 0 end flg_mature_fpd0,
case when obs_min_inst_def10 >=1 then 1 else 0 end flg_mature_fpd10,
case when obs_min_inst_def30 >=1 then 1 else 0 end flg_mature_fpd30,
case when obs_min_inst_def30 >=2 then 1 else 0 end flg_mature_fspd_30,
case when obs_min_inst_def30 >=3 then 1 else 0 end flg_mature_fstpd_30
from prj-prod-dataplatform.risk_credit_mis.loan_deliquency_data),
base as 
  (select distinct r.customerId,
  r.digitalLoanAccountId,
  loanmaster.loanAccountNumber,
  r.modelDisplayName,
  r.Beta_cash_stack_score,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month, 
  'Test' Data_selection,  
  del.deffstpd30,
  del.flg_mature_fstpd_30,
  loanmaster.new_loan_type,
  modelVersionId, trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile' 
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall' 
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
  from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  inner join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
   left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
  where loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and r.Beta_cash_stack_score is not null
  and del.flg_mature_fstpd_30 = 1
  )
  select *  from base
  ;
"""
dfd = client.query(sq).to_dataframe()
dfd = dfd.drop_duplicates(keep="first")
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()


# %%
df1 = dfd.copy()

# %% [markdown]
# ### Train

# %%
sq = """
WITH parsed as (
select customerId, digitalLoanAccountId,modelDisplayName,modelVersionId,start_time,end_time,prediction,trenchCategory,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeatures,
FROM prj-prod-dataplatform.dap_ds_poweruser_playground.ml_training_model_run_details
where modelDisplayName in  ('Beta-Cash-Stack-Model', 'beta_stack_model_cash')
),
  modelname as (
  select customerId,
  digitalLoanAccountId,
  start_time,
  prediction Beta_cash_stack_score,
    case when modelDisplayName like 'Beta-Cash-Stack-Model' then 'beta_stack_model_cash' else modelDisplayName end as modelDisplayName,
    modelVersionId,
  trenchCategory
  from parsed
  )
,
  deliquency as
(select loanAccountNumber,
case when obs_min_inst_def0 >= 1 and min_inst_def0 = 1 then 1 else 0 end deffpd0,
case when obs_min_inst_def10 >=1 and min_inst_def10 =1 then 1 else 0 end deffpd10,
case when obs_min_inst_def30 >=1 and min_inst_def30 =1 then 1 else 0 end deffpd30,
case when obs_min_inst_def30 >=2 and min_inst_def30 in (1,2) then 1 else 0 end deffspd30,
case when obs_min_inst_def30 >=3 and min_inst_def30 in (1,2,3) then 1 else 0 end deffstpd30,
case when obs_min_inst_def0 >= 1 then 1 else 0 end flg_mature_fpd0,
case when obs_min_inst_def10 >=1 then 1 else 0 end flg_mature_fpd10,
case when obs_min_inst_def30 >=1 then 1 else 0 end flg_mature_fpd30,
case when obs_min_inst_def30 >=2 then 1 else 0 end flg_mature_fspd_30,
case when obs_min_inst_def30 >=3 then 1 else 0 end flg_mature_fstpd_30
from prj-prod-dataplatform.risk_credit_mis.loan_deliquency_data),
base as
(select distinct r.customerId,
  r.digitalLoanAccountId,
  loanmaster.loanAccountNumber,
  r.modelDisplayName,
  r.Beta_cash_stack_score,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month, 
  'Train' Data_selection,  
    del.deffstpd30,
  del.flg_mature_fstpd_30,
  loanmaster.new_loan_type,
  modelVersionId,
    trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile' 
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall' 
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
    from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  inner join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
  left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
  where loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and r.Beta_cash_stack_score is not null
  and del.flg_mature_fstpd_30 = 1
  )
  select * from base;
  """
dfd = client.query(sq).to_dataframe()
dfd = dfd.drop_duplicates(keep="first")
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()

# %%
df2 = dfd.copy()

# %% [markdown]
# ### Concat

# %%
# df_concat = pd.concat([df1, df2], ignore_index=True)
# print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")

# df_concat = (df_concat
#              .sort_values(by=['digitalLoanAccountId', 'Data_selection'],
#                           key=lambda s: s.map({'Train': 0, 'Test': 1}))
#              .drop_duplicates(subset=['digitalLoanAccountId'], keep='first')
#              .reset_index(drop=True))
# print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")
# df_concat.head()


# 1) Get all IDs present in Train
train_ids = set(df2["digitalLoanAccountId"])

# 2) Keep only Test rows whose ID is NOT in Train
df1_no_dupes = df1[~df1["digitalLoanAccountId"].isin(train_ids)]

# 3) Concatenate
df_concat = pd.concat([df1_no_dupes, df2], ignore_index=True)

print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")
df_concat.head()


# %%
df_concat["Beta_cash_stack_score"] = pd.to_numeric(
    df_concat["Beta_cash_stack_score"], errors="coerce"
)

# %%
fact_table, dimension_table = calculate_periodic_gini_prod_ver_trench_dimfact(
    df_concat,
    "Beta_cash_stack_score",
    "deffstpd30",
    "FSTPD30",
    model_version_column="modelVersionId",
    trench_column="trenchCategory",
    loan_type_column="new_loan_type",
    loan_product_type_column="loan_product_type",
    account_id_column="digitalLoanAccountId",
)

# %%
fact_table, dimension_table = update_tables(
    fact_table, dimension_table, model_name="beta_stack_model_cash", product="CASH"
)
print(f"The shape of the fact table is:\t {fact_table.shape}")
print(f"The shape of the dimension table is:\t {dimension_table.shape}")

# %%
# Upload to BigQuery
table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_table"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(fact_table, table_id, job_config=job_config)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_table"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(dimension_table, table_id, job_config=job_config)
job.result()  # Wait for the job to complete

# %%


# %% [markdown]
# # End

# %% [markdown]
#

# %% [markdown]
#
