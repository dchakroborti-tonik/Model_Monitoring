# %% [markdown]
# # Define Library

import io
import os
import pickle
import tempfile
import time
import uuid
from datetime import datetime
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

# %% [markdown]
## Configure Settings
# Set options or configurations as needed
pd.set_option("display.max_columns", None)
pd.set_option("Display.max_rows", 100)

# %% [markdown]
# # Function

# %% [markdown]
# ## expand_calc_features

import json

# %%
import pandas as pd


def expand_calc_features(df):
    """
    Expand the calcFeatures JSON column into separate columns and return the complete DataFrame.

    Parameters:
    df (pd.DataFrame): Input DataFrame with calcFeatures column containing JSON data

    Returns:
    pd.DataFrame: Expanded DataFrame with all original columns plus JSON features as separate columns
    """

    # Make a copy to avoid modifying the original DataFrame
    df_expanded = df.copy()

    # Parse the calcFeatures JSON column
    calc_features_list = []

    for idx, calc_features_str in enumerate(df["calcFeatures"]):
        try:
            # Parse the JSON string
            features_dict = json.loads(
                calc_features_str.replace("'", '"')
            )  # Replace single quotes with double quotes for valid JSON
            calc_features_list.append(features_dict)
        except (json.JSONDecodeError, AttributeError) as e:
            # If parsing fails, create an empty dict and print warning
            print(f"Warning: Could not parse calcFeatures at index {idx}: {e}")
            calc_features_list.append({})

    # Create DataFrame from the parsed JSON data
    calc_features_df = pd.DataFrame(calc_features_list)

    # Add prefix to JSON-derived columns to avoid conflicts
    calc_features_df = calc_features_df.add_prefix("calc_")

    # Reset index to ensure proper alignment
    df_expanded = df_expanded.reset_index(drop=True)
    calc_features_df = calc_features_df.reset_index(drop=True)

    # Combine original DataFrame with expanded calcFeatures
    result_df = pd.concat([df_expanded, calc_features_df], axis=1)

    return result_df


# %% [markdown]
# ## expand_calc_features_robust

import json

# %%
import pandas as pd


def expand_calc_features_robust(df):
    """
    Expand the calcFeatures JSON column into separate columns with better error handling.

    Parameters:
    df (pd.DataFrame): Input DataFrame with calcFeatures column containing JSON data

    Returns:
    pd.DataFrame: Expanded DataFrame with all original columns plus JSON features as separate columns
    """

    # Make a copy to avoid modifying the original DataFrame
    df_expanded = df.copy()

    # Parse the calcFeatures JSON column
    calc_features_data = []

    for idx, row in df.iterrows():
        calc_features_str = row["calcFeatures"]

        if pd.isna(calc_features_str) or calc_features_str == "":
            calc_features_data.append({})
            continue

        try:
            # Clean the string and parse JSON
            cleaned_str = (
                calc_features_str.replace("'", '"')
                .replace("None", "null")
                .replace("True", "true")
                .replace("False", "false")
            )
            features_dict = json.loads(cleaned_str)
            calc_features_data.append(features_dict)
        except Exception as e:
            print(f"Warning: Could not parse calcFeatures at index {idx}: {e}")
            print(
                f"Problematic string: {calc_features_str[:100]}..."
            )  # Print first 100 chars
            calc_features_data.append({})

    # Create DataFrame from the parsed JSON data
    calc_features_df = pd.DataFrame(calc_features_data)

    # Add prefix to JSON-derived columns to avoid conflicts with existing columns
    calc_features_df = calc_features_df.add_prefix("feat_")

    # Combine DataFrames
    result_df = pd.concat([df_expanded, calc_features_df], axis=1)

    print(f"Original DataFrame shape: {df.shape}")
    print(f"Expanded DataFrame shape: {result_df.shape}")
    print(f"Added {len(calc_features_df.columns)} new columns from calcFeatures")

    return result_df


import warnings
from typing import Dict, List, Tuple

import numpy as np

# %%
import pandas as pd

warnings.filterwarnings("ignore")


def identify_feature_types(
    df: pd.DataFrame, feature_list: List[str]
) -> Dict[str, List[str]]:
    """
    Identify categorical and numerical features from the feature list.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    feature_list : List[str]
        List of features to classify

    Returns:
    --------
    Dict with 'categorical' and 'numerical' keys containing respective feature lists
    """
    categorical_features = []
    numerical_features = []

    for feature in feature_list:
        if feature not in df.columns:
            print(f"Warning: Feature '{feature}' not found in dataframe")
            continue

        # Check if feature is numeric
        if pd.api.types.is_numeric_dtype(df[feature]):
            # If unique values are less than 15 and all integers, treat as categorical
            unique_vals = df[feature].nunique()
            if (
                unique_vals < 15
                and df[feature]
                .dropna()
                .apply(lambda x: x == int(x) if isinstance(x, (int, float)) else False)
                .all()
            ):
                categorical_features.append(feature)
            else:
                numerical_features.append(feature)
        else:
            categorical_features.append(feature)

    return {"categorical": categorical_features, "numerical": numerical_features}


def create_bins_for_features(
    df: pd.DataFrame,
    numerical_features: List[str],
    categorical_features: List[str],
    base_month: str,
    month_col: str = "Application_month",
) -> Dict:
    """
    Create bins for numerical features (deciles) and categorical features (top 6 + others).

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    numerical_features : List[str]
        List of numerical features
    categorical_features : List[str]
        List of categorical features
    base_month : str
        Base month to determine binning strategy
    month_col : str
        Name of month column

    Returns:
    --------
    Dictionary containing binning information for each feature
    """
    base_df = df[df[month_col] == base_month].copy()
    binning_info = {}

    # Create bins for numerical features (deciles)
    for feature in numerical_features:
        valid_data = base_df[feature].dropna()

        if len(valid_data) == 0:
            binning_info[feature] = {
                "type": "numerical",
                "bins": None,
                "bin_ranges": {},
            }
            continue

        # Create decile bins
        try:
            bins = np.percentile(valid_data, np.arange(0, 101, 10))
            # Remove duplicates and sort
            bins = np.unique(bins)

            # Create bin ranges dictionary
            bin_ranges = {}
            for i in range(len(bins) - 1):
                bin_name = f"Bin_{i+1}"
                bin_ranges[bin_name] = {
                    "min": bins[i],
                    "max": bins[i + 1],
                    "range_str": f"[{bins[i]:.2f}, {bins[i+1]:.2f}]",
                }

            binning_info[feature] = {
                "type": "numerical",
                "bins": bins,
                "bin_ranges": bin_ranges,
            }
        except:
            binning_info[feature] = {
                "type": "numerical",
                "bins": None,
                "bin_ranges": {},
            }

    # Create bins for categorical features (top 6 + others)
    for feature in categorical_features:
        value_counts = base_df[feature].value_counts()
        top_6 = value_counts.nlargest(6).index.tolist()

        binning_info[feature] = {
            "type": "categorical",
            "top_categories": top_6,
            "bin_ranges": {},  # No ranges for categorical
        }

    return binning_info


def apply_binning(df: pd.DataFrame, feature: str, binning_info: Dict) -> pd.Series:
    """
    Apply binning to a feature based on binning information.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    feature : str
        Feature name
    binning_info : Dict
        Binning information for the feature

    Returns:
    --------
    pd.Series with binned values
    """
    if binning_info["type"] == "numerical":
        if binning_info["bins"] is None:
            return pd.Series(["Missing"] * len(df), index=df.index)

        bins = binning_info["bins"]
        labels = [f"Bin_{i+1}" for i in range(len(bins) - 1)]

        binned = pd.cut(
            df[feature],
            bins=bins,
            labels=labels,
            include_lowest=True,
            duplicates="drop",
        )

        # Handle nulls - convert to string and then replace
        binned = binned.astype(str)
        binned[df[feature].isna()] = "Missing"

        return binned

    else:  # categorical
        top_cats = binning_info["top_categories"]

        # Convert categorical to object if needed to avoid category errors
        if pd.api.types.is_categorical_dtype(df[feature]):
            feature_data = df[feature].astype(str)
        else:
            feature_data = df[feature].astype(str)

        # Replace NaN string representation with 'Missing'
        feature_data = feature_data.replace("nan", "Missing")

        # Apply binning logic
        binned = feature_data.apply(
            lambda x: (
                x if x in top_cats else ("Missing" if x == "Missing" else "Others")
            )
        )

        return binned


def calculate_psi(
    expected_pct: pd.Series, actual_pct: pd.Series, epsilon: float = 0.0001
) -> float:
    """
    Calculate Population Stability Index.

    Parameters:
    -----------
    expected_pct : pd.Series
        Expected (baseline) percentages
    actual_pct : pd.Series
        Actual percentages
    epsilon : float
        Small value to avoid log(0)

    Returns:
    --------
    PSI value
    """
    # Align indices
    all_bins = expected_pct.index.union(actual_pct.index)
    expected_pct = expected_pct.reindex(all_bins, fill_value=0)
    actual_pct = actual_pct.reindex(all_bins, fill_value=0)

    # Add epsilon to avoid log(0)
    expected_pct = expected_pct + epsilon
    actual_pct = actual_pct + epsilon

    # Calculate PSI
    psi_value = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))

    return psi_value


def calculate_month_on_month_psi(
    df: pd.DataFrame,
    feature_list: List[str],
    segment_columns: List[str],
    month_col: str = "Application_month",
) -> pd.DataFrame:
    """
    Calculate month-on-month PSI for each feature, overall and by segments.
    Each row shows PSI comparing first month vs each subsequent month.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    feature_list : List[str]
        List of features to calculate PSI for
    segment_columns : List[str]
        List of segment columns
    month_col : str
        Name of month column

    Returns:
    --------
    pd.DataFrame with PSI values with one row per feature-month-segment combination
    """
    # Create a copy to avoid modifying original
    df = df.copy()

    # Identify feature types
    feature_types = identify_feature_types(df, feature_list)

    # Get sorted months
    months = sorted(df[month_col].unique())
    base_month = months[0]

    # Create binning strategy based on base month
    binning_info = create_bins_for_features(
        df,
        feature_types["numerical"],
        feature_types["categorical"],
        base_month,
        month_col,
    )

    results = []

    # Calculate overall PSI
    for feature in feature_list:
        if feature not in df.columns:
            continue

        # Apply binning
        df[f"{feature}_binned"] = apply_binning(df, feature, binning_info[feature])

        # Get base month distribution
        base_dist = df[df[month_col] == base_month][f"{feature}_binned"].value_counts(
            normalize=True
        )

        # For each month (including base month for reference)
        for month in months:
            actual_dist = df[df[month_col] == month][f"{feature}_binned"].value_counts(
                normalize=True
            )
            psi_value = (
                calculate_psi(base_dist, actual_dist) if month != base_month else 0.0
            )

            results.append(
                {
                    "Feature": feature,
                    "Feature_Type": binning_info[feature]["type"],
                    "Segment_Column": "Overall",
                    "Segment_Value": "All",
                    "Month": f"{month}",
                    "Base_Month": base_month,
                    "Current_Month": month,
                    "PSI": psi_value,
                }
            )

    # Calculate PSI by segments
    for segment_col in segment_columns:
        if segment_col not in df.columns:
            continue

        segments = df[segment_col].dropna().unique()

        for segment_val in segments:
            segment_df = df[df[segment_col] == segment_val]

            for feature in feature_list:
                if feature not in df.columns:
                    continue

                # Get base month distribution for segment
                base_segment = segment_df[segment_df[month_col] == base_month]
                if len(base_segment) == 0:
                    continue

                base_dist = base_segment[f"{feature}_binned"].value_counts(
                    normalize=True
                )

                # For each month (including base month for reference)
                for month in months:
                    actual_segment = segment_df[segment_df[month_col] == month]
                    if len(actual_segment) == 0:
                        continue

                    actual_dist = actual_segment[f"{feature}_binned"].value_counts(
                        normalize=True
                    )
                    psi_value = (
                        calculate_psi(base_dist, actual_dist)
                        if month != base_month
                        else 0.0
                    )

                    results.append(
                        {
                            "Feature": feature,
                            "Feature_Type": binning_info[feature]["type"],
                            "Segment_Column": segment_col,
                            "Segment_Value": segment_val,
                            "Month": f"{month}",
                            "Base_Month": base_month,
                            "Current_Month": month,
                            "PSI": psi_value,
                        }
                    )

    return pd.DataFrame(results)


def calculate_bin_level_psi(
    df: pd.DataFrame,
    feature_list: List[str],
    segment_columns: List[str],
    month_col: str = "Application_month",
) -> pd.DataFrame:
    """
    Calculate bin-level PSI for each feature, overall and by segments.
    Each row shows bin-level PSI comparing first month vs each month.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    feature_list : List[str]
        List of features to calculate PSI for
    segment_columns : List[str]
        List of segment columns
    month_col : str
        Name of month column

    Returns:
    --------
    pd.DataFrame with bin-level PSI details including bin ranges
    """
    # Create a copy to avoid modifying original
    df = df.copy()

    # Identify feature types
    feature_types = identify_feature_types(df, feature_list)

    # Get sorted months
    months = sorted(df[month_col].unique())
    base_month = months[0]

    # Create binning strategy based on base month
    binning_info = create_bins_for_features(
        df,
        feature_types["numerical"],
        feature_types["categorical"],
        base_month,
        month_col,
    )

    results = []
    epsilon = 0.0001

    # Calculate overall bin-level PSI
    for feature in feature_list:
        if feature not in df.columns:
            continue

        # Apply binning
        df[f"{feature}_binned"] = apply_binning(df, feature, binning_info[feature])

        # Get base month distribution
        base_dist = df[df[month_col] == base_month][f"{feature}_binned"].value_counts(
            normalize=True
        )

        # Calculate bin-level PSI for each month (including base month)
        for month in months:
            actual_dist = df[df[month_col] == month][f"{feature}_binned"].value_counts(
                normalize=True
            )

            # Get all bins
            all_bins = base_dist.index.union(actual_dist.index)

            for bin_name in all_bins:
                expected_pct = base_dist.get(bin_name, 0) + epsilon
                actual_pct = actual_dist.get(bin_name, 0) + epsilon

                bin_psi = (
                    (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)
                    if month != base_month
                    else 0.0
                )

                # Get bin range information
                bin_ranges = binning_info[feature]["bin_ranges"]
                if bin_name in bin_ranges:
                    bin_min = bin_ranges[bin_name]["min"]
                    bin_max = bin_ranges[bin_name]["max"]
                    bin_range = bin_ranges[bin_name]["range_str"]
                else:
                    # For categorical or special bins (Missing, Others)
                    bin_min = None
                    bin_max = None
                    bin_range = bin_name

                results.append(
                    {
                        "Feature": feature,
                        "Feature_Type": binning_info[feature]["type"],
                        "Segment_Column": "Overall",
                        "Segment_Value": "All",
                        "Month": f"{month}",
                        "Base_Month": base_month,
                        "Current_Month": month,
                        "Bin": bin_name,
                        "Bin_Range": bin_range,
                        "Bin_Min": bin_min,
                        "Bin_Max": bin_max,
                        "Base_Percentage": (base_dist.get(bin_name, 0) * 100),
                        "Actual_Percentage": (actual_dist.get(bin_name, 0) * 100),
                        "Bin_PSI": bin_psi,
                    }
                )

    # Calculate bin-level PSI by segments
    for segment_col in segment_columns:
        if segment_col not in df.columns:
            continue

        segments = df[segment_col].dropna().unique()

        for segment_val in segments:
            segment_df = df[df[segment_col] == segment_val]

            for feature in feature_list:
                if feature not in df.columns:
                    continue

                # Get base month distribution for segment
                base_segment = segment_df[segment_df[month_col] == base_month]
                if len(base_segment) == 0:
                    continue

                base_dist = base_segment[f"{feature}_binned"].value_counts(
                    normalize=True
                )

                # Calculate bin-level PSI for each month (including base month)
                for month in months:
                    actual_segment = segment_df[segment_df[month_col] == month]
                    if len(actual_segment) == 0:
                        continue

                    actual_dist = actual_segment[f"{feature}_binned"].value_counts(
                        normalize=True
                    )

                    # Get all bins
                    all_bins = base_dist.index.union(actual_dist.index)

                    for bin_name in all_bins:
                        expected_pct = base_dist.get(bin_name, 0) + epsilon
                        actual_pct = actual_dist.get(bin_name, 0) + epsilon

                        bin_psi = (
                            (actual_pct - expected_pct)
                            * np.log(actual_pct / expected_pct)
                            if month != base_month
                            else 0.0
                        )

                        # Get bin range information
                        bin_ranges = binning_info[feature]["bin_ranges"]
                        if bin_name in bin_ranges:
                            bin_min = bin_ranges[bin_name]["min"]
                            bin_max = bin_ranges[bin_name]["max"]
                            bin_range = bin_ranges[bin_name]["range_str"]
                        else:
                            # For categorical or special bins (Missing, Others)
                            bin_min = None
                            bin_max = None
                            bin_range = bin_name

                        results.append(
                            {
                                "Feature": feature,
                                "Feature_Type": binning_info[feature]["type"],
                                "Segment_Column": segment_col,
                                "Segment_Value": segment_val,
                                "Month": f"{month}",
                                "Base_Month": base_month,
                                "Current_Month": month,
                                "Bin": bin_name,
                                "Bin_Range": bin_range,
                                "Bin_Min": bin_min,
                                "Bin_Max": bin_max,
                                "Base_Percentage": (base_dist.get(bin_name, 0) * 100),
                                "Actual_Percentage": (
                                    actual_dist.get(bin_name, 0) * 100
                                ),
                                "Bin_PSI": bin_psi,
                            }
                        )

    return pd.DataFrame(results)


sq = r"""  
WITH parsed as (
  select customerId, digitalLoanAccountId,modelDisplayName,modelVersionId,start_time,end_time,prediction,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeatures,
--REPLACE(REPLACE(prediction, "'", '"'), "None", "null") AS prediction_clean
FROM `prj-prod-dataplatform.audit_balance.ml_model_run_details`
where modelDisplayName = 'Alpha-Cash-CIC-Model'
),
latest_request as (
select * from parsed
QUALIFY ROW_NUMBER() OVER (PARTITION BY customerId, digitalLoanAccountId,modelDisplayName ORDER BY start_time DESC ) = 1),

model_run as (
select customerId,digitalLoanAccountId,modelName, publish_time,requestPayload as requestPayload_clean
--REPLACE(REPLACE(requestPayload, "'", '"'), "None", "null") AS requestPayload_clean
from `prj-prod-dataplatform.audit_balance.ml_request_details` 
WHERE modelName = 'Alpha-Cash-Model-response'
QUALIFY ROW_NUMBER() OVER (PARTITION BY customerId, digitalLoanAccountId,modelName ORDER BY publish_time DESC ) = 1),
base as (
select * from (
  select 
 r.customerId,
 r.digitalLoanAccountId,
 r.prediction,
 r.start_time,
 r.end_time,
 r.modelDisplayName,
 r.modelVersionId,
 loanmaster.new_loan_type,
 loanmaster.gender,
 'Cash' as product,
 'Alpha-Cash-CIC-Model_All_Trench' Model_Name,
  REGEXP_EXTRACT(m.requestPayload_clean, r"osType[:=]['\"]?([^'\"]+)['\"]?") AS osType,
  REGEXP_EXTRACT(m.requestPayload_clean, r"loanType[:=]['\"]?([^'\"]+)['\"]?") AS loanType,
  REGEXP_EXTRACT(m.requestPayload_clean, r"trenchCategory[:=]['\"]?([^'\"]+)['\"]?") AS trenchCategory,
   SAFE_CAST(REGEXP_EXTRACT(m.requestPayload_clean, r"aStackScore[:= ]([0-9\.]+)") AS FLOAT64) AS aStackScore,
  SAFE_CAST(REGEXP_EXTRACT(m.requestPayload_clean, r"aCicScore[:= ]([0-9\.]+)") AS FLOAT64) AS aCicScore,
  calcFeatures,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  r.start_time) AS appln_submit_datetime,
  loanmaster.disbursementDateTime,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  r.start_time)) as Application_month, 
FROM latest_request r
left join model_run m
on r.digitalLoanAccountId = m.digitalLoanAccountId 
left join risk_credit_mis.loan_master_table loanmaster 
  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
qualify row_number() over (partition by r.customerId,r.digitalLoanAccountId order by r.start_time desc) = 1
 )
)
select * from base;
"""
dfd = client.query(sq).to_dataframe()
dfd.head()

# %% [markdown]
# ### expand df

# %%
# Expand the calcFeatures column
expanded_df = expand_calc_features(dfd)

# Display the result
print(f"Original columns: {dfd.shape[1]}")
print(f"Expanded columns: {expanded_df.shape[1]}")
expanded_df.head()


# %%
df = expanded_df.drop(columns=["calcFeatures"]).copy()

# %% [markdown]
# ### Drop table temp_csi_new_monitoring_data

# %%
sq = """drop table if exists prj-prod-dataplatform.dap_ds_poweruser_playground.temp_csi_new_monitoring_data;"""

job = client.query(sq)
job.result()  # Wait for job to complete
print(
    f"Table  prj-prod-dataplatform.dap_ds_poweruser_playground.temp_csi_new_monitoring_data dropped successfully."
)

sq = """drop table if exists prj-prod-dataplatform.dap_ds_poweruser_playground.temp_csi_new_monitoring_data_feature_bin_level;"""

job = client.query(sq)
job.result()  # Wait for job to complete
print(
    f"Table  prj-prod-dataplatform.dap_ds_poweruser_playground.temp_csi_new_monitoring_data_feature_bin_level dropped successfully."
)

# %% [markdown]
# ### Run the PSI calculation

# %%
df = df.copy()

# Define feature list
feature_list = [
    "aCicScore",
    "calc_cic_max_age_all_contracts_snapshot",
    "calc_cic_ratio_overdue_contracts_to_granted_contracts",
    "calc_cic_ScoreRange",
    "calc_cic_ln_loan_level_user_type",
    "calc_cic_has_ever_been_overdue",
    "calc_cic_latest_granted_contract_overdue_flag",
    "calc_cic_ratio_closed_over_new_granted_cnt_24M",
    "calc_cic_ratio_risky_contracts_to_granted_contracts",
    "calc_cic_Short_and_Term_Loans_granted_contracts_cnt_24M",
    "calc_cic_flg_zero_non_granted_ever",
    "calc_cic_Personal_Loans_granted_contracts_amt_24M",
    "calc_cic_CreditAvgCreditLimit",
    "calc_cic_flg_zero_granted_ever",
]

# Define segment columns
segment_columns = ["new_loan_type", "gender", "osType", "trenchCategory", "product"]
# Calculate month-on-month PSI
psi_results = calculate_month_on_month_psi(df, feature_list, segment_columns)
psi_results["modelDisplayName"] = df["modelDisplayName"].iloc[0]
psi_results["Model_Name"] = df["Model_Name"].iloc[0]
psi_results["modelVersionId"] = df["modelVersionId"].iloc[0]
psi_results["trenchCategory"] = df["trenchCategory"].iloc[0]
psi_results = psi_results[
    [
        "modelDisplayName",
        "Model_Name",
        "modelVersionId",
        "trenchCategory",
        "Feature",
        "Feature_Type",
        "Segment_Column",
        "Segment_Value",
        "Month",
        "Base_Month",
        "Current_Month",
        "PSI",
    ]
].copy()

# Calculate bin-level PSI
bin_psi_results = calculate_bin_level_psi(df, feature_list, segment_columns)
bin_psi_results["modelDisplayName"] = df["modelDisplayName"].iloc[0]
bin_psi_results["Model_Name"] = df["Model_Name"].iloc[0]
bin_psi_results["modelVersionId"] = df["modelVersionId"].iloc[0]
bin_psi_results["trenchCategory"] = df["trenchCategory"].iloc[0]
bin_psi_results = bin_psi_results[
    [
        "modelDisplayName",
        "Model_Name",
        "modelVersionId",
        "trenchCategory",
        "Feature",
        "Feature_Type",
        "Segment_Column",
        "Segment_Value",
        "Month",
        "Base_Month",
        "Current_Month",
        "Bin",
        "Bin_Range",
        "Bin_Min",
        "Bin_Max",
        "Base_Percentage",
        "Actual_Percentage",
        "Bin_PSI",
    ]
].copy()

# Upload to BigQuery
table_id = (
    "prj-prod-dataplatform.dap_ds_poweruser_playground.temp_csi_new_monitoring_data"
)
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_TRUNCATE",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(psi_results, table_id, job_config=job_config)
job.result()  # Wait for the job to complete


# Upload to BigQuery
table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.temp_csi_new_monitoring_data_feature_bin_level"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_TRUNCATE",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(bin_psi_results, table_id, job_config=job_config)
job.result()  # Wait for the job to complete

# %%
bin_psi_results.columns

# %% [markdown]
# ## Alpha-Cash-Stack-Model

# %%
sq = r""" 
WITH parsed as (
  select customerId, digitalLoanAccountId,modelDisplayName,modelVersionId,start_time,end_time,prediction,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeatures,
--REPLACE(REPLACE(prediction, "'", '"'), "None", "null") AS prediction_clean
FROM `prj-prod-dataplatform.audit_balance.ml_model_run_details`
where modelDisplayName = 'Alpha-Cash-Stack-Model'
),
latest_request as (
select * from parsed
QUALIFY ROW_NUMBER() OVER (PARTITION BY customerId, digitalLoanAccountId,modelDisplayName ORDER BY start_time DESC ) = 1),

model_run as (
select customerId,digitalLoanAccountId,modelName, publish_time,requestPayload as requestPayload_clean
--REPLACE(REPLACE(requestPayload, "'", '"'), "None", "null") AS requestPayload_clean
from `prj-prod-dataplatform.audit_balance.ml_request_details` 
WHERE modelName = 'Alpha-Cash-Model-response'
QUALIFY ROW_NUMBER() OVER (PARTITION BY customerId, digitalLoanAccountId,modelName ORDER BY publish_time DESC ) = 1),
base as (
select * from (
  select 
 r.customerId,
 r.digitalLoanAccountId,
 r.prediction ,
 r.start_time,
 r.end_time,
 r.modelDisplayName,
 r.modelVersionId,
 loanmaster.new_loan_type,
 loanmaster.gender,
 'Alpha-Cash-Stack-Model_Trench1' Model_Name,
 'Cash' as product,
  REGEXP_EXTRACT(m.requestPayload_clean, r"osType[:=]['\"]?([^'\"]+)['\"]?") AS osType,
  REGEXP_EXTRACT(m.requestPayload_clean, r"loanType[:=]['\"]?([^'\"]+)['\"]?") AS loanType,
  REGEXP_EXTRACT(m.requestPayload_clean, r"trenchCategory[:=]['\"]?([^'\"]+)['\"]?") AS trenchCategory,
   SAFE_CAST(REGEXP_EXTRACT(m.requestPayload_clean, r"aStackScore[:= ]([0-9\.]+)") AS FLOAT64) AS aStackScore,
  SAFE_CAST(REGEXP_EXTRACT(m.requestPayload_clean, r"aCicScore[:= ]([0-9\.]+)") AS FLOAT64) AS aCicScore,
  calcFeatures,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  r.start_time) AS appln_submit_datetime,
  loanmaster.disbursementDateTime,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  r.start_time)) as Application_month, 
FROM latest_request r
left join model_run m
on r.digitalLoanAccountId = m.digitalLoanAccountId 
left join risk_credit_mis.loan_master_table loanmaster 
  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
qualify row_number() over (partition by r.customerId,r.digitalLoanAccountId order by r.start_time desc) = 1
 )
)
select * from base where trenchCategory = 'Trench 1';
"""

dfd = client.query(sq).to_dataframe()
dfd.head()

# %% [markdown]
# ### Expand df

# %%
# Expand the calcFeatures column
expanded_df = expand_calc_features(dfd)

# Display the result
print(f"Original columns: {dfd.shape[1]}")
print(f"Expanded columns: {expanded_df.shape[1]}")
expanded_df.head()


# %%
df = expanded_df.drop(columns=["calcFeatures"]).copy()

# %%
# sq = r"""WITH parsed as (
#   select customerId, digitalLoanAccountId,modelDisplayName,modelVersionId,start_time,end_time,prediction,
# REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeatures,
# FROM `prj-prod-dataplatform.audit_balance.ml_model_run_details`
# where modelDisplayName = 'Alpha-Cash-Stack-Model'),
# latest_request as (
# select * from parsed
# QUALIFY ROW_NUMBER() OVER (PARTITION BY customerId, digitalLoanAccountId,modelDisplayName ORDER BY start_time DESC ) = 1),
# model_run as (
# select customerId,digitalLoanAccountId,modelName, publish_time,requestPayload as requestPayload_clean
# from `prj-prod-dataplatform.audit_balance.ml_request_details`
# WHERE modelName = 'Alpha-Cash-Model-response'
# QUALIFY ROW_NUMBER() OVER (PARTITION BY customerId, digitalLoanAccountId,modelName ORDER BY publish_time DESC ) = 1)
# select * from (
#   select
#  r.customerId,
#  r.digitalLoanAccountId,
#  r.prediction Alpha_Cash_Stack_Score,
#  r.start_time,
#  r.end_time,
#  r.modelDisplayName,
#  r.modelVersionId,
#   loanmaster.new_loan_type,
#  loanmaster.gender,

#   REGEXP_EXTRACT(m.requestPayload_clean, r"osType[:=]['\"]?([^'\"]+)['\"]?") AS osType,
#   REGEXP_EXTRACT(m.requestPayload_clean, r"loanType[:=]['\"]?([^'\"]+)['\"]?") AS loanType,
#   REGEXP_EXTRACT(m.requestPayload_clean, r"trenchCategory[:=]['\"]?([^'\"]+)['\"]?") AS trenchCategory,

#  SAFE_CAST(JSON_VALUE(r.calcFeatures, "$.apps_score") AS FLOAT64) AS  apps_score,
#  SAFE_CAST(JSON_VALUE(r.calcFeatures, "$.c_demo_score") AS FLOAT64) AS  c_demo_score,
#  SAFE_CAST(JSON_VALUE(r.calcFeatures, "$.c_credo_score") AS FLOAT64) AS  c_credo_score,
#  SAFE_CAST(JSON_VALUE(r.calcFeatures, "$.c_tx_score") AS FLOAT64) AS  c_tx_score,
#  SAFE_CAST(JSON_VALUE(r.calcFeatures, "$.ca_cic_score") AS FLOAT64) AS  ca_cic_score,
# coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  r.start_time) AS appln_submit_datetime,
#   loanmaster.disbursementDateTime,
#   format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  r.start_time)) as Application_month,
# FROM latest_request r
# left join model_run m
# on r.digitalLoanAccountId = m.digitalLoanAccountId
# left join risk_credit_mis.loan_master_table loanmaster
#   ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
# ) where trenchCategory = 'Trench 1'
# ;
# """

# df = client.query(sq).to_dataframe()
# df.head()

# %% [markdown]
# ### run the psi calculation

# %%
df = df.copy()

# Define feature list
feature_list = [
    "aStackScore",
    "calc_apps_score",
    "calc_c_demo_score",
    "calc_c_credo_score",
    "calc_c_tx_score",
    "calc_ca_cic_score",
]

# Define segment columns
segment_columns = ["new_loan_type", "gender", "osType", "trenchCategory", "product"]
# Calculate month-on-month PSI
psi_results = calculate_month_on_month_psi(df, feature_list, segment_columns)
psi_results["modelDisplayName"] = df["modelDisplayName"].iloc[0]
psi_results["Model_Name"] = df["Model_Name"].iloc[0]
psi_results["modelVersionId"] = df["modelVersionId"].iloc[0]
psi_results["trenchCategory"] = df["trenchCategory"].iloc[0]
psi_results = psi_results[
    [
        "modelDisplayName",
        "Model_Name",
        "modelVersionId",
        "trenchCategory",
        "Feature",
        "Feature_Type",
        "Segment_Column",
        "Segment_Value",
        "Month",
        "Base_Month",
        "Current_Month",
        "PSI",
    ]
].copy()

# Calculate bin-level PSI
bin_psi_results = calculate_bin_level_psi(df, feature_list, segment_columns)
bin_psi_results["modelDisplayName"] = df["modelDisplayName"].iloc[0]
bin_psi_results["Model_Name"] = df["Model_Name"].iloc[0]
bin_psi_results["modelVersionId"] = df["modelVersionId"].iloc[0]
bin_psi_results["trenchCategory"] = df["trenchCategory"].iloc[0]
bin_psi_results = bin_psi_results[
    [
        "modelDisplayName",
        "Model_Name",
        "modelVersionId",
        "trenchCategory",
        "Feature",
        "Feature_Type",
        "Segment_Column",
        "Segment_Value",
        "Month",
        "Base_Month",
        "Current_Month",
        "Bin",
        "Bin_Range",
        "Bin_Min",
        "Bin_Max",
        "Base_Percentage",
        "Actual_Percentage",
        "Bin_PSI",
    ]
].copy()

# Upload to BigQuery
table_id = (
    "prj-prod-dataplatform.dap_ds_poweruser_playground.temp_csi_new_monitoring_data"
)
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(psi_results, table_id, job_config=job_config)
job.result()  # Wait for the job to complete


# Upload to BigQuery
table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.temp_csi_new_monitoring_data_feature_bin_level"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(bin_psi_results, table_id, job_config=job_config)
job.result()  # Wait for the job to complete

# %%
psi_results

# %% [markdown]
# ## Beta-Cash-Demo-Model (Trench1,2,3)

# %%
sq = r"""
WITH parsed as (
  select customerId, digitalLoanAccountId,modelDisplayName,modelVersionId,start_time,end_time,prediction,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeatures
FROM `prj-prod-dataplatform.audit_balance.ml_model_run_details`
where modelDisplayName = 'Beta-Cash-Demo-Model'),

latest_request as (
select * from parsed
QUALIFY ROW_NUMBER() OVER (PARTITION BY customerId, digitalLoanAccountId,modelDisplayName ORDER BY start_time DESC ) = 1),

model_run as (
select customerId,digitalLoanAccountId,modelName, publish_time,
REPLACE(REPLACE(requestPayload, "'", '"'), "None", "null") AS requestPayload_clean
from `prj-prod-dataplatform.audit_balance.ml_request_details` 
WHERE modelName = 'Beta-Cash-Model-response'
QUALIFY ROW_NUMBER() OVER (PARTITION BY customerId, digitalLoanAccountId,modelName ORDER BY publish_time DESC ) = 1)

select * from (select 
 r.customerId,
 r.digitalLoanAccountId,
 r.prediction Beta_Cash_Demo_Score,
 r.start_time,
 r.end_time,
 r.modelDisplayName,
 r.modelVersionId,
 loanmaster.new_loan_type,
 loanmaster.gender,
  'Beta-Cash-Demo-Model_All_Trench' Model_Name,
  'Cash' as product,
  JSON_VALUE(m.requestPayload_clean, "$.predictions.trenchCategory") AS trenchCategory,
  JSON_VALUE(m.requestPayload_clean, "$.predictions.loanType") AS loanType,
  --  Demo Score Model Features for all Trench 1,2,3
  calcFeatures,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  r.start_time) AS appln_submit_datetime,
  loanmaster.disbursementDateTime,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  r.start_time)) as Application_month, 
FROM latest_request r
left join model_run m
on r.digitalLoanAccountId = m.digitalLoanAccountId 
left join risk_credit_mis.loan_master_table loanmaster 
  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
qualify row_number() over (partition by r.customerId,r.digitalLoanAccountId order by r.start_time desc) = 1
)
-- where trenchCategory = 'Trench 1'
;
"""

dfd = client.query(sq).to_dataframe()
dfd.head()

# %% [markdown]
# ### expand df

# %%
# Expand the calcFeatures column
expanded_df = expand_calc_features(dfd)

# Display the result
print(f"Original columns: {dfd.shape[1]}")
print(f"Expanded columns: {expanded_df.shape[1]}")
expanded_df.head()


# %%
df = expanded_df.drop(columns=["calcFeatures"]).copy()

# %%
bin_columns = [col for col in df.columns if "bin" in col.lower()]
df[bin_columns] = df[bin_columns].astype("category")

# %% [markdown]
# ### run the psi calculation

# %%
df = df.copy()

# Define feature list
feature_list = [
    "Beta_Cash_Demo_Score",
    "calc_ln_vas_opted_flag",
    "calc_ln_self_dec_income",
    "calc_ln_age",
    "calc_ln_source_funds_new_bin",
    "calc_ln_loan_level_user_type",
    "calc_ln_industry_new_cat_bin",
    "calc_ln_marital_status",
    "calc_ln_doc_type_rolled",
    "calc_ln_education_level",
    "calc_ln_ref2_type",
    "calc_ln_email_primary_domain",
    "calc_ln_os_type",
    "calc_ln_province_bin",
]

# Define segment columns
segment_columns = ["new_loan_type", "gender", "osType", "trenchCategory", "product"]
# Calculate month-on-month PSI
psi_results = calculate_month_on_month_psi(df, feature_list, segment_columns)
psi_results["modelDisplayName"] = df["modelDisplayName"].iloc[0]
psi_results["Model_Name"] = df["Model_Name"].iloc[0]
psi_results["modelVersionId"] = df["modelVersionId"].iloc[0]
psi_results["trenchCategory"] = df["trenchCategory"].iloc[0]
psi_results = psi_results[
    [
        "modelDisplayName",
        "Model_Name",
        "modelVersionId",
        "trenchCategory",
        "Feature",
        "Feature_Type",
        "Segment_Column",
        "Segment_Value",
        "Month",
        "Base_Month",
        "Current_Month",
        "PSI",
    ]
].copy()

# Calculate bin-level PSI
bin_psi_results = calculate_bin_level_psi(df, feature_list, segment_columns)
bin_psi_results["modelDisplayName"] = df["modelDisplayName"].iloc[0]
bin_psi_results["Model_Name"] = df["Model_Name"].iloc[0]
bin_psi_results["modelVersionId"] = df["modelVersionId"].iloc[0]
bin_psi_results["trenchCategory"] = df["trenchCategory"].iloc[0]
bin_psi_results = bin_psi_results[
    [
        "modelDisplayName",
        "Model_Name",
        "modelVersionId",
        "trenchCategory",
        "Feature",
        "Feature_Type",
        "Segment_Column",
        "Segment_Value",
        "Month",
        "Base_Month",
        "Current_Month",
        "Bin",
        "Bin_Range",
        "Bin_Min",
        "Bin_Max",
        "Base_Percentage",
        "Actual_Percentage",
        "Bin_PSI",
    ]
].copy()

# Upload to BigQuery
table_id = (
    "prj-prod-dataplatform.dap_ds_poweruser_playground.temp_csi_new_monitoring_data"
)
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(psi_results, table_id, job_config=job_config)
job.result()  # Wait for the job to complete


# Upload to BigQuery
table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.temp_csi_new_monitoring_data_feature_bin_level"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(bin_psi_results, table_id, job_config=job_config)
job.result()  # Wait for the job to complete

# %%
psi_results

# %% [markdown]
# ## Beta-Cash-AppScore-Model

# %%
sq = r"""  
WITH parsed as (
  select customerId, digitalLoanAccountId,modelDisplayName,modelVersionId,start_time,end_time,prediction,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeatures,
REPLACE(REPLACE(prediction, "'", '"'), "None", "null") AS prediction_clean
FROM `prj-prod-dataplatform.audit_balance.ml_model_run_details`
where modelDisplayName = 'Beta-Cash-AppScore-Model'),

latest_request as (
select * from parsed
QUALIFY ROW_NUMBER() OVER (PARTITION BY customerId, digitalLoanAccountId,modelDisplayName ORDER BY start_time DESC ) = 1),

model_run as (
select customerId,digitalLoanAccountId,modelName, publish_time,
REPLACE(REPLACE(requestPayload, "'", '"'), "None", "null") AS requestPayload_clean
from `prj-prod-dataplatform.audit_balance.ml_request_details` 
WHERE modelName = 'Beta-Cash-Model-response'
QUALIFY ROW_NUMBER() OVER (PARTITION BY customerId, digitalLoanAccountId,modelName ORDER BY publish_time DESC ) = 1)

select * from (
  select 
 r.customerId,
 r.digitalLoanAccountId,
 --r.prediction,
 r.start_time,
 r.end_time,
 r.modelDisplayName,
 r.modelVersionId,
 loanmaster.new_loan_type,
 loanmaster.gender,
 'Beta-Cash-AppScore-Model_Trench1' Model_Name,
 'Cash' as product,
  JSON_VALUE(m.requestPayload_clean, "$.predictions.trenchCategory") AS trenchCategory,
  JSON_VALUE(m.requestPayload_clean, "$.predictions.loanType") AS loanType,
  --  Beta App Score Model Features for all Trench 1
  SAFE_CAST(JSON_VALUE(r.prediction_clean, "$.combined_score") AS Float64) AS beta_cash_app_score,
  calcFeatures,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  r.start_time) AS appln_submit_datetime,
  loanmaster.disbursementDateTime,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  r.start_time)) as Application_month,  
FROM latest_request r
left join model_run m
on r.digitalLoanAccountId = m.digitalLoanAccountId 
left join risk_credit_mis.loan_master_table loanmaster 
  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
qualify row_number() over (partition by r.customerId,r.digitalLoanAccountId order by r.start_time desc) = 1
)
where trenchCategory = 'Trench 1'
;
"""

dfd = client.query(sq).to_dataframe()
dfd.head()

# %% [markdown]
# ### expand df

# %%
# Expand the calcFeatures column
expanded_df = expand_calc_features(dfd)

# Display the result
print(f"Original columns: {dfd.shape[1]}")
print(f"Expanded columns: {expanded_df.shape[1]}")
expanded_df.head()


# %%
df = expanded_df.drop(columns=["calcFeatures"]).copy()

# %% [markdown]
# ### run the psi calculation

# %%
df = df.copy()

# Define feature list
feature_list = [
    "beta_cash_app_score",
    "calc_app_cnt_health_and_fitness_ever",
    "calc_app_cnt_shopping_ever",
    "calc_app_cnt_crypto_ever",
    "calc_app_cnt_driver_ever",
    "calc_app_cnt_payday_180d",
    "calc_app_cnt_gambling_180d",
    "calc_app_avg_time_bw_installed_mins_3d",
    "calc_app_median_time_bw_installed_mins_3d",
    "calc_app_avg_time_bw_installed_mins_ever",
    "calc_app_median_time_bw_installed_mins_ever",
]

# Define segment columns
segment_columns = ["new_loan_type", "gender", "osType", "trenchCategory", "product"]
# Calculate month-on-month PSI
psi_results = calculate_month_on_month_psi(df, feature_list, segment_columns)
psi_results["modelDisplayName"] = df["modelDisplayName"].iloc[0]
psi_results["Model_Name"] = df["Model_Name"].iloc[0]
psi_results["modelVersionId"] = df["modelVersionId"].iloc[0]
psi_results["trenchCategory"] = df["trenchCategory"].iloc[0]
psi_results = psi_results[
    [
        "modelDisplayName",
        "Model_Name",
        "modelVersionId",
        "trenchCategory",
        "Feature",
        "Feature_Type",
        "Segment_Column",
        "Segment_Value",
        "Month",
        "Base_Month",
        "Current_Month",
        "PSI",
    ]
].copy()

# Calculate bin-level PSI
bin_psi_results = calculate_bin_level_psi(df, feature_list, segment_columns)
bin_psi_results["modelDisplayName"] = df["modelDisplayName"].iloc[0]
bin_psi_results["Model_Name"] = df["Model_Name"].iloc[0]
bin_psi_results["modelVersionId"] = df["modelVersionId"].iloc[0]
bin_psi_results["trenchCategory"] = df["trenchCategory"].iloc[0]
bin_psi_results = bin_psi_results[
    [
        "modelDisplayName",
        "Model_Name",
        "modelVersionId",
        "trenchCategory",
        "Feature",
        "Feature_Type",
        "Segment_Column",
        "Segment_Value",
        "Month",
        "Base_Month",
        "Current_Month",
        "Bin",
        "Bin_Range",
        "Bin_Min",
        "Bin_Max",
        "Base_Percentage",
        "Actual_Percentage",
        "Bin_PSI",
    ]
].copy()

# Upload to BigQuery
table_id = (
    "prj-prod-dataplatform.dap_ds_poweruser_playground.temp_csi_new_monitoring_data"
)
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(psi_results, table_id, job_config=job_config)
job.result()  # Wait for the job to complete


# Upload to BigQuery
table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.temp_csi_new_monitoring_data_feature_bin_level"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(bin_psi_results, table_id, job_config=job_config)
job.result()  # Wait for the job to complete

# %%
psi_results

# %% [markdown]
# ## Beta-Cash-Stack-Model Trench1

# %%
sq = r""" 
WITH parsed as (
  select customerId, digitalLoanAccountId,modelDisplayName,modelVersionId,start_time,end_time,prediction,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeatures,
--REPLACE(REPLACE(prediction, "'", '"'), "None", "null") AS prediction_clean
FROM `prj-prod-dataplatform.audit_balance.ml_model_run_details`
where modelDisplayName = 'Beta-Cash-Stack-Model'),

latest_request as (
select * from parsed
QUALIFY ROW_NUMBER() OVER (PARTITION BY customerId, digitalLoanAccountId,modelDisplayName ORDER BY start_time DESC ) = 1),

model_run as (
select customerId,digitalLoanAccountId,modelName, publish_time,
REPLACE(REPLACE(requestPayload, "'", '"'), "None", "null") AS requestPayload_clean
from `prj-prod-dataplatform.audit_balance.ml_request_details` 
WHERE modelName = 'Beta-Cash-Model-response'
QUALIFY ROW_NUMBER() OVER (PARTITION BY customerId, digitalLoanAccountId,modelName ORDER BY publish_time DESC ) = 1)

select * from (
  select 
 r.customerId,
 r.digitalLoanAccountId,
 r.prediction Beta_cash_stack_score,
 r.start_time,
 r.end_time,
 r.modelDisplayName,
  r.modelVersionId,
  loanmaster.new_loan_type,
 loanmaster.gender,
 'Beta-Cash-Stack-Model_Trench1' Model_Name,
 'Cash' as product,
 JSON_VALUE(m.requestPayload_clean, "$.predictions.trenchCategory") AS trenchCategory,
r.calcFeatures,
coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  r.start_time) AS appln_submit_datetime,
  loanmaster.disbursementDateTime,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  r.start_time)) as Application_month,  
FROM latest_request r
left join model_run m
on r.digitalLoanAccountId = m.digitalLoanAccountId 
left join risk_credit_mis.loan_master_table loanmaster 
  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
qualify row_number() over (partition by r.customerId,r.digitalLoanAccountId order by r.start_time desc) = 1
  )
 where trenchCategory = 'Trench 1'
;
"""

dfd = client.query(sq).to_dataframe()
dfd.head()

# %% [markdown]
# ### Expand_df

# %%
# Expand the calcFeatures column
expanded_df = expand_calc_features(dfd)

# Display the result
print(f"Original columns: {dfd.shape[1]}")
print(f"Expanded columns: {expanded_df.shape[1]}")
expanded_df.head()


# %%
df = expanded_df.drop(columns=["calcFeatures"]).copy()

# %% [markdown]
# ### run the psi calculation

# %%
df = df.copy()

# Define feature list
feature_list = [
    "Beta_cash_stack_score",
    "calc_demo_score",
    "calc_credo_score",
    "calc_trx_score",
    "calc_app_score",
]

# Define segment columns
segment_columns = ["new_loan_type", "gender", "osType", "trenchCategory", "product"]
# Calculate month-on-month PSI
psi_results = calculate_month_on_month_psi(df, feature_list, segment_columns)
psi_results["modelDisplayName"] = df["modelDisplayName"].iloc[0]
psi_results["Model_Name"] = df["Model_Name"].iloc[0]
psi_results["modelVersionId"] = df["modelVersionId"].iloc[0]
psi_results["trenchCategory"] = df["trenchCategory"].iloc[0]
psi_results = psi_results[
    [
        "modelDisplayName",
        "Model_Name",
        "modelVersionId",
        "trenchCategory",
        "Feature",
        "Feature_Type",
        "Segment_Column",
        "Segment_Value",
        "Month",
        "Base_Month",
        "Current_Month",
        "PSI",
    ]
].copy()

# Calculate bin-level PSI
bin_psi_results = calculate_bin_level_psi(df, feature_list, segment_columns)
bin_psi_results["modelDisplayName"] = df["modelDisplayName"].iloc[0]
bin_psi_results["Model_Name"] = df["Model_Name"].iloc[0]
bin_psi_results["modelVersionId"] = df["modelVersionId"].iloc[0]
bin_psi_results["trenchCategory"] = df["trenchCategory"].iloc[0]
bin_psi_results = bin_psi_results[
    [
        "modelDisplayName",
        "Model_Name",
        "modelVersionId",
        "trenchCategory",
        "Feature",
        "Feature_Type",
        "Segment_Column",
        "Segment_Value",
        "Month",
        "Base_Month",
        "Current_Month",
        "Bin",
        "Bin_Range",
        "Bin_Min",
        "Bin_Max",
        "Base_Percentage",
        "Actual_Percentage",
        "Bin_PSI",
    ]
].copy()
# Upload to BigQuery
table_id = (
    "prj-prod-dataplatform.dap_ds_poweruser_playground.temp_csi_new_monitoring_data"
)
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(psi_results, table_id, job_config=job_config)
job.result()  # Wait for the job to complete


# Upload to BigQuery
table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.temp_csi_new_monitoring_data_feature_bin_level"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(bin_psi_results, table_id, job_config=job_config)
job.result()  # Wait for the job to complete

# %%
psi_results

# %% [markdown]
# ## Alpha-Cash-Stack-Model_Trench2

# %%
sq = r""" 
WITH parsed as (
  select customerId, digitalLoanAccountId,modelDisplayName,modelVersionId,start_time,end_time,prediction,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeatures,
FROM `prj-prod-dataplatform.audit_balance.ml_model_run_details`
where modelDisplayName = 'Alpha-Cash-Stack-Model'),

latest_request as (
select * from parsed
QUALIFY ROW_NUMBER() OVER (PARTITION BY customerId, digitalLoanAccountId,modelDisplayName ORDER BY start_time DESC ) = 1),

model_run as (
select customerId,digitalLoanAccountId,modelName, publish_time,requestPayload as requestPayload_clean
from `prj-prod-dataplatform.audit_balance.ml_request_details` 
WHERE modelName = 'Alpha-Cash-Model-response'
QUALIFY ROW_NUMBER() OVER (PARTITION BY customerId, digitalLoanAccountId,modelName ORDER BY publish_time DESC ) = 1)

select * from (
  select 
 r.customerId,
 r.digitalLoanAccountId,
 r.prediction aStackScore,
 r.start_time,
 r.end_time,
 r.modelDisplayName,
 r.modelVersionId,
 loanmaster.new_loan_type,
 loanmaster.gender,
'Alpha-Cash-Stack-Model_Trench2' Model_Name,
'Cash' as product,
  REGEXP_EXTRACT(m.requestPayload_clean, r"osType[:=]['\"]?([^'\"]+)['\"]?") AS osType,
  REGEXP_EXTRACT(m.requestPayload_clean, r"loanType[:=]['\"]?([^'\"]+)['\"]?") AS loanType,
  REGEXP_EXTRACT(m.requestPayload_clean, r"trenchCategory[:=]['\"]?([^'\"]+)['\"]?") AS trenchCategory,
r.calcFeatures,
 coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  r.start_time) AS appln_submit_datetime,
  loanmaster.disbursementDateTime,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  r.start_time)) as Application_month,  
FROM latest_request r
left join model_run m
on r.digitalLoanAccountId = m.digitalLoanAccountId 
left join risk_credit_mis.loan_master_table loanmaster 
  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
qualify row_number() over (partition by r.customerId,r.digitalLoanAccountId order by r.start_time desc) = 1
  
) where trenchCategory = 'Trench 2'
;
"""

dfd = client.query(sq).to_dataframe()
dfd.head()

# %% [markdown]
# ### Expand_df

# %%
# Expand the calcFeatures column
expanded_df = expand_calc_features(dfd)

# Display the result
print(f"Original columns: {dfd.shape[1]}")
print(f"Expanded columns: {expanded_df.shape[1]}")
expanded_df.head()


# %%
df = expanded_df.drop(columns=["calcFeatures"]).copy()

# %% [markdown]
# ### run the psi calculation

# %%
df = df.copy()

# Define feature list
feature_list = [
    "aStackScore",
    "calc_apps_score",
    "calc_c_demo_score",
    "calc_c_credo_score",
    "calc_c_tx_score",
    "calc_ca_cic_score",
]

# Define segment columns
segment_columns = ["new_loan_type", "gender", "osType", "trenchCategory", "product"]
# Calculate month-on-month PSI
psi_results = calculate_month_on_month_psi(df, feature_list, segment_columns)
psi_results["modelDisplayName"] = df["modelDisplayName"].iloc[0]
psi_results["Model_Name"] = df["Model_Name"].iloc[0]
psi_results["modelVersionId"] = df["modelVersionId"].iloc[0]
psi_results["trenchCategory"] = df["trenchCategory"].iloc[0]
psi_results = psi_results[
    [
        "modelDisplayName",
        "Model_Name",
        "modelVersionId",
        "trenchCategory",
        "Feature",
        "Feature_Type",
        "Segment_Column",
        "Segment_Value",
        "Month",
        "Base_Month",
        "Current_Month",
        "PSI",
    ]
].copy()

# Calculate bin-level PSI
bin_psi_results = calculate_bin_level_psi(df, feature_list, segment_columns)
bin_psi_results["modelDisplayName"] = df["modelDisplayName"].iloc[0]
bin_psi_results["Model_Name"] = df["Model_Name"].iloc[0]
bin_psi_results["modelVersionId"] = df["modelVersionId"].iloc[0]
bin_psi_results["trenchCategory"] = df["trenchCategory"].iloc[0]
bin_psi_results = bin_psi_results[
    [
        "modelDisplayName",
        "Model_Name",
        "modelVersionId",
        "trenchCategory",
        "Feature",
        "Feature_Type",
        "Segment_Column",
        "Segment_Value",
        "Month",
        "Base_Month",
        "Current_Month",
        "Bin",
        "Bin_Range",
        "Bin_Min",
        "Bin_Max",
        "Base_Percentage",
        "Actual_Percentage",
        "Bin_PSI",
    ]
].copy()

# Upload to BigQuery
table_id = (
    "prj-prod-dataplatform.dap_ds_poweruser_playground.temp_csi_new_monitoring_data"
)
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(psi_results, table_id, job_config=job_config)
job.result()  # Wait for the job to complete


# Upload to BigQuery
table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.temp_csi_new_monitoring_data_feature_bin_level"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(bin_psi_results, table_id, job_config=job_config)
job.result()  # Wait for the job to complete

# %%
psi_results

# %% [markdown]
# ## Beta-Cash-AppScore-Model_Trench2

# %%
sq = r""" 
WITH parsed as (
  select customerId, digitalLoanAccountId,modelDisplayName,modelVersionId,start_time,end_time,prediction,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeatures,
REPLACE(REPLACE(prediction, "'", '"'), "None", "null") AS prediction_clean
FROM `prj-prod-dataplatform.audit_balance.ml_model_run_details`
where modelDisplayName = 'Beta-Cash-AppScore-Model'),

latest_request as (
select * from parsed
QUALIFY ROW_NUMBER() OVER (PARTITION BY customerId, digitalLoanAccountId,modelDisplayName ORDER BY start_time DESC ) = 1),

model_run as (
select customerId,digitalLoanAccountId,modelName, publish_time,
REPLACE(REPLACE(requestPayload, "'", '"'), "None", "null") AS requestPayload_clean
from `prj-prod-dataplatform.audit_balance.ml_request_details` 
WHERE modelName = 'Beta-Cash-Model-response'
QUALIFY ROW_NUMBER() OVER (PARTITION BY customerId, digitalLoanAccountId,modelName ORDER BY publish_time DESC ) = 1)

select * from (
  select 
 r.customerId,
 r.digitalLoanAccountId,
 --r.prediction,
 r.start_time,
 r.end_time,
 r.modelDisplayName,
 r.modelVersionId,
 loanmaster.new_loan_type,
 loanmaster.gender,
 'Beta-Cash-AppScore-Model_Trench2' Model_Name,
 'Cash' as product,
  JSON_VALUE(m.requestPayload_clean, "$.predictions.trenchCategory") AS trenchCategory,
  JSON_VALUE(m.requestPayload_clean, "$.predictions.loanType") AS loanType,
  --  Beta App Score Model Features for all Trench 1
  SAFE_CAST(JSON_VALUE(r.prediction_clean, "$.combined_score") AS Float64) AS beta_cash_app_score,
  r.calcFeatures,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  r.start_time) AS appln_submit_datetime,
  loanmaster.disbursementDateTime,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  r.start_time)) as Application_month,  
FROM latest_request r
left join model_run m
on r.digitalLoanAccountId = m.digitalLoanAccountId 
left join risk_credit_mis.loan_master_table loanmaster 
  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
qualify row_number() over (partition by r.customerId,r.digitalLoanAccountId order by r.start_time desc) = 1
)
where trenchCategory = 'Trench 2'
;
"""

dfd = client.query(sq).to_dataframe()
dfd.head()

# %% [markdown]
# ### Expand_df

# %%
# Expand the calcFeatures column
expanded_df = expand_calc_features(dfd)

# Display the result
print(f"Original columns: {dfd.shape[1]}")
print(f"Expanded columns: {expanded_df.shape[1]}")
expanded_df.head()


# %%
df = expanded_df.drop(columns=["calcFeatures"]).copy()

# %%
df.info()

# %% [markdown]
# ### run the psi calculation

# %%
df = df.copy()

# Define feature list
feature_list = [
    "beta_cash_app_score",
    "calc_app_cnt_health_and_fitness_ever",
    "calc_app_cnt_shopping_ever",
    "calc_app_cnt_crypto_ever",
    "calc_app_cnt_driver_ever",
    "calc_app_cnt_payday_180d",
    "calc_app_cnt_gambling_180d",
    "calc_app_avg_time_bw_installed_mins_3d",
    "calc_app_median_time_bw_installed_mins_3d",
    "calc_app_avg_time_bw_installed_mins_ever",
    "calc_app_median_time_bw_installed_mins_ever",
]

# Define segment columns
segment_columns = ["new_loan_type", "gender", "osType", "trenchCategory", "product"]
# Calculate month-on-month PSI
psi_results = calculate_month_on_month_psi(df, feature_list, segment_columns)
psi_results["modelDisplayName"] = df["modelDisplayName"].iloc[0]
psi_results["Model_Name"] = df["Model_Name"].iloc[0]
psi_results["modelVersionId"] = df["modelVersionId"].iloc[0]
psi_results["trenchCategory"] = df["trenchCategory"].iloc[0]
psi_results = psi_results[
    [
        "modelDisplayName",
        "Model_Name",
        "modelVersionId",
        "trenchCategory",
        "Feature",
        "Feature_Type",
        "Segment_Column",
        "Segment_Value",
        "Month",
        "Base_Month",
        "Current_Month",
        "PSI",
    ]
].copy()

# Calculate bin-level PSI
bin_psi_results = calculate_bin_level_psi(df, feature_list, segment_columns)
bin_psi_results["modelDisplayName"] = df["modelDisplayName"].iloc[0]
bin_psi_results["Model_Name"] = df["Model_Name"].iloc[0]
bin_psi_results["modelVersionId"] = df["modelVersionId"].iloc[0]
bin_psi_results["trenchCategory"] = df["trenchCategory"].iloc[0]
bin_psi_results = bin_psi_results[
    [
        "modelDisplayName",
        "Model_Name",
        "modelVersionId",
        "trenchCategory",
        "Feature",
        "Feature_Type",
        "Segment_Column",
        "Segment_Value",
        "Month",
        "Base_Month",
        "Current_Month",
        "Bin",
        "Bin_Range",
        "Bin_Min",
        "Bin_Max",
        "Base_Percentage",
        "Actual_Percentage",
        "Bin_PSI",
    ]
].copy()

# Upload to BigQuery
table_id = (
    "prj-prod-dataplatform.dap_ds_poweruser_playground.temp_csi_new_monitoring_data"
)
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(psi_results, table_id, job_config=job_config)
job.result()  # Wait for the job to complete


# Upload to BigQuery
table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.temp_csi_new_monitoring_data_feature_bin_level"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(bin_psi_results, table_id, job_config=job_config)
job.result()  # Wait for the job to complete

# %%
psi_results

# %% [markdown]
# ## Beta-Cash-Stack-Model_Trench2

# %%
sq = r""" 
WITH parsed as (
  select customerId, digitalLoanAccountId,modelDisplayName,modelVersionId,start_time,end_time,prediction,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeatures,
--REPLACE(REPLACE(prediction, "'", '"'), "None", "null") AS prediction_clean
FROM `prj-prod-dataplatform.audit_balance.ml_model_run_details`
where modelDisplayName = 'Beta-Cash-Stack-Model'),

latest_request as (
select * from parsed
QUALIFY ROW_NUMBER() OVER (PARTITION BY customerId, digitalLoanAccountId,modelDisplayName ORDER BY start_time DESC ) = 1),

model_run as (
select customerId,digitalLoanAccountId,modelName, publish_time,
REPLACE(REPLACE(requestPayload, "'", '"'), "None", "null") AS requestPayload_clean
from `prj-prod-dataplatform.audit_balance.ml_request_details` 
WHERE modelName = 'Beta-Cash-Model-response'
QUALIFY ROW_NUMBER() OVER (PARTITION BY customerId, digitalLoanAccountId,modelName ORDER BY publish_time DESC ) = 1)

select * from (
  select 
 r.customerId,
 r.digitalLoanAccountId,
 r.prediction Beta_cash_stack_score,
 r.start_time,
 r.end_time,
 r.modelDisplayName,
 r.modelVersionId,
 loanmaster.new_loan_type,
 loanmaster.gender,
 'Beta-Cash-Stack-Model_Trench2' Model_Name,
'Cash' as product,
 JSON_VALUE(m.requestPayload_clean, "$.predictions.trenchCategory") AS trenchCategory,
 r.calcFeatures,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  r.start_time) AS appln_submit_datetime,
  loanmaster.disbursementDateTime,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  r.start_time)) as Application_month, 
FROM latest_request r
left join model_run m
on r.digitalLoanAccountId = m.digitalLoanAccountId 
left join risk_credit_mis.loan_master_table loanmaster 
  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
qualify row_number() over (partition by r.customerId,r.digitalLoanAccountId order by r.start_time desc) = 1
  )
 where trenchCategory = 'Trench 2'
;
"""

dfd = client.query(sq).to_dataframe()
dfd.head()

# %% [markdown]
# ### Expand_df

# %%
# Expand the calcFeatures column
expanded_df = expand_calc_features(dfd)

# Display the result
print(f"Original columns: {dfd.shape[1]}")
print(f"Expanded columns: {expanded_df.shape[1]}")
expanded_df.head()


# %%
df = expanded_df.drop(columns=["calcFeatures"]).copy()

# %% [markdown]
# ### run the psi calculation

# %%
df = df.copy()

# Define feature list
feature_list = [
    "Beta_cash_stack_score",
    "calc_demo_score",
    "calc_credo_score",
    "calc_trx_score",
    "calc_app_score",
]

# Define segment columns
segment_columns = ["new_loan_type", "gender", "osType", "trenchCategory", "product"]
# Calculate month-on-month PSI
psi_results = calculate_month_on_month_psi(df, feature_list, segment_columns)
psi_results["modelDisplayName"] = df["modelDisplayName"].iloc[0]
psi_results["Model_Name"] = df["Model_Name"].iloc[0]
psi_results["modelVersionId"] = df["modelVersionId"].iloc[0]
psi_results["trenchCategory"] = df["trenchCategory"].iloc[0]
psi_results = psi_results[
    [
        "modelDisplayName",
        "Model_Name",
        "modelVersionId",
        "trenchCategory",
        "Feature",
        "Feature_Type",
        "Segment_Column",
        "Segment_Value",
        "Month",
        "Base_Month",
        "Current_Month",
        "PSI",
    ]
].copy()

# Calculate bin-level PSI
bin_psi_results = calculate_bin_level_psi(df, feature_list, segment_columns)
bin_psi_results["modelDisplayName"] = df["modelDisplayName"].iloc[0]
bin_psi_results["Model_Name"] = df["Model_Name"].iloc[0]
bin_psi_results["modelVersionId"] = df["modelVersionId"].iloc[0]
bin_psi_results["trenchCategory"] = df["trenchCategory"].iloc[0]
bin_psi_results = bin_psi_results[
    [
        "modelDisplayName",
        "Model_Name",
        "modelVersionId",
        "trenchCategory",
        "Feature",
        "Feature_Type",
        "Segment_Column",
        "Segment_Value",
        "Month",
        "Base_Month",
        "Current_Month",
        "Bin",
        "Bin_Range",
        "Bin_Min",
        "Bin_Max",
        "Base_Percentage",
        "Actual_Percentage",
        "Bin_PSI",
    ]
].copy()

# Upload to BigQuery
table_id = (
    "prj-prod-dataplatform.dap_ds_poweruser_playground.temp_csi_new_monitoring_data"
)
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(psi_results, table_id, job_config=job_config)
job.result()  # Wait for the job to complete


# Upload to BigQuery
table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.temp_csi_new_monitoring_data_feature_bin_level"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(bin_psi_results, table_id, job_config=job_config)
job.result()  # Wait for the job to complete

# %%
psi_results

# %%


# %% [markdown]
# ## Alpha-Cash-Stack-Model_Trench3

# %%
sq = r""" 
WITH parsed as (
  select customerId, digitalLoanAccountId,modelDisplayName,modelVersionId,start_time,end_time,prediction,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeatures,
FROM `prj-prod-dataplatform.audit_balance.ml_model_run_details`
where modelDisplayName = 'Alpha-Cash-Stack-Model'),
latest_request as (
select * from parsed
QUALIFY ROW_NUMBER() OVER (PARTITION BY customerId, digitalLoanAccountId,modelDisplayName ORDER BY start_time DESC ) = 1),
model_run as (
select customerId,digitalLoanAccountId,modelName, publish_time,requestPayload as requestPayload_clean
from `prj-prod-dataplatform.audit_balance.ml_request_details` 
WHERE modelName = 'Alpha-Cash-Model-response'
QUALIFY ROW_NUMBER() OVER (PARTITION BY customerId, digitalLoanAccountId,modelName ORDER BY publish_time DESC ) = 1)
select * from (
  select 
 r.customerId,
 r.digitalLoanAccountId,
 r.prediction aStackScore,
 r.start_time,
 r.end_time,
 r.modelDisplayName,
 r.modelVersionId,
 loanmaster.new_loan_type,
 loanmaster.gender,
 'Alpha-Cash-Stack-Model_Trench3' Model_Name,
 'Cash' as product,
  REGEXP_EXTRACT(m.requestPayload_clean, r"osType[:=]['\"]?([^'\"]+)['\"]?") AS osType,
  REGEXP_EXTRACT(m.requestPayload_clean, r"loanType[:=]['\"]?([^'\"]+)['\"]?") AS loanType,
  REGEXP_EXTRACT(m.requestPayload_clean, r"trenchCategory[:=]['\"]?([^'\"]+)['\"]?") AS trenchCategory,
 r.calcFeatures,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  r.start_time) AS appln_submit_datetime,
  loanmaster.disbursementDateTime,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  r.start_time)) as Application_month,  
FROM latest_request r
left join model_run m
on r.digitalLoanAccountId = m.digitalLoanAccountId 
left join risk_credit_mis.loan_master_table loanmaster 
  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
qualify row_number() over (partition by r.customerId,r.digitalLoanAccountId order by r.start_time desc) = 1
) where trenchCategory = 'Trench 3'
;
"""

dfd = client.query(sq).to_dataframe()
dfd.head()

# %% [markdown]
# ### Expand_df

# %%
# Expand the calcFeatures column
expanded_df = expand_calc_features(dfd)

# Display the result
print(f"Original columns: {dfd.shape[1]}")
print(f"Expanded columns: {expanded_df.shape[1]}")
expanded_df.head()


# %%
df = expanded_df.drop(columns=["calcFeatures"]).copy()

# %% [markdown]
# ### run the psi calculation

# %%
df = df.copy()

# Define feature list
feature_list = [
    "aStackScore",
    "calc_apps_score",
    "calc_c_demo_score",
    "calc_c_credo_score",
    "calc_c_tx_score",
    "calc_ca_cic_score",
]

# Define segment columns
segment_columns = ["new_loan_type", "gender", "osType", "trenchCategory", "product"]
# Calculate month-on-month PSI
psi_results = calculate_month_on_month_psi(df, feature_list, segment_columns)
psi_results["modelDisplayName"] = df["modelDisplayName"].iloc[0]
psi_results["Model_Name"] = df["Model_Name"].iloc[0]
psi_results["modelVersionId"] = df["modelVersionId"].iloc[0]
psi_results["trenchCategory"] = df["trenchCategory"].iloc[0]
psi_results = psi_results[
    [
        "modelDisplayName",
        "Model_Name",
        "modelVersionId",
        "trenchCategory",
        "Feature",
        "Feature_Type",
        "Segment_Column",
        "Segment_Value",
        "Month",
        "Base_Month",
        "Current_Month",
        "PSI",
    ]
].copy()

# Calculate bin-level PSI
bin_psi_results = calculate_bin_level_psi(df, feature_list, segment_columns)
bin_psi_results["modelDisplayName"] = df["modelDisplayName"].iloc[0]
bin_psi_results["Model_Name"] = df["Model_Name"].iloc[0]
bin_psi_results["modelVersionId"] = df["modelVersionId"].iloc[0]
bin_psi_results["trenchCategory"] = df["trenchCategory"].iloc[0]
bin_psi_results = bin_psi_results[
    [
        "modelDisplayName",
        "Model_Name",
        "modelVersionId",
        "trenchCategory",
        "Feature",
        "Feature_Type",
        "Segment_Column",
        "Segment_Value",
        "Month",
        "Base_Month",
        "Current_Month",
        "Bin",
        "Bin_Range",
        "Bin_Min",
        "Bin_Max",
        "Base_Percentage",
        "Actual_Percentage",
        "Bin_PSI",
    ]
].copy()

# Upload to BigQuery
table_id = (
    "prj-prod-dataplatform.dap_ds_poweruser_playground.temp_csi_new_monitoring_data"
)
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(psi_results, table_id, job_config=job_config)
job.result()  # Wait for the job to complete


# Upload to BigQuery
table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.temp_csi_new_monitoring_data_feature_bin_level"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(bin_psi_results, table_id, job_config=job_config)
job.result()  # Wait for the job to complete

# %%
psi_results

# %% [markdown]
# ## Beta-Cash-AppScore-Model_Trench3

# %%
sq = r""" WITH parsed as (
  select customerId, digitalLoanAccountId,modelDisplayName,modelVersionId,start_time,end_time,prediction,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeatures,
REPLACE(REPLACE(prediction, "'", '"'), "None", "null") AS prediction_clean
FROM `prj-prod-dataplatform.audit_balance.ml_model_run_details`
where modelDisplayName = 'Beta-Cash-AppScore-Model'),

latest_request as (
select * from parsed
QUALIFY ROW_NUMBER() OVER (PARTITION BY customerId, digitalLoanAccountId,modelDisplayName ORDER BY start_time DESC ) = 1),

model_run as (
select customerId,digitalLoanAccountId,modelName, publish_time,
REPLACE(REPLACE(requestPayload, "'", '"'), "None", "null") AS requestPayload_clean
from `prj-prod-dataplatform.audit_balance.ml_request_details` 
WHERE modelName = 'Beta-Cash-Model-response'
QUALIFY ROW_NUMBER() OVER (PARTITION BY customerId, digitalLoanAccountId,modelName ORDER BY publish_time DESC ) = 1)

select * from (
  select 
 r.customerId,
 r.digitalLoanAccountId,
 --r.prediction,
 r.start_time,
 r.end_time,
 r.modelDisplayName,
 r.modelVersionId,
 loanmaster.new_loan_type,
 loanmaster.gender,
 'Beta-Cash-AppScore-Model_Trench3' Model_Name,
  JSON_VALUE(m.requestPayload_clean, "$.predictions.trenchCategory") AS trenchCategory,
  JSON_VALUE(m.requestPayload_clean, "$.predictions.loanType") AS loanType,
  --  Beta App Score Model Features for Trench 3
  SAFE_CAST(JSON_VALUE(r.prediction_clean, "$.combined_score") AS Float64) AS beta_cash_app_score,
 r.calcFeatures,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  r.start_time) AS appln_submit_datetime,
  loanmaster.disbursementDateTime,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  r.start_time)) as Application_month,  
FROM latest_request r
left join model_run m
on r.digitalLoanAccountId = m.digitalLoanAccountId 
left join risk_credit_mis.loan_master_table loanmaster 
  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
qualify row_number() over (partition by r.customerId,r.digitalLoanAccountId order by r.start_time desc) = 1
)
where trenchCategory = 'Trench 3'

;"""
dfd = client.query(sq).to_dataframe()
dfd.head()

# %% [markdown]
# ### Expand_df

# %%
# Expand the calcFeatures column
expanded_df = expand_calc_features(dfd)

# Display the result
print(f"Original columns: {dfd.shape[1]}")
print(f"Expanded columns: {expanded_df.shape[1]}")
expanded_df.head()


# %%
df = expanded_df.drop(columns=["calcFeatures"]).copy()

# %% [markdown]
# ### run the psi calculation

# %%
df = df.copy()

# Define feature list
feature_list = [
    "beta_cash_app_score",
    "calc_app_cnt_health_and_fitness_ever",
    "calc_app_cnt_productivity_ever",
    "calc_app_cnt_rated_for_18plus_ever",
    "calc_app_cnt_books_and_reference_ever",
    "calc_app_cnt_gaming_180d",
    "calc_app_cnt_absence_tag_365d",
    "calc_app_last_payday_install_to_apply_days",
    "calc_app_cnt_absence_tag_365d_binned",
    "calc_app_cnt_gaming_180d_binned",
    "calc_app_cnt_productivity_ever_binned",
    "calc_app_cnt_rated_for_18plus_ever_binned",
    "calc_app_cnt_health_and_fitness_ever_binned",
    "calc_app_cnt_books_and_reference_ever_binned",
    "calc_app_last_payday_install_to_apply_days_binned",
    "calc_ln_user_type",
]

# Define segment columns
segment_columns = ["new_loan_type", "gender", "osType", "trenchCategory"]
# Calculate month-on-month PSI
psi_results = calculate_month_on_month_psi(df, feature_list, segment_columns)
psi_results["modelDisplayName"] = df["modelDisplayName"].iloc[0]
psi_results["Model_Name"] = df["Model_Name"].iloc[0]
psi_results["modelVersionId"] = df["modelVersionId"].iloc[0]
psi_results["trenchCategory"] = df["trenchCategory"].iloc[0]
psi_results = psi_results[
    [
        "modelDisplayName",
        "Model_Name",
        "modelVersionId",
        "trenchCategory",
        "Feature",
        "Feature_Type",
        "Segment_Column",
        "Segment_Value",
        "Month",
        "Base_Month",
        "Current_Month",
        "PSI",
    ]
].copy()

# Calculate bin-level PSI
bin_psi_results = calculate_bin_level_psi(df, feature_list, segment_columns)
bin_psi_results["modelDisplayName"] = df["modelDisplayName"].iloc[0]
bin_psi_results["Model_Name"] = df["Model_Name"].iloc[0]
bin_psi_results["modelVersionId"] = df["modelVersionId"].iloc[0]
bin_psi_results["trenchCategory"] = df["trenchCategory"].iloc[0]
bin_psi_results = bin_psi_results[
    [
        "modelDisplayName",
        "Model_Name",
        "modelVersionId",
        "trenchCategory",
        "Feature",
        "Feature_Type",
        "Segment_Column",
        "Segment_Value",
        "Month",
        "Base_Month",
        "Current_Month",
        "Bin",
        "Bin_Range",
        "Bin_Min",
        "Bin_Max",
        "Base_Percentage",
        "Actual_Percentage",
        "Bin_PSI",
    ]
].copy()

# Upload to BigQuery
table_id = (
    "prj-prod-dataplatform.dap_ds_poweruser_playground.temp_csi_new_monitoring_data"
)
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(psi_results, table_id, job_config=job_config)
job.result()  # Wait for the job to complete


# Upload to BigQuery
table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.temp_csi_new_monitoring_data_feature_bin_level"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(bin_psi_results, table_id, job_config=job_config)
job.result()  # Wait for the job to complete

# %%
psi_results

# %% [markdown]
# ## Beta-Cash-Stack-Model_Trench3

# %%
sq = r"""WITH parsed as (
  select customerId, digitalLoanAccountId,modelDisplayName,modelVersionId,start_time,end_time,prediction,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeatures,
--REPLACE(REPLACE(prediction, "'", '"'), "None", "null") AS prediction_clean
FROM `prj-prod-dataplatform.audit_balance.ml_model_run_details`
where modelDisplayName = 'Beta-Cash-Stack-Model'),

latest_request as (
select * from parsed
QUALIFY ROW_NUMBER() OVER (PARTITION BY customerId, digitalLoanAccountId,modelDisplayName ORDER BY start_time DESC ) = 1),

model_run as (
select customerId,digitalLoanAccountId,modelName, publish_time,
REPLACE(REPLACE(requestPayload, "'", '"'), "None", "null") AS requestPayload_clean
from `prj-prod-dataplatform.audit_balance.ml_request_details` 
WHERE modelName = 'Beta-Cash-Model-response'
QUALIFY ROW_NUMBER() OVER (PARTITION BY customerId, digitalLoanAccountId,modelName ORDER BY publish_time DESC ) = 1)

select * from (
  select 
 r.customerId,
 r.digitalLoanAccountId,
 r.prediction Beta_cash_stack_score,
 r.start_time,
 r.end_time,
 r.modelDisplayName,
 r.modelVersionId,
 loanmaster.new_loan_type,
 loanmaster.gender,
 'Beta-Cash-Stack-Model_Trench3' Model_Name,
 'Cash' as product,
 JSON_VALUE(m.requestPayload_clean, "$.predictions.trenchCategory") AS trenchCategory,
  r.calcFeatures,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  r.start_time) AS appln_submit_datetime,
  loanmaster.disbursementDateTime,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  r.start_time)) as Application_month,  
FROM latest_request r
left join model_run m
on r.digitalLoanAccountId = m.digitalLoanAccountId 
left join risk_credit_mis.loan_master_table loanmaster 
  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
qualify row_number() over (partition by r.customerId,r.digitalLoanAccountId order by r.start_time desc) = 1
  )
 where trenchCategory = 'Trench 3'

;"""
dfd = client.query(sq).to_dataframe()
dfd.head()

# %% [markdown]
# ### Expand_df

# %%
# Expand the calcFeatures column
expanded_df = expand_calc_features(dfd)

# Display the result
print(f"Original columns: {dfd.shape[1]}")
print(f"Expanded columns: {expanded_df.shape[1]}")
expanded_df.head()


# %%
df = expanded_df.drop(columns=["calcFeatures"]).copy()

# %% [markdown]
# ### run the psi calculation

# %%
df = df.copy()

# Define feature list
feature_list = [
    "Beta_cash_stack_score",
    "calc_demo_score",
    "calc_credo_score",
    "calc_trx_score",
    "calc_app_score",
]

# Define segment columns
segment_columns = ["new_loan_type", "gender", "osType", "trenchCategory", "product"]
# Calculate month-on-month PSI
psi_results = calculate_month_on_month_psi(df, feature_list, segment_columns)
psi_results["modelDisplayName"] = df["modelDisplayName"].iloc[0]
psi_results["Model_Name"] = df["Model_Name"].iloc[0]
psi_results["modelVersionId"] = df["modelVersionId"].iloc[0]
psi_results["trenchCategory"] = df["trenchCategory"].iloc[0]
psi_results = psi_results[
    [
        "modelDisplayName",
        "Model_Name",
        "modelVersionId",
        "trenchCategory",
        "Feature",
        "Feature_Type",
        "Segment_Column",
        "Segment_Value",
        "Month",
        "Base_Month",
        "Current_Month",
        "PSI",
    ]
].copy()

# Calculate bin-level PSI
bin_psi_results = calculate_bin_level_psi(df, feature_list, segment_columns)
bin_psi_results["modelDisplayName"] = df["modelDisplayName"].iloc[0]
bin_psi_results["Model_Name"] = df["Model_Name"].iloc[0]
bin_psi_results["modelVersionId"] = df["modelVersionId"].iloc[0]
bin_psi_results["trenchCategory"] = df["trenchCategory"].iloc[0]
bin_psi_results = bin_psi_results[
    [
        "modelDisplayName",
        "Model_Name",
        "modelVersionId",
        "trenchCategory",
        "Feature",
        "Feature_Type",
        "Segment_Column",
        "Segment_Value",
        "Month",
        "Base_Month",
        "Current_Month",
        "Bin",
        "Bin_Range",
        "Bin_Min",
        "Bin_Max",
        "Base_Percentage",
        "Actual_Percentage",
        "Bin_PSI",
    ]
].copy()

# Upload to BigQuery
table_id = (
    "prj-prod-dataplatform.dap_ds_poweruser_playground.temp_csi_new_monitoring_data"
)
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(psi_results, table_id, job_config=job_config)
job.result()  # Wait for the job to complete


# Upload to BigQuery
table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.temp_csi_new_monitoring_data_feature_bin_level"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(bin_psi_results, table_id, job_config=job_config)
job.result()  # Wait for the job to complete

# %%
psi_results

# %% [markdown]
# # SIL

# %% [markdown]
# ## Alpha - CIC-SIL-Model

# %%
sq = """ 
WITH cleaned AS (
  SELECT
    customerId,digitalLoanAccountId,prediction,start_time,end_time,modelDisplayName,modelVersionId,

    REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature
  FROM `prj-prod-dataplatform.audit_balance.ml_model_run_details`
  WHERE modelDisplayName = 'Alpha - CIC-SIL-Model')
SELECT
  r.customerId,r.digitalLoanAccountId,prediction Alpha_cic_sil_score
    ,start_time,end_time,modelDisplayName,modelVersionId,
   loanmaster.new_loan_type,
 loanmaster.gender,
 'Alpha - CIC-SIL-Model' Model_Name,
 'SIL' as product,
 'NA' trenchCategory,
  r.calcFeature calcFeatures,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  r.start_time) AS appln_submit_datetime,
  loanmaster.disbursementDateTime,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  r.start_time)) as Application_month,  
FROM cleaned r 
left join risk_credit_mis.loan_master_table loanmaster 
  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
qualify row_number() over (partition by r.customerId,r.digitalLoanAccountId order by r.start_time desc) = 1
;
"""

dfd = client.query(sq).to_dataframe()
dfd.head()

# %% [markdown]
# ### Expand df

# %%
# Expand the calcFeatures column
expanded_df = expand_calc_features(dfd)

# Display the result
print(f"Original columns: {dfd.shape[1]}")
print(f"Expanded columns: {expanded_df.shape[1]}")
expanded_df.head()


# %%
df = expanded_df.drop(columns=["calcFeatures"]).copy()

# %%
df["Alpha_cic_sil_score"] = pd.to_numeric(df["Alpha_cic_sil_score"], errors="coerce")
df.info()

# %% [markdown]
# ### run the psi calculation

# %%
df = df.copy()

# Define feature list
feature_list = [
    "Alpha_cic_sil_score",
    "calc_cic_Personal_Loans_granted_contracts_amt_24M",
    "calc_cic_days_since_last_inquiry",
    "calc_cic_cnt_active_contracts",
    "calc_cic_vel_contract_nongranted_cnt_12on24",
    "calc_cic_max_amt_granted_24M",
    "calc_cic_zero_non_granted_ever_flag",
    "calc_cic_tot_active_contracts_util",
    "calc_cic_vel_contract_granted_amt_12on24",
    "calc_cic_zero_granted_ever_flag",
]

# Define segment columns
segment_columns = ["new_loan_type", "gender", "osType", "trenchCategory", "product"]
# Calculate month-on-month PSI
psi_results = calculate_month_on_month_psi(df, feature_list, segment_columns)
psi_results["modelDisplayName"] = df["modelDisplayName"].iloc[0]
psi_results["Model_Name"] = df["Model_Name"].iloc[0]
psi_results["modelVersionId"] = df["modelVersionId"].iloc[0]
psi_results["trenchCategory"] = df["trenchCategory"].iloc[0]
psi_results = psi_results[
    [
        "modelDisplayName",
        "Model_Name",
        "modelVersionId",
        "trenchCategory",
        "Feature",
        "Feature_Type",
        "Segment_Column",
        "Segment_Value",
        "Month",
        "Base_Month",
        "Current_Month",
        "PSI",
    ]
].copy()

# Calculate bin-level PSI
bin_psi_results = calculate_bin_level_psi(df, feature_list, segment_columns)
bin_psi_results["modelDisplayName"] = df["modelDisplayName"].iloc[0]
bin_psi_results["Model_Name"] = df["Model_Name"].iloc[0]
bin_psi_results["modelVersionId"] = df["modelVersionId"].iloc[0]
bin_psi_results["trenchCategory"] = df["trenchCategory"].iloc[0]
bin_psi_results = bin_psi_results[
    [
        "modelDisplayName",
        "Model_Name",
        "modelVersionId",
        "trenchCategory",
        "Feature",
        "Feature_Type",
        "Segment_Column",
        "Segment_Value",
        "Month",
        "Base_Month",
        "Current_Month",
        "Bin",
        "Bin_Range",
        "Bin_Min",
        "Bin_Max",
        "Base_Percentage",
        "Actual_Percentage",
        "Bin_PSI",
    ]
].copy()

# Upload to BigQuery
table_id = (
    "prj-prod-dataplatform.dap_ds_poweruser_playground.temp_csi_new_monitoring_data"
)
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(psi_results, table_id, job_config=job_config)
job.result()  # Wait for the job to complete


# Upload to BigQuery
table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.temp_csi_new_monitoring_data_feature_bin_level"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(bin_psi_results, table_id, job_config=job_config)
job.result()  # Wait for the job to complete

# %%
psi_results

# %% [markdown]
# ## Alpha - IncomeEstimationModel

# %%
sq = """ 
WITH cleaned AS (
  SELECT
    customerId,digitalLoanAccountId,prediction,start_time,end_time,modelDisplayName,modelVersionId,
    REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature
  FROM `prj-prod-dataplatform.audit_balance.ml_model_run_details`
  WHERE modelDisplayName = 'Alpha  - IncomeEstimationModel')
SELECT
  r.customerId,r.digitalLoanAccountId,prediction Alpha_Income_Estimated_score,start_time,end_time,modelDisplayName,modelVersionId,
  loanmaster.new_loan_type,
 loanmaster.gender,
  'Alpha  - IncomeEstimationModel' Model_Name,
  'SIL' as product,
  'NA' trenchCategory,
  r.calcFeature calcFeatures,
  JSON_VALUE(calcFeature, "$.inc_alpha_cic_credit_avg_credit_limit") AS inc_alpha_cic_credit_avg_credit_limit,
  JSON_VALUE(calcFeature, "$.inc_alpha_cic_max_active_contracts_amt") AS inc_alpha_cic_max_active_contracts_amt,
  JSON_VALUE(calcFeature, "$.inc_alpha_ln_age") AS inc_alpha_ln_age,
  JSON_VALUE(calcFeature, "$.inc_alpha_doc_type_rolled") AS inc_alpha_doc_type_rolled,
  JSON_VALUE(calcFeature, "$.inc_alpha_ln_brand") AS inc_alpha_ln_brand,
  JSON_VALUE(calcFeature, "$.inc_alpha_ln_city") AS inc_alpha_ln_city,
  JSON_VALUE(calcFeature, "$.inc_alpha_ln_cnt_dependents") AS inc_alpha_ln_cnt_dependents,
  JSON_VALUE(calcFeature, "$.inc_alpha_ln_education_level") AS inc_alpha_ln_education_level,
  JSON_VALUE(calcFeature, "$.inc_alpha_ln_employment_type_new") AS inc_alpha_ln_employment_type_new,
  JSON_VALUE(calcFeature, "$.inc_alpha_ln_gender") AS inc_alpha_ln_gender,
  JSON_VALUE(calcFeature, "$.inc_alpha_ln_industry_new") AS inc_alpha_ln_industry_new,
  JSON_VALUE(calcFeature, "$.inc_alpha_ln_loan_prod_type") AS inc_alpha_ln_loan_prod_type,
  JSON_VALUE(calcFeature, "$.inc_alpha_ln_marital_status_new") AS inc_alpha_ln_marital_status_new,
  JSON_VALUE(calcFeature, "$.inc_alpha_ln_nature_of_work_new") AS inc_alpha_ln_nature_of_work_new,
  JSON_VALUE(calcFeature, "$.inc_alpha_ln_osversion_bin") AS inc_alpha_ln_osversion_bin,
  JSON_VALUE(calcFeature, "$.inc_alpha_ln_purpose") AS inc_alpha_ln_purpose,
  JSON_VALUE(calcFeature, "$.inc_alpha_ln_source_of_funds_new") AS inc_alpha_ln_source_of_funds_new,
  JSON_VALUE(calcFeature, "$.inc_alpha_encoded_company_name_grouped") AS inc_alpha_encoded_company_name_grouped,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  r.start_time) AS appln_submit_datetime,
  loanmaster.disbursementDateTime,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  r.start_time)) as Application_month,
FROM cleaned r
left join risk_credit_mis.loan_master_table loanmaster 
  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
qualify row_number() over (partition by r.customerId,r.digitalLoanAccountId order by r.start_time desc) = 1
  ;
"""
dfd = client.query(sq).to_dataframe()
dfd.head()

# %% [markdown]
# ### Expand df
#

# %%
# Expand the calcFeatures column
expanded_df = dfd.copy()

# Display the result
print(f"Original columns: {dfd.shape[1]}")
print(f"Expanded columns: {expanded_df.shape[1]}")
expanded_df.head()


# %%
df = expanded_df.drop(columns=["calcFeatures"]).copy()

# %%
df["Alpha_Income_Estimated_score"] = pd.to_numeric(
    df["Alpha_Income_Estimated_score"], errors="coerce"
)
df.columns

# %% [markdown]
# ### run psi calculation

# %%
df = df.copy()

# Define feature list
feature_list = [
    "Alpha_Income_Estimated_score",
    "inc_alpha_cic_credit_avg_credit_limit",
    "inc_alpha_cic_max_active_contracts_amt",
    "inc_alpha_ln_age",
    "inc_alpha_doc_type_rolled",
    "inc_alpha_ln_brand",
    "inc_alpha_ln_city",
    "inc_alpha_ln_cnt_dependents",
    "inc_alpha_ln_education_level",
    "inc_alpha_ln_employment_type_new",
    "inc_alpha_ln_gender",
    "inc_alpha_ln_industry_new",
    "inc_alpha_ln_loan_prod_type",
    "inc_alpha_ln_marital_status_new",
    "inc_alpha_ln_nature_of_work_new",
    "inc_alpha_ln_osversion_bin",
    "inc_alpha_ln_purpose",
    "inc_alpha_ln_source_of_funds_new",
    "inc_alpha_encoded_company_name_grouped",
]

# Define segment columns
segment_columns = ["new_loan_type", "gender", "osType", "trenchCategory", "product"]
# Calculate month-on-month PSI
psi_results = calculate_month_on_month_psi(df, feature_list, segment_columns)
psi_results["modelDisplayName"] = df["modelDisplayName"].iloc[0]
psi_results["Model_Name"] = df["Model_Name"].iloc[0]
psi_results["modelVersionId"] = df["modelVersionId"].iloc[0]
psi_results["trenchCategory"] = df["trenchCategory"].iloc[0]
psi_results = psi_results[
    [
        "modelDisplayName",
        "Model_Name",
        "modelVersionId",
        "trenchCategory",
        "Feature",
        "Feature_Type",
        "Segment_Column",
        "Segment_Value",
        "Month",
        "Base_Month",
        "Current_Month",
        "PSI",
    ]
].copy()

# Calculate bin-level PSI
bin_psi_results = calculate_bin_level_psi(df, feature_list, segment_columns)
bin_psi_results["modelDisplayName"] = df["modelDisplayName"].iloc[0]
bin_psi_results["Model_Name"] = df["Model_Name"].iloc[0]
bin_psi_results["modelVersionId"] = df["modelVersionId"].iloc[0]
bin_psi_results["trenchCategory"] = df["trenchCategory"].iloc[0]
bin_psi_results = bin_psi_results[
    [
        "modelDisplayName",
        "Model_Name",
        "modelVersionId",
        "trenchCategory",
        "Feature",
        "Feature_Type",
        "Segment_Column",
        "Segment_Value",
        "Month",
        "Base_Month",
        "Current_Month",
        "Bin",
        "Bin_Range",
        "Bin_Min",
        "Bin_Max",
        "Base_Percentage",
        "Actual_Percentage",
        "Bin_PSI",
    ]
].copy()

# Upload to BigQuery
table_id = (
    "prj-prod-dataplatform.dap_ds_poweruser_playground.temp_csi_new_monitoring_data"
)
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(psi_results, table_id, job_config=job_config)
job.result()  # Wait for the job to complete


# Upload to BigQuery
table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.temp_csi_new_monitoring_data_feature_bin_level"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(bin_psi_results, table_id, job_config=job_config)
job.result()  # Wait for the job to complete

# %%
psi_results

# %% [markdown]
# ## SIL Alpha - StackingModel

# %%
sq = """ WITH cleaned AS (
  SELECT
    customerId,digitalLoanAccountId,prediction,start_time,end_time,modelDisplayName,modelVersionId,
    REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature
  FROM `prj-prod-dataplatform.audit_balance.ml_model_run_details`
  WHERE modelDisplayName = 'Alpha - StackingModel')
SELECT distinct
 r.customerId,r.digitalLoanAccountId,prediction Sil_Alpha_Stack_score,start_time,end_time,modelDisplayName,modelVersionId,
 loanmaster.new_loan_type,
 loanmaster.gender,
 'SIL Alpha - StackingModel' Model_Name,
 'SIL' as product,
 'NA' trenchCategory,
  r.calcFeature calcFeatures,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  r.start_time) AS appln_submit_datetime,
  loanmaster.disbursementDateTime,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  r.start_time)) as Application_month,  
FROM cleaned r 
left join risk_credit_mis.loan_master_table loanmaster 
  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
qualify row_number() over(partition by r.digitalLoanAccountId order by r.start_time desc)=1
  ;"""
dfd = client.query(sq).to_dataframe()
dfd = dfd.drop_duplicates(keep="first")
dfd.head()

# %% [markdown]
# ### Expand df

# %%
# Expand the calcFeatures column
expanded_df = expand_calc_features(dfd)

# Display the result
print(f"Original columns: {dfd.shape[1]}")
print(f"Expanded columns: {expanded_df.shape[1]}")
expanded_df.head()


# %%
df = expanded_df.drop(columns=["calcFeatures"]).copy()
df = df.drop_duplicates(keep="first")

# %%
df["Sil_Alpha_Stack_score"] = pd.to_numeric(
    df["Sil_Alpha_Stack_score"], errors="coerce"
)
df.info()

# %% [markdown]
# ### run psi calculation

# %%
df = df.copy()

# Define feature list
feature_list = [
    "Sil_Alpha_Stack_score",
    "calc_sb_demo_score",
    "calc_s_cic_score",
    "calc_s_credo_score",
    "calc_s_apps_score",
]

# Define segment columns
segment_columns = ["new_loan_type", "gender", "osType", "trenchCategory", "product"]
# Calculate month-on-month PSI
psi_results = calculate_month_on_month_psi(df, feature_list, segment_columns)
psi_results["modelDisplayName"] = df["modelDisplayName"].iloc[0]
psi_results["Model_Name"] = df["Model_Name"].iloc[0]
psi_results["modelVersionId"] = df["modelVersionId"].iloc[0]
psi_results["trenchCategory"] = df["trenchCategory"].iloc[0]
psi_results = psi_results[
    [
        "modelDisplayName",
        "Model_Name",
        "modelVersionId",
        "trenchCategory",
        "Feature",
        "Feature_Type",
        "Segment_Column",
        "Segment_Value",
        "Month",
        "Base_Month",
        "Current_Month",
        "PSI",
    ]
].copy()

# Calculate bin-level PSI
bin_psi_results = calculate_bin_level_psi(df, feature_list, segment_columns)
bin_psi_results["modelDisplayName"] = df["modelDisplayName"].iloc[0]
bin_psi_results["Model_Name"] = df["Model_Name"].iloc[0]
bin_psi_results["modelVersionId"] = df["modelVersionId"].iloc[0]
bin_psi_results["trenchCategory"] = df["trenchCategory"].iloc[0]
bin_psi_results = bin_psi_results[
    [
        "modelDisplayName",
        "Model_Name",
        "modelVersionId",
        "trenchCategory",
        "Feature",
        "Feature_Type",
        "Segment_Column",
        "Segment_Value",
        "Month",
        "Base_Month",
        "Current_Month",
        "Bin",
        "Bin_Range",
        "Bin_Min",
        "Bin_Max",
        "Base_Percentage",
        "Actual_Percentage",
        "Bin_PSI",
    ]
].copy()

# Upload to BigQuery
table_id = (
    "prj-prod-dataplatform.dap_ds_poweruser_playground.temp_csi_new_monitoring_data"
)
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(psi_results, table_id, job_config=job_config)
job.result()  # Wait for the job to complete


# Upload to BigQuery
table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.temp_csi_new_monitoring_data_feature_bin_level"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(bin_psi_results, table_id, job_config=job_config)
job.result()  # Wait for the job to complete

# %%
psi_results

# %% [markdown]
# ## SIL Beta - AppsScoreModel

# %%
sq = """ 
WITH cleaned AS (
  SELECT
    customerId,digitalLoanAccountId,prediction,start_time,end_time,modelDisplayName,modelVersionId,

    REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature,
    REPLACE(REPLACE(prediction, "'", '"'), "None", "null") AS prediction_clean
  FROM `prj-prod-dataplatform.audit_balance.ml_model_run_details`
  WHERE modelDisplayName = 'Beta - AppsScoreModel')
SELECT
  r.customerId,r.digitalLoanAccountId,prediction,start_time,end_time,
  modelDisplayName,modelVersionId,
     loanmaster.new_loan_type,
 loanmaster.gender,
 'SIL Beta - AppsScoreModel' Model_Name,
 'SIL' as product,
  'NA' trenchCategory,
  safe_cast(JSON_VALUE(prediction_clean, "$.combined_score") AS float64) as sil_beta_app_score,
 calcFeature calcFeatures, 
    coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  r.start_time) AS appln_submit_datetime,
    loanmaster.disbursementDateTime,
    format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  r.start_time)) as Application_month,
 FROM cleaned r
left join risk_credit_mis.loan_master_table loanmaster 
  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  qualify row_number() over(partition by r.customerId, r.digitalLoanAccountid order by start_time desc) = 1
;
"""
dfd = client.query(sq).to_dataframe()
dfd.head()

# %% [markdown]
# ### Expand df

# %%
# Expand the calcFeatures column
expanded_df = expand_calc_features(dfd)

# Display the result
print(f"Original columns: {dfd.shape[1]}")
print(f"Expanded columns: {expanded_df.shape[1]}")
expanded_df.head()


# %%
df = expanded_df.drop(columns=["calcFeatures"]).copy()
df = df.drop_duplicates(keep="first")

# %%
df["sil_beta_app_score"] = pd.to_numeric(df["sil_beta_app_score"], errors="coerce")


# %% [markdown]
# ### run psi calculation

# %%
df = df.copy()

# Define feature list
feature_list = [
    "sil_beta_app_score",
    "calc_app_cnt_rated_for_3plus_ever",
    "calc_app_cnt_education_ever",
    "calc_app_cnt_business_ever",
    "calc_app_cnt_music_and_audio_ever",
    "calc_app_cnt_travel_and_local_ever",
    "calc_app_cnt_finance_7d",
    "calc_app_cnt_absence_tag_30d",
    "calc_app_cnt_competitors_30d",
    "calc_app_cnt_finance_30d",
    "calc_app_cnt_absence_tag_90d",
    "calc_app_cnt_finance_90d",
    "calc_app_cnt_competitors_90d",
    "calc_app_cnt_payday_90d",
    "calc_app_avg_time_bw_installed_mins_30d",
    "calc_app_median_time_bw_installed_mins_30d",
    "calc_app_first_competitors_install_to_apply_days",
    "calc_app_first_payday_install_to_apply_days",
    "calc_app_vel_finance_30_over_365",
]

# Define segment columns
segment_columns = ["new_loan_type", "gender", "osType", "trenchCategory", "product"]
# Calculate month-on-month PSI
psi_results = calculate_month_on_month_psi(df, feature_list, segment_columns)
psi_results["modelDisplayName"] = df["modelDisplayName"].iloc[0]
psi_results["Model_Name"] = df["Model_Name"].iloc[0]
psi_results["modelVersionId"] = df["modelVersionId"].iloc[0]
psi_results["trenchCategory"] = df["trenchCategory"].iloc[0]
psi_results = psi_results[
    [
        "modelDisplayName",
        "Model_Name",
        "modelVersionId",
        "trenchCategory",
        "Feature",
        "Feature_Type",
        "Segment_Column",
        "Segment_Value",
        "Month",
        "Base_Month",
        "Current_Month",
        "PSI",
    ]
].copy()

# Calculate bin-level PSI
bin_psi_results = calculate_bin_level_psi(df, feature_list, segment_columns)
bin_psi_results["modelDisplayName"] = df["modelDisplayName"].iloc[0]
bin_psi_results["Model_Name"] = df["Model_Name"].iloc[0]
bin_psi_results["modelVersionId"] = df["modelVersionId"].iloc[0]
bin_psi_results["trenchCategory"] = df["trenchCategory"].iloc[0]
bin_psi_results = bin_psi_results[
    [
        "modelDisplayName",
        "Model_Name",
        "modelVersionId",
        "trenchCategory",
        "Feature",
        "Feature_Type",
        "Segment_Column",
        "Segment_Value",
        "Month",
        "Base_Month",
        "Current_Month",
        "Bin",
        "Bin_Range",
        "Bin_Min",
        "Bin_Max",
        "Base_Percentage",
        "Actual_Percentage",
        "Bin_PSI",
    ]
].copy()

# Upload to BigQuery
table_id = (
    "prj-prod-dataplatform.dap_ds_poweruser_playground.temp_csi_new_monitoring_data"
)
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(psi_results, table_id, job_config=job_config)
job.result()  # Wait for the job to complete


# Upload to BigQuery
table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.temp_csi_new_monitoring_data_feature_bin_level"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(bin_psi_results, table_id, job_config=job_config)
job.result()  # Wait for the job to complete

# %% [markdown]
# ## SIL Beta - DemoScoreModel

# %%
sq = """ WITH cleaned AS (
  SELECT
  customerId,digitalLoanAccountId,prediction,start_time,end_time,modelDisplayName,modelVersionId,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature_cleaned
  FROM `prj-prod-dataplatform.audit_balance.ml_model_run_details`
  WHERE modelDisplayName = 'Beta - DemoScoreModel')
SELECT
  r.customerId,  r.digitalLoanAccountId,start_time, prediction sil_beta_demo_score, modelDisplayName,modelVersionId,
      loanmaster.new_loan_type,
 loanmaster.gender,
 'SIL Beta - DemoScoreModel' Model_Name,
 'SIL' as product,
 'NA' trenchCategory,
  r.calcFeature_cleaned calcFeatures,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  r.start_time) AS appln_submit_datetime,
  loanmaster.disbursementDateTime,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  r.start_time)) as Application_month,
FROM cleaned r
left join risk_credit_mis.loan_master_table loanmaster 
  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
qualify row_number() over(partition by r.customerId, r.digitalLoanAccountId order by start_time desc) = 1;"""

dfd = client.query(sq).to_dataframe()
dfd.head()

# %% [markdown]
# ### Expand df

# %%
# Expand the calcFeatures column
expanded_df = expand_calc_features(dfd)

# Display the result
print(f"Original columns: {dfd.shape[1]}")
print(f"Expanded columns: {expanded_df.shape[1]}")
expanded_df.head()


# %%
df = expanded_df.drop(columns=["calcFeatures"]).copy()
df = df.drop_duplicates(keep="first")

# %%
df["sil_beta_demo_score"] = pd.to_numeric(df["sil_beta_demo_score"], errors="coerce")


# %% [markdown]
# ### run psi calculation

# %%
df = df.copy()

# Define feature list
feature_list = [
    "sil_beta_demo_score",
    "calc_beta_de_ln_vas_opted_flag",
    "calc_beta_de_ln_doc_type_rolled",
    "calc_beta_de_ln_marital_status",
    "calc_beta_de_ln_age_bin",
    "calc_beta_de_ln_province_bin",
    "calc_beta_de_ln_ref2_type",
    "calc_beta_de_ln_education_level",
    "calc_beta_de_ln_ref1_type",
    "calc_beta_de_ln_industry_new_bin",
    "calc_beta_de_ln_appln_day_of_week",
    "calc_beta_de_onb_name_email_match_score",
    "calc_beta_de_ln_employment_type_new_bin",
    "calc_beta_de_ln_telconame",
    "calc_beta_de_time_bw_onb_loan_appln_mins",
    "calc_beta_de_ln_source_of_funds_new_bin",
    "calc_beta_de_ln_brand_bin",
    "calc_beta_de_ln_email_primary_domain",
]

# Define segment columns
segment_columns = ["new_loan_type", "gender", "osType", "trenchCategory", "product"]
# Calculate month-on-month PSI
psi_results = calculate_month_on_month_psi(df, feature_list, segment_columns)
psi_results["modelDisplayName"] = df["modelDisplayName"].iloc[0]
psi_results["Model_Name"] = df["Model_Name"].iloc[0]
psi_results["modelVersionId"] = df["modelVersionId"].iloc[0]
psi_results["trenchCategory"] = df["trenchCategory"].iloc[0]
psi_results = psi_results[
    [
        "modelDisplayName",
        "Model_Name",
        "modelVersionId",
        "trenchCategory",
        "Feature",
        "Feature_Type",
        "Segment_Column",
        "Segment_Value",
        "Month",
        "Base_Month",
        "Current_Month",
        "PSI",
    ]
].copy()

# Calculate bin-level PSI
bin_psi_results = calculate_bin_level_psi(df, feature_list, segment_columns)
bin_psi_results["modelDisplayName"] = df["modelDisplayName"].iloc[0]
bin_psi_results["Model_Name"] = df["Model_Name"].iloc[0]
bin_psi_results["modelVersionId"] = df["modelVersionId"].iloc[0]
bin_psi_results["trenchCategory"] = df["trenchCategory"].iloc[0]
bin_psi_results = bin_psi_results[
    [
        "modelDisplayName",
        "Model_Name",
        "modelVersionId",
        "trenchCategory",
        "Feature",
        "Feature_Type",
        "Segment_Column",
        "Segment_Value",
        "Month",
        "Base_Month",
        "Current_Month",
        "Bin",
        "Bin_Range",
        "Bin_Min",
        "Bin_Max",
        "Base_Percentage",
        "Actual_Percentage",
        "Bin_PSI",
    ]
].copy()

# Upload to BigQuery
table_id = (
    "prj-prod-dataplatform.dap_ds_poweruser_playground.temp_csi_new_monitoring_data"
)
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(psi_results, table_id, job_config=job_config)
job.result()  # Wait for the job to complete


# Upload to BigQuery
table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.temp_csi_new_monitoring_data_feature_bin_level"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(bin_psi_results, table_id, job_config=job_config)
job.result()  # Wait for the job to complete

# %%
psi_results

# %% [markdown]
# ## SIL_Beta - StackScoreModel

# %%
sq = """ WITH cleaned AS (
  SELECT
    customerId,digitalLoanAccountId,prediction,start_time,end_time,modelDisplayName,modelVersionId,
    REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature
  FROM `prj-prod-dataplatform.audit_balance.ml_model_run_details`
  WHERE modelDisplayName = 'Beta - StackScoreModel')
SELECT
  r.customerId,r.digitalLoanAccountId,prediction,start_time,end_time,modelDisplayName,modelVersionId,
 loanmaster.new_loan_type,
 loanmaster.gender,
 'SIL_Beta - StackScoreModel' Model_Name,
 'SIL' as product,
 'NA' trenchCategory,
  r.calcFeature calcFeatures,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  r.start_time) AS appln_submit_datetime,
  loanmaster.disbursementDateTime,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  r.start_time)) as Application_month,  
FROM cleaned r 
left join risk_credit_mis.loan_master_table loanmaster 
  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
qualify row_number() over (partition by r.customerId, r.digitalLoanAccountId order by start_time desc) = 1
;
"""
dfd = client.query(sq).to_dataframe()
dfd.head()


# %% [markdown]
# ### Expand df

# %%
# Expand the calcFeatures column
expanded_df = expand_calc_features(dfd)

# Display the result
print(f"Original columns: {dfd.shape[1]}")
print(f"Expanded columns: {expanded_df.shape[1]}")
expanded_df.head()


# %%
df = expanded_df.drop(columns=["calcFeatures"]).copy()
df.rename(columns={"prediction": "sil_beta_stack_score"}, inplace=True)
df = df.drop_duplicates(keep="first")

# %%
df["sil_beta_stack_score"] = pd.to_numeric(df["sil_beta_stack_score"], errors="coerce")


# %% [markdown]
# ### run psi calculation

# %%
df.columns

# %%
df = df.copy()

# Define feature list
feature_list = [
    "sil_beta_stack_score",
    "calc_s_apps_score",
    "calc_s_credo_score",
    "calc_sb_demo_score",
]

# Define segment columns
segment_columns = ["new_loan_type", "gender", "osType", "trenchCategory", "product"]
# Calculate month-on-month PSI
psi_results = calculate_month_on_month_psi(df, feature_list, segment_columns)
psi_results["modelDisplayName"] = df["modelDisplayName"].iloc[0]
psi_results["Model_Name"] = df["Model_Name"].iloc[0]
psi_results["modelVersionId"] = df["modelVersionId"].iloc[0]
psi_results["trenchCategory"] = df["trenchCategory"].iloc[0]
psi_results = psi_results[
    [
        "modelDisplayName",
        "Model_Name",
        "modelVersionId",
        "trenchCategory",
        "Feature",
        "Feature_Type",
        "Segment_Column",
        "Segment_Value",
        "Month",
        "Base_Month",
        "Current_Month",
        "PSI",
    ]
].copy()

# Calculate bin-level PSI
bin_psi_results = calculate_bin_level_psi(df, feature_list, segment_columns)
bin_psi_results["modelDisplayName"] = df["modelDisplayName"].iloc[0]
bin_psi_results["Model_Name"] = df["Model_Name"].iloc[0]
bin_psi_results["modelVersionId"] = df["modelVersionId"].iloc[0]
bin_psi_results["trenchCategory"] = df["trenchCategory"].iloc[0]
bin_psi_results = bin_psi_results[
    [
        "modelDisplayName",
        "Model_Name",
        "modelVersionId",
        "trenchCategory",
        "Feature",
        "Feature_Type",
        "Segment_Column",
        "Segment_Value",
        "Month",
        "Base_Month",
        "Current_Month",
        "Bin",
        "Bin_Range",
        "Bin_Min",
        "Bin_Max",
        "Base_Percentage",
        "Actual_Percentage",
        "Bin_PSI",
    ]
].copy()

# Upload to BigQuery
table_id = (
    "prj-prod-dataplatform.dap_ds_poweruser_playground.temp_csi_new_monitoring_data"
)
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(psi_results, table_id, job_config=job_config)
job.result()  # Wait for the job to complete


# Upload to BigQuery
table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.temp_csi_new_monitoring_data_feature_bin_level"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(bin_psi_results, table_id, job_config=job_config)
job.result()  # Wait for the job to complete

# %%
psi_results

# %%


# %% [markdown]
# ## Beta - IncomeEstimationModel

# %%
sq = """ 
WITH cleaned AS (
  SELECT
    customerId,digitalLoanAccountId,prediction,start_time,end_time,modelDisplayName,modelVersionId,

    REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature
  FROM `prj-prod-dataplatform.audit_balance.ml_model_run_details`
  WHERE modelDisplayName = 'Beta - IncomeEstimationModel')
SELECT
  r.customerId,r.digitalLoanAccountId,prediction sil_beta_income_estimation_score,start_time,end_time,modelDisplayName,modelVersionId,
  JSON_VALUE(calcFeature, "$.inc_beta_ln_loan_type") AS inc_beta_ln_loan_type,
  JSON_VALUE(calcFeature, "$.inc_beta_ln_education_level") AS inc_beta_ln_education_level,
  JSON_VALUE(calcFeature, "$.inc_beta_ln_employment_type_new") AS inc_beta_ln_employment_type_new,
  JSON_VALUE(calcFeature, "$.inc_beta_ln_industry_new") AS inc_beta_ln_industry_new,
  JSON_VALUE(calcFeature, "$.inc_beta_ln_age") AS inc_beta_ln_age,
  JSON_VALUE(calcFeature, "$.inc_beta_ln_brand") AS inc_beta_ln_brand,
  JSON_VALUE(calcFeature, "$.inc_beta_ln_city") AS inc_beta_ln_city,
  JSON_VALUE(calcFeature, "$.inc_beta_ln_purpose") AS inc_beta_ln_purpose,
  JSON_VALUE(calcFeature, "$.inc_beta_ln_osversion_bin") AS inc_beta_ln_osversion_bin,
  JSON_VALUE(calcFeature, "$.inc_beta_ln_postal_code") AS inc_beta_ln_postal_code,
  JSON_VALUE(calcFeature, "$.inc_beta_ln_gender") AS inc_beta_ln_gender,
  JSON_VALUE(calcFeature, "$.inc_beta_ln_doc_type_rolled") AS inc_beta_ln_doc_type_rolled,
  JSON_VALUE(calcFeature, "$.inc_beta_ln_cnt_dependents") AS inc_beta_ln_cnt_dependents,
  JSON_VALUE(calcFeature, "$.inc_beta_ln_source_of_funds_new") AS inc_beta_ln_source_of_funds_new,
  JSON_VALUE(calcFeature, "$.inc_beta_ln_marital_status_new") AS inc_beta_ln_marital_status_new,
  JSON_VALUE(calcFeature, "$.inc_beta_encoded_company_name_grouped") AS inc_beta_encoded_company_name_grouped,
   loanmaster.new_loan_type,
 loanmaster.gender,
 'SIL Beta - IncomeEstimationModel' Model_Name,
 'SIL' as product,
 'NA' trenchCategory,
  r.calcFeature calcFeatures,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  r.start_time) AS appln_submit_datetime,
  loanmaster.disbursementDateTime,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  r.start_time)) as Application_month,  
FROM cleaned r 
left join risk_credit_mis.loan_master_table loanmaster 
  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
qualify row_number() over (partition by r.customerId, r.digitalLoanAccountId order by start_time desc) = 1
  ;
"""
dfd = client.query(sq).to_dataframe()
dfd.head()

# %% [markdown]
# ### Expand df
#

# %%
# Expand the calcFeatures column
expanded_df = dfd.copy()

# Display the result
print(f"Original columns: {dfd.shape[1]}")
print(f"Expanded columns: {expanded_df.shape[1]}")
expanded_df.head()


# %%
df = expanded_df.drop(columns=["calcFeatures"]).copy()

# %%
df["sil_beta_income_estimation_score"] = pd.to_numeric(
    df["sil_beta_income_estimation_score"], errors="coerce"
)
df.columns

# %% [markdown]
# ### run psi calculation

# %%
df = df.copy()

# Define feature list
feature_list = [
    "sil_beta_income_estimation_score",
    "inc_beta_ln_loan_type",
    "inc_beta_ln_education_level",
    "inc_beta_ln_employment_type_new",
    "inc_beta_ln_industry_new",
    "inc_beta_ln_age",
    "inc_beta_ln_brand",
    "inc_beta_ln_city",
    "inc_beta_ln_purpose",
    "inc_beta_ln_osversion_bin",
    "inc_beta_ln_postal_code",
    "inc_beta_ln_gender",
    "inc_beta_ln_doc_type_rolled",
    "inc_beta_ln_cnt_dependents",
    "inc_beta_ln_source_of_funds_new",
    "inc_beta_ln_marital_status_new",
    "inc_beta_encoded_company_name_grouped",
]

# Define segment columns
segment_columns = ["new_loan_type", "gender", "osType", "trenchCategory", "product"]
# Calculate month-on-month PSI
psi_results = calculate_month_on_month_psi(df, feature_list, segment_columns)
psi_results["modelDisplayName"] = df["modelDisplayName"].iloc[0]
psi_results["Model_Name"] = df["Model_Name"].iloc[0]
psi_results["modelVersionId"] = df["modelVersionId"].iloc[0]
psi_results["trenchCategory"] = df["trenchCategory"].iloc[0]
psi_results = psi_results[
    [
        "modelDisplayName",
        "Model_Name",
        "modelVersionId",
        "trenchCategory",
        "Feature",
        "Feature_Type",
        "Segment_Column",
        "Segment_Value",
        "Month",
        "Base_Month",
        "Current_Month",
        "PSI",
    ]
].copy()

# Calculate bin-level PSI
bin_psi_results = calculate_bin_level_psi(df, feature_list, segment_columns)
bin_psi_results["modelDisplayName"] = df["modelDisplayName"].iloc[0]
bin_psi_results["Model_Name"] = df["Model_Name"].iloc[0]
bin_psi_results["modelVersionId"] = df["modelVersionId"].iloc[0]
bin_psi_results["trenchCategory"] = df["trenchCategory"].iloc[0]
bin_psi_results = bin_psi_results[
    [
        "modelDisplayName",
        "Model_Name",
        "modelVersionId",
        "trenchCategory",
        "Feature",
        "Feature_Type",
        "Segment_Column",
        "Segment_Value",
        "Month",
        "Base_Month",
        "Current_Month",
        "Bin",
        "Bin_Range",
        "Bin_Min",
        "Bin_Max",
        "Base_Percentage",
        "Actual_Percentage",
        "Bin_PSI",
    ]
].copy()

# Upload to BigQuery
table_id = (
    "prj-prod-dataplatform.dap_ds_poweruser_playground.temp_csi_new_monitoring_data"
)
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(psi_results, table_id, job_config=job_config)
job.result()  # Wait for the job to complete


# Upload to BigQuery
table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.temp_csi_new_monitoring_data_feature_bin_level"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(bin_psi_results, table_id, job_config=job_config)
job.result()  # Wait for the job to complete

# %%
bin_psi_results

# %% [markdown]
# # End

# %% [markdown]
#
