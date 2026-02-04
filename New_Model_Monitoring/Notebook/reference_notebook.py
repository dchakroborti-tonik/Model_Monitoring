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

# %% [markdown]
## Configure Settings
# Set options or configurations as needed
pd.set_option("display.max_columns", None)
pd.set_option("Display.max_rows", 100)

# %% [markdown]
# # Function

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
            binning_info[feature] = {"type": "numerical", "bins": None}
            continue

        # Create decile bins
        try:
            bins = np.percentile(valid_data, np.arange(0, 101, 10))
            # Remove duplicates and sort
            bins = np.unique(bins)
            binning_info[feature] = {"type": "numerical", "bins": bins}
        except:
            binning_info[feature] = {"type": "numerical", "bins": None}

    # Create bins for categorical features (top 6 + others)
    for feature in categorical_features:
        value_counts = base_df[feature].value_counts()
        top_6 = value_counts.nlargest(6).index.tolist()

        binning_info[feature] = {"type": "categorical", "top_categories": top_6}

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

        # Handle nulls
        binned = binned.astype(str)
        binned[df[feature].isna()] = "Missing"

        return binned

    else:  # categorical
        top_cats = binning_info["top_categories"]
        binned = df[feature].apply(lambda x: x if x in top_cats else "Others")
        binned = binned.fillna("Missing")

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

    # Clean up temporary columns
    for feature in feature_list:
        if f"{feature}_binned" in df.columns:
            df.drop(f"{feature}_binned", axis=1, inplace=True)

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
    pd.DataFrame with bin-level PSI details
    """
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

                        results.append(
                            {
                                "Feature": feature,
                                "Feature_Type": binning_info[feature]["type"],
                                "Segment_Column": segment_col,
                                "Segment_Value": segment_val,
                                "Month": f"{base_month} vs {month}",
                                "Base_Month": base_month,
                                "Current_Month": month,
                                "Bin": bin_name,
                                "Base_Percentage": (base_dist.get(bin_name, 0) * 100),
                                "Actual_Percentage": (
                                    actual_dist.get(bin_name, 0) * 100
                                ),
                                "Bin_PSI": bin_psi,
                            }
                        )

    # Clean up temporary columns
    for feature in feature_list:
        if f"{feature}_binned" in df.columns:
            df.drop(f"{feature}_binned", axis=1, inplace=True)

    return pd.DataFrame(results)


# # Example usage:
# if __name__ == "__main__":
#     # Load your data
#     # df = pd.read_json('CSI_testing_data.json', lines=True)

#     # Define feature list
#     feature_list = [
#         'cic_max_age_all_contracts_snapshot',
#         'cic_ratio_overdue_contracts_to_granted_contracts',
#         'cic_ScoreRange',
#         'cic_ln_loan_level_user_type',
#         'cic_has_ever_been_overdue',
#         'cic_latest_granted_contract_overdue_flag',
#         'cic_ratio_closed_over_new_granted_cnt_24M',
#         'cic_ratio_risky_contracts_to_granted_contracts',
#         'cic_Short_and_Term_Loans_granted_contracts_cnt_24M',
#         'cic_flg_zero_non_granted_ever',
#         'cic_Personal_Loans_granted_contracts_amt_24M',
#         'cic_CreditAvgCreditLimit',
#         'cic_flg_zero_granted_ever',
#     ]

#     # Define segment columns
#     segment_columns = ['new_loan_type', 'gender', 'osType', 'loanType', 'trenchCategory']

#     # Calculate month-on-month PSI
#     # psi_results = calculate_month_on_month_psi(df, feature_list, segment_columns)
#     # print(psi_results)

#     # Calculate bin-level PSI
#     # bin_psi_results = calculate_bin_level_psi(df, feature_list, segment_columns)
#     # print(bin_psi_results)

# %%
# sq = """drop table if exists prj-prod-dataplatform.dap_ds_poweruser_playground.temp_csi_new_monitoring_data;"""

# job = client.query(sq)
# job.result()  # Wait for job to complete
# print(f"Table  prj-prod-dataplatform.dap_ds_poweruser_playground.temp_csi_new_monitoring_data dropped successfully.")

# %% [markdown]
# # Queries

# %% [markdown]
# ## Alpha-Cash-CIC-Model

# %%
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
  REGEXP_EXTRACT(m.requestPayload_clean, r"osType[:=]['\"]?([^'\"]+)['\"]?") AS osType,
  REGEXP_EXTRACT(m.requestPayload_clean, r"loanType[:=]['\"]?([^'\"]+)['\"]?") AS loanType,
  REGEXP_EXTRACT(m.requestPayload_clean, r"trenchCategory[:=]['\"]?([^'\"]+)['\"]?") AS trenchCategory,
   SAFE_CAST(REGEXP_EXTRACT(m.requestPayload_clean, r"aStackScore[:= ]([0-9\.]+)") AS FLOAT64) AS aStackScore,
  SAFE_CAST(REGEXP_EXTRACT(m.requestPayload_clean, r"aCicScore[:= ]([0-9\.]+)") AS FLOAT64) AS aCicScore,
  --  Alpha CIC Score Model Features for Trench 1
  SAFE_CAST(JSON_VALUE(r.calcFeatures, "$.cic_max_age_all_contracts_snapshot") AS INT64) AS cic_max_age_all_contracts_snapshot,
  SAFE_CAST(JSON_VALUE(r.calcFeatures, "$.cic_ratio_overdue_contracts_to_granted_contracts") AS FLOAT64) AS cic_ratio_overdue_contracts_to_granted_contracts,
  JSON_VALUE(r.calcFeatures, "$.cic_ScoreRange") AS cic_ScoreRange,
  JSON_VALUE(r.calcFeatures, "$.cic_ln_loan_level_user_type") AS cic_ln_loan_level_user_type,
  JSON_VALUE(r.calcFeatures, "$.cic_has_ever_been_overdue") AS cic_has_ever_been_overdue,
  JSON_VALUE(r.calcFeatures, "$.cic_latest_granted_contract_overdue_flag") AS cic_latest_granted_contract_overdue_flag,
  JSON_VALUE(r.calcFeatures, "$.cic_ratio_closed_over_new_granted_cnt_24M") AS cic_ratio_closed_over_new_granted_cnt_24M,
  JSON_VALUE(r.calcFeatures, "$.cic_ratio_risky_contracts_to_granted_contracts") AS cic_ratio_risky_contracts_to_granted_contracts,
  JSON_VALUE(r.calcFeatures, "$.cic_Short_and_Term_Loans_granted_contracts_cnt_24M") AS cic_Short_and_Term_Loans_granted_contracts_cnt_24M,
  JSON_VALUE(r.calcFeatures, "$.cic_flg_zero_non_granted_ever") AS cic_flg_zero_non_granted_ever,
  JSON_VALUE(r.calcFeatures, "$.cic_Personal_Loans_granted_contracts_amt_24M") AS cic_Personal_Loans_granted_contracts_amt_24M,
  JSON_VALUE(r.calcFeatures, "$.cic_CreditAvgCreditLimit") AS cic_CreditAvgCreditLimit,
  JSON_VALUE(r.calcFeatures, "$.cic_flg_zero_granted_ever") AS cic_flg_zero_granted_ever,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  r.start_time) AS appln_submit_datetime,
  loanmaster.disbursementDateTime,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  r.start_time)) as Application_month, 
FROM latest_request r
left join model_run m
on r.digitalLoanAccountId = m.digitalLoanAccountId 
left join risk_credit_mis.loan_master_table loanmaster 
  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
 )
where trenchCategory = 'Trench 1'
)
select *, case when appln_submit_datetime <= '2025-09-30' then 'Train' else 'Test' end dataselection from base
"""

df = client.query(sq).to_dataframe()
df.head()

# %%
df.head(100).to_json(
    r"D:\OneDrive - Tonik Financial Pte Ltd\MyStuff\Data Engineering\Model_Monitoring\New_Model_Monitoring\Notebook\CSI_testing_data.json",
    orient="records",
    lines=True,
)

# %%
df["cic_flg_zero_granted_ever"].value_counts(dropna=False)

# %%
# convert the object column to numeric for correct population stability index calculation
columns_to_convert = [
    "cic_ratio_closed_over_new_granted_cnt_24M",
    "cic_ratio_risky_contracts_to_granted_contracts",
    "cic_Short_and_Term_Loans_granted_contracts_cnt_24M",
    "cic_Personal_Loans_granted_contracts_amt_24M",
    "cic_CreditAvgCreditLimit",
]

for col in columns_to_convert:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# %%
df.info()

# %%
df.to_csv(r"sample.csv", index=False)

# %%
sq = """drop table if exists prj-prod-dataplatform.dap_ds_poweruser_playground.temp_csi_new_monitoring_data;"""

job = client.query(sq)
job.result()  # Wait for job to complete
print(
    f"Table  prj-prod-dataplatform.dap_ds_poweruser_playground.temp_csi_new_monitoring_data dropped successfully."
)

# %%
df = df.copy()

# Define feature list
feature_list = [
    "aCicScore",
    "cic_max_age_all_contracts_snapshot",
    "cic_ratio_overdue_contracts_to_granted_contracts",
    "cic_ScoreRange",
    "cic_ln_loan_level_user_type",
    "cic_has_ever_been_overdue",
    "cic_latest_granted_contract_overdue_flag",
    "cic_ratio_closed_over_new_granted_cnt_24M",
    "cic_ratio_risky_contracts_to_granted_contracts",
    "cic_Short_and_Term_Loans_granted_contracts_cnt_24M",
    "cic_flg_zero_non_granted_ever",
    "cic_Personal_Loans_granted_contracts_amt_24M",
    "cic_CreditAvgCreditLimit",
    "cic_flg_zero_granted_ever",
]

# Define segment columns
segment_columns = ["new_loan_type", "gender", "osType", "loanType", "trenchCategory"]
# Calculate month-on-month PSI
psi_results = calculate_month_on_month_psi(df, feature_list, segment_columns)
psi_results["modelDisplayName"] = df["modelDisplayName"].iloc[0]
psi_results["modelVersionId"] = df["modelVersionId"].iloc[0]
psi_results["trenchCategory"] = df["trenchCategory"].iloc[0]
psi_results = psi_results[
    [
        "modelDisplayName",
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
bin_psi_results["modelVersionId"] = df["modelVersionId"].iloc[0]
bin_psi_results["trenchCategory"] = df["trenchCategory"].iloc[0]
bin_psi_results = bin_psi_results[
    [
        "modelDisplayName",
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
psi_results

# %%


# %% [markdown]
# ## Alpha-Cash-Stack-Model

# %%
sq = r"""WITH parsed as (
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
 r.prediction Alpha_Cash_Stack_Score,
 r.start_time,
 r.end_time,
 r.modelDisplayName,
 r.modelVersionId,
  loanmaster.new_loan_type,
 loanmaster.gender,

  REGEXP_EXTRACT(m.requestPayload_clean, r"osType[:=]['\"]?([^'\"]+)['\"]?") AS osType,
  REGEXP_EXTRACT(m.requestPayload_clean, r"loanType[:=]['\"]?([^'\"]+)['\"]?") AS loanType,
  REGEXP_EXTRACT(m.requestPayload_clean, r"trenchCategory[:=]['\"]?([^'\"]+)['\"]?") AS trenchCategory,

 SAFE_CAST(JSON_VALUE(r.calcFeatures, "$.apps_score") AS FLOAT64) AS  apps_score,
 SAFE_CAST(JSON_VALUE(r.calcFeatures, "$.c_demo_score") AS FLOAT64) AS  c_demo_score,
 SAFE_CAST(JSON_VALUE(r.calcFeatures, "$.c_credo_score") AS FLOAT64) AS  c_credo_score,
 SAFE_CAST(JSON_VALUE(r.calcFeatures, "$.c_tx_score") AS FLOAT64) AS  c_tx_score,
 SAFE_CAST(JSON_VALUE(r.calcFeatures, "$.ca_cic_score") AS FLOAT64) AS  ca_cic_score,
coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  r.start_time) AS appln_submit_datetime,
  loanmaster.disbursementDateTime,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  r.start_time)) as Application_month, 
FROM latest_request r
left join model_run m
on r.digitalLoanAccountId = m.digitalLoanAccountId 
left join risk_credit_mis.loan_master_table loanmaster 
  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
) where trenchCategory = 'Trench 1'
;
"""

df = client.query(sq).to_dataframe()
df.head()

# %%
df.columns

# %%
df.info()

# %%
df = df.copy()

# Define feature list
feature_list = [
    "Alpha_Cash_Stack_Score",
    "apps_score",
    "c_demo_score",
    "c_credo_score",
    "c_tx_score",
    "ca_cic_score",
]

# Define segment columns
segment_columns = ["new_loan_type", "gender", "osType", "loanType", "trenchCategory"]
# Calculate month-on-month PSI
psi_results = calculate_month_on_month_psi(df, feature_list, segment_columns)
psi_results["modelDisplayName"] = df["modelDisplayName"].iloc[0]
psi_results["modelVersionId"] = df["modelVersionId"].iloc[0]
psi_results["trenchCategory"] = df["trenchCategory"].iloc[0]
psi_results = psi_results[
    [
        "modelDisplayName",
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
bin_psi_results["modelVersionId"] = df["modelVersionId"].iloc[0]
bin_psi_results["trenchCategory"] = df["trenchCategory"].iloc[0]
bin_psi_results = bin_psi_results[
    [
        "modelDisplayName",
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
 r.prediction,
 r.start_time,
 r.end_time,
 r.modelDisplayName,
 r.modelVersionId,
 loanmaster.new_loan_type,
 loanmaster.gender,

  JSON_VALUE(m.requestPayload_clean, "$.predictions.trenchCategory") AS trenchCategory,
  JSON_VALUE(m.requestPayload_clean, "$.predictions.loanType") AS loanType,
  --  Demo Score Model Features for all Trench 1,2,3
  SAFE_CAST(JSON_VALUE(r.calcFeatures, "$.ln_vas_opted_flag") AS INT64) AS ln_vas_opted_flag,
  SAFE_CAST(JSON_VALUE(r.calcFeatures, "$.ln_self_dec_income") AS FLOAT64) AS ln_self_dec_income,
  SAFE_CAST(JSON_VALUE(r.calcFeatures, "$.ln_age") AS INT64) AS ln_age,
  JSON_VALUE(r.calcFeatures, "$.ln_source_funds_new_bin") AS ln_source_funds_new_bin,
  JSON_VALUE(r.calcFeatures, "$.ln_loan_level_user_type") AS ln_loan_level_user_type,
  JSON_VALUE(r.calcFeatures, "$.ln_industry_new_cat_bin") AS ln_industry_new_cat_bin,
  JSON_VALUE(r.calcFeatures, "$.ln_marital_status") AS ln_marital_status,
  JSON_VALUE(r.calcFeatures, "$.ln_doc_type_rolled") AS ln_doc_type_rolled,
  JSON_VALUE(r.calcFeatures, "$.ln_education_level") AS ln_education_level,
  JSON_VALUE(r.calcFeatures, "$.ln_ref2_type") AS ln_ref2_type,
  JSON_VALUE(r.calcFeatures, "$.ln_email_primary_domain") AS ln_email_primary_domain,
  JSON_VALUE(r.calcFeatures, "$.ln_os_type") AS ln_os_type,
  JSON_VALUE(r.calcFeatures, "$.ln_province_bin") AS ln_province_bin,
  SAFE_CAST(JSON_VALUE(r.calcFeatures, "$.ln_mature_fspd30_flag") AS INT64) AS ln_mature_fspd30_flag,
  SAFE_CAST(JSON_VALUE(r.calcFeatures, "$.ln_fspd30_flag") AS INT64) AS ln_fspd30_flag,
  SAFE_CAST(JSON_VALUE(r.calcFeatures, "$.ln_appln_submit_datetime") AS TIMESTAMP) AS ln_appln_submit_datetime,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  r.start_time) AS appln_submit_datetime,
  loanmaster.disbursementDateTime,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  r.start_time)) as Application_month, 
FROM latest_request r
left join model_run m
on r.digitalLoanAccountId = m.digitalLoanAccountId 
left join risk_credit_mis.loan_master_table loanmaster 
  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
)
where trenchCategory = 'Trench 1'

"""

df = client.query(sq).to_dataframe()
df.head()

# %%
df.columns

# %%
df.info()

# %%
df = df.copy()

# Define feature list
feature_list = [
    "Alpha_Cash_Stack_Score",
    "apps_score",
    "c_demo_score",
    "c_credo_score",
    "c_tx_score",
    "ca_cic_score",
]

# Define segment columns
segment_columns = ["new_loan_type", "gender", "osType", "loanType", "trenchCategory"]
# Calculate month-on-month PSI
psi_results = calculate_month_on_month_psi(df, feature_list, segment_columns)
psi_results["modelDisplayName"] = df["modelDisplayName"].iloc[0]
psi_results["modelVersionId"] = df["modelVersionId"].iloc[0]
psi_results["trenchCategory"] = df["trenchCategory"].iloc[0]
psi_results = psi_results[
    [
        "modelDisplayName",
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
bin_psi_results["modelVersionId"] = df["modelVersionId"].iloc[0]
bin_psi_results["trenchCategory"] = df["trenchCategory"].iloc[0]
bin_psi_results = bin_psi_results[
    [
        "modelDisplayName",
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
