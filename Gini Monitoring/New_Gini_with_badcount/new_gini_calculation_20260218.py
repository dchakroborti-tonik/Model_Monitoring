# %% [markdown]
# # Gini Calculation

# %% [markdown]
# ## Define Library

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
# ## Functions

# %% [markdown]
# ### calculate_gini


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
# ### calculate_periodic_gini_prod_ver_trench_dimfact

from datetime import timedelta
from itertools import combinations, product

import numpy as np

# %%
import pandas as pd
from sklearn.metrics import roc_auc_score


def calculate_gini(scores, labels):
    """
    Calculate Gini coefficient using ROC AUC score.

    Gini = 2 * AUC - 1

    The Gini coefficient measures the discriminatory power of a model:
    - Gini = 1.0: Perfect model (all positives ranked before negatives)
    - Gini = 0.0: Random model (no discriminatory power)
    - Gini = -1.0: Worst model (all negatives ranked before positives)

    Returns np.nan when:
    - Fewer than 2 observations
    - All labels are the same (no variation in labels)
    """
    labels_array = np.asarray(labels)
    scores_array = np.asarray(scores)

    if labels_array.size < 2:
        return np.nan

    if np.unique(labels_array).size < 2:  # all 0 or all 1
        return np.nan

    try:
        auc = roc_auc_score(labels_array, scores_array)
        gini = 2 * auc - 1
        return gini
    except Exception:
        return np.nan


def calculate_periodic_gini_prod_ver_trench_dimfact(
    df,
    score_column,
    label_column,
    namecolumn,
    data_selection_column=None,
    model_version_column=None,
    trench_column=None,
    loan_type_column=None,
    loan_product_type_column=None,
    ostype_column=None,
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
    data_selection_column: (optional) name of column for data selection (Test/Train)
    model_version_column: (optional) name of column for model version
    trench_column: (optional) name of column for trench category
    loan_type_column: (optional) name of loan type column
    loan_product_type_column: (optional) name of loan product type column
    ostype_column: (optional) name of column for OS type
    account_id_column: (optional) name of column for distinct account IDs
    """
    # Input validation
    required_columns = ["disbursementdate", score_column, label_column]
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"Missing required columns. Need: {required_columns}")

    optional_columns = {
        "data_selection": data_selection_column,
        "model_version": model_version_column,
        "trench": trench_column,
        "loan_type": loan_type_column,
        "loan_product_type": loan_product_type_column,
        "ostype": ostype_column,
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

    # Add week and month columns once to avoid recalculation
    df["week"] = df["disbursementdate"].dt.to_period("W")
    df["month"] = df["disbursementdate"].dt.to_period("M")

    # Create list of available segment columns
    segment_columns = []
    if data_selection_column:
        segment_columns.append(("DataSelection", data_selection_column))
    if model_version_column:
        segment_columns.append(("ModelVersion", model_version_column))
    if trench_column:
        segment_columns.append(("Trench", trench_column))
    if loan_type_column:
        segment_columns.append(("LoanType", loan_type_column))
    if loan_product_type_column:
        segment_columns.append(("ProductType", loan_product_type_column))
    if ostype_column:
        segment_columns.append(("OSType", ostype_column))

    # Build all segment combinations more efficiently
    datasets_to_process = [("Overall", df, {})]

    if segment_columns:
        # Generate combinations iteratively
        for r in range(1, len(segment_columns) + 1):
            for combo in combinations(segment_columns, r):
                # Create all combinations for this specific combo
                combo_cols = [col[1] for col in combo]
                combo_names = [col[0] for col in combo]

                # Get unique values for each column in combo
                unique_values = [
                    sorted(df[col].dropna().unique()) for col in combo_cols
                ]

                # Create cartesian product of unique values
                for values in product(*unique_values):
                    # Filter dataframe for this combination
                    filtered_df = df.copy()
                    combo_metadata = {}
                    combo_name_parts = []

                    for col, name, val in zip(combo_cols, combo_names, values):
                        filtered_df = filtered_df[filtered_df[col] == val]
                        combo_metadata[col] = val
                        combo_name_parts.append(f"{name}_{val}")

                    if len(filtered_df) > 0:
                        combo_name = "_".join(combo_name_parts)
                        datasets_to_process.append(
                            (combo_name, filtered_df, combo_metadata)
                        )

    all_results = []

    # Process each dataset
    for dataset_name, dataset_df, metadata in datasets_to_process:
        # Calculate weekly metrics in a single groupby operation
        weekly_groups = dataset_df.groupby("week")

        weekly_data = []
        for week, group in weekly_groups:
            group_size = len(group)

            # Only process if we have enough data
            if group_size >= 10:
                gini_value = calculate_gini(group[score_column], group[label_column])
            else:
                gini_value = np.nan

            distinct_accounts = (
                group[account_id_column].nunique() if account_id_column else None
            )
            bad_count = (
                group[group[label_column] == 1][account_id_column].nunique()
                if account_id_column
                else None
            )

            weekly_data.append(
                {
                    "week": week,
                    "gini_value": gini_value,
                    "distinct_accounts": distinct_accounts,
                    "bad_count": bad_count if bad_count else 0,
                    "period": "Week",
                }
            )

        weekly_gini = pd.DataFrame(weekly_data)

        if len(weekly_gini) > 0:
            weekly_gini["start_date"] = weekly_gini["week"].apply(
                lambda x: x.to_timestamp()
            )
            weekly_gini["end_date"] = weekly_gini["start_date"] + timedelta(days=6)
            weekly_gini = weekly_gini[
                [
                    "start_date",
                    "end_date",
                    "gini_value",
                    "period",
                    "distinct_accounts",
                    "bad_count",
                ]
            ]

        # Calculate monthly metrics in a single groupby operation
        monthly_groups = dataset_df.groupby("month")

        monthly_data = []
        for month, group in monthly_groups:
            group_size = len(group)

            # Only process if we have enough data
            if group_size >= 20:
                gini_value = calculate_gini(group[score_column], group[label_column])
            else:
                gini_value = np.nan

            distinct_accounts = (
                group[account_id_column].nunique() if account_id_column else None
            )
            bad_count = (
                group[group[label_column] == 1][account_id_column].nunique()
                if account_id_column
                else None
            )

            monthly_data.append(
                {
                    "month": month,
                    "gini_value": gini_value,
                    "distinct_accounts": distinct_accounts,
                    "bad_count": bad_count if bad_count else 0,
                    "period": "Month",
                }
            )

        monthly_gini = pd.DataFrame(monthly_data)

        if len(monthly_gini) > 0:
            monthly_gini["start_date"] = monthly_gini["month"].apply(
                lambda x: x.to_timestamp()
            )
            monthly_gini["end_date"] = (
                monthly_gini["start_date"]
                + pd.DateOffset(months=1)
                - pd.Timedelta(days=1)
            )
            monthly_gini = monthly_gini[
                [
                    "start_date",
                    "end_date",
                    "gini_value",
                    "period",
                    "distinct_accounts",
                    "bad_count",
                ]
            ]

        # Combine results for this dataset
        if len(weekly_gini) > 0 and len(monthly_gini) > 0:
            gini_results = pd.concat([weekly_gini, monthly_gini], ignore_index=True)
        elif len(weekly_gini) > 0:
            gini_results = weekly_gini.copy()
        elif len(monthly_gini) > 0:
            gini_results = monthly_gini.copy()
        else:
            continue  # Skip if no results

        gini_results = gini_results.sort_values(by="start_date").reset_index(drop=True)

        # Add metadata columns
        gini_results["Model_Name"] = score_column
        gini_results["bad_rate"] = namecolumn
        gini_results["segment_type"] = dataset_name
        gini_results["data_selection"] = (
            metadata.get(data_selection_column, None) if data_selection_column else None
        )
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
        gini_results["ostype"] = (
            metadata.get(ostype_column, None) if ostype_column else None
        )

        all_results.append(gini_results)

    # Combine all results
    if len(all_results) == 0:
        # Return empty dataframes if no results
        fact_table = pd.DataFrame()
        dimension_table = pd.DataFrame()
        return fact_table, dimension_table

    fact_table = pd.concat(all_results, ignore_index=True)

    # Create dimension table (unique segment combinations for filtering)
    dimension_cols = [
        "Model_Name",
        "bad_rate",
        "segment_type",
        "data_selection",
        "model_version",
        "trench_category",
        "loan_type",
        "loan_product_type",
        "ostype",
    ]

    dimension_table = (
        fact_table[dimension_cols].drop_duplicates().reset_index(drop=True)
    )
    dimension_table["segment_id"] = range(len(dimension_table))

    # Add segment_id to fact table using merge
    fact_table = fact_table.merge(
        dimension_table[["segment_id"] + dimension_cols],
        on=dimension_cols,
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
            "bad_count",
            "Model_Name",
            "bad_rate",
            "segment_type",
            "data_selection",
            "model_version",
            "trench_category",
            "loan_type",
            "loan_product_type",
            "ostype",
        ]
    ]

    # Reorder columns in dimension table
    dimension_table = dimension_table[
        [
            "segment_id",
            "Model_Name",
            "bad_rate",
            "segment_type",
            "data_selection",
            "model_version",
            "trench_category",
            "loan_type",
            "loan_product_type",
            "ostype",
        ]
    ]

    return fact_table, dimension_table


# %% [markdown]
# ### update_tables


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
        "ostype",
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
# ### Models

# %% [markdown]
# #### cic_model_sil

# %% [markdown]
# ##### FPD0

# %% [markdown]
# ###### Test

# %%
sq = """
with modelname as
  (SELECT
    mmrd.customerId,mmrd.digitalLoanAccountId,prediction Alpha_cic_sil_score,start_time,end_time,modelDisplayName,modelVersionId,
    case when trenchCategory is null then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench1' end)
     when trenchCategory = '' then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench 1' end)
    else trenchCategory end  as trenchCategory,
    REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature,
    deviceOs osType,                                                                                                                --- Added ostype
  FROM prj-prod-dataplatform.audit_balance.ml_model_run_details mmrd
  left join prj-prod-dataplatform.risk_credit_mis.model_loan_score_mart mt on mt.digitalLoanAccountId = mmrd.digitalLoanAccountId
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
  'Prod' Data_selection,
  deffpd0,
  flg_mature_fpd0,
  loanmaster.new_loan_type,
  modelVersionId, trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile'
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall'
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
    coalesce((case when lower(r.osType) like '%andro%' then 'android'
                  when lower(r.osType) like '%os%' then 'ios' else lower(r.osType) end),
            (case when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%andro%' then 'android'
                  when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%os%' then 'ios'
                  when lower(loanmaster.deviceType) like '%andro%' then 'android'
                  else 'ios' end)
            ) as osType
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
  qualify row_number() over(partition by digitalLoanAccountId, modelVersionId order by appln_submit_datetime) = 1
  ;
"""
dfd = client.query(sq).to_dataframe()
# dfd = dfd.drop_duplicates(keep='first')
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()

# %%
df1 = dfd.copy()

# %% [markdown]
# ###### Train

# %%
sq = """
  with modelname as
  (
   SELECT
    mmrd.customerId,mmrd.digitalLoanAccountId,prediction Alpha_cic_sil_score,start_time,end_time,modelDisplayName,modelVersionId,
    case when trenchCategory is null then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench1' end)
     when trenchCategory = '' then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench 1' end)
    else trenchCategory end  as trenchCategory,
    REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature,
    Data_selection,
    deviceOs osType,
  FROM prj-prod-dataplatform.dap_ds_poweruser_playground.ml_training_model_run_details_20260116 mmrd
  left join prj-prod-dataplatform.risk_credit_mis.model_loan_score_mart mt on mt.digitalLoanAccountId = mmrd.digitalLoanAccountId
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
  Data_selection,
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
        coalesce((case when lower(r.osType) like '%andro%' then 'android'
                  when lower(r.osType) like '%os%' then 'ios' else lower(r.osType) end),
            (case when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%andro%' then 'android'
                  when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%os%' then 'ios'
                  when lower(loanmaster.deviceType) like '%andro%' then 'android'
                  else 'ios' end)
            ) as osType
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
  select * from base
  qualify row_number() over(partition by digitalLoanAccountId, modelVersionId order by appln_submit_datetime) = 1
  ;
  """
dfd = client.query(sq).to_dataframe()
# dfd = dfd.drop_duplicates(keep='first')
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()

# %%
df2 = dfd.copy()

# %% [markdown]
# ###### Concatenate both Test and Train Datasets

# %%
# 1) Get all IDs present in Train
train_ids = set(df2["digitalLoanAccountId"])

# 2) Keep only Test rows whose ID is NOT in Train
df1_no_dupes = df1[~df1["digitalLoanAccountId"].isin(train_ids)]

# 3) Concatenate
df_concat = pd.concat([df1_no_dupes, df2], ignore_index=True)

print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")
df_concat.head()

# %% [markdown]
# ###### Making sure the score column values are numerical

# %%
df_concat["Alpha_cic_sil_score"] = pd.to_numeric(
    df_concat["Alpha_cic_sil_score"], errors="coerce"
)

# %% [markdown]
# ###### Calculate Gini

# %%
#### Calculating the Gini

fact_table, dimension_table = calculate_periodic_gini_prod_ver_trench_dimfact(
    df_concat,
    "Alpha_cic_sil_score",
    "deffpd0",
    "FPD0",
    data_selection_column="Data_selection",
    model_version_column="modelVersionId",
    trench_column="trenchCategory",
    loan_type_column="new_loan_type",
    loan_product_type_column="loan_product_type",
    ostype_column="osType",  # Add this
    account_id_column="digitalLoanAccountId",
)


# %% [markdown]
# ###### Update Tables

# %%
#### Updating Fact and Dimension Table
fact_table, dimension_table = update_tables(
    fact_table, dimension_table, "cic_model_sil", "SIL"
)


# %%
df_f_fpd0_cicsil = fact_table.copy()
df_d_fpd0_cicsil = dimension_table.copy()

print(
    f"The shape of fact table and copied dataframe are:\t {fact_table.shape} - {df_f_fpd0_cicsil.shape}"
)
print(
    f"The shape of dimension table and copied dataframe are:\t {dimension_table.shape} - {df_d_fpd0_cicsil.shape}"
)

# %%
facttable_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_cicsil"
dimtable_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_cicsil"

# %%
# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_TRUNCATE",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_f_fpd0_cicsil, facttable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_TRUNCATE",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_d_fpd0_cicsil, dimtable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# %% [markdown]
# ##### FPD10

# %% [markdown]
# ###### Test

# %%
#### Test
sq = """
with modelname as
  (SELECT
    mmrd.customerId,mmrd.digitalLoanAccountId,prediction Alpha_cic_sil_score,start_time,end_time,modelDisplayName,modelVersionId,
    case when trenchCategory is null then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench1' end)
     when trenchCategory = '' then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench 1' end)
    else trenchCategory end  as trenchCategory,
    REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature,
    deviceOs osType,
  FROM prj-prod-dataplatform.audit_balance.ml_model_run_details mmrd
  left join prj-prod-dataplatform.risk_credit_mis.model_loan_score_mart mt on mt.digitalLoanAccountId = mmrd.digitalLoanAccountId
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
  'Prod' Data_selection,
  del.deffpd10,
  del.flg_mature_fpd10,
  loanmaster.new_loan_type,
  modelVersionId, trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile'
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall'
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
     coalesce((case when lower(r.osType) like '%andro%' then 'android'
                  when lower(r.osType) like '%os%' then 'ios' else lower(r.osType) end),
            (case when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%andro%' then 'android'
                  when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%os%' then 'ios'
                  when lower(loanmaster.deviceType) like '%andro%' then 'android'
                  else 'ios' end)
            ) as osType
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
  qualify row_number() over(partition by digitalLoanAccountId, modelVersionId order by appln_submit_datetime) = 1
  ;
"""
dfd = client.query(sq).to_dataframe()
# dfd = dfd.drop_duplicates(keep='first')
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()


# %%
df1 = dfd.copy()

# %% [markdown]
# ###### Train

# %%
#### Train
sq = """
  with modelname as
  (SELECT
    mmrd.customerId,mmrd.digitalLoanAccountId,prediction Alpha_cic_sil_score,start_time,end_time,modelDisplayName,modelVersionId,
  case when trenchCategory is null then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench1' end)
     when trenchCategory = '' then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench 1' end)
    else trenchCategory end  as trenchCategory,
    REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature,
    Data_selection,
    deviceOs osType,
  FROM prj-prod-dataplatform.dap_ds_poweruser_playground.ml_training_model_run_details_20260116 mmrd
  left join prj-prod-dataplatform.risk_credit_mis.model_loan_score_mart mt on mt.digitalLoanAccountId = mmrd.digitalLoanAccountId
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
  Data_selection,
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
      coalesce((case when lower(r.osType) like '%andro%' then 'android'
                  when lower(r.osType) like '%os%' then 'ios' else lower(r.osType) end),
            (case when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%andro%' then 'android'
                  when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%os%' then 'ios'
                  when lower(loanmaster.deviceType) like '%andro%' then 'android'
                  else 'ios' end)
            ) as osType
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
  select * from base
  qualify row_number() over(partition by digitalLoanAccountId, modelVersionId order by appln_submit_datetime) = 1
  ;
  """
dfd = client.query(sq).to_dataframe()
# dfd = dfd.drop_duplicates(keep='first')
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()


# %%
df2 = dfd.copy()

# %% [markdown]
# ###### Concatenate both Test and Train Dataset

# %%
#### Concatenate both Test and Train Dataset

# 1) Get all IDs present in Train
train_ids = set(df2["digitalLoanAccountId"])

# 2) Keep only Test rows whose ID is NOT in Train
df1_no_dupes = df1[~df1["digitalLoanAccountId"].isin(train_ids)]

# 3) Concatenate
df_concat = pd.concat([df1_no_dupes, df2], ignore_index=True)

print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")
df_concat.head()

# %% [markdown]
# ###### Making Sure the Score is Numeric

# %%
df_concat["Alpha_cic_sil_score"] = pd.to_numeric(
    df_concat["Alpha_cic_sil_score"], errors="coerce"
)

# %% [markdown]
# ###### Calculate Gini

# %%
fact_table, dimension_table = calculate_periodic_gini_prod_ver_trench_dimfact(
    df_concat,
    "Alpha_cic_sil_score",
    "deffpd10",
    "FPD10",
    data_selection_column="Data_selection",  # Add this
    model_version_column="modelVersionId",
    trench_column="trenchCategory",
    loan_type_column="new_loan_type",
    loan_product_type_column="loan_product_type",
    ostype_column="osType",  # Add this
    account_id_column="digitalLoanAccountId",
)

# %% [markdown]
# ###### Update Tables

# %%
fact_table, dimension_table = update_tables(
    fact_table, dimension_table, model_name="cic_model_sil", product="SIL"
)

# %%
df_f_fpd10_cicsil = fact_table.copy()
df_d_fpd10_cicsil = dimension_table.copy()

print(
    f"The shape of fact table and copied dataframe are:\t {fact_table.shape} - {df_f_fpd10_cicsil.shape}"
)
print(
    f"The shape of dimension table and copied dataframe are:\t {dimension_table.shape} - {df_d_fpd10_cicsil.shape}"
)

# %%
# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_f_fpd10_cicsil, facttable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_d_fpd10_cicsil, dimtable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# %% [markdown]
# ##### FPD30

# %% [markdown]
# ###### Test

# %%
sq = """
with modelname as
  (SELECT
    mmrd.customerId,mmrd.digitalLoanAccountId,prediction Alpha_cic_sil_score,start_time,end_time,modelDisplayName,modelVersionId,
  case when trenchCategory is null then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench1' end)
     when trenchCategory = '' then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench 1' end)
    else trenchCategory end  as trenchCategory,
    REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature,
    deviceOs osType,
  FROM prj-prod-dataplatform.audit_balance.ml_model_run_details mmrd
  left join prj-prod-dataplatform.risk_credit_mis.model_loan_score_mart mt on mt.digitalLoanAccountId = mmrd.digitalLoanAccountId
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
  'Prod' Data_selection,
  del.deffpd30,
  del.flg_mature_fpd30,
  loanmaster.new_loan_type,
  modelVersionId, trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile'
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall'
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
     coalesce((case when lower(r.osType) like '%andro%' then 'android'
                  when lower(r.osType) like '%os%' then 'ios' else lower(r.osType) end),
            (case when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%andro%' then 'android'
                  when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%os%' then 'ios'
                  when lower(loanmaster.deviceType) like '%andro%' then 'android'
                  else 'ios' end)
            ) as osType
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
  qualify row_number() over(partition by digitalLoanAccountId, modelVersionId order by appln_submit_datetime) = 1
  ;
"""
dfd = client.query(sq).to_dataframe()
# dfd = dfd.drop_duplicates(keep='first')
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()


# %%
df1 = dfd.copy()

# %% [markdown]
# ###### Train

# %%
sq = """
  with modelname as
  (SELECT
    mmrd.customerId,mmrd.digitalLoanAccountId,prediction Alpha_cic_sil_score,start_time,end_time,modelDisplayName,modelVersionId,
     case when trenchCategory is null then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench1' end)
     when trenchCategory = '' then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench 1' end)
    else trenchCategory end  as trenchCategory,
    REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature,
    Data_selection,
    deviceOs osType,
  FROM prj-prod-dataplatform.dap_ds_poweruser_playground.ml_training_model_run_details_20260116 mmrd
  left join prj-prod-dataplatform.risk_credit_mis.model_loan_score_mart mt on mt.digitalLoanAccountId = mmrd.digitalLoanAccountId
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
  Data_selection,
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
       coalesce((case when lower(r.osType) like '%andro%' then 'android'
                  when lower(r.osType) like '%os%' then 'ios' else lower(r.osType) end),
            (case when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%andro%' then 'android'
                  when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%os%' then 'ios'
                  when lower(loanmaster.deviceType) like '%andro%' then 'android'
                  else 'ios' end)
            ) as osType
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
  select * from base
  qualify row_number() over(partition by digitalLoanAccountId, modelVersionId order by appln_submit_datetime) = 1
  ;
  """
dfd = client.query(sq).to_dataframe()
# dfd = dfd.drop_duplicates(keep='first')
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()


# %%
df2 = dfd.copy()

# %% [markdown]
# ###### Concatenate both test and train dataset

# %%
# 1) Get all IDs present in Train
train_ids = set(df2["digitalLoanAccountId"])

# 2) Keep only Test rows whose ID is NOT in Train
df1_no_dupes = df1[~df1["digitalLoanAccountId"].isin(train_ids)]

# 3) Concatenate
df_concat = pd.concat([df1_no_dupes, df2], ignore_index=True)

print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")
df_concat.head()

# %% [markdown]
# ###### Making sure the score column in numerical

# %%
df_concat["Alpha_cic_sil_score"] = pd.to_numeric(
    df_concat["Alpha_cic_sil_score"], errors="coerce"
)

# %% [markdown]
# ###### Calculate Gini

# %%
fact_table, dimension_table = calculate_periodic_gini_prod_ver_trench_dimfact(
    df_concat,
    "Alpha_cic_sil_score",
    "deffpd30",
    "FPD30",
    data_selection_column="Data_selection",  # Add this
    model_version_column="modelVersionId",
    trench_column="trenchCategory",
    loan_type_column="new_loan_type",
    loan_product_type_column="loan_product_type",
    ostype_column="osType",  # Add this
    account_id_column="digitalLoanAccountId",
)

# %% [markdown]
# ###### Update Tables

# %%
fact_table, dimension_table = update_tables(
    fact_table, dimension_table, model_name="cic_model_sil", product="SIL"
)

# %%
df_f_fpd30_cicsil = fact_table.copy()
df_d_fpd30_cicsil = dimension_table.copy()

print(
    f"The shape of fact table and copied dataframe are:\t {fact_table.shape} - {df_f_fpd30_cicsil.shape}"
)
print(
    f"The shape of dimension table and copied dataframe are:\t {dimension_table.shape} - {df_d_fpd30_cicsil.shape}"
)

# %%
# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_f_fpd30_cicsil, facttable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_d_fpd30_cicsil, dimtable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# %% [markdown]
# ##### FSPD30

# %% [markdown]
# ###### Test

# %%
sq = """
with modelname as
  (SELECT
    mmrd.customerId,mmrd.digitalLoanAccountId,prediction Alpha_cic_sil_score,start_time,end_time,modelDisplayName,modelVersionId,
    case when trenchCategory is null then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench1' end)
     when trenchCategory = '' then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench 1' end)
    else trenchCategory end  as trenchCategory,
    REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature,
    deviceOs osType,
    FROM prj-prod-dataplatform.audit_balance.ml_model_run_details mmrd
  left join prj-prod-dataplatform.risk_credit_mis.model_loan_score_mart mt on mt.digitalLoanAccountId = mmrd.digitalLoanAccountId
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
  'Prod' Data_selection,
  del.deffspd30,
  del.flg_mature_fspd_30,
  loanmaster.new_loan_type,
  modelVersionId, trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile'
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall'
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
     coalesce((case when lower(r.osType) like '%andro%' then 'android'
                  when lower(r.osType) like '%os%' then 'ios' else lower(r.osType) end),
            (case when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%andro%' then 'android'
                  when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%os%' then 'ios'
                  when lower(loanmaster.deviceType) like '%andro%' then 'android'
                  else 'ios' end)
            ) as osType

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
  qualify row_number() over(partition by digitalLoanAccountId, modelVersionId order by appln_submit_datetime) = 1
  ;
"""
dfd = client.query(sq).to_dataframe()
# dfd = dfd.drop_duplicates(keep='first')
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()


# %%
df1 = dfd.copy()

# %% [markdown]
# ###### Train

# %%
sq = """
  with modelname as
  (SELECT
    mmrd.customerId,mmrd.digitalLoanAccountId,prediction Alpha_cic_sil_score,start_time,end_time,modelDisplayName,modelVersionId,
   case when trenchCategory is null then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench1' end)
     when trenchCategory = '' then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench 1' end)
    else trenchCategory end  as trenchCategory,
    REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature,
    Data_selection,
    deviceOs osType,
  FROM prj-prod-dataplatform.dap_ds_poweruser_playground.ml_training_model_run_details_20260116 mmrd
  left join prj-prod-dataplatform.risk_credit_mis.model_loan_score_mart mt on mt.digitalLoanAccountId = mmrd.digitalLoanAccountId
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
  Data_selection,
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
     coalesce((case when lower(r.osType) like '%andro%' then 'android'
                  when lower(r.osType) like '%os%' then 'ios' else lower(r.osType) end),
            (case when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%andro%' then 'android'
                  when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%os%' then 'ios'
                  when lower(loanmaster.deviceType) like '%andro%' then 'android'
                  else 'ios' end)
            ) as osType
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
  select * from base
  qualify row_number() over(partition by digitalLoanAccountId, modelVersionId order by appln_submit_datetime) = 1
  ;
  """
dfd = client.query(sq).to_dataframe()
# dfd = dfd.drop_duplicates(keep='first')
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()

# %%
df2 = dfd.copy()

# %% [markdown]
# ###### Concatenate both test and train dataset

# %%
# 1) Get all IDs present in Train
train_ids = set(df2["digitalLoanAccountId"])

# 2) Keep only Test rows whose ID is NOT in Train
df1_no_dupes = df1[~df1["digitalLoanAccountId"].isin(train_ids)]

# 3) Concatenate
df_concat = pd.concat([df1_no_dupes, df2], ignore_index=True)

print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")
df_concat.head()

# %% [markdown]
# ###### Making sure the score column is numeric

# %%
df_concat["Alpha_cic_sil_score"] = pd.to_numeric(
    df_concat["Alpha_cic_sil_score"], errors="coerce"
)

# %% [markdown]
# ###### Calculate Gini

# %%
fact_table, dimension_table = calculate_periodic_gini_prod_ver_trench_dimfact(
    df_concat,
    "Alpha_cic_sil_score",
    "deffspd30",
    "FSPD30",
    data_selection_column="Data_selection",  # Add this
    model_version_column="modelVersionId",
    trench_column="trenchCategory",
    loan_type_column="new_loan_type",
    loan_product_type_column="loan_product_type",
    ostype_column="osType",
    account_id_column="digitalLoanAccountId",
)


# %% [markdown]
# ###### Update Tables

# %%
fact_table, dimension_table = update_tables(
    fact_table, dimension_table, model_name="cic_model_sil", product="SIL"
)

# %%
df_f_fspd30_cicsil = fact_table.copy()
df_d_fspd30_cicsil = dimension_table.copy()

print(
    f"The shape of fact table and copied dataframe are:\t {fact_table.shape} - {df_f_fspd30_cicsil.shape}"
)
print(
    f"The shape of dimension table and copied dataframe are:\t {dimension_table.shape} - {df_d_fspd30_cicsil.shape}"
)

# %%
# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_f_fspd30_cicsil, facttable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_d_fspd30_cicsil, dimtable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# %% [markdown]
# ##### FSTPD30

# %% [markdown]
# ###### Test

# %%
sq = """
with modelname as
  (SELECT
    mmrd.customerId,mmrd.digitalLoanAccountId,prediction Alpha_cic_sil_score,start_time,end_time,modelDisplayName,modelVersionId,
    case when trenchCategory is null then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench1' end)
     when trenchCategory = '' then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench 1' end)
    else trenchCategory end  as trenchCategory,
    REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature,
    deviceOs osType,
  FROM prj-prod-dataplatform.audit_balance.ml_model_run_details mmrd
  left join prj-prod-dataplatform.risk_credit_mis.model_loan_score_mart mt on mt.digitalLoanAccountId = mmrd.digitalLoanAccountId
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
  'Prod' Data_selection,
  del.deffstpd30,
  del.flg_mature_fstpd_30,
  loanmaster.new_loan_type,
  modelVersionId, trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile'
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall'
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
     coalesce((case when lower(r.osType) like '%andro%' then 'android'
                  when lower(r.osType) like '%os%' then 'ios' else lower(r.osType) end),
            (case when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%andro%' then 'android'
                  when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%os%' then 'ios'
                  when lower(loanmaster.deviceType) like '%andro%' then 'android'
                  else 'ios' end)
            ) as osType
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
  qualify row_number() over(partition by digitalLoanAccountId, modelVersionId order by appln_submit_datetime) = 1
  ;
"""
dfd = client.query(sq).to_dataframe()
# dfd = dfd.drop_duplicates(keep='first')
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()


# %%
df1 = dfd.copy()

# %% [markdown]
# ###### Train

# %%
sq = """
  with modelname as
  (  SELECT
    mmrd.customerId,mmrd.digitalLoanAccountId,prediction Alpha_cic_sil_score,start_time,end_time,modelDisplayName,modelVersionId,
   case when trenchCategory is null then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench1' end)
     when trenchCategory = '' then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench 1' end)
    else trenchCategory end  as trenchCategory,
    REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature,
    Data_selection,
    deviceOs osType,
  FROM prj-prod-dataplatform.dap_ds_poweruser_playground.ml_training_model_run_details_20260116 mmrd
  left join prj-prod-dataplatform.risk_credit_mis.model_loan_score_mart mt on mt.digitalLoanAccountId = mmrd.digitalLoanAccountId
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
   Data_selection,
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
       coalesce((case when lower(r.osType) like '%andro%' then 'android'
                  when lower(r.osType) like '%os%' then 'ios' else lower(r.osType) end),
            (case when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%andro%' then 'android'
                  when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%os%' then 'ios'
                  when lower(loanmaster.deviceType) like '%andro%' then 'android'
                  else 'ios' end)
            ) as osType
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
  select * from base
  qualify row_number() over(partition by digitalLoanAccountId, modelVersionId order by appln_submit_datetime) = 1
  ;
  """
dfd = client.query(sq).to_dataframe()
# dfd = dfd.drop_duplicates(keep='first')
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()


# %%
df2 = dfd.copy()

# %% [markdown]
# ###### Concatenate both Test and Train Datasets

# %%
# %% [markdown]

# 1) Get all IDs present in Train
train_ids = set(df2["digitalLoanAccountId"])

# 2) Keep only Test rows whose ID is NOT in Train
df1_no_dupes = df1[~df1["digitalLoanAccountId"].isin(train_ids)]

# 3) Concatenate
df_concat = pd.concat([df1_no_dupes, df2], ignore_index=True)

print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")
df_concat.head()


# %% [markdown]
# ###### Making sure the score column is numeric

# %%
df_concat["Alpha_cic_sil_score"] = pd.to_numeric(
    df_concat["Alpha_cic_sil_score"], errors="coerce"
)


# %% [markdown]
# ###### Calculate Gini

# %%

fact_table, dimension_table = calculate_periodic_gini_prod_ver_trench_dimfact(
    df_concat,
    "Alpha_cic_sil_score",
    "deffstpd30",
    "FSTPD30",
    data_selection_column="Data_selection",  # Add this
    model_version_column="modelVersionId",
    trench_column="trenchCategory",
    loan_type_column="new_loan_type",
    loan_product_type_column="loan_product_type",
    ostype_column="osType",
    account_id_column="digitalLoanAccountId",
)


# %% [markdown]
# ###### Update Tables

# %%
fact_table, dimension_table = update_tables(
    fact_table, dimension_table, model_name="cic_model_sil", product="SIL"
)

# %%
df_f_fstpd30_cicsil = fact_table.copy()
df_d_fstpd30_cicsil = dimension_table.copy()

print(
    f"The shape of fact table and copied dataframe are:\t {fact_table.shape} - {df_f_fstpd30_cicsil.shape}"
)
print(
    f"The shape of dimension table and copied dataframe are:\t {dimension_table.shape} - {df_d_fstpd30_cicsil.shape}"
)

# %%
# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_f_fstpd30_cicsil, facttable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_d_fstpd30_cicsil, dimtable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# %%
df_fact_cic_sil = pd.concat(
    [
        df_f_fpd0_cicsil,
        df_f_fpd10_cicsil,
        df_f_fpd30_cicsil,
        df_f_fspd30_cicsil,
        df_f_fstpd30_cicsil,
    ],
    ignore_index=True,
)
df_dim_cic_sil = pd.concat(
    [
        df_d_fpd0_cicsil,
        df_d_fpd10_cicsil,
        df_d_fpd30_cicsil,
        df_d_fspd30_cicsil,
        df_d_fstpd30_cicsil,
    ],
    ignore_index=True,
)

# %% [markdown]
# #### alpha_stack_model_sil_credo_score

# %% [markdown]
# ##### FPD0

# %% [markdown]
# ###### Test

# %%
# sq = f"""with modelname as
#   (  SELECT
#     mmrd.customerId,mmrd.digitalLoanAccountId,prediction,start_time,end_time,modelDisplayName,modelVersionId,
#   case when trenchCategory is null then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
#     when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
#     else 'Trench1' end)
#      when trenchCategory = '' then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
#     when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
#     else 'Trench 1' end)
#     else trenchCategory end  as trenchCategory,
#     REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature,
#       deviceOs osType,
#     FROM prj-prod-dataplatform.audit_balance.ml_model_run_details mmrd
#   left join prj-prod-dataplatform.risk_credit_mis.model_loan_score_mart mt on mt.digitalLoanAccountId = mmrd.digitalLoanAccountId
#   WHERE modelDisplayName in  ('Alpha - StackingModel', 'alpha_stack_model_sil')
#   -- and modelVersionId = 'v1'
#       ),
#   deliquency as
# (select loanAccountNumber,
# case when obs_min_inst_def0 >= 1 and min_inst_def0 = 1 then 1 else 0 end deffpd0,
# case when obs_min_inst_def10 >=1 and min_inst_def10 =1 then 1 else 0 end deffpd10,
# case when obs_min_inst_def30 >=1 and min_inst_def30 =1 then 1 else 0 end deffpd30,
# case when obs_min_inst_def30 >=2 and min_inst_def30 in (1,2) then 1 else 0 end deffspd30,
# case when obs_min_inst_def30 >=3 and min_inst_def30 in (1,2,3) then 1 else 0 end deffstpd30,
# case when obs_min_inst_def0 >= 1 then 1 else 0 end flg_mature_fpd0,
# case when obs_min_inst_def10 >=1 then 1 else 0 end flg_mature_fpd10,
# case when obs_min_inst_def30 >=1 then 1 else 0 end flg_mature_fpd30,
# case when obs_min_inst_def30 >=2 then 1 else 0 end flg_mature_fspd_30,
# case when obs_min_inst_def30 >=3 then 1 else 0 end flg_mature_fstpd_30
# from prj-prod-dataplatform.risk_credit_mis.loan_deliquency_data),
# base as
#   (select distinct r.customerId,
#   r.digitalLoanAccountId,
#   loanmaster.loanAccountNumber,
#   r.modelDisplayName,
#   coalesce(cast(JSON_VALUE(SAFE.PARSE_JSON(CAST(calcFeature AS STRING)), '$.credo_score')AS FLOAT64),
#   cast(JSON_VALUE(SAFE.PARSE_JSON(CAST(calcFeature AS STRING)), '$.s_credo_score')AS FLOAT64)) AS credo_score,
#   calcFeature,
#   coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
#   date(loanmaster.disbursementDateTime) disbursementdate,
#   format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month,
#   'Prod' Data_selection,
#   deffpd0,
#   flg_mature_fpd0,
#   loanmaster.new_loan_type,
#   modelVersionId, trenchCategory,
#     case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
#     when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile'
#     when loanmaster.loantype='BNPL' and store_type =3 then 'Mall'
#     when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
#     else 'not applicable' end as loan_product_type,
#         coalesce((case when lower(r.osType) like '%andro%' then 'android'
#                   when lower(r.osType) like '%os%' then 'ios' else lower(r.osType) end),
#             (case when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%andro%' then 'android'
#                   when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%os%' then 'ios'
#                   when lower(loanmaster.deviceType) like '%andro%' then 'android'
#                   else 'ios' end)
#             ) as osType
#   from modelname r
#   left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
#   left join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
#    left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
#   left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
#  qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
#   where
#   loanmaster.flagDisbursement = 1
#   and loanmaster.disbursementDateTime is not null
#   and flg_mature_fpd0 = 1
#   )
#   select *
#   from base
#   where credo_score is not null
#  qualify row_number() over(partition by digitalLoanAccountId, modelVersionId order by appln_submit_datetime) = 1
#    ;
# """
# dfd = client.query(sq).to_dataframe()
# # dfd = dfd.drop_duplicates(keep='first')
# print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
# dfd.head()

# %%
# df1 = dfd.copy()

# %% [markdown]
# ###### Train

# %%
# sq = """
# with modelname as
#   (  SELECT
#     mmrd.customerId,mmrd.digitalLoanAccountId,prediction,start_time,end_time,modelDisplayName,modelVersionId,
#   case when trenchCategory is null then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
#     when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
#     else 'Trench1' end)
#      when trenchCategory = '' then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
#     when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
#     else 'Trench 1' end)
#     else trenchCategory end  as trenchCategory,
#     REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature,
#     Data_selection,
#        deviceOs osType,
#     FROM prj-prod-dataplatform.dap_ds_poweruser_playground.ml_training_model_run_details_20260116 mmrd
#   left join prj-prod-dataplatform.risk_credit_mis.model_loan_score_mart mt on mt.digitalLoanAccountId = mmrd.digitalLoanAccountId
#   WHERE modelDisplayName in  ('Alpha - StackingModel', 'alpha_stack_model_sil')
#   -- and modelVersionId = 'v1'
#       ),
#   deliquency as
# (select loanAccountNumber,
# case when obs_min_inst_def0 >= 1 and min_inst_def0 = 1 then 1 else 0 end deffpd0,
# case when obs_min_inst_def10 >=1 and min_inst_def10 =1 then 1 else 0 end deffpd10,
# case when obs_min_inst_def30 >=1 and min_inst_def30 =1 then 1 else 0 end deffpd30,
# case when obs_min_inst_def30 >=2 and min_inst_def30 in (1,2) then 1 else 0 end deffspd30,
# case when obs_min_inst_def30 >=3 and min_inst_def30 in (1,2,3) then 1 else 0 end deffstpd30,
# case when obs_min_inst_def0 >= 1 then 1 else 0 end flg_mature_fpd0,
# case when obs_min_inst_def10 >=1 then 1 else 0 end flg_mature_fpd10,
# case when obs_min_inst_def30 >=1 then 1 else 0 end flg_mature_fpd30,
# case when obs_min_inst_def30 >=2 then 1 else 0 end flg_mature_fspd_30,
# case when obs_min_inst_def30 >=3 then 1 else 0 end flg_mature_fstpd_30
# from prj-prod-dataplatform.risk_credit_mis.loan_deliquency_data),
# base as
#   (select distinct r.customerId,
#   r.digitalLoanAccountId,
#   loanmaster.loanAccountNumber,
#   r.modelDisplayName,
#   coalesce(cast(JSON_VALUE(SAFE.PARSE_JSON(CAST(calcFeature AS STRING)), '$.credo_score')AS FLOAT64),
#   cast(JSON_VALUE(SAFE.PARSE_JSON(CAST(calcFeature AS STRING)), '$.s_credo_score')AS FLOAT64)) AS credo_score,
#   calcFeature,
#   coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
#   date(loanmaster.disbursementDateTime) disbursementdate,
#   format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month,
#   Data_selection,
#   deffpd0,
#   flg_mature_fpd0,
#   loanmaster.new_loan_type,
#   modelVersionId, trenchCategory,
#     case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
#     when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile'
#     when loanmaster.loantype='BNPL' and store_type =3 then 'Mall'
#     when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
#     else 'not applicable' end as loan_product_type,
#         coalesce((case when lower(r.osType) like '%andro%' then 'android'
#                   when lower(r.osType) like '%os%' then 'ios' else lower(r.osType) end),
#             (case when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%andro%' then 'android'
#                   when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%os%' then 'ios'
#                   when lower(loanmaster.deviceType) like '%andro%' then 'android'
#                   else 'ios' end)
#             ) as osType
#   from modelname r
#   left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
#   left join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
#    left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
#   left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
#  qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
#   where
#   loanmaster.flagDisbursement = 1
#   and loanmaster.disbursementDateTime is not null
#   and flg_mature_fpd0 = 1
#   )
#   select *
#   from base
#   where credo_score is not null
#  qualify row_number() over(partition by digitalLoanAccountId, modelVersionId order by appln_submit_datetime) = 1
#    ;
#   """
# dfd = client.query(sq).to_dataframe()
# # dfd = dfd.drop_duplicates(keep='first')
# print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
# dfd.head()

# %%
# df2 = dfd.copy()

# %% [markdown]
# ###### Concatenate test and train datasets

# %%
# # 1) Get all IDs present in Train
# train_ids = set(df2["digitalLoanAccountId"])

# # 2) Keep only Test rows whose ID is NOT in Train
# df1_no_dupes = df1[~df1["digitalLoanAccountId"].isin(train_ids)]

# # 3) Concatenate
# df_concat = pd.concat([df1_no_dupes, df2], ignore_index=True)

# print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")
# df_concat.head()

# %% [markdown]
# ###### Making sure credo score for alpha sil is numeric

# %%
# df_concat["credo_score"] = pd.to_numeric(df_concat["credo_score"], errors="coerce")

# %% [markdown]
# ###### Calculate Gini

# %%
# fact_table, dimension_table = calculate_periodic_gini_prod_ver_trench_dimfact(
#     df_concat,
#     "credo_score",
#     "deffpd0",
#     "FPD0",
#     data_selection_column="Data_selection",
#     model_version_column="modelVersionId",
#     trench_column="trenchCategory",
#     loan_type_column="new_loan_type",
#     loan_product_type_column="loan_product_type",
#     ostype_column="osType",
#     account_id_column="digitalLoanAccountId",
# )

# %% [markdown]
# ###### Update Tables

# %%
# fact_table, dimension_table = update_tables(
#     fact_table, dimension_table, model_name="alpha_stack_credo_score_sil", product="SIL"
# )
# print(f"The shape of the fact table is:\t {fact_table.shape}")
# print(f"The shape of the dimension table is:\t {dimension_table.shape}")

# %%
# df_f_fpd0_alphacredosil = fact_table.copy()
# df_d_fpd0_alphacredosil = dimension_table.copy()

# print(f"The shape of fact table and copied dataframe are:\t {fact_table.shape} - {df_f_fpd0_alphacredosil.shape}")
# print(f"The shape of dimension table and copied dataframe are:\t {dimension_table.shape} - {df_d_fpd0_alphacredosil.shape}")

# %%
# facttable_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_cicsil"
# dimtable_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_cicsil"

# %% [markdown]
# ##### FPD10

# %% [markdown]
# ###### Test

# %%
# sq = f"""with modelname as
#   (  SELECT
#     mmrd.customerId,mmrd.digitalLoanAccountId,prediction,start_time,end_time,modelDisplayName,modelVersionId,
#   case when trenchCategory is null then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
#     when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
#     else 'Trench1' end)
#      when trenchCategory = '' then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
#     when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
#     else 'Trench 1' end)
#     else trenchCategory end  as trenchCategory,
#     REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature,
#         deviceOs osType,
#     FROM prj-prod-dataplatform.audit_balance.ml_model_run_details mmrd
#   left join prj-prod-dataplatform.risk_credit_mis.model_loan_score_mart mt on mt.digitalLoanAccountId = mmrd.digitalLoanAccountId
#   WHERE modelDisplayName in  ('Alpha - StackingModel', 'alpha_stack_model_sil')
#   -- and modelVersionId = 'v1'
#       ),
#   deliquency as
# (select loanAccountNumber,
# case when obs_min_inst_def0 >= 1 and min_inst_def0 = 1 then 1 else 0 end deffpd0,
# case when obs_min_inst_def10 >=1 and min_inst_def10 =1 then 1 else 0 end deffpd10,
# case when obs_min_inst_def30 >=1 and min_inst_def30 =1 then 1 else 0 end deffpd30,
# case when obs_min_inst_def30 >=2 and min_inst_def30 in (1,2) then 1 else 0 end deffspd30,
# case when obs_min_inst_def30 >=3 and min_inst_def30 in (1,2,3) then 1 else 0 end deffstpd30,
# case when obs_min_inst_def0 >= 1 then 1 else 0 end flg_mature_fpd0,
# case when obs_min_inst_def10 >=1 then 1 else 0 end flg_mature_fpd10,
# case when obs_min_inst_def30 >=1 then 1 else 0 end flg_mature_fpd30,
# case when obs_min_inst_def30 >=2 then 1 else 0 end flg_mature_fspd_30,
# case when obs_min_inst_def30 >=3 then 1 else 0 end flg_mature_fstpd_30
# from prj-prod-dataplatform.risk_credit_mis.loan_deliquency_data),
# base as
#   (select distinct r.customerId,
#   r.digitalLoanAccountId,
#   loanmaster.loanAccountNumber,
#   r.modelDisplayName,
#   coalesce(cast(JSON_VALUE(SAFE.PARSE_JSON(CAST(calcFeature AS STRING)), '$.credo_score')AS FLOAT64),
#   cast(JSON_VALUE(SAFE.PARSE_JSON(CAST(calcFeature AS STRING)), '$.s_credo_score')AS FLOAT64)) AS credo_score,
#   calcFeature,
#   coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
#   date(loanmaster.disbursementDateTime) disbursementdate,
#   format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month,
#   'Prod' Data_selection,
#   del.deffpd10,
#   del.flg_mature_fpd10,
#   loanmaster.new_loan_type,
#   modelVersionId, trenchCategory,
#     case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
#     when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile'
#     when loanmaster.loantype='BNPL' and store_type =3 then 'Mall'
#     when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
#     else 'not applicable' end as loan_product_type,
#         coalesce((case when lower(r.osType) like '%andro%' then 'android'
#                   when lower(r.osType) like '%os%' then 'ios' else lower(r.osType) end),
#             (case when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%andro%' then 'android'
#                   when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%os%' then 'ios'
#                   when lower(loanmaster.deviceType) like '%andro%' then 'android'
#                   else 'ios' end)
#             ) as osType
#   from modelname r
#   left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
#   left join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
#    left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
#   left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
#  qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
#   where
#   loanmaster.flagDisbursement = 1
#   and loanmaster.disbursementDateTime is not null
#   and del.flg_mature_fpd10 = 1
#   )
#   select *
#   from base
#   where credo_score is not null
#  qualify row_number() over(partition by digitalLoanAccountId, modelVersionId order by appln_submit_datetime) = 1
#    ;
# """
# dfd = client.query(sq).to_dataframe()
# # dfd = dfd.drop_duplicates(keep='first')
# print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
# dfd.head()

# %%
# df1 = dfd.copy()

# %% [markdown]
# ###### Train

# %%
# sq = """
# with modelname as
#   (  SELECT
#     mmrd.customerId,mmrd.digitalLoanAccountId,prediction,start_time,end_time,modelDisplayName,modelVersionId,
#   case when trenchCategory is null then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
#     when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
#     else 'Trench1' end)
#      when trenchCategory = '' then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
#     when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
#     else 'Trench 1' end)
#     else trenchCategory end  as trenchCategory,
#     REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature,
#     Data_selection,
#         deviceOs osType,
#     FROM prj-prod-dataplatform.dap_ds_poweruser_playground.ml_training_model_run_details_20260116 mmrd
#   left join prj-prod-dataplatform.risk_credit_mis.model_loan_score_mart mt on mt.digitalLoanAccountId = mmrd.digitalLoanAccountId
#   WHERE modelDisplayName in  ('Alpha - StackingModel', 'alpha_stack_model_sil')
#   -- and modelVersionId = 'v1'
#       ),
#   deliquency as
# (select loanAccountNumber,
# case when obs_min_inst_def0 >= 1 and min_inst_def0 = 1 then 1 else 0 end deffpd0,
# case when obs_min_inst_def10 >=1 and min_inst_def10 =1 then 1 else 0 end deffpd10,
# case when obs_min_inst_def30 >=1 and min_inst_def30 =1 then 1 else 0 end deffpd30,
# case when obs_min_inst_def30 >=2 and min_inst_def30 in (1,2) then 1 else 0 end deffspd30,
# case when obs_min_inst_def30 >=3 and min_inst_def30 in (1,2,3) then 1 else 0 end deffstpd30,
# case when obs_min_inst_def0 >= 1 then 1 else 0 end flg_mature_fpd0,
# case when obs_min_inst_def10 >=1 then 1 else 0 end flg_mature_fpd10,
# case when obs_min_inst_def30 >=1 then 1 else 0 end flg_mature_fpd30,
# case when obs_min_inst_def30 >=2 then 1 else 0 end flg_mature_fspd_30,
# case when obs_min_inst_def30 >=3 then 1 else 0 end flg_mature_fstpd_30
# from prj-prod-dataplatform.risk_credit_mis.loan_deliquency_data),
# base as
#   (select distinct r.customerId,
#   r.digitalLoanAccountId,
#   loanmaster.loanAccountNumber,
#   r.modelDisplayName,
#   coalesce(cast(JSON_VALUE(SAFE.PARSE_JSON(CAST(calcFeature AS STRING)), '$.credo_score')AS FLOAT64),
#   cast(JSON_VALUE(SAFE.PARSE_JSON(CAST(calcFeature AS STRING)), '$.s_credo_score')AS FLOAT64)) AS credo_score,
#   calcFeature,
#   coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
#   date(loanmaster.disbursementDateTime) disbursementdate,
#   format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month,
#   'Train' Data_selection,
#   del.deffpd10,
#   del.flg_mature_fpd10,
#   loanmaster.new_loan_type,
#   modelVersionId, trenchCategory,
#     case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
#     when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile'
#     when loanmaster.loantype='BNPL' and store_type =3 then 'Mall'
#     when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
#     else 'not applicable' end as loan_product_type,
#         coalesce((case when lower(r.osType) like '%andro%' then 'android'
#                   when lower(r.osType) like '%os%' then 'ios' else lower(r.osType) end),
#             (case when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%andro%' then 'android'
#                   when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%os%' then 'ios'
#                   when lower(loanmaster.deviceType) like '%andro%' then 'android'
#                   else 'ios' end)
#             ) as osType
#   from modelname r
#   left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
#   left join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
#    left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
#   left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
#  qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
#   where
#   loanmaster.flagDisbursement = 1
#   and loanmaster.disbursementDateTime is not null
#   and del.flg_mature_fpd10 = 1
#   )
#   select *
#   from base
#   where credo_score is not null
#  qualify row_number() over(partition by digitalLoanAccountId, modelVersionId order by appln_submit_datetime) = 1
#    ;
#   """
# dfd = client.query(sq).to_dataframe()
# # dfd = dfd.drop_duplicates(keep='first')
# print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
# dfd.head()

# %%
# df2 = dfd.copy()

# %% [markdown]
# ###### Concatenate Test and Train dataset

# %%
# # 1) Get all IDs present in Train
# train_ids = set(df2["digitalLoanAccountId"])

# # 2) Keep only Test rows whose ID is NOT in Train
# df1_no_dupes = df1[~df1["digitalLoanAccountId"].isin(train_ids)]

# # 3) Concatenate
# df_concat = pd.concat([df1_no_dupes, df2], ignore_index=True)

# print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")
# df_concat.head()

# %% [markdown]
# ###### Making sure the alpha credo score is numeric

# %%
# df_concat["credo_score"] = pd.to_numeric(df_concat["credo_score"], errors="coerce")

# %% [markdown]
# ###### Calculate Gini

# %%
# fact_table, dimension_table = calculate_periodic_gini_prod_ver_trench_dimfact(
#     df_concat,
#     "credo_score",
#     "deffpd10",
#     "FPD10",
#     data_selection_column="Data_selection",
#     model_version_column="modelVersionId",
#     trench_column="trenchCategory",
#     loan_type_column="new_loan_type",
#     loan_product_type_column="loan_product_type",
#     ostype_column="osType",
#     account_id_column="digitalLoanAccountId",
# )

# %%
# fact_table, dimension_table = update_tables(
#     fact_table, dimension_table, model_name="alpha_stack_credo_score_sil", product="SIL"
# )
# print(f"The shape of the fact table is:\t {fact_table.shape}")
# print(f"The shape of the dimension table is:\t {dimension_table.shape}")

# %%
# df_f_fpd10_alphacredosil = fact_table.copy()
# df_d_fpd10_alphacredosil = dimension_table.copy()

# print(f"The shape of fact table and copied dataframe are:\t {fact_table.shape} - {df_f_fpd10_alphacredosil.shape}")
# print(f"The shape of dimension table and copied dataframe are:\t {dimension_table.shape} - {df_d_fpd10_alphacredosil.shape}")

# %% [markdown]
# ##### FPD30

# %% [markdown]
# ###### Test

# %%
# sq = f"""with modelname as
#   (  SELECT
#     mmrd.customerId,mmrd.digitalLoanAccountId,prediction,start_time,end_time,modelDisplayName,modelVersionId,
#   case when trenchCategory is null then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
#     when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
#     else 'Trench1' end)
#      when trenchCategory = '' then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
#     when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
#     else 'Trench 1' end)
#     else trenchCategory end  as trenchCategory,
#     REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature,
#         deviceOs osType,
#     FROM prj-prod-dataplatform.audit_balance.ml_model_run_details mmrd
#   left join prj-prod-dataplatform.risk_credit_mis.model_loan_score_mart mt on mt.digitalLoanAccountId = mmrd.digitalLoanAccountId
#   WHERE modelDisplayName in  ('Alpha - StackingModel', 'alpha_stack_model_sil')
#   -- and modelVersionId = 'v1'
#       ),
#   deliquency as
# (select loanAccountNumber,
# case when obs_min_inst_def0 >= 1 and min_inst_def0 = 1 then 1 else 0 end deffpd0,
# case when obs_min_inst_def10 >=1 and min_inst_def10 =1 then 1 else 0 end deffpd10,
# case when obs_min_inst_def30 >=1 and min_inst_def30 =1 then 1 else 0 end deffpd30,
# case when obs_min_inst_def30 >=2 and min_inst_def30 in (1,2) then 1 else 0 end deffspd30,
# case when obs_min_inst_def30 >=3 and min_inst_def30 in (1,2,3) then 1 else 0 end deffstpd30,
# case when obs_min_inst_def0 >= 1 then 1 else 0 end flg_mature_fpd0,
# case when obs_min_inst_def10 >=1 then 1 else 0 end flg_mature_fpd10,
# case when obs_min_inst_def30 >=1 then 1 else 0 end flg_mature_fpd30,
# case when obs_min_inst_def30 >=2 then 1 else 0 end flg_mature_fspd_30,
# case when obs_min_inst_def30 >=3 then 1 else 0 end flg_mature_fstpd_30
# from prj-prod-dataplatform.risk_credit_mis.loan_deliquency_data),
# base as
#   (select distinct r.customerId,
#   r.digitalLoanAccountId,
#   loanmaster.loanAccountNumber,
#   r.modelDisplayName,
#   coalesce(cast(JSON_VALUE(SAFE.PARSE_JSON(CAST(calcFeature AS STRING)), '$.credo_score')AS FLOAT64),
#   cast(JSON_VALUE(SAFE.PARSE_JSON(CAST(calcFeature AS STRING)), '$.s_credo_score')AS FLOAT64)) AS credo_score,
#   calcFeature,
#   coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
#   date(loanmaster.disbursementDateTime) disbursementdate,
#   format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month,
#   'Prod' Data_selection,
#   del.deffpd30,
#   del.flg_mature_fpd30,
#   loanmaster.new_loan_type,
#   modelVersionId, trenchCategory,
#     case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
#     when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile'
#     when loanmaster.loantype='BNPL' and store_type =3 then 'Mall'
#     when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
#     else 'not applicable' end as loan_product_type,
#         coalesce((case when lower(r.osType) like '%andro%' then 'android'
#                   when lower(r.osType) like '%os%' then 'ios' else lower(r.osType) end),
#             (case when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%andro%' then 'android'
#                   when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%os%' then 'ios'
#                   when lower(loanmaster.deviceType) like '%andro%' then 'android'
#                   else 'ios' end)
#             ) as osType
#   from modelname r
#   left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
#   left join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
#    left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
#   left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
#  qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
#   where
#   loanmaster.flagDisbursement = 1
#   and loanmaster.disbursementDateTime is not null
#   and del.flg_mature_fpd30 = 1
#   )
#   select *
#   from base
#   where credo_score is not null
#  qualify row_number() over(partition by digitalLoanAccountId, modelVersionId order by appln_submit_datetime) = 1
#    ;
# """
# dfd = client.query(sq).to_dataframe()
# # dfd = dfd.drop_duplicates(keep='first')
# print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
# dfd.head()

# %%
# df1 = dfd.copy()

# %% [markdown]
# ###### Train

# %%
# sq = """
# with modelname as
#   (  SELECT
#     mmrd.customerId,mmrd.digitalLoanAccountId,prediction,start_time,end_time,modelDisplayName,modelVersionId,
#   case when trenchCategory is null then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
#     when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
#     else 'Trench1' end)
#      when trenchCategory = '' then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
#     when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
#     else 'Trench 1' end)
#     else trenchCategory end  as trenchCategory,
#     REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature,
#     Data_selection,
#         deviceOs osType,
#     FROM prj-prod-dataplatform.dap_ds_poweruser_playground.ml_training_model_run_details_20260116 mmrd
#   left join prj-prod-dataplatform.risk_credit_mis.model_loan_score_mart mt on mt.digitalLoanAccountId = mmrd.digitalLoanAccountId
#   WHERE modelDisplayName in  ('Alpha - StackingModel', 'alpha_stack_model_sil')
#   -- and modelVersionId = 'v1'
#       ),
#   deliquency as
# (select loanAccountNumber,
# case when obs_min_inst_def0 >= 1 and min_inst_def0 = 1 then 1 else 0 end deffpd0,
# case when obs_min_inst_def10 >=1 and min_inst_def10 =1 then 1 else 0 end deffpd10,
# case when obs_min_inst_def30 >=1 and min_inst_def30 =1 then 1 else 0 end deffpd30,
# case when obs_min_inst_def30 >=2 and min_inst_def30 in (1,2) then 1 else 0 end deffspd30,
# case when obs_min_inst_def30 >=3 and min_inst_def30 in (1,2,3) then 1 else 0 end deffstpd30,
# case when obs_min_inst_def0 >= 1 then 1 else 0 end flg_mature_fpd0,
# case when obs_min_inst_def10 >=1 then 1 else 0 end flg_mature_fpd10,
# case when obs_min_inst_def30 >=1 then 1 else 0 end flg_mature_fpd30,
# case when obs_min_inst_def30 >=2 then 1 else 0 end flg_mature_fspd_30,
# case when obs_min_inst_def30 >=3 then 1 else 0 end flg_mature_fstpd_30
# from prj-prod-dataplatform.risk_credit_mis.loan_deliquency_data),
# base as
#   (select distinct r.customerId,
#   r.digitalLoanAccountId,
#   loanmaster.loanAccountNumber,
#   r.modelDisplayName,
#   coalesce(cast(JSON_VALUE(SAFE.PARSE_JSON(CAST(calcFeature AS STRING)), '$.credo_score')AS FLOAT64),
#   cast(JSON_VALUE(SAFE.PARSE_JSON(CAST(calcFeature AS STRING)), '$.s_credo_score')AS FLOAT64)) AS credo_score,
#   calcFeature,
#   coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
#   date(loanmaster.disbursementDateTime) disbursementdate,
#   format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month,
#   Data_selection,
#   del.deffpd30,
#   del.flg_mature_fpd30,
#   loanmaster.new_loan_type,
#   modelVersionId, trenchCategory,
#     case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
#     when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile'
#     when loanmaster.loantype='BNPL' and store_type =3 then 'Mall'
#     when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
#     else 'not applicable' end as loan_product_type,
#         coalesce((case when lower(r.osType) like '%andro%' then 'android'
#                   when lower(r.osType) like '%os%' then 'ios' else lower(r.osType) end),
#             (case when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%andro%' then 'android'
#                   when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%os%' then 'ios'
#                   when lower(loanmaster.deviceType) like '%andro%' then 'android'
#                   else 'ios' end)
#             ) as osType
#   from modelname r
#   left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
#   left join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
#    left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
#   left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
#  qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
#   where
#   loanmaster.flagDisbursement = 1
#   and loanmaster.disbursementDateTime is not null
#   and del.flg_mature_fpd30 = 1
#   )
#   select *
#   from base
#   where credo_score is not null
#  qualify row_number() over(partition by digitalLoanAccountId, modelVersionId order by appln_submit_datetime) = 1
#    ;
#   """
# dfd = client.query(sq).to_dataframe()
# # dfd = dfd.drop_duplicates(keep='first')
# print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
# dfd.head()

# %%
# df2 = dfd.copy()

# %% [markdown]
# ###### Concatenate Test and Train Dataset

# %%
# # 1) Get all IDs present in Train
# train_ids = set(df2["digitalLoanAccountId"])

# # 2) Keep only Test rows whose ID is NOT in Train
# df1_no_dupes = df1[~df1["digitalLoanAccountId"].isin(train_ids)]

# # 3) Concatenate
# df_concat = pd.concat([df1_no_dupes, df2], ignore_index=True)

# print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")
# df_concat.head()

# %%
# df_concat["credo_score"] = pd.to_numeric(df_concat["credo_score"], errors="coerce")

# %% [markdown]
# ###### Calculate Gini

# %%
# fact_table, dimension_table = calculate_periodic_gini_prod_ver_trench_dimfact(
#     df_concat,
#     "credo_score",
#     "deffpd30",
#     "FPD30",
#     data_selection_column="Data_selection",
#     model_version_column="modelVersionId",
#     trench_column="trenchCategory",
#     loan_type_column="new_loan_type",
#     loan_product_type_column="loan_product_type",
#     ostype_column="osType",
#     account_id_column="digitalLoanAccountId",
# )

# %% [markdown]
# ###### Update Table

# %%
# fact_table, dimension_table = update_tables(
#     fact_table, dimension_table, model_name="alpha_stack_credo_score_sil", product="SIL"
# )
# print(f"The shape of the fact table is:\t {fact_table.shape}")
# print(f"The shape of the dimension table is:\t {dimension_table.shape}")

# %%
# df_f_fpd30_alphacredosil = fact_table.copy()
# df_d_fpd30_alphacredosil = dimension_table.copy()

# print(f"The shape of fact table and copied dataframe are:\t {fact_table.shape} - {df_f_fpd30_alphacredosil.shape}")
# print(f"The shape of dimension table and copied dataframe are:\t {dimension_table.shape} - {df_d_fpd30_alphacredosil.shape}")

# %% [markdown]
# ##### FSPD30

# %% [markdown]
# ###### Test

# %%
# sq = f"""with modelname as
#   (  SELECT
#     mmrd.customerId,mmrd.digitalLoanAccountId,prediction,start_time,end_time,modelDisplayName,modelVersionId,
#   case when trenchCategory is null then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
#     when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
#     else 'Trench1' end)
#      when trenchCategory = '' then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
#     when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
#     else 'Trench 1' end)
#     else trenchCategory end  as trenchCategory,
#     REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature,
#         deviceOs osType,
#     FROM prj-prod-dataplatform.audit_balance.ml_model_run_details mmrd
#   left join prj-prod-dataplatform.risk_credit_mis.model_loan_score_mart mt on mt.digitalLoanAccountId = mmrd.digitalLoanAccountId
#   WHERE modelDisplayName in  ('Alpha - StackingModel', 'alpha_stack_model_sil')
#   -- and modelVersionId = 'v1'
#       ),
#   deliquency as
# (select loanAccountNumber,
# case when obs_min_inst_def0 >= 1 and min_inst_def0 = 1 then 1 else 0 end deffpd0,
# case when obs_min_inst_def10 >=1 and min_inst_def10 =1 then 1 else 0 end deffpd10,
# case when obs_min_inst_def30 >=1 and min_inst_def30 =1 then 1 else 0 end deffpd30,
# case when obs_min_inst_def30 >=2 and min_inst_def30 in (1,2) then 1 else 0 end deffspd30,
# case when obs_min_inst_def30 >=3 and min_inst_def30 in (1,2,3) then 1 else 0 end deffstpd30,
# case when obs_min_inst_def0 >= 1 then 1 else 0 end flg_mature_fpd0,
# case when obs_min_inst_def10 >=1 then 1 else 0 end flg_mature_fpd10,
# case when obs_min_inst_def30 >=1 then 1 else 0 end flg_mature_fpd30,
# case when obs_min_inst_def30 >=2 then 1 else 0 end flg_mature_fspd_30,
# case when obs_min_inst_def30 >=3 then 1 else 0 end flg_mature_fstpd_30
# from prj-prod-dataplatform.risk_credit_mis.loan_deliquency_data),
# base as
#   (select distinct r.customerId,
#   r.digitalLoanAccountId,
#   loanmaster.loanAccountNumber,
#   r.modelDisplayName,
#   coalesce(cast(JSON_VALUE(SAFE.PARSE_JSON(CAST(calcFeature AS STRING)), '$.credo_score')AS FLOAT64),
#   cast(JSON_VALUE(SAFE.PARSE_JSON(CAST(calcFeature AS STRING)), '$.s_credo_score')AS FLOAT64)) AS credo_score,
#   calcFeature,
#   coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
#   date(loanmaster.disbursementDateTime) disbursementdate,
#   format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month,
#   'Prod' Data_selection,
#   del.deffspd30,
#   del.flg_mature_fspd_30,
#   loanmaster.new_loan_type,
#   modelVersionId, trenchCategory,
#     case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
#     when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile'
#     when loanmaster.loantype='BNPL' and store_type =3 then 'Mall'
#     when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
#     else 'not applicable' end as loan_product_type,
#         coalesce((case when lower(r.osType) like '%andro%' then 'android'
#                   when lower(r.osType) like '%os%' then 'ios' else lower(r.osType) end),
#             (case when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%andro%' then 'android'
#                   when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%os%' then 'ios'
#                   when lower(loanmaster.deviceType) like '%andro%' then 'android'
#                   else 'ios' end)
#             ) as osType
#   from modelname r
#   left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
#   left join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
#    left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
#   left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
#  qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
#   where
#   loanmaster.flagDisbursement = 1
#   and loanmaster.disbursementDateTime is not null
#   and del.flg_mature_fspd_30 = 1
#   )
#   select *
#   from base
#   where credo_score is not null
#  qualify row_number() over(partition by digitalLoanAccountId, modelVersionId order by appln_submit_datetime) = 1
#    ;
# """
# dfd = client.query(sq).to_dataframe()
# # dfd = dfd.drop_duplicates(keep='first')
# print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
# dfd.head()

# %%
# df1 = dfd.copy()

# %% [markdown]
# ###### Train

# %%
# sq = """
# with modelname as
#   (  SELECT
#     mmrd.customerId,mmrd.digitalLoanAccountId,prediction,start_time,end_time,modelDisplayName,modelVersionId,
#   case when trenchCategory is null then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
#     when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
#     else 'Trench1' end)
#      when trenchCategory = '' then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
#     when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
#     else 'Trench 1' end)
#     else trenchCategory end  as trenchCategory,
#     REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature,
#     Data_selection,
#         deviceOs osType,
#     FROM prj-prod-dataplatform.dap_ds_poweruser_playground.ml_training_model_run_details_20260116 mmrd
#   left join prj-prod-dataplatform.risk_credit_mis.model_loan_score_mart mt on mt.digitalLoanAccountId = mmrd.digitalLoanAccountId
#   WHERE modelDisplayName in  ('Alpha - StackingModel', 'alpha_stack_model_sil')
#   -- and modelVersionId = 'v1'
#       ),
#   deliquency as
# (select loanAccountNumber,
# case when obs_min_inst_def0 >= 1 and min_inst_def0 = 1 then 1 else 0 end deffpd0,
# case when obs_min_inst_def10 >=1 and min_inst_def10 =1 then 1 else 0 end deffpd10,
# case when obs_min_inst_def30 >=1 and min_inst_def30 =1 then 1 else 0 end deffpd30,
# case when obs_min_inst_def30 >=2 and min_inst_def30 in (1,2) then 1 else 0 end deffspd30,
# case when obs_min_inst_def30 >=3 and min_inst_def30 in (1,2,3) then 1 else 0 end deffstpd30,
# case when obs_min_inst_def0 >= 1 then 1 else 0 end flg_mature_fpd0,
# case when obs_min_inst_def10 >=1 then 1 else 0 end flg_mature_fpd10,
# case when obs_min_inst_def30 >=1 then 1 else 0 end flg_mature_fpd30,
# case when obs_min_inst_def30 >=2 then 1 else 0 end flg_mature_fspd_30,
# case when obs_min_inst_def30 >=3 then 1 else 0 end flg_mature_fstpd_30
# from prj-prod-dataplatform.risk_credit_mis.loan_deliquency_data),
# base as
#   (select distinct r.customerId,
#   r.digitalLoanAccountId,
#   loanmaster.loanAccountNumber,
#   r.modelDisplayName,
#   coalesce(cast(JSON_VALUE(SAFE.PARSE_JSON(CAST(calcFeature AS STRING)), '$.credo_score')AS FLOAT64),
#   cast(JSON_VALUE(SAFE.PARSE_JSON(CAST(calcFeature AS STRING)), '$.s_credo_score')AS FLOAT64)) AS credo_score,
#   calcFeature,
#   coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
#   date(loanmaster.disbursementDateTime) disbursementdate,
#   format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month,
#   Data_selection,
#   del.deffspd30,
#   del.flg_mature_fspd_30,
#   loanmaster.new_loan_type,
#   modelVersionId, trenchCategory,
#     case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
#     when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile'
#     when loanmaster.loantype='BNPL' and store_type =3 then 'Mall'
#     when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
#     else 'not applicable' end as loan_product_type,
#         coalesce((case when lower(r.osType) like '%andro%' then 'android'
#                   when lower(r.osType) like '%os%' then 'ios' else lower(r.osType) end),
#             (case when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%andro%' then 'android'
#                   when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%os%' then 'ios'
#                   when lower(loanmaster.deviceType) like '%andro%' then 'android'
#                   else 'ios' end)
#             ) as osType
#   from modelname r
#   left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
#   left join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
#    left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
#   left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
#  qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
#   where
#   loanmaster.flagDisbursement = 1
#   and loanmaster.disbursementDateTime is not null
#   and del.flg_mature_fspd_30 = 1
#   )
#   select *
#   from base
#   where credo_score is not null
#  qualify row_number() over(partition by digitalLoanAccountId, modelVersionId order by appln_submit_datetime) = 1
#    ;
#   """
# dfd = client.query(sq).to_dataframe()
# # dfd = dfd.drop_duplicates(keep='first')
# print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
# dfd.head()

# %%
# df2 = dfd.copy()

# %% [markdown]
# ###### Concatenate Test and Train dataset

# %%
# # 1) Get all IDs present in Train
# train_ids = set(df2["digitalLoanAccountId"])

# # 2) Keep only Test rows whose ID is NOT in Train
# df1_no_dupes = df1[~df1["digitalLoanAccountId"].isin(train_ids)]

# # 3) Concatenate
# df_concat = pd.concat([df1_no_dupes, df2], ignore_index=True)

# print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")
# df_concat.head()

# %%
# df_concat["credo_score"] = pd.to_numeric(df_concat["credo_score"], errors="coerce")

# %% [markdown]
# ###### Calculate Gini

# %%
# fact_table, dimension_table = calculate_periodic_gini_prod_ver_trench_dimfact(
#     df_concat,
#     "credo_score",
#     "deffspd30",
#     "FSPD30",
#     data_selection_column="Data_selection",
#     model_version_column="modelVersionId",
#     trench_column="trenchCategory",
#     loan_type_column="new_loan_type",
#     loan_product_type_column="loan_product_type",
#     ostype_column="osType",
#     account_id_column="digitalLoanAccountId",
# )

# %%
# fact_table, dimension_table = update_tables(
#     fact_table, dimension_table, model_name="alpha_stack_credo_score_sil", product="SIL"
# )
# print(f"The shape of the fact table is:\t {fact_table.shape}")
# print(f"The shape of the dimension table is:\t {dimension_table.shape}")

# %%
# df_f_fspd30_alphacredosil = fact_table.copy()
# df_d_fspd30_alphacredosil = dimension_table.copy()

# print(f"The shape of fact table and copied dataframe are:\t {fact_table.shape} - {df_f_fspd30_alphacredosil.shape}")
# print(f"The shape of dimension table and copied dataframe are:\t {dimension_table.shape} - {df_d_fspd30_alphacredosil.shape}")

# %% [markdown]
# ###### FSTPD30

# %% [markdown]
# ###### Test

# %%
# sq = f"""with modelname as
#   (  SELECT
#     mmrd.customerId,mmrd.digitalLoanAccountId,prediction,start_time,end_time,modelDisplayName,modelVersionId,
#   case when trenchCategory is null then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
#     when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
#     else 'Trench1' end)
#      when trenchCategory = '' then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
#     when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
#     else 'Trench 1' end)
#     else trenchCategory end  as trenchCategory,
#     REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature,
#         deviceOs osType,
#     FROM prj-prod-dataplatform.audit_balance.ml_model_run_details mmrd
#   left join prj-prod-dataplatform.risk_credit_mis.model_loan_score_mart mt on mt.digitalLoanAccountId = mmrd.digitalLoanAccountId
#   WHERE modelDisplayName in  ('Alpha - StackingModel', 'alpha_stack_model_sil')
#   -- and modelVersionId = 'v1'
#       ),
#   deliquency as
# (select loanAccountNumber,
# case when obs_min_inst_def0 >= 1 and min_inst_def0 = 1 then 1 else 0 end deffpd0,
# case when obs_min_inst_def10 >=1 and min_inst_def10 =1 then 1 else 0 end deffpd10,
# case when obs_min_inst_def30 >=1 and min_inst_def30 =1 then 1 else 0 end deffpd30,
# case when obs_min_inst_def30 >=2 and min_inst_def30 in (1,2) then 1 else 0 end deffspd30,
# case when obs_min_inst_def30 >=3 and min_inst_def30 in (1,2,3) then 1 else 0 end deffstpd30,
# case when obs_min_inst_def0 >= 1 then 1 else 0 end flg_mature_fpd0,
# case when obs_min_inst_def10 >=1 then 1 else 0 end flg_mature_fpd10,
# case when obs_min_inst_def30 >=1 then 1 else 0 end flg_mature_fpd30,
# case when obs_min_inst_def30 >=2 then 1 else 0 end flg_mature_fspd_30,
# case when obs_min_inst_def30 >=3 then 1 else 0 end flg_mature_fstpd_30
# from prj-prod-dataplatform.risk_credit_mis.loan_deliquency_data),
# base as
#   (select distinct r.customerId,
#   r.digitalLoanAccountId,
#   loanmaster.loanAccountNumber,
#   r.modelDisplayName,
#   coalesce(cast(JSON_VALUE(SAFE.PARSE_JSON(CAST(calcFeature AS STRING)), '$.credo_score')AS FLOAT64),
#   cast(JSON_VALUE(SAFE.PARSE_JSON(CAST(calcFeature AS STRING)), '$.s_credo_score')AS FLOAT64)) AS credo_score,
#   calcFeature,
#   coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
#   date(loanmaster.disbursementDateTime) disbursementdate,
#   format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month,
#   'Prod' Data_selection,
#   del.deffstpd30,
#   del.flg_mature_fstpd_30,
#   loanmaster.new_loan_type,
#   modelVersionId, trenchCategory,
#     case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
#     when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile'
#     when loanmaster.loantype='BNPL' and store_type =3 then 'Mall'
#     when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
#     else 'not applicable' end as loan_product_type,
#         coalesce((case when lower(r.osType) like '%andro%' then 'android'
#                   when lower(r.osType) like '%os%' then 'ios' else lower(r.osType) end),
#             (case when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%andro%' then 'android'
#                   when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%os%' then 'ios'
#                   when lower(loanmaster.deviceType) like '%andro%' then 'android'
#                   else 'ios' end)
#             ) as osType
#   from modelname r
#   left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
#   left join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
#    left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
#   left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
#  qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
#   where
#   loanmaster.flagDisbursement = 1
#   and loanmaster.disbursementDateTime is not null
#   and del.flg_mature_fstpd_30 = 1
#   )
#   select *
#   from base
#   where credo_score is not null
#  qualify row_number() over(partition by digitalLoanAccountId, modelVersionId order by appln_submit_datetime) = 1
#    ;
# """
# dfd = client.query(sq).to_dataframe()
# # dfd = dfd.drop_duplicates(keep='first')
# print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
# dfd.head()

# df1 = dfd.copy()

# # ### Train

# sq = """
# with modelname as
#   (  SELECT
#     mmrd.customerId,mmrd.digitalLoanAccountId,prediction,start_time,end_time,modelDisplayName,modelVersionId,
#   case when trenchCategory is null then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
#     when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
#     else 'Trench1' end)
#      when trenchCategory = '' then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
#     when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
#     else 'Trench 1' end)
#     else trenchCategory end  as trenchCategory,
#     REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature,
#     Data_selection,
#         deviceOs osType,


#     FROM prj-prod-dataplatform.dap_ds_poweruser_playground.ml_training_model_run_details_20260116 mmrd
#   left join prj-prod-dataplatform.risk_credit_mis.model_loan_score_mart mt on mt.digitalLoanAccountId = mmrd.digitalLoanAccountId
#   WHERE modelDisplayName in  ('Alpha - StackingModel', 'alpha_stack_model_sil')
#   -- and modelVersionId = 'v1'
#       ),
#   deliquency as
# (select loanAccountNumber,
# case when obs_min_inst_def0 >= 1 and min_inst_def0 = 1 then 1 else 0 end deffpd0,
# case when obs_min_inst_def10 >=1 and min_inst_def10 =1 then 1 else 0 end deffpd10,
# case when obs_min_inst_def30 >=1 and min_inst_def30 =1 then 1 else 0 end deffpd30,
# case when obs_min_inst_def30 >=2 and min_inst_def30 in (1,2) then 1 else 0 end deffspd30,
# case when obs_min_inst_def30 >=3 and min_inst_def30 in (1,2,3) then 1 else 0 end deffstpd30,
# case when obs_min_inst_def0 >= 1 then 1 else 0 end flg_mature_fpd0,
# case when obs_min_inst_def10 >=1 then 1 else 0 end flg_mature_fpd10,
# case when obs_min_inst_def30 >=1 then 1 else 0 end flg_mature_fpd30,
# case when obs_min_inst_def30 >=2 then 1 else 0 end flg_mature_fspd_30,
# case when obs_min_inst_def30 >=3 then 1 else 0 end flg_mature_fstpd_30
# from prj-prod-dataplatform.risk_credit_mis.loan_deliquency_data),
# base as
#   (select distinct r.customerId,
#   r.digitalLoanAccountId,
#   loanmaster.loanAccountNumber,
#   r.modelDisplayName,
#   coalesce(cast(JSON_VALUE(SAFE.PARSE_JSON(CAST(calcFeature AS STRING)), '$.credo_score')AS FLOAT64),
#   cast(JSON_VALUE(SAFE.PARSE_JSON(CAST(calcFeature AS STRING)), '$.s_credo_score')AS FLOAT64)) AS credo_score,
#   calcFeature,
#   coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
#   date(loanmaster.disbursementDateTime) disbursementdate,
#   format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month,
#   Data_selection,
#   del.deffstpd30,
#   del.flg_mature_fstpd_30,
#   loanmaster.new_loan_type,
#   modelVersionId, trenchCategory,
#     case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
#     when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile'
#     when loanmaster.loantype='BNPL' and store_type =3 then 'Mall'
#     when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
#     else 'not applicable' end as loan_product_type,

#     coalesce((case when lower(r.osType) like '%andro%' then 'android'
#                   when lower(r.osType) like '%os%' then 'ios' else lower(r.osType) end),
#             (case when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%andro%' then 'android'
#                   when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%os%' then 'ios'
#                   when lower(loanmaster.deviceType) like '%andro%' then 'android'
#                   else 'ios' end)
#             ) as osType
#   from modelname r
#   left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
#   left join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
#    left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
#   left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
#  qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
#   where
#   loanmaster.flagDisbursement = 1
#   and loanmaster.disbursementDateTime is not null
#   and del.flg_mature_fstpd_30 = 1
#   )
#   select *
#   from base
#   where credo_score is not null
#  qualify row_number() over(partition by digitalLoanAccountId, modelVersionId order by appln_submit_datetime) = 1
#    ;
#   """
# dfd = client.query(sq).to_dataframe()
# # dfd = dfd.drop_duplicates(keep='first')
# print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
# dfd.head()

# df2 = dfd.copy()

# # %% [markdown]
# # ### Concat

# # 1) Get all IDs present in Train
# train_ids = set(df2["digitalLoanAccountId"])

# # 2) Keep only Test rows whose ID is NOT in Train
# df1_no_dupes = df1[~df1["digitalLoanAccountId"].isin(train_ids)]

# # 3) Concatenate
# df_concat = pd.concat([df1_no_dupes, df2], ignore_index=True)

# print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")
# df_concat.head()


# df_concat["credo_score"] = pd.to_numeric(df_concat["credo_score"], errors="coerce")


# fact_table, dimension_table = calculate_periodic_gini_prod_ver_trench_dimfact(
#     df_concat,
#     "credo_score",
#     "deffstpd30",
#     "FSTPD30",
#     data_selection_column="Data_selection",
#     model_version_column="modelVersionId",
#     trench_column="trenchCategory",
#     loan_type_column="new_loan_type",
#     loan_product_type_column="loan_product_type",
#     ostype_column="osType",
#     account_id_column="digitalLoanAccountId",
# )

# fact_table, dimension_table = update_tables(
#     fact_table, dimension_table, model_name="alpha_stack_credo_score_sil", product="SIL"
# )
# print(f"The shape of the fact table is:\t {fact_table.shape}")
# print(f"The shape of the dimension table is:\t {dimension_table.shape}")

# df_f_fstpd30_alphacredosil = fact_table.copy()
# df_d_fstpd30_alphacredosil = dimension_table.copy()

# print(f"The shape of fact table and copied dataframe are:\t {fact_table.shape} - {df_f_fstpd30_alphacredosil.shape}")
# print(f"The shape of dimension table and copied dataframe are:\t {dimension_table.shape} - {df_d_fstpd30_alphacredosil.shape}")


# %% [markdown]
# #### alpha_stack_model_sil

# %%
# ## alpha_stack_model_sil

# ### FPD0
# #### Test

# %%
sq = """
with modelname as
  (  SELECT
    mmrd.customerId,mmrd.digitalLoanAccountId,prediction Sil_Alpha_Stack_score,start_time,end_time,modelDisplayName,modelVersionId,
  case when trenchCategory is null then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench1' end)
     when trenchCategory = '' then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench 1' end)
    else trenchCategory end  as trenchCategory,
    REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature,
     deviceOs osType,
  FROM prj-prod-dataplatform.audit_balance.ml_model_run_details mmrd
  left join prj-prod-dataplatform.risk_credit_mis.model_loan_score_mart mt on mt.digitalLoanAccountId = mmrd.digitalLoanAccountId
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
  'Prod' Data_selection,
  deffpd0,
  flg_mature_fpd0,
  loanmaster.new_loan_type,
  modelVersionId, trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile'
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall'
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
      coalesce((case when lower(r.osType) like '%andro%' then 'android'
                  when lower(r.osType) like '%os%' then 'ios' else lower(r.osType) end),
            (case when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%andro%' then 'android'
                  when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%os%' then 'ios'
                  when lower(loanmaster.deviceType) like '%andro%' then 'android'
                  else 'ios' end)
            ) as osType
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
  qualify row_number() over(partition by digitalLoanAccountId, modelVersionId order by appln_submit_datetime) = 1
  ;
"""
dfd = client.query(sq).to_dataframe()
# dfd = dfd.drop_duplicates(keep='first')
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()


# %%
df1 = dfd.copy()

# #### Train

# %%
sq = """
  with modelname as
  (  SELECT
    mmrd.customerId,mmrd.digitalLoanAccountId,prediction Sil_Alpha_Stack_score,start_time,end_time,modelDisplayName,modelVersionId,
  case when trenchCategory is null then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench1' end)
     when trenchCategory = '' then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench 1' end)
    else trenchCategory end  as trenchCategory,
    REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature,
    Data_selection,
    deviceOs osType,
  FROM prj-prod-dataplatform.dap_ds_poweruser_playground.ml_training_model_run_details_20260116 mmrd
  left join prj-prod-dataplatform.risk_credit_mis.model_loan_score_mart mt on mt.digitalLoanAccountId = mmrd.digitalLoanAccountId
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
  Data_selection,
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
        coalesce((case when lower(r.osType) like '%andro%' then 'android'
                  when lower(r.osType) like '%os%' then 'ios' else lower(r.osType) end),
            (case when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%andro%' then 'android'
                  when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%os%' then 'ios'
                  when lower(loanmaster.deviceType) like '%andro%' then 'android'
                  else 'ios' end)
            ) as osType
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
  select * from base
  qualify row_number() over(partition by digitalLoanAccountId, modelVersionId order by appln_submit_datetime) = 1
  ;
  """
dfd = client.query(sq).to_dataframe()
# dfd = dfd.drop_duplicates(keep='first')
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()

# %%
df2 = dfd.copy()


# 1) Get all IDs present in Train
train_ids = set(df2["digitalLoanAccountId"])

# 2) Keep only Test rows whose ID is NOT in Train
df1_no_dupes = df1[~df1["digitalLoanAccountId"].isin(train_ids)]

# 3) Concatenate
df_concat = pd.concat([df1_no_dupes, df2], ignore_index=True)

print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")
df_concat.head()


# #### Making Sure the Score is Numeric

# %%
df_concat["Sil_Alpha_Stack_score"] = pd.to_numeric(
    df_concat["Sil_Alpha_Stack_score"], errors="coerce"
)

# #### Calculating the Gini

# %%
fact_table, dimension_table = calculate_periodic_gini_prod_ver_trench_dimfact(
    df_concat,
    "Sil_Alpha_Stack_score",
    "deffpd0",
    "FPD0",
    data_selection_column="Data_selection",  # Add this
    model_version_column="modelVersionId",
    trench_column="trenchCategory",
    loan_type_column="new_loan_type",
    loan_product_type_column="loan_product_type",
    ostype_column="osType",
    account_id_column="digitalLoanAccountId",
)

# #### Updating Fact and Dimension Table

# %%
fact_table, dimension_table = update_tables(
    fact_table, dimension_table, model_name="alpha_stack_model_sil", product="SIL"
)
print(f"The shape of the fact table is:\t {fact_table.shape}")
print(f"The shape of the dimension table is:\t {dimension_table.shape}")

df_f_fpd0_alphastacksil = fact_table.copy()
df_d_fpd0_alphastacksil = dimension_table.copy()

print(
    f"The shape of fact table and copied dataframe are:\t {fact_table.shape} - {df_f_fpd0_alphastacksil.shape}"
)
print(
    f"The shape of dimension table and copied dataframe are:\t {dimension_table.shape} - {df_d_fpd0_alphastacksil.shape}"
)

facttable_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_alphastacksil"
dimtable_id = (
    "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_alphastacksil"
)

# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_TRUNCATE",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_f_fpd0_alphastacksil, facttable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_TRUNCATE",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_d_fpd0_alphastacksil, dimtable_id, job_config=job_config
)
job.result()  # Wait for the job to complete


# ### FPD10


# #### Test

# %%
sq = """
with modelname as
  ( SELECT
    mmrd.customerId,mmrd.digitalLoanAccountId,prediction Sil_Alpha_Stack_score,start_time,end_time,modelDisplayName,modelVersionId,
   case when trenchCategory is null then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench1' end)
     when trenchCategory = '' then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench 1' end)
    else trenchCategory end  as trenchCategory,
    deviceOs osType,
  FROM prj-prod-dataplatform.audit_balance.ml_model_run_details mmrd
  left join prj-prod-dataplatform.risk_credit_mis.model_loan_score_mart mt on mt.digitalLoanAccountId = mmrd.digitalLoanAccountId
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
  'Prod' Data_selection,
  del.deffpd10,
  del.flg_mature_fpd10,
  loanmaster.new_loan_type,
  modelVersionId, trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile'
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall'
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
       coalesce((case when lower(r.osType) like '%andro%' then 'android'
                  when lower(r.osType) like '%os%' then 'ios' else lower(r.osType) end),
            (case when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%andro%' then 'android'
                  when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%os%' then 'ios'
                  when lower(loanmaster.deviceType) like '%andro%' then 'android'
                  else 'ios' end)
            ) as osType
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
  qualify row_number() over(partition by digitalLoanAccountId, modelVersionId order by appln_submit_datetime) = 1
  ;
"""
dfd = client.query(sq).to_dataframe()
# dfd = dfd.drop_duplicates(keep='first')
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()


# %%
df1 = dfd.copy()

# #### Train

# %%
sq = """
  with modelname as
  (  SELECT
    mmrd.customerId,mmrd.digitalLoanAccountId,prediction Sil_Alpha_Stack_score,start_time,end_time,modelDisplayName,modelVersionId,
    case when trenchCategory is null then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench1' end)
     when trenchCategory = '' then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench 1' end)
    else trenchCategory end  as trenchCategory,
    REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature,
    Data_selection,
    deviceOs osType,
  FROM prj-prod-dataplatform.dap_ds_poweruser_playground.ml_training_model_run_details_20260116 mmrd
  left join prj-prod-dataplatform.risk_credit_mis.model_loan_score_mart mt on mt.digitalLoanAccountId = mmrd.digitalLoanAccountId
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
  Data_selection,
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
        coalesce((case when lower(r.osType) like '%andro%' then 'android'
                  when lower(r.osType) like '%os%' then 'ios' else lower(r.osType) end),
            (case when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%andro%' then 'android'
                  when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%os%' then 'ios'
                  when lower(loanmaster.deviceType) like '%andro%' then 'android'
                  else 'ios' end)
            ) as osType
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
  select * from base
  qualify row_number() over(partition by digitalLoanAccountId, modelVersionId order by appln_submit_datetime) = 1
  ;
  """
dfd = client.query(sq).to_dataframe()
# dfd = dfd.drop_duplicates(keep='first')
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()

# %%
df2 = dfd.copy()


# 1) Get all IDs present in Train
train_ids = set(df2["digitalLoanAccountId"])

# 2) Keep only Test rows whose ID is NOT in Train
df1_no_dupes = df1[~df1["digitalLoanAccountId"].isin(train_ids)]

# 3) Concatenate
df_concat = pd.concat([df1_no_dupes, df2], ignore_index=True)

print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")
df_concat.head()


# #### Making Sure the Score is Numeric

# %%
df_concat["Sil_Alpha_Stack_score"] = pd.to_numeric(
    df_concat["Sil_Alpha_Stack_score"], errors="coerce"
)


# #### Calculating the Gini

# %%
fact_table, dimension_table = calculate_periodic_gini_prod_ver_trench_dimfact(
    df_concat,
    "Sil_Alpha_Stack_score",
    "deffpd10",
    "FPD10",
    data_selection_column="Data_selection",  # Add this
    model_version_column="modelVersionId",
    trench_column="trenchCategory",
    loan_type_column="new_loan_type",
    loan_product_type_column="loan_product_type",
    ostype_column="osType",
    account_id_column="digitalLoanAccountId",
)

# #### Updating Fact and Dimension Table

# %%
fact_table, dimension_table = update_tables(
    fact_table, dimension_table, model_name="alpha_stack_model_sil", product="SIL"
)


df_f_fpd10_alphastacksil = fact_table.copy()
df_d_fpd10_alphastacksil = dimension_table.copy()

print(
    f"The shape of fact table and copied dataframe are:\t {fact_table.shape} - {df_f_fpd10_alphastacksil.shape}"
)
print(
    f"The shape of dimension table and copied dataframe are:\t {dimension_table.shape} - {df_d_fpd10_alphastacksil.shape}"
)

# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_f_fpd10_alphastacksil, facttable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_d_fpd10_alphastacksil, dimtable_id, job_config=job_config
)
job.result()  # Wait for the job to complete


# ### FPD30


# #### Test

# %%
sq = """
with modelname as
  (SELECT
    mmrd.customerId,mmrd.digitalLoanAccountId,prediction Sil_Alpha_Stack_score,start_time,end_time,modelDisplayName,modelVersionId,
    case when trenchCategory is null then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench1' end)
     when trenchCategory = '' then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench 1' end)
    else trenchCategory end  as trenchCategory,
    REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature,
      deviceOs osType,
  FROM prj-prod-dataplatform.audit_balance.ml_model_run_details mmrd
  left join prj-prod-dataplatform.risk_credit_mis.model_loan_score_mart mt on mt.digitalLoanAccountId = mmrd.digitalLoanAccountId
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
  'Prod' Data_selection,
  del.deffpd30,
  del.flg_mature_fpd30,
  loanmaster.new_loan_type,
  modelVersionId, trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile'
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall'
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
       coalesce((case when lower(r.osType) like '%andro%' then 'android'
                  when lower(r.osType) like '%os%' then 'ios' else lower(r.osType) end),
            (case when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%andro%' then 'android'
                  when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%os%' then 'ios'
                  when lower(loanmaster.deviceType) like '%andro%' then 'android'
                  else 'ios' end)
            ) as osType
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
  qualify row_number() over(partition by digitalLoanAccountId, modelVersionId order by appln_submit_datetime) = 1
  ;
"""
dfd = client.query(sq).to_dataframe()
# dfd = dfd.drop_duplicates(keep='first')
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()


# %%
df1 = dfd.copy()

# #### Train

# %%
sq = """
  with modelname as
  (
  SELECT
    mmrd.customerId,mmrd.digitalLoanAccountId,prediction Sil_Alpha_Stack_score,start_time,end_time,modelDisplayName,modelVersionId,
  case when trenchCategory is null then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench1' end)
     when trenchCategory = '' then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench 1' end)
    else trenchCategory end  as trenchCategory,
    REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature,
    Data_selection,
       deviceOs osType,
  FROM prj-prod-dataplatform.dap_ds_poweruser_playground.ml_training_model_run_details_20260116 mmrd
  left join prj-prod-dataplatform.risk_credit_mis.model_loan_score_mart mt on mt.digitalLoanAccountId = mmrd.digitalLoanAccountId
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
  Data_selection,
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
     coalesce((case when lower(r.osType) like '%andro%' then 'android'
                  when lower(r.osType) like '%os%' then 'ios' else lower(r.osType) end),
            (case when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%andro%' then 'android'
                  when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%os%' then 'ios'
                  when lower(loanmaster.deviceType) like '%andro%' then 'android'
                  else 'ios' end)
            ) as osType
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
  select * from base
  qualify row_number() over(partition by digitalLoanAccountId, modelVersionId order by appln_submit_datetime) = 1
  ;
  """
dfd = client.query(sq).to_dataframe()
# dfd = dfd.drop_duplicates(keep='first')
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()

# %%
df2 = dfd.copy()

# #### Concatenate both Test and Train Datasets


# 1) Get all IDs present in Train
train_ids = set(df2["digitalLoanAccountId"])

# 2) Keep only Test rows whose ID is NOT in Train
df1_no_dupes = df1[~df1["digitalLoanAccountId"].isin(train_ids)]

# 3) Concatenate
df_concat = pd.concat([df1_no_dupes, df2], ignore_index=True)

print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")
df_concat.head()


# #### Making Sure the Score is Numeric

# %%
df_concat["Sil_Alpha_Stack_score"] = pd.to_numeric(
    df_concat["Sil_Alpha_Stack_score"], errors="coerce"
)


# #### Calculating the Gini

# %%
fact_table, dimension_table = calculate_periodic_gini_prod_ver_trench_dimfact(
    df_concat,
    "Sil_Alpha_Stack_score",
    "deffpd30",
    "FPD30",
    data_selection_column="Data_selection",  # Add this
    model_version_column="modelVersionId",
    trench_column="trenchCategory",
    loan_type_column="new_loan_type",
    loan_product_type_column="loan_product_type",
    ostype_column="osType",
    account_id_column="digitalLoanAccountId",
)


# #### Updating the Fact and Dimension Table

# %%
fact_table, dimension_table = update_tables(
    fact_table, dimension_table, model_name="alpha_stack_model_sil", product="SIL"
)


df_f_fpd30_alphastacksil = fact_table.copy()
df_d_fpd30_alphastacksil = dimension_table.copy()

print(
    f"The shape of fact table and copied dataframe are:\t {fact_table.shape} - {df_f_fpd30_alphastacksil.shape}"
)
print(
    f"The shape of dimension table and copied dataframe are:\t {dimension_table.shape} - {df_d_fpd30_alphastacksil.shape}"
)

# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_f_fpd30_alphastacksil, facttable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_d_fpd30_alphastacksil, dimtable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# ### FSPD30


# #### Test

# %%
sq = """
with modelname as
  (  SELECT
    mmrd.customerId,mmrd.digitalLoanAccountId,prediction Sil_Alpha_Stack_score,start_time,end_time,modelDisplayName,modelVersionId,
  case when trenchCategory is null then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench1' end)
     when trenchCategory = '' then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench 1' end)
    else trenchCategory end  as trenchCategory,
    REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature,
    deviceOs osType,
  FROM prj-prod-dataplatform.audit_balance.ml_model_run_details mmrd
  left join prj-prod-dataplatform.risk_credit_mis.model_loan_score_mart mt on mt.digitalLoanAccountId = mmrd.digitalLoanAccountId
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
  'Prod' Data_selection,
  del.deffspd30,
  del.flg_mature_fspd_30,
  loanmaster.new_loan_type,
  modelVersionId, trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile'
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall'
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
   coalesce((case when lower(r.osType) like '%andro%' then 'android'
                  when lower(r.osType) like '%os%' then 'ios' else lower(r.osType) end),
            (case when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%andro%' then 'android'
                  when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%os%' then 'ios'
                  when lower(loanmaster.deviceType) like '%andro%' then 'android'
                  else 'ios' end)
            ) as osType
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
  qualify row_number() over(partition by digitalLoanAccountId, modelVersionId order by appln_submit_datetime) = 1
  ;
"""
dfd = client.query(sq).to_dataframe()
# dfd = dfd.drop_duplicates(keep='first')
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()


# %%
df1 = dfd.copy()


# #### Train

# %%
sq = """
  with modelname as
  (  SELECT
    mmrd.customerId,mmrd.digitalLoanAccountId,prediction Sil_Alpha_Stack_score,start_time,end_time,modelDisplayName,modelVersionId,
   case when trenchCategory is null then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench1' end)
     when trenchCategory = '' then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench 1' end)
    else trenchCategory end  as trenchCategory,
    REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature,
    Data_selection,
        deviceOs osType,
  FROM prj-prod-dataplatform.dap_ds_poweruser_playground.ml_training_model_run_details_20260116 mmrd
  left join prj-prod-dataplatform.risk_credit_mis.model_loan_score_mart mt on mt.digitalLoanAccountId = mmrd.digitalLoanAccountId
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
  Data_selection,
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
    coalesce((case when lower(r.osType) like '%andro%' then 'android'
                  when lower(r.osType) like '%os%' then 'ios' else lower(r.osType) end),
            (case when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%andro%' then 'android'
                  when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%os%' then 'ios'
                  when lower(loanmaster.deviceType) like '%andro%' then 'android'
                  else 'ios' end)
            ) as osType
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
  select * from base
  qualify row_number() over(partition by digitalLoanAccountId, modelVersionId order by appln_submit_datetime) = 1
  ;
  """
dfd = client.query(sq).to_dataframe()
# dfd = dfd.drop_duplicates(keep='first')
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()

# %%
df2 = dfd.copy()


# #### Concatenate both Test and Train Datasets


# 1) Get all IDs present in Train
train_ids = set(df2["digitalLoanAccountId"])

# 2) Keep only Test rows whose ID is NOT in Train
df1_no_dupes = df1[~df1["digitalLoanAccountId"].isin(train_ids)]

# 3) Concatenate
df_concat = pd.concat([df1_no_dupes, df2], ignore_index=True)

print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")
df_concat.head()


# #### Making sure the score column in numeric

# %%
df_concat["Sil_Alpha_Stack_score"] = pd.to_numeric(
    df_concat["Sil_Alpha_Stack_score"], errors="coerce"
)


# #### Calculating the Gini

# %%
fact_table, dimension_table = calculate_periodic_gini_prod_ver_trench_dimfact(
    df_concat,
    "Sil_Alpha_Stack_score",
    "deffspd30",
    "FSPD30",
    data_selection_column="Data_selection",  # Add this
    model_version_column="modelVersionId",
    trench_column="trenchCategory",
    loan_type_column="new_loan_type",
    loan_product_type_column="loan_product_type",
    ostype_column="osType",
    account_id_column="digitalLoanAccountId",
)


# #### Updating the Fact and Dimension Table

# %%
fact_table, dimension_table = update_tables(
    fact_table, dimension_table, model_name="alpha_stack_model_sil", product="SIL"
)


df_f_fspd30_alphastacksil = fact_table.copy()
df_d_fspd30_alphastacksil = dimension_table.copy()

print(
    f"The shape of fact table and copied dataframe are:\t {fact_table.shape} - {df_f_fspd30_alphastacksil.shape}"
)
print(
    f"The shape of dimension table and copied dataframe are:\t {dimension_table.shape} - {df_d_fspd30_alphastacksil.shape}"
)

# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_f_fspd30_alphastacksil, facttable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_d_fspd30_alphastacksil, dimtable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# ### FSTPD30

# #### Test

# %%
sq = """
with modelname as
  (  SELECT
    mmrd.customerId,mmrd.digitalLoanAccountId,prediction Sil_Alpha_Stack_score,start_time,end_time,modelDisplayName,modelVersionId,
  case when trenchCategory is null then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench1' end)
     when trenchCategory = '' then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench 1' end)
    else trenchCategory end  as trenchCategory,
    REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature,
    deviceOs osType,
  FROM prj-prod-dataplatform.audit_balance.ml_model_run_details mmrd
  left join prj-prod-dataplatform.risk_credit_mis.model_loan_score_mart mt on mt.digitalLoanAccountId = mmrd.digitalLoanAccountId
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
  'Prod' Data_selection,
  del.deffstpd30,
  del.flg_mature_fstpd_30,
  loanmaster.new_loan_type,
  modelVersionId, trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile'
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall'
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
        coalesce((case when lower(r.osType) like '%andro%' then 'android'
                  when lower(r.osType) like '%os%' then 'ios' else lower(r.osType) end),
            (case when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%andro%' then 'android'
                  when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%os%' then 'ios'
                  when lower(loanmaster.deviceType) like '%andro%' then 'android'
                  else 'ios' end)
            ) as osType
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
  qualify row_number() over(partition by digitalLoanAccountId, modelVersionId order by appln_submit_datetime) = 1
  ;
"""
dfd = client.query(sq).to_dataframe()
# dfd = dfd.drop_duplicates(keep='first')
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()


# %%
df1 = dfd.copy()


# #### Train

# %%
sq = """
  with modelname as
  (  SELECT
    mmrd.customerId,mmrd.digitalLoanAccountId,prediction Sil_Alpha_Stack_score,start_time,end_time,modelDisplayName,modelVersionId,
   case when trenchCategory is null then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench1' end)
     when trenchCategory = '' then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench 1' end)
    else trenchCategory end  as trenchCategory,
    REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature,
    Data_selection,
        deviceOs osType,
  FROM prj-prod-dataplatform.dap_ds_poweruser_playground.ml_training_model_run_details_20260116 mmrd
  left join prj-prod-dataplatform.risk_credit_mis.model_loan_score_mart mt on mt.digitalLoanAccountId = mmrd.digitalLoanAccountId
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
  Data_selection,
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
        coalesce((case when lower(r.osType) like '%andro%' then 'android'
                  when lower(r.osType) like '%os%' then 'ios' else lower(r.osType) end),
            (case when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%andro%' then 'android'
                  when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%os%' then 'ios'
                  when lower(loanmaster.deviceType) like '%andro%' then 'android'
                  else 'ios' end)
            ) as osType
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
  select * from base
  qualify row_number() over(partition by digitalLoanAccountId, modelVersionId order by appln_submit_datetime) = 1
  ;
  """
dfd = client.query(sq).to_dataframe()
# dfd = dfd.drop_duplicates(keep='first')
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()

# %%
df2 = dfd.copy()


# #### Concatenate both Test and Train Datasets


# 1) Get all IDs present in Train
train_ids = set(df2["digitalLoanAccountId"])

# 2) Keep only Test rows whose ID is NOT in Train
df1_no_dupes = df1[~df1["digitalLoanAccountId"].isin(train_ids)]

# 3) Concatenate
df_concat = pd.concat([df1_no_dupes, df2], ignore_index=True)

print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")
df_concat.head()


# #### Making sure the score column is numeric

# %%
df_concat["Sil_Alpha_Stack_score"] = pd.to_numeric(
    df_concat["Sil_Alpha_Stack_score"], errors="coerce"
)


# #### Calculating the Gini

# %%
fact_table, dimension_table = calculate_periodic_gini_prod_ver_trench_dimfact(
    df_concat,
    "Sil_Alpha_Stack_score",
    "deffstpd30",
    "FSTPD30",
    data_selection_column="Data_selection",  # Add this
    model_version_column="modelVersionId",
    trench_column="trenchCategory",
    loan_type_column="new_loan_type",
    loan_product_type_column="loan_product_type",
    ostype_column="osType",
    account_id_column="digitalLoanAccountId",
)


# #### Updating the Fact and Dimension Tables

# %%
fact_table, dimension_table = update_tables(
    fact_table, dimension_table, model_name="alpha_stack_model_sil", product="SIL"
)
print(f"The shape of the fact table is:\t {fact_table.shape}")
print(f"The shape of the dimension table is:\t {dimension_table.shape}")

df_f_fstpd30_alphastacksil = fact_table.copy()
df_d_fstpd30_alphastacksil = dimension_table.copy()

print(
    f"The shape of fact table and copied dataframe are:\t {fact_table.shape} - {df_f_fstpd30_alphastacksil.shape}"
)
print(
    f"The shape of dimension table and copied dataframe are:\t {dimension_table.shape} - {df_d_fstpd30_alphastacksil.shape}"
)

# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_f_fstpd30_alphastacksil, facttable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_d_fstpd30_alphastacksil, dimtable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

factallaplhastacksil = pd.concat(
    [
        df_f_fpd0_alphastacksil,
        df_f_fpd10_alphastacksil,
        df_f_fpd30_alphastacksil,
        df_f_fspd30_alphastacksil,
        df_f_fstpd30_alphastacksil,
    ],
    ignore_index=True,
)
dimallaplhastacksil = pd.concat(
    [
        df_d_fpd0_alphastacksil,
        df_d_fpd10_alphastacksil,
        df_d_fpd30_alphastacksil,
        df_d_fspd30_alphastacksil,
        df_d_fstpd30_alphastacksil,
    ],
    ignore_index=True,
)

print(f"alpha_stack_model_sil model Gini Calculation and Upload Completed!")


# %% [markdown]
# #### Beta Sil App Score

# %%
# ### FPD0

# #### Test

# %%
sq = """
WITH cleaned AS (
    SELECT
    mmrd.customerId,mmrd.digitalLoanAccountId,prediction,start_time,end_time,modelDisplayName,modelVersionId,
   case when trenchCategory is null then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench1' end)
     when trenchCategory = '' then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench 1' end)
    else trenchCategory end  as trenchCategory,
    REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature,
    REPLACE(REPLACE(prediction, "'", '"'), "None", "null") AS prediction_clean,
    deviceOs osType,
   FROM prj-prod-dataplatform.audit_balance.ml_model_run_details mmrd
  left join prj-prod-dataplatform.risk_credit_mis.model_loan_score_mart mt on mt.digitalLoanAccountId = mmrd.digitalLoanAccountId
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
  osType,
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
  'Prod' Data_selection,
  deffpd0,
  flg_mature_fpd0,
  loanmaster.new_loan_type,
  modelVersionId, trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile'
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall'
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
        coalesce((case when lower(r.osType) like '%andro%' then 'android'
                  when lower(r.osType) like '%os%' then 'ios' else lower(r.osType) end),
            (case when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%andro%' then 'android'
                  when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%os%' then 'ios'
                  when lower(loanmaster.deviceType) like '%andro%' then 'android'
                  else 'ios' end)
            ) as osType
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
  qualify row_number() over(partition by digitalLoanAccountId, modelVersionId order by appln_submit_datetime) = 1
  ;
"""
dfd = client.query(sq).to_dataframe()
# dfd = dfd.drop_duplicates(keep='first')
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()


# %%
df1 = dfd.copy()


# #### Train

# %%
sq = """
WITH cleaned AS (
  SELECT
    mmrd.customerId,mmrd.digitalLoanAccountId,prediction,start_time,end_time,modelDisplayName,modelVersionId,
  case when trenchCategory is null then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench1' end)
     when trenchCategory = '' then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench 1' end)
    else trenchCategory end  as trenchCategory,
    REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature,
    REPLACE(REPLACE(cast(prediction as string), "'", '"'), "None", "null") AS prediction_clean,
    Data_selection,
    deviceOs osType,
  FROM prj-prod-dataplatform.dap_ds_poweruser_playground.ml_training_model_run_details_20260116 mmrd
    left join prj-prod-dataplatform.risk_credit_mis.model_loan_score_mart mt on mt.digitalLoanAccountId = mmrd.digitalLoanAccountId
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
  trenchCategory,
  Data_selection,
  osType,
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
  Data_selection,
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
        coalesce((case when lower(r.osType) like '%andro%' then 'android'
                  when lower(r.osType) like '%os%' then 'ios' else lower(r.osType) end),
            (case when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%andro%' then 'android'
                  when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%os%' then 'ios'
                  when lower(loanmaster.deviceType) like '%andro%' then 'android'
                  else 'ios' end)
            ) as osType
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
  select * from base
  qualify row_number() over(partition by digitalLoanAccountId, modelVersionId order by appln_submit_datetime) = 1
  ;
  """
dfd = client.query(sq).to_dataframe()
# dfd = dfd.drop_duplicates(keep='first')
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()

# %%
df2 = dfd.copy()


# #### Concatenate both Test and Train Datasets


# 1) Get all IDs present in Train
train_ids = set(df2["digitalLoanAccountId"])

# 2) Keep only Test rows whose ID is NOT in Train
df1_no_dupes = df1[~df1["digitalLoanAccountId"].isin(train_ids)]

# 3) Concatenate
df_concat = pd.concat([df1_no_dupes, df2], ignore_index=True)

print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")
df_concat.head()


# #### Making Sure the Score is Numeric

# %%
df_concat["sil_beta_app_score"] = pd.to_numeric(
    df_concat["sil_beta_app_score"], errors="coerce"
)


# #### Calculating the Gini

# %%
fact_table, dimension_table = calculate_periodic_gini_prod_ver_trench_dimfact(
    df_concat,
    "sil_beta_app_score",
    "deffpd0",
    "FPD0",
    data_selection_column="Data_selection",
    model_version_column="modelVersionId",
    trench_column="trenchCategory",
    loan_type_column="new_loan_type",
    loan_product_type_column="loan_product_type",
    ostype_column="osType",
    account_id_column="digitalLoanAccountId",
)


# #### Updating Fact and Dimension Table

# %%
fact_table, dimension_table = update_tables(
    fact_table, dimension_table, model_name="apps_score_model_sil", product="SIL"
)
print(f"The shape of the fact table is:\t {fact_table.shape}")
print(f"The shape of the dimension table is:\t {dimension_table.shape}")

df_f_fpd0_appscoresil = fact_table.copy()
df_d_fpd0_appscoresil = dimension_table.copy()

facttable_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_appscoresil"
dimtable_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_appscoresil"

# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_TRUNCATE",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_f_fpd0_appscoresil, facttable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_TRUNCATE",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_d_fpd0_appscoresil, dimtable_id, job_config=job_config
)
job.result()  # Wait for the job to complete


# ### FPD10


# #### Test

# %%
sq = """
WITH cleaned AS (
    SELECT
    mmrd.customerId,mmrd.digitalLoanAccountId,prediction,start_time,end_time,modelDisplayName,modelVersionId,
   case when trenchCategory is null then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench1' end)
     when trenchCategory = '' then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench 1' end)
    else trenchCategory end  as trenchCategory,
    REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature,
    REPLACE(REPLACE(prediction, "'", '"'), "None", "null") AS prediction_clean,
    deviceOs osType,
   FROM prj-prod-dataplatform.audit_balance.ml_model_run_details mmrd
  left join prj-prod-dataplatform.risk_credit_mis.model_loan_score_mart mt on mt.digitalLoanAccountId = mmrd.digitalLoanAccountId
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
  osType,
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
  'Prod' Data_selection,
  del.deffpd10,
  del.flg_mature_fpd10,
  loanmaster.new_loan_type,
  modelVersionId, trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile'
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall'
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
     coalesce((case when lower(r.osType) like '%andro%' then 'android'
                  when lower(r.osType) like '%os%' then 'ios' else lower(r.osType) end),
            (case when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%andro%' then 'android'
                  when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%os%' then 'ios'
                  when lower(loanmaster.deviceType) like '%andro%' then 'android'
                  else 'ios' end)
            ) as osType
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
  qualify row_number() over(partition by digitalLoanAccountId, modelVersionId order by appln_submit_datetime) = 1
  ;
"""
dfd = client.query(sq).to_dataframe()
# dfd = dfd.drop_duplicates(keep='first')
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()


# %%
df1 = dfd.copy()


# #### Train

# %%
sq = """
WITH cleaned AS (
  SELECT
    mmrd.customerId,mmrd.digitalLoanAccountId,prediction,start_time,end_time,modelDisplayName,modelVersionId,
  case when trenchCategory is null then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench1' end)
     when trenchCategory = '' then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench 1' end)
    else trenchCategory end  as trenchCategory,
    REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature,
    REPLACE(REPLACE(cast(prediction as string), "'", '"'), "None", "null") AS prediction_clean,
    Data_selection,
    deviceOs osType,
  FROM prj-prod-dataplatform.dap_ds_poweruser_playground.ml_training_model_run_details_20260116 mmrd
    left join prj-prod-dataplatform.risk_credit_mis.model_loan_score_mart mt on mt.digitalLoanAccountId = mmrd.digitalLoanAccountId
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
  trenchCategory,
  Data_selection,
  osType,
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
   Data_selection,
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
        coalesce((case when lower(r.osType) like '%andro%' then 'android'
                  when lower(r.osType) like '%os%' then 'ios' else lower(r.osType) end),
            (case when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%andro%' then 'android'
                  when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%os%' then 'ios'
                  when lower(loanmaster.deviceType) like '%andro%' then 'android'
                  else 'ios' end)
            ) as osType
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
  select * from base
  qualify row_number() over(partition by digitalLoanAccountId, modelVersionId order by appln_submit_datetime) = 1
  ;
  """
dfd = client.query(sq).to_dataframe()
# dfd = dfd.drop_duplicates(keep='first')
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()

# %%
df2 = dfd.copy()


# #### Concatenate both Test and Train Dataset


# 1) Get all IDs present in Train
train_ids = set(df2["digitalLoanAccountId"])

# 2) Keep only Test rows whose ID is NOT in Train
df1_no_dupes = df1[~df1["digitalLoanAccountId"].isin(train_ids)]

# 3) Concatenate
df_concat = pd.concat([df1_no_dupes, df2], ignore_index=True)

print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")
df_concat.head()


# #### Making Sure the Score is Numeric

# %%
df_concat["sil_beta_app_score"] = pd.to_numeric(
    df_concat["sil_beta_app_score"], errors="coerce"
)


# #### Calculating the Gini

# %%
fact_table, dimension_table = calculate_periodic_gini_prod_ver_trench_dimfact(
    df_concat,
    "sil_beta_app_score",
    "deffpd10",
    "FPD10",
    data_selection_column="Data_selection",
    model_version_column="modelVersionId",
    trench_column="trenchCategory",
    loan_type_column="new_loan_type",
    loan_product_type_column="loan_product_type",
    ostype_column="osType",
    account_id_column="digitalLoanAccountId",
)


# #### Updating Fact and Dimension Table

# %%
fact_table, dimension_table = update_tables(
    fact_table, dimension_table, model_name="apps_score_model_sil", product="SIL"
)

df_f_fpd10_appscoresil = fact_table.copy()
df_d_fpd10_appscoresil = dimension_table.copy()

# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_f_fpd10_appscoresil, facttable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_d_fpd10_appscoresil, dimtable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# ### FPD30


# #### Test

# %%
sq = """
WITH cleaned AS (
    SELECT
    mmrd.customerId,mmrd.digitalLoanAccountId,prediction,start_time,end_time,modelDisplayName,modelVersionId,
   case when trenchCategory is null then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench1' end)
     when trenchCategory = '' then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench 1' end)
    else trenchCategory end  as trenchCategory,
    REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature,
    REPLACE(REPLACE(prediction, "'", '"'), "None", "null") AS prediction_clean,
     deviceOs osType,
   FROM prj-prod-dataplatform.audit_balance.ml_model_run_details mmrd
  left join prj-prod-dataplatform.risk_credit_mis.model_loan_score_mart mt on mt.digitalLoanAccountId = mmrd.digitalLoanAccountId
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
   osType,
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
  'Prod' Data_selection,
  del.deffpd30,
  del.flg_mature_fpd30,
  loanmaster.new_loan_type,
  modelVersionId, trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile'
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall'
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
        coalesce((case when lower(r.osType) like '%andro%' then 'android'
                  when lower(r.osType) like '%os%' then 'ios' else lower(r.osType) end),
            (case when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%andro%' then 'android'
                  when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%os%' then 'ios'
                  when lower(loanmaster.deviceType) like '%andro%' then 'android'
                  else 'ios' end)
            ) as osType
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
  qualify row_number() over(partition by digitalLoanAccountId, modelVersionId order by appln_submit_datetime) = 1
  ;
"""
dfd = client.query(sq).to_dataframe()
# dfd = dfd.drop_duplicates(keep='first')
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()


# %%
df1 = dfd.copy()


# #### Train

# %%
sq = """
WITH cleaned AS (
  SELECT
    mmrd.customerId,mmrd.digitalLoanAccountId,prediction,start_time,end_time,modelDisplayName,modelVersionId,
  case when trenchCategory is null then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench1' end)
     when trenchCategory = '' then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench 1' end)
    else trenchCategory end  as trenchCategory,
    REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature,
    REPLACE(REPLACE(cast(prediction as string), "'", '"'), "None", "null") AS prediction_clean,
    Data_selection,
    deviceOs osType,
  FROM prj-prod-dataplatform.dap_ds_poweruser_playground.ml_training_model_run_details_20260116 mmrd
    left join prj-prod-dataplatform.risk_credit_mis.model_loan_score_mart mt on mt.digitalLoanAccountId = mmrd.digitalLoanAccountId
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
  trenchCategory,
  Data_selection,
  osType,
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
   Data_selection,
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
     coalesce((case when lower(r.osType) like '%andro%' then 'android'
                  when lower(r.osType) like '%os%' then 'ios' else lower(r.osType) end),
            (case when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%andro%' then 'android'
                  when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%os%' then 'ios'
                  when lower(loanmaster.deviceType) like '%andro%' then 'android'
                  else 'ios' end)
            ) as osType
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
  select * from base
  qualify row_number() over(partition by digitalLoanAccountId, modelVersionId order by appln_submit_datetime) = 1
  ;
  """
dfd = client.query(sq).to_dataframe()
# dfd = dfd.drop_duplicates(keep='first')
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()

# %%
df2 = dfd.copy()


# #### Concatenate both Test and Train Datasets


# 1) Get all IDs present in Train
train_ids = set(df2["digitalLoanAccountId"])

# 2) Keep only Test rows whose ID is NOT in Train
df1_no_dupes = df1[~df1["digitalLoanAccountId"].isin(train_ids)]

# 3) Concatenate
df_concat = pd.concat([df1_no_dupes, df2], ignore_index=True)

print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")
df_concat.head()


# #### Making Sure the Score is Numeric

# %%
df_concat["sil_beta_app_score"] = pd.to_numeric(
    df_concat["sil_beta_app_score"], errors="coerce"
)


# #### Calculating the Gini

# %%
fact_table, dimension_table = calculate_periodic_gini_prod_ver_trench_dimfact(
    df_concat,
    "sil_beta_app_score",
    "deffpd30",
    "FPD30",
    data_selection_column="Data_selection",
    model_version_column="modelVersionId",
    trench_column="trenchCategory",
    loan_type_column="new_loan_type",
    loan_product_type_column="loan_product_type",
    ostype_column="osType",
    account_id_column="digitalLoanAccountId",
)


# #### Updating the Fact and Dimension Table

# %%
fact_table, dimension_table = update_tables(
    fact_table, dimension_table, model_name="apps_score_model_sil", product="SIL"
)

df_f_fpd30_appscoresil = fact_table.copy()
df_d_fpd30_appscoresil = dimension_table.copy()


# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_f_fpd30_appscoresil, facttable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_d_fpd30_appscoresil, dimtable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# ### FSPD30


# #### Test

# %%
sq = """
WITH cleaned AS (
    SELECT
    mmrd.customerId,mmrd.digitalLoanAccountId,prediction,start_time,end_time,modelDisplayName,modelVersionId,
  case when trenchCategory is null then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench1' end)
     when trenchCategory = '' then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench 1' end)
    else trenchCategory end  as trenchCategory,
    REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature,
    REPLACE(REPLACE(prediction, "'", '"'), "None", "null") AS prediction_clean,
     deviceOs osType,
   FROM prj-prod-dataplatform.audit_balance.ml_model_run_details mmrd
  left join prj-prod-dataplatform.risk_credit_mis.model_loan_score_mart mt on mt.digitalLoanAccountId = mmrd.digitalLoanAccountId
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
 osType,
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
  'Prod' Data_selection,
  del.deffspd30,
  del.flg_mature_fspd_30,
  loanmaster.new_loan_type,
  modelVersionId, trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile'
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall'
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
     coalesce((case when lower(r.osType) like '%andro%' then 'android'
                  when lower(r.osType) like '%os%' then 'ios' else lower(r.osType) end),
            (case when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%andro%' then 'android'
                  when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%os%' then 'ios'
                  when lower(loanmaster.deviceType) like '%andro%' then 'android'
                  else 'ios' end)
            ) as osType
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
  qualify row_number() over(partition by digitalLoanAccountId, modelVersionId order by appln_submit_datetime) = 1
  ;
"""
dfd = client.query(sq).to_dataframe()
# dfd = dfd.drop_duplicates(keep='first')
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()


# %%
df1 = dfd.copy()


# #### Train

# %%
sq = """
WITH cleaned AS (
  SELECT
    mmrd.customerId,mmrd.digitalLoanAccountId,prediction,start_time,end_time,modelDisplayName,modelVersionId,
  case when trenchCategory is null then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench1' end)
     when trenchCategory = '' then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench 1' end)
    else trenchCategory end  as trenchCategory,
    REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature,
    REPLACE(REPLACE(cast(prediction as string), "'", '"'), "None", "null") AS prediction_clean,
    Data_selection,
    deviceOs osType,
  FROM prj-prod-dataplatform.dap_ds_poweruser_playground.ml_training_model_run_details_20260116 mmrd
    left join prj-prod-dataplatform.risk_credit_mis.model_loan_score_mart mt on mt.digitalLoanAccountId = mmrd.digitalLoanAccountId
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
  trenchCategory,
  Data_selection,
  osType,
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
  Data_selection,
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
     coalesce((case when lower(r.osType) like '%andro%' then 'android'
                  when lower(r.osType) like '%os%' then 'ios' else lower(r.osType) end),
            (case when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%andro%' then 'android'
                  when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%os%' then 'ios'
                  when lower(loanmaster.deviceType) like '%andro%' then 'android'
                  else 'ios' end)
            ) as osType
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
  select * from base
  qualify row_number() over(partition by digitalLoanAccountId, modelVersionId order by appln_submit_datetime) = 1
  ;
  """
dfd = client.query(sq).to_dataframe()
# dfd = dfd.drop_duplicates(keep='first')
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()

# %%
df2 = dfd.copy()


# #### Concatenate both Test and Train Datasets


# 1) Get all IDs present in Train
train_ids = set(df2["digitalLoanAccountId"])

# 2) Keep only Test rows whose ID is NOT in Train
df1_no_dupes = df1[~df1["digitalLoanAccountId"].isin(train_ids)]

# 3) Concatenate
df_concat = pd.concat([df1_no_dupes, df2], ignore_index=True)

print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")
df_concat.head()


# #### Making sure the score column in numeric

# %%
df_concat["sil_beta_app_score"] = pd.to_numeric(
    df_concat["sil_beta_app_score"], errors="coerce"
)


# #### Calculating the Gini

# %%
fact_table, dimension_table = calculate_periodic_gini_prod_ver_trench_dimfact(
    df_concat,
    "sil_beta_app_score",
    "deffspd30",
    "FSPD30",
    data_selection_column="Data_selection",
    model_version_column="modelVersionId",
    trench_column="trenchCategory",
    loan_type_column="new_loan_type",
    loan_product_type_column="loan_product_type",
    ostype_column="osType",
    account_id_column="digitalLoanAccountId",
)


# #### Updating the Fact and Dimension Table

# %%
fact_table, dimension_table = update_tables(
    fact_table, dimension_table, model_name="apps_score_model_sil", product="SIL"
)

df_f_fspd30_appscoresil = fact_table.copy()
df_d_fspd30_appscoresil = dimension_table.copy()


# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_f_fspd30_appscoresil, facttable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_d_fspd30_appscoresil, dimtable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# ### FSTPD30


# #### Test

# %%
sq = """
WITH cleaned AS (
    SELECT
    mmrd.customerId,mmrd.digitalLoanAccountId,prediction,start_time,end_time,modelDisplayName,modelVersionId,
  case when trenchCategory is null then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench1' end)
     when trenchCategory = '' then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench 1' end)
    else trenchCategory end  as trenchCategory,
    REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature,
    REPLACE(REPLACE(prediction, "'", '"'), "None", "null") AS prediction_clean,
      deviceOs osType,
   FROM prj-prod-dataplatform.audit_balance.ml_model_run_details mmrd
  left join prj-prod-dataplatform.risk_credit_mis.model_loan_score_mart mt on mt.digitalLoanAccountId = mmrd.digitalLoanAccountId
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
  osType,
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
  'Prod' Data_selection,
  del.deffstpd30,
  del.flg_mature_fstpd_30,
  loanmaster.new_loan_type,
  modelVersionId, trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile'
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall'
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
    coalesce((case when lower(r.osType) like '%andro%' then 'android'
                  when lower(r.osType) like '%os%' then 'ios' else lower(r.osType) end),
            (case when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%andro%' then 'android'
                  when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%os%' then 'ios'
                  when lower(loanmaster.deviceType) like '%andro%' then 'android'
                  else 'ios' end)
            ) as osType
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
  qualify row_number() over(partition by digitalLoanAccountId, modelVersionId order by appln_submit_datetime) = 1
  ;
"""
dfd = client.query(sq).to_dataframe()
# dfd = dfd.drop_duplicates(keep='first')
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()


# %%
df1 = dfd.copy()


# #### Train

# %%
sq = """
WITH cleaned AS (
  SELECT
    mmrd.customerId,mmrd.digitalLoanAccountId,prediction,start_time,end_time,modelDisplayName,modelVersionId,
   case when trenchCategory is null then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench1' end)
     when trenchCategory = '' then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench 1' end)
    else trenchCategory end  as trenchCategory,
    REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature,
    REPLACE(REPLACE(cast(prediction as string), "'", '"'), "None", "null") AS prediction_clean,
    Data_selection,
    deviceOs osType,
  FROM prj-prod-dataplatform.dap_ds_poweruser_playground.ml_training_model_run_details_20260116 mmrd
    left join prj-prod-dataplatform.risk_credit_mis.model_loan_score_mart mt on mt.digitalLoanAccountId = mmrd.digitalLoanAccountId
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
  trenchCategory,
  Data_selection,
  osType,
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
  Data_selection,
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
    coalesce((case when lower(r.osType) like '%andro%' then 'android'
                  when lower(r.osType) like '%os%' then 'ios' else lower(r.osType) end),
            (case when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%andro%' then 'android'
                  when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%os%' then 'ios'
                  when lower(loanmaster.deviceType) like '%andro%' then 'android'
                  else 'ios' end)
            ) as osType
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
  select * from base
  qualify row_number() over(partition by digitalLoanAccountId, modelVersionId order by appln_submit_datetime) = 1
  ;
  """
dfd = client.query(sq).to_dataframe()
# dfd = dfd.drop_duplicates(keep='first')
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()

# %%
df2 = dfd.copy()


# #### Concatenate both Test and Train Datasets

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


# #### Making sure the score column is numeric

# %%
df_concat["sil_beta_app_score"] = pd.to_numeric(
    df_concat["sil_beta_app_score"], errors="coerce"
)


# #### Calculating the Gini

# %%
fact_table, dimension_table = calculate_periodic_gini_prod_ver_trench_dimfact(
    df_concat,
    "sil_beta_app_score",
    "deffstpd30",
    "FSTPD30",
    data_selection_column="Data_selection",
    model_version_column="modelVersionId",
    trench_column="trenchCategory",
    loan_type_column="new_loan_type",
    loan_product_type_column="loan_product_type",
    ostype_column="osType",
    account_id_column="digitalLoanAccountId",
)


# #### Updating the Fact and Dimension Tables

# %%
fact_table, dimension_table = update_tables(
    fact_table, dimension_table, model_name="apps_score_model_sil", product="SIL"
)
print(f"The shape of the fact table is:\t {fact_table.shape}")
print(f"The shape of the dimension table is:\t {dimension_table.shape}")

df_f_fstpd30_appscoresil = fact_table.copy()
df_d_fstpd30_appscoresil = dimension_table.copy()

# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_f_fstpd30_appscoresil, facttable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_d_fstpd30_appscoresil, dimtable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

factappscoresil = pd.concat(
    [
        df_f_fpd0_appscoresil,
        df_f_fpd10_appscoresil,
        df_f_fpd30_appscoresil,
        df_f_fspd30_appscoresil,
        df_f_fstpd30_appscoresil,
    ],
    ignore_index=True,
)
demoappscoresil = pd.concat(
    [
        df_d_fpd0_appscoresil,
        df_d_fpd10_appscoresil,
        df_d_fpd30_appscoresil,
        df_d_fspd30_appscoresil,
        df_d_fstpd30_appscoresil,
    ],
    ignore_index=True,
)


# %% [markdown]
# #### Beta SIL Demo Score

# %%
facttable_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_betademosil"
dimtable_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_betademosil"

# ### FPD0

# #### Test

# %%
sq = """
WITH cleaned AS (
    SELECT
  mmrd.customerId,mmrd.digitalLoanAccountId,prediction,start_time,end_time,modelDisplayName,modelVersionId,
  case when trenchCategory is null then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench1' end)
     when trenchCategory = '' then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench 1' end)
    else trenchCategory end  as trenchCategory,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature_cleaned,
deviceOs osType,
  FROM `prj-prod-dataplatform.audit_balance.ml_model_run_details` mmrd
  left join prj-prod-dataplatform.risk_credit_mis.model_loan_score_mart mt on mt.digitalLoanAccountId = mmrd.digitalLoanAccountId
  WHERE modelDisplayName in ('Beta - DemoScoreModel', 'beta_demo_model_sil')
  ),
  modelname as (
  select customerId,
  digitalLoanAccountId,
  start_time,
  prediction sil_beta_demo_score,
  modelDisplayName,
  modelVersionId, trenchCategory,
  osType,
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
  'Prod' Data_selection,
  deffpd0,
  flg_mature_fpd0,
  loanmaster.new_loan_type,
  modelVersionId, trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile'
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall'
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
        coalesce((case when lower(r.osType) like '%andro%' then 'android'
                  when lower(r.osType) like '%os%' then 'ios' else lower(r.osType) end),
            (case when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%andro%' then 'android'
                  when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%os%' then 'ios'
                  when lower(loanmaster.deviceType) like '%andro%' then 'android'
                  else 'ios' end)
            ) as osType
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
  qualify row_number() over(partition by digitalLoanAccountId, modelVersionId order by appln_submit_datetime) = 1
  ;
"""
dfd = client.query(sq).to_dataframe()
# dfd = dfd.drop_duplicates(keep='first')
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()


# %%
df1 = dfd.copy()


# ### Train

# %%
sq = """
WITH cleaned AS (
    SELECT
  mmrd.customerId, mmrd.digitalLoanAccountId,prediction,start_time,end_time,modelDisplayName,modelVersionId,
   case when trenchCategory is null then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench1' end)
     when trenchCategory = '' then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench 1' end)
    else trenchCategory end  as trenchCategory,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature_cleaned,
Data_selection,
   deviceOs osType,
  FROM prj-prod-dataplatform.dap_ds_poweruser_playground.ml_training_model_run_details_20260116 mmrd
    left join prj-prod-dataplatform.risk_credit_mis.model_loan_score_mart mt on mt.digitalLoanAccountId = mmrd.digitalLoanAccountId
  WHERE modelDisplayName in ('Beta - DemoScoreModel', 'beta_demo_model_sil')
  ),
  modelname as (
  select customerId,
  digitalLoanAccountId,
  start_time,
  prediction sil_beta_demo_score,
  case when modelDisplayName = 'Beta - DemoScoreModel' then 'beta_demo_model_sil' else modelDisplayName end as modelDisplayName,
  modelVersionId, trenchCategory,Data_selection,
     osType,
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
  Data_selection,
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
         coalesce((case when lower(r.osType) like '%andro%' then 'android'
                  when lower(r.osType) like '%os%' then 'ios' else lower(r.osType) end),
            (case when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%andro%' then 'android'
                  when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%os%' then 'ios'
                  when lower(loanmaster.deviceType) like '%andro%' then 'android'
                  else 'ios' end)
            ) as osType
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
  select * from base
  qualify row_number() over(partition by digitalLoanAccountId, modelVersionId order by appln_submit_datetime) = 1
  ;
  """
dfd = client.query(sq).to_dataframe()
# dfd = dfd.drop_duplicates(keep='first')
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()

# %%
df2 = dfd.copy()


# #### Concatenate Test and Train Datasets


# 1) Get all IDs present in Train
train_ids = set(df2["digitalLoanAccountId"])

# 2) Keep only Test rows whose ID is NOT in Train
df1_no_dupes = df1[~df1["digitalLoanAccountId"].isin(train_ids)]

# 3) Concatenate
df_concat = pd.concat([df1_no_dupes, df2], ignore_index=True)

print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")
df_concat.head()


# #### Making sure the score is numerical

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
    data_selection_column="Data_selection",
    model_version_column="modelVersionId",
    trench_column="trenchCategory",
    loan_type_column="new_loan_type",
    loan_product_type_column="loan_product_type",
    ostype_column="osType",
    account_id_column="digitalLoanAccountId",
)

# %%
fact_table, dimension_table = update_tables(
    fact_table, dimension_table, model_name="beta_demo_model_sil", product="SIL"
)
print(f"The shape of the fact table is:\t {fact_table.shape}")
print(f"The shape of the dimension table is:\t {dimension_table.shape}")

df_f_fpd0_betademoscoresil = fact_table.copy()
df_d_fpd0_betademoscoresil = dimension_table.copy()

# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_TRUNCATE",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_f_fpd0_betademoscoresil, facttable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_TRUNCATE",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_d_fpd0_betademoscoresil, dimtable_id, job_config=job_config
)
job.result()  # Wait for the job to complete


### FPD10


# ### Test

# %%
sq = """
WITH cleaned AS (
     SELECT
  mmrd.customerId,mmrd.digitalLoanAccountId,prediction,start_time,end_time,modelDisplayName,modelVersionId,
    case when trenchCategory is null then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench1' end)
     when trenchCategory = '' then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench 1' end)
    else trenchCategory end  as trenchCategory,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature_cleaned,
    deviceOs osType,
  FROM `prj-prod-dataplatform.audit_balance.ml_model_run_details` mmrd
  left join prj-prod-dataplatform.risk_credit_mis.model_loan_score_mart mt on mt.digitalLoanAccountId = mmrd.digitalLoanAccountId
  WHERE modelDisplayName in ('Beta - DemoScoreModel', 'beta_demo_model_sil')
  ),
  modelname as (
  select customerId,
  digitalLoanAccountId,
  start_time,
  prediction sil_beta_demo_score,
  modelDisplayName,
  modelVersionId, trenchCategory,
      osType,
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
  'Prod' Data_selection,
  del.deffpd10,
  del.flg_mature_fpd10,
  loanmaster.new_loan_type,
  modelVersionId, trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile'
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall'
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
        coalesce((case when lower(r.osType) like '%andro%' then 'android'
                  when lower(r.osType) like '%os%' then 'ios' else lower(r.osType) end),
            (case when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%andro%' then 'android'
                  when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%os%' then 'ios'
                  when lower(loanmaster.deviceType) like '%andro%' then 'android'
                  else 'ios' end)
            ) as osType
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
  qualify row_number() over(partition by digitalLoanAccountId, modelVersionId order by appln_submit_datetime) = 1
  ;
"""
dfd = client.query(sq).to_dataframe()
# dfd = dfd.drop_duplicates(keep='first')
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()


# %%
df1 = dfd.copy()


# ### Train

# %%
sq = """
WITH cleaned AS (
    SELECT
  mmrd.customerId,mmrd.digitalLoanAccountId,prediction,start_time,end_time,modelDisplayName,modelVersionId,
  case when trenchCategory is null then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench1' end)
     when trenchCategory = '' then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench 1' end)
    else trenchCategory end  as trenchCategory,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature_cleaned,
Data_selection,
    deviceOs osType,
  FROM prj-prod-dataplatform.dap_ds_poweruser_playground.ml_training_model_run_details_20260116 mmrd
    left join prj-prod-dataplatform.risk_credit_mis.model_loan_score_mart mt on mt.digitalLoanAccountId = mmrd.digitalLoanAccountId
  WHERE modelDisplayName in ('Beta - DemoScoreModel', 'beta_demo_model_sil')
  ),
  modelname as (
  select customerId,
  digitalLoanAccountId,
  start_time,
  prediction sil_beta_demo_score,
  case when modelDisplayName = 'Beta - DemoScoreModel' then 'beta_demo_model_sil' else modelDisplayName end as modelDisplayName,
  modelVersionId, trenchCategory,Data_selection,
      osType,
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
  Data_selection,
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
        coalesce((case when lower(r.osType) like '%andro%' then 'android'
                  when lower(r.osType) like '%os%' then 'ios' else lower(r.osType) end),
            (case when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%andro%' then 'android'
                  when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%os%' then 'ios'
                  when lower(loanmaster.deviceType) like '%andro%' then 'android'
                  else 'ios' end)
            ) as osType
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
  select * from base
  qualify row_number() over(partition by digitalLoanAccountId, modelVersionId order by appln_submit_datetime) = 1
  ;
  """
dfd = client.query(sq).to_dataframe()
# dfd = dfd.drop_duplicates(keep='first')
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()

# %%
df2 = dfd.copy()


# #### Concatenate Test and Train Set


# 1) Get all IDs present in Train
train_ids = set(df2["digitalLoanAccountId"])

# 2) Keep only Test rows whose ID is NOT in Train
df1_no_dupes = df1[~df1["digitalLoanAccountId"].isin(train_ids)]

# 3) Concatenate
df_concat = pd.concat([df1_no_dupes, df2], ignore_index=True)

print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")
df_concat.head()


# #### Making sure the score is numerical

# %%
df_concat["sil_beta_demo_score"] = pd.to_numeric(
    df_concat["sil_beta_demo_score"], errors="coerce"
)


# #### Calculate Gini

# %%
fact_table, dimension_table = calculate_periodic_gini_prod_ver_trench_dimfact(
    df_concat,
    "sil_beta_demo_score",
    "deffpd10",
    "FPD10",
    data_selection_column="Data_selection",
    model_version_column="modelVersionId",
    trench_column="trenchCategory",
    loan_type_column="new_loan_type",
    loan_product_type_column="loan_product_type",
    ostype_column="osType",
    account_id_column="digitalLoanAccountId",
)


# #### Update fact and dimension table

# %%
fact_table, dimension_table = update_tables(
    fact_table, dimension_table, model_name="beta_demo_model_sil", product="SIL"
)

df_f_fpd10_betademoscoresil = fact_table.copy()
df_d_fpd10_betademoscoresil = dimension_table.copy()


# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_f_fpd10_betademoscoresil, facttable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_d_fpd10_betademoscoresil, dimtable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# ### FPD30


# ### Test

# %%
sq = """
WITH cleaned AS (
 SELECT
  mmrd.customerId,mmrd.digitalLoanAccountId,prediction,start_time,end_time,modelDisplayName,modelVersionId,
   case when trenchCategory is null then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench1' end)
     when trenchCategory = '' then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench 1' end)
    else trenchCategory end  as trenchCategory,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature_cleaned,
   deviceOs osType,

  FROM `prj-prod-dataplatform.audit_balance.ml_model_run_details` mmrd
  left join prj-prod-dataplatform.risk_credit_mis.model_loan_score_mart mt on mt.digitalLoanAccountId = mmrd.digitalLoanAccountId
  WHERE modelDisplayName in ('Beta - DemoScoreModel', 'beta_demo_model_sil')
  ),
  modelname as (
  select customerId,
  digitalLoanAccountId,
  start_time,
  prediction sil_beta_demo_score,
  modelDisplayName,
  modelVersionId, trenchCategory,
     osType,
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
  'Prod' Data_selection,
  del.deffpd30,
  del.flg_mature_fpd30,
  loanmaster.new_loan_type,
  modelVersionId, trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile'
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall'
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,

    coalesce((case when lower(r.osType) like '%andro%' then 'android'
                  when lower(r.osType) like '%os%' then 'ios' else lower(r.osType) end),
            (case when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%andro%' then 'android'
                  when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%os%' then 'ios'
                  when lower(loanmaster.deviceType) like '%andro%' then 'android'
                  else 'ios' end)
            ) as osType
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
  qualify row_number() over(partition by digitalLoanAccountId, modelVersionId order by appln_submit_datetime) = 1
  ;
"""
dfd = client.query(sq).to_dataframe()
# dfd = dfd.drop_duplicates(keep='first')
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()


# %%
df1 = dfd.copy()


# ### Train

# %%
sq = """
WITH cleaned AS (
    SELECT
  mmrd.customerId,mmrd.digitalLoanAccountId,prediction,start_time,end_time,modelDisplayName,modelVersionId,
   case when trenchCategory is null then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench1' end)
     when trenchCategory = '' then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench 1' end)
    else trenchCategory end  as trenchCategory,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature_cleaned,
Data_selection,
    deviceOs osType,
  FROM prj-prod-dataplatform.dap_ds_poweruser_playground.ml_training_model_run_details_20260116 mmrd
    left join prj-prod-dataplatform.risk_credit_mis.model_loan_score_mart mt on mt.digitalLoanAccountId = mmrd.digitalLoanAccountId
  WHERE modelDisplayName in ('Beta - DemoScoreModel', 'beta_demo_model_sil')
  ),
  modelname as (
  select customerId,
  digitalLoanAccountId,
  start_time,
  prediction sil_beta_demo_score,
  case when modelDisplayName = 'Beta - DemoScoreModel' then 'beta_demo_model_sil' else modelDisplayName end as modelDisplayName,
  modelVersionId, trenchCategory, Data_selection,
      osType,
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
  Data_selection,
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
        coalesce((case when lower(r.osType) like '%andro%' then 'android'
                  when lower(r.osType) like '%os%' then 'ios' else lower(r.osType) end),
            (case when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%andro%' then 'android'
                  when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%os%' then 'ios'
                  when lower(loanmaster.deviceType) like '%andro%' then 'android'
                  else 'ios' end)
            ) as osType
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
  select * from base
  qualify row_number() over(partition by digitalLoanAccountId, modelVersionId order by appln_submit_datetime) = 1
  ;
  """
dfd = client.query(sq).to_dataframe()
# dfd = dfd.drop_duplicates(keep='first')
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()

# %%
df2 = dfd.copy()


# ### Concat


# 1) Get all IDs present in Train
train_ids = set(df2["digitalLoanAccountId"])

# 2) Keep only Test rows whose ID is NOT in Train
df1_no_dupes = df1[~df1["digitalLoanAccountId"].isin(train_ids)]

# 3) Concatenate
df_concat = pd.concat([df1_no_dupes, df2], ignore_index=True)

print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")
df_concat.head()


# #### Making sure the score is numerical

# %%
df_concat["sil_beta_demo_score"] = pd.to_numeric(
    df_concat["sil_beta_demo_score"], errors="coerce"
)


# #### Calculate the Gini

# %%
fact_table, dimension_table = calculate_periodic_gini_prod_ver_trench_dimfact(
    df_concat,
    "sil_beta_demo_score",
    "deffpd30",
    "FPD30",
    data_selection_column="Data_selection",
    model_version_column="modelVersionId",
    trench_column="trenchCategory",
    loan_type_column="new_loan_type",
    loan_product_type_column="loan_product_type",
    ostype_column="osType",
    account_id_column="digitalLoanAccountId",
)


# #### Updating fact and dimension tables

# %%
fact_table, dimension_table = update_tables(
    fact_table, dimension_table, model_name="beta_demo_model_sil", product="SIL"
)

df_f_fpd30_betademoscoresil = fact_table.copy()
df_d_fpd30_betademoscoresil = dimension_table.copy()

# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_f_fpd30_betademoscoresil, facttable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_d_fpd30_betademoscoresil, dimtable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# ### FSPD30


# ### Test

# %%
sq = """
WITH cleaned AS (
     SELECT
  mmrd.customerId,mmrd.digitalLoanAccountId,prediction,start_time,end_time,modelDisplayName,modelVersionId,
    case when trenchCategory is null then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench1' end)
     when trenchCategory = '' then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench 1' end)
    else trenchCategory end  as trenchCategory,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature_cleaned,
   deviceOs osType,
  FROM `prj-prod-dataplatform.audit_balance.ml_model_run_details` mmrd
  left join prj-prod-dataplatform.risk_credit_mis.model_loan_score_mart mt on mt.digitalLoanAccountId = mmrd.digitalLoanAccountId
  WHERE modelDisplayName in ('Beta - DemoScoreModel', 'beta_demo_model_sil')
  ),
  modelname as (
  select customerId,
  digitalLoanAccountId,
  start_time,
  prediction sil_beta_demo_score,
  modelDisplayName,
  modelVersionId, trenchCategory,
     osType,
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
  'Prod' Data_selection,
  del.deffspd30,
  del.flg_mature_fspd_30,
  loanmaster.new_loan_type,
  modelVersionId, trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile'
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall'
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
        coalesce((case when lower(r.osType) like '%andro%' then 'android'
                  when lower(r.osType) like '%os%' then 'ios' else lower(r.osType) end),
            (case when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%andro%' then 'android'
                  when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%os%' then 'ios'
                  when lower(loanmaster.deviceType) like '%andro%' then 'android'
                  else 'ios' end)
            ) as osType
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
  qualify row_number() over(partition by digitalLoanAccountId, modelVersionId order by appln_submit_datetime) = 1
  ;
"""
dfd = client.query(sq).to_dataframe()
# dfd = dfd.drop_duplicates(keep='first')
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()


# %%
df1 = dfd.copy()


# ### Train

# %%
sq = """
WITH cleaned AS (
    SELECT
  mmrd.customerId,mmrd.digitalLoanAccountId,prediction,start_time,end_time,modelDisplayName,modelVersionId,
    case when trenchCategory is null then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench1' end)
     when trenchCategory = '' then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench 1' end)
    else trenchCategory end  as trenchCategory,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature_cleaned,
Data_selection,
    deviceOs osType,
  FROM prj-prod-dataplatform.dap_ds_poweruser_playground.ml_training_model_run_details_20260116 mmrd
    left join prj-prod-dataplatform.risk_credit_mis.model_loan_score_mart mt on mt.digitalLoanAccountId = mmrd.digitalLoanAccountId
  WHERE modelDisplayName in ('Beta - DemoScoreModel', 'beta_demo_model_sil')
  ),
  modelname as (
  select customerId,
  digitalLoanAccountId,
  start_time,
  prediction sil_beta_demo_score,
  case when modelDisplayName = 'Beta - DemoScoreModel' then 'beta_demo_model_sil' else modelDisplayName end as modelDisplayName,
  modelVersionId, trenchCategory, Data_selection,
      osType,
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
  Data_selection,
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
        coalesce((case when lower(r.osType) like '%andro%' then 'android'
                  when lower(r.osType) like '%os%' then 'ios' else lower(r.osType) end),
            (case when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%andro%' then 'android'
                  when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%os%' then 'ios'
                  when lower(loanmaster.deviceType) like '%andro%' then 'android'
                  else 'ios' end)
            ) as osType
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
  select * from base
  qualify row_number() over(partition by digitalLoanAccountId, modelVersionId order by appln_submit_datetime) = 1
  ;
  """
dfd = client.query(sq).to_dataframe()
# dfd = dfd.drop_duplicates(keep='first')
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()

# %%
df2 = dfd.copy()


# #### Concatenate Test and Train Dataset


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
    data_selection_column="Data_selection",
    model_version_column="modelVersionId",
    trench_column="trenchCategory",
    loan_type_column="new_loan_type",
    loan_product_type_column="loan_product_type",
    ostype_column="osType",
    account_id_column="digitalLoanAccountId",
)

fact_table, dimension_table = update_tables(
    fact_table, dimension_table, model_name="beta_demo_model_sil", product="SIL"
)

df_f_fspd30_betademoscoresil = fact_table.copy()
df_d_fspd30_betademoscoresil = dimension_table.copy()

# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_f_fspd30_betademoscoresil, facttable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_d_fspd30_betademoscoresil, dimtable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# ### FSTPD30


# ### Test

# %%
sq = """
WITH cleaned AS (
SELECT
  mmrd.customerId,mmrd.digitalLoanAccountId,prediction,start_time,end_time,modelDisplayName,modelVersionId,
  case when trenchCategory is null then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench1' end)
     when trenchCategory = '' then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench 1' end)
    else trenchCategory end  as trenchCategory,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature_cleaned,
    deviceOs osType,
  FROM `prj-prod-dataplatform.audit_balance.ml_model_run_details` mmrd
  left join prj-prod-dataplatform.risk_credit_mis.model_loan_score_mart mt on mt.digitalLoanAccountId = mmrd.digitalLoanAccountId
  WHERE modelDisplayName in ('Beta - DemoScoreModel', 'beta_demo_model_sil')
  ),
  modelname as (
  select customerId,
  digitalLoanAccountId,
  start_time,
  prediction sil_beta_demo_score,
  modelDisplayName,
  modelVersionId, trenchCategory,
      osType,
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
  'Prod' Data_selection,
  del.deffstpd30,
  del.flg_mature_fstpd_30,
  loanmaster.new_loan_type,
  modelVersionId, trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile'
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall'
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
    coalesce((case when lower(r.osType) like '%andro%' then 'android'
                  when lower(r.osType) like '%os%' then 'ios' else lower(r.osType) end),
            (case when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%andro%' then 'android'
                  when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%os%' then 'ios'
                  when lower(loanmaster.deviceType) like '%andro%' then 'android'
                  else 'ios' end)
            ) as osType
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
  qualify row_number() over(partition by digitalLoanAccountId, modelVersionId order by appln_submit_datetime) = 1
  ;
"""
dfd = client.query(sq).to_dataframe()
# dfd = dfd.drop_duplicates(keep='first')
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()


# %%
df1 = dfd.copy()


# ### Train

# %%
sq = """
WITH cleaned AS (
  SELECT
  mmrd.customerId,mmrd.digitalLoanAccountId,prediction,start_time,end_time,modelDisplayName,modelVersionId,
  case when trenchCategory is null then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench1' end)
     when trenchCategory = '' then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench 1' end)
    else trenchCategory end  as trenchCategory,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature_cleaned,
Data_selection,
    deviceOs osType,
  FROM prj-prod-dataplatform.dap_ds_poweruser_playground.ml_training_model_run_details_20260116 mmrd
    left join prj-prod-dataplatform.risk_credit_mis.model_loan_score_mart mt on mt.digitalLoanAccountId = mmrd.digitalLoanAccountId
  WHERE modelDisplayName in ('Beta - DemoScoreModel', 'beta_demo_model_sil')
  ),
  modelname as (
  select customerId,
  digitalLoanAccountId,
  start_time,
  prediction sil_beta_demo_score,
  case when modelDisplayName = 'Beta - DemoScoreModel' then 'beta_demo_model_sil' else modelDisplayName end as modelDisplayName,
  modelVersionId, trenchCategory, Data_selection,
      osType,
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
  Data_selection,
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
        coalesce((case when lower(r.osType) like '%andro%' then 'android'
                  when lower(r.osType) like '%os%' then 'ios' else lower(r.osType) end),
            (case when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%andro%' then 'android'
                  when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%os%' then 'ios'
                  when lower(loanmaster.deviceType) like '%andro%' then 'android'
                  else 'ios' end)
            ) as osType
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
  select * from base
  qualify row_number() over(partition by digitalLoanAccountId, modelVersionId order by appln_submit_datetime) = 1
  ;
  """
dfd = client.query(sq).to_dataframe()
# dfd = dfd.drop_duplicates(keep='first')
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()

# %%
df2 = dfd.copy()


# ### Concat


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
    data_selection_column="Data_selection",
    model_version_column="modelVersionId",
    trench_column="trenchCategory",
    loan_type_column="new_loan_type",
    loan_product_type_column="loan_product_type",
    ostype_column="osType",
    account_id_column="digitalLoanAccountId",
)

# %%
fact_table, dimension_table = update_tables(
    fact_table, dimension_table, model_name="beta_demo_model_sil", product="SIL"
)
print(f"The shape of the fact table is:\t {fact_table.shape}")
print(f"The shape of the dimension table is:\t {dimension_table.shape}")

df_f_fstpd30_betademoscoresil = fact_table.copy()
df_d_fstpd30_betademoscoresil = dimension_table.copy()


# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_f_fstpd30_betademoscoresil, facttable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_d_fstpd30_betademoscoresil, dimtable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

factbetademosil = pd.concat(
    [
        df_f_fpd0_betademoscoresil,
        df_f_fpd10_betademoscoresil,
        df_f_fpd30_betademoscoresil,
        df_f_fspd30_betademoscoresil,
        df_f_fstpd30_betademoscoresil,
    ],
    ignore_index=False,
)
dimbetademosil = pd.concat(
    [
        df_d_fpd0_betademoscoresil,
        df_d_fpd10_betademoscoresil,
        df_d_fpd30_betademoscoresil,
        df_d_fspd30_betademoscoresil,
        df_d_fstpd30_betademoscoresil,
    ],
    ignore_index=False,
)

print(f"beta_demo_model_sil model Gini Calculation and Upload Completed!")


# %% [markdown]
# #### Beta SIL STACK Score Model

# %%
# ##
facttable_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_betastacksil"
dimtable_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_betastacksil"

# ### FPD0
# #### Test
# %%
sq = """
WITH cleaned AS (
  SELECT
    mmrd.customerId,mmrd.digitalLoanAccountId,prediction,start_time,end_time,modelDisplayName,modelVersionId,
  case when trenchCategory is null then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench1' end)
     when trenchCategory = '' then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench 1' end)
    else trenchCategory end  as trenchCategory,
    REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature,
        deviceOs osType,
   FROM prj-prod-dataplatform.audit_balance.ml_model_run_details mmrd
  left join prj-prod-dataplatform.risk_credit_mis.model_loan_score_mart mt on mt.digitalLoanAccountId = mmrd.digitalLoanAccountId
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
    osType,
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
  'Prod' Data_selection,
  deffpd0,
  flg_mature_fpd0,
  loanmaster.new_loan_type,
  modelVersionId, trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile'
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall'
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
     coalesce((case when lower(r.osType) like '%andro%' then 'android'
                  when lower(r.osType) like '%os%' then 'ios' else lower(r.osType) end),
            (case when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%andro%' then 'android'
                  when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%os%' then 'ios'
                  when lower(loanmaster.deviceType) like '%andro%' then 'android'
                  else 'ios' end)
            ) as osType
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
  qualify row_number() over(partition by digitalLoanAccountId, modelVersionId order by appln_submit_datetime) = 1
  ;
"""
dfd = client.query(sq).to_dataframe()
# dfd = dfd.drop_duplicates(keep='first')
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()


# %%
df1 = dfd.copy()


# #### Train

# %%
sq = """
WITH cleaned AS (
  SELECT
    mmrd.customerId,mmrd.digitalLoanAccountId,prediction,start_time,end_time,modelDisplayName,modelVersionId,
  case when trenchCategory is null then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench1' end)
     when trenchCategory = '' then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench 1' end)
    else trenchCategory end  as trenchCategory,
    REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature,
    Data_selection,
        deviceOs osType,
   FROM prj-prod-dataplatform.dap_ds_poweruser_playground.ml_training_model_run_details_20260116 mmrd
  left join prj-prod-dataplatform.risk_credit_mis.model_loan_score_mart mt on mt.digitalLoanAccountId = mmrd.digitalLoanAccountId
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
  Data_selection,
    osType,
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
  Data_selection,
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
        coalesce((case when lower(r.osType) like '%andro%' then 'android'
                  when lower(r.osType) like '%os%' then 'ios' else lower(r.osType) end),
            (case when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%andro%' then 'android'
                  when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%os%' then 'ios'
                  when lower(loanmaster.deviceType) like '%andro%' then 'android'
                  else 'ios' end)
            ) as osType
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
  select * from base
  qualify row_number() over(partition by digitalLoanAccountId, modelVersionId order by appln_submit_datetime) = 1
  ;
  """
dfd = client.query(sq).to_dataframe()
# dfd = dfd.drop_duplicates(keep='first')
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()

# %%
df2 = dfd.copy()


# #### Concatenate both Test and Train Datasets


# 1) Get all IDs present in Train
train_ids = set(df2["digitalLoanAccountId"])

# 2) Keep only Test rows whose ID is NOT in Train
df1_no_dupes = df1[~df1["digitalLoanAccountId"].isin(train_ids)]

# 3) Concatenate
df_concat = pd.concat([df1_no_dupes, df2], ignore_index=True)

print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")
df_concat.head()


# #### Making Sure the Score is Numeric

# %%
df_concat["sil_beta_stack_score"] = pd.to_numeric(
    df_concat["sil_beta_stack_score"], errors="coerce"
)


# #### Calculating the Gini

# %%
fact_table, dimension_table = calculate_periodic_gini_prod_ver_trench_dimfact(
    df_concat,
    "sil_beta_stack_score",
    "deffpd0",
    "FPD0",
    data_selection_column="Data_selection",
    model_version_column="modelVersionId",
    trench_column="trenchCategory",
    loan_type_column="new_loan_type",
    loan_product_type_column="loan_product_type",
    ostype_column="osType",
    account_id_column="digitalLoanAccountId",
)


# #### Updating Fact and Dimension Table

# %%
fact_table, dimension_table = update_tables(
    fact_table, dimension_table, model_name="beta_stack_model_sil", product="SIL"
)
print(f"The shape of the fact table is:\t {fact_table.shape}")
print(f"The shape of the dimension table is:\t {dimension_table.shape}")


# #### Assigning the Table Name

df_f_fpd0_betastackscoresil = fact_table.copy()
df_d_fpd0_betastackscoresil = dimension_table.copy()


# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_TRUNCATE",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_f_fpd0_betastackscoresil, facttable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_TRUNCATE",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_d_fpd0_betastackscoresil, dimtable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# #### Creating the table


# ### FPD10


# #### Test

# %%
sq = """
WITH cleaned AS (
   SELECT
    mmrd.customerId,mmrd.digitalLoanAccountId,prediction,start_time,end_time,modelDisplayName,modelVersionId,
  case when trenchCategory is null then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench1' end)
     when trenchCategory = '' then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench 1' end)
    else trenchCategory end  as trenchCategory,
    REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature,
    deviceOs osType,
   FROM prj-prod-dataplatform.audit_balance.ml_model_run_details mmrd
  left join prj-prod-dataplatform.risk_credit_mis.model_loan_score_mart mt on mt.digitalLoanAccountId = mmrd.digitalLoanAccountId
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
  osType
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
  'Prod' Data_selection,
  del.deffpd10,
  del.flg_mature_fpd10,
  loanmaster.new_loan_type,
  modelVersionId, trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile'
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall'
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
        coalesce((case when lower(r.osType) like '%andro%' then 'android'
                  when lower(r.osType) like '%os%' then 'ios' else lower(r.osType) end),
            (case when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%andro%' then 'android'
                  when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%os%' then 'ios'
                  when lower(loanmaster.deviceType) like '%andro%' then 'android'
                  else 'ios' end)
            ) as osType
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
  qualify row_number() over(partition by digitalLoanAccountId, modelVersionId order by appln_submit_datetime) = 1

  ;
"""
dfd = client.query(sq).to_dataframe()
# dfd = dfd.drop_duplicates(keep='first')
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()


# %%
df1 = dfd.copy()


# #### Train

# %%
sq = """
WITH cleaned AS (
  SELECT
    mmrd.customerId,mmrd.digitalLoanAccountId,prediction,start_time,end_time,modelDisplayName,modelVersionId,
  case when trenchCategory is null then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench1' end)
     when trenchCategory = '' then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench 1' end)
    else trenchCategory end  as trenchCategory,
    REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature,
    Data_selection,
     deviceOs osType,
   FROM prj-prod-dataplatform.dap_ds_poweruser_playground.ml_training_model_run_details_20260116 mmrd
  left join prj-prod-dataplatform.risk_credit_mis.model_loan_score_mart mt on mt.digitalLoanAccountId = mmrd.digitalLoanAccountId
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
  Data_selection,
   osType,
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
  Data_selection,
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
    coalesce((case when lower(r.osType) like '%andro%' then 'android'
                  when lower(r.osType) like '%os%' then 'ios' else lower(r.osType) end),
            (case when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%andro%' then 'android'
                  when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%os%' then 'ios'
                  when lower(loanmaster.deviceType) like '%andro%' then 'android'
                  else 'ios' end)
            ) as osType

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
  select * from base
  qualify row_number() over(partition by digitalLoanAccountId, modelVersionId order by appln_submit_datetime) = 1
  ;
  """
dfd = client.query(sq).to_dataframe()
# dfd = dfd.drop_duplicates(keep='first')
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()

# %%
df2 = dfd.copy()


# #### Concatenate both Test and Train Dataset


# 1) Get all IDs present in Train
train_ids = set(df2["digitalLoanAccountId"])

# 2) Keep only Test rows whose ID is NOT in Train
df1_no_dupes = df1[~df1["digitalLoanAccountId"].isin(train_ids)]

# 3) Concatenate
df_concat = pd.concat([df1_no_dupes, df2], ignore_index=True)

print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")
df_concat.head()


# #### Making Sure the Score is Numeric

# %%
df_concat["sil_beta_stack_score"] = pd.to_numeric(
    df_concat["sil_beta_stack_score"], errors="coerce"
)


# #### Calculating the Gini

# %%
fact_table, dimension_table = calculate_periodic_gini_prod_ver_trench_dimfact(
    df_concat,
    "sil_beta_stack_score",
    "deffpd10",
    "FPD10",
    data_selection_column="Data_selection",
    model_version_column="modelVersionId",
    trench_column="trenchCategory",
    loan_type_column="new_loan_type",
    loan_product_type_column="loan_product_type",
    ostype_column="osType",
    account_id_column="digitalLoanAccountId",
)


# #### Updating Fact and Dimension Table

# %%
fact_table, dimension_table = update_tables(
    fact_table, dimension_table, model_name="beta_stack_model_sil", product="SIL"
)


df_f_fpd10_betastackscoresil = fact_table.copy()
df_d_fpd10_betastackscoresil = dimension_table.copy()

# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_f_fpd10_betastackscoresil, facttable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_d_fpd10_betastackscoresil, dimtable_id, job_config=job_config
)
job.result()  # Wait for the job to complete


# ### FPD30


# #### Test

# %%
sq = """
WITH cleaned AS (
   SELECT
    mmrd.customerId,mmrd.digitalLoanAccountId,prediction,start_time,end_time,modelDisplayName,modelVersionId,
   case when trenchCategory is null then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench1' end)
     when trenchCategory = '' then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench 1' end)
    else trenchCategory end  as trenchCategory,
    REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature,
    deviceOs osType,
   FROM prj-prod-dataplatform.audit_balance.ml_model_run_details mmrd
  left join prj-prod-dataplatform.risk_credit_mis.model_loan_score_mart mt on mt.digitalLoanAccountId = mmrd.digitalLoanAccountId
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
  osType,
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
  'Prod' Data_selection,
  del.deffpd30,
  del.flg_mature_fpd30,
  loanmaster.new_loan_type,
  modelVersionId, trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile'
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall'
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
    coalesce((case when lower(r.osType) like '%andro%' then 'android'
                  when lower(r.osType) like '%os%' then 'ios' else lower(r.osType) end),
            (case when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%andro%' then 'android'
                  when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%os%' then 'ios'
                  when lower(loanmaster.deviceType) like '%andro%' then 'android'
                  else 'ios' end)
            ) as osType
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
  qualify row_number() over(partition by digitalLoanAccountId, modelVersionId order by appln_submit_datetime) = 1
  ;
"""
dfd = client.query(sq).to_dataframe()
# dfd = dfd.drop_duplicates(keep='first')
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()


# %%
df1 = dfd.copy()


# #### Train

# %%
sq = """
WITH cleaned AS (
  SELECT
    mmrd.customerId,mmrd.digitalLoanAccountId,prediction,start_time,end_time,modelDisplayName,modelVersionId,
  case when trenchCategory is null then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench1' end)
     when trenchCategory = '' then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench 1' end)
    else trenchCategory end  as trenchCategory,
    REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature,
    Data_selection,
    deviceOs osType,
   FROM prj-prod-dataplatform.dap_ds_poweruser_playground.ml_training_model_run_details_20260116 mmrd
  left join prj-prod-dataplatform.risk_credit_mis.model_loan_score_mart mt on mt.digitalLoanAccountId = mmrd.digitalLoanAccountId
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
  Data_selection,
  osType,
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
  Data_selection,
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
        coalesce((case when lower(r.osType) like '%andro%' then 'android'
                  when lower(r.osType) like '%os%' then 'ios' else lower(r.osType) end),
            (case when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%andro%' then 'android'
                  when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%os%' then 'ios'
                  when lower(loanmaster.deviceType) like '%andro%' then 'android'
                  else 'ios' end)
            ) as osType
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
  select * from base
  qualify row_number() over(partition by digitalLoanAccountId, modelVersionId order by appln_submit_datetime) = 1
  ;
  """
dfd = client.query(sq).to_dataframe()
# dfd = dfd.drop_duplicates(keep='first')
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()

# %%
df2 = dfd.copy()


# #### Concatenate both Test and Train Datasets


# 1) Get all IDs present in Train
train_ids = set(df2["digitalLoanAccountId"])

# 2) Keep only Test rows whose ID is NOT in Train
df1_no_dupes = df1[~df1["digitalLoanAccountId"].isin(train_ids)]

# 3) Concatenate
df_concat = pd.concat([df1_no_dupes, df2], ignore_index=True)

print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")
df_concat.head()


# #### Making Sure the Score is Numeric

# %%
df_concat["sil_beta_stack_score"] = pd.to_numeric(
    df_concat["sil_beta_stack_score"], errors="coerce"
)


# #### Calculating the Gini

# %%
fact_table, dimension_table = calculate_periodic_gini_prod_ver_trench_dimfact(
    df_concat,
    "sil_beta_stack_score",
    "deffpd30",
    "FPD30",
    data_selection_column="Data_selection",
    model_version_column="modelVersionId",
    trench_column="trenchCategory",
    loan_type_column="new_loan_type",
    loan_product_type_column="loan_product_type",
    ostype_column="osType",
    account_id_column="digitalLoanAccountId",
)


# #### Updating the Fact and Dimension Table

# %%
fact_table, dimension_table = update_tables(
    fact_table, dimension_table, model_name="beta_stack_model_sil", product="SIL"
)

df_f_fpd30_betastackscoresil = fact_table.copy()
df_d_fpd30_betastackscoresil = dimension_table.copy()

# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_f_fpd30_betastackscoresil, facttable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_d_fpd30_betastackscoresil, dimtable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# #### Inserting the data into Fact and Dimension tables


# ### FSPD30


# #### Test

# %%
sq = """
WITH cleaned AS (
   SELECT
    mmrd.customerId,mmrd.digitalLoanAccountId,prediction,start_time,end_time,modelDisplayName,modelVersionId,
  case when trenchCategory is null then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench1' end)
     when trenchCategory = '' then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench 1' end)
    else trenchCategory end  as trenchCategory,
    REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature,
    deviceOs osType,
   FROM prj-prod-dataplatform.audit_balance.ml_model_run_details mmrd
  left join prj-prod-dataplatform.risk_credit_mis.model_loan_score_mart mt on mt.digitalLoanAccountId = mmrd.digitalLoanAccountId
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
  osType,
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
  'Prod' Data_selection,
  del.deffspd30,
  del.flg_mature_fspd_30,
  loanmaster.new_loan_type,
  modelVersionId, trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile'
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall'
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
     coalesce((case when lower(r.osType) like '%andro%' then 'android'
                  when lower(r.osType) like '%os%' then 'ios' else lower(r.osType) end),
            (case when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%andro%' then 'android'
                  when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%os%' then 'ios'
                  when lower(loanmaster.deviceType) like '%andro%' then 'android'
                  else 'ios' end)
            ) as osType
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
  qualify row_number() over(partition by digitalLoanAccountId, modelVersionId order by appln_submit_datetime) = 1
  ;
"""
dfd = client.query(sq).to_dataframe()
# dfd = dfd.drop_duplicates(keep='first')
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()


# %%
df1 = dfd.copy()


# #### Train

# %%
sq = """
WITH cleaned AS (
  SELECT
    mmrd.customerId,mmrd.digitalLoanAccountId,prediction,start_time,end_time,modelDisplayName,modelVersionId,
  case when trenchCategory is null then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench1' end)
     when trenchCategory = '' then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench 1' end)
    else trenchCategory end  as trenchCategory,
    REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature,
    Data_selection,
    deviceOs osType,
   FROM prj-prod-dataplatform.dap_ds_poweruser_playground.ml_training_model_run_details_20260116 mmrd
  left join prj-prod-dataplatform.risk_credit_mis.model_loan_score_mart mt on mt.digitalLoanAccountId = mmrd.digitalLoanAccountId
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
  Data_selection,
  osType,
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
  Data_selection,
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
        coalesce((case when lower(r.osType) like '%andro%' then 'android'
                  when lower(r.osType) like '%os%' then 'ios' else lower(r.osType) end),
            (case when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%andro%' then 'android'
                  when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%os%' then 'ios'
                  when lower(loanmaster.deviceType) like '%andro%' then 'android'
                  else 'ios' end)
            ) as osType
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
  select * from base
  qualify row_number() over(partition by digitalLoanAccountId, modelVersionId order by appln_submit_datetime) = 1
  ;
  """
dfd = client.query(sq).to_dataframe()
# dfd = dfd.drop_duplicates(keep='first')
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()

# %%
df2 = dfd.copy()


# #### Concatenate both Test and Train Datasets


# 1) Get all IDs present in Train
train_ids = set(df2["digitalLoanAccountId"])

# 2) Keep only Test rows whose ID is NOT in Train
df1_no_dupes = df1[~df1["digitalLoanAccountId"].isin(train_ids)]

# 3) Concatenate
df_concat = pd.concat([df1_no_dupes, df2], ignore_index=True)

print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")
df_concat.head()


# #### Making sure the score column in numeric

# %%
df_concat["sil_beta_stack_score"] = pd.to_numeric(
    df_concat["sil_beta_stack_score"], errors="coerce"
)


# #### Calculating the Gini

# %%
fact_table, dimension_table = calculate_periodic_gini_prod_ver_trench_dimfact(
    df_concat,
    "sil_beta_stack_score",
    "deffspd30",
    "FSPD30",
    data_selection_column="Data_selection",
    model_version_column="modelVersionId",
    trench_column="trenchCategory",
    loan_type_column="new_loan_type",
    loan_product_type_column="loan_product_type",
    ostype_column="osType",
    account_id_column="digitalLoanAccountId",
)


# #### Updating the Fact and Dimension Table

# %%
fact_table, dimension_table = update_tables(
    fact_table, dimension_table, model_name="beta_stack_model_sil", product="SIL"
)


# #### Inserting the Data into Fact and Dimension Table

df_f_fspd30_betastackscoresil = fact_table.copy()
df_d_fspd30_betastackscoresil = dimension_table.copy()

# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_f_fspd30_betastackscoresil, facttable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_d_fspd30_betastackscoresil, dimtable_id, job_config=job_config
)
job.result()  # Wait for the job to complete


# ### FSTPD30


# #### Test

# %%
sq = """
WITH cleaned AS (
   SELECT
    mmrd.customerId,mmrd.digitalLoanAccountId,prediction,start_time,end_time,modelDisplayName,modelVersionId,
  case when trenchCategory is null then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench1' end)
     when trenchCategory = '' then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench 1' end)
    else trenchCategory end  as trenchCategory,
    REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature,
    deviceOs osType,
   FROM prj-prod-dataplatform.audit_balance.ml_model_run_details mmrd
  left join prj-prod-dataplatform.risk_credit_mis.model_loan_score_mart mt on mt.digitalLoanAccountId = mmrd.digitalLoanAccountId
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
  osType,
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
  'Prod' Data_selection,
  del.deffstpd30,
  del.flg_mature_fstpd_30,
  loanmaster.new_loan_type,
  modelVersionId, trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile'
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall'
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
      coalesce((case when lower(r.osType) like '%andro%' then 'android'
                  when lower(r.osType) like '%os%' then 'ios' else lower(r.osType) end),
            (case when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%andro%' then 'android'
                  when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%os%' then 'ios'
                  when lower(loanmaster.deviceType) like '%andro%' then 'android'
                  else 'ios' end)
            ) as osType
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
  qualify row_number() over(partition by digitalLoanAccountId, modelVersionId order by appln_submit_datetime) = 1
  ;
"""
dfd = client.query(sq).to_dataframe()
# dfd = dfd.drop_duplicates(keep='first')
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()


# %%
df1 = dfd.copy()


# #### Train

# %%
sq = """
WITH cleaned AS (
  SELECT
    mmrd.customerId,mmrd.digitalLoanAccountId,prediction,start_time,end_time,modelDisplayName,modelVersionId,
  case when trenchCategory is null then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench1' end)
     when trenchCategory = '' then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench 1' end)
    else trenchCategory end  as trenchCategory,
    REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature,
    Data_selection,
    deviceOs osType,
   FROM prj-prod-dataplatform.dap_ds_poweruser_playground.ml_training_model_run_details_20260116 mmrd
  left join prj-prod-dataplatform.risk_credit_mis.model_loan_score_mart mt on mt.digitalLoanAccountId = mmrd.digitalLoanAccountId
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
  Data_selection,
  osType,
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
  Data_selection,
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
     coalesce((case when lower(r.osType) like '%andro%' then 'android'
                  when lower(r.osType) like '%os%' then 'ios' else lower(r.osType) end),
            (case when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%andro%' then 'android'
                  when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%os%' then 'ios'
                  when lower(loanmaster.deviceType) like '%andro%' then 'android'
                  else 'ios' end)
            ) as osType
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
  select * from base
  qualify row_number() over(partition by digitalLoanAccountId, modelVersionId order by appln_submit_datetime) = 1
  ;
  """
dfd = client.query(sq).to_dataframe()
# dfd = dfd.drop_duplicates(keep='first')
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()

# %%
df2 = dfd.copy()


# #### Concatenate both Test and Train Datasets


# 1) Get all IDs present in Train
train_ids = set(df2["digitalLoanAccountId"])

# 2) Keep only Test rows whose ID is NOT in Train
df1_no_dupes = df1[~df1["digitalLoanAccountId"].isin(train_ids)]

# 3) Concatenate
df_concat = pd.concat([df1_no_dupes, df2], ignore_index=True)

print(f"The shape of the concatenated dataframe is:\t {df_concat.shape}")
df_concat.head()


# #### Making sure the score column is numeric

# %%
df_concat["sil_beta_stack_score"] = pd.to_numeric(
    df_concat["sil_beta_stack_score"], errors="coerce"
)


# #### Calculating the Gini

# %%
fact_table, dimension_table = calculate_periodic_gini_prod_ver_trench_dimfact(
    df_concat,
    "sil_beta_stack_score",
    "deffstpd30",
    "FSTPD30",
    data_selection_column="Data_selection",
    model_version_column="modelVersionId",
    trench_column="trenchCategory",
    loan_type_column="new_loan_type",
    loan_product_type_column="loan_product_type",
    ostype_column="osType",
    account_id_column="digitalLoanAccountId",
)


# #### Updating the Fact and Dimension Tables

# %%
fact_table, dimension_table = update_tables(
    fact_table, dimension_table, model_name="beta_stack_model_sil", product="SIL"
)
print(f"The shape of the fact table is:\t {fact_table.shape}")
print(f"The shape of the dimension table is:\t {dimension_table.shape}")


df_f_fstpd30_betastackscoresil = fact_table.copy()
df_d_fstpd30_betastackscoresil = dimension_table.copy()

# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_f_fstpd30_betastackscoresil, facttable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_d_fstpd30_betastackscoresil, dimtable_id, job_config=job_config
)
job.result()  # Wait for the job to complete


factbetademosil = pd.concat(
    [
        df_f_fpd0_betastackscoresil,
        df_f_fpd10_betastackscoresil,
        df_f_fpd30_betastackscoresil,
        df_f_fspd30_betastackscoresil,
        df_f_fstpd30_betastackscoresil,
    ],
    ignore_index=False,
)
dimbetademosil = pd.concat(
    [
        df_d_fpd0_betastackscoresil,
        df_d_fpd10_betastackscoresil,
        df_d_fpd30_betastackscoresil,
        df_d_fspd30_betastackscoresil,
        df_d_fstpd30_betastackscoresil,
    ],
    ignore_index=False,
)

# %%

print(f"beta_stack_model_sil model Gini Calculation and Upload Completed!")


# %% [markdown]
# #### beta_stack_model_sil_credo_score

# %%
facttable_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_credosil"
dimtable_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_credosil"


# ## FPD0


# ## Test

# %%
sq = f"""with modelname as
  (  SELECT
    mmrd.customerId,mmrd.digitalLoanAccountId,prediction,start_time,end_time,modelDisplayName,modelVersionId,
  case when trenchCategory is null then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench1' end)
     when trenchCategory = '' then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench 1' end)
    else trenchCategory end  as trenchCategory,
    REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature,
        deviceOs osType,
    FROM prj-prod-dataplatform.audit_balance.ml_model_run_details mmrd
  left join prj-prod-dataplatform.risk_credit_mis.model_loan_score_mart mt on mt.digitalLoanAccountId = mmrd.digitalLoanAccountId
  WHERE modelDisplayName in ('Beta - StackScoreModel', 'beta_stack_model_sil')
  -- and modelVersionId = 'v1'
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
   coalesce(coalesce(cast(JSON_VALUE(SAFE.PARSE_JSON(CAST(calcFeature AS STRING)), '$.credo_score')AS FLOAT64),
  cast(JSON_VALUE(SAFE.PARSE_JSON(CAST(calcFeature AS STRING)), '$.s_credo_score')AS FLOAT64)),
  CAST(JSON_EXTRACT_SCALAR(calcFeature, '$.stackScoreModel.s_credo_score') AS FLOAT64))   AS credo_score,
  calcFeature,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month,
  'Prod' Data_selection,
  deffpd0,
  flg_mature_fpd0,
  loanmaster.new_loan_type,
  modelVersionId, trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile'
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall'
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
      coalesce((case when lower(r.osType) like '%andro%' then 'android'
                  when lower(r.osType) like '%os%' then 'ios' else lower(r.osType) end),
            (case when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%andro%' then 'android'
                  when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%os%' then 'ios'
                  when lower(loanmaster.deviceType) like '%andro%' then 'android'
                  else 'ios' end)
            ) as osType
  from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  left join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
   left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
  where
  loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and flg_mature_fpd0 = 1
  )
  select *
  from base
  where credo_score is not null
 qualify row_number() over(partition by digitalLoanAccountId, modelVersionId order by appln_submit_datetime) = 1
   ;
"""
dfd = client.query(sq).to_dataframe()
# dfd = dfd.drop_duplicates(keep='first')
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()


# %%
df1 = dfd.copy()


# ### Train

# %%
sq = """
with modelname as
  (  SELECT
    mmrd.customerId,mmrd.digitalLoanAccountId,prediction,start_time,end_time,modelDisplayName,modelVersionId,
  case when trenchCategory is null then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench1' end)
     when trenchCategory = '' then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench 1' end)
    else trenchCategory end  as trenchCategory,
    REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature,
    Data_selection,
        deviceOs osType,
    FROM prj-prod-dataplatform.dap_ds_poweruser_playground.ml_training_model_run_details_20260116 mmrd
  left join prj-prod-dataplatform.risk_credit_mis.model_loan_score_mart mt on mt.digitalLoanAccountId = mmrd.digitalLoanAccountId
  WHERE modelDisplayName in ('Beta - StackScoreModel', 'beta_stack_model_sil')
  -- and modelVersionId = 'v1'
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
    coalesce(coalesce(cast(JSON_VALUE(SAFE.PARSE_JSON(CAST(calcFeature AS STRING)), '$.credo_score')AS FLOAT64),
  cast(JSON_VALUE(SAFE.PARSE_JSON(CAST(calcFeature AS STRING)), '$.s_credo_score')AS FLOAT64))
  , cast(JSON_VALUE(SAFE.PARSE_JSON(CAST(calcFeature AS STRING)), '$.credo_gen_score')AS FLOAT64))
   AS credo_score,
  calcFeature,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month,
  Data_selection,
  deffpd0,
  flg_mature_fpd0,
  loanmaster.new_loan_type,
  modelVersionId, trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile'
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall'
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
        coalesce((case when lower(r.osType) like '%andro%' then 'android'
                  when lower(r.osType) like '%os%' then 'ios' else lower(r.osType) end),
            (case when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%andro%' then 'android'
                  when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%os%' then 'ios'
                  when lower(loanmaster.deviceType) like '%andro%' then 'android'
                  else 'ios' end)
            ) as osType
  from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  left join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
   left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
  where
  loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and flg_mature_fpd0 = 1
  )
  select *
  from base
  where credo_score is not null
 qualify row_number() over(partition by digitalLoanAccountId, modelVersionId order by appln_submit_datetime) = 1
   ;
  """
dfd = client.query(sq).to_dataframe()
# dfd = dfd.drop_duplicates(keep='first')
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()

# %%
df2 = dfd.copy()


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
# df_concat = df1.copy()

df_concat["credo_score"] = pd.to_numeric(df_concat["credo_score"], errors="coerce")

# %%
fact_table, dimension_table = calculate_periodic_gini_prod_ver_trench_dimfact(
    df_concat,
    "credo_score",
    "deffpd0",
    "FPD0",
    data_selection_column="Data_selection",
    model_version_column="modelVersionId",
    trench_column="trenchCategory",
    loan_type_column="new_loan_type",
    loan_product_type_column="loan_product_type",
    ostype_column="osType",
    account_id_column="digitalLoanAccountId",
)

# %%
fact_table, dimension_table = update_tables(
    fact_table, dimension_table, model_name="beta_stack_credo_score_sil", product="SIL"
)
print(f"The shape of the fact table is:\t {fact_table.shape}")
print(f"The shape of the dimension table is:\t {dimension_table.shape}")

# %%
df_f_fpd0_betastackcredoscoresil = fact_table.copy()
df_d_fpd0_betastackcredoscoresil = dimension_table.copy()

# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_TRUNCATE",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_f_fpd0_betastackcredoscoresil, facttable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_TRUNCATE",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_d_fpd0_betastackcredoscoresil, dimtable_id, job_config=job_config
)
job.result()  # Wait for the job to complete


# ## FPD10


# ## Test

# %%
sq = f"""with modelname as
  (  SELECT
    mmrd.customerId,mmrd.digitalLoanAccountId,prediction,start_time,end_time,modelDisplayName,modelVersionId,
  case when trenchCategory is null then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench1' end)
     when trenchCategory = '' then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench 1' end)
    else trenchCategory end  as trenchCategory,
    REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature,
        deviceOs osType,

    FROM prj-prod-dataplatform.audit_balance.ml_model_run_details mmrd
  left join prj-prod-dataplatform.risk_credit_mis.model_loan_score_mart mt on mt.digitalLoanAccountId = mmrd.digitalLoanAccountId
  WHERE modelDisplayName in ('Beta - StackScoreModel', 'beta_stack_model_sil')
  -- and modelVersionId = 'v1'
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
    coalesce(coalesce(cast(JSON_VALUE(SAFE.PARSE_JSON(CAST(calcFeature AS STRING)), '$.credo_score')AS FLOAT64),
  cast(JSON_VALUE(SAFE.PARSE_JSON(CAST(calcFeature AS STRING)), '$.s_credo_score')AS FLOAT64)),
  CAST(JSON_EXTRACT_SCALAR(calcFeature, '$.stackScoreModel.s_credo_score') AS FLOAT64))   AS credo_score,
  calcFeature,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month,
  'Prod' Data_selection,
  del.deffpd10,
  del.flg_mature_fpd10,
  loanmaster.new_loan_type,
  modelVersionId, trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile'
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall'
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
        coalesce((case when lower(r.osType) like '%andro%' then 'android'
                  when lower(r.osType) like '%os%' then 'ios' else lower(r.osType) end),
            (case when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%andro%' then 'android'
                  when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%os%' then 'ios'
                  when lower(loanmaster.deviceType) like '%andro%' then 'android'
                  else 'ios' end)
            ) as osType
  from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  left join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
   left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
  where
  loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and del.flg_mature_fpd10 = 1
  )
  select *
  from base
  where credo_score is not null
 qualify row_number() over(partition by digitalLoanAccountId, modelVersionId order by appln_submit_datetime) = 1
   ;
"""
dfd = client.query(sq).to_dataframe()
# dfd = dfd.drop_duplicates(keep='first')
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()


# %%
df1 = dfd.copy()


# ### Train

# %%
sq = """
with modelname as
  (  SELECT
    mmrd.customerId,mmrd.digitalLoanAccountId,prediction,start_time,end_time,modelDisplayName,modelVersionId,
  case when trenchCategory is null then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench1' end)
     when trenchCategory = '' then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench 1' end)
    else trenchCategory end  as trenchCategory,
    REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature,
    Data_selection,
        deviceOs osType,
    FROM prj-prod-dataplatform.dap_ds_poweruser_playground.ml_training_model_run_details_20260116 mmrd
  left join prj-prod-dataplatform.risk_credit_mis.model_loan_score_mart mt on mt.digitalLoanAccountId = mmrd.digitalLoanAccountId
  WHERE modelDisplayName in ('Beta - StackScoreModel', 'beta_stack_model_sil')
  -- and modelVersionId = 'v1'
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
  coalesce(coalesce(cast(JSON_VALUE(SAFE.PARSE_JSON(CAST(calcFeature AS STRING)), '$.credo_score')AS FLOAT64),
  cast(JSON_VALUE(SAFE.PARSE_JSON(CAST(calcFeature AS STRING)), '$.s_credo_score')AS FLOAT64))
  , cast(JSON_VALUE(SAFE.PARSE_JSON(CAST(calcFeature AS STRING)), '$.credo_gen_score')AS FLOAT64))
   AS credo_score,
  calcFeature,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month,
  Data_selection,
  del.deffpd10,
  del.flg_mature_fpd10,
  loanmaster.new_loan_type,
  modelVersionId, trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile'
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall'
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
        coalesce((case when lower(r.osType) like '%andro%' then 'android'
                  when lower(r.osType) like '%os%' then 'ios' else lower(r.osType) end),
            (case when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%andro%' then 'android'
                  when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%os%' then 'ios'
                  when lower(loanmaster.deviceType) like '%andro%' then 'android'
                  else 'ios' end)
            ) as osType
  from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  left join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
   left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
  where
  loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and del.flg_mature_fpd0 = 1
  )
  select *
  from base
  where credo_score is not null
 qualify row_number() over(partition by digitalLoanAccountId, modelVersionId order by appln_submit_datetime) = 1
   ;
  """
dfd = client.query(sq).to_dataframe()
# dfd = dfd.drop_duplicates(keep='first')
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()

# %%
df2 = dfd.copy()


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
# df_concat = df1.copy()

df_concat["credo_score"] = pd.to_numeric(df_concat["credo_score"], errors="coerce")

# %%
fact_table, dimension_table = calculate_periodic_gini_prod_ver_trench_dimfact(
    df_concat,
    "credo_score",
    "deffpd10",
    "FPD10",
    data_selection_column="Data_selection",
    model_version_column="modelVersionId",
    trench_column="trenchCategory",
    loan_type_column="new_loan_type",
    loan_product_type_column="loan_product_type",
    ostype_column="osType",
    account_id_column="digitalLoanAccountId",
)

# %%
fact_table, dimension_table = update_tables(
    fact_table, dimension_table, model_name="beta_stack_credo_score_sil", product="SIL"
)
print(f"The shape of the fact table is:\t {fact_table.shape}")
print(f"The shape of the dimension table is:\t {dimension_table.shape}")

df_f_fpd10_betastackcredoscoresil = fact_table.copy()
df_d_fpd10_betastackcredoscoresil = dimension_table.copy()

# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_f_fpd10_betastackcredoscoresil, facttable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_d_fpd10_betastackcredoscoresil, dimtable_id, job_config=job_config
)
job.result()  # Wait for the job to complete


# ## FPD30


# ## Test

# %%
sq = f"""with modelname as
  (  SELECT
    mmrd.customerId,mmrd.digitalLoanAccountId,prediction,start_time,end_time,modelDisplayName,modelVersionId,
  case when trenchCategory is null then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench1' end)
     when trenchCategory = '' then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench 1' end)
    else trenchCategory end  as trenchCategory,
    REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature,
       deviceOs osType,
    FROM prj-prod-dataplatform.audit_balance.ml_model_run_details mmrd
  left join prj-prod-dataplatform.risk_credit_mis.model_loan_score_mart mt on mt.digitalLoanAccountId = mmrd.digitalLoanAccountId
  WHERE modelDisplayName in ('Beta - StackScoreModel', 'beta_stack_model_sil')
  -- and modelVersionId = 'v1'
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
  coalesce(coalesce(cast(JSON_VALUE(SAFE.PARSE_JSON(CAST(calcFeature AS STRING)), '$.credo_score')AS FLOAT64),
  cast(JSON_VALUE(SAFE.PARSE_JSON(CAST(calcFeature AS STRING)), '$.s_credo_score')AS FLOAT64)),
  CAST(JSON_EXTRACT_SCALAR(calcFeature, '$.stackScoreModel.s_credo_score') AS FLOAT64))   AS credo_score,
  calcFeature,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month,
  'Prod' Data_selection,
  del.deffpd30,
  del.flg_mature_fpd30,
  loanmaster.new_loan_type,
  modelVersionId, trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile'
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall'
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
        coalesce((case when lower(r.osType) like '%andro%' then 'android'
                  when lower(r.osType) like '%os%' then 'ios' else lower(r.osType) end),
            (case when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%andro%' then 'android'
                  when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%os%' then 'ios'
                  when lower(loanmaster.deviceType) like '%andro%' then 'android'
                  else 'ios' end)
            ) as osType
  from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  left join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
   left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
  where
  loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and del.flg_mature_fpd30 = 1
  )
  select *
  from base
  where credo_score is not null
 qualify row_number() over(partition by digitalLoanAccountId, modelVersionId order by appln_submit_datetime) = 1
   ;
"""
dfd = client.query(sq).to_dataframe()
# dfd = dfd.drop_duplicates(keep='first')
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()


# %%
df1 = dfd.copy()


# ### Train

# %%
sq = """
with modelname as
  (  SELECT
    mmrd.customerId,mmrd.digitalLoanAccountId,prediction,start_time,end_time,modelDisplayName,modelVersionId,
  case when trenchCategory is null then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench1' end)
     when trenchCategory = '' then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench 1' end)
    else trenchCategory end  as trenchCategory,
    REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature,
    Data_selection,
        deviceOs osType,
    FROM prj-prod-dataplatform.dap_ds_poweruser_playground.ml_training_model_run_details_20260116 mmrd
  left join prj-prod-dataplatform.risk_credit_mis.model_loan_score_mart mt on mt.digitalLoanAccountId = mmrd.digitalLoanAccountId
  WHERE modelDisplayName in ('Beta - StackScoreModel', 'beta_stack_model_sil')
  -- and modelVersionId = 'v1'
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
  coalesce(coalesce(cast(JSON_VALUE(SAFE.PARSE_JSON(CAST(calcFeature AS STRING)), '$.credo_score')AS FLOAT64),
  cast(JSON_VALUE(SAFE.PARSE_JSON(CAST(calcFeature AS STRING)), '$.s_credo_score')AS FLOAT64))
  , cast(JSON_VALUE(SAFE.PARSE_JSON(CAST(calcFeature AS STRING)), '$.credo_gen_score')AS FLOAT64))
   AS credo_score,
  calcFeature,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month,
  Data_selection,
  del.deffpd30,
  del.flg_mature_fpd30,
  loanmaster.new_loan_type,
  modelVersionId, trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile'
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall'
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
        coalesce((case when lower(r.osType) like '%andro%' then 'android'
                  when lower(r.osType) like '%os%' then 'ios' else lower(r.osType) end),
            (case when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%andro%' then 'android'
                  when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%os%' then 'ios'
                  when lower(loanmaster.deviceType) like '%andro%' then 'android'
                  else 'ios' end)
            ) as osType
  from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  left join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
   left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
  where
  loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and del.flg_mature_fpd30 = 1
  )
  select *
  from base
  where credo_score is not null
 qualify row_number() over(partition by digitalLoanAccountId, modelVersionId order by appln_submit_datetime) = 1
   ;
  """
dfd = client.query(sq).to_dataframe()
# dfd = dfd.drop_duplicates(keep='first')
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()

# %%
df2 = dfd.copy()


# ### Concat

# %%
df_concat = pd.concat([df1, df2], ignore_index=True)
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
# df_concat = df1.copy()

df_concat["credo_score"] = pd.to_numeric(df_concat["credo_score"], errors="coerce")

# %%
fact_table, dimension_table = calculate_periodic_gini_prod_ver_trench_dimfact(
    df_concat,
    "credo_score",
    "deffpd30",
    "FPD30",
    data_selection_column="Data_selection",
    model_version_column="modelVersionId",
    trench_column="trenchCategory",
    loan_type_column="new_loan_type",
    loan_product_type_column="loan_product_type",
    ostype_column="osType",
    account_id_column="digitalLoanAccountId",
)

# %%
fact_table, dimension_table = update_tables(
    fact_table, dimension_table, model_name="beta_stack_credo_score_sil", product="SIL"
)
print(f"The shape of the fact table is:\t {fact_table.shape}")
print(f"The shape of the dimension table is:\t {dimension_table.shape}")

# %%
df_f_fpd30_betastackcredoscoresil = fact_table.copy()
df_d_fpd30_betastackcredoscoresil = dimension_table.copy()

# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_f_fpd30_betastackcredoscoresil, facttable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_d_fpd30_betastackcredoscoresil, dimtable_id, job_config=job_config
)
job.result()  # Wait for the job to complete


# ## FSPD30


# ## Test

# %%
sq = f"""with modelname as
  (  SELECT
    mmrd.customerId,mmrd.digitalLoanAccountId,prediction,start_time,end_time,modelDisplayName,modelVersionId,
  case when trenchCategory is null then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench1' end)
     when trenchCategory = '' then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench 1' end)
    else trenchCategory end  as trenchCategory,
    REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature,
        deviceOs osType,
    FROM prj-prod-dataplatform.audit_balance.ml_model_run_details mmrd
  left join prj-prod-dataplatform.risk_credit_mis.model_loan_score_mart mt on mt.digitalLoanAccountId = mmrd.digitalLoanAccountId
  WHERE modelDisplayName in ('Beta - StackScoreModel', 'beta_stack_model_sil')
  -- and modelVersionId = 'v1'
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
  coalesce(coalesce(cast(JSON_VALUE(SAFE.PARSE_JSON(CAST(calcFeature AS STRING)), '$.credo_score')AS FLOAT64),
  cast(JSON_VALUE(SAFE.PARSE_JSON(CAST(calcFeature AS STRING)), '$.s_credo_score')AS FLOAT64)),
  CAST(JSON_EXTRACT_SCALAR(calcFeature, '$.stackScoreModel.s_credo_score') AS FLOAT64))   AS credo_score,
  calcFeature,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month,
  'Prod' Data_selection,
  del.deffspd30,
  del.flg_mature_fspd_30,
  loanmaster.new_loan_type,
  modelVersionId, trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile'
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall'
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
        coalesce((case when lower(r.osType) like '%andro%' then 'android'
                  when lower(r.osType) like '%os%' then 'ios' else lower(r.osType) end),
            (case when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%andro%' then 'android'
                  when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%os%' then 'ios'
                  when lower(loanmaster.deviceType) like '%andro%' then 'android'
                  else 'ios' end)
            ) as osType
  from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  left join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
   left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
  where
  loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and del.flg_mature_fspd_30 = 1
  )
  select *
  from base
  where credo_score is not null
 qualify row_number() over(partition by digitalLoanAccountId, modelVersionId order by appln_submit_datetime) = 1
   ;
"""
dfd = client.query(sq).to_dataframe()
# dfd = dfd.drop_duplicates(keep='first')
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()


# %%
df1 = dfd.copy()


# ### Train

# %%
sq = """
with modelname as
  (  SELECT
    mmrd.customerId,mmrd.digitalLoanAccountId,prediction,start_time,end_time,modelDisplayName,modelVersionId,
  case when trenchCategory is null then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench1' end)
     when trenchCategory = '' then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench 1' end)
    else trenchCategory end  as trenchCategory,
    REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature,
    Data_selection,
      deviceOs osType,
    FROM prj-prod-dataplatform.dap_ds_poweruser_playground.ml_training_model_run_details_20260116 mmrd
  left join prj-prod-dataplatform.risk_credit_mis.model_loan_score_mart mt on mt.digitalLoanAccountId = mmrd.digitalLoanAccountId
  WHERE modelDisplayName in ('Beta - StackScoreModel', 'beta_stack_model_sil')
  -- and modelVersionId = 'v1'
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
  coalesce(coalesce(cast(JSON_VALUE(SAFE.PARSE_JSON(CAST(calcFeature AS STRING)), '$.credo_score')AS FLOAT64),
  cast(JSON_VALUE(SAFE.PARSE_JSON(CAST(calcFeature AS STRING)), '$.s_credo_score')AS FLOAT64))
  , cast(JSON_VALUE(SAFE.PARSE_JSON(CAST(calcFeature AS STRING)), '$.credo_gen_score')AS FLOAT64))
   AS credo_score,
  calcFeature,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month,
  Data_selection,
  deffspd30,
  flg_mature_fspd_30,
  loanmaster.new_loan_type,
  modelVersionId, trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile'
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall'
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
        coalesce((case when lower(r.osType) like '%andro%' then 'android'
                  when lower(r.osType) like '%os%' then 'ios' else lower(r.osType) end),
            (case when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%andro%' then 'android'
                  when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%os%' then 'ios'
                  when lower(loanmaster.deviceType) like '%andro%' then 'android'
                  else 'ios' end)
            ) as osType
  from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  left join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
   left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
  where
  loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and flg_mature_fspd_30 = 1
  )
  select *
  from base
  where credo_score is not null
 qualify row_number() over(partition by digitalLoanAccountId, modelVersionId order by appln_submit_datetime) = 1
   ;
  """
dfd = client.query(sq).to_dataframe()
# dfd = dfd.drop_duplicates(keep='first')
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()

# %%
df2 = dfd.copy()


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
# df_concat = df1.copy()

df_concat["credo_score"] = pd.to_numeric(df_concat["credo_score"], errors="coerce")

# %%
fact_table, dimension_table = calculate_periodic_gini_prod_ver_trench_dimfact(
    df_concat,
    "credo_score",
    "deffspd30",
    "FSPD30",
    data_selection_column="Data_selection",
    model_version_column="modelVersionId",
    trench_column="trenchCategory",
    loan_type_column="new_loan_type",
    loan_product_type_column="loan_product_type",
    ostype_column="osType",
    account_id_column="digitalLoanAccountId",
)

# %%
fact_table, dimension_table = update_tables(
    fact_table, dimension_table, model_name="beta_stack_credo_score_sil", product="SIL"
)
print(f"The shape of the fact table is:\t {fact_table.shape}")
print(f"The shape of the dimension table is:\t {dimension_table.shape}")

df_f_fspd30_betastackcredoscoresil = fact_table.copy()
df_d_fspd30_betastackcredoscoresil = dimension_table.copy()


# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_f_fspd30_betastackcredoscoresil, facttable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_d_fspd30_betastackcredoscoresil, dimtable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# ## FSTPD30


# ## Test

# %%
sq = f"""with modelname as
  (  SELECT
    mmrd.customerId,mmrd.digitalLoanAccountId,prediction,start_time,end_time,modelDisplayName,modelVersionId,
  case when trenchCategory is null then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench1' end)
     when trenchCategory = '' then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench 1' end)
    else trenchCategory end  as trenchCategory,
    REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature,
        deviceOs osType,
    FROM prj-prod-dataplatform.audit_balance.ml_model_run_details mmrd
  left join prj-prod-dataplatform.risk_credit_mis.model_loan_score_mart mt on mt.digitalLoanAccountId = mmrd.digitalLoanAccountId
  WHERE modelDisplayName in ('Beta - StackScoreModel', 'beta_stack_model_sil')
  -- and modelVersionId = 'v1'
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
  coalesce(coalesce(cast(JSON_VALUE(SAFE.PARSE_JSON(CAST(calcFeature AS STRING)), '$.credo_score')AS FLOAT64),
  cast(JSON_VALUE(SAFE.PARSE_JSON(CAST(calcFeature AS STRING)), '$.s_credo_score')AS FLOAT64)),
  CAST(JSON_EXTRACT_SCALAR(calcFeature, '$.stackScoreModel.s_credo_score') AS FLOAT64))   AS credo_score,
  calcFeature,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month,
  'Prod' Data_selection,
  del.deffstpd30,
  del.flg_mature_fstpd_30,
  loanmaster.new_loan_type,
  modelVersionId, trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile'
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall'
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
        coalesce((case when lower(r.osType) like '%andro%' then 'android'
                  when lower(r.osType) like '%os%' then 'ios' else lower(r.osType) end),
            (case when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%andro%' then 'android'
                  when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%os%' then 'ios'
                  when lower(loanmaster.deviceType) like '%andro%' then 'android'
                  else 'ios' end)
            ) as osType
  from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  left join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
   left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
  where
  loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and del.flg_mature_fstpd_30 = 1
  )
  select *
  from base
  where credo_score is not null
 qualify row_number() over(partition by digitalLoanAccountId, modelVersionId order by appln_submit_datetime) = 1
   ;
"""
dfd = client.query(sq).to_dataframe()
# dfd = dfd.drop_duplicates(keep='first')
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()


# %%
df1 = dfd.copy()


# ### Train

# %%
sq = """
with modelname as
  (  SELECT
    mmrd.customerId,mmrd.digitalLoanAccountId,prediction,start_time,end_time,modelDisplayName,modelVersionId,
  case when trenchCategory is null then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench1' end)
     when trenchCategory = '' then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench 1' end)
    else trenchCategory end  as trenchCategory,
    REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature,
    Data_selection,
        deviceOs osType,
    FROM prj-prod-dataplatform.dap_ds_poweruser_playground.ml_training_model_run_details_20260116 mmrd
  left join prj-prod-dataplatform.risk_credit_mis.model_loan_score_mart mt on mt.digitalLoanAccountId = mmrd.digitalLoanAccountId
  WHERE modelDisplayName in ('Beta - StackScoreModel', 'beta_stack_model_sil')
  -- and modelVersionId = 'v1'
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
  coalesce(coalesce(cast(JSON_VALUE(SAFE.PARSE_JSON(CAST(calcFeature AS STRING)), '$.credo_score')AS FLOAT64),
  cast(JSON_VALUE(SAFE.PARSE_JSON(CAST(calcFeature AS STRING)), '$.s_credo_score')AS FLOAT64))
  , cast(JSON_VALUE(SAFE.PARSE_JSON(CAST(calcFeature AS STRING)), '$.credo_gen_score')AS FLOAT64))
   AS credo_score,
  calcFeature,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month,
  Data_selection,
  deffstpd30,
  flg_mature_fstpd_30,
  loanmaster.new_loan_type,
  modelVersionId, trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile'
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall'
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
        coalesce((case when lower(r.osType) like '%andro%' then 'android'
                  when lower(r.osType) like '%os%' then 'ios' else lower(r.osType) end),
            (case when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%andro%' then 'android'
                  when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%os%' then 'ios'
                  when lower(loanmaster.deviceType) like '%andro%' then 'android'
                  else 'ios' end)
            ) as osType
  from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  left join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
   left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
  where
  loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and flg_mature_fstpd_30 = 1
  )
  select *
  from base
  where credo_score is not null
 qualify row_number() over(partition by digitalLoanAccountId, modelVersionId order by appln_submit_datetime) = 1
   ;
  """
dfd = client.query(sq).to_dataframe()
# dfd = dfd.drop_duplicates(keep='first')
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()

# %%
df2 = dfd.copy()


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
# df_concat = df1.copy()

df_concat["credo_score"] = pd.to_numeric(df_concat["credo_score"], errors="coerce")

# %%
fact_table, dimension_table = calculate_periodic_gini_prod_ver_trench_dimfact(
    df_concat,
    "credo_score",
    "deffstpd30",
    "FSTPD30",
    data_selection_column="Data_selection",
    model_version_column="modelVersionId",
    trench_column="trenchCategory",
    loan_type_column="new_loan_type",
    loan_product_type_column="loan_product_type",
    ostype_column="osType",
    account_id_column="digitalLoanAccountId",
)

# %%
fact_table, dimension_table = update_tables(
    fact_table, dimension_table, model_name="beta_stack_credo_score_sil", product="SIL"
)
print(f"The shape of the fact table is:\t {fact_table.shape}")
print(f"The shape of the dimension table is:\t {dimension_table.shape}")

df_f_fstpd30_betastackcredoscoresil = fact_table.copy()
df_d_fstpd30_betastackcredoscoresil = dimension_table.copy()
# %%

# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_f_fstpd30_betastackcredoscoresil, facttable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_d_fstpd30_betastackcredoscoresil, dimtable_id, job_config=job_config
)
job.result()  # Wait for the job to complete


factbetacredosil = pd.concat(
    [
        df_f_fpd0_betastackcredoscoresil,
        df_f_fpd10_betastackcredoscoresil,
        df_f_fpd30_betastackcredoscoresil,
        df_f_fspd30_betastackcredoscoresil,
        df_f_fstpd30_betastackcredoscoresil,
    ],
    ignore_index=True,
)
dimbetacredosil = pd.concat(
    [
        df_d_fpd0_betastackcredoscoresil,
        df_d_fpd10_betastackcredoscoresil,
        df_d_fpd30_betastackcredoscoresil,
        df_d_fspd30_betastackcredoscoresil,
        df_d_fstpd30_betastackcredoscoresil,
    ],
    ignore_index=True,
)

print(f"beta_stack_credo_score_sil model Gini Calculation and Upload Completed!")


# %% [markdown]
# #### Alpha-Cash-CIC-Model

# %%
facttable_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_alphaciccash"
dimtable_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_alphaciccash"

# ### FPD0

# ### Test

# %%
sq = r"""
WITH parsed as (
select customerId, digitalLoanAccountId,modelDisplayName,modelVersionId,start_time,end_time,prediction,trenchCategory,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeatures,
    deviceOs osType,
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
  coalesce (p.trenchCategory, REGEXP_EXTRACT(m.requestPayload_clean, r"trenchCategory[:=]['\"]?([^'\"]+)['\"]?")) trenchCategory,
      osType,
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
  'Prod' Data_selection,
  deffpd0,
  flg_mature_fpd0,
  loanmaster.new_loan_type,
  modelVersionId, trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile'
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall'
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
        coalesce((case when lower(r.osType) like '%andro%' then 'android'
                  when lower(r.osType) like '%os%' then 'ios' else lower(r.osType) end),
            (case when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%andro%' then 'android'
                  when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%os%' then 'ios'
                  when lower(loanmaster.deviceType) like '%andro%' then 'android'
                  else 'ios' end)
            ) as osType
  from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  inner join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
   left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
  where loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and r.aCicScore is not null
  and del.flg_mature_fpd0 = 1
  )
  select *  from base
  qualify row_number() over(partition by digitalLoanAccountId, modelVersionId order by appln_submit_datetime) = 1
  ;
"""
dfd = client.query(sq).to_dataframe()
# dfd = dfd.drop_duplicates(keep='first')
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()


# %%
df1 = dfd.copy()


# ### Train

# %%
sq = """
WITH parsed as (
select customerId, digitalLoanAccountId,modelDisplayName,modelVersionId,start_time,end_time,prediction,trenchCategory,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeatures,
Data_selection,
 deviceOs osType,
FROM prj-prod-dataplatform.dap_ds_poweruser_playground.ml_training_model_run_details_20260116
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
  trenchCategory,
  Data_selection,
   osType,
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
  Data_selection,
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
        coalesce((case when lower(r.osType) like '%andro%' then 'android'
                  when lower(r.osType) like '%os%' then 'ios' else lower(r.osType) end),
            (case when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%andro%' then 'android'
                  when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%os%' then 'ios'
                  when lower(loanmaster.deviceType) like '%andro%' then 'android'
                  else 'ios' end)
            ) as osType
    from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  inner join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
  left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
  where loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and r.aCicScore is not null
  and del.flg_mature_fpd0 = 1
  )
  select * from base
  qualify row_number() over(partition by digitalLoanAccountId, modelVersionId order by appln_submit_datetime) = 1
  ;
  """
dfd = client.query(sq).to_dataframe()
# dfd = dfd.drop_duplicates(keep='first')
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()

# %%
df2 = dfd.copy()


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
    data_selection_column="Data_selection",
    model_version_column="modelVersionId",
    trench_column="trenchCategory",
    loan_type_column="new_loan_type",
    loan_product_type_column="loan_product_type",
    ostype_column="osType",
    account_id_column="digitalLoanAccountId",
)

# %%
fact_table, dimension_table = update_tables(
    fact_table, dimension_table, model_name="cic_model_cash", product="CASH"
)
print(f"The shape of the fact table is:\t {fact_table.shape}")
print(f"The shape of the dimension table is:\t {dimension_table.shape}")

df_f_fpd0_ciccash = fact_table.copy()
df_d_fpd0_ciccash = dimension_table.copy()

# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_TRUNCATE",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_f_fpd0_ciccash, facttable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_TRUNCATE",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_d_fpd0_ciccash, dimtable_id, job_config=job_config
)
job.result()  # Wait for the job to complete


# ### FPD10

# ### Test

# %%
sq = r"""
WITH parsed as (
select customerId, digitalLoanAccountId,modelDisplayName,modelVersionId,start_time,end_time,prediction,trenchCategory,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeatures,
    deviceOs osType,
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
  coalesce (p.trenchCategory, REGEXP_EXTRACT(m.requestPayload_clean, r"trenchCategory[:=]['\"]?([^'\"]+)['\"]?")) trenchCategory,
      osType,
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
  'Prod' Data_selection,
  del.deffpd10,
  del.flg_mature_fpd10,
  loanmaster.new_loan_type,
  modelVersionId, trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile'
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall'
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
        coalesce((case when lower(r.osType) like '%andro%' then 'android'
                  when lower(r.osType) like '%os%' then 'ios' else lower(r.osType) end),
            (case when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%andro%' then 'android'
                  when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%os%' then 'ios'
                  when lower(loanmaster.deviceType) like '%andro%' then 'android'
                  else 'ios' end)
            ) as osType
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
  qualify row_number() over(partition by digitalLoanAccountId, modelVersionId order by appln_submit_datetime) = 1
  ;
"""
dfd = client.query(sq).to_dataframe()
# dfd = dfd.drop_duplicates(keep='first')
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()


# %%
df1 = dfd.copy()


# ### Train

# %%
sq = """
WITH parsed as (
select customerId, digitalLoanAccountId,modelDisplayName,modelVersionId,start_time,end_time,prediction,trenchCategory,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeatures,
Data_selection,
    deviceOs osType,
FROM prj-prod-dataplatform.dap_ds_poweruser_playground.ml_training_model_run_details_20260116
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
  trenchCategory,
  Data_selection,
      osType,
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
  Data_selection,
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
        coalesce((case when lower(r.osType) like '%andro%' then 'android'
                  when lower(r.osType) like '%os%' then 'ios' else lower(r.osType) end),
            (case when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%andro%' then 'android'
                  when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%os%' then 'ios'
                  when lower(loanmaster.deviceType) like '%andro%' then 'android'
                  else 'ios' end)
            ) as osType
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
  select * from base
  qualify row_number() over(partition by digitalLoanAccountId, modelVersionId order by appln_submit_datetime) = 1
  ;
  """
dfd = client.query(sq).to_dataframe()
# dfd = dfd.drop_duplicates(keep='first')
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()

# %%
df2 = dfd.copy()


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
    data_selection_column="Data_selection",
    model_version_column="modelVersionId",
    trench_column="trenchCategory",
    loan_type_column="new_loan_type",
    loan_product_type_column="loan_product_type",
    ostype_column="osType",
    account_id_column="digitalLoanAccountId",
)

# %%
fact_table, dimension_table = update_tables(
    fact_table, dimension_table, model_name="cic_model_cash", product="CASH"
)

df_f_fpd10_ciccash = fact_table.copy()
df_d_fpd10_ciccash = dimension_table.copy()

# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_f_fpd10_ciccash, facttable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_d_fpd10_ciccash, dimtable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# %%


# ### FPD30


# ### Test

# %%
sq = r"""
WITH parsed as (
select customerId, digitalLoanAccountId,modelDisplayName,modelVersionId,start_time,end_time,prediction,trenchCategory,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeatures,
   deviceOs osType,
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
  coalesce (p.trenchCategory, REGEXP_EXTRACT(m.requestPayload_clean, r"trenchCategory[:=]['\"]?([^'\"]+)['\"]?")) trenchCategory,
     osType,
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
  'Prod' Data_selection,
  del.deffpd30,
  del.flg_mature_fpd30,
  loanmaster.new_loan_type,
  modelVersionId, trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile'
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall'
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
        coalesce((case when lower(r.osType) like '%andro%' then 'android'
                  when lower(r.osType) like '%os%' then 'ios' else lower(r.osType) end),
            (case when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%andro%' then 'android'
                  when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%os%' then 'ios'
                  when lower(loanmaster.deviceType) like '%andro%' then 'android'
                  else 'ios' end)
            ) as osType
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
  qualify row_number() over(partition by digitalLoanAccountId, modelVersionId order by appln_submit_datetime) = 1
  ;
"""
dfd = client.query(sq).to_dataframe()
# dfd = dfd.drop_duplicates(keep='first')
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()


# %%
df1 = dfd.copy()


# ### Train

# %%
sq = """
WITH parsed as (
select customerId, digitalLoanAccountId,modelDisplayName,modelVersionId,start_time,end_time,prediction,trenchCategory,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeatures,
Data_selection,
    deviceOs osType,
FROM prj-prod-dataplatform.dap_ds_poweruser_playground.ml_training_model_run_details_20260116
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
  trenchCategory,
  Data_selection,
      osType,
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
  Data_selection,
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
        coalesce((case when lower(r.osType) like '%andro%' then 'android'
                  when lower(r.osType) like '%os%' then 'ios' else lower(r.osType) end),
            (case when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%andro%' then 'android'
                  when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%os%' then 'ios'
                  when lower(loanmaster.deviceType) like '%andro%' then 'android'
                  else 'ios' end)
            ) as osType
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
  select * from base
  qualify row_number() over(partition by digitalLoanAccountId, modelVersionId order by appln_submit_datetime) = 1
  ;
  """
dfd = client.query(sq).to_dataframe()
# dfd = dfd.drop_duplicates(keep='first')
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()

# %%
df2 = dfd.copy()


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
    data_selection_column="Data_selection",
    model_version_column="modelVersionId",
    trench_column="trenchCategory",
    loan_type_column="new_loan_type",
    loan_product_type_column="loan_product_type",
    ostype_column="osType",
    account_id_column="digitalLoanAccountId",
)

# %%
fact_table, dimension_table = update_tables(
    fact_table, dimension_table, model_name="cic_model_cash", product="CASH"
)

df_f_fpd30_ciccash = fact_table.copy()
df_d_fpd30_ciccash = dimension_table.copy()

# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_f_fpd30_ciccash, facttable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_d_fpd30_ciccash, dimtable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# %%


# ### FSPD30


# ### Test

# %%
sq = r"""
WITH parsed as (
select customerId, digitalLoanAccountId,modelDisplayName,modelVersionId,start_time,end_time,prediction,trenchCategory,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeatures,
    deviceOs osType,
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
  coalesce (p.trenchCategory, REGEXP_EXTRACT(m.requestPayload_clean, r"trenchCategory[:=]['\"]?([^'\"]+)['\"]?")) trenchCategory,
      osType,
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
  'Prod' Data_selection,
  del.deffspd30,
  del.flg_mature_fspd_30,
  loanmaster.new_loan_type,
  modelVersionId, trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile'
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall'
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
        coalesce((case when lower(r.osType) like '%andro%' then 'android'
                  when lower(r.osType) like '%os%' then 'ios' else lower(r.osType) end),
            (case when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%andro%' then 'android'
                  when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%os%' then 'ios'
                  when lower(loanmaster.deviceType) like '%andro%' then 'android'
                  else 'ios' end)
            ) as osType
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
  qualify row_number() over(partition by digitalLoanAccountId, modelVersionId order by appln_submit_datetime) = 1
  ;
"""
dfd = client.query(sq).to_dataframe()
# dfd = dfd.drop_duplicates(keep='first')
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()


# %%
df1 = dfd.copy()


# ### Train

# %%
sq = """
WITH parsed as (
select customerId, digitalLoanAccountId,modelDisplayName,modelVersionId,start_time,end_time,prediction,trenchCategory,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeatures,
Data_selection,
    deviceOs osType,
FROM prj-prod-dataplatform.dap_ds_poweruser_playground.ml_training_model_run_details_20260116
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
  trenchCategory,
  Data_selection,
      osType,
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
  Data_selection,
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
        coalesce((case when lower(r.osType) like '%andro%' then 'android'
                  when lower(r.osType) like '%os%' then 'ios' else lower(r.osType) end),
            (case when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%andro%' then 'android'
                  when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%os%' then 'ios'
                  when lower(loanmaster.deviceType) like '%andro%' then 'android'
                  else 'ios' end)
            ) as osType
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
  select * from base
  qualify row_number() over(partition by digitalLoanAccountId, modelVersionId order by appln_submit_datetime) = 1
  ;
  """
dfd = client.query(sq).to_dataframe()
# dfd = dfd.drop_duplicates(keep='first')
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()

# %%
df2 = dfd.copy()


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
    data_selection_column="Data_selection",
    model_version_column="modelVersionId",
    trench_column="trenchCategory",
    loan_type_column="new_loan_type",
    loan_product_type_column="loan_product_type",
    ostype_column="osType",
    account_id_column="digitalLoanAccountId",
)

# %%
fact_table, dimension_table = update_tables(
    fact_table, dimension_table, model_name="cic_model_cash", product="CASH"
)

df_f_fspd30_ciccash = fact_table.copy()
df_d_fspd30_ciccash = dimension_table.copy()


# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_f_fspd30_ciccash, facttable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_d_fspd30_ciccash, dimtable_id, job_config=job_config
)
job.result()  # Wait for the job to complete
# %%


# ### FSTPD30


# ### Test

# %%
sq = r"""
WITH parsed as (
select customerId, digitalLoanAccountId,modelDisplayName,modelVersionId,start_time,end_time,prediction,trenchCategory,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeatures,
  deviceOs osType,
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
  coalesce (p.trenchCategory, REGEXP_EXTRACT(m.requestPayload_clean, r"trenchCategory[:=]['\"]?([^'\"]+)['\"]?")) trenchCategory,
     osType,
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
  'Prod' Data_selection,
  del.deffstpd30,
  del.flg_mature_fstpd_30,
  loanmaster.new_loan_type,
  modelVersionId, trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile'
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall'
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
        coalesce((case when lower(r.osType) like '%andro%' then 'android'
                  when lower(r.osType) like '%os%' then 'ios' else lower(r.osType) end),
            (case when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%andro%' then 'android'
                  when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%os%' then 'ios'
                  when lower(loanmaster.deviceType) like '%andro%' then 'android'
                  else 'ios' end)
            ) as osType
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
  qualify row_number() over(partition by digitalLoanAccountId, modelVersionId order by appln_submit_datetime) = 1
  ;
"""
dfd = client.query(sq).to_dataframe()
# dfd = dfd.drop_duplicates(keep='first')
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()


# %%
df1 = dfd.copy()


# ### Train

# %%
sq = """
WITH parsed as (
select customerId, digitalLoanAccountId,modelDisplayName,modelVersionId,start_time,end_time,prediction,trenchCategory,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeatures,
Data_selection,     deviceOs osType,
FROM prj-prod-dataplatform.dap_ds_poweruser_playground.ml_training_model_run_details_20260116
where modelDisplayName in ('Alpha-Cash-CIC-Model','Alpha Cash CIC Model','cic_model_cash')
-- and modelVersionId = 'v1'
),
  modelname as (
  select customerId,
  digitalLoanAccountId,
  start_time,
  prediction aCicScore,
  case when modelDisplayName like 'Alpha%' then 'cic_model_cash' else modelDisplayName end as modelDisplayName,
  modelVersionId,
  trenchCategory,
  Data_selection,
      osType,
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
  Data_selection,
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
        coalesce((case when lower(r.osType) like '%andro%' then 'android'
                  when lower(r.osType) like '%os%' then 'ios' else lower(r.osType) end),
            (case when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%andro%' then 'android'
                  when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%os%' then 'ios'
                  when lower(loanmaster.deviceType) like '%andro%' then 'android'
                  else 'ios' end)
            ) as osType
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
  select * from base
  qualify row_number() over(partition by digitalLoanAccountId, modelVersionId order by appln_submit_datetime) = 1
  ;
  """
dfd = client.query(sq).to_dataframe()
# dfd = dfd.drop_duplicates(keep='first')
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()

# %%
df2 = dfd.copy()


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
    data_selection_column="Data_selection",
    model_version_column="modelVersionId",
    trench_column="trenchCategory",
    loan_type_column="new_loan_type",
    loan_product_type_column="loan_product_type",
    ostype_column="osType",
    account_id_column="digitalLoanAccountId",
)

# %%
fact_table, dimension_table = update_tables(
    fact_table, dimension_table, model_name="cic_model_cash", product="CASH"
)
print(f"The shape of the fact table is:\t {fact_table.shape}")
print(f"The shape of the dimension table is:\t {dimension_table.shape}")

df_f_fstpd30_ciccash = fact_table.copy()
df_d_fstpd30_ciccash = dimension_table.copy()


# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_f_fstpd30_ciccash, facttable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_d_fstpd30_ciccash, dimtable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# %%

factciccash = pd.concat(
    [
        df_f_fpd0_ciccash,
        df_f_fpd10_ciccash,
        df_f_fpd30_ciccash,
        df_f_fspd30_ciccash,
        df_f_fstpd30_ciccash,
    ],
    ignore_index=True,
)
demociccash = pd.concat(
    [
        df_d_fpd0_ciccash,
        df_d_fpd10_ciccash,
        df_d_fpd30_ciccash,
        df_d_fspd30_ciccash,
        df_d_fstpd30_ciccash,
    ],
    ignore_index=True,
)

print(f"cic_model_cash model Gini Calculation and Upload Completed!")


# %% [markdown]
# #### Alpha-Cash-Stack-Model

# %%
facttable_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_alphastackcash"
dimtable_id = (
    "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_alphastackcash"
)


# ##


# ### FPD0


# ### Test

# %%
sq = r"""
WITH parsed as (
select customerId, digitalLoanAccountId,modelDisplayName,modelVersionId,start_time,end_time,prediction,trenchCategory,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeatures,
    deviceOs osType,
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
  coalesce (p.trenchCategory, REGEXP_EXTRACT(m.requestPayload_clean, r"trenchCategory[:=]['\"]?([^'\"]+)['\"]?")) trenchCategory,
       osType,
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
  'Prod' Data_selection,
  deffpd0,
  flg_mature_fpd0,
  loanmaster.new_loan_type,
  modelVersionId, trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile'
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall'
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
        coalesce((case when lower(r.osType) like '%andro%' then 'android'
                  when lower(r.osType) like '%os%' then 'ios' else lower(r.osType) end),
            (case when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%andro%' then 'android'
                  when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%os%' then 'ios'
                  when lower(loanmaster.deviceType) like '%andro%' then 'android'
                  else 'ios' end)
            ) as osType
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
  qualify row_number() over(partition by digitalLoanAccountId, modelVersionId order by appln_submit_datetime) = 1
  ;
"""
dfd = client.query(sq).to_dataframe()
# dfd = dfd.drop_duplicates(keep='first')
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()


# %%
df1 = dfd.copy()


# ### Train

# %%
sq = """
WITH parsed as (
select customerId, digitalLoanAccountId,modelDisplayName,modelVersionId,start_time,end_time,prediction,trenchCategory,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeatures,
Data_selection,     deviceOs osType,


FROM prj-prod-dataplatform.dap_ds_poweruser_playground.ml_training_model_run_details_20260116
where modelDisplayName in ('Alpha-Cash-Stack-Model', 'alpha_stack_model_cash')
),
  modelname as (
  select customerId,
  digitalLoanAccountId,
  start_time,
  prediction aStackScore,
  case when p.modelDisplayName like 'Alpha%' then 'alpha_stack_model_cash' else p.modelDisplayName end as modelDisplayName,
  p.modelVersionId,
  trenchCategory,
  Data_selection,
       osType,


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
  Data_selection,
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

    coalesce((case when lower(r.osType) like '%andro%' then 'android'
                  when lower(r.osType) like '%os%' then 'ios' else lower(r.osType) end),
            (case when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%andro%' then 'android'
                  when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%os%' then 'ios'
                  when lower(loanmaster.deviceType) like '%andro%' then 'android'
                  else 'ios' end)
            ) as osType
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
  select * from base
  qualify row_number() over(partition by digitalLoanAccountId, modelVersionId order by appln_submit_datetime) = 1
  ;
  """
dfd = client.query(sq).to_dataframe()
# dfd = dfd.drop_duplicates(keep='first')
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()

# %%
df2 = dfd.copy()


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
    data_selection_column="Data_selection",
    model_version_column="modelVersionId",
    trench_column="trenchCategory",
    loan_type_column="new_loan_type",
    loan_product_type_column="loan_product_type",
    ostype_column="osType",
    account_id_column="digitalLoanAccountId",
)

# %%
fact_table, dimension_table = update_tables(
    fact_table, dimension_table, model_name="alpha_stack_model_cash", product="CASH"
)
print(f"The shape of the fact table is:\t {fact_table.shape}")
print(f"The shape of the dimension table is:\t {dimension_table.shape}")

df_f_fpd0_alphastackcash = fact_table.copy()
df_d_fpd0_alphastackcash = dimension_table.copy()


# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_TRUNCATE",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_f_fpd0_alphastackcash, facttable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_TRUNCATE",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_d_fpd0_alphastackcash, dimtable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# ### FPD10


# ### Test

# %%
sq = r"""
WITH parsed as (
select customerId, digitalLoanAccountId,modelDisplayName,modelVersionId,start_time,end_time,prediction,trenchCategory,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeatures,
   deviceOs osType,
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
  coalesce (p.trenchCategory, REGEXP_EXTRACT(m.requestPayload_clean, r"trenchCategory[:=]['\"]?([^'\"]+)['\"]?")) trenchCategory,
     osType,
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
  'Prod' Data_selection,
  del.deffpd10,
  del.flg_mature_fpd10,
  loanmaster.new_loan_type,
  modelVersionId, trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile'
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall'
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
    coalesce((case when lower(r.osType) like '%andro%' then 'android'
                  when lower(r.osType) like '%os%' then 'ios' else lower(r.osType) end),
            (case when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%andro%' then 'android'
                  when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%os%' then 'ios'
                  when lower(loanmaster.deviceType) like '%andro%' then 'android'
                  else 'ios' end)
            ) as osType
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
  qualify row_number() over(partition by digitalLoanAccountId, modelVersionId order by appln_submit_datetime) = 1
  ;
"""
dfd = client.query(sq).to_dataframe()
# dfd = dfd.drop_duplicates(keep='first')
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()


# %%
df1 = dfd.copy()


# ### Train

# %%
sq = """
WITH parsed as (
select customerId, digitalLoanAccountId,modelDisplayName,modelVersionId,start_time,end_time,prediction,trenchCategory,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeatures,
Data_selection,
    deviceOs osType,


FROM prj-prod-dataplatform.dap_ds_poweruser_playground.ml_training_model_run_details_20260116
where modelDisplayName in ('Alpha-Cash-Stack-Model', 'alpha_stack_model_cash')
),
  modelname as (
  select customerId,
  digitalLoanAccountId,
  start_time,
  prediction aStackScore,
  case when p.modelDisplayName like 'Alpha%' then 'alpha_stack_model_cash' else p.modelDisplayName end as modelDisplayName,
  p.modelVersionId,
  trenchCategory,
  Data_selection,
    osType,

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
  Data_selection,
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
coalesce((case when lower(r.osType) like '%andro%' then 'android'
                  when lower(r.osType) like '%os%' then 'ios' else lower(r.osType) end),
            (case when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%andro%' then 'android'
                  when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%os%' then 'ios'
                  when lower(loanmaster.deviceType) like '%andro%' then 'android'
                  else 'ios' end)
            ) as osType
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
  select * from base
  qualify row_number() over(partition by digitalLoanAccountId, modelVersionId order by appln_submit_datetime) = 1
  ;
  """
dfd = client.query(sq).to_dataframe()
# dfd = dfd.drop_duplicates(keep='first')
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()

# %%
df2 = dfd.copy()


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
    data_selection_column="Data_selection",
    model_version_column="modelVersionId",
    trench_column="trenchCategory",
    loan_type_column="new_loan_type",
    loan_product_type_column="loan_product_type",
    ostype_column="osType",
    account_id_column="digitalLoanAccountId",
)

# %%
fact_table, dimension_table = update_tables(
    fact_table, dimension_table, model_name="alpha_stack_model_cash", product="CASH"
)

df_f_fpd10_alphastackcash = fact_table.copy()
df_d_fpd10_alphastackcash = dimension_table.copy()


# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_f_fpd10_alphastackcash, facttable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_d_fpd10_alphastackcash, dimtable_id, job_config=job_config
)
job.result()  # Wait for the job to complete


# %%


# ### FPD30


# ### Test

# %%
sq = r"""
WITH parsed as (
select customerId, digitalLoanAccountId,modelDisplayName,modelVersionId,start_time,end_time,prediction,trenchCategory,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeatures,
    deviceOs osType,
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
  coalesce (p.trenchCategory, REGEXP_EXTRACT(m.requestPayload_clean, r"trenchCategory[:=]['\"]?([^'\"]+)['\"]?")) trenchCategory,
       osType,
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
  'Prod' Data_selection,
  del.deffpd30,
  del.flg_mature_fpd30,
  loanmaster.new_loan_type,
  modelVersionId, trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile'
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall'
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
    coalesce((case when lower(r.osType) like '%andro%' then 'android'
                  when lower(r.osType) like '%os%' then 'ios' else lower(r.osType) end),
            (case when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%andro%' then 'android'
                  when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%os%' then 'ios'
                  when lower(loanmaster.deviceType) like '%andro%' then 'android'
                  else 'ios' end)
            ) as osType
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
  qualify row_number() over(partition by digitalLoanAccountId, modelVersionId order by appln_submit_datetime) = 1
  ;
"""
dfd = client.query(sq).to_dataframe()
# dfd = dfd.drop_duplicates(keep='first')
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()


# %%
df1 = dfd.copy()


# ### Train

# %%
sq = """
WITH parsed as (
select customerId, digitalLoanAccountId,modelDisplayName,modelVersionId,start_time,end_time,prediction,trenchCategory,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeatures,
Data_selection,
  deviceOs osType,

FROM prj-prod-dataplatform.dap_ds_poweruser_playground.ml_training_model_run_details_20260116
where modelDisplayName in ('Alpha-Cash-Stack-Model', 'alpha_stack_model_cash')

),
  modelname as (
  select customerId,
  digitalLoanAccountId,
  start_time,
  prediction aStackScore,
  case when p.modelDisplayName like 'Alpha%' then 'alpha_stack_model_cash' else p.modelDisplayName end as modelDisplayName,
  p.modelVersionId,
  trenchCategory,
  Data_selection,
    osType,
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
  Data_selection,
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
    coalesce((case when lower(r.osType) like '%andro%' then 'android'
                  when lower(r.osType) like '%os%' then 'ios' else lower(r.osType) end),
            (case when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%andro%' then 'android'
                  when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%os%' then 'ios'
                  when lower(loanmaster.deviceType) like '%andro%' then 'android'
                  else 'ios' end)
            ) as osType
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
  select * from base
  qualify row_number() over(partition by digitalLoanAccountId, modelVersionId order by appln_submit_datetime) = 1
  ;
  """
dfd = client.query(sq).to_dataframe()
# dfd = dfd.drop_duplicates(keep='first')
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()

# %%
df2 = dfd.copy()


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
    data_selection_column="Data_selection",
    model_version_column="modelVersionId",
    trench_column="trenchCategory",
    loan_type_column="new_loan_type",
    loan_product_type_column="loan_product_type",
    ostype_column="osType",
    account_id_column="digitalLoanAccountId",
)

# %%
fact_table, dimension_table = update_tables(
    fact_table, dimension_table, model_name="alpha_stack_model_cash", product="CASH"
)

df_f_fpd30_alphastackcash = fact_table.copy()
df_d_fpd30_alphastackcash = dimension_table.copy()

# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_f_fpd30_alphastackcash, facttable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_d_fpd30_alphastackcash, dimtable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# %%


# ### FSPD30


# ### Test

# %%
sq = r"""
WITH parsed as (
select customerId, digitalLoanAccountId,modelDisplayName,modelVersionId,start_time,end_time,prediction,trenchCategory,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeatures,
    deviceOs osType,
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
  coalesce (p.trenchCategory, REGEXP_EXTRACT(m.requestPayload_clean, r"trenchCategory[:=]['\"]?([^'\"]+)['\"]?")) trenchCategory,
        osType,

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
  'Prod' Data_selection,
  del.deffspd30,
  del.flg_mature_fspd_30,
  loanmaster.new_loan_type,
  modelVersionId, trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile'
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall'
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
        coalesce((case when lower(r.osType) like '%andro%' then 'android'
                  when lower(r.osType) like '%os%' then 'ios' else lower(r.osType) end),
            (case when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%andro%' then 'android'
                  when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%os%' then 'ios'
                  when lower(loanmaster.deviceType) like '%andro%' then 'android'
                  else 'ios' end)
            ) as osType
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
  qualify row_number() over(partition by digitalLoanAccountId, modelVersionId order by appln_submit_datetime) = 1
  ;
"""
dfd = client.query(sq).to_dataframe()
# dfd = dfd.drop_duplicates(keep='first')
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()


# %%
df1 = dfd.copy()


# ### Train

# %%
sq = """
WITH parsed as (
select customerId, digitalLoanAccountId,modelDisplayName,modelVersionId,start_time,end_time,prediction,trenchCategory,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeatures,
Data_selection,
    deviceOs osType,
FROM prj-prod-dataplatform.dap_ds_poweruser_playground.ml_training_model_run_details_20260116
where modelDisplayName in ('Alpha-Cash-Stack-Model', 'alpha_stack_model_cash')
),
  modelname as (
  select customerId,
  digitalLoanAccountId,
  start_time,
  prediction aStackScore,
  case when p.modelDisplayName like 'Alpha%' then 'alpha_stack_model_cash' else p.modelDisplayName end as modelDisplayName,
  p.modelVersionId,
  trenchCategory,
  Data_selection,
       osType,

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
  Data_selection,
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


    coalesce((case when lower(r.osType) like '%andro%' then 'android'
                  when lower(r.osType) like '%os%' then 'ios' else lower(r.osType) end),
            (case when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%andro%' then 'android'
                  when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%os%' then 'ios'
                  when lower(loanmaster.deviceType) like '%andro%' then 'android'
                  else 'ios' end)
            ) as osType

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
  select * from base
  qualify row_number() over(partition by digitalLoanAccountId, modelVersionId order by appln_submit_datetime) = 1
  ;
  """
dfd = client.query(sq).to_dataframe()
# dfd = dfd.drop_duplicates(keep='first')
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()

# %%
df2 = dfd.copy()


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
    data_selection_column="Data_selection",
    model_version_column="modelVersionId",
    trench_column="trenchCategory",
    loan_type_column="new_loan_type",
    loan_product_type_column="loan_product_type",
    ostype_column="osType",
    account_id_column="digitalLoanAccountId",
)

# %%
fact_table, dimension_table = update_tables(
    fact_table, dimension_table, model_name="alpha_stack_model_cash", product="CASH"
)

df_f_fspd30_alphastackcash = fact_table.copy()
df_d_fspd30_alphastackcash = dimension_table.copy()

# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_f_fspd30_alphastackcash, facttable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_d_fspd30_alphastackcash, dimtable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# %%


# ### FSTPD30


# ### Test

# %%
sq = r"""
WITH parsed as (
select customerId, digitalLoanAccountId,modelDisplayName,modelVersionId,start_time,end_time,prediction,trenchCategory,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeatures,
    deviceOs osType,
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
  coalesce (p.trenchCategory, REGEXP_EXTRACT(m.requestPayload_clean, r"trenchCategory[:=]['\"]?([^'\"]+)['\"]?")) trenchCategory,
      osType,
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
  'Prod' Data_selection,
  del.deffstpd30,
  del.flg_mature_fstpd_30,
  loanmaster.new_loan_type,
  modelVersionId, trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile'
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall'
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
    coalesce((case when lower(r.osType) like '%andro%' then 'android'
                  when lower(r.osType) like '%os%' then 'ios' else lower(r.osType) end),
            (case when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%andro%' then 'android'
                  when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%os%' then 'ios'
                  when lower(loanmaster.deviceType) like '%andro%' then 'android'
                  else 'ios' end)
            ) as osType
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
  qualify row_number() over(partition by digitalLoanAccountId, modelVersionId order by appln_submit_datetime) = 1
  ;
"""
dfd = client.query(sq).to_dataframe()
# dfd = dfd.drop_duplicates(keep='first')
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()


# %%
df1 = dfd.copy()


# ### Train

# %%
sq = """
WITH parsed as (
select customerId, digitalLoanAccountId,modelDisplayName,modelVersionId,start_time,end_time,prediction,trenchCategory,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeatures,
Data_selection,
    deviceOs osType,
FROM prj-prod-dataplatform.dap_ds_poweruser_playground.ml_training_model_run_details_20260116
where modelDisplayName in ('Alpha-Cash-Stack-Model', 'alpha_stack_model_cash')
),
  modelname as (
  select customerId,
  digitalLoanAccountId,
  start_time,
  prediction aStackScore,
  case when p.modelDisplayName like 'Alpha%' then 'alpha_stack_model_cash' else p.modelDisplayName end as modelDisplayName,
  p.modelVersionId,
  trenchCategory,
  Data_selection,
      osType,
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
  Data_selection,
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
    coalesce((case when lower(r.osType) like '%andro%' then 'android'
                  when lower(r.osType) like '%os%' then 'ios' else lower(r.osType) end),
            (case when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%andro%' then 'android'
                  when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%os%' then 'ios'
                  when lower(loanmaster.deviceType) like '%andro%' then 'android'
                  else 'ios' end)
            ) as osType
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
  select * from base
  qualify row_number() over(partition by digitalLoanAccountId, modelVersionId order by appln_submit_datetime) = 1
  ;
  """
dfd = client.query(sq).to_dataframe()
# dfd = dfd.drop_duplicates(keep='first')
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()

# %%
df2 = dfd.copy()


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
    data_selection_column="Data_selection",
    model_version_column="modelVersionId",
    trench_column="trenchCategory",
    loan_type_column="new_loan_type",
    loan_product_type_column="loan_product_type",
    ostype_column="osType",
    account_id_column="digitalLoanAccountId",
)

# %%
fact_table, dimension_table = update_tables(
    fact_table, dimension_table, model_name="alpha_stack_model_cash", product="CASH"
)
print(f"The shape of the fact table is:\t {fact_table.shape}")
print(f"The shape of the dimension table is:\t {dimension_table.shape}")

df_f_fstpd30_alphastackcash = fact_table.copy()
df_d_fstpd30_alphastackcash = dimension_table.copy()

# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_f_fstpd30_alphastackcash, facttable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_d_fstpd30_alphastackcash, dimtable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

factalphastackcash = pd.concat(
    [
        df_f_fpd0_alphastackcash,
        df_f_fpd10_alphastackcash,
        df_f_fpd30_alphastackcash,
        df_f_fspd30_alphastackcash,
        df_f_fstpd30_alphastackcash,
    ],
    ignore_index=True,
)
dimalphastackcash = pd.concat(
    [
        df_d_fpd0_alphastackcash,
        df_d_fpd10_alphastackcash,
        df_d_fpd30_alphastackcash,
        df_d_fspd30_alphastackcash,
        df_d_fstpd30_alphastackcash,
    ],
    ignore_index=True,
)

print(f"The alpha_stack_model_cash gini calculation completed")

# %% [markdown]
# ####  Beta-Cash-Stack-Model- Credo Score

# %%
# %%
facttable_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_betacredocash"
dimtable_id = (
    "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_betacredocash"
)

# %%
# ## Beta-Cash-Stack-Model- Credo Score


# ### FPD0


# ### Test

# %%

sq = """
with modelname as
  (  SELECT
    mmrd.customerId,mmrd.digitalLoanAccountId,prediction,start_time,end_time,modelDisplayName,modelVersionId,
  case when trenchCategory is null then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench1' end)
     when trenchCategory = '' then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench 1' end)
    else trenchCategory end  as trenchCategory,
    REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature,
    deviceOs osType,
    FROM prj-prod-dataplatform.audit_balance.ml_model_run_details mmrd
  left join prj-prod-dataplatform.risk_credit_mis.model_loan_score_mart mt on mt.digitalLoanAccountId = mmrd.digitalLoanAccountId
  WHERE modelDisplayName in ('Beta-Cash-Stack-Model', 'beta_stack_model_cash')
  -- and modelVersionId = 'v1'
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
  coalesce(cast(JSON_VALUE(SAFE.PARSE_JSON(CAST(calcFeature AS STRING)), '$.credo_score')AS FLOAT64), cast(JSON_VALUE(SAFE.PARSE_JSON(CAST(calcFeature AS STRING)), '$.credo_score')AS FLOAT64)) AS credo_score,
  calcFeature,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month,
  'Prod' Data_selection,
  deffpd0,
  flg_mature_fpd0,
  loanmaster.new_loan_type,
  modelVersionId, trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile'
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall'
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,

    coalesce((case when lower(r.osType) like '%andro%' then 'android'
                  when lower(r.osType) like '%os%' then 'ios' else lower(r.osType) end),
            (case when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%andro%' then 'android'
                  when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%os%' then 'ios'
                  when lower(loanmaster.deviceType) like '%andro%' then 'android'
                  else 'ios' end)
            ) as osType
  from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  left join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
   left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
  where
  loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and flg_mature_fpd0 = 1
  )
  select *
  from base
  where credo_score is not null
 qualify row_number() over(partition by digitalLoanAccountId, modelVersionId order by appln_submit_datetime) = 1
   ;



"""
dfd = client.query(sq).to_dataframe()
# dfd = dfd.drop_duplicates(keep='first')
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()


# %%
df1 = dfd.copy()


# ### Train

# %%
sq = """
with modelname as
  (  SELECT
    mmrd.customerId,mmrd.digitalLoanAccountId,prediction,start_time,end_time,modelDisplayName,modelVersionId,
  case when trenchCategory is null then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench1' end)
     when trenchCategory = '' then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench 1' end)
    else trenchCategory end  as trenchCategory,
    REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature,
    Data_selection, deviceOs osType,
    FROM prj-prod-dataplatform.dap_ds_poweruser_playground.ml_training_model_run_details_20260116 mmrd
  left join prj-prod-dataplatform.risk_credit_mis.model_loan_score_mart mt on mt.digitalLoanAccountId = mmrd.digitalLoanAccountId
  WHERE modelDisplayName in ('Beta-Cash-Stack-Model', 'beta_stack_model_cash')
  -- and modelVersionId = 'v1'
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
  coalesce(cast(JSON_VALUE(SAFE.PARSE_JSON(CAST(calcFeature AS STRING)), '$.credo_score')AS FLOAT64), cast(JSON_VALUE(SAFE.PARSE_JSON(CAST(calcFeature AS STRING)), '$.credo_score')AS FLOAT64)) AS credo_score,
  calcFeature,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month,
  Data_selection,
  deffpd0,
  flg_mature_fpd0,
  loanmaster.new_loan_type,
  modelVersionId, trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile'
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall'
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,

    coalesce((case when lower(r.osType) like '%andro%' then 'android'
                  when lower(r.osType) like '%os%' then 'ios' else lower(r.osType) end),
            (case when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%andro%' then 'android'
                  when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%os%' then 'ios'
                  when lower(loanmaster.deviceType) like '%andro%' then 'android'
                  else 'ios' end)
            ) as osType
  from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  left join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
   left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
  where
  loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and flg_mature_fpd0 = 1
  )
  select *
  from base
  where credo_score is not null
 qualify row_number() over(partition by digitalLoanAccountId, modelVersionId order by appln_submit_datetime) = 1
   ;
  """
dfd = client.query(sq).to_dataframe()
# dfd = dfd.drop_duplicates(keep='first')
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()

# %%
df2 = dfd.copy()


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
# df_concat = df1.copy()

df_concat["credo_score"] = pd.to_numeric(df_concat["credo_score"], errors="coerce")

# %%
fact_table, dimension_table = calculate_periodic_gini_prod_ver_trench_dimfact(
    df_concat,
    "credo_score",
    "deffpd0",
    "FPD0",
    data_selection_column="Data_selection",
    model_version_column="modelVersionId",
    trench_column="trenchCategory",
    loan_type_column="new_loan_type",
    loan_product_type_column="loan_product_type",
    ostype_column="osType",
    account_id_column="digitalLoanAccountId",
)

# %%
fact_table, dimension_table = update_tables(
    fact_table,
    dimension_table,
    model_name="credo_score_cash",
    product="CASH",
)
print(f"The shape of the fact table is:\t {fact_table.shape}")
print(f"The shape of the dimension table is:\t {dimension_table.shape}")

# %%
# Upload to BigQuery
df_f_fpd0_credoscorecash = fact_table.copy()
df_d_fpd0_credoscorecash = dimension_table.copy()

# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_TRUNCATE",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_f_fpd0_credoscorecash, facttable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_TRUNCATE",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_d_fpd0_credoscorecash, dimtable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# ### FPD10


# ### Test

# %%

sq = """
with modelname as
  (  SELECT
    mmrd.customerId,mmrd.digitalLoanAccountId,prediction,start_time,end_time,modelDisplayName,modelVersionId,
  case when trenchCategory is null then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench1' end)
     when trenchCategory = '' then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench 1' end)
    else trenchCategory end  as trenchCategory,
    REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature,
    deviceOs osType,
    FROM prj-prod-dataplatform.audit_balance.ml_model_run_details mmrd
  left join prj-prod-dataplatform.risk_credit_mis.model_loan_score_mart mt on mt.digitalLoanAccountId = mmrd.digitalLoanAccountId
  WHERE modelDisplayName in ('Beta-Cash-Stack-Model', 'beta_stack_model_cash')
  -- and modelVersionId = 'v1'
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
  coalesce(cast(JSON_VALUE(SAFE.PARSE_JSON(CAST(calcFeature AS STRING)), '$.credo_score')AS FLOAT64), cast(JSON_VALUE(SAFE.PARSE_JSON(CAST(calcFeature AS STRING)), '$.credo_score')AS FLOAT64)) AS credo_score,
  calcFeature,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month,
  'Prod' Data_selection,
  del.deffpd10,
  del.flg_mature_fpd10,
  loanmaster.new_loan_type,
  modelVersionId, trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile'
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall'
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
        coalesce((case when lower(r.osType) like '%andro%' then 'android'
                  when lower(r.osType) like '%os%' then 'ios' else lower(r.osType) end),
            (case when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%andro%' then 'android'
                  when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%os%' then 'ios'
                  when lower(loanmaster.deviceType) like '%andro%' then 'android'
                  else 'ios' end)
            ) as osType
  from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  left join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
   left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
  where
  loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and del.flg_mature_fpd10 = 1
  )
  select *
  from base
  where credo_score is not null
 qualify row_number() over(partition by digitalLoanAccountId, modelVersionId order by appln_submit_datetime) = 1
   ;



"""
dfd = client.query(sq).to_dataframe()
# dfd = dfd.drop_duplicates(keep='first')
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()


# %%
df1 = dfd.copy()


# ### Train

# %%
sq = """
with modelname as
  (  SELECT
    mmrd.customerId,mmrd.digitalLoanAccountId,prediction,start_time,end_time,modelDisplayName,modelVersionId,
  case when trenchCategory is null then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench1' end)
     when trenchCategory = '' then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench 1' end)
    else trenchCategory end  as trenchCategory,
    REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature, deviceOs osType,
    Data_selection
    FROM prj-prod-dataplatform.dap_ds_poweruser_playground.ml_training_model_run_details_20260116 mmrd
  left join prj-prod-dataplatform.risk_credit_mis.model_loan_score_mart mt on mt.digitalLoanAccountId = mmrd.digitalLoanAccountId
  WHERE modelDisplayName in ('Beta-Cash-Stack-Model', 'beta_stack_model_cash')
  -- and modelVersionId = 'v1'
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
  coalesce(cast(JSON_VALUE(SAFE.PARSE_JSON(CAST(calcFeature AS STRING)), '$.credo_score')AS FLOAT64), cast(JSON_VALUE(SAFE.PARSE_JSON(CAST(calcFeature AS STRING)), '$.credo_score')AS FLOAT64)) AS credo_score,
  calcFeature,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month,
  Data_selection,
  del.deffpd10,
  del.flg_mature_fpd10,
  loanmaster.new_loan_type,
  modelVersionId, trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile'
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall'
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
        coalesce((case when lower(r.osType) like '%andro%' then 'android'
                  when lower(r.osType) like '%os%' then 'ios' else lower(r.osType) end),
            (case when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%andro%' then 'android'
                  when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%os%' then 'ios'
                  when lower(loanmaster.deviceType) like '%andro%' then 'android'
                  else 'ios' end)
            ) as osType
  from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  left join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
   left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
  where
  loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and del.flg_mature_fpd10 = 1
  )
  select *
  from base
  where credo_score is not null
 qualify row_number() over(partition by digitalLoanAccountId, modelVersionId order by appln_submit_datetime) = 1
   ;
  """
dfd = client.query(sq).to_dataframe()
# dfd = dfd.drop_duplicates(keep='first')
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()

# %%
df2 = dfd.copy()


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
# df_concat = df1.copy()

df_concat["credo_score"] = pd.to_numeric(df_concat["credo_score"], errors="coerce")

# %%
fact_table, dimension_table = calculate_periodic_gini_prod_ver_trench_dimfact(
    df_concat,
    "credo_score",
    "deffpd10",
    "FPD10",
    data_selection_column="Data_selection",
    model_version_column="modelVersionId",
    trench_column="trenchCategory",
    loan_type_column="new_loan_type",
    loan_product_type_column="loan_product_type",
    ostype_column="osType",
    account_id_column="digitalLoanAccountId",
)

# %%
fact_table, dimension_table = update_tables(
    fact_table,
    dimension_table,
    model_name="credo_score_cash",
    product="CASH",
)
print(f"The shape of the fact table is:\t {fact_table.shape}")
print(f"The shape of the dimension table is:\t {dimension_table.shape}")

# %%
df_f_fpd10_credoscorecash = fact_table.copy()
df_d_fpd10_credoscorecash = dimension_table.copy()

# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_f_fpd10_credoscorecash, facttable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_d_fpd10_credoscorecash, dimtable_id, job_config=job_config
)
job.result()  # Wait for the job to complete
# %%


# ### FPD30


# ### Test

# %%

sq = """
with modelname as
  (  SELECT
    mmrd.customerId,mmrd.digitalLoanAccountId,prediction,start_time,end_time,modelDisplayName,modelVersionId,
  case when trenchCategory is null then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench1' end)
     when trenchCategory = '' then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench 1' end)
    else trenchCategory end  as trenchCategory,
    REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature,  deviceOs osType,
    FROM prj-prod-dataplatform.audit_balance.ml_model_run_details mmrd
  left join prj-prod-dataplatform.risk_credit_mis.model_loan_score_mart mt on mt.digitalLoanAccountId = mmrd.digitalLoanAccountId
  WHERE modelDisplayName in ('Beta-Cash-Stack-Model', 'beta_stack_model_cash')
  -- and modelVersionId = 'v1'
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
  coalesce(cast(JSON_VALUE(SAFE.PARSE_JSON(CAST(calcFeature AS STRING)), '$.credo_score')AS FLOAT64), cast(JSON_VALUE(SAFE.PARSE_JSON(CAST(calcFeature AS STRING)), '$.credo_score')AS FLOAT64)) AS credo_score,
  calcFeature,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month,
  'Prod' Data_selection,
  del.deffpd30,
  del.flg_mature_fpd30,
  loanmaster.new_loan_type,
  modelVersionId, trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile'
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall'
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
        coalesce((case when lower(r.osType) like '%andro%' then 'android'
                  when lower(r.osType) like '%os%' then 'ios' else lower(r.osType) end),
            (case when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%andro%' then 'android'
                  when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%os%' then 'ios'
                  when lower(loanmaster.deviceType) like '%andro%' then 'android'
                  else 'ios' end)
            ) as osType
  from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  left join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
   left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
  where
  loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and del.flg_mature_fpd30 = 1
  )
  select *
  from base
  where credo_score is not null
 qualify row_number() over(partition by digitalLoanAccountId, modelVersionId order by appln_submit_datetime) = 1
   ;



"""
dfd = client.query(sq).to_dataframe()
# dfd = dfd.drop_duplicates(keep='first')
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()


# %%
df1 = dfd.copy()


# ### Train

# %%
sq = """
with modelname as
  (  SELECT
    mmrd.customerId,mmrd.digitalLoanAccountId,prediction,start_time,end_time,modelDisplayName,modelVersionId,
  case when trenchCategory is null then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench1' end)
     when trenchCategory = '' then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench 1' end)
    else trenchCategory end  as trenchCategory,
    REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature,  deviceOs osType,
    Data_selection
    FROM prj-prod-dataplatform.dap_ds_poweruser_playground.ml_training_model_run_details_20260116 mmrd
  left join prj-prod-dataplatform.risk_credit_mis.model_loan_score_mart mt on mt.digitalLoanAccountId = mmrd.digitalLoanAccountId
  WHERE modelDisplayName in ('Beta-Cash-Stack-Model', 'beta_stack_model_cash')
  -- and modelVersionId = 'v1'
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
  coalesce(cast(JSON_VALUE(SAFE.PARSE_JSON(CAST(calcFeature AS STRING)), '$.credo_score')AS FLOAT64), cast(JSON_VALUE(SAFE.PARSE_JSON(CAST(calcFeature AS STRING)), '$.credo_score')AS FLOAT64)) AS credo_score,
  calcFeature,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month,
  Data_selection,
  del.deffpd30,
  del.flg_mature_fpd30,
  loanmaster.new_loan_type,
  modelVersionId, trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile'
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall'
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
        coalesce((case when lower(r.osType) like '%andro%' then 'android'
                  when lower(r.osType) like '%os%' then 'ios' else lower(r.osType) end),
            (case when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%andro%' then 'android'
                  when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%os%' then 'ios'
                  when lower(loanmaster.deviceType) like '%andro%' then 'android'
                  else 'ios' end)
            ) as osType
  from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  left join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
   left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
  where
  loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and del.flg_mature_fpd0 = 1
  )
  select *
  from base
  where credo_score is not null
 qualify row_number() over(partition by digitalLoanAccountId, modelVersionId order by appln_submit_datetime) = 1
   ;
  """
dfd = client.query(sq).to_dataframe()
# dfd = dfd.drop_duplicates(keep='first')
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()

# %%
df2 = dfd.copy()


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
# df_concat = df1.copy()

df_concat["credo_score"] = pd.to_numeric(df_concat["credo_score"], errors="coerce")

# %%
fact_table, dimension_table = calculate_periodic_gini_prod_ver_trench_dimfact(
    df_concat,
    "credo_score",
    "deffpd30",
    "FPD30",
    data_selection_column="Data_selection",
    model_version_column="modelVersionId",
    trench_column="trenchCategory",
    loan_type_column="new_loan_type",
    loan_product_type_column="loan_product_type",
    ostype_column="osType",
    account_id_column="digitalLoanAccountId",
)

# %%
fact_table, dimension_table = update_tables(
    fact_table,
    dimension_table,
    model_name="credo_score_cash",
    product="CASH",
)
print(f"The shape of the fact table is:\t {fact_table.shape}")
print(f"The shape of the dimension table is:\t {dimension_table.shape}")

df_f_fpd30_credoscorecash = fact_table.copy()
df_d_fpd30_credoscorecash = dimension_table.copy()

# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_f_fpd30_credoscorecash, facttable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_d_fpd30_credoscorecash, dimtable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# %%


# ### FSPD30


# ### Test

# %%

sq = """
with modelname as
  (  SELECT
    mmrd.customerId,mmrd.digitalLoanAccountId,prediction,start_time,end_time,modelDisplayName,modelVersionId,
  case when trenchCategory is null then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench1' end)
     when trenchCategory = '' then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench 1' end)
    else trenchCategory end  as trenchCategory,
    REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature,
        deviceOs osType,
    FROM prj-prod-dataplatform.audit_balance.ml_model_run_details mmrd
  left join prj-prod-dataplatform.risk_credit_mis.model_loan_score_mart mt on mt.digitalLoanAccountId = mmrd.digitalLoanAccountId
  WHERE modelDisplayName in ('Beta-Cash-Stack-Model', 'beta_stack_model_cash')
  -- and modelVersionId = 'v1'
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
  coalesce(cast(JSON_VALUE(SAFE.PARSE_JSON(CAST(calcFeature AS STRING)), '$.credo_score')AS FLOAT64), cast(JSON_VALUE(SAFE.PARSE_JSON(CAST(calcFeature AS STRING)), '$.credo_score')AS FLOAT64)) AS credo_score,
  calcFeature,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month,
  'Prod' Data_selection,
  del.deffspd30,
  del.flg_mature_fspd_30,
  loanmaster.new_loan_type,
  modelVersionId, trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile'
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall'
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
        coalesce((case when lower(r.osType) like '%andro%' then 'android'
                  when lower(r.osType) like '%os%' then 'ios' else lower(r.osType) end),
            (case when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%andro%' then 'android'
                  when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%os%' then 'ios'
                  when lower(loanmaster.deviceType) like '%andro%' then 'android'
                  else 'ios' end)
            ) as osType
  from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  left join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
   left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
  where
  loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and del.flg_mature_fspd_30 = 1
  )
  select *
  from base
  where credo_score is not null
 qualify row_number() over(partition by digitalLoanAccountId, modelVersionId order by appln_submit_datetime) = 1
   ;



"""
dfd = client.query(sq).to_dataframe()
# dfd = dfd.drop_duplicates(keep='first')
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()


# %%
df1 = dfd.copy()


# ### Train

# %%
sq = """
with modelname as
  (  SELECT
    mmrd.customerId,mmrd.digitalLoanAccountId,prediction,start_time,end_time,modelDisplayName,modelVersionId,
  case when trenchCategory is null then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench1' end)
     when trenchCategory = '' then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench 1' end)
    else trenchCategory end  as trenchCategory,
    REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature,     deviceOs osType,
    Data_selection
    FROM prj-prod-dataplatform.dap_ds_poweruser_playground.ml_training_model_run_details_20260116 mmrd
  left join prj-prod-dataplatform.risk_credit_mis.model_loan_score_mart mt on mt.digitalLoanAccountId = mmrd.digitalLoanAccountId
  WHERE modelDisplayName in ('Beta-Cash-Stack-Model', 'beta_stack_model_cash')
  -- and modelVersionId = 'v1'
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
  coalesce(cast(JSON_VALUE(SAFE.PARSE_JSON(CAST(calcFeature AS STRING)), '$.credo_score')AS FLOAT64), cast(JSON_VALUE(SAFE.PARSE_JSON(CAST(calcFeature AS STRING)), '$.credo_score')AS FLOAT64)) AS credo_score,
  calcFeature,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month,
  Data_selection,
  del.deffspd30,
  del.flg_mature_fspd_30,
  loanmaster.new_loan_type,
  modelVersionId, trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile'
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall'
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
        coalesce((case when lower(r.osType) like '%andro%' then 'android'
                  when lower(r.osType) like '%os%' then 'ios' else lower(r.osType) end),
            (case when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%andro%' then 'android'
                  when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%os%' then 'ios'
                  when lower(loanmaster.deviceType) like '%andro%' then 'android'
                  else 'ios' end)
            ) as osType
  from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  left join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
   left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
  where
  loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and del.flg_mature_fspd_30 = 1
  )
  select *
  from base
  where credo_score is not null
 qualify row_number() over(partition by digitalLoanAccountId, modelVersionId order by appln_submit_datetime) = 1
   ;
  """
dfd = client.query(sq).to_dataframe()
# dfd = dfd.drop_duplicates(keep='first')
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()

# %%
df2 = dfd.copy()


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
# df_concat = df1.copy()

df_concat["credo_score"] = pd.to_numeric(df_concat["credo_score"], errors="coerce")

# %%
fact_table, dimension_table = calculate_periodic_gini_prod_ver_trench_dimfact(
    df_concat,
    "credo_score",
    "deffspd30",
    "FSPD30",
    data_selection_column="Data_selection",
    model_version_column="modelVersionId",
    trench_column="trenchCategory",
    loan_type_column="new_loan_type",
    loan_product_type_column="loan_product_type",
    ostype_column="osType",
    account_id_column="digitalLoanAccountId",
)

# %%
fact_table, dimension_table = update_tables(
    fact_table,
    dimension_table,
    model_name="credo_score_cash",
    product="CASH",
)
print(f"The shape of the fact table is:\t {fact_table.shape}")
print(f"The shape of the dimension table is:\t {dimension_table.shape}")

# %%
df_f_fspd30_credoscorecash = fact_table.copy()
df_d_fspd30_credoscorecash = dimension_table.copy()
# %%

# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_f_fspd30_credoscorecash, facttable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_d_fspd30_credoscorecash, dimtable_id, job_config=job_config
)
job.result()  # Wait for the job to complete


# ### FSTPD30


# ### Test

# %%

sq = """
with modelname as
  (  SELECT
    mmrd.customerId,mmrd.digitalLoanAccountId,prediction,start_time,end_time,modelDisplayName,modelVersionId,
  case when trenchCategory is null then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench1' end)
     when trenchCategory = '' then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench 1' end)
    else trenchCategory end  as trenchCategory,
    REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature,
     deviceOs osType,
    FROM prj-prod-dataplatform.audit_balance.ml_model_run_details mmrd
  left join prj-prod-dataplatform.risk_credit_mis.model_loan_score_mart mt on mt.digitalLoanAccountId = mmrd.digitalLoanAccountId
  WHERE modelDisplayName in ('Beta-Cash-Stack-Model', 'beta_stack_model_cash')
  -- and modelVersionId = 'v1'
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
  coalesce(cast(JSON_VALUE(SAFE.PARSE_JSON(CAST(calcFeature AS STRING)), '$.credo_score')AS FLOAT64), cast(JSON_VALUE(SAFE.PARSE_JSON(CAST(calcFeature AS STRING)), '$.credo_score')AS FLOAT64)) AS credo_score,
  calcFeature,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month,
  'Prod' Data_selection,
  del.deffstpd30,
  del.flg_mature_fstpd_30,
  loanmaster.new_loan_type,
  modelVersionId, trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile'
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall'
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
        coalesce((case when lower(r.osType) like '%andro%' then 'android'
                  when lower(r.osType) like '%os%' then 'ios' else lower(r.osType) end),
            (case when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%andro%' then 'android'
                  when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%os%' then 'ios'
                  when lower(loanmaster.deviceType) like '%andro%' then 'android'
                  else 'ios' end)
            ) as osType
  from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  left join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
   left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
  where
  loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and del.flg_mature_fstpd_30 = 1
  )
  select *
  from base
  where credo_score is not null
 qualify row_number() over(partition by digitalLoanAccountId, modelVersionId order by appln_submit_datetime) = 1
   ;



"""
dfd = client.query(sq).to_dataframe()
# dfd = dfd.drop_duplicates(keep='first')
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()


# %%
df1 = dfd.copy()


# ### Train

# %%
sq = """
with modelname as
  (  SELECT
    mmrd.customerId,mmrd.digitalLoanAccountId,prediction,start_time,end_time,modelDisplayName,modelVersionId,
  case when trenchCategory is null then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench1' end)
     when trenchCategory = '' then (case when mt.ln_user_type='1_Repeat Applicant' then 'Trench 3'
    when mt.ln_user_type <>'1_Repeat Applicant' and DATE_DIFF(current_date(), mt.onb_tsa_onboarding_datetime, DAY) >30 then 'Trench 2'
    else 'Trench 1' end)
    else trenchCategory end  as trenchCategory,
    REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature,  deviceOs osType,
    Data_selection
    FROM prj-prod-dataplatform.dap_ds_poweruser_playground.ml_training_model_run_details_20260116 mmrd
  left join prj-prod-dataplatform.risk_credit_mis.model_loan_score_mart mt on mt.digitalLoanAccountId = mmrd.digitalLoanAccountId
  WHERE modelDisplayName in ('Beta-Cash-Stack-Model', 'beta_stack_model_cash')
  -- and modelVersionId = 'v1'
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
  coalesce(cast(JSON_VALUE(SAFE.PARSE_JSON(CAST(calcFeature AS STRING)), '$.credo_score')AS FLOAT64), cast(JSON_VALUE(SAFE.PARSE_JSON(CAST(calcFeature AS STRING)), '$.credo_score')AS FLOAT64)) AS credo_score,
  calcFeature,
  coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime)) AS appln_submit_datetime,
  date(loanmaster.disbursementDateTime) disbursementdate,
  format_date('%Y-%m', coalesce(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime),  cast(r.start_time as datetime))) as Application_month,
   Data_selection,
  del.deffstpd30,
  del.flg_mature_fstpd_30,
  loanmaster.new_loan_type,
  modelVersionId, trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile'
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall'
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
        coalesce((case when lower(r.osType) like '%andro%' then 'android'
                  when lower(r.osType) like '%os%' then 'ios' else lower(r.osType) end),
            (case when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%andro%' then 'android'
                  when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%os%' then 'ios'
                  when lower(loanmaster.deviceType) like '%andro%' then 'android'
                  else 'ios' end)
            ) as osType
  from modelname r
  left join risk_credit_mis.loan_master_table loanmaster  ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
  left join deliquency del on del.loanAccountNumber = loanmaster.loanAccountNumber
   left join(SELECT DISTINCT mer_refferal_code, mer_name mer_name,store_type,store_tagging FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
  left join worktable_datachampions.TARGET_SPLIT P on P.STORE_NAME = mer_name
 qualify row_number() over(partition by mer_refferal_code order by  created_dt desc)=1) sil_category on loanmaster.purpleKey=sil_category.mer_refferal_code
  where
  loanmaster.flagDisbursement = 1
  and loanmaster.disbursementDateTime is not null
  and flg_mature_fstpd_30 = 1
  )
  select *
  from base
  where credo_score is not null
 qualify row_number() over(partition by digitalLoanAccountId, modelVersionId order by appln_submit_datetime) = 1
   ;
  """
dfd = client.query(sq).to_dataframe()
# dfd = dfd.drop_duplicates(keep='first')
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()

# %%
df2 = dfd.copy()


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
# df_concat = df1.copy()

df_concat["credo_score"] = pd.to_numeric(df_concat["credo_score"], errors="coerce")

# %%
fact_table, dimension_table = calculate_periodic_gini_prod_ver_trench_dimfact(
    df_concat,
    "credo_score",
    "deffstpd30",
    "FSTPD30",
    data_selection_column="Data_selection",
    model_version_column="modelVersionId",
    trench_column="trenchCategory",
    loan_type_column="new_loan_type",
    loan_product_type_column="loan_product_type",
    ostype_column="osType",
    account_id_column="digitalLoanAccountId",
)

# %%
fact_table, dimension_table = update_tables(
    fact_table,
    dimension_table,
    model_name="credo_score_cash",
    product="CASH",
)
print(f"The shape of the fact table is:\t {fact_table.shape}")
print(f"The shape of the dimension table is:\t {dimension_table.shape}")

df_f_fstpd30_credoscorecash = fact_table.copy()
df_d_fstpd30_credoscorecash = dimension_table.copy()

# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_f_fstpd30_credoscorecash, facttable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_d_fstpd30_credoscorecash, dimtable_id, job_config=job_config
)
job.result()  # Wait for the job to complete


factcredoscorecash = pd.concat(
    [
        df_f_fpd0_credoscorecash,
        df_f_fpd10_credoscorecash,
        df_f_fpd30_credoscorecash,
        df_f_fspd30_credoscorecash,
        df_f_fstpd30_credoscorecash,
    ],
    ignore_index=True,
)
dimcredoscorecash = pd.concat(
    [
        df_d_fpd0_credoscorecash,
        df_d_fpd10_credoscorecash,
        df_d_fpd30_credoscorecash,
        df_d_fspd30_credoscorecash,
        df_d_fstpd30_credoscorecash,
    ],
    ignore_index=True,
)


print(f" credo_score_cash gini calculation completed")


# %% [markdown]
# #### Beta-Cash-AppScore-Model

# %%

# %%

facttable_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_appscorecash"
dimtable_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_appscorecash"

# ## Beta-Cash-AppScore-Model


# ### FPD0


# ### Test

# %%

sq = """
WITH parsed as (
select customerId, digitalLoanAccountId,modelDisplayName,modelVersionId,start_time,end_time,prediction,trenchCategory,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeatures,
REPLACE(REPLACE(prediction, "'", '"'), "None", "null") AS prediction_clean,  deviceOs osType,
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
  coalesce(p.trenchCategory, JSON_VALUE(m.requestPayload_clean, "$.predictions.trenchCategory"))  trenchCategory, osType
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
  'Prod' Data_selection,
  deffpd0,
  flg_mature_fpd0,
  loanmaster.new_loan_type,
  modelVersionId, trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile'
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall'
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
      coalesce((case when lower(r.osType) like '%andro%' then 'android'
                  when lower(r.osType) like '%os%' then 'ios' else lower(r.osType) end),
            (case when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%andro%' then 'android'
                  when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%os%' then 'ios'
                  when lower(loanmaster.deviceType) like '%andro%' then 'android'
                  else 'ios' end)
            ) as osType
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
  qualify row_number() over(partition by digitalLoanAccountId, modelVersionId order by appln_submit_datetime) = 1
  ;
"""
dfd = client.query(sq).to_dataframe()
# dfd = dfd.drop_duplicates(keep='first')
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()


# %%
df1 = dfd.copy()


# ### Train

# %%
sq = """WITH parsed as (
select customerId, digitalLoanAccountId,modelDisplayName,modelVersionId,start_time,end_time,prediction,trenchCategory,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeatures,  deviceOs osType,
Data_selection
FROM prj-prod-dataplatform.dap_ds_poweruser_playground.ml_training_model_run_details_20260116
where modelDisplayName in  ('Beta-Cash-AppScore-Model', 'apps_score_cash')
),
  modelname as (
  select customerId,
  digitalLoanAccountId,
  start_time,
  prediction beta_cash_app_score,
  case when modelDisplayName like 'Beta-Cash-AppScore-Model' then 'apps_score_cash' else modelDisplayName end as modelDisplayName,
  modelVersionId,
  trenchCategory,
  Data_selection, osType
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
  Data_selection,
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
      coalesce((case when lower(r.osType) like '%andro%' then 'android'
                  when lower(r.osType) like '%os%' then 'ios' else lower(r.osType) end),
            (case when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%andro%' then 'android'
                  when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%os%' then 'ios'
                  when lower(loanmaster.deviceType) like '%andro%' then 'android'
                  else 'ios' end)
            ) as osType
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
  select * from base
  qualify row_number() over(partition by digitalLoanAccountId, modelVersionId order by appln_submit_datetime) = 1
  ;
  """
dfd = client.query(sq).to_dataframe()
# dfd = dfd.drop_duplicates(keep='first')
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()

# %%
df2 = dfd.copy()


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
    data_selection_column="Data_selection",
    model_version_column="modelVersionId",
    trench_column="trenchCategory",
    loan_type_column="new_loan_type",
    loan_product_type_column="loan_product_type",
    ostype_column="osType",
    account_id_column="digitalLoanAccountId",
)

# %%
fact_table, dimension_table = update_tables(
    fact_table, dimension_table, model_name="apps_score_cash", product="CASH"
)
print(f"The shape of the fact table is:\t {fact_table.shape}")
print(f"The shape of the dimension table is:\t {dimension_table.shape}")


df_f_fpd0_appscorecash = fact_table.copy()
df_d_fpd0_appscorecash = dimension_table.copy()

# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_TRUNCATE",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_f_fpd0_appscorecash, facttable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_TRUNCATE",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_d_fpd0_appscorecash, dimtable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# ### FPD10


# ### Test

# %%
sq = """
WITH parsed as (
select customerId, digitalLoanAccountId,modelDisplayName,modelVersionId,start_time,end_time,prediction,trenchCategory,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeatures,
REPLACE(REPLACE(prediction, "'", '"'), "None", "null") AS prediction_clean,    deviceOs osType,
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
  coalesce(p.trenchCategory, JSON_VALUE(m.requestPayload_clean, "$.predictions.trenchCategory"))  trenchCategory, osType
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
  'Prod' Data_selection,
  del.deffpd10,
  del.flg_mature_fpd10,
  loanmaster.new_loan_type,
  modelVersionId, trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile'
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall'
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
      coalesce((case when lower(r.osType) like '%andro%' then 'android'
                  when lower(r.osType) like '%os%' then 'ios' else lower(r.osType) end),
            (case when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%andro%' then 'android'
                  when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%os%' then 'ios'
                  when lower(loanmaster.deviceType) like '%andro%' then 'android'
                  else 'ios' end)
            ) as osType
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
  qualify row_number() over(partition by digitalLoanAccountId, modelVersionId order by appln_submit_datetime) = 1
  ;
"""
dfd = client.query(sq).to_dataframe()
# dfd = dfd.drop_duplicates(keep='first')
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()


# %%
df1 = dfd.copy()


# ### Train

# %%
sq = """WITH parsed as (
select customerId, digitalLoanAccountId,modelDisplayName,modelVersionId,start_time,end_time,prediction,trenchCategory,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeatures,    deviceOs osType,
Data_selection
FROM prj-prod-dataplatform.dap_ds_poweruser_playground.ml_training_model_run_details_20260116
where modelDisplayName in  ('Beta-Cash-AppScore-Model', 'apps_score_cash')
),
  modelname as (
  select customerId,
  digitalLoanAccountId,
  start_time,
  prediction beta_cash_app_score,
  case when modelDisplayName like 'Beta-Cash-AppScore-Model' then 'apps_score_cash' else modelDisplayName end as modelDisplayName,
  modelVersionId,
  trenchCategory,
  Data_selection, osType
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
  Data_selection,
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
      coalesce((case when lower(r.osType) like '%andro%' then 'android'
                  when lower(r.osType) like '%os%' then 'ios' else lower(r.osType) end),
            (case when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%andro%' then 'android'
                  when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%os%' then 'ios'
                  when lower(loanmaster.deviceType) like '%andro%' then 'android'
                  else 'ios' end)
            ) as osType
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
  select * from base
  qualify row_number() over(partition by digitalLoanAccountId, modelVersionId order by appln_submit_datetime) = 1
  ;
  """
dfd = client.query(sq).to_dataframe()
# dfd = dfd.drop_duplicates(keep='first')
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()

# %%
df2 = dfd.copy()


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
    data_selection_column="Data_selection",
    model_version_column="modelVersionId",
    trench_column="trenchCategory",
    loan_type_column="new_loan_type",
    loan_product_type_column="loan_product_type",
    ostype_column="osType",
    account_id_column="digitalLoanAccountId",
)

# %%
fact_table, dimension_table = update_tables(
    fact_table, dimension_table, model_name="apps_score_cash", product="CASH"
)

df_f_fpd10_appscorecash = fact_table.copy()
df_d_fpd10_appscorecash = dimension_table.copy()

# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_f_fpd10_appscorecash, facttable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimensi1on_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_d_fpd10_appscorecash, dimtable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# %%


# ### FPD30


# ### Test

# %%
sq = """
WITH parsed as (
select customerId, digitalLoanAccountId,modelDisplayName,modelVersionId,start_time,end_time,prediction,trenchCategory,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeatures,
REPLACE(REPLACE(prediction, "'", '"'), "None", "null") AS prediction_clean,
deviceOs osType,
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
  coalesce(p.trenchCategory, JSON_VALUE(m.requestPayload_clean, "$.predictions.trenchCategory"))  trenchCategory, osType
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
  'Prod' Data_selection,
  del.deffpd30,
  del.flg_mature_fpd30,
  loanmaster.new_loan_type,
  modelVersionId, trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile'
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall'
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
        coalesce((case when lower(r.osType) like '%andro%' then 'android'
                  when lower(r.osType) like '%os%' then 'ios' else lower(r.osType) end),
            (case when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%andro%' then 'android'
                  when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%os%' then 'ios'
                  when lower(loanmaster.deviceType) like '%andro%' then 'android'
                  else 'ios' end)
            ) as osType
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
  qualify row_number() over(partition by digitalLoanAccountId, modelVersionId order by appln_submit_datetime) = 1
  ;
"""
dfd = client.query(sq).to_dataframe()
# dfd = dfd.drop_duplicates(keep='first')
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()


# %%
df1 = dfd.copy()


# ### Train

# %%
sq = """WITH parsed as (
select customerId, digitalLoanAccountId,modelDisplayName,modelVersionId,start_time,end_time,prediction,trenchCategory,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeatures,
Data_selection, deviceOs osType,
FROM prj-prod-dataplatform.dap_ds_poweruser_playground.ml_training_model_run_details_20260116
where modelDisplayName in  ('Beta-Cash-AppScore-Model', 'apps_score_cash')
),
  modelname as (
  select customerId,
  digitalLoanAccountId,
  start_time,
  prediction beta_cash_app_score,
  case when modelDisplayName like 'Beta-Cash-AppScore-Model' then 'apps_score_cash' else modelDisplayName end as modelDisplayName,
  modelVersionId,
  trenchCategory,
  Data_selection, osType
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
  Data_selection,
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
        coalesce((case when lower(r.osType) like '%andro%' then 'android'
                  when lower(r.osType) like '%os%' then 'ios' else lower(r.osType) end),
            (case when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%andro%' then 'android'
                  when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%os%' then 'ios'
                  when lower(loanmaster.deviceType) like '%andro%' then 'android'
                  else 'ios' end)
            ) as osType
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
  select * from base
  qualify row_number() over(partition by digitalLoanAccountId, modelVersionId order by appln_submit_datetime) = 1
  ;
  """
dfd = client.query(sq).to_dataframe()
# dfd = dfd.drop_duplicates(keep='first')
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()

# %%
df2 = dfd.copy()


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
    data_selection_column="Data_selection",
    model_version_column="modelVersionId",
    trench_column="trenchCategory",
    loan_type_column="new_loan_type",
    loan_product_type_column="loan_product_type",
    ostype_column="osType",
    account_id_column="digitalLoanAccountId",
)

# %%
fact_table, dimension_table = update_tables(
    fact_table, dimension_table, model_name="apps_score_cash", product="CASH"
)

df_f_fpd30_appscorecash = fact_table.copy()
df_d_fpd30_appscorecash = dimension_table.copy()

# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_f_fpd30_appscorecash, facttable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimensi1on_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_d_fpd30_appscorecash, dimtable_id, job_config=job_config
)
job.result()  # Wait for the job to complete


# %%


# ### FSPD30


# ### Test

# %%
sq = """
WITH parsed as (
select customerId, digitalLoanAccountId,modelDisplayName,modelVersionId,start_time,end_time,prediction,trenchCategory,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeatures,
REPLACE(REPLACE(prediction, "'", '"'), "None", "null") AS prediction_clean,     deviceOs osType,
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
  coalesce(p.trenchCategory, JSON_VALUE(m.requestPayload_clean, "$.predictions.trenchCategory"))  trenchCategory, osType
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
  'Prod' Data_selection,
  del.deffspd30,
  del.flg_mature_fspd_30,
  loanmaster.new_loan_type,
  modelVersionId, trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile'
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall'
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
        coalesce((case when lower(r.osType) like '%andro%' then 'android'
                  when lower(r.osType) like '%os%' then 'ios' else lower(r.osType) end),
            (case when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%andro%' then 'android'
                  when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%os%' then 'ios'
                  when lower(loanmaster.deviceType) like '%andro%' then 'android'
                  else 'ios' end)
            ) as osType
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
  qualify row_number() over(partition by digitalLoanAccountId, modelVersionId order by appln_submit_datetime) = 1
  ;
"""
dfd = client.query(sq).to_dataframe()
# dfd = dfd.drop_duplicates(keep='first')
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()


# %%
df1 = dfd.copy()


# ### Train

# %%
sq = """WITH parsed as (
select customerId, digitalLoanAccountId,modelDisplayName,modelVersionId,start_time,end_time,prediction,trenchCategory,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeatures,
Data_selection,     deviceOs osType,
FROM prj-prod-dataplatform.dap_ds_poweruser_playground.ml_training_model_run_details_20260116
where modelDisplayName in  ('Beta-Cash-AppScore-Model', 'apps_score_cash')
),
  modelname as (
  select customerId,
  digitalLoanAccountId,
  start_time,
  prediction beta_cash_app_score,
  case when modelDisplayName like 'Beta-Cash-AppScore-Model' then 'apps_score_cash' else modelDisplayName end as modelDisplayName,
  modelVersionId,
  trenchCategory,
  Data_selection, osType
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
  Data_selection,
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
        coalesce((case when lower(r.osType) like '%andro%' then 'android'
                  when lower(r.osType) like '%os%' then 'ios' else lower(r.osType) end),
            (case when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%andro%' then 'android'
                  when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%os%' then 'ios'
                  when lower(loanmaster.deviceType) like '%andro%' then 'android'
                  else 'ios' end)
            ) as osType
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
  select * from base
  qualify row_number() over(partition by digitalLoanAccountId, modelVersionId order by appln_submit_datetime) = 1
  ;
  """
dfd = client.query(sq).to_dataframe()
# dfd = dfd.drop_duplicates(keep='first')
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()

# %%
df2 = dfd.copy()


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
    data_selection_column="Data_selection",
    model_version_column="modelVersionId",
    trench_column="trenchCategory",
    loan_type_column="new_loan_type",
    loan_product_type_column="loan_product_type",
    ostype_column="osType",
    account_id_column="digitalLoanAccountId",
)

# %%
fact_table, dimension_table = update_tables(
    fact_table, dimension_table, model_name="apps_score_cash", product="CASH"
)

# %%
df_f_fspd30_appscorecash = fact_table.copy()
df_d_fspd30_appscorecash = dimension_table.copy()

# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_f_fspd30_appscorecash, facttable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimensi1on_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_d_fspd30_appscorecash, dimtable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# %%


# ### FSTPD30


# ### Test

# %%
sq = """
WITH parsed as (
select customerId, digitalLoanAccountId,modelDisplayName,modelVersionId,start_time,end_time,prediction,trenchCategory,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeatures,
REPLACE(REPLACE(prediction, "'", '"'), "None", "null") AS prediction_clean,
deviceOs osType,
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
  coalesce(p.trenchCategory, JSON_VALUE(m.requestPayload_clean, "$.predictions.trenchCategory"))  trenchCategory, osType
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
  'Prod' Data_selection,
  del.deffstpd30,
  del.flg_mature_fstpd_30,
  loanmaster.new_loan_type,
  modelVersionId, trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile'
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall'
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
        coalesce((case when lower(r.osType) like '%andro%' then 'android'
                  when lower(r.osType) like '%os%' then 'ios' else lower(r.osType) end),
            (case when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%andro%' then 'android'
                  when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%os%' then 'ios'
                  when lower(loanmaster.deviceType) like '%andro%' then 'android'
                  else 'ios' end)
            ) as osType
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
  qualify row_number() over(partition by digitalLoanAccountId, modelVersionId order by appln_submit_datetime) = 1
  ;
"""
dfd = client.query(sq).to_dataframe()
# dfd = dfd.drop_duplicates(keep='first')
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()


# %%
df1 = dfd.copy()


# ### Train

# %%
sq = """WITH parsed as (
select customerId, digitalLoanAccountId,modelDisplayName,modelVersionId,start_time,end_time,prediction,trenchCategory,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeatures,
Data_selection, deviceOs osType,
FROM prj-prod-dataplatform.dap_ds_poweruser_playground.ml_training_model_run_details_20260116
where modelDisplayName in  ('Beta-Cash-AppScore-Model', 'apps_score_cash')
),
  modelname as (
  select customerId,
  digitalLoanAccountId,
  start_time,
  prediction beta_cash_app_score,
  case when modelDisplayName like 'Beta-Cash-AppScore-Model' then 'apps_score_cash' else modelDisplayName end as modelDisplayName,
  modelVersionId,
  trenchCategory,
  Data_selection, osType
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
  Data_selection,
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
        coalesce((case when lower(r.osType) like '%andro%' then 'android'
                  when lower(r.osType) like '%os%' then 'ios' else lower(r.osType) end),
            (case when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%andro%' then 'android'
                  when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%os%' then 'ios'
                  when lower(loanmaster.deviceType) like '%andro%' then 'android'
                  else 'ios' end)
            ) as osType
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
  select * from base
  qualify row_number() over(partition by digitalLoanAccountId, modelVersionId order by appln_submit_datetime) = 1
  ;
  """
dfd = client.query(sq).to_dataframe()
# dfd = dfd.drop_duplicates(keep='first')
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()

# %%
df2 = dfd.copy()


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
    data_selection_column="Data_selection",
    model_version_column="modelVersionId",
    trench_column="trenchCategory",
    loan_type_column="new_loan_type",
    loan_product_type_column="loan_product_type",
    ostype_column="osType",
    account_id_column="digitalLoanAccountId",
)

# %%
fact_table, dimension_table = update_tables(
    fact_table, dimension_table, model_name="apps_score_cash", product="CASH"
)
print(f"The shape of the fact table is:\t {fact_table.shape}")
print(f"The shape of the dimension table is:\t {dimension_table.shape}")

df_f_fstpd30_appscorecash = fact_table.copy()
df_d_fstpd30_appscorecash = dimension_table.copy()

# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_f_fstpd30_appscorecash, facttable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimensi1on_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_d_fstpd30_appscorecash, dimtable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

factappscorecash = pd.concat(
    [
        df_f_fpd0_appscorecash,
        df_f_fpd10_appscorecash,
        df_f_fpd30_appscorecash,
        df_f_fspd30_appscorecash,
        df_f_fstpd30_appscorecash,
    ],
    ignore_index=True,
)
dimappscorecash = pd.concat(
    [
        df_d_fpd0_appscorecash,
        df_d_fpd10_appscorecash,
        df_d_fpd30_appscorecash,
        df_d_fspd30_appscorecash,
        df_d_fstpd30_appscorecash,
    ],
    ignore_index=True,
)


# %%

# %% [markdown]
# #### Beta-Cash-Demo-Model

# %%

facttable_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_betademocash"
dimtable_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_betademocash"

# ## Beta-Cash-Demo-Model


# ### FPD0


# ### Test

# %%
sq = """
WITH parsed as (
select customerId, digitalLoanAccountId,modelDisplayName,modelVersionId,start_time,end_time,prediction,trenchCategory,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeatures,
deviceOs osType,
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
  p.modelVersionId,
  osType,
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
  'Prod' Data_selection,
  deffpd0,
  flg_mature_fpd0,
  loanmaster.new_loan_type,
  modelVersionId, trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile'
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall'
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
       coalesce((case when lower(r.osType) like '%andro%' then 'android'
                  when lower(r.osType) like '%os%' then 'ios' else lower(r.osType) end),
            (case when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%andro%' then 'android'
                  when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%os%' then 'ios'
                  when lower(loanmaster.deviceType) like '%andro%' then 'android'
                  else 'ios' end)
            ) as osType
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
  qualify row_number() over(partition by digitalLoanAccountId, modelVersionId order by appln_submit_datetime) = 1
  ;
"""
dfd = client.query(sq).to_dataframe()
# dfd = dfd.drop_duplicates(keep='first')
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()


# %%
df1 = dfd.copy()


# ### Train

# %%
sq = """
WITH parsed as (
select customerId, digitalLoanAccountId,modelDisplayName,modelVersionId,start_time,end_time,prediction,trenchCategory,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeatures,
Data_selection,
  deviceOs osType,
FROM prj-prod-dataplatform.dap_ds_poweruser_playground.ml_training_model_run_details_20260116
where modelDisplayName in ('Beta-Cash-Demo-Model', 'beta_demo_model_cash')
),
  modelname as (
  select customerId,
  digitalLoanAccountId,
  start_time,
  prediction Beta_Cash_Demo_Score,
  case when modelDisplayName like 'Beta-Cash-Demo-Model' then 'beta_demo_model_cash' else modelDisplayName end as modelDisplayName,
  modelVersionId,
  trenchCategory,
  Data_selection,
     osType,
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
  Data_selection,
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
        coalesce((case when lower(r.osType) like '%andro%' then 'android'
                  when lower(r.osType) like '%os%' then 'ios' else lower(r.osType) end),
            (case when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%andro%' then 'android'
                  when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%os%' then 'ios'
                  when lower(loanmaster.deviceType) like '%andro%' then 'android'
                  else 'ios' end)
            ) as osType
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
  select * from base
  qualify row_number() over(partition by digitalLoanAccountId, modelVersionId order by appln_submit_datetime) = 1
  ;
  """
dfd = client.query(sq).to_dataframe()
# dfd = dfd.drop_duplicates(keep='first')
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()

# %%
df2 = dfd.copy()


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
    data_selection_column="Data_selection",
    model_version_column="modelVersionId",
    trench_column="trenchCategory",
    loan_type_column="new_loan_type",
    loan_product_type_column="loan_product_type",
    ostype_column="osType",
    account_id_column="digitalLoanAccountId",
)

# %%
fact_table, dimension_table = update_tables(
    fact_table, dimension_table, model_name="beta_demo_model_cash", product="CASH"
)
print(f"The shape of the fact table is:\t {fact_table.shape}")
print(f"The shape of the dimension table is:\t {dimension_table.shape}")

df_f_fpd0_betademocash = fact_table.copy()
df_d_fpd0_betademocash = dimension_table.copy()


# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_TRUNCATE",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_f_fpd0_betademocash, facttable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimensi1on_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_TRUNCATE",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_d_fpd0_betademocash, dimtable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# ### FPD10


# ### Test

# %%
sq = """
WITH parsed as (
select customerId, digitalLoanAccountId,modelDisplayName,modelVersionId,start_time,end_time,prediction,trenchCategory,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeatures,  deviceOs osType,
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
  p.modelVersionId,  osType,
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
  'Prod' Data_selection,
  del.deffpd10,
  del.flg_mature_fpd10,
  loanmaster.new_loan_type,
  modelVersionId, trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile'
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall'
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
        coalesce((case when lower(r.osType) like '%andro%' then 'android'
                  when lower(r.osType) like '%os%' then 'ios' else lower(r.osType) end),
            (case when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%andro%' then 'android'
                  when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%os%' then 'ios'
                  when lower(loanmaster.deviceType) like '%andro%' then 'android'
                  else 'ios' end)
            ) as osType
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
  qualify row_number() over(partition by digitalLoanAccountId, modelVersionId order by appln_submit_datetime) = 1
  ;
"""
dfd = client.query(sq).to_dataframe()
# dfd = dfd.drop_duplicates(keep='first')
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()


# %%
df1 = dfd.copy()


# ### Train

# %%
sq = """
WITH parsed as (
select customerId, digitalLoanAccountId,modelDisplayName,modelVersionId,start_time,end_time,prediction,trenchCategory,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeatures,
Data_selection, deviceOs osType,
FROM prj-prod-dataplatform.dap_ds_poweruser_playground.ml_training_model_run_details_20260116
where modelDisplayName in ('Beta-Cash-Demo-Model', 'beta_demo_model_cash')
),
  modelname as (
  select customerId,
  digitalLoanAccountId,
  start_time,
  prediction Beta_Cash_Demo_Score,
  case when modelDisplayName like 'Beta-Cash-Demo-Model' then 'beta_demo_model_cash' else modelDisplayName end as modelDisplayName,
  modelVersionId,
  trenchCategory,
  Data_selection, osType
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
  Data_selection,
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
        coalesce((case when lower(r.osType) like '%andro%' then 'android'
                  when lower(r.osType) like '%os%' then 'ios' else lower(r.osType) end),
            (case when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%andro%' then 'android'
                  when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%os%' then 'ios'
                  when lower(loanmaster.deviceType) like '%andro%' then 'android'
                  else 'ios' end)
            ) as osType
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
  select * from base
  qualify row_number() over(partition by digitalLoanAccountId, modelVersionId order by appln_submit_datetime) = 1;
  """
dfd = client.query(sq).to_dataframe()
# dfd = dfd.drop_duplicates(keep='first')
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()

# %%
df2 = dfd.copy()


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
    data_selection_column="Data_selection",
    model_version_column="modelVersionId",
    trench_column="trenchCategory",
    loan_type_column="new_loan_type",
    loan_product_type_column="loan_product_type",
    ostype_column="osType",
    account_id_column="digitalLoanAccountId",
)

# %%
fact_table, dimension_table = update_tables(
    fact_table, dimension_table, model_name="beta_demo_model_cash", product="CASH"
)

df_f_fpd10_betademocash = fact_table.copy()
df_d_fpd10_betademocash = dimension_table.copy()
# %%
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_f_fpd10_betademocash, facttable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimensi1on_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_d_fpd10_betademocash, dimtable_id, job_config=job_config
)
job.result()  # Wait for the job to complete


# ### FPD30


# ### Test

# %%
sq = """
WITH parsed as (
select customerId, digitalLoanAccountId,modelDisplayName,modelVersionId,start_time,end_time,prediction,trenchCategory,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeatures,     deviceOs osType,
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
  p.modelVersionId, osType
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
  'Prod' Data_selection,
  del.deffpd30,
  del.flg_mature_fpd30,
  loanmaster.new_loan_type,
  modelVersionId, trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile'
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall'
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
        coalesce((case when lower(r.osType) like '%andro%' then 'android'
                  when lower(r.osType) like '%os%' then 'ios' else lower(r.osType) end),
            (case when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%andro%' then 'android'
                  when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%os%' then 'ios'
                  when lower(loanmaster.deviceType) like '%andro%' then 'android'
                  else 'ios' end)
            ) as osType
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
  qualify row_number() over(partition by digitalLoanAccountId, modelVersionId order by appln_submit_datetime) = 1
  ;
"""
dfd = client.query(sq).to_dataframe()
# dfd = dfd.drop_duplicates(keep='first')
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()


# %%
df1 = dfd.copy()


# ### Train

# %%
sq = """
WITH parsed as (
select customerId, digitalLoanAccountId,modelDisplayName,modelVersionId,start_time,end_time,prediction,trenchCategory,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeatures,
Data_selection,   deviceOs osType,
FROM prj-prod-dataplatform.dap_ds_poweruser_playground.ml_training_model_run_details_20260116
where modelDisplayName in ('Beta-Cash-Demo-Model', 'beta_demo_model_cash')
),
  modelname as (
  select customerId,
  digitalLoanAccountId,
  start_time,
  prediction Beta_Cash_Demo_Score,
  case when modelDisplayName like 'Beta-Cash-Demo-Model' then 'beta_demo_model_cash' else modelDisplayName end as modelDisplayName,
  modelVersionId,
  trenchCategory,
  Data_selection, osType
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
  Data_selection,
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
        coalesce((case when lower(r.osType) like '%andro%' then 'android'
                  when lower(r.osType) like '%os%' then 'ios' else lower(r.osType) end),
            (case when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%andro%' then 'android'
                  when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%os%' then 'ios'
                  when lower(loanmaster.deviceType) like '%andro%' then 'android'
                  else 'ios' end)
            ) as osType
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
  select * from base
  qualify row_number() over(partition by digitalLoanAccountId, modelVersionId order by appln_submit_datetime) = 1
  ;
  """
dfd = client.query(sq).to_dataframe()
# dfd = dfd.drop_duplicates(keep='first')
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()

# %%
df2 = dfd.copy()


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
    data_selection_column="Data_selection",
    model_version_column="modelVersionId",
    trench_column="trenchCategory",
    loan_type_column="new_loan_type",
    loan_product_type_column="loan_product_type",
    ostype_column="osType",
    account_id_column="digitalLoanAccountId",
)

# %%
fact_table, dimension_table = update_tables(
    fact_table, dimension_table, model_name="beta_demo_model_cash", product="CASH"
)

df_f_fpd30_betademocash = fact_table.copy()
df_d_fpd30_betademocash = dimension_table.copy()

job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_f_fpd30_betademocash, facttable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimensi1on_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_d_fpd30_betademocash, dimtable_id, job_config=job_config
)
job.result()  # Wait for the job to complete


# %%


# ### FSPD30


# ### Test

# %%
sq = """
WITH parsed as (
select customerId, digitalLoanAccountId,modelDisplayName,modelVersionId,start_time,end_time,prediction,trenchCategory,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeatures,
    deviceOs osType,
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
  p.modelVersionId, osType
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
  'Prod' Data_selection,
  del.deffspd30,
  del.flg_mature_fspd_30,
  loanmaster.new_loan_type,
  modelVersionId, trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile'
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall'
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,

    coalesce((case when lower(r.osType) like '%andro%' then 'android'
                  when lower(r.osType) like '%os%' then 'ios' else lower(r.osType) end),
            (case when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%andro%' then 'android'
                  when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%os%' then 'ios'
                  when lower(loanmaster.deviceType) like '%andro%' then 'android'
                  else 'ios' end)
            ) as osType
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
  qualify row_number() over(partition by digitalLoanAccountId, modelVersionId order by appln_submit_datetime) = 1
  ;
"""
dfd = client.query(sq).to_dataframe()
# dfd = dfd.drop_duplicates(keep='first')
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()


# %%
df1 = dfd.copy()


# ### Train

# %%
sq = """
WITH parsed as (
select customerId, digitalLoanAccountId,modelDisplayName,modelVersionId,start_time,end_time,prediction,trenchCategory,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeatures,
Data_selection,     deviceOs osType,
FROM prj-prod-dataplatform.dap_ds_poweruser_playground.ml_training_model_run_details_20260116
where modelDisplayName in ('Beta-Cash-Demo-Model', 'beta_demo_model_cash')
),
  modelname as (
  select customerId,
  digitalLoanAccountId,
  start_time,
  prediction Beta_Cash_Demo_Score,
  case when modelDisplayName like 'Beta-Cash-Demo-Model' then 'beta_demo_model_cash' else modelDisplayName end as modelDisplayName,
  modelVersionId,
  trenchCategory,
  Data_selection, osType
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
  Data_selection,
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
        coalesce((case when lower(r.osType) like '%andro%' then 'android'
                  when lower(r.osType) like '%os%' then 'ios' else lower(r.osType) end),
            (case when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%andro%' then 'android'
                  when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%os%' then 'ios'
                  when lower(loanmaster.deviceType) like '%andro%' then 'android'
                  else 'ios' end)
            ) as osType
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
  select * from base
  qualify row_number() over(partition by digitalLoanAccountId, modelVersionId order by appln_submit_datetime) = 1
  ;
  """
dfd = client.query(sq).to_dataframe()
# dfd = dfd.drop_duplicates(keep='first')
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()

# %%
df2 = dfd.copy()


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
    data_selection_column="Data_selection",
    model_version_column="modelVersionId",
    trench_column="trenchCategory",
    loan_type_column="new_loan_type",
    loan_product_type_column="loan_product_type",
    ostype_column="osType",
    account_id_column="digitalLoanAccountId",
)

# %%
fact_table, dimension_table = update_tables(
    fact_table, dimension_table, model_name="beta_demo_model_cash", product="CASH"
)

df_f_fspd30_betademocash = fact_table.copy()
df_d_fspd30_betademocash = dimension_table.copy()

job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_f_fspd30_betademocash, facttable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimensi1on_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_d_fspd30_betademocash, dimtable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# %%


# ### FSTPD30


# ### Test

# %%
sq = """
WITH parsed as (
select customerId, digitalLoanAccountId,modelDisplayName,modelVersionId,start_time,end_time,prediction,trenchCategory,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeatures,   deviceOs osType,
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
  p.modelVersionId, osType
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
  'Prod' Data_selection,
  del.deffstpd30,
  del.flg_mature_fstpd_30,
  loanmaster.new_loan_type,
  modelVersionId, trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile'
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall'
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
        coalesce((case when lower(r.osType) like '%andro%' then 'android'
                  when lower(r.osType) like '%os%' then 'ios' else lower(r.osType) end),
            (case when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%andro%' then 'android'
                  when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%os%' then 'ios'
                  when lower(loanmaster.deviceType) like '%andro%' then 'android'
                  else 'ios' end)
            ) as osType
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
  qualify row_number() over(partition by digitalLoanAccountId, modelVersionId order by appln_submit_datetime) = 1
  ;
"""
dfd = client.query(sq).to_dataframe()
# dfd = dfd.drop_duplicates(keep='first')
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()


# %%
df1 = dfd.copy()


# ### Train

# %%
sq = """
WITH parsed as (
select customerId, digitalLoanAccountId,modelDisplayName,modelVersionId,start_time,end_time,prediction,trenchCategory,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeatures,   deviceOs osType,
Data_selection
FROM prj-prod-dataplatform.dap_ds_poweruser_playground.ml_training_model_run_details_20260116
where modelDisplayName in ('Beta-Cash-Demo-Model', 'beta_demo_model_cash')
),
  modelname as (
  select customerId,
  digitalLoanAccountId,
  start_time,
  prediction Beta_Cash_Demo_Score,
  case when modelDisplayName like 'Beta-Cash-Demo-Model' then 'beta_demo_model_cash' else modelDisplayName end as modelDisplayName,
  modelVersionId,
  trenchCategory,
  Data_selection, osType
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
  Data_selection,
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
        coalesce((case when lower(r.osType) like '%andro%' then 'android'
                  when lower(r.osType) like '%os%' then 'ios' else lower(r.osType) end),
            (case when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%andro%' then 'android'
                  when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%os%' then 'ios'
                  when lower(loanmaster.deviceType) like '%andro%' then 'android'
                  else 'ios' end)
            ) as osType
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
  select * from base
  qualify row_number() over(partition by digitalLoanAccountId, modelVersionId order by appln_submit_datetime) = 1
  ;
  """
dfd = client.query(sq).to_dataframe()
# dfd = dfd.drop_duplicates(keep='first')
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()

# %%
df2 = dfd.copy()


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
    data_selection_column="Data_selection",
    model_version_column="modelVersionId",
    trench_column="trenchCategory",
    loan_type_column="new_loan_type",
    loan_product_type_column="loan_product_type",
    ostype_column="osType",
    account_id_column="digitalLoanAccountId",
)

# %%
fact_table, dimension_table = update_tables(
    fact_table, dimension_table, model_name="beta_demo_model_cash", product="CASH"
)
print(f"The shape of the fact table is:\t {fact_table.shape}")
print(f"The shape of the dimension table is:\t {dimension_table.shape}")

df_f_fstpd30_betademocash = fact_table.copy()
df_d_fstpd30_betademocash = dimension_table.copy()

job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_f_fstpd30_betademocash, facttable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimensi1on_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_d_fstpd30_betademocash, dimtable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

factbetademocash = pd.concat(
    [
        df_f_fpd0_betademocash,
        df_f_fpd10_betademocash,
        df_f_fpd30_betademocash,
        df_f_fspd30_betademocash,
        df_f_fstpd30_betademocash,
    ],
    ignore_index=True,
)
dimbetademocash = pd.concat(
    [
        df_d_fpd0_betademocash,
        df_d_fpd10_betademocash,
        df_d_fpd30_betademocash,
        df_d_fspd30_betademocash,
        df_d_fstpd30_betademocash,
    ],
    ignore_index=True,
)


print("beta_demo_model_cash gini calculation completed")


# %% [markdown]
# #### Beta-Cash-Stack-Model

# %%

facttable_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_betastackcash"
dimtable_id = (
    "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_betastackcash"
)

# %%
# ## Beta-Cash-Stack-Model


# ### FPD0


# ### Test

# %%

sq = """
WITH parsed as (
select customerId, digitalLoanAccountId,modelDisplayName,modelVersionId,start_time,end_time,prediction,trenchCategory,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeatures,
REPLACE(REPLACE(prediction, "'", '"'), "None", "null") AS prediction_clean,
 deviceOs osType,
FROM `prj-prod-dataplatform.audit_balance.ml_model_run_details`
where modelDisplayName in  ('Beta-Cash-Stack-Model', 'beta_stack_model_cash')
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
  p.modelVersionId, osType
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
  'Prod' Data_selection,
  deffpd0,
  flg_mature_fpd0,
  loanmaster.new_loan_type,
  modelVersionId, trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile'
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall'
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
        coalesce((case when lower(r.osType) like '%andro%' then 'android'
                  when lower(r.osType) like '%os%' then 'ios' else lower(r.osType) end),
            (case when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%andro%' then 'android'
                  when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%os%' then 'ios'
                  when lower(loanmaster.deviceType) like '%andro%' then 'android'
                  else 'ios' end)
            ) as osType
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
  qualify row_number() over(partition by digitalLoanAccountId, modelVersionId order by appln_submit_datetime) = 1
  ;
"""
dfd = client.query(sq).to_dataframe()
# dfd = dfd.drop_duplicates(keep='first')
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()


# %%
df1 = dfd.copy()


# ### Train

# %%
sq = """
WITH parsed as (
select customerId, digitalLoanAccountId,modelDisplayName,modelVersionId,start_time,end_time,prediction,trenchCategory,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeatures,
Data_selection,  deviceOs osType,
FROM prj-prod-dataplatform.dap_ds_poweruser_playground.ml_training_model_run_details_20260116
where modelDisplayName in  ('Beta-Cash-Stack-Model', 'beta_stack_model_cash')
),
  modelname as (
  select customerId,
  digitalLoanAccountId,
  start_time,
  prediction Beta_cash_stack_score,
    case when modelDisplayName like 'Beta-Cash-Stack-Model' then 'beta_stack_model_cash' else modelDisplayName end as modelDisplayName,
    modelVersionId,
  trenchCategory,
  Data_selection, osType
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
  Data_selection,
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
        coalesce((case when lower(r.osType) like '%andro%' then 'android'
                  when lower(r.osType) like '%os%' then 'ios' else lower(r.osType) end),
            (case when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%andro%' then 'android'
                  when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%os%' then 'ios'
                  when lower(loanmaster.deviceType) like '%andro%' then 'android'
                  else 'ios' end)
            ) as osType
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
  select * from base
  qualify row_number() over(partition by digitalLoanAccountId, modelVersionId order by appln_submit_datetime) = 1
  ;
  """
dfd = client.query(sq).to_dataframe()
# dfd = dfd.drop_duplicates(keep='first')
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()

# %%
df2 = dfd.copy()


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
    data_selection_column="Data_selection",
    model_version_column="modelVersionId",
    trench_column="trenchCategory",
    loan_type_column="new_loan_type",
    loan_product_type_column="loan_product_type",
    ostype_column="osType",
    account_id_column="digitalLoanAccountId",
)

# %%
fact_table, dimension_table = update_tables(
    fact_table, dimension_table, model_name="beta_stack_model_cash", product="CASH"
)
print(f"The shape of the fact table is:\t {fact_table.shape}")
print(f"The shape of the dimension table is:\t {dimension_table.shape}")

df_f_fpd0_betastackcash = fact_table.copy()
df_d_fpd0_betastackcash = dimension_table.copy()


job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_TRUNCATE",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_f_fpd0_betastackcash, facttable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimensi1on_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_TRUNCATE",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_d_fpd0_betastackcash, dimtable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# ### FPD10


# ### Test

# %%
sq = """
WITH parsed as (
select customerId, digitalLoanAccountId,modelDisplayName,modelVersionId,start_time,end_time,prediction,trenchCategory,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeatures,
REPLACE(REPLACE(prediction, "'", '"'), "None", "null") AS prediction_clean,
deviceOs osType,
FROM `prj-prod-dataplatform.audit_balance.ml_model_run_details`
where modelDisplayName in  ('Beta-Cash-Stack-Model', 'beta_stack_model_cash')
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
  p.modelVersionId, osType
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
  'Prod' Data_selection,
  del.deffpd10,
  del.flg_mature_fpd10,
  loanmaster.new_loan_type,
  modelVersionId, trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile'
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall'
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
        coalesce((case when lower(r.osType) like '%andro%' then 'android'
                  when lower(r.osType) like '%os%' then 'ios' else lower(r.osType) end),
            (case when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%andro%' then 'android'
                  when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%os%' then 'ios'
                  when lower(loanmaster.deviceType) like '%andro%' then 'android'
                  else 'ios' end)
            ) as osType
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
  qualify row_number() over(partition by digitalLoanAccountId, modelVersionId order by appln_submit_datetime) = 1
  ;
"""
dfd = client.query(sq).to_dataframe()
# dfd = dfd.drop_duplicates(keep='first')
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()


# %%
df1 = dfd.copy()


# ### Train

# %%
sq = """
WITH parsed as (
select customerId, digitalLoanAccountId,modelDisplayName,modelVersionId,start_time,end_time,prediction,trenchCategory,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeatures,
Data_selection, deviceOs osType,
FROM prj-prod-dataplatform.dap_ds_poweruser_playground.ml_training_model_run_details_20260116
where modelDisplayName in  ('Beta-Cash-Stack-Model', 'beta_stack_model_cash')
),
  modelname as (
  select customerId,
  digitalLoanAccountId,
  start_time,
  prediction Beta_cash_stack_score,
    case when modelDisplayName like 'Beta-Cash-Stack-Model' then 'beta_stack_model_cash' else modelDisplayName end as modelDisplayName,
    modelVersionId,
  trenchCategory,
  Data_selection, osType
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
  Data_selection,
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
        coalesce((case when lower(r.osType) like '%andro%' then 'android'
                  when lower(r.osType) like '%os%' then 'ios' else lower(r.osType) end),
            (case when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%andro%' then 'android'
                  when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%os%' then 'ios'
                  when lower(loanmaster.deviceType) like '%andro%' then 'android'
                  else 'ios' end)
            ) as osType
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
  select * from base
  qualify row_number() over(partition by digitalLoanAccountId, modelVersionId order by appln_submit_datetime) = 1
  ;
  """
dfd = client.query(sq).to_dataframe()
# dfd = dfd.drop_duplicates(keep='first')
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()

# %%
df2 = dfd.copy()


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
    data_selection_column="Data_selection",
    model_version_column="modelVersionId",
    trench_column="trenchCategory",
    loan_type_column="new_loan_type",
    loan_product_type_column="loan_product_type",
    ostype_column="osType",
    account_id_column="digitalLoanAccountId",
)

# %%
fact_table, dimension_table = update_tables(
    fact_table, dimension_table, model_name="beta_stack_model_cash", product="CASH"
)

df_f_fpd10_betastackcash = fact_table.copy()
df_d_fpd10_betastackcash = dimension_table.copy()

job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_f_fpd10_betastackcash, facttable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimensi1on_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_d_fpd10_betastackcash, dimtable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# %%


# ### FPD30


# ### Test

# %%
sq = """
WITH parsed as (
select customerId, digitalLoanAccountId,modelDisplayName,modelVersionId,start_time,end_time,prediction,trenchCategory,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeatures,
REPLACE(REPLACE(prediction, "'", '"'), "None", "null") AS prediction_clean,
deviceOs osType,
FROM `prj-prod-dataplatform.audit_balance.ml_model_run_details`
where modelDisplayName in  ('Beta-Cash-Stack-Model', 'beta_stack_model_cash')
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
  p.modelVersionId, osType
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
  'Prod' Data_selection,
  del.deffpd30,
  del.flg_mature_fpd30,
  loanmaster.new_loan_type,
  modelVersionId, trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile'
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall'
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
        coalesce((case when lower(r.osType) like '%andro%' then 'android'
                  when lower(r.osType) like '%os%' then 'ios' else lower(r.osType) end),
            (case when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%andro%' then 'android'
                  when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%os%' then 'ios'
                  when lower(loanmaster.deviceType) like '%andro%' then 'android'
                  else 'ios' end)
            ) as osType
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
  qualify row_number() over(partition by digitalLoanAccountId, modelVersionId order by appln_submit_datetime) = 1
  ;
"""
dfd = client.query(sq).to_dataframe()
# dfd = dfd.drop_duplicates(keep='first')
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()


# %%
df1 = dfd.copy()


# ### Train

# %%
sq = """
WITH parsed as (
select customerId, digitalLoanAccountId,modelDisplayName,modelVersionId,start_time,end_time,prediction,trenchCategory,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeatures,
Data_selection, deviceOs osType,
FROM prj-prod-dataplatform.dap_ds_poweruser_playground.ml_training_model_run_details_20260116
where modelDisplayName in  ('Beta-Cash-Stack-Model', 'beta_stack_model_cash')
),
  modelname as (
  select customerId,
  digitalLoanAccountId,
  start_time,
  prediction Beta_cash_stack_score,
    case when modelDisplayName like 'Beta-Cash-Stack-Model' then 'beta_stack_model_cash' else modelDisplayName end as modelDisplayName,
    modelVersionId,
  trenchCategory,
  Data_selection, osType
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
  Data_selection,
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
        coalesce((case when lower(r.osType) like '%andro%' then 'android'
                  when lower(r.osType) like '%os%' then 'ios' else lower(r.osType) end),
            (case when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%andro%' then 'android'
                  when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%os%' then 'ios'
                  when lower(loanmaster.deviceType) like '%andro%' then 'android'
                  else 'ios' end)
            ) as osType
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
  select * from base
  qualify row_number() over(partition by digitalLoanAccountId, modelVersionId order by appln_submit_datetime) = 1
  ;
  """
dfd = client.query(sq).to_dataframe()
# dfd = dfd.drop_duplicates(keep='first')
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()

# %%
df2 = dfd.copy()


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
    data_selection_column="Data_selection",
    model_version_column="modelVersionId",
    trench_column="trenchCategory",
    loan_type_column="new_loan_type",
    loan_product_type_column="loan_product_type",
    ostype_column="osType",
    account_id_column="digitalLoanAccountId",
)

# %%
fact_table, dimension_table = update_tables(
    fact_table, dimension_table, model_name="beta_stack_model_cash", product="CASH"
)

df_f_fpd30_betastackcash = fact_table.copy()
df_d_fpd30_betastackcash = dimension_table.copy()

job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_f_fpd30_betastackcash, facttable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimensi1on_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_d_fpd30_betastackcash, dimtable_id, job_config=job_config
)
job.result()  # Wait for the job to complete


# %%


# ### FSPD30


# ### Test

# %%
sq = """
WITH parsed as (
select customerId, digitalLoanAccountId,modelDisplayName,modelVersionId,start_time,end_time,prediction,trenchCategory,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeatures,
REPLACE(REPLACE(prediction, "'", '"'), "None", "null") AS prediction_clean, deviceOs osType,
FROM `prj-prod-dataplatform.audit_balance.ml_model_run_details`
where modelDisplayName in  ('Beta-Cash-Stack-Model', 'beta_stack_model_cash')
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
  p.modelVersionId, osType
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
  'Prod' Data_selection,
  del.deffspd30,
  del.flg_mature_fspd_30,
  loanmaster.new_loan_type,
  modelVersionId, trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile'
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall'
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
        coalesce((case when lower(r.osType) like '%andro%' then 'android'
                  when lower(r.osType) like '%os%' then 'ios' else lower(r.osType) end),
            (case when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%andro%' then 'android'
                  when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%os%' then 'ios'
                  when lower(loanmaster.deviceType) like '%andro%' then 'android'
                  else 'ios' end)
            ) as osType
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
  qualify row_number() over(partition by digitalLoanAccountId, modelVersionId order by appln_submit_datetime) = 1
  ;
"""
dfd = client.query(sq).to_dataframe()
# dfd = dfd.drop_duplicates(keep='first')
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()


# %%
df1 = dfd.copy()


# ### Train

# %%
sq = """
WITH parsed as (
select customerId, digitalLoanAccountId,modelDisplayName,modelVersionId,start_time,end_time,prediction,trenchCategory,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeatures,
Data_selection, deviceOs osType,
FROM prj-prod-dataplatform.dap_ds_poweruser_playground.ml_training_model_run_details_20260116
where modelDisplayName in  ('Beta-Cash-Stack-Model', 'beta_stack_model_cash')
),
  modelname as (
  select customerId,
  digitalLoanAccountId,
  start_time,
  prediction Beta_cash_stack_score,
    case when modelDisplayName like 'Beta-Cash-Stack-Model' then 'beta_stack_model_cash' else modelDisplayName end as modelDisplayName,
    modelVersionId,
  trenchCategory,
  Data_selection, osType
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
  Data_selection,
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
        coalesce((case when lower(r.osType) like '%andro%' then 'android'
                  when lower(r.osType) like '%os%' then 'ios' else lower(r.osType) end),
            (case when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%andro%' then 'android'
                  when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%os%' then 'ios'
                  when lower(loanmaster.deviceType) like '%andro%' then 'android'
                  else 'ios' end)
            ) as osType
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
  select * from base
  qualify row_number() over(partition by digitalLoanAccountId, modelVersionId order by appln_submit_datetime) = 1
  ;
  """
dfd = client.query(sq).to_dataframe()
# dfd = dfd.drop_duplicates(keep='first')
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()

# %%
df2 = dfd.copy()


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
    data_selection_column="Data_selection",
    model_version_column="modelVersionId",
    trench_column="trenchCategory",
    loan_type_column="new_loan_type",
    loan_product_type_column="loan_product_type",
    ostype_column="osType",
    account_id_column="digitalLoanAccountId",
)

# %%
fact_table, dimension_table = update_tables(
    fact_table, dimension_table, model_name="beta_stack_model_cash", product="CASH"
)

# %%
df_f_fspd30_betastackcash = fact_table.copy()
df_d_fspd30_betastackcash = dimension_table.copy()

job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_f_fspd30_betastackcash, facttable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimensi1on_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_d_fspd30_betastackcash, dimtable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# %%


# ### FSTPD30


# ### Test

# %%
sq = """
WITH parsed as (
select customerId, digitalLoanAccountId,modelDisplayName,modelVersionId,start_time,end_time,prediction,trenchCategory,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeatures,
REPLACE(REPLACE(prediction, "'", '"'), "None", "null") AS prediction_clean, deviceOs osType,
FROM `prj-prod-dataplatform.audit_balance.ml_model_run_details`
where modelDisplayName in  ('Beta-Cash-Stack-Model', 'beta_stack_model_cash')
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
  p.modelVersionId, osType
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
  'Prod' Data_selection,
  del.deffstpd30,
  del.flg_mature_fstpd_30,
  loanmaster.new_loan_type,
  modelVersionId, trenchCategory,
    case when loanmaster.loantype='BNPL' and store_type =1 then 'Appliance'
    when loanmaster.loantype='BNPL' and store_type =2 then 'Mobile'
    when loanmaster.loantype='BNPL' and store_type =3 then 'Mall'
    when loanmaster.loantype='BNPL' and store_type not in (1,2,3) then store_tagging
    else 'not applicable' end as loan_product_type,
        coalesce((case when lower(r.osType) like '%andro%' then 'android'
                  when lower(r.osType) like '%os%' then 'ios' else lower(r.osType) end),
            (case when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%andro%' then 'android'
                  when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%os%' then 'ios'
                  when lower(loanmaster.deviceType) like '%andro%' then 'android'
                  else 'ios' end)
            ) as osType
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
  qualify row_number() over(partition by digitalLoanAccountId, modelVersionId order by appln_submit_datetime) = 1
  ;
"""
dfd = client.query(sq).to_dataframe()
# dfd = dfd.drop_duplicates(keep='first')
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()


# %%
df1 = dfd.copy()


# ### Train

# %%
sq = """
WITH parsed as (
select customerId, digitalLoanAccountId,modelDisplayName,modelVersionId,start_time,end_time,prediction,trenchCategory,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeatures,
Data_selection, deviceOs osType,
FROM prj-prod-dataplatform.dap_ds_poweruser_playground.ml_training_model_run_details_20260116
where modelDisplayName in  ('Beta-Cash-Stack-Model', 'beta_stack_model_cash')
),
  modelname as (
  select customerId,
  digitalLoanAccountId,
  start_time,
  prediction Beta_cash_stack_score,
    case when modelDisplayName like 'Beta-Cash-Stack-Model' then 'beta_stack_model_cash' else modelDisplayName end as modelDisplayName,
    modelVersionId,
  trenchCategory,
  Data_selection, osType
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
  Data_selection,
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
        coalesce((case when lower(r.osType) like '%andro%' then 'android'
                  when lower(r.osType) like '%os%' then 'ios' else lower(r.osType) end),
            (case when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%andro%' then 'android'
                  when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%os%' then 'ios'
                  when lower(loanmaster.deviceType) like '%andro%' then 'android'
                  else 'ios' end)
            ) as osType
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
  select * from base
  qualify row_number() over(partition by digitalLoanAccountId, modelVersionId order by appln_submit_datetime) = 1
  ;
  """
dfd = client.query(sq).to_dataframe()
# dfd = dfd.drop_duplicates(keep='first')
print(f"The shape of the dataframe downloaded is:\t {dfd.shape}")
dfd.head()

# %%
df2 = dfd.copy()


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
    data_selection_column="Data_selection",
    model_version_column="modelVersionId",
    trench_column="trenchCategory",
    loan_type_column="new_loan_type",
    loan_product_type_column="loan_product_type",
    ostype_column="osType",
    account_id_column="digitalLoanAccountId",
)

# %%
fact_table, dimension_table = update_tables(
    fact_table, dimension_table, model_name="beta_stack_model_cash", product="CASH"
)
print(f"The shape of the fact table is:\t {fact_table.shape}")
print(f"The shape of the dimension table is:\t {dimension_table.shape}")

df_f_fstpd30_betastackcash = fact_table.copy()
df_d_fstpd30_betastackcash = dimension_table.copy()

job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_f_fstpd30_betastackcash, facttable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimensi1on_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(
    df_d_fstpd30_betastackcash, dimtable_id, job_config=job_config
)
job.result()  # Wait for the job to complete

# %%

print("beta_stack_model_cash gini calculation completed")


# %% [markdown]
# ##### Concate All Dataframes

# %%
factalldf = pd.concat(
    [
        df_f_fpd0_cicsil,
        df_f_fpd10_cicsil,
        df_f_fpd30_cicsil,
        df_f_fspd30_cicsil,
        df_f_fstpd30_cicsil,
        #    df_f_fpd0_alphacredosil, df_f_fpd10_alphacredosil, df_f_fpd30_alphacredosil, df_f_fspd30_alphacredosil, df_f_fstpd30_alphacredosil,
        df_f_fpd0_alphastacksil,
        df_f_fpd10_alphastacksil,
        df_f_fpd30_alphastacksil,
        df_f_fspd30_alphastacksil,
        df_f_fstpd30_alphastacksil,
        df_f_fpd0_appscoresil,
        df_f_fpd10_appscoresil,
        df_f_fpd30_appscoresil,
        df_f_fspd30_appscoresil,
        df_f_fstpd30_appscoresil,
        df_f_fpd0_betademoscoresil,
        df_f_fpd10_betademoscoresil,
        df_f_fpd30_betademoscoresil,
        df_f_fspd30_betademoscoresil,
        df_f_fstpd30_betademoscoresil,
        df_f_fpd0_betastackscoresil,
        df_f_fpd10_betastackscoresil,
        df_f_fpd30_betastackscoresil,
        df_f_fspd30_betastackscoresil,
        df_f_fstpd30_betastackscoresil,
        df_f_fpd0_betastackcredoscoresil,
        df_f_fpd10_betastackcredoscoresil,
        df_f_fpd30_betastackcredoscoresil,
        df_f_fspd30_betastackcredoscoresil,
        df_f_fstpd30_betastackcredoscoresil,
        df_f_fpd0_ciccash,
        df_f_fpd10_ciccash,
        df_f_fpd30_ciccash,
        df_f_fspd30_ciccash,
        df_f_fstpd30_ciccash,
        df_f_fpd0_alphastackcash,
        df_f_fpd10_alphastackcash,
        df_f_fpd30_alphastackcash,
        df_f_fspd30_alphastackcash,
        df_f_fstpd30_alphastackcash,
        df_f_fpd0_credoscorecash,
        df_f_fpd10_credoscorecash,
        df_f_fpd30_credoscorecash,
        df_f_fspd30_credoscorecash,
        df_f_fstpd30_credoscorecash,
        df_f_fpd0_appscorecash,
        df_f_fpd10_appscorecash,
        df_f_fpd30_appscorecash,
        df_f_fspd30_appscorecash,
        df_f_fstpd30_appscorecash,
        df_f_fpd0_betademocash,
        df_f_fpd10_betademocash,
        df_f_fpd30_betademocash,
        df_f_fspd30_betademocash,
        df_f_fstpd30_betademocash,
        df_f_fpd0_betastackcash,
        df_f_fpd10_betastackcash,
        df_f_fpd30_betastackcash,
        df_f_fspd30_betastackcash,
        df_f_fstpd30_betastackcash,
    ],
    ignore_index=True,
)
dimalldf = pd.concat(
    [
        df_d_fpd0_cicsil,
        df_d_fpd10_cicsil,
        df_d_fpd30_cicsil,
        df_d_fspd30_cicsil,
        df_d_fstpd30_cicsil,
        #   df_d_fpd0_alphacredosil, df_d_fpd10_alphacredosil, df_d_fpd30_alphacredosil, df_d_fspd30_alphacredosil, df_d_fstpd30_alphacredosil,
        df_d_fpd0_alphastacksil,
        df_d_fpd10_alphastacksil,
        df_d_fpd30_alphastacksil,
        df_d_fspd30_alphastacksil,
        df_d_fstpd30_alphastacksil,
        df_d_fpd0_appscoresil,
        df_d_fpd10_appscoresil,
        df_d_fpd30_appscoresil,
        df_d_fspd30_appscoresil,
        df_d_fstpd30_appscoresil,
        df_d_fpd0_betademoscoresil,
        df_d_fpd10_betademoscoresil,
        df_d_fpd30_betademoscoresil,
        df_d_fspd30_betademoscoresil,
        df_d_fstpd30_betademoscoresil,
        df_d_fpd0_betastackscoresil,
        df_d_fpd10_betastackscoresil,
        df_d_fpd30_betastackscoresil,
        df_d_fspd30_betastackscoresil,
        df_d_fstpd30_betastackscoresil,
        df_d_fpd0_betastackcredoscoresil,
        df_d_fpd10_betastackcredoscoresil,
        df_d_fpd30_betastackcredoscoresil,
        df_d_fspd30_betastackcredoscoresil,
        df_d_fstpd30_betastackcredoscoresil,
        df_d_fpd0_ciccash,
        df_d_fpd10_ciccash,
        df_d_fpd30_ciccash,
        df_d_fspd30_ciccash,
        df_d_fstpd30_ciccash,
        df_d_fpd0_alphastackcash,
        df_d_fpd10_alphastackcash,
        df_d_fpd30_alphastackcash,
        df_d_fspd30_alphastackcash,
        df_d_fstpd30_alphastackcash,
        df_d_fpd0_credoscorecash,
        df_d_fpd10_credoscorecash,
        df_d_fpd30_credoscorecash,
        df_d_fspd30_credoscorecash,
        df_d_fstpd30_credoscorecash,
        df_d_fpd0_appscorecash,
        df_d_fpd10_appscorecash,
        df_d_fpd30_appscorecash,
        df_d_fspd30_appscorecash,
        df_d_fstpd30_appscorecash,
        df_d_fpd0_betademocash,
        df_d_fpd10_betademocash,
        df_d_fpd30_betademocash,
        df_d_fspd30_betademocash,
        df_d_fstpd30_betademocash,
        df_d_fpd0_betastackcash,
        df_d_fpd10_betastackcash,
        df_d_fpd30_betastackcash,
        df_d_fspd30_betastackcash,
        df_d_fstpd30_betastackcash,
    ],
    ignore_index=True,
)

print(
    f"The shape of concatenated Fact and dimension table is :\ {factalldf.shape} & {dimalldf.shape}"
)

# %%
factbetademosil = pd.concat(
    [
        df_f_fpd0_betademoscoresil,
        df_f_fpd10_betademoscoresil,
        df_f_fpd30_betademoscoresil,
        df_f_fspd30_betademoscoresil,
        df_f_fstpd30_betademoscoresil,
    ],
    ignore_index=False,
)
dimbetademosil = pd.concat(
    [
        df_d_fpd0_betademoscoresil,
        df_d_fpd10_betademoscoresil,
        df_d_fpd30_betademoscoresil,
        df_d_fspd30_betademoscoresil,
        df_d_fstpd30_betademoscoresil,
    ],
    ignore_index=False,
)

# %%
factcicsil = pd.concat(
    [
        df_f_fpd0_cicsil,
        df_f_fpd10_cicsil,
        df_f_fpd30_cicsil,
        df_f_fspd30_cicsil,
        df_f_fstpd30_cicsil,
    ],
    ignore_index=True,
)
democicsil = pd.concat(
    [
        df_d_fpd0_cicsil,
        df_d_fpd10_cicsil,
        df_d_fpd30_cicsil,
        df_d_fspd30_cicsil,
        df_d_fstpd30_cicsil,
    ],
    ignore_index=True,
)

# %%
df_f_fspd30_cicsil.head()

# %%
dd.query(
    """select distinct Model_display_name, model_version, bad_rate from democicsil order by 1,2,3"""
).df()

# %%
dimalldf.columns.values

# %%
dd.query(
    """select * from factalldf where bad_rate = 'FSPD30'
and Model_display_name = 'cic_model_sil'
;"""
).df()

# %% [markdown]
# #### Create a tables for Fact and Dimension dataset

# %%
facttable_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_table4"
dimtable_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_table4"

# %%
# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.fact_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_TRUNCATE",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(factalldf, facttable_id, job_config=job_config)
job.result()  # Wait for the job to complete

# %%
# Upload to BigQuery
# table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_table3"
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_TRUNCATE",  # or "WRITE_APPEND"
)
job = client.load_table_from_dataframe(dimalldf, dimtable_id, job_config=job_config)
job.result()  # Wait for the job to complete

# %%
df_clean = dimalldf.drop_duplicates(keep="first").reset_index(drop=True)
print(
    f"The shape of the dimension table after dropping duplicates is:\t {df_clean.shape}"
)

# %% [markdown]
#
# create or replace table prj-prod-dataplatform.dap_ds_poweruser_playground.fact_table3_copy
# as
# select * from prj-prod-dataplatform.dap_ds_poweruser_playground.fact_table3
# ;
#
# create or replace table prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_table3_copy as
# select * from prj-prod-dataplatform.dap_ds_poweruser_playground.dimension_table3;

# %% [markdown]
# # End
