# region Define Libraries
import io
import json
import logging
import os
import pickle
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import wraps
from itertools import combinations
from typing import Dict, List, Optional, Tuple, Union

import duckdb as dd
import gcsfs
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from google.api_core.exceptions import GoogleAPIError
from google.cloud import bigquery, storage
from sklearn.metrics import roc_auc_score


# Configure logging
def setup_logging(
    log_level: str = "INFO", log_file: Optional[str] = None
) -> logging.Logger:
    """Configure comprehensive logging with file and console handlers."""
    logger = logging.getLogger("gini_pipeline")
    logger.setLevel(getattr(logging, log_level.upper()))

    # Clear existing handlers
    logger.handlers = []

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


logger = setup_logging("INFO", "gini_pipeline.log")


# Configuration class for centralized settings
@dataclass
class PipelineConfig:
    """Centralized configuration for the Gini calculation pipeline."""

    project_id: str = "prj-prod-dataplatform"
    dataset_id: str = "dap_ds_poweruser_playground"
    fact_table: str = "fact_table4"
    dimension_table: str = "dimension_table4"

    # Model configurations
    models: Dict[str, Dict] = None

    # Delinquency metrics to process
    delinquency_metrics: List[str] = None

    # Minimum observations for Gini calculation
    min_weekly_obs: int = 10
    min_monthly_obs: int = 20

    def __post_init__(self):
        self.models = {
            "cic_model_sil": {
                "display_name": "cic_model_sil",
                "product": "SIL",
                "model_names": [
                    "Alpha - CIC-SIL-Model",
                    "cic_model_sil",
                    "Sil-Alpha-CIC-SIL-Model",
                ],
                "score_column": "Alpha_cic_sil_score",
                "score_extractor": None,  # Direct column
            },
            "alpha_stack_credo_score_sil": {
                "display_name": "alpha_stack_credo_score_sil",
                "product": "SIL",
                "model_names": ["Alpha - StackingModel", "alpha_stack_model_sil"],
                "score_column": "credo_score",
                "score_extractor": "credo_score",  # JSON path
            },
            "alpha_stack_model_sil": {
                "display_name": "alpha_stack_model_sil",
                "product": "SIL",
                "model_names": ["Alpha - StackingModel", "alpha_stack_model_sil"],
                "score_column": "Sil_Alpha_Stack_score",
                "score_extractor": None,
            },
            "apps_score_model_sil": {
                "display_name": "apps_score_model_sil",
                "product": "SIL",
                "model_names": ["Beta - AppsScoreModel", "apps_score_model_sil"],
                "score_column": "sil_beta_app_score",
                "score_extractor": "combined_score",
            },
            "beta_demo_model_sil": {
                "display_name": "beta_demo_model_sil",
                "product": "SIL",
                "model_names": ["Beta - DemoScoreModel", "beta_demo_model_sil"],
                "score_column": "sil_beta_demo_score",
                "score_extractor": None,
            },
            "beta_stack_model_sil": {
                "display_name": "beta_stack_model_sil",
                "product": "SIL",
                "model_names": ["Beta - StackScoreModel", "beta_stack_model_sil"],
                "score_column": "sil_beta_stack_score",
                "score_extractor": None,
            },
        }

        self.delinquency_metrics = [
            ("FPD0", "deffpd0", "flg_mature_fpd0"),
            ("FPD10", "deffpd10", "flg_mature_fpd10"),
            ("FPD30", "deffpd30", "flg_mature_fpd30"),
            ("FSPD30", "deffspd30", "flg_mature_fspd_30"),
            ("FSTPD30", "deffstpd30", "flg_mature_fstpd_30"),
        ]


# Initialize configuration
config = PipelineConfig()


# Set up BigQuery client with retry logic
def get_bigquery_client(project_id: str) -> bigquery.Client:
    """Initialize BigQuery client with proper authentication."""
    try:
        path = r"C:\Users\Dwaipayan\AppData\Roaming\gcloud\legacy_credentials\dchakroborti@tonikbank.com\adc.json"
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = path
        os.environ["GOOGLE_CLOUD_PROJECT"] = project_id

        client = bigquery.Client(project=project_id)
        logger.info(f"BigQuery client initialized for project: {project_id}")
        return client
    except Exception as e:
        logger.error(f"Failed to initialize BigQuery client: {e}")
        raise


client = get_bigquery_client(config.project_id)

# Set pandas options
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 100)
import warnings

warnings.filterwarnings("ignore")

# endregion

# region Utility Decorators and Functions


def timer(func):
    """Decorator to measure execution time of functions."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        logger.info(f"Starting {func.__name__}...")
        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            logger.info(f"Completed {func.__name__} in {elapsed:.2f} seconds")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"Failed {func.__name__} after {elapsed:.2f} seconds: {e}")
            raise

    return wrapper


def retry_on_error(max_retries: int = 3, delay: float = 1.0):
    """Decorator to retry functions on failure."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    logger.warning(
                        f"Attempt {attempt + 1} failed for {func.__name__}: {e}. Retrying..."
                    )
                    time.sleep(delay * (attempt + 1))

        return wrapper

    return decorator


# endregion

# region Gini Calculation Functions


def calculate_gini(pd_scores: np.ndarray, bad_indicators: np.ndarray) -> float:
    """
    Calculate Gini coefficient from scores and binary indicators.

    Args:
        pd_scores: Array-like of scores/probabilities
        bad_indicators: Array-like of binary outcomes (0/1)

    Returns:
        Gini coefficient or np.nan if calculation fails
    """
    try:
        pd_scores = np.array(pd_scores, dtype=float)
        bad_indicators = np.array(bad_indicators, dtype=int)

        if len(pd_scores) < 2 or len(np.unique(bad_indicators)) < 2:
            return np.nan

        auc = roc_auc_score(bad_indicators, pd_scores)
        gini = 2 * auc - 1
        return gini

    except (ValueError, TypeError) as e:
        logger.debug(f"Gini calculation failed: {e}")
        return np.nan


def calculate_gini_alternative(scores: np.ndarray, labels: np.ndarray) -> float:
    """
    Alternative Gini calculation with proper edge case handling.

    Returns np.nan when:
    - Fewer than 2 observations
    - No positive labels (sum of labels = 0)
    """
    n = len(scores)
    if n < 2:
        return np.nan

    label_sum = np.sum(labels)
    if label_sum == 0:
        return np.nan

    sorted_indices = np.argsort(scores)
    sorted_labels = labels[sorted_indices]
    cumsum_labels = np.cumsum(sorted_labels)

    gini = 1 - 2 * np.sum(cumsum_labels) / (n * label_sum)
    return gini


# endregion

# region SQL Query Builder


class SQLQueryBuilder:
    """Builds SQL queries dynamically for different models and metrics."""

    # Base CTEs that are common across queries
    DELINQUENCY_CTE = """
    delinquency AS (
        SELECT loanAccountNumber,
            CASE WHEN obs_min_inst_def0 >= 1 AND min_inst_def0 = 1 THEN 1 ELSE 0 END AS deffpd0,
            CASE WHEN obs_min_inst_def10 >= 1 AND min_inst_def10 = 1 THEN 1 ELSE 0 END AS deffpd10,
            CASE WHEN obs_min_inst_def30 >= 1 AND min_inst_def30 = 1 THEN 1 ELSE 0 END AS deffpd30,
            CASE WHEN obs_min_inst_def30 >= 2 AND min_inst_def30 IN (1,2) THEN 1 ELSE 0 END AS deffspd30,
            CASE WHEN obs_min_inst_def30 >= 3 AND min_inst_def30 IN (1,2,3) THEN 1 ELSE 0 END AS deffstpd30,
            CASE WHEN obs_min_inst_def0 >= 1 THEN 1 ELSE 0 END AS flg_mature_fpd0,
            CASE WHEN obs_min_inst_def10 >= 1 THEN 1 ELSE 0 END AS flg_mature_fpd10,
            CASE WHEN obs_min_inst_def30 >= 1 THEN 1 ELSE 0 END AS flg_mature_fpd30,
            CASE WHEN obs_min_inst_def30 >= 2 THEN 1 ELSE 0 END AS flg_mature_fspd_30,
            CASE WHEN obs_min_inst_def30 >= 3 THEN 1 ELSE 0 END AS flg_mature_fstpd_30
        FROM `{project}.risk_credit_mis.loan_deliquency_data`
    )"""

    MERCHANT_CTE = """
    merchant_mapping AS (
        SELECT DISTINCT mer_refferal_code, mer_name, store_type, store_tagging
        FROM `dl_loans_db_raw.tdbk_merchant_refferal_mtb`
        LEFT JOIN `worktable_datachampions.TARGET_SPLIT` P ON P.STORE_NAME = mer_name
        QUALIFY ROW_NUMBER() OVER (PARTITION BY mer_refferal_code ORDER BY created_dt DESC) = 1
    )"""

    @staticmethod
    def get_trench_logic() -> str:
        """Returns the standardized trench category logic."""
        return """
        CASE 
            WHEN trenchCategory IS NULL THEN (
                CASE 
                    WHEN mt.ln_user_type = '1_Repeat Applicant' THEN 'Trench 3'
                    WHEN mt.ln_user_type <> '1_Repeat Applicant' 
                        AND DATE_DIFF(CURRENT_DATE(), mt.onb_tsa_onboarding_datetime, DAY) > 30 
                    THEN 'Trench 2'
                    ELSE 'Trench1' 
                END
            )
            WHEN trenchCategory = '' THEN (
                CASE 
                    WHEN mt.ln_user_type = '1_Repeat Applicant' THEN 'Trench 3'
                    WHEN mt.ln_user_type <> '1_Repeat Applicant' 
                        AND DATE_DIFF(CURRENT_DATE(), mt.onb_tsa_onboarding_datetime, DAY) > 30 
                    THEN 'Trench 2'
                    ELSE 'Trench 1' 
                END
            )
            ELSE trenchCategory 
        END AS trenchCategory"""

    @staticmethod
    def get_product_type_logic() -> str:
        """Returns the standardized loan product type logic."""
        return """
        CASE 
            WHEN loanmaster.loantype = 'BNPL' AND store_type = 1 THEN 'Appliance'
            WHEN loanmaster.loantype = 'BNPL' AND store_type = 2 THEN 'Mobile'
            WHEN loanmaster.loantype = 'BNPL' AND store_type = 3 THEN 'Mall'
            WHEN loanmaster.loantype = 'BNPL' AND store_type NOT IN (1,2,3) THEN store_tagging
            ELSE 'not applicable'
        END AS loan_product_type"""

    @staticmethod
    def get_ostype_logic(source_table: str = "r") -> str:
        """Returns the standardized OS type logic."""
        return f"""
        COALESCE(
            CASE 
                WHEN LOWER({source_table}.osType) LIKE '%andro%' THEN 'android'
                WHEN LOWER({source_table}.osType) LIKE '%os%' THEN 'ios'
                ELSE LOWER({source_table}.osType)
            END,
            CASE 
                WHEN LOWER(COALESCE(loanmaster.osversion_v2, loanmaster.osVersion)) LIKE '%andro%' THEN 'android'
                WHEN LOWER(COALESCE(loanmaster.osversion_v2, loanmaster.osVersion)) LIKE '%os%' THEN 'ios'
                WHEN LOWER(loanmaster.deviceType) LIKE '%andro%' THEN 'android'
                ELSE 'ios'
            END
        ) AS osType"""

    @classmethod
    def build_model_cte(cls, model_config: Dict, is_train: bool = False) -> str:
        """Builds the model-specific CTE."""
        table = (
            "ml_training_model_run_details_20260116"
            if is_train
            else "ml_model_run_details"
        )
        schema = "dap_ds_poweruser_playground" if is_train else "audit_balance"
        data_selection = "Data_selection" if is_train else "'Prod' AS Data_selection"

        model_names = "', '".join(model_config["model_names"])

        # Build score extraction logic
        score_col = model_config["score_column"]
        extractor = model_config.get("score_extractor")

        if extractor and "credo" in score_col:
            score_logic = f"""
            COALESCE(
                CAST(JSON_VALUE(SAFE.PARSE_JSON(CAST(calcFeature AS STRING)), '$.{extractor}') AS FLOAT64),
                CAST(JSON_VALUE(SAFE.PARSE_JSON(CAST(calcFeature AS STRING)), '$.s_{extractor}') AS FLOAT64)
            ) AS {score_col}"""
        elif extractor:
            score_logic = f"""
            SAFE_CAST(JSON_VALUE(prediction_clean, '$.{extractor}') AS FLOAT64) AS {score_col}"""
        else:
            score_logic = f"prediction AS {score_col}"

        # Handle special case for apps_score_model_sil which needs prediction cleaning
        if "app_score" in score_col:
            return f"""
    cleaned AS (
        SELECT
            mmrd.customerId,
            mmrd.digitalLoanAccountId,
            prediction,
            start_time,
            end_time,
            modelDisplayName,
            modelVersionId,
            {cls.get_trench_logic()},
            REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature,
            REPLACE(REPLACE(CAST(prediction AS STRING), "'", '"'), "None", "null") AS prediction_clean,
            {data_selection},
            deviceOs AS osType
        FROM `{config.project_id}.{schema}.{table}` mmrd
        LEFT JOIN `{config.project_id}.risk_credit_mis.model_loan_score_mart` mt 
            ON mt.digitalLoanAccountId = mmrd.digitalLoanAccountId
        WHERE modelDisplayName IN ('{model_names}')
    ),
    modelname AS (
        SELECT
            customerId,
            digitalLoanAccountId,
            start_time,
            COALESCE(prediction, {score_logic}) AS {score_col},
            CASE 
                WHEN modelDisplayName = 'Beta - AppsScoreModel' THEN 'apps_score_model_sil'
                ELSE modelDisplayName 
            END AS modelDisplayName,
            modelVersionId,
            trenchCategory,
            Data_selection,
            osType
        FROM cleaned
        WHERE {score_col} IS NOT NULL
    )"""
        else:
            return f"""
    modelname AS (
        SELECT
            mmrd.customerId,
            mmrd.digitalLoanAccountId,
            {score_logic},
            start_time,
            end_time,
            CASE 
                WHEN modelDisplayName = 'Beta - DemoScoreModel' THEN 'beta_demo_model_sil'
                WHEN modelDisplayName = 'Beta - StackScoreModel' THEN 'beta_stack_model_sil'
                ELSE modelDisplayName 
            END AS modelDisplayName,
            modelVersionId,
            {cls.get_trench_logic()},
            REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature,
            {data_selection},
            deviceOs AS osType
        FROM `{config.project_id}.{schema}.{table}` mmrd
        LEFT JOIN `{config.project_id}.risk_credit_mis.model_loan_score_mart` mt 
            ON mt.digitalLoanAccountId = mmrd.digitalLoanAccountId
        WHERE modelDisplayName IN ('{model_names}')
    )"""

    @classmethod
    def build_base_query(
        cls, model_config: Dict, metric: Tuple[str, str, str], is_train: bool = False
    ) -> str:
        """Builds the complete query for a specific model and metric."""
        metric_name, def_col, mature_col = metric

        query = f"""
WITH {cls.build_model_cte(model_config, is_train)},
{cls.DELINQUENCY_CTE.format(project=config.project_id)},
{cls.MERCHANT_CTE},
base AS (
    SELECT DISTINCT
        r.customerId,
        r.digitalLoanAccountId,
        loanmaster.loanAccountNumber,
        r.modelDisplayName,
        r.{model_config['score_column']},
        COALESCE(
            IF(loanmaster.new_loan_type = 'Flex-up', 
               loanmaster.startApplyDateTime, 
               loanmaster.termsAndConditionsSubmitDateTime),
            CAST(r.start_time AS DATETIME)
        ) AS appln_submit_datetime,
        DATE(loanmaster.disbursementDateTime) AS disbursementdate,
        FORMAT_DATE('%Y-%m', COALESCE(
            IF(loanmaster.new_loan_type = 'Flex-up', 
               loanmaster.startApplyDateTime, 
               loanmaster.termsAndConditionsSubmitDateTime),
            CAST(r.start_time AS DATETIME)
        )) AS Application_month,
        r.Data_selection,
        del.{def_col},
        del.{mature_col},
        loanmaster.new_loan_type,
        r.modelVersionId,
        r.trenchCategory,
        {cls.get_product_type_logic()},
        {cls.get_ostype_logic("r")}
    FROM modelname r
    LEFT JOIN `risk_credit_mis.loan_master_table` loanmaster 
        ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
    INNER JOIN delinquency del 
        ON del.loanAccountNumber = loanmaster.loanAccountNumber
    LEFT JOIN merchant_mapping sil_category 
        ON loanmaster.purpleKey = sil_category.mer_refferal_code
    WHERE loanmaster.flagDisbursement = 1
        AND loanmaster.disbursementDateTime IS NOT NULL
        AND r.{model_config['score_column']} IS NOT NULL
        AND del.{mature_col} = 1
)
SELECT * FROM base
QUALIFY ROW_NUMBER() OVER (
    PARTITION BY digitalLoanAccountId, modelVersionId 
    ORDER BY appln_submit_datetime
) = 1
"""
        return query


# endregion

# region Data Processing Functions


@timer
@retry_on_error(max_retries=3)
def execute_query(query: str) -> pd.DataFrame:
    """Execute BigQuery query with error handling and logging."""
    try:
        logger.debug(f"Executing query (first 500 chars): {query[:500]}...")
        df = client.query(query).to_dataframe()
        logger.info(f"Query returned {len(df)} rows")
        return df
    except GoogleAPIError as e:
        logger.error(f"BigQuery error: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error executing query: {e}")
        raise


def combine_train_test(
    df_test: pd.DataFrame, df_train: pd.DataFrame, id_col: str = "digitalLoanAccountId"
) -> pd.DataFrame:
    """
    Efficiently combine test and train datasets, removing duplicates.
    Train data takes precedence for overlapping IDs.
    """
    if df_test.empty and df_train.empty:
        logger.warning("Both test and train dataframes are empty")
        return pd.DataFrame()

    if df_test.empty:
        logger.info("Test dataframe empty, using train only")
        return df_train.copy()

    if df_train.empty:
        logger.info("Train dataframe empty, using test only")
        return df_test.copy()

    # Use set difference for efficient filtering
    train_ids = set(df_train[id_col].unique())
    mask = ~df_test[id_col].isin(train_ids)
    df_test_filtered = df_test[mask].copy()

    logger.info(
        f"Filtered {len(df_test) - len(df_test_filtered)} duplicate rows from test set"
    )

    # Concatenate efficiently
    result = pd.concat([df_test_filtered, df_train], ignore_index=True, copy=False)
    logger.info(f"Combined dataframe shape: {result.shape}")

    return result


@timer
def calculate_periodic_gini(
    df: pd.DataFrame,
    score_column: str,
    label_column: str,
    namecolumn: str,
    data_selection_column: Optional[str] = None,
    model_version_column: Optional[str] = None,
    trench_column: Optional[str] = None,
    loan_type_column: Optional[str] = None,
    loan_product_type_column: Optional[str] = None,
    ostype_column: Optional[str] = None,
    account_id_column: Optional[str] = None,
    min_weekly: int = 10,
    min_monthly: int = 20,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculate periodic Gini coefficients with optimized performance.
    Returns fact_table and dimension_table.
    """
    logger.info(f"Starting Gini calculation for {namecolumn} with score {score_column}")

    # Input validation
    required_columns = ["disbursementdate", score_column, label_column]
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Create working copy with only needed columns to save memory
    needed_cols = [
        c
        for c in [
            "disbursementdate",
            score_column,
            label_column,
            data_selection_column,
            model_version_column,
            trench_column,
            loan_type_column,
            loan_product_type_column,
            ostype_column,
            account_id_column,
        ]
        if c
    ]

    df = df[needed_cols].copy()

    # Convert types efficiently
    df["disbursementdate"] = pd.to_datetime(df["disbursementdate"], errors="coerce")
    df[score_column] = pd.to_numeric(df[score_column], errors="coerce")
    df[label_column] = pd.to_numeric(df[label_column], errors="coerce")

    # Drop invalid rows
    initial_len = len(df)
    df = df.dropna(subset=[score_column, label_column, "disbursementdate"])
    logger.info(f"Dropped {initial_len - len(df)} rows with invalid values")

    if len(df) == 0:
        logger.warning("No valid data after cleaning")
        return pd.DataFrame(), pd.DataFrame()

    # Build segment columns list with proper mapping
    # Key: output column name in dimension table, Value: input column name from df
    segment_mapping = {}
    if data_selection_column:
        segment_mapping["data_selection"] = data_selection_column
    if model_version_column:
        segment_mapping["model_version"] = model_version_column
    if trench_column:
        segment_mapping["trench_category"] = trench_column
    if loan_type_column:
        segment_mapping["loan_type"] = loan_type_column
    if loan_product_type_column:
        segment_mapping["loan_product_type"] = loan_product_type_column
    if ostype_column:
        segment_mapping["ostype"] = ostype_column

    # Build list for generating combinations (using original column names)
    segment_cols = []
    if data_selection_column:
        segment_cols.append(("DataSelection", data_selection_column))
    if model_version_column:
        segment_cols.append(("ModelVersion", model_version_column))
    if trench_column:
        segment_cols.append(("Trench", trench_column))
    if loan_type_column:
        segment_cols.append(("LoanType", loan_type_column))
    if loan_product_type_column:
        segment_cols.append(("ProductType", loan_product_type_column))
    if ostype_column:
        segment_cols.append(("OSType", ostype_column))

    # Generate all segment combinations efficiently
    datasets = [("Overall", df, {})]

    for r in range(1, len(segment_cols) + 1):
        for combo in combinations(segment_cols, r):
            combo_names = [c[0] for c in combo]
            combo_cols = [c[1] for c in combo]

            for keys, group in df.groupby(combo_cols, observed=True):
                if len(group) < min_weekly:
                    continue

                filter_dict = dict(
                    zip(combo_cols, keys if isinstance(keys, tuple) else [keys])
                )
                segment_name = "_".join(
                    [
                        f"{n}_{k}"
                        for n, k in zip(
                            combo_names, keys if isinstance(keys, tuple) else [keys]
                        )
                    ]
                )

                # Build metadata with standardized column names
                metadata = {}
                for output_col, input_col in segment_mapping.items():
                    metadata[output_col] = filter_dict.get(input_col)

                datasets.append((segment_name, group, metadata))

    logger.info(f"Processing {len(datasets)} segment combinations")

    # Process all datasets
    all_results = []

    for dataset_name, dataset_df, metadata in datasets:
        if len(dataset_df) < min_weekly:
            continue

        # Calculate weekly Gini
        weekly = calculate_time_period_gini(
            dataset_df, score_column, label_column, "W", min_weekly, account_id_column
        )

        # Calculate monthly Gini
        monthly = calculate_time_period_gini(
            dataset_df, score_column, label_column, "M", min_monthly, account_id_column
        )

        # Combine results
        combined = pd.concat([weekly, monthly], ignore_index=True, copy=False)

        if combined.empty:
            continue

        # Add metadata columns
        combined["Model_Name"] = score_column
        combined["bad_rate"] = namecolumn
        combined["segment_type"] = dataset_name

        # Add segment columns (these will be NA for "Overall")
        for col in segment_mapping.keys():
            combined[col] = metadata.get(col)

        all_results.append(combined)

    if not all_results:
        logger.warning("No Gini results calculated")
        return pd.DataFrame(), pd.DataFrame()

    # Combine all results
    fact_table = pd.concat(all_results, ignore_index=True, copy=False)

    # Create dimension table - use only columns that actually exist
    dim_cols = ["Model_Name", "bad_rate", "segment_type"] + list(segment_mapping.keys())

    # Ensure all columns exist
    for col in dim_cols:
        if col not in fact_table.columns:
            fact_table[col] = None

    dimension_table = fact_table[dim_cols].drop_duplicates().reset_index(drop=True)
    dimension_table["segment_id"] = range(len(dimension_table))

    # Merge segment_id back to fact table
    fact_table = fact_table.merge(
        dimension_table[["segment_id"] + dim_cols], on=dim_cols, how="left"
    )

    # Reorder columns
    final_cols = [
        "segment_id",
        "start_date",
        "end_date",
        "period",
        "gini_value",
        "distinct_accounts",
        "Model_Name",
        "bad_rate",
        "segment_type",
    ] + list(segment_mapping.keys())

    fact_table = fact_table[final_cols]

    dimension_table = dimension_table[
        ["segment_id", "Model_Name", "bad_rate", "segment_type"]
        + list(segment_mapping.keys())
    ]

    logger.info(
        f"Generated fact_table ({len(fact_table)} rows) and dimension_table ({len(dimension_table)} rows)"
    )

    return fact_table, dimension_table


def calculate_time_period_gini(
    df: pd.DataFrame,
    score_col: str,
    label_col: str,
    period: str,  # 'W' or 'M'
    min_obs: int,
    account_id_col: Optional[str] = None,
) -> pd.DataFrame:
    """Calculate Gini for a specific time period."""
    df = df.copy()

    # Create period column
    if period == "W":
        df["period_key"] = df["disbursementdate"].dt.to_period("W")
    else:
        df["period_key"] = df["disbursementdate"].dt.to_period("M")

    # Calculate Gini by period
    results = []

    for name, group in df.groupby("period_key", observed=True):
        if len(group) < min_obs:
            continue

        gini = calculate_gini(group[score_col].values, group[label_col].values)

        if np.isnan(gini):
            continue

        # Calculate start and end dates properly
        start_date = name.to_timestamp()

        if period == "W":
            end_date = start_date + timedelta(days=6)
        else:
            # For monthly: add 1 month, then subtract 1 day
            end_date = (start_date + pd.DateOffset(months=1)) - pd.Timedelta(days=1)

        result = {
            "start_date": start_date,
            "end_date": end_date,
            "gini_value": gini,
            "period": "Week" if period == "W" else "Month",
        }

        # Add distinct account count if specified
        if account_id_col and account_id_col in group.columns:
            result["distinct_accounts"] = group[account_id_col].nunique()
        else:
            result["distinct_accounts"] = None

        results.append(result)

    return pd.DataFrame(results)


def update_tables(
    fact_table: pd.DataFrame,
    dimension_table: pd.DataFrame,
    model_name: str,
    product: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Update tables with model display name and fill NA values."""
    cols_to_replace = [
        "model_version",
        "trench_category",
        "loan_type",
        "loan_product_type",
        "ostype",
    ]

    fact_table = fact_table.copy()
    dimension_table = dimension_table.copy()

    fact_table["Model_display_name"] = model_name
    fact_table["Product_Category"] = product
    fact_table[cols_to_replace] = fact_table[cols_to_replace].fillna("Overall")

    dimension_table["Model_display_name"] = model_name
    dimension_table["Product_Category"] = product
    dimension_table[cols_to_replace] = dimension_table[cols_to_replace].fillna(
        "Overall"
    )

    logger.info(f"Updated tables for model: {model_name}, product: {product}")

    return fact_table, dimension_table


# endregion

# region BigQuery Upload


@timer
@retry_on_error(max_retries=3)
def upload_to_bigquery(
    df: pd.DataFrame, table_id: str, write_disposition: str = "WRITE_APPEND"
) -> None:
    """Upload dataframe to BigQuery with error handling."""
    if df.empty:
        logger.warning(f"Empty dataframe, skipping upload to {table_id}")
        return

    try:
        job_config = bigquery.LoadJobConfig(write_disposition=write_disposition)
        job = client.load_table_from_dataframe(df, table_id, job_config=job_config)
        job.result()
        logger.info(f"Successfully uploaded {len(df)} rows to {table_id}")
    except Exception as e:
        logger.error(f"Failed to upload to {table_id}: {e}")
        raise


# endregion

# region Main Pipeline


class GiniPipeline:
    """Main pipeline class for orchestrating Gini calculations."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.fact_table_id = (
            f"{config.project_id}.{config.dataset_id}.{config.fact_table}"
        )
        self.dim_table_id = (
            f"{config.project_id}.{config.dataset_id}.{config.dimension_table}"
        )
        self.total_rows_processed = 0

    @timer
    def process_model_metric(
        self, model_key: str, metric: Tuple[str, str, str]
    ) -> None:
        """Process a single model-metric combination."""
        model_config = self.config.models[model_key]
        metric_name, def_col, mature_col = metric

        logger.info(f"{'='*60}")
        logger.info(f"Processing: {model_key} - {metric_name}")
        logger.info(f"{'='*60}")

        # Build and execute queries
        query_test = SQLQueryBuilder.build_base_query(
            model_config, metric, is_train=False
        )
        query_train = SQLQueryBuilder.build_base_query(
            model_config, metric, is_train=True
        )

        # Execute queries
        df_test = execute_query(query_test)
        df_train = execute_query(query_train)

        logger.info(f"Test data: {len(df_test)} rows, Train data: {len(df_train)} rows")

        # Combine datasets
        df_combined = combine_train_test(df_test, df_train)

        if df_combined.empty:
            logger.warning(f"No data for {model_key} - {metric_name}, skipping")
            return

        self.total_rows_processed += len(df_combined)

        # Ensure score is numeric
        score_col = model_config["score_column"]
        df_combined[score_col] = pd.to_numeric(df_combined[score_col], errors="coerce")

        # Calculate Gini
        fact_table, dimension_table = calculate_periodic_gini(
            df_combined,
            score_col,
            def_col,
            metric_name,
            data_selection_column="Data_selection",
            model_version_column="modelVersionId",
            trench_column="trenchCategory",
            loan_type_column="new_loan_type",
            loan_product_type_column="loan_product_type",
            ostype_column="osType",
            account_id_column="digitalLoanAccountId",
            min_weekly=self.config.min_weekly_obs,
            min_monthly=self.config.min_monthly_obs,
        )

        if fact_table.empty:
            logger.warning(f"No Gini results for {model_key} - {metric_name}")
            return

        # Update tables
        fact_table, dimension_table = update_tables(
            fact_table,
            dimension_table,
            model_config["display_name"],
            model_config["product"],
        )

        # Upload to BigQuery
        upload_to_bigquery(fact_table, self.fact_table_id, "WRITE_APPEND")
        upload_to_bigquery(dimension_table, self.dim_table_id, "WRITE_APPEND")

        logger.info(f"Completed {model_key} - {metric_name}")

    @timer
    def run(
        self,
        specific_model: Optional[str] = None,
        specific_metric: Optional[str] = None,
    ) -> None:
        """Run the complete pipeline or specific model/metric."""
        logger.info("Starting Gini Calculation Pipeline")
        logger.info(f"Target tables: {self.fact_table_id}, {self.dim_table_id}")

        models_to_process = (
            [specific_model] if specific_model else list(self.config.models.keys())
        )

        for model_key in models_to_process:
            if model_key not in self.config.models:
                logger.error(f"Unknown model: {model_key}")
                continue

            metrics_to_process = (
                [m for m in self.config.delinquency_metrics if m[0] == specific_metric]
                if specific_metric
                else self.config.delinquency_metrics
            )

            for metric in metrics_to_process:
                try:
                    self.process_model_metric(model_key, metric)
                except Exception as e:
                    logger.error(
                        f"Failed to process {model_key} - {metric[0]}: {e}",
                        exc_info=True,
                    )
                    continue

        logger.info(f"{'='*60}")
        logger.info("Pipeline completed!")
        logger.info(f"Total rows processed: {self.total_rows_processed}")
        logger.info(f"{'='*60}")


# endregion

# region Entry Point

if __name__ == "__main__":
    # Initialize and run pipeline
    pipeline = GiniPipeline(config)

    # Run complete pipeline
    pipeline.run()

    # Or run specific model/metric:
    # pipeline.run(specific_model="cic_model_sil", specific_metric="FPD0")
