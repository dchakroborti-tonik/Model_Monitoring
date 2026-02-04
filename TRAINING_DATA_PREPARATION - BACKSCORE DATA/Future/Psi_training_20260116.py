"""
psi_training_20260116.py
Converted from psi_training_20260116.ipynb
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from google.cloud import bigquery
from google.cloud import storage
import os
import tempfile
import time
from datetime import datetime
import uuid
import joblib
import json
import gcsfs
import duckdb as dd
import pickle
from typing import Union, List, Dict, Tuple
import io
import warnings
import logging
from typing import List

# Set up logging
def setup_logging(log_file='psi_training_20260116.log'):
    """Set up logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

# Initialize logger
logger = setup_logging()

# Configure settings
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
warnings.filterwarnings('ignore')

# Set environment variables
path = r'C:\\Users\\Dwaipayan\\AppData\\Roaming\\gcloud\\legacy_credentials\\dchakroborti@tonikbank.com\\adc.json'
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = path
os.environ["GOOGLE_CLOUD_PROJECT"] = "prj-prod-dataplatform"

# Initialize BigQuery client
client = bigquery.Client(project='prj-prod-dataplatform')

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def expand_calc_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expand the calcFeatures JSON column into separate columns and return the complete DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame with calcFeatures column containing JSON data
    
    Returns:
    pd.DataFrame: Expanded DataFrame with all original columns plus JSON features as separate columns
    """
    df_expanded = df.copy()
    calc_features_list = []
    
    for idx, calc_features_str in enumerate(df['calcFeatures']):
        try:
            features_dict = json.loads(calc_features_str.replace("'", '"'))
            calc_features_list.append(features_dict)
        except (json.JSONDecodeError, AttributeError) as e:
            logger.warning(f"Could not parse calcFeatures at index {idx}: {e}")
            calc_features_list.append({})
    
    calc_features_df = pd.DataFrame(calc_features_list)
    calc_features_df = calc_features_df.add_prefix('calc_')
    
    df_expanded = df_expanded.reset_index(drop=True)
    calc_features_df = calc_features_df.reset_index(drop=True)
    
    result_df = pd.concat([df_expanded, calc_features_df], axis=1)
    return result_df

def expand_calc_features_robust(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expand the calcFeatures JSON column into separate columns with better error handling.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame with calcFeatures column containing JSON data
    
    Returns:
    pd.DataFrame: Expanded DataFrame with all original columns plus JSON features as separate columns
    """
    df_expanded = df.copy()
    calc_features_data = []
    
    for idx, row in df.iterrows():
        calc_features_str = row['calcFeatures']
        
        if pd.isna(calc_features_str) or calc_features_str == '':
            calc_features_data.append({})
            continue
        
        try:
            cleaned_str = calc_features_str.replace("'", '"').replace('None', 'null').replace('True', 'true').replace('False', 'false')
            features_dict = json.loads(cleaned_str)
            calc_features_data.append(features_dict)
        except Exception as e:
            logger.warning(f"Could not parse calcFeatures at index {idx}: {e}")
            logger.warning(f"Problematic string: {calc_features_str[:100]}...")
            calc_features_data.append({})
    
    calc_features_df = pd.DataFrame(calc_features_data)
    calc_features_df = calc_features_df.add_prefix('feat_')
    result_df = pd.concat([df_expanded, calc_features_df], axis=1)
    
    logger.info(f"Original DataFrame shape: {df.shape}")
    logger.info(f"Expanded DataFrame shape: {result_df.shape}")
    logger.info(f"Added {len(calc_features_df.columns)} new columns from calcFeatures")
    
    return result_df

def transform_data(
    d1: pd.DataFrame, 
    feature_column: List[str], 
    a: str = 'demo_score', 
    modelDisplayName: str = 'Cash_beta_trench1_Demo_backscore', 
    tc: str = "", 
    subscription_name: str = 'sil_march 25 models'
) -> pd.DataFrame:
    """
    Transforms input data into a structured format suitable for model scoring output.
    
    Parameters:
    - d1 (pd.DataFrame): Input DataFrame containing raw data.
    - feature_column (List[str]): List of column names to include in the 'calcFeature' JSON.
    - a (str): Column name containing the prediction score. Default is 'demo_score'.
    - modelDisplayName (str): Name of the model used for scoring.
    - tc (str): Trench category (optional).
    - subscription_name (str): Name of the subscription or model group.
    
    Returns:
    - pd.DataFrame: Transformed DataFrame with structured output.
    """
    df = d1.copy()
    output_data = []
    
    for _, row in df.iterrows():
        calc_feature = {}
        
        for col in feature_column:
            if col in row and pd.notna(row[col]):
                if isinstance(row[col], pd.Timestamp):
                    calc_feature[col] = row[col].isoformat()
                else:
                    calc_feature[col] = row[col]
        
        current_time = datetime.now().isoformat()
        
        output_row = {
            "customerId": row['customer_id'],
            "digitalLoanAccountId": row['digitalLoanAccountId'],
            "crifApplicationId": str(uuid.uuid4()),
            "prediction": row.get(a, 0),
            "start_time": current_time,
            "end_time": current_time,
            "modelDisplayName": modelDisplayName,
            "modelVersionId": "v1",
            "calcFeature": json.dumps(calc_feature, default=str),
            "subscription_name": subscription_name,
            "message_id": str(uuid.uuid4()),
            "publish_time": current_time,
            "attributes": "{}",
            "trenchCategory": tc,
            "deviceOs": row['osType'],
            "Data_selection": row['Data_selection'],
            "Application_date": row['application_date'],
        }
        
        output_data.append(output_row)
    
    output_df = pd.DataFrame(output_data)
    return output_df

def transform_datav2(
    d1: pd.DataFrame, 
    feature_column: List[str], 
    a: str = 'demo_score', 
    modelDisplayName: str = 'Cash_beta_trench1_Demo_backscore', 
    tc: str = "", 
    subscription_name: str = 'sil_march 25 models'
) -> pd.DataFrame:
    """
    Transforms input data into a structured format suitable for model scoring output (v2).
    
    Parameters:
    - d1 (pd.DataFrame): Input DataFrame containing raw data.
    - feature_column (List[str]): List of column names to include in the 'calcFeature' JSON.
    - a (str): Column name containing the prediction score. Default is 'demo_score'.
    - modelDisplayName (str): Name of the model used for scoring.
    - tc (str): Trench category (optional).
    - subscription_name (str): Name of the subscription or model group.
    
    Returns:
    - pd.DataFrame: Transformed DataFrame with structured output.
    """
    df = d1.copy()
    output_data = []
    
    for _, row in df.iterrows():
        calc_feature = {}
        
        for col in feature_column:
            if col in row and pd.notna(row[col]):
                if isinstance(row[col], pd.Timestamp):
                    calc_feature[col] = row[col].isoformat()
                else:
                    calc_feature[col] = row[col]
        
        current_time = datetime.now().isoformat()
        
        output_row = {
            "customerId": row['customer_id'],
            "digitalLoanAccountId": row['digitalLoanAccountId'],
            "crifApplicationId": str(uuid.uuid4()),
            "prediction": row.get(a, 0),
            "start_time": current_time,
            "end_time": current_time,
            "modelDisplayName": modelDisplayName,
            "modelVersionId": "v2",
            "calcFeature": json.dumps(calc_feature, default=str),
            "subscription_name": subscription_name,
            "message_id": str(uuid.uuid4()),
            "publish_time": current_time,
            "attributes": "{}",
            "trenchCategory": tc,
            "deviceOs": row['osType'],
            "Data_selection": row['Data_selection'],
            "Application_date": row['application_date'],
        }
        
        output_data.append(output_row)
    
    output_df = pd.DataFrame(output_data)
    return output_df

# ============================================================================
# PSI FUNCTIONS
# ============================================================================

def identify_feature_types(df: pd.DataFrame, feature_list: List[str]) -> Dict[str, List[str]]:
    """
    Identify categorical and numerical features from the feature list.
    """
    categorical_features = []
    numerical_features = []
    
    for feature in feature_list:
        if feature not in df.columns:
            logger.warning(f"Feature '{feature}' not found in dataframe")
            continue
        
        if pd.api.types.is_numeric_dtype(df[feature]):
            unique_vals = df[feature].nunique()
            if unique_vals < 15 and df[feature].dropna().apply(lambda x: x == int(x) if isinstance(x, (int, float)) else False).all():
                categorical_features.append(feature)
            else:
                numerical_features.append(feature)
        else:
            categorical_features.append(feature)
    
    return {
        'categorical': categorical_features,
        'numerical': numerical_features
    }

def create_bins_for_features(
    df: pd.DataFrame,
    numerical_features: List[str],
    categorical_features: List[str],
    train_period_df: pd.DataFrame
) -> Dict:
    """
    Create bins for numerical features (deciles with fallback) and categorical features (top 6 + others)
    based on the entire training period data.
    """
    binning_info = {}
    
    # Create bins for numerical features with fallback strategy
    for feature in numerical_features:
        valid_data = train_period_df[feature].dropna()
        
        if len(valid_data) == 0:
            binning_info[feature] = {'type': 'numerical', 'bins': None, 'bin_ranges': {}}
            continue
        
        bins = None
        bin_count = None
        
        # Try 10 bins (deciles)
        try:
            test_bins = np.percentile(valid_data, np.arange(0, 101, 10))
            test_bins = np.unique(test_bins)
            if len(test_bins) >= 11:
                bins = test_bins
                bin_count = 10
        except Exception as e:
            pass
        
        # If 10 bins not possible, try 5 bins
        if bins is None:
            try:
                test_bins = np.percentile(valid_data, np.arange(0, 101, 20))
                test_bins = np.unique(test_bins)
                if len(test_bins) >= 6:
                    bins = test_bins
                    bin_count = 5
            except Exception as e:
                pass
        
        # If 5 bins not possible, try 3 bins
        if bins is None:
            try:
                test_bins = np.percentile(valid_data, [0, 33.33, 66.67, 100])
                test_bins = np.unique(test_bins)
                if len(test_bins) >= 4:
                    bins = test_bins
                    bin_count = 3
            except Exception as e:
                pass
        
        # If still no bins possible, use equal distance bins of 5
        if bins is None:
            logger.warning(f"Feature '{feature}' has insufficient variance - cannot create standard bins")
            logger.info(f"Feature '{feature}': Using equal distance bins of 5")
            
            min_val = valid_data.min()
            max_val = valid_data.max()
            
            bins = np.linspace(min_val, max_val, 6)
            bins = np.unique(bins)
            bin_count = len(bins) - 1
            
            if bin_count == 1:
                bins = np.array([min_val - 0.1, min_val, min_val + 0.1])
                bin_count = 2
                logger.info(f"Feature '{feature}': Constant value ({min_val}). Created 2 equal distance bins with buffer")
        
        # Add infinity edges to capture all values
        bins = bins.copy()
        bins[0] = -np.inf
        bins[-1] = np.inf
        
        logger.info(f"Feature '{feature}': Created {bin_count} bins")
        
        bin_ranges = {}
        for i in range(len(bins)-1):
            bin_name = f"Bin_{i+1}"
            bin_ranges[bin_name] = {
                'min': bins[i],
                'max': bins[i+1],
                'range_str': f"[{bins[i]:.2f}, {bins[i+1]:.2f}]" if not np.isinf(bins[i]) and not np.isinf(bins[i+1]) else f"({bins[i]}, {bins[i+1]})"
            }
        
        binning_info[feature] = {
            'type': 'numerical',
            'bins': bins,
            'bin_ranges': bin_ranges,
            'bin_count': bin_count
        }
    
    # Create bins for categorical features (top 6 + others) using training period
    for feature in categorical_features:
        value_counts = train_period_df[feature].value_counts()
        unique_categories = value_counts.index.tolist()
        logger.info(f"Unique categories: {unique_categories}")
        
        if len(unique_categories) <= 6:
            top_categories = unique_categories
        else:
            top_categories = value_counts.nlargest(6).index.tolist()
        
        logger.info(f"Top categories for feature '{feature}': {top_categories}")
        
        binning_info[feature] = {
            'type': 'categorical',
            'top_categories': top_categories,
            'bin_ranges': {}
        }
    
    return binning_info

def apply_binning(df: pd.DataFrame, feature: str, binning_info: Dict) -> pd.Series:
    """
    Apply binning to a feature based on binning information.
    """
    if binning_info['type'] == 'numerical':
        if binning_info['bins'] is None:
            return pd.Series(['Missing'] * len(df), index=df.index)
        
        bins = binning_info['bins']
        labels = [f"Bin_{i+1}" for i in range(len(bins)-1)]
        
        binned = pd.cut(df[feature],
                       bins=bins,
                       labels=labels,
                       include_lowest=True,
                       duplicates='drop')
        
        binned = binned.astype(str)
        binned[df[feature].isna()] = 'Missing'
        
        return binned
    else:
        top_cats = binning_info['top_categories']
        
        if pd.api.types.is_categorical_dtype(df[feature]):
            feature_data = df[feature].astype(str)
        else:
            feature_data = df[feature].astype(str)
        
        feature_data = feature_data.replace('nan', 'Missing')
        top_cats_str = [str(cat) for cat in top_cats]
        
        binned = feature_data.apply(lambda x: x if x in top_cats_str else ('Others' if x != 'Missing' else 'Missing'))
        
        return binned

def calculate_psi(expected_pct: pd.Series, actual_pct: pd.Series, epsilon: float = 0.0001) -> float:
    """
    Calculate Population Stability Index with proper epsilon handling and renormalization.
    """
    all_bins = expected_pct.index.union(actual_pct.index)
    expected_pct = expected_pct.reindex(all_bins, fill_value=0)
    actual_pct = actual_pct.reindex(all_bins, fill_value=0)
    
    expected_pct = expected_pct.apply(lambda x: epsilon if x == 0 else x)
    actual_pct = actual_pct.apply(lambda x: epsilon if x == 0 else x)
    
    expected_pct = expected_pct / expected_pct.sum()
    actual_pct = actual_pct / actual_pct.sum()
    
    psi_value = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    
    return psi_value

def calculate_month_on_month_psi(
    df: pd.DataFrame,
    feature_list: List[str],
    segment_columns: List[str],
    month_col: str = 'Application_month',
    data_selection_col: str = 'Data_selection',
    account_id_col: str = 'digitalLoanAccountId'
) -> pd.DataFrame:
    """
    Calculate PSI for each feature comparing training period (June 2024 to March 2025)
    vs each month after March 2025, overall and by segments.
    """
    df = df.copy()
    train_df = df[df[data_selection_col] == 'Train'].copy()
    test_df = df[df[data_selection_col] != 'Train'].copy()
    
    if len(train_df) == 0:
        raise ValueError("No training data found. Check Data_selection column.")
    
    logger.info(f"Training period: {train_df[month_col].min()} to {train_df[month_col].max()}")
    logger.info(f"Test period: {test_df[month_col].min()} to {test_df[month_col].max()}")
    
    feature_types = identify_feature_types(df, feature_list)
    binning_info = create_bins_for_features(df, feature_types['numerical'], feature_types['categorical'], train_df)
    test_months = sorted(test_df[month_col].unique())
    results = []
    
    # Calculate overall PSI
    for feature in feature_list:
        if feature not in df.columns:
            continue
        
        df[f'{feature}_binned'] = apply_binning(df, feature, binning_info[feature])
        train_baseline = df[df[data_selection_col] == 'Train'][f'{feature}_binned'].value_counts(normalize=True)
        
        for month in test_months:
            actual_dist = df[df[month_col] == month][f'{feature}_binned'].value_counts(normalize=True)
            psi_value = calculate_psi(train_baseline, actual_dist)
            
            expected_avg_pct = train_baseline.mean() * 100
            actual_avg_pct = actual_dist.mean() * 100
            
            results.append({
                'Feature': feature,
                'Feature_Type': binning_info[feature]['type'],
                'Segment_Column': 'Overall',
                'Segment_Value': 'All',
                'Month': f"{month}",
                'Base_Month': 'Train (Jun 2024 - Mar 2025)',
                'Current_Month': month,
                'Expected_Percentage': expected_avg_pct,
                'Actual_Percentage': actual_avg_pct,
                'PSI': psi_value
            })
    
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
                
                train_segment = segment_df[segment_df[data_selection_col] == 'Train']
                if len(train_segment) == 0:
                    continue
                
                train_baseline = train_segment[f'{feature}_binned'].value_counts(normalize=True)
                
                for month in test_months:
                    actual_segment = segment_df[segment_df[month_col] == month]
                    if len(actual_segment) == 0:
                        continue
                    
                    actual_dist = actual_segment[f'{feature}_binned'].value_counts(normalize=True)
                    psi_value = calculate_psi(train_baseline, actual_dist)
                    
                    expected_avg_pct = train_baseline.mean() * 100
                    actual_avg_pct = actual_dist.mean() * 100
                    
                    base_segment_count = train_segment[account_id_col].nunique()
                    actual_segment_count = actual_segment[account_id_col].nunique()
                    
                    results.append({
                        'Feature': feature,
                        'Feature_Type': binning_info[feature]['type'],
                        'Segment_Column': segment_col,
                        'Segment_Value': segment_val,
                        'Month': f"{month}",
                        'Base_Month': 'Train (Jun 2024 - Mar 2025)',
                        'Current_Month': month,
                        'Base_Count': base_segment_count,
                        'Actual_Count': actual_segment_count,
                        'Expected_Percentage': expected_avg_pct,
                        'Actual_Percentage': actual_avg_pct,
                        'PSI': psi_value
                    })
    
    return pd.DataFrame(results)

def calculate_bin_level_psi(
    df: pd.DataFrame,
    feature_list: List[str],
    segment_columns: List[str],
    month_col: str = 'Application_month',
    data_selection_col: str = 'Data_selection',
    account_id_col: str = 'digitalLoanAccountId'
) -> pd.DataFrame:
    """
    Calculate bin-level PSI for each feature comparing training period
    vs each month after March 2025, overall and by segments.
    """
    df = df.copy()
    train_df = df[df[data_selection_col] == 'Train'].copy()
    test_df = df[df[data_selection_col] != 'Train'].copy()
    
    if len(train_df) == 0:
        raise ValueError("No training data found. Check Data_selection column.")
    
    logger.info(f"Training period: {train_df[month_col].min()} to {train_df[month_col].max()}")
    logger.info(f"Test period: {test_df[month_col].min()} to {test_df[month_col].max()}")
    
    feature_types = identify_feature_types(df, feature_list)
    binning_info = create_bins_for_features(df, feature_types['numerical'], feature_types['categorical'], train_df)
    test_months = sorted(test_df[month_col].unique())
    results = []
    epsilon = 0.0001
    
    # Calculate overall bin-level PSI
    for feature in feature_list:
        if feature not in df.columns:
            continue
        
        df[f'{feature}_binned'] = apply_binning(df, feature, binning_info[feature])
        train_baseline = df[df[data_selection_col] == 'Train'][f'{feature}_binned'].value_counts(normalize=True)
        
        for month in test_months:
            month_data = df[df[month_col] == month]
            actual_dist = month_data[f'{feature}_binned'].value_counts(normalize=True)
            
            base_count = df[df[data_selection_col] == 'Train'][account_id_col].nunique()
            actual_count = month_data[account_id_col].nunique()
            
            all_bins = train_baseline.index.union(actual_dist.index)
            
            for bin_name in all_bins:
                expected_pct = train_baseline.get(bin_name, 0)
                actual_pct = actual_dist.get(bin_name, 0)
                
                expected_pct = epsilon if expected_pct == 0 else expected_pct
                actual_pct = epsilon if actual_pct == 0 else actual_pct
                
                bin_psi = (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)
                
                bin_ranges = binning_info[feature]['bin_ranges']
                if bin_name in bin_ranges:
                    bin_min = bin_ranges[bin_name]['min']
                    bin_max = bin_ranges[bin_name]['max']
                    bin_range = bin_ranges[bin_name]['range_str']
                else:
                    bin_min = None
                    bin_max = None
                    bin_range = bin_name
                
                results.append({
                    'Feature': feature,
                    'Feature_Type': binning_info[feature]['type'],
                    'Segment_Column': 'Overall',
                    'Segment_Value': 'All',
                    'Month': f"{month}",
                    'Base_Month': 'Train (Jun 2024 - Mar 2025)',
                    'Current_Month': month,
                    'Base_Count': base_count,
                    'Actual_Count': actual_count,
                    'Bin': bin_name,
                    'Bin_Range': bin_range,
                    'Bin_Min': bin_min,
                    'Bin_Max': bin_max,
                    'Base_Percentage': (train_baseline.get(bin_name, 0) * 100),
                    'Actual_Percentage': (actual_dist.get(bin_name, 0) * 100),
                    'Bin_PSI': bin_psi
                })
    
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
                
                train_segment = segment_df[segment_df[data_selection_col] == 'Train']
                if len(train_segment) == 0:
                    continue
                
                train_baseline = train_segment[f'{feature}_binned'].value_counts(normalize=True)
                
                for month in test_months:
                    actual_segment = segment_df[segment_df[month_col] == month]
                    if len(actual_segment) == 0:
                        continue
                    
                    actual_dist = actual_segment[f'{feature}_binned'].value_counts(normalize=True)
                    
                    base_segment_count = train_segment[account_id_col].nunique()
                    actual_segment_count = actual_segment[account_id_col].nunique()
                    
                    all_bins = train_baseline.index.union(actual_dist.index)
                    
                    for bin_name in all_bins:
                        expected_pct = train_baseline.get(bin_name, 0)
                        actual_pct = actual_dist.get(bin_name, 0)
                        
                        expected_pct = epsilon if expected_pct == 0 else expected_pct
                        actual_pct = epsilon if actual_pct == 0 else actual_pct
                        
                        bin_psi = (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)
                        
                        bin_ranges = binning_info[feature]['bin_ranges']
                        if bin_name in bin_ranges:
                            bin_min = bin_ranges[bin_name]['min']
                            bin_max = bin_ranges[bin_name]['max']
                            bin_range = bin_ranges[bin_name]['range_str']
                        else:
                            bin_min = None
                            bin_max = None
                            bin_range = bin_name
                        
                        results.append({
                            'Feature': feature,
                            'Feature_Type': binning_info[feature]['type'],
                            'Segment_Column': segment_col,
                            'Segment_Value': segment_val,
                            'Month': f"{month}",
                            'Base_Month': 'Train (Jun 2024 - Mar 2025)',
                            'Current_Month': month,
                            'Base_Count': base_segment_count,
                            'Actual_Count': actual_segment_count,
                            'Bin': bin_name,
                            'Bin_Range': bin_range,
                            'Bin_Min': bin_min,
                            'Bin_Max': bin_max,
                            'Base_Percentage': (train_baseline.get(bin_name, 0) * 100),
                            'Actual_Percentage': (actual_dist.get(bin_name, 0) * 100),
                            'Bin_PSI': bin_psi
                        })
    
    return pd.DataFrame(results)

# ============================================================================
# SIL V1 MODELS
# ============================================================================

def run_sil_models():
    """Execute all SIL model transformations and upload to BigQuery"""
    logger.info("Starting SIL V1 models processing...")
    
    # ============================================================================
    # Alpha - CIC-SIL-Model
    # ============================================================================
    logger.info("Processing Alpha - CIC-SIL-Model...")
    sq = """
    select distinct
        r.customerId customer_id ,
        r.digitalLoanAccountId,
        r.cic_score,
        r.cic_Personal_Loans_granted_contracts_amt_24M,
        r.cic_days_since_last_inquiry, 
        r.cic_cnt_active_contracts,
        r.cic_vel_contract_nongranted_cnt_12on24,
        r.cic_max_amt_granted_24M, 
        r.cic_zero_non_granted_ever_flag,
        r.cic_tot_active_contracts_util,
        r.cic_vel_contract_granted_amt_12on24,
        r.cic_zero_granted_ever_flag,
        case when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%andro%' then 'android'
        when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%os%' then 'ios'
        when lower(loanmaster.deviceType) like '%andro%' then 'android'
        else 'ios' end osType,
        date(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime)) application_date,
        case when date(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime))
             between '2024-06-01' and '2024-09-30' then 'Dev_Train'
             when date(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime)) < '2024-06-01' then 'Pre_Train'
                      else 'Dev_Test' end as Data_selection 
    from risk_mart.sil_risk_ds_master_20230101_20250309_v2 r
    left join risk_credit_mis.loan_master_table loanmaster
      ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
    where cic_score is not null
    and date(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime)) < '2025-03-24'
    ;
    """
    
    data = client.query(sq).to_dataframe(progress_bar_type='tqdm')
    logger.info(f"The shape of the dataframe is: {data.shape}")
    
    feature_column = ['cic_Personal_Loans_granted_contracts_amt_24M',
           'cic_days_since_last_inquiry', 'cic_cnt_active_contracts',
           'cic_vel_contract_nongranted_cnt_12on24', 'cic_max_amt_granted_24M',
           'cic_zero_non_granted_ever_flag', 'cic_tot_active_contracts_util',
           'cic_vel_contract_granted_amt_12on24', 'cic_zero_granted_ever_flag']
    
    dfd = transform_data(data, feature_column, a='cic_score', modelDisplayName='Alpha - CIC-SIL-Model')
    
    # Log results
    result = dfd.groupby('Data_selection').agg(
        digitalLoanAccountId_count=('digitalLoanAccountId', 'count'),
        Application_date_min=('Application_date', 'min'),
        Application_date_max=('Application_date', 'max')
    ).reset_index()
    
    logger.info("\nAlpha - CIC-SIL-Model Data Summary:")
    logger.info(result.to_string())
    
    # Upload to BigQuery
    table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.ml_training_model_run_details_20260116"
    job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
    job = client.load_table_from_dataframe(dfd, table_id, job_config=job_config)
    job.result()
    logger.info("Alpha - CIC-SIL-Model uploaded to BigQuery")
    
    # ============================================================================
    # Alpha - StackingModel
    # ============================================================================
    logger.info("\nProcessing Alpha - StackingModel...")
    sq = """
    select distinct 
    r.customerId customer_id ,
    r.digitalLoanAccountId,
    r.alpha_stack_score,
    r.beta_demo_score,
    r.cic_score,
    r.apps_score,
    r.credo_gen_score,
        case when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%andro%' then 'android'
        when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%os%' then 'ios'
        when lower(loanmaster.deviceType) like '%andro%' then 'android'
        else 'ios' end osType,
    date(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime)) application_date,
    case when date(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime))
            between '2024-06-01' and '2024-09-30' then 'Dev_Train'
            when date(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime)) < '2024-06-01' then 'Pre_Train'
                    else 'Dev_Test' end as Data_selection 
    from `risk_mart.sil_risk_ds_master_20230101_20250309_v2` r
    left join risk_credit_mis.loan_master_table loanmaster
      ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
    where alpha_stack_score is not null
    and date(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime)) < '2025-03-24'
    ;
    """
    
    data = client.query(sq).to_dataframe(progress_bar_type='tqdm')
    logger.info(f"The shape of the dataframe is: {data.shape}")
    
    data.rename(columns={'beta_demo_score':'sb_demo_score', 'cic_score':'s_cic_score', 
                        'apps_score':'s_apps_score', 'credo_gen_score':'s_credo_score'}, inplace=True)
    
    feature_column = ['sb_demo_score', 's_cic_score', 's_apps_score', 's_credo_score']
    dfd = transform_data(data, feature_column, a='alpha_stack_score', modelDisplayName='Alpha - StackingModel')
    
    # Log results
    result = dfd.groupby('Data_selection').agg(
        digitalLoanAccountId_count=('digitalLoanAccountId', 'count'),
        Application_date_min=('Application_date', 'min'),
        Application_date_max=('Application_date', 'max')
    ).reset_index()
    
    logger.info("\nAlpha - StackingModel Data Summary:")
    logger.info(result.to_string())
    
    # Upload to BigQuery
    job_config = bigquery.LoadJobConfig(write_disposition="WRITE_APPEND")
    job = client.load_table_from_dataframe(dfd, table_id, job_config=job_config)
    job.result()
    logger.info("Alpha - StackingModel uploaded to BigQuery")
    
    # ============================================================================
    # Beta - AppsScoreModel
    # ============================================================================
    logger.info("\nProcessing Beta - AppsScoreModel...")
    sq = """
    select distinct
    r.customerId customer_id ,
    r.digitalLoanAccountId,
    r.apps_score,
    r.app_cnt_absence_tag_30d,
    r.app_cnt_absence_tag_90d ,
    r.app_cnt_business_ever ,
    r.app_cnt_competitors_30d ,
    r.app_cnt_competitors_90d ,
    r.app_cnt_education_ever ,
    r.app_cnt_finance_7d ,
    r.app_cnt_finance_90d ,
    r.app_cnt_music_and_audio_ever ,
    r.app_cnt_payday_90d ,
    r.app_cnt_rated_for_3plus_ever ,
    r.app_cnt_travel_and_local_ever ,
    r.app_first_competitors_install_to_apply_days ,
    r.app_first_payday_install_to_apply_days ,
    r.app_median_time_bw_installed_mins_30d ,
    r.app_vel_finance_30_over_365 ,
        case when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%andro%' then 'android'
        when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%os%' then 'ios'
        when lower(loanmaster.deviceType) like '%andro%' then 'android'
        else 'ios' end osType,
        date(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime)) application_date,
        case when date(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime))
             between '2023-12-01' and '2024-06-30' then 'Dev_Train'
             when date(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime)) < '2023-12-01' then 'Pre_Train'
                      else 'Dev_Test' end as Data_selection 
    from `risk_mart.sil_risk_ds_master_20230101_20250309_v2` r
    left join risk_credit_mis.loan_master_table loanmaster
      ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
    where apps_score is not null
    and date(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime)) < '2025-03-20'
    """
    
    data = client.query(sq).to_dataframe(progress_bar_type='tqdm')
    logger.info(f"The shape of the dataframe is: {data.shape}")
    
    feature_column = ['app_cnt_rated_for_3plus_ever',
           'app_cnt_education_ever', 'app_cnt_business_ever',
           'app_cnt_music_and_audio_ever',
           'app_cnt_travel_and_local_ever', 'app_cnt_finance_7d',
           'app_cnt_competitors_30d', 'app_cnt_absence_tag_30d',
            'app_cnt_absence_tag_90d',
           'app_cnt_finance_90d', 'app_cnt_competitors_90d',
           'app_cnt_payday_90d',
           'app_median_time_bw_installed_mins_30d',
           'app_first_competitors_install_to_apply_days',
           'app_first_payday_install_to_apply_days',
           'app_vel_finance_30_over_365']
    
    dfd = transform_data(data, feature_column, a='apps_score', modelDisplayName='Beta - AppsScoreModel')
    
    # Log results
    result = dfd.groupby('Data_selection').agg(
        digitalLoanAccountId_count=('digitalLoanAccountId', 'count'),
        Application_date_min=('Application_date', 'min'),
        Application_date_max=('Application_date', 'max')
    ).reset_index()
    
    logger.info("\nBeta - AppsScoreModel Data Summary:")
    logger.info(result.to_string())
    
    # Upload to BigQuery
    job = client.load_table_from_dataframe(dfd, table_id, job_config=job_config)
    job.result()
    logger.info("Beta - AppsScoreModel uploaded to BigQuery")
    
    # ============================================================================
    # Beta - DemoScoreModel
    # ============================================================================
    logger.info("\nProcessing Beta - DemoScoreModel...")
    sq = """
    select distinct
    r.customerId customer_id ,
    r.digitalLoanAccountId,
    r.beta_demo_score,
    r.beta_de_ln_vas_opted_flag ,
    r.beta_de_ln_doc_type_rolled ,
    r.beta_de_ln_marital_status ,
    r.beta_de_ln_age_bin ,
    r.beta_de_ln_province_bin ,
    r.beta_de_ln_ref2_type ,
    r.beta_de_ln_education_level ,
    r.beta_de_ln_ref1_type ,
    r.beta_de_ln_industry_new_bin ,
    r.beta_de_ln_appln_day_of_week ,
    r.beta_de_onb_name_email_match_score ,
    r.beta_de_ln_employment_type_new_bin ,
    r.beta_de_ln_telconame ,
    r.beta_de_time_bw_onb_loan_appln_mins ,
    r.beta_de_ln_source_of_funds_new_bin ,
    r.beta_de_ln_brand_bin ,
    r.beta_de_ln_email_primary_domain ,
        case when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%andro%' then 'android'
        when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%os%' then 'ios'
        when lower(loanmaster.deviceType) like '%andro%' then 'android'
        else 'ios' end osType,
    date(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime)) application_date,
    case when date(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime))
            between '2023-07-01' and '2024-06-30' then 'Dev_Train'
            when date(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime)) < '2023-07-01' then 'Pre_Train'
                    else 'Dev_Test' end as Data_selection 
    from `risk_mart.sil_risk_ds_master_20230101_20250309_v2` r
    left join risk_credit_mis.loan_master_table loanmaster
      ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
    where beta_demo_score is not null
    and date(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime)) < '2025-03-20'
    ;
    """
    
    data = client.query(sq).to_dataframe(progress_bar_type='tqdm')
    logger.info(f"The shape of the dataframe is: {data.shape}")
    
    feature_column = ['beta_de_ln_vas_opted_flag',
           'beta_de_ln_doc_type_rolled', 'beta_de_ln_marital_status',
           'beta_de_ln_age_bin', 'beta_de_ln_province_bin',
           'beta_de_ln_ref2_type', 'beta_de_ln_education_level',
           'beta_de_ln_ref1_type', 'beta_de_ln_industry_new_bin',
           'beta_de_ln_appln_day_of_week',
           'beta_de_onb_name_email_match_score',
           'beta_de_ln_employment_type_new_bin', 'beta_de_ln_telconame',
           'beta_de_time_bw_onb_loan_appln_mins',
           'beta_de_ln_source_of_funds_new_bin', 'beta_de_ln_brand_bin',
           'beta_de_ln_email_primary_domain']
    
    dfd = transform_data(data, feature_column, a='beta_demo_score', modelDisplayName='Beta - DemoScoreModel')
    
    # Log results
    result = dfd.groupby('Data_selection').agg(
        digitalLoanAccountId_count=('digitalLoanAccountId', 'count'),
        Application_date_min=('Application_date', 'min'),
        Application_date_max=('Application_date', 'max')
    ).reset_index()
    
    logger.info("\nBeta - DemoScoreModel Data Summary:")
    logger.info(result.to_string())
    
    # Upload to BigQuery
    job = client.load_table_from_dataframe(dfd, table_id, job_config=job_config)
    job.result()
    logger.info("Beta - DemoScoreModel uploaded to BigQuery")
    
    # ============================================================================
    # Beta - StackScoreModel
    # ============================================================================
    logger.info("\nProcessing Beta - StackScoreModel...")
    sq = """
    select  distinct
    r.customerId customer_id ,
    r.digitalLoanAccountId,
    r.beta_stack_score,
    r.apps_score,
    r.credo_gen_score,
    r.beta_demo_score,
        case when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%andro%' then 'android'
        when lower(coalesce(loanmaster.osversion_v2, loanmaster.osVersion)) like '%os%' then 'ios'
        when lower(loanmaster.deviceType) like '%andro%' then 'android'
        else 'ios' end osType,
    date(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime)) application_date,
    case when date(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime))
            between '2023-07-01' and '2024-06-30' then 'Dev_Train'
            when date(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime)) < '2023-07-01' then 'Pre_Train'
                    else 'Dev_Test' end as Data_selection 
    from `risk_mart.sil_risk_ds_master_20230101_20250309_v2` r
    left join risk_credit_mis.loan_master_table loanmaster
      ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
    where beta_stack_score is not null
    and date(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime)) < '2025-03-20'
    """
    
    data = client.query(sq).to_dataframe(progress_bar_type='tqdm')
    logger.info(f"The shape of the dataframe is: {data.shape}")
    
    feature_column = ['apps_score', 'credo_gen_score', 'beta_demo_score']
    dfd = transform_data(data, feature_column, a='beta_stack_score', modelDisplayName='Beta - StackScoreModel')
    
    # Log results
    result = dfd.groupby('Data_selection').agg(
        digitalLoanAccountId_count=('digitalLoanAccountId', 'count'),
        Application_date_min=('Application_date', 'min'),
        Application_date_max=('Application_date', 'max')
    ).reset_index()
    
    logger.info("\nBeta - StackScoreModel Data Summary:")
    logger.info(result.to_string())
    
    # Upload to BigQuery
    job = client.load_table_from_dataframe(dfd, table_id, job_config=job_config)
    job.result()
    logger.info("Beta - StackScoreModel uploaded to BigQuery")
    
    logger.info("\nSIL V1 models processing completed!")

# ============================================================================
# CASH V1 MODELS
# ============================================================================

def run_cash_models():
    """Execute all Cash model transformations and upload to BigQuery"""
    logger.info("\nStarting Cash V1 models processing...")
    
    # ============================================================================
    # Alpha-Cash-Stack-Model - Trench 1
    # ============================================================================
    logger.info("\nProcessing Alpha-Cash-Stack-Model - Trench 1...")
    sq = """
    select 
      r.customer_id,
      r.digitalLoanAccountId, 
      r.stack_score,
      r.demo_score,
      r.apps_score,
      r.credo_score,
      r.cic_score,
      r.stack_score_norm,
      r.ln_os_type osType,
       date(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime)) application_date,
        case when date(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime))
             between '2024-10-01' and '2025-02-28' then 'Dev_Train'
             when date(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime)) < '2024-10-01' then 'Pre_Train'
                      else 'Dev_Test' end as Data_selection 
    from worktable_data_analysis.cash_alpha_trench1_applied_loans_backscored_20241001_20250930 r
    left join risk_credit_mis.loan_master_table loanmaster
      ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
    where r.stack_score is not null
    and date(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime))
    < '2025-09-24'
    ;
    """
    
    data = client.query(sq).to_dataframe(progress_bar_type='tqdm')
    logger.info(f"The shape of the dataframe is: {data.shape}")
    
    feature_column = ['demo_score', 'apps_score', 'credo_score', 'cic_score', 'stack_score_norm']
    dfd = transform_data(data, feature_column, a='stack_score', modelDisplayName='Alpha-Cash-Stack-Model', tc='Trench 1', subscription_name='Cash September 25 Models')
    
    # Log results
    result = dfd.groupby('Data_selection').agg(
        digitalLoanAccountId_count=('digitalLoanAccountId', 'count'),
        Application_date_min=('Application_date', 'min'),
        Application_date_max=('Application_date', 'max')
    ).reset_index()
    
    logger.info("\nAlpha-Cash-Stack-Model (Trench 1) Data Summary:")
    logger.info(result.to_string())
    
    # Upload to BigQuery
    table_id = "prj-prod-dataplatform.dap_ds_poweruser_playground.ml_training_model_run_details_20260116"
    job_config = bigquery.LoadJobConfig(write_disposition="WRITE_APPEND")
    job = client.load_table_from_dataframe(dfd, table_id, job_config=job_config)
    job.result()
    logger.info("Alpha-Cash-Stack-Model (Trench 1) uploaded to BigQuery")
    
    # ============================================================================
    # Alpha-Cash-Stack-Model - Trench 2
    # ============================================================================
    logger.info("\nProcessing Alpha-Cash-Stack-Model - Trench 2...")
    sq = """
    select 
      r.customer_id,
      r.digitalLoanAccountId, 
      r.stack_score,
      r.demo_score,
      r.apps_score,
      r.credo_score,
      r.cic_score,
      r.stack_score_norm,
      r.ln_os_type osType,
      date(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime)) application_date,
        case when date(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime))
             between '2024-10-01' and '2025-02-28' then 'Dev_Train'
             when date(if(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime)) < '2024-10-01' then 'Pre_Train'
                      else 'Dev_Test' end as Data_selection 
    from worktable_data_analysis.cash_alpha_trench2_applied_loans_backscored_20241001_20250930 r
    left join risk_credit_mis.loan_master_table loanmaster
      ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
    where r.stack_score is not null
    and date(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime)) < '2025-09-24'
    ;
    """
    
    data = client.query(sq).to_dataframe(progress_bar_type='tqdm')
    logger.info(f"The shape of the dataframe is: {data.shape}")
    
    dfd = transform_data(data, feature_column, a='stack_score', modelDisplayName='Alpha-Cash-Stack-Model', tc='Trench 2', subscription_name='Cash September 25 Models')
    
    # Log results
    result = dfd.groupby('Data_selection').agg(
        digitalLoanAccountId_count=('digitalLoanAccountId', 'count'),
        Application_date_min=('Application_date', 'min'),
        Application_date_max=('Application_date', 'max')
    ).reset_index()
    
    logger.info("\nAlpha-Cash-Stack-Model (Trench 2) Data Summary:")
    logger.info(result.to_string())
    
    # Upload to BigQuery
    job = client.load_table_from_dataframe(dfd, table_id, job_config=job_config)
    job.result()
    logger.info("Alpha-Cash-Stack-Model (Trench 2) uploaded to BigQuery")
    
    # ============================================================================
    # Alpha-Cash-Stack-Model - Trench 3
    # ============================================================================
    logger.info("\nProcessing Alpha-Cash-Stack-Model - Trench 3...")
    sq = """
    select 
      r.customer_id,
      r.digitalLoanAccountId, 
      r.stack_score,
      r.demo_score,
      r.apps_score,
      r.credo_score,
      r.cic_score,
      r.stack_score_norm,
      r.ln_os_type osType,
      date(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime)) application_date,
        case when date(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime))
             between '2024-10-01' and '2025-02-28' then 'Dev_Train'
             when date(if(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime)) < '2024-10-01' then 'Pre_Train'
                      else 'Dev_Test' end as Data_selection 
    from worktable_data_analysis.cash_alpha_trench3_applied_loans_backscored_20241001_20250930 r
    left join risk_credit_mis.loan_master_table loanmaster
      ON loanmaster.digitalLoanAccountId = r.digitalLoanAccountId
    where r.stack_score is not null
    and date(IF(loanmaster.new_loan_type = 'Flex-up', loanmaster.startApplyDateTime, loanmaster.termsAndConditionsSubmitDateTime))< '2025-09-24'
    ;
    """
    
    data = client.query(sq).to_dataframe(progress_bar_type='tqdm')
    logger.info(f"The shape of the dataframe is: {data.shape}")
    
    dfd = transform_data(data, feature_column, a='stack_score', modelDisplayName='Alpha-Cash-Stack-Model', tc='Trench 3', subscription_name='Cash September 25 Models')
    
    # Log results
    result = dfd.groupby('Data_selection').agg(
        digitalLoanAccountId_count=('digitalLoanAccountId', 'count'),
        Application_date_min=('Application_date', 'min'),
        Application_date_max=('Application_date', 'max')
    ).reset_index()
    
    logger.info("\nAlpha-Cash-Stack-Model (Trench 3) Data Summary:")
    logger.info(result.to_string())
    
    # Upload to BigQuery
    job = client.load_table_from_dataframe(dfd, table_id, job_config=job_config)
    job.result()
    logger.info("Alpha-Cash-Stack-Model (Trench 3) uploaded to BigQuery")
    
    logger.info("\nCash V1 models processing completed!")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    logger.info("=" * 80)
    logger.info("Starting PSI Training Pipeline")
    logger.info("=" * 80)
    
    try:
        # Run SIL models
        run_sil_models()
        
        # Run Cash models
        run_cash_models()
        
        logger.info("\n" + "=" * 80)
        logger.info("All models processed successfully!")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Error in pipeline execution: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()