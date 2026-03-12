from datetime import timedelta
from itertools import combinations, product

import numpy as np
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
