


def csi_monitoring():
    # Import libraries
    import time
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from google.cloud import bigquery
    import os
    client = bigquery.Client(project='prj-prod-dataplatform')


    """Create a backup table"""

    # Changes
    # sq = """Create table `dap_ds_poweruser_playground.F_CSI_MODEL_FEATURES_BIN_TAB_backup` as
    # select * from `dap_ds_poweruser_playground.F_CSI_MODEL_FEATURES_BIN_TAB_new`;
    # """

    # query_job = client.query(sq)
    # query_job.result()

    """Check the backup table"""

    # sq = """select * from  `dap_ds_poweruser_playground.F_CSI_MODEL_FEATURES_BIN_TAB_backup`;"""

    # dd = client.query(sq).to_dataframe(progress_bar_type='tqdm')
    # print(dd.shape)

    """Drop the old table"""

    # Changes
    sq = """drop table dap_ds_poweruser_playground.F_CSI_MODEL_FEATURES_BIN_TAB_new"""

    query_job = client.query(sq)
    query_job.result()

    """
    `1. Calculate the CSI the same way as we have calculated PSI but for top 5 features of a Model.`<br>
    `2. We will calculate it for overall, then for each categories of user_type, prod_type and os_type.`<br>

    # CIC Score CIS
    """

    sq = """
    WITH cicscorebase AS (
    SELECT
        digitalLoanAccountId,
        FORMAT_DATE('%Y-%m', ln_appln_submit_datetime) AS Application_month,
        FORMAT_DATE('%F', DATE_TRUNC(ln_appln_submit_datetime, WEEK(MONDAY))) AS Appl_week_start_date,
        EXTRACT(WEEK(MONDAY) FROM ln_appln_submit_datetime) AS Appl_week_number,
        ln_user_type,
        ln_loan_type,
        ln_prod_type,
        ln_os_type,
        cic_hit_flag,
        cic_score,
        CASE
        WHEN DATE_TRUNC(ln_appln_submit_datetime, DAY) BETWEEN '2024-06-01' AND '2024-09-30' THEN 'Train'
        WHEN DATE_TRUNC(ln_appln_submit_datetime, DAY) >= '2024-10-01' THEN 'Test'
        ELSE 'Other'
        END AS dataselection,
        CASE
        WHEN cic_Personal_Loans_granted_contracts_amt_24M IS NULL THEN 'g. missing'
        WHEN CAST(cic_Personal_Loans_granted_contracts_amt_24M AS FLOAT64) BETWEEN 0.999 AND 5650.6 THEN 'a. 0.999-5650.6'
        WHEN CAST(cic_Personal_Loans_granted_contracts_amt_24M AS FLOAT64) BETWEEN 5650.7 AND 12447.8 THEN 'b. 5650.7-12447.8'
        WHEN CAST(cic_Personal_Loans_granted_contracts_amt_24M AS FLOAT64) BETWEEN 12447.9 AND 24000.0 THEN 'c. 12447.9-24000.0'
        WHEN CAST(cic_Personal_Loans_granted_contracts_amt_24M AS FLOAT64) BETWEEN 24000.1 AND 50500.0 THEN 'd. 24000.1-50500'
        WHEN CAST(cic_Personal_Loans_granted_contracts_amt_24M AS FLOAT64) BETWEEN 50500.1 AND 2738545.0 THEN 'e. 50500.1-2738545.0'
        WHEN CAST(cic_Personal_Loans_granted_contracts_amt_24M AS FLOAT64) > 2738545.0 THEN 'f. >2738545.0'
        END AS cic_Personal_Loans_granted_contracts_amt_24M_bin,
        CASE
        WHEN cic_cnt_active_contracts IS NULL THEN 'd. missing'
        WHEN CAST(cic_cnt_active_contracts AS FLOAT64) BETWEEN 0.999 AND 2.0 THEN 'a. 0.999-2.0'
        WHEN CAST(cic_cnt_active_contracts AS FLOAT64) BETWEEN 2.1 AND 88.0 THEN 'b. 2.1-88.0'
        WHEN CAST(cic_cnt_active_contracts AS FLOAT64) > 88.0 THEN 'c. >88.0'
        END AS cic_cnt_active_contracts_bin,
        CASE
        WHEN cic_vel_contract_nongranted_cnt_12on24 IS NULL THEN 'd. missing'
        WHEN CAST(cic_vel_contract_nongranted_cnt_12on24 AS FLOAT64) BETWEEN 0.285 AND 1.994 THEN 'a. 0.285-1.994'
        WHEN CAST(cic_vel_contract_nongranted_cnt_12on24 AS FLOAT64) BETWEEN 1.995 AND 2.012 THEN 'b. 1.995-2.012'
        WHEN CAST(cic_vel_contract_nongranted_cnt_12on24 AS FLOAT64) > 2.012 THEN 'c. >2.012'
        END AS cic_vel_contract_nongranted_cnt_12on24_bin,
        CASE
        WHEN cic_days_since_last_inquiry IS NULL THEN 'g. missing'
        WHEN CAST(cic_days_since_last_inquiry AS FLOAT64) BETWEEN -0.001 AND 10.0 THEN 'a. -0.001-10.0'
        WHEN CAST(cic_days_since_last_inquiry AS FLOAT64) BETWEEN 10.1 AND 117.0 THEN 'b. 10.1-117.0'
        WHEN CAST(cic_days_since_last_inquiry AS FLOAT64) BETWEEN 117.1 AND 281.0 THEN 'c. 117.1-281.0'
        WHEN CAST(cic_days_since_last_inquiry AS FLOAT64) BETWEEN 281.1 AND 832.0 THEN 'd. 281.1-832.0'
        WHEN CAST(cic_days_since_last_inquiry AS FLOAT64) BETWEEN 832.1 AND 10844.0 THEN 'e. 832.1-10844.0'
        WHEN CAST(cic_days_since_last_inquiry AS FLOAT64) > 10844.0 THEN 'f. >10844.0'
        END AS cic_days_since_last_inquiry_bin,
        CASE
        WHEN cic_max_amt_granted_24M IS NULL THEN 'g. missing'
        WHEN CAST(cic_max_amt_granted_24M AS FLOAT64) BETWEEN -0.001 AND 5394.2 THEN 'a. -0.001-5394.2'
        WHEN CAST(cic_max_amt_granted_24M AS FLOAT64) BETWEEN 5394.3 AND 10502.4 THEN 'b. 5394.3-10502.4'
        WHEN CAST(cic_max_amt_granted_24M AS FLOAT64) BETWEEN 10502.5 AND 20000.0 THEN 'c. 10502.5-20000.0'
        WHEN CAST(cic_max_amt_granted_24M AS FLOAT64) BETWEEN 20000.1 AND 40000.0 THEN 'd. 20000.1-40000.0'
        WHEN CAST(cic_max_amt_granted_24M AS FLOAT64) BETWEEN 40000.1 AND 8000000.0 THEN 'e. 40000.1-8000000.0'
        WHEN CAST(cic_max_amt_granted_24M AS FLOAT64) > 8000000.0 THEN 'f. >8000000.0'
        END AS cic_max_amt_granted_24M_bin
    FROM `prj-prod-dataplatform.risk_credit_mis.application_score_master`
    WHERE cic_called_flag = 1
    AND DATE_TRUNC(ln_appln_submit_datetime, DAY) >= '2024-06-01'
    )
    SELECT * FROM cicscorebase;

    """

    cicscoredf = client.query(sq).to_dataframe(progress_bar_type='tqdm')

    cicscoredf['Application_month'][cicscoredf['dataselection'] == 'Test'].unique()

    """For Overall CSI calculation"""

    def calculate_categorical_csi(train_dist, test_dist):
        """
        Calculate csi for categorical features.

        Args:
            train_dist: Distribution of categories in training set
            test_dist: Distribution of categories in test set

        Returns:
            float: csi value
        """
        # Ensure both distributions have the same categories
        all_categories = set(train_dist.index) | set(test_dist.index)

        # Align distributions
        train_dist_aligned = train_dist.reindex(all_categories, fill_value=0.0001)  # Small value to avoid division by zero
        test_dist_aligned = test_dist.reindex(all_categories, fill_value=0.0001)

        # Calculate csi
        csi_values = (test_dist_aligned - train_dist_aligned) * np.log(test_dist_aligned / train_dist_aligned)
        return csi_values.sum()

    def calculate_bin_csi(train_df, test_df, feature):
        """
        Calculate csi for each bin value within a feature.

        Args:
            train_df: Training DataFrame
            test_df: Test DataFrame
            feature: Feature name to calculate bin-level csi for

        Returns:
            DataFrame: csi results for each bin value
        """
        # Get all unique bin values across both datasets
        all_bins = set(train_df[feature].dropna().unique()) | set(test_df[feature].dropna().unique())

        # Results list for bin-level csi
        bin_csi_results = []

        # Calculate distribution for the entire feature in training set (for reference)
        train_counts = train_df[feature].value_counts(dropna=True)
        train_distribution = train_counts / train_counts.sum()

        # Calculate distribution for the entire feature in test set (for reference)
        test_counts = test_df[feature].value_counts(dropna=True)
        test_distribution = test_counts / test_counts.sum()

        # Calculate overall csi for the feature
        overall_csi = calculate_categorical_csi(train_distribution, test_distribution)

        # Calculate csi for each bin value
        for bin_value in all_bins:
            # Calculate percentage of this bin in train set
            train_bin_count = train_df[train_df[feature] == bin_value].shape[0]
            train_total = train_df.shape[0]
            train_bin_pct = train_bin_count / train_total if train_total > 0 else 0.0001

            # Calculate percentage of this bin in test set
            test_bin_count = test_df[test_df[feature] == bin_value].shape[0]
            test_total = test_df.shape[0]
            test_bin_pct = test_bin_count / test_total if test_total > 0 else 0.0001

            # Calculate csi for this bin
            if train_bin_pct < 0.0001:
                train_bin_pct = 0.0001  # Avoid division by zero
            if test_bin_pct < 0.0001:
                test_bin_pct = 0.0001  # Avoid division by zero

            bin_csi = (test_bin_pct - train_bin_pct) * np.log(test_bin_pct / train_bin_pct)

            # Store result
            bin_csi_results.append({
                'feature': feature,
                'bin_value': bin_value,
                'train_pct': train_bin_pct,
                'test_pct': test_bin_pct,
                'bin_csi': bin_csi,
                'feature_csi': overall_csi
            })

        return pd.DataFrame(bin_csi_results)

    def calculate_segmented_bin_csi(df, feature_list, segment_columns=None):
        """
        Calculate csi for each bin value within multiple features, overall and by segments.

        Args:
            df: DataFrame containing the data
            feature_list: List of feature names to calculate csi for
            segment_columns: List of columns to segment by (e.g., ['ln_user_type', 'ln_os_type'])

        Returns:
            DataFrame: csi results for each bin value by month and segment
        """
        # Initialize results list
        all_results = []

        # If no segment columns are provided, use an empty list
        if segment_columns is None:
            segment_columns = []

        # First, calculate overall csi for each bin
        overall_results = calculate_feature_bin_csi(df, feature_list)
        overall_results['segment_type'] = 'Overall'
        overall_results['segment_value'] = 'All'
        all_results.append(overall_results)

        # Then calculate csi for each segment column
        for segment_col in segment_columns:
            if segment_col not in df.columns:
                print(f"Warning: {segment_col} not found in DataFrame. Skipping.")
                continue

            # Get unique segment values
            segment_values = df[segment_col].dropna().unique()

            for segment_val in segment_values:
                # Filter data for this segment
                segment_df = df[df[segment_col] == segment_val]

                # Skip if not enough data
                if len(segment_df) < 50:  # Arbitrary threshold
                    print(f"Skipping {segment_col}={segment_val} due to insufficient data ({len(segment_df)} rows).")
                    continue

                # Calculate csi for this segment
                try:
                    segment_results = calculate_feature_bin_csi(segment_df, feature_list)
                    segment_results['segment_type'] = segment_col
                    segment_results['segment_value'] = segment_val
                    all_results.append(segment_results)
                except Exception as e:
                    print(f"Error calculating csi for {segment_col}={segment_val}: {e}")

        # Combine all results
        if all_results:
            combined_results = pd.concat(all_results, ignore_index=True)
            return combined_results
        else:
            return pd.DataFrame()

    def calculate_feature_bin_csi(df, feature_list):
        """
        Calculate csi for each bin value within multiple features.

        Args:
            df: DataFrame containing the data
            feature_list: List of feature names to calculate csi for

        Returns:
            DataFrame: csi results for each bin value by month
        """
        # Make a copy to avoid modifying the original DataFrame
        df_copy = df.copy()

        # Separate train and test data
        train_df = df_copy[df_copy['dataselection'] == 'Train']
        test_df = df_copy[df_copy['dataselection'] == 'Test']

        # Skip if either dataset is empty
        if train_df.empty or test_df.empty:
            print("Warning: Either train or test dataset is empty. Skipping csi calculation.")
            return pd.DataFrame()

        # Handle Application_month based on its type
        if isinstance(df_copy['Application_month'].iloc[0], str):
            # If it's a string in format 'YYYY-MM-DD', extract just 'YYYY-MM'
            last_train_month_str = str(train_df['Application_month'].max())
            if len(last_train_month_str) >= 7:  # Ensure we have at least YYYY-MM
                last_train_month_str = last_train_month_str[:7]  # Extract YYYY-MM part
        else:
            # If it's already a datetime object
            try:
                last_train_month = pd.to_datetime(train_df['Application_month'].max())
                last_train_month_str = last_train_month.strftime('%Y-%m')
            except:
                # Fallback if conversion fails
                last_train_month_str = str(train_df['Application_month'].max())

        # Store all bin-level csi results
        all_bin_results = []

        # Calculate distribution for each feature in the training set
        for feature in feature_list:
            if feature not in train_df.columns:
                print(f"Warning: Feature {feature} not found in training data. Skipping.")
                continue

            # Calculate bin-level csi for the training set against itself (always 0)
            train_bins = train_df[feature].dropna().unique()
            for bin_value in train_bins:
                all_bin_results.append({
                    'Month': last_train_month_str,
                    'feature': feature,
                    'bin_value': bin_value,
                    'DateCategory': 'a_Training',
                    'train_pct': (train_df[feature] == bin_value).mean(),
                    'test_pct': (train_df[feature] == bin_value).mean(),  # Same as train for training data
                    'bin_csi': 0.0,  # csi against itself is 0
                    'feature_csi': 0.0,  # Overall csi against itself is 0
                    'account_count': train_df['digitalLoanAccountId'].nunique()
                })

        # Get unique months from test set and sort them
        test_months = sorted(test_df['Application_month'].unique())

        # Create mapping of months to prefixed labels (b, c, d, etc.)
        prefix_map = {}
        for i, month in enumerate(test_months):
            month_str = str(month)
            if isinstance(month, str) and len(month_str) >= 7:
                month_str = month_str[:7]  # Extract YYYY-MM part

            # Use letters b, c, d, etc. for subsequent months (a is reserved for Training)
            prefix = chr(98 + i)  # ASCII: b=98, c=99, etc.
            prefix_map[month] = f"{prefix}_{month_str}"

        # Calculate monthly csi for each feature and bin in the test set
        for month in test_months:
            original_month_str = str(month)
            if isinstance(month, str) and len(original_month_str) >= 7:
                original_month_str = original_month_str[:7]  # Extract YYYY-MM part

            # Use the prefixed month string for sorting
            month_str = prefix_map[month]

            month_df = test_df[test_df['Application_month'] == month]

            if not month_df.empty:
                month_accounts = month_df['digitalLoanAccountId'].nunique()

                for feature in feature_list:
                    if feature not in month_df.columns:
                        continue

                    # Calculate bin-level csi for this feature in this month
                    try:
                        # Get all unique bin values for this feature across train and test
                        all_bins = set(train_df[feature].dropna().unique()) | set(month_df[feature].dropna().unique())

                        # Calculate overall feature csi for reference
                        train_counts = train_df[feature].value_counts(dropna=True, normalize=True)
                        test_counts = month_df[feature].value_counts(dropna=True, normalize=True)
                        overall_csi = calculate_categorical_csi(train_counts, test_counts)

                        # Calculate csi for each bin
                        for bin_value in all_bins:
                            # Calculate percentages
                            train_pct = (train_df[feature] == bin_value).mean()
                            test_pct = (month_df[feature] == bin_value).mean()

                            # Add small value to avoid division by zero
                            if train_pct < 0.0001:
                                train_pct = 0.0001
                            if test_pct < 0.0001:
                                test_pct = 0.0001

                            # Calculate csi for this bin
                            bin_csi = (test_pct - train_pct) * np.log(test_pct / train_pct)

                            # Store result
                            all_bin_results.append({
                                'Month': original_month_str,
                                'MonthSortKey': month_str,
                                'feature': feature,
                                'bin_value': bin_value,
                                'DateCategory': 'b_Monthly',
                                'train_pct': train_pct,
                                'test_pct': test_pct,
                                'bin_csi': bin_csi,
                                'feature_csi': overall_csi,
                                'account_count': month_accounts
                            })
                    except Exception as e:
                        print(f"Error calculating bin csi for {feature} in {month}: {e}")

        # Create the output DataFrame
        return pd.DataFrame(all_bin_results)

    # Features list
    feature_list = [
        'cic_Personal_Loans_granted_contracts_amt_24M_bin',
        'cic_cnt_active_contracts_bin',
        'cic_vel_contract_nongranted_cnt_12on24_bin',
        'cic_days_since_last_inquiry_bin',
        'cic_max_amt_granted_24M_bin'
    ]

    # Define segment columns
    segment_columns = ['ln_user_type', 'ln_prod_type', 'ln_os_type']

    # Calculate bin-level csi for overall and by segments
    bin_results = calculate_segmented_bin_csi(cicscoredf, feature_list, segment_columns)


    # Try to combine with s_apps_score results if they exist (continued)
    try:
        # First ensure the s_apps_score_output_df has the same structure
        if 'MonthSortKey' not in s_apps_score_output_df.columns:
            s_apps_score_output_df['MonthSortKey'] = s_apps_score_output_df['Month']
            # Update DateCategory with prefix
            s_apps_score_output_df['DateCategory'] = s_apps_score_output_df['DateCategory'].apply(
                lambda x: 'a_Training' if x == 'Training' else 'b_Monthly'
            )

        # Add segment info to s_apps_score_output_df
        s_apps_score_output_df['segment_type'] = 'Overall'
        s_apps_score_output_df['segment_value'] = 'All'

        # Add bin_value column to s_apps_score_output_df (as 'All' for feature-level csi)
        s_apps_score_output_df['bin_value'] = 'All'

        # Rename csivalues to feature_csi for consistency
        if 'csivalues' in s_apps_score_output_df.columns:
            s_apps_score_output_df = s_apps_score_output_df.rename(columns={'csivalues': 'feature_csi'})

        # Add bin_csi column (same as feature_csi for feature-level csi)
        if 'feature_csi' in s_apps_score_output_df.columns:
            s_apps_score_output_df['bin_csi'] = s_apps_score_output_df['feature_csi']

        # Replace 'scorename' with 'feature' for consistency
        if 'scorename' in s_apps_score_output_df.columns:
            s_apps_score_output_df['feature'] = s_apps_score_output_df['feature'].fillna(s_apps_score_output_df['scorename'])
            s_apps_score_output_df = s_apps_score_output_df.drop('scorename', axis=1)

        # Combine with bin_results
        combined_results = pd.concat([s_apps_score_output_df, bin_results], ignore_index=True)
    except NameError:
        # If s_apps_score_output_df doesn't exist, just use bin_results
        combined_results = bin_results

    # Sort by segment_type, segment_value, feature, bin_value, and MonthSortKey
    sort_columns = ['segment_type', 'segment_value', 'feature', 'bin_value']
    if 'MonthSortKey' in combined_results.columns:
        sort_columns.append('MonthSortKey')
    else:
        sort_columns.append('Month')

    combined_results = combined_results.sort_values(sort_columns)

    # Save the detailed bin-level results
    # Changes
    # combined_results.to_csv('bin_level_csi_results_cicscore.csv', index=False)

    # Function to create pivot table for a given segment and feature
    def create_bin_pivot(data, segment_type, segment_value, feature=None):
        # Filter by segment
        segment_data = data[(data['segment_type'] == segment_type) &
                        (data['segment_value'] == segment_value)]

        # Further filter by feature if specified
        if feature:
            segment_data = segment_data[segment_data['feature'] == feature]

        # Create pivot table - rows are bin values, columns are months
        pivot = segment_data.pivot_table(
            index=['feature', 'bin_value'],
            columns=['MonthSortKey'] if 'MonthSortKey' in segment_data.columns else ['Month'],
            values='bin_csi',
            aggfunc='first'
        )

        return pivot

    # Create bin pivot tables for overall and by segments
    unique_segment_combos = combined_results[['segment_type', 'segment_value']].drop_duplicates()
    unique_features = combined_results['feature'].unique()

    # # Question - do we need this Excel file?
    # # Create Excel writer to save all pivots in one file
    # with pd.ExcelWriter('bin_level_csi_pivots_cicscore.xlsx') as writer:
    #     # First, create overall pivot with all features and bins
    #     overall_pivot = create_bin_pivot(combined_results, 'Overall', 'All')
    #     overall_pivot.to_excel(writer, sheet_name='Overall_All_Features')
    #     print("Created overall pivot table for all features")

    #     # Create separate pivot for each feature (across all segments)
    #     for feature in unique_features:
    #         # Create pivot for this feature - Overall segment
    #         feature_pivot = create_bin_pivot(combined_results, 'Overall', 'All', feature)

    #         # Make sheet name Excel-friendly (31 char limit, no special chars)
    #         sheet_name = f"Overall_{feature[-20:]}"
    #         sheet_name = sheet_name.replace("/", "_").replace("\\", "_")[:31]

    #         feature_pivot.to_excel(writer, sheet_name=sheet_name)
    #         print(f"Created pivot for feature: {feature}")

    #     # Create separate pivot for each segment and feature combination
    #     for _, segment_row in unique_segment_combos.iterrows():
    #         segment_type = segment_row['segment_type']
    #         segment_value = segment_row['segment_value']

    #         # Skip Overall segment as we already handled it
    #         if segment_type == 'Overall' and segment_value == 'All':
    #             continue

    #         # Create segment-specific pivots for each feature
    #         for feature in unique_features:
    #             # Filter data for this segment and feature
    #             segment_feature_data = combined_results[
    #                 (combined_results['segment_type'] == segment_type) &
    #                 (combined_results['segment_value'] == segment_value) &
    #                 (combined_results['feature'] == feature)
    #             ]

    #             # Skip if no data
    #             if segment_feature_data.empty:
    #                 continue

    #             # Create pivot
    #             pivot = segment_feature_data.pivot_table(
    #                 index=['bin_value'],
    #                 columns=['MonthSortKey'] if 'MonthSortKey' in segment_feature_data.columns else ['Month'],
    #                 values='bin_csi',
    #                 aggfunc='first'
    #             )

    #             # Make sheet name Excel-friendly
    #             segment_name = f"{segment_type}_{segment_value}"
    #             feature_name = feature[-10:]  # Use last 10 chars of feature name to keep sheet name short
    #             sheet_name = f"{segment_name}_{feature_name}"
    #             sheet_name = sheet_name.replace("/", "_").replace("\\", "_")[:31]

    #             pivot.to_excel(writer, sheet_name=sheet_name)
    #             print(f"Created pivot for {segment_type}={segment_value}, feature={feature}")


    # Calculate bin contribution to total csi
    summary_data = []

    for segment_type in combined_results['segment_type'].unique():
        for segment_value in combined_results[combined_results['segment_type'] == segment_type]['segment_value'].unique():
            for feature in combined_results['feature'].unique():
                # Get data for this segment and feature
                segment_feature_data = combined_results[
                    (combined_results['segment_type'] == segment_type) &
                    (combined_results['segment_value'] == segment_value) &
                    (combined_results['feature'] == feature)
                ]

                if segment_feature_data.empty:
                    continue

                # Get unique months
                months = segment_feature_data['Month'].unique()
                print(months)

                for month in months:
                    month_data = segment_feature_data[segment_feature_data['Month'] == month]

                    # Get feature csi (should be same for all bins in this feature/month/segment)
                    feature_csi = month_data['feature_csi'].iloc[0] if not month_data.empty else 0

                    # Get top contributing bins
                    if not month_data.empty and 'bin_csi' in month_data.columns:
                        # Sort by absolute bin_csi value to get top contributors
                        top_bins = month_data.sort_values('bin_csi', key=abs, ascending=False)

                        # Take top 3 bins
                        for i, (_, bin_row) in enumerate(top_bins.iterrows()):
                            if i >= 3:  # Limit to top 3
                                break

                            bin_value = bin_row['bin_value']
                            bin_csi = bin_row['bin_csi']

                            # Calculate contribution percentage
                            pct_contribution = (bin_csi / feature_csi * 100) if feature_csi != 0 else 0

                            summary_data.append({
                                'segment_type': segment_type,
                                'segment_value': segment_value,
                                'feature': feature,
                                'Month': month,
                                'feature_csi': feature_csi,
                                'bin_value': bin_value,
                                'bin_csi': bin_csi,
                                'pct_contribution': pct_contribution,
                                'rank': i + 1
                            })

    # Question - do we need this Excel file?
    # Create summary DataFrame
    if summary_data:
        summary_df = pd.DataFrame(summary_data)

        # Pivot to get a table with top contributors
        contribution_pivot = summary_df.pivot_table(
            index=['segment_type', 'segment_value', 'feature', 'Month', 'feature_csi'],
            columns=['rank'],
            values=['bin_value', 'bin_csi', 'pct_contribution'],
            aggfunc='first'
        )

        # Save to Excel
        # Changes
        # contribution_pivot.to_excel('bin_contribution_summary_cic_score.xlsx')

    else:
        print("No data available for bin contribution summary")

    combined_results

    combined_results['Month'] = combined_results['Month'].replace('2024-09', '2024-06-2024-09')
    combined_results['MonthSortKey'] = combined_results['MonthSortKey'].fillna('a_2024-06-2024-09')
    combined_results['Month'] = combined_results['Month'].apply(lambda x: x.split(' 00:00:00')[0] if'00:00:00' in x else x)
    combined_results['scorename'] = 'CIC_Score'
    combined_results['Modelname'] = 'SIL CIC Model'
    combined_results['Description'] = 'Train period from 2024-06 to 2024-09'
    combined_results.sort_values(by='Month', ascending=False)

    dataset_id = 'dap_ds_poweruser_playground'
    table_id = 'F_CSI_MODEL_FEATURES_BIN_TAB_new' # Changes
    # Define the table schema as per your DataFrame columns
    schema = [
        bigquery.SchemaField("Month", "string"),
        bigquery.SchemaField("feature", "string"),
        bigquery.SchemaField("bin_value", "string"),
        bigquery.SchemaField("DateCategory", "string"),
        bigquery.SchemaField("train_pct", "float64"),
        bigquery.SchemaField("test_pct", "float64"),
        bigquery.SchemaField("bin_csi", "float64"),
        bigquery.SchemaField("feature_csi", "float64"),
        bigquery.SchemaField("account_count", "int64"),
        bigquery.SchemaField("MonthSortKey", "string"),
        bigquery.SchemaField("segment_type", "string"),
        bigquery.SchemaField("segment_value", "string"),
        bigquery.SchemaField("scorename", "string"),
        bigquery.SchemaField("Modelname", "string"),
        bigquery.SchemaField("Description", "string"),
        ]
    # Create the dataset reference
    dataset_ref = client.dataset(dataset_id)
    # Define the table reference
    table_ref = dataset_ref.table(table_id)
    # Configure the job to overwrite the table if it already exists
    job_config = bigquery.LoadJobConfig(schema = schema)
    # Load the DataFrame into BigQuery
    job = client.load_table_from_dataframe(combined_results, table_ref, job_config=job_config)
    # Wait for the job to complete
    job.result()

    """beta_demo_score"""

    sq = """
    with sildemo as
    (select
    digitalLoanAccountId,
    FORMAT_DATE('%Y-%m', ln_appln_submit_datetime) Application_month,
    FORMAT_DATE('%F', DATE_TRUNC(ln_appln_submit_datetime, WEEK(MONDAY))) as Appl_week_start_date,
    EXTRACT(WEEK(MONDAY) FROM ln_appln_submit_datetime) as Appl_week_number,
    ln_user_type,
    ln_loan_type,
    ln_prod_type,
    ln_os_type,
    beta_demo_score,
    case when date_trunc(ln_appln_submit_datetime, day) between '2023-07-01' and '2024-06-30' then 'Train'
        when date_trunc(ln_appln_submit_datetime, day) >= '2024-07-01' then 'Test'
        Else 'Other' end dataselection,
    case when beta_de_ln_vas_opted_flag is null then 'c. missing'
        when beta_de_ln_vas_opted_flag = '1' then 'a. 1'
        when beta_de_ln_vas_opted_flag = '0' then 'b. 0'
        end beta_de_ln_vas_opted_flag_bin,
    case when beta_de_ln_doc_type_rolled is null then 'i. missing'
        when beta_de_ln_doc_type_rolled like 'Driving License' then 'a. Driving License'
        when beta_de_ln_doc_type_rolled like 'Others' then 'b. Others'
        when beta_de_ln_doc_type_rolled like 'Passport' then 'c. Passport'
        when beta_de_ln_doc_type_rolled like 'Postal ID Card' then 'd. Postal ID Card'
        when beta_de_ln_doc_type_rolled like 'Professional ID Card' then 'e. Professional ID Card'
        when beta_de_ln_doc_type_rolled like 'Social Security Card' then 'f. Social Security Card'
        when beta_de_ln_doc_type_rolled like 'UMID Card' then 'g. UMID Card'
        when beta_de_ln_doc_type_rolled like 'Voter Card' then 'h. Voter Card'
        else 'j. NA' end beta_de_ln_doc_type_rolled_bin,
    case when (beta_de_ln_marital_status is null or beta_de_ln_marital_status like 'nan') then 'f. missing'
        when beta_de_ln_marital_status like 'Annulled / Separated' then 'a. Annulled / Separated'
        when beta_de_ln_marital_status like 'Married' then 'b. Married'
        when beta_de_ln_marital_status like 'Single' then 'c. Single'
        when beta_de_ln_marital_status like 'Widow / Widower' then 'd. Widow / Widower'
        when beta_de_ln_marital_status like 'With a Live-in Partner' then 'e. With a Live-in Partner'
        else 'g. NA' end beta_de_ln_marital_status_bin,
    beta_de_ln_age_bin,
    case when (beta_de_ln_ref2_type is null or beta_de_ln_ref2_type like 'nan') then 'g. missing'
        when beta_de_ln_ref2_type like 'Child' then 'a. Child'
        when beta_de_ln_ref2_type like 'Co-worker' then 'b. Co-worker'
        when beta_de_ln_ref2_type like 'Friend' then 'c. Friend'
        when beta_de_ln_ref2_type like 'Parent' then 'd. Parent'
        when beta_de_ln_ref2_type like 'Sibling' then 'e. Sibling'
        when beta_de_ln_ref2_type like 'Spouse' then 'f. Spouse'
        else 'h. NA' end beta_de_ln_ref2_type_bin
    from prj-prod-dataplatform.risk_credit_mis.application_score_master
    where date_trunc(ln_appln_submit_datetime, day) >= '2023-07-01'
    )
    select * from sildemo;
    """

    sildemodf = client.query(sq).to_dataframe(progress_bar_type='tqdm')

    def calculate_categorical_csi(train_dist, test_dist):
        """
        Calculate csi for categorical features.

        Args:
            train_dist: Distribution of categories in training set
            test_dist: Distribution of categories in test set

        Returns:
            float: csi value
        """
        # Ensure both distributions have the same categories
        all_categories = set(train_dist.index) | set(test_dist.index)

        # Align distributions
        train_dist_aligned = train_dist.reindex(all_categories, fill_value=0.0001)  # Small value to avoid division by zero
        test_dist_aligned = test_dist.reindex(all_categories, fill_value=0.0001)

        # Calculate csi
        csi_values = (test_dist_aligned - train_dist_aligned) * np.log(test_dist_aligned / train_dist_aligned)
        return csi_values.sum()

    def calculate_bin_csi(train_df, test_df, feature):
        """
        Calculate csi for each bin value within a feature.

        Args:
            train_df: Training DataFrame
            test_df: Test DataFrame
            feature: Feature name to calculate bin-level csi for

        Returns:
            DataFrame: csi results for each bin value
        """
        # Get all unique bin values across both datasets
        all_bins = set(train_df[feature].dropna().unique()) | set(test_df[feature].dropna().unique())

        # Results list for bin-level csi
        bin_csi_results = []

        # Calculate distribution for the entire feature in training set (for reference)
        train_counts = train_df[feature].value_counts(dropna=True)
        train_distribution = train_counts / train_counts.sum()

        # Calculate distribution for the entire feature in test set (for reference)
        test_counts = test_df[feature].value_counts(dropna=True)
        test_distribution = test_counts / test_counts.sum()

        # Calculate overall csi for the feature
        overall_csi = calculate_categorical_csi(train_distribution, test_distribution)

        # Calculate csi for each bin value
        for bin_value in all_bins:
            # Calculate percentage of this bin in train set
            train_bin_count = train_df[train_df[feature] == bin_value].shape[0]
            train_total = train_df.shape[0]
            train_bin_pct = train_bin_count / train_total if train_total > 0 else 0.0001

            # Calculate percentage of this bin in test set
            test_bin_count = test_df[test_df[feature] == bin_value].shape[0]
            test_total = test_df.shape[0]
            test_bin_pct = test_bin_count / test_total if test_total > 0 else 0.0001

            # Calculate csi for this bin
            if train_bin_pct < 0.0001:
                train_bin_pct = 0.0001  # Avoid division by zero
            if test_bin_pct < 0.0001:
                test_bin_pct = 0.0001  # Avoid division by zero

            bin_csi = (test_bin_pct - train_bin_pct) * np.log(test_bin_pct / train_bin_pct)

            # Store result
            bin_csi_results.append({
                'feature': feature,
                'bin_value': bin_value,
                'train_pct': train_bin_pct,
                'test_pct': test_bin_pct,
                'bin_csi': bin_csi,
                'feature_csi': overall_csi
            })

        return pd.DataFrame(bin_csi_results)

    def calculate_segmented_bin_csi(df, feature_list, segment_columns=None):
        """
        Calculate csi for each bin value within multiple features, overall and by segments.

        Args:
            df: DataFrame containing the data
            feature_list: List of feature names to calculate csi for
            segment_columns: List of columns to segment by (e.g., ['ln_user_type', 'ln_os_type'])

        Returns:
            DataFrame: csi results for each bin value by month and segment
        """
        # Initialize results list
        all_results = []

        # If no segment columns are provided, use an empty list
        if segment_columns is None:
            segment_columns = []

        # First, calculate overall csi for each bin
        overall_results = calculate_feature_bin_csi(df, feature_list)
        overall_results['segment_type'] = 'Overall'
        overall_results['segment_value'] = 'All'
        all_results.append(overall_results)

        # Then calculate csi for each segment column
        for segment_col in segment_columns:
            if segment_col not in df.columns:
                print(f"Warning: {segment_col} not found in DataFrame. Skipping.")
                continue

            # Get unique segment values
            segment_values = df[segment_col].dropna().unique()

            for segment_val in segment_values:
                # Filter data for this segment
                segment_df = df[df[segment_col] == segment_val]

                # Skip if not enough data
                if len(segment_df) < 50:  # Arbitrary threshold
                    print(f"Skipping {segment_col}={segment_val} due to insufficient data ({len(segment_df)} rows).")
                    continue

                # Calculate csi for this segment
                try:
                    segment_results = calculate_feature_bin_csi(segment_df, feature_list)
                    segment_results['segment_type'] = segment_col
                    segment_results['segment_value'] = segment_val
                    all_results.append(segment_results)
                except Exception as e:
                    print(f"Error calculating csi for {segment_col}={segment_val}: {e}")

        # Combine all results
        if all_results:
            combined_results = pd.concat(all_results, ignore_index=True)
            return combined_results
        else:
            return pd.DataFrame()

    def calculate_feature_bin_csi(df, feature_list):
        """
        Calculate csi for each bin value within multiple features.

        Args:
            df: DataFrame containing the data
            feature_list: List of feature names to calculate csi for

        Returns:
            DataFrame: csi results for each bin value by month
        """
        # Make a copy to avoid modifying the original DataFrame
        df_copy = df.copy()

        # Separate train and test data
        train_df = df_copy[df_copy['dataselection'] == 'Train']
        test_df = df_copy[df_copy['dataselection'] == 'Test']

        # Skip if either dataset is empty
        if train_df.empty or test_df.empty:
            print("Warning: Either train or test dataset is empty. Skipping csi calculation.")
            return pd.DataFrame()

        # Handle Application_month based on its type
        if isinstance(df_copy['Application_month'].iloc[0], str):
            # If it's a string in format 'YYYY-MM-DD', extract just 'YYYY-MM'
            last_train_month_str = str(train_df['Application_month'].max())
            if len(last_train_month_str) >= 7:  # Ensure we have at least YYYY-MM
                last_train_month_str = last_train_month_str[:7]  # Extract YYYY-MM part
        else:
            # If it's already a datetime object
            try:
                last_train_month = pd.to_datetime(train_df['Application_month'].max())
                last_train_month_str = last_train_month.strftime('%Y-%m')
            except:
                # Fallback if conversion fails
                last_train_month_str = str(train_df['Application_month'].max())

        # Store all bin-level csi results
        all_bin_results = []

        # Calculate distribution for each feature in the training set
        for feature in feature_list:
            if feature not in train_df.columns:
                print(f"Warning: Feature {feature} not found in training data. Skipping.")
                continue

            # Calculate bin-level csi for the training set against itself (always 0)
            train_bins = train_df[feature].dropna().unique()
            for bin_value in train_bins:
                all_bin_results.append({
                    'Month': last_train_month_str,
                    'feature': feature,
                    'bin_value': bin_value,
                    'DateCategory': 'a_Training',
                    'train_pct': (train_df[feature] == bin_value).mean(),
                    'test_pct': (train_df[feature] == bin_value).mean(),  # Same as train for training data
                    'bin_csi': 0.0,  # csi against itself is 0
                    'feature_csi': 0.0,  # Overall csi against itself is 0
                    'account_count': train_df['digitalLoanAccountId'].nunique()
                })

        # Get unique months from test set and sort them
        test_months = sorted(test_df['Application_month'].unique())

        # Create mapping of months to prefixed labels (b, c, d, etc.)
        prefix_map = {}
        for i, month in enumerate(test_months):
            month_str = str(month)
            if isinstance(month, str) and len(month_str) >= 7:
                month_str = month_str[:7]  # Extract YYYY-MM part

            # Use letters b, c, d, etc. for subsequent months (a is reserved for Training)
            prefix = chr(98 + i)  # ASCII: b=98, c=99, etc.
            prefix_map[month] = f"{prefix}_{month_str}"

        # Calculate monthly csi for each feature and bin in the test set
        for month in test_months:
            original_month_str = str(month)
            if isinstance(month, str) and len(original_month_str) >= 7:
                original_month_str = original_month_str[:7]  # Extract YYYY-MM part

            # Use the prefixed month string for sorting
            month_str = prefix_map[month]

            month_df = test_df[test_df['Application_month'] == month]

            if not month_df.empty:
                month_accounts = month_df['digitalLoanAccountId'].nunique()

                for feature in feature_list:
                    if feature not in month_df.columns:
                        continue

                    # Calculate bin-level csi for this feature in this month
                    try:
                        # Get all unique bin values for this feature across train and test
                        all_bins = set(train_df[feature].dropna().unique()) | set(month_df[feature].dropna().unique())

                        # Calculate overall feature csi for reference
                        train_counts = train_df[feature].value_counts(dropna=True, normalize=True)
                        test_counts = month_df[feature].value_counts(dropna=True, normalize=True)
                        overall_csi = calculate_categorical_csi(train_counts, test_counts)

                        # Calculate csi for each bin
                        for bin_value in all_bins:
                            # Calculate percentages
                            train_pct = (train_df[feature] == bin_value).mean()
                            test_pct = (month_df[feature] == bin_value).mean()

                            # Add small value to avoid division by zero
                            if train_pct < 0.0001:
                                train_pct = 0.0001
                            if test_pct < 0.0001:
                                test_pct = 0.0001

                            # Calculate csi for this bin
                            bin_csi = (test_pct - train_pct) * np.log(test_pct / train_pct)

                            # Store result
                            all_bin_results.append({
                                'Month': original_month_str,
                                'MonthSortKey': month_str,
                                'feature': feature,
                                'bin_value': bin_value,
                                'DateCategory': 'b_Monthly',
                                'train_pct': train_pct,
                                'test_pct': test_pct,
                                'bin_csi': bin_csi,
                                'feature_csi': overall_csi,
                                'account_count': month_accounts
                            })
                    except Exception as e:
                        print(f"Error calculating bin csi for {feature} in {month}: {e}")

        # Create the output DataFrame
        return pd.DataFrame(all_bin_results)

    # Features list
    feature_list = [
        'beta_de_ln_vas_opted_flag_bin',
        'beta_de_ln_doc_type_rolled_bin',
        'beta_de_ln_marital_status_bin',
        'beta_de_ln_age_bin',
        'beta_de_ln_ref2_type_bin'
    ]

    # Define segment columns
    segment_columns = ['ln_user_type', 'ln_prod_type', 'ln_os_type']

    # Calculate bin-level csi for overall and by segments
    bin_results = calculate_segmented_bin_csi(sildemodf, feature_list, segment_columns)



    # Try to combine with s_apps_score results if they exist (continued)
    try:
        # First ensure the s_apps_score_output_df has the same structure
        if 'MonthSortKey' not in s_apps_score_output_df.columns:
            s_apps_score_output_df['MonthSortKey'] = s_apps_score_output_df['Month']
            # Update DateCategory with prefix
            s_apps_score_output_df['DateCategory'] = s_apps_score_output_df['DateCategory'].apply(
                lambda x: 'a_Training' if x == 'Training' else 'b_Monthly'
            )

        # Add segment info to s_apps_score_output_df
        s_apps_score_output_df['segment_type'] = 'Overall'
        s_apps_score_output_df['segment_value'] = 'All'

        # Add bin_value column to s_apps_score_output_df (as 'All' for feature-level csi)
        s_apps_score_output_df['bin_value'] = 'All'

        # Rename csivalues to feature_csi for consistency
        if 'csivalues' in s_apps_score_output_df.columns:
            s_apps_score_output_df = s_apps_score_output_df.rename(columns={'csivalues': 'feature_csi'})

        # Add bin_csi column (same as feature_csi for feature-level csi)
        if 'feature_csi' in s_apps_score_output_df.columns:
            s_apps_score_output_df['bin_csi'] = s_apps_score_output_df['feature_csi']

        # Replace 'scorename' with 'feature' for consistency
        if 'scorename' in s_apps_score_output_df.columns:
            s_apps_score_output_df['feature'] = s_apps_score_output_df['feature'].fillna(s_apps_score_output_df['scorename'])
            s_apps_score_output_df = s_apps_score_output_df.drop('scorename', axis=1)

        # Combine with bin_results
        combined_results = pd.concat([s_apps_score_output_df, bin_results], ignore_index=True)
    except NameError:
        # If s_apps_score_output_df doesn't exist, just use bin_results
        combined_results = bin_results

    # Sort by segment_type, segment_value, feature, bin_value, and MonthSortKey
    sort_columns = ['segment_type', 'segment_value', 'feature', 'bin_value']
    if 'MonthSortKey' in combined_results.columns:
        sort_columns.append('MonthSortKey')
    else:
        sort_columns.append('Month')

    combined_results = combined_results.sort_values(sort_columns)

    # Question - do we need this CSV file?
    # Save the detailed bin-level results
    # combined_results.to_csv('bin_level_csi_results_sildemo.csv', index=False)


    # Function to create pivot table for a given segment and feature
    def create_bin_pivot(data, segment_type, segment_value, feature=None):
        # Filter by segment
        segment_data = data[(data['segment_type'] == segment_type) &
                        (data['segment_value'] == segment_value)]

        # Further filter by feature if specified
        if feature:
            segment_data = segment_data[segment_data['feature'] == feature]

        # Create pivot table - rows are bin values, columns are months
        pivot = segment_data.pivot_table(
            index=['feature', 'bin_value'],
            columns=['MonthSortKey'] if 'MonthSortKey' in segment_data.columns else ['Month'],
            values='bin_csi',
            aggfunc='first'
        )

        return pivot

    # Create bin pivot tables for overall and by segments
    unique_segment_combos = combined_results[['segment_type', 'segment_value']].drop_duplicates()
    unique_features = combined_results['feature'].unique()

    # Question - do we need this Excel file?
    # Create Excel writer to save all pivots in one file
    # with pd.ExcelWriter('bin_level_csi_pivots_sildemo.xlsx') as writer:
    #     # First, create overall pivot with all features and bins
    #     overall_pivot = create_bin_pivot(combined_results, 'Overall', 'All')
    #     overall_pivot.to_excel(writer, sheet_name='Overall_All_Features')


    #     # Create separate pivot for each feature (across all segments)
    #     for feature in unique_features:
    #         # Create pivot for this feature - Overall segment
    #         feature_pivot = create_bin_pivot(combined_results, 'Overall', 'All', feature)

    #         # Make sheet name Excel-friendly (31 char limit, no special chars)
    #         sheet_name = f"Overall_{feature[-20:]}"
    #         sheet_name = sheet_name.replace("/", "_").replace("\\", "_")[:31]

    #         feature_pivot.to_excel(writer, sheet_name=sheet_name)
    #         print(f"Created pivot for feature: {feature}")

    #     # Create separate pivot for each segment and feature combination
    #     for _, segment_row in unique_segment_combos.iterrows():
    #         segment_type = segment_row['segment_type']
    #         segment_value = segment_row['segment_value']

    #         # Skip Overall segment as we already handled it
    #         if segment_type == 'Overall' and segment_value == 'All':
    #             continue

    #         # Create segment-specific pivots for each feature
    #         for feature in unique_features:
    #             # Filter data for this segment and feature
    #             segment_feature_data = combined_results[
    #                 (combined_results['segment_type'] == segment_type) &
    #                 (combined_results['segment_value'] == segment_value) &
    #                 (combined_results['feature'] == feature)
    #             ]

    #             # Skip if no data
    #             if segment_feature_data.empty:
    #                 continue

    #             # Create pivot
    #             pivot = segment_feature_data.pivot_table(
    #                 index=['bin_value'],
    #                 columns=['MonthSortKey'] if 'MonthSortKey' in segment_feature_data.columns else ['Month'],
    #                 values='bin_csi',
    #                 aggfunc='first'
    #             )

    #             # Make sheet name Excel-friendly
    #             segment_name = f"{segment_type}_{segment_value}"
    #             feature_name = feature[-10:]  # Use last 10 chars of feature name to keep sheet name short
    #             sheet_name = f"{segment_name}_{feature_name}"
    #             sheet_name = sheet_name.replace("/", "_").replace("\\", "_")[:31]

    #             pivot.to_excel(writer, sheet_name=sheet_name)
    #             print(f"Created pivot for {segment_type}={segment_value}, feature={feature}")



    # Calculate bin contribution to total csi
    summary_data = []

    for segment_type in combined_results['segment_type'].unique():
        for segment_value in combined_results[combined_results['segment_type'] == segment_type]['segment_value'].unique():
            for feature in combined_results['feature'].unique():
                # Get data for this segment and feature
                segment_feature_data = combined_results[
                    (combined_results['segment_type'] == segment_type) &
                    (combined_results['segment_value'] == segment_value) &
                    (combined_results['feature'] == feature)
                ]

                if segment_feature_data.empty:
                    continue

                # Get unique months
                months = segment_feature_data['Month'].unique()

                for month in months:
                    month_data = segment_feature_data[segment_feature_data['Month'] == month]

                    # Get feature csi (should be same for all bins in this feature/month/segment)
                    feature_csi = month_data['feature_csi'].iloc[0] if not month_data.empty else 0

                    # Get top contributing bins
                    if not month_data.empty and 'bin_csi' in month_data.columns:
                        # Sort by absolute bin_csi value to get top contributors
                        top_bins = month_data.sort_values('bin_csi', key=abs, ascending=False)

                        # Take top 3 bins
                        for i, (_, bin_row) in enumerate(top_bins.iterrows()):
                            if i >= 3:  # Limit to top 3
                                break

                            bin_value = bin_row['bin_value']
                            bin_csi = bin_row['bin_csi']

                            # Calculate contribution percentage
                            pct_contribution = (bin_csi / feature_csi * 100) if feature_csi != 0 else 0

                            summary_data.append({
                                'segment_type': segment_type,
                                'segment_value': segment_value,
                                'feature': feature,
                                'Month': month,
                                'feature_csi': feature_csi,
                                'bin_value': bin_value,
                                'bin_csi': bin_csi,
                                'pct_contribution': pct_contribution,
                                'rank': i + 1
                            })

    # Create summary DataFrame
    if summary_data:
        summary_df = pd.DataFrame(summary_data)

        # Pivot to get a table with top contributors
        contribution_pivot = summary_df.pivot_table(
            index=['segment_type', 'segment_value', 'feature', 'Month', 'feature_csi'],
            columns=['rank'],
            values=['bin_value', 'bin_csi', 'pct_contribution'],
            aggfunc='first'
        )

        # Question - do we need this Excel file?
        # Save to Excel
        # Changes
        # contribution_pivot.to_excel('bin_contribution_summary_sildemo.xlsx')
    else:
        print("No data available for bin contribution summary")

    combined_results[['Month','MonthSortKey']].value_counts(dropna=False)

    combined_results['Month'] = combined_results['Month'].replace('2024-06', '2023-07-2024-06')
    combined_results['MonthSortKey'] = combined_results['MonthSortKey'].fillna('a_2023-07-2024-06')
    combined_results['Month'] = combined_results['Month'].apply(lambda x: x.split(' 00:00:00')[0] if'00:00:00' in x else x)
    combined_results['scorename'] = 'beta_demo_score'
    combined_results['Modelname'] = 'SIL Beta Demo'
    combined_results['Description'] = 'Train period from 2023-07 to 2024-06'
    combined_results

    dataset_id = 'dap_ds_poweruser_playground'
    table_id = 'F_CSI_MODEL_FEATURES_BIN_TAB_new' # Changes
    # Define the table schema as per your DataFrame columns
    schema = [
        bigquery.SchemaField("Month", "string"),
        bigquery.SchemaField("feature", "string"),
        bigquery.SchemaField("bin_value", "string"),
        bigquery.SchemaField("DateCategory", "string"),
        bigquery.SchemaField("train_pct", "float64"),
        bigquery.SchemaField("test_pct", "float64"),
        bigquery.SchemaField("bin_csi", "float64"),
        bigquery.SchemaField("feature_csi", "float64"),
        bigquery.SchemaField("account_count", "int64"),
        bigquery.SchemaField("MonthSortKey", "string"),
        bigquery.SchemaField("segment_type", "string"),
        bigquery.SchemaField("segment_value", "string"),
        bigquery.SchemaField("scorename", "string"),
        bigquery.SchemaField("Modelname", "string"),
        bigquery.SchemaField("Description", "string"),
        ]
    # Create the dataset reference
    dataset_ref = client.dataset(dataset_id)
    # Define the table reference
    table_ref = dataset_ref.table(table_id)
    # Configure the job to overwrite the table if it already exists
    job_config = bigquery.LoadJobConfig(schema = schema)
    # Load the DataFrame into BigQuery
    job = client.load_table_from_dataframe(combined_results, table_ref, job_config=job_config)
    # Wait for the job to complete
    job.result()
    print(f"Table {table_id} created in dataset {dataset_id}.")

    """# App Score"""

    sq = """
    WITH appscore AS (
    SELECT
        digitalLoanAccountId,
        FORMAT_DATE('%Y-%m', ln_appln_submit_datetime) AS Application_month,
        FORMAT_DATE('%F', DATE_TRUNC(ln_appln_submit_datetime, WEEK(MONDAY))) AS Appl_week_start_date,
        EXTRACT(WEEK(MONDAY) FROM ln_appln_submit_datetime) AS Appl_week_number,
        ln_user_type,
        ln_loan_type,
        ln_prod_type,
        ln_os_type,
        beta_apps_score AS apps_score,
        CASE
        WHEN DATE_TRUNC(ln_appln_submit_datetime, DAY) BETWEEN '2023-12-01' AND '2024-06-30' THEN 'Train'
        WHEN DATE_TRUNC(ln_appln_submit_datetime, DAY) >= '2024-07-01' THEN 'Test'
        ELSE 'Other'
        END AS dataselection,
        CASE
        WHEN app_first_competitors_install_to_apply_days IS NULL THEN 'k. missing'
        WHEN CAST(app_first_competitors_install_to_apply_days AS FLOAT64) BETWEEN -2.001 AND 4.1 THEN 'a. -2.001-4.1'
        WHEN CAST(app_first_competitors_install_to_apply_days AS FLOAT64) BETWEEN 4.2 AND 40.1 THEN 'b. 4.2-40.1'
        WHEN CAST(app_first_competitors_install_to_apply_days AS FLOAT64) BETWEEN 40.2 AND 88.4 THEN 'c. 40.2-88.4'
        WHEN CAST(app_first_competitors_install_to_apply_days AS FLOAT64) BETWEEN 88.5 AND 143.2 THEN 'd. 88.5-143.2'
        WHEN CAST(app_first_competitors_install_to_apply_days AS FLOAT64) BETWEEN 143.3 AND 206.7 THEN 'e. 143.3-206.7'
        WHEN CAST(app_first_competitors_install_to_apply_days AS FLOAT64) BETWEEN 206.8 AND 288.3 THEN 'f. 206.8-288.2'
        WHEN CAST(app_first_competitors_install_to_apply_days AS FLOAT64) BETWEEN 288.4 AND 391.9 THEN 'g. 288.4-391.9'
        WHEN CAST(app_first_competitors_install_to_apply_days AS FLOAT64) BETWEEN 392.0 AND 547.3 THEN 'h. 392.0-547.3'
        WHEN CAST(app_first_competitors_install_to_apply_days AS FLOAT64) BETWEEN 547.4 AND 826.1 THEN 'i. 547.4-826.1'
        WHEN CAST(app_first_competitors_install_to_apply_days AS FLOAT64) BETWEEN 826.2 AND 5242.2 THEN 'j. 826.2-5242.2'
        WHEN CAST(app_first_competitors_install_to_apply_days AS FLOAT64) > 5242.2 THEN 'l. >5242.2'
        END AS app_first_competitors_install_to_apply_days_bin,
        CASE
        WHEN app_median_time_bw_installed_mins_30d IS NULL THEN 'l. missing'
        WHEN CAST(app_median_time_bw_installed_mins_30d AS FLOAT64) BETWEEN -0.001 AND 96.022 THEN 'a. -0.001-96.022'
        WHEN CAST(app_median_time_bw_installed_mins_30d AS FLOAT64) BETWEEN 96.023 AND 1166.377 THEN 'b. 96.023-1166.377'
        WHEN CAST(app_median_time_bw_installed_mins_30d AS FLOAT64) BETWEEN 1166.378 AND 2259.803 THEN 'c. 1166.378-2259.803'
        WHEN CAST(app_median_time_bw_installed_mins_30d AS FLOAT64) BETWEEN 2259.804 AND 3532.3 THEN 'd. 2259.804-3532.3'
        WHEN CAST(app_median_time_bw_installed_mins_30d AS FLOAT64) BETWEEN 3532.4 AND 5067.042 THEN 'e. 3532.4-5067.042'
        WHEN CAST(app_median_time_bw_installed_mins_30d AS FLOAT64) BETWEEN 5067.043 AND 7065.507 THEN 'f. 5067.043-7065.507'
        WHEN CAST(app_median_time_bw_installed_mins_30d AS FLOAT64) BETWEEN 7065.508 AND 9891.612 THEN 'g. 7065.508-9891.612'
        WHEN CAST(app_median_time_bw_installed_mins_30d AS FLOAT64) BETWEEN 9891.613 AND 14384.46 THEN 'h. 9891.613-14384.46'
        WHEN CAST(app_median_time_bw_installed_mins_30d AS FLOAT64) BETWEEN 14384.461 AND 20358.378 THEN 'j. 14384.461-20358.378'
        WHEN CAST(app_median_time_bw_installed_mins_30d AS FLOAT64) BETWEEN 20358.379 AND 112663145.6 THEN 'k. 20358.379-112663145.6'
        ELSE 'm. NA'
        END AS app_median_time_bw_installed_mins_30d_bin,
        CASE
        WHEN app_cnt_absence_tag_90d IS NULL THEN 'g. missing'
        WHEN CAST(app_cnt_absence_tag_90d AS FLOAT64) BETWEEN -0.001 AND 1.0 THEN 'a. (-0.001, 1.0]'
        WHEN CAST(app_cnt_absence_tag_90d AS FLOAT64) BETWEEN 1.001 AND 2.0 THEN 'b. (1.0, 2.0]'
        WHEN CAST(app_cnt_absence_tag_90d AS FLOAT64) BETWEEN 2.001 AND 3.0 THEN 'c. (2.0, 3.0]'
        WHEN CAST(app_cnt_absence_tag_90d AS FLOAT64) BETWEEN 3.001 AND 4.0 THEN 'd. (3.0, 4.0]'
        WHEN CAST(app_cnt_absence_tag_90d AS FLOAT64) BETWEEN 4.001 AND 7.0 THEN 'e. (4.0, 7.0]'
        WHEN CAST(app_cnt_absence_tag_90d AS FLOAT64) BETWEEN 7.001 AND 154.0 THEN 'f. (7.0, 154.0]'
        ELSE 'h. NA'
        END AS app_cnt_absence_tag_90d_bin,
        CASE
        WHEN app_cnt_finance_90d IS NULL THEN 'f. missing'
        WHEN CAST(app_cnt_finance_90d AS FLOAT64) BETWEEN -0.001 AND 1.0 THEN 'a. (-0.001, 1.0]'
        WHEN CAST(app_cnt_finance_90d AS FLOAT64) BETWEEN 1.001 AND 2.0 THEN 'b. (1.0, 2.0]'
        WHEN CAST(app_cnt_finance_90d AS FLOAT64) BETWEEN 2.001 AND 3.0 THEN 'c. (2.0, 3.0]'
        WHEN CAST(app_cnt_finance_90d AS FLOAT64) BETWEEN 3.001 AND 4.0 THEN 'd. (3.0, 4.0]'
        WHEN CAST(app_cnt_finance_90d AS FLOAT64) BETWEEN 4.001 AND 30.0 THEN 'e. (4.0, 30.0]'
        ELSE 'g. NA'
        END AS app_cnt_finance_90d_bin,
        CASE
        WHEN app_first_payday_install_to_apply_days IS NULL THEN 'h. missing'
        WHEN CAST(app_first_payday_install_to_apply_days AS FLOAT64) BETWEEN -1.001 AND 0.0 THEN 'a. (-1.001, 0.0]'
        WHEN CAST(app_first_payday_install_to_apply_days AS FLOAT64) BETWEEN 0.001 AND 0.1 THEN 'b. (0.0, 0.1]'
        WHEN CAST(app_first_payday_install_to_apply_days AS FLOAT64) BETWEEN 0.101 AND 42.1 THEN 'c. (0.1, 42.1]'
        WHEN CAST(app_first_payday_install_to_apply_days AS FLOAT64) BETWEEN 42.101 AND 138.8 THEN 'd. (42.1, 138.8]'
        WHEN CAST(app_first_payday_install_to_apply_days AS FLOAT64) BETWEEN 138.801 AND 273.54 THEN 'e. (138.8, 273.54]'
        WHEN CAST(app_first_payday_install_to_apply_days AS FLOAT64) BETWEEN 273.541 AND 532.2 THEN 'f. (273.54, 532.2]'
        WHEN CAST(app_first_payday_install_to_apply_days AS FLOAT64) BETWEEN 532.201 AND 5242.2 THEN 'g. (532.2, 5242.2]'
        ELSE 'i. NA'  -- Add an else clause to handle values outside the defined ranges
        END AS app_first_payday_install_to_apply_days_bin
    FROM `prj-prod-dataplatform.risk_credit_mis.application_score_master`
    WHERE DATE_TRUNC(ln_appln_submit_datetime, DAY) >= '2023-12-01'
    )
    SELECT * FROM appscore;

    """

    appscoredf = client.query(sq).to_dataframe(progress_bar_type='tqdm')

    def calculate_categorical_csi(train_dist, test_dist):
        """
        Calculate csi for categorical features.

        Args:
            train_dist: Distribution of categories in training set
            test_dist: Distribution of categories in test set

        Returns:
            float: csi value
        """
        # Ensure both distributions have the same categories
        all_categories = set(train_dist.index) | set(test_dist.index)

        # Align distributions
        train_dist_aligned = train_dist.reindex(all_categories, fill_value=0.0001)  # Small value to avoid division by zero
        test_dist_aligned = test_dist.reindex(all_categories, fill_value=0.0001)

        # Calculate csi
        csi_values = (test_dist_aligned - train_dist_aligned) * np.log(test_dist_aligned / train_dist_aligned)
        return csi_values.sum()

    def calculate_bin_csi(train_df, test_df, feature):
        """
        Calculate csi for each bin value within a feature.

        Args:
            train_df: Training DataFrame
            test_df: Test DataFrame
            feature: Feature name to calculate bin-level csi for

        Returns:
            DataFrame: csi results for each bin value
        """
        # Get all unique bin values across both datasets
        all_bins = set(train_df[feature].dropna().unique()) | set(test_df[feature].dropna().unique())

        # Results list for bin-level csi
        bin_csi_results = []

        # Calculate distribution for the entire feature in training set (for reference)
        train_counts = train_df[feature].value_counts(dropna=True)
        train_distribution = train_counts / train_counts.sum()

        # Calculate distribution for the entire feature in test set (for reference)
        test_counts = test_df[feature].value_counts(dropna=True)
        test_distribution = test_counts / test_counts.sum()

        # Calculate overall csi for the feature
        overall_csi = calculate_categorical_csi(train_distribution, test_distribution)

        # Calculate csi for each bin value
        for bin_value in all_bins:
            # Calculate percentage of this bin in train set
            train_bin_count = train_df[train_df[feature] == bin_value].shape[0]
            train_total = train_df.shape[0]
            train_bin_pct = train_bin_count / train_total if train_total > 0 else 0.0001

            # Calculate percentage of this bin in test set
            test_bin_count = test_df[test_df[feature] == bin_value].shape[0]
            test_total = test_df.shape[0]
            test_bin_pct = test_bin_count / test_total if test_total > 0 else 0.0001

            # Calculate csi for this bin
            if train_bin_pct < 0.0001:
                train_bin_pct = 0.0001  # Avoid division by zero
            if test_bin_pct < 0.0001:
                test_bin_pct = 0.0001  # Avoid division by zero

            bin_csi = (test_bin_pct - train_bin_pct) * np.log(test_bin_pct / train_bin_pct)

            # Store result
            bin_csi_results.append({
                'feature': feature,
                'bin_value': bin_value,
                'train_pct': train_bin_pct,
                'test_pct': test_bin_pct,
                'bin_csi': bin_csi,
                'feature_csi': overall_csi
            })

        return pd.DataFrame(bin_csi_results)

    def calculate_segmented_bin_csi(df, feature_list, segment_columns=None):
        """
        Calculate csi for each bin value within multiple features, overall and by segments.

        Args:
            df: DataFrame containing the data
            feature_list: List of feature names to calculate csi for
            segment_columns: List of columns to segment by (e.g., ['ln_user_type', 'ln_os_type'])

        Returns:
            DataFrame: csi results for each bin value by month and segment
        """
        # Initialize results list
        all_results = []

        # If no segment columns are provided, use an empty list
        if segment_columns is None:
            segment_columns = []

        # First, calculate overall csi for each bin
        overall_results = calculate_feature_bin_csi(df, feature_list)
        overall_results['segment_type'] = 'Overall'
        overall_results['segment_value'] = 'All'
        all_results.append(overall_results)

        # Then calculate csi for each segment column
        for segment_col in segment_columns:
            if segment_col not in df.columns:
                print(f"Warning: {segment_col} not found in DataFrame. Skipping.")
                continue

            # Get unique segment values
            segment_values = df[segment_col].dropna().unique()

            for segment_val in segment_values:
                # Filter data for this segment
                segment_df = df[df[segment_col] == segment_val]

                # Skip if not enough data
                if len(segment_df) < 50:  # Arbitrary threshold
                    print(f"Skipping {segment_col}={segment_val} due to insufficient data ({len(segment_df)} rows).")
                    continue

                # Calculate csi for this segment
                try:
                    segment_results = calculate_feature_bin_csi(segment_df, feature_list)
                    segment_results['segment_type'] = segment_col
                    segment_results['segment_value'] = segment_val
                    all_results.append(segment_results)
                except Exception as e:
                    print(f"Error calculating csi for {segment_col}={segment_val}: {e}")

        # Combine all results
        if all_results:
            combined_results = pd.concat(all_results, ignore_index=True)
            return combined_results
        else:
            return pd.DataFrame()

    def calculate_feature_bin_csi(df, feature_list):
        """
        Calculate csi for each bin value within multiple features.

        Args:
            df: DataFrame containing the data
            feature_list: List of feature names to calculate csi for

        Returns:
            DataFrame: csi results for each bin value by month
        """
        # Make a copy to avoid modifying the original DataFrame
        df_copy = df.copy()

        # Separate train and test data
        train_df = df_copy[df_copy['dataselection'] == 'Train']
        test_df = df_copy[df_copy['dataselection'] == 'Test']

        # Skip if either dataset is empty
        if train_df.empty or test_df.empty:
            print("Warning: Either train or test dataset is empty. Skipping csi calculation.")
            return pd.DataFrame()

        # Handle Application_month based on its type
        if isinstance(df_copy['Application_month'].iloc[0], str):
            # If it's a string in format 'YYYY-MM-DD', extract just 'YYYY-MM'
            last_train_month_str = str(train_df['Application_month'].max())
            if len(last_train_month_str) >= 7:  # Ensure we have at least YYYY-MM
                last_train_month_str = last_train_month_str[:7]  # Extract YYYY-MM part
        else:
            # If it's already a datetime object
            try:
                last_train_month = pd.to_datetime(train_df['Application_month'].max())
                last_train_month_str = last_train_month.strftime('%Y-%m')
            except:
                # Fallback if conversion fails
                last_train_month_str = str(train_df['Application_month'].max())

        # Store all bin-level csi results
        all_bin_results = []

        # Calculate distribution for each feature in the training set
        for feature in feature_list:
            if feature not in train_df.columns:
                print(f"Warning: Feature {feature} not found in training data. Skipping.")
                continue

            # Calculate bin-level csi for the training set against itself (always 0)
            train_bins = train_df[feature].dropna().unique()
            for bin_value in train_bins:
                all_bin_results.append({
                    'Month': last_train_month_str,
                    'feature': feature,
                    'bin_value': bin_value,
                    'DateCategory': 'a_Training',
                    'train_pct': (train_df[feature] == bin_value).mean(),
                    'test_pct': (train_df[feature] == bin_value).mean(),  # Same as train for training data
                    'bin_csi': 0.0,  # csi against itself is 0
                    'feature_csi': 0.0,  # Overall csi against itself is 0
                    'account_count': train_df['digitalLoanAccountId'].nunique()
                })

        # Get unique months from test set and sort them
        test_months = sorted(test_df['Application_month'].unique())

        # Create mapping of months to prefixed labels (b, c, d, etc.)
        prefix_map = {}
        for i, month in enumerate(test_months):
            month_str = str(month)
            if isinstance(month, str) and len(month_str) >= 7:
                month_str = month_str[:7]  # Extract YYYY-MM part

            # Use letters b, c, d, etc. for subsequent months (a is reserved for Training)
            prefix = chr(98 + i)  # ASCII: b=98, c=99, etc.
            prefix_map[month] = f"{prefix}_{month_str}"

        # Calculate monthly csi for each feature and bin in the test set
        for month in test_months:
            original_month_str = str(month)
            if isinstance(month, str) and len(original_month_str) >= 7:
                original_month_str = original_month_str[:7]  # Extract YYYY-MM part

            # Use the prefixed month string for sorting
            month_str = prefix_map[month]

            month_df = test_df[test_df['Application_month'] == month]

            if not month_df.empty:
                month_accounts = month_df['digitalLoanAccountId'].nunique()

                for feature in feature_list:
                    if feature not in month_df.columns:
                        continue

                    # Calculate bin-level csi for this feature in this month
                    try:
                        # Get all unique bin values for this feature across train and test
                        all_bins = set(train_df[feature].dropna().unique()) | set(month_df[feature].dropna().unique())

                        # Calculate overall feature csi for reference
                        train_counts = train_df[feature].value_counts(dropna=True, normalize=True)
                        test_counts = month_df[feature].value_counts(dropna=True, normalize=True)
                        overall_csi = calculate_categorical_csi(train_counts, test_counts)

                        # Calculate csi for each bin
                        for bin_value in all_bins:
                            # Calculate percentages
                            train_pct = (train_df[feature] == bin_value).mean()
                            test_pct = (month_df[feature] == bin_value).mean()

                            # Add small value to avoid division by zero
                            if train_pct < 0.0001:
                                train_pct = 0.0001
                            if test_pct < 0.0001:
                                test_pct = 0.0001

                            # Calculate csi for this bin
                            bin_csi = (test_pct - train_pct) * np.log(test_pct / train_pct)

                            # Store result
                            all_bin_results.append({
                                'Month': original_month_str,
                                'MonthSortKey': month_str,
                                'feature': feature,
                                'bin_value': bin_value,
                                'DateCategory': 'b_Monthly',
                                'train_pct': train_pct,
                                'test_pct': test_pct,
                                'bin_csi': bin_csi,
                                'feature_csi': overall_csi,
                                'account_count': month_accounts
                            })
                    except Exception as e:
                        print(f"Error calculating bin csi for {feature} in {month}: {e}")

        # Create the output DataFrame
        return pd.DataFrame(all_bin_results)

    # Features list
    feature_list = [
    'app_first_competitors_install_to_apply_days_bin',
        'app_median_time_bw_installed_mins_30d_bin',
        'app_cnt_absence_tag_90d_bin',
        'app_cnt_finance_90d_bin',
        'app_first_payday_install_to_apply_days_bin'
    ]

    # Define segment columns
    segment_columns = ['ln_user_type', 'ln_prod_type', 'ln_os_type']

    # Calculate bin-level csi for overall and by segments
    bin_results = calculate_segmented_bin_csi(appscoredf, feature_list, segment_columns)



    # Try to combine with s_apps_score results if they exist
    try:
        # First ensure the s_apps_score_output_df has the same structure
        if 'MonthSortKey' not in s_apps_score_output_df.columns:
            s_apps_score_output_df['MonthSortKey'] = s_apps_score_output_df['Month']
            # Update DateCategory with prefix
            s_apps_score_output_df['DateCategory'] = s_apps_score_output_df['DateCategory'].apply(
                lambda x: 'a_Training' if x == 'Training' else 'b_Monthly'
            )

        # Add segment info to s_apps_score_output_df
        s_apps_score_output_df['segment_type'] = 'Overall'
        s_apps_score_output_df['segment_value'] = 'All'

        # Add bin_value column to s_apps_score_output_df (as 'All' for feature-level csi)
        s_apps_score_output_df['bin_value'] = 'All'

        # Rename csivalues to feature_csi for consistency
        if 'csivalues' in s_apps_score_output_df.columns:
            s_apps_score_output_df = s_apps_score_output_df.rename(columns={'csivalues': 'feature_csi'})

        # Add bin_csi column (same as feature_csi for feature-level csi)
        if 'feature_csi' in s_apps_score_output_df.columns:
            s_apps_score_output_df['bin_csi'] = s_apps_score_output_df['feature_csi']

        # Replace 'scorename' with 'feature' for consistency
        if 'scorename' in s_apps_score_output_df.columns:
            s_apps_score_output_df['feature'] = s_apps_score_output_df['feature'].fillna(s_apps_score_output_df['scorename'])
            s_apps_score_output_df = s_apps_score_output_df.drop('scorename', axis=1)

        # Combine with bin_results
        combined_results = pd.concat([s_apps_score_output_df, bin_results], ignore_index=True)
    except NameError:
        # If s_apps_score_output_df doesn't exist, just use bin_results
        combined_results = bin_results

    # Sort by segment_type, segment_value, feature, bin_value, and MonthSortKey
    sort_columns = ['segment_type', 'segment_value', 'feature', 'bin_value']
    if 'MonthSortKey' in combined_results.columns:
        sort_columns.append('MonthSortKey')
    else:
        sort_columns.append('Month')

    combined_results = combined_results.sort_values(sort_columns)

    # Question - do we need this CSV file?
    # Save the detailed bin-level results
    # combined_results.to_csv('bin_level_csi_results_appscore.csv', index=False)


    # Function to create pivot table for a given segment and feature - FIXED to avoid overlapping ranges
    def create_bin_pivot(data, segment_type, segment_value, feature=None):
        # Filter by segment
        segment_data = data[(data['segment_type'] == segment_type) &
                        (data['segment_value'] == segment_value)]

        # Further filter by feature if specified
        if feature:
            segment_data = segment_data[segment_data['feature'] == feature]

        # Create pivot table - rows are bin values, columns are months
        pivot = segment_data.pivot_table(
            index=['feature', 'bin_value'],
            columns=['MonthSortKey'] if 'MonthSortKey' in segment_data.columns else ['Month'],
            values='bin_csi',
            aggfunc='first'
        )

        return pivot

    # Create bin pivot tables for overall and by segments
    unique_segment_combos = combined_results[['segment_type', 'segment_value']].drop_duplicates()
    unique_features = combined_results['feature'].unique()

    # Question - do we need this Excel file?
    # Create Excel writer to save all pivots in one file - FIXED writing to avoid overlapping ranges
    # with pd.ExcelWriter('bin_level_csi_pivots_app_score.xlsx') as writer:
    #     # First, create overall pivot with all features and bins
    #     overall_pivot = create_bin_pivot(combined_results, 'Overall', 'All')
    #     # Add merge_cells=False to avoid overlapping range error
    #     overall_pivot.to_excel(writer, sheet_name='Overall_All_Features', merge_cells=False)
    #     print("Created overall pivot table for all features")

    #     # Create separate pivot for each feature (across all segments)
    #     for feature in unique_features:
    #         # Create pivot for this feature - Overall segment
    #         feature_pivot = create_bin_pivot(combined_results, 'Overall', 'All', feature)

    #         # Make sheet name Excel-friendly (31 char limit, no special chars)
    #         sheet_name = f"Overall_{feature[-20:]}"
    #         sheet_name = sheet_name.replace("/", "_").replace("\\", "_")[:31]

    #         # Add merge_cells=False to avoid overlapping range error
    #         feature_pivot.to_excel(writer, sheet_name=sheet_name, merge_cells=False)
    #         print(f"Created pivot for feature: {feature}")

    #     # Create separate pivot for each segment and feature combination
    #     for _, segment_row in unique_segment_combos.iterrows():
    #         segment_type = segment_row['segment_type']
    #         segment_value = segment_row['segment_value']

    #         # Skip Overall segment as we already handled it
    #         if segment_type == 'Overall' and segment_value == 'All':
    #             continue

    #         # Create segment-specific pivots for each feature
    #         for feature in unique_features:
    #             # Filter data for this segment and feature
    #             segment_feature_data = combined_results[
    #                 (combined_results['segment_type'] == segment_type) &
    #                 (combined_results['segment_value'] == segment_value) &
    #                 (combined_results['feature'] == feature)
    #             ]

    #             # Skip if no data
    #             if segment_feature_data.empty:
    #                 continue

    #             # Create pivot
    #             pivot = segment_feature_data.pivot_table(
    #                 index=['bin_value'],
    #                 columns=['MonthSortKey'] if 'MonthSortKey' in segment_feature_data.columns else ['Month'],
    #                 values='bin_csi',
    #                 aggfunc='first'
    #             )

    #             # Make sheet name Excel-friendly
    #             segment_name = f"{segment_type}_{segment_value}"
    #             feature_name = feature[-10:]  # Use last 10 chars of feature name to keep sheet name short
    #             sheet_name = f"{segment_name}_{feature_name}"
    #             sheet_name = sheet_name.replace("/", "_").replace("\\", "_")[:31]

    #             # Add merge_cells=False to avoid overlapping range error
    #             pivot.to_excel(writer, sheet_name=sheet_name, merge_cells=False)
    #             print(f"Created pivot for {segment_type}={segment_value}, feature={feature}")


    # Calculate bin contribution to total csi
    summary_data = []

    for segment_type in combined_results['segment_type'].unique():
        for segment_value in combined_results[combined_results['segment_type'] == segment_type]['segment_value'].unique():
            for feature in combined_results['feature'].unique():
                # Get data for this segment and feature
                segment_feature_data = combined_results[
                    (combined_results['segment_type'] == segment_type) &
                    (combined_results['segment_value'] == segment_value) &
                    (combined_results['feature'] == feature)
                ]

                if segment_feature_data.empty:
                    continue

                # Get unique months
                months = segment_feature_data['Month'].unique()

                for month in months:
                    month_data = segment_feature_data[segment_feature_data['Month'] == month]

                    # Get feature csi (should be same for all bins in this feature/month/segment)
                    feature_csi = month_data['feature_csi'].iloc[0] if not month_data.empty else 0

                    # Get top contributing bins
                    if not month_data.empty and 'bin_csi' in month_data.columns:
                        # Sort by absolute bin_csi value to get top contributors
                        top_bins = month_data.sort_values('bin_csi', key=abs, ascending=False)

                        # Take top 3 bins
                        for i, (_, bin_row) in enumerate(top_bins.iterrows()):
                            if i >= 3:  # Limit to top 3
                                break

                            bin_value = bin_row['bin_value']
                            bin_csi = bin_row['bin_csi']

                            # Calculate contribution percentage
                            pct_contribution = (bin_csi / feature_csi * 100) if feature_csi != 0 else 0

                            summary_data.append({
                                'segment_type': segment_type,
                                'segment_value': segment_value,
                                'feature': feature,
                                'Month': month,
                                'feature_csi': feature_csi,
                                'bin_value': bin_value,
                                'bin_csi': bin_csi,
                                'pct_contribution': pct_contribution,
                                'rank': i + 1
                            })

    # Create summary DataFrame
    if summary_data:
        summary_df = pd.DataFrame(summary_data)

        # Pivot to get a table with top contributors
        contribution_pivot = summary_df.pivot_table(
            index=['segment_type', 'segment_value', 'feature', 'Month', 'feature_csi'],
            columns=['rank'],
            values=['bin_value', 'bin_csi', 'pct_contribution'],
            aggfunc='first'
        )

    else:
        print("No data available for bin contribution summary")

    print("\nAnalysis complete!")

    combined_results['Month'] = combined_results['Month'].replace('2024-06', '2023-12-2024-06')
    combined_results['MonthSortKey'] = combined_results['MonthSortKey'].fillna('a_2023-12-2024-06')
    combined_results['Month'] = combined_results['Month'].apply(lambda x: x.split(' 00:00:00')[0] if'00:00:00' in x else x)
    combined_results['scorename'] = 'apps_score'
    combined_results['Modelname'] = 'Android_SIL_Apps_Score'
    combined_results['Description'] = 'Train period from 2023-12 to 2024-06'
    combined_results

    dataset_id = 'dap_ds_poweruser_playground'
    table_id = 'F_CSI_MODEL_FEATURES_BIN_TAB_new' # Changes
    # Define the table schema as per your DataFrame columns
    schema = [
        bigquery.SchemaField("Month", "string"),
        bigquery.SchemaField("feature", "string"),
        bigquery.SchemaField("bin_value", "string"),
        bigquery.SchemaField("DateCategory", "string"),
        bigquery.SchemaField("train_pct", "float64"),
        bigquery.SchemaField("test_pct", "float64"),
        bigquery.SchemaField("bin_csi", "float64"),
        bigquery.SchemaField("feature_csi", "float64"),
        bigquery.SchemaField("account_count", "int64"),
        bigquery.SchemaField("MonthSortKey", "string"),
        bigquery.SchemaField("segment_type", "string"),
        bigquery.SchemaField("segment_value", "string"),
        bigquery.SchemaField("scorename", "string"),
        bigquery.SchemaField("Modelname", "string"),
        bigquery.SchemaField("Description", "string"),
        ]
    # Create the dataset reference
    dataset_ref = client.dataset(dataset_id)
    # Define the table reference
    table_ref = dataset_ref.table(table_id)
    # Configure the job to overwrite the table if it already exists
    job_config = bigquery.LoadJobConfig(schema = schema)
    # Load the DataFrame into BigQuery
    job = client.load_table_from_dataframe(combined_results, table_ref, job_config=job_config)
    # Wait for the job to complete
    job.result()
    print(f"Table {table_id} created in dataset {dataset_id}.")

