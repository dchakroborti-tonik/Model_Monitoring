

def psi_monitoring():
    # Import Libraries
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from google.cloud import bigquery
    from datetime import datetime
    import os

    client = bigquery.Client(project='prj-prod-dataplatform')

    a = "`prj-prod-dataplatform.risk_credit_mis.application_score_master`"

    sq = f"""
    with base as 
    (select 
    a.digitalLoanAccountId, 
    FORMAT_DATE('%Y-%m', a.ln_appln_submit_datetime) Application_month,
    FORMAT_DATE('%F', DATE_TRUNC(a.ln_appln_submit_datetime, WEEK(MONDAY))) as Appl_week_start_date,
    EXTRACT(WEEK(MONDAY) FROM a.ln_appln_submit_datetime) as Appl_week_number,
    a.ln_loan_type,
    case when date_trunc(a.ln_appln_submit_datetime, day) between '2023-07-01' and '2024-06-30' then 'Train'
        when date_trunc(a.ln_appln_submit_datetime, day) >= '2024-08-01' then 'Test'
        Else 'Other' end dataselection,
    a.beta_apps_score s_apps_score,
    from
    {a} a
    where a.ln_loan_applied_flag = 1 and ln_dl_rule_reject_flag = 0
    and date_trunc(a.ln_appln_submit_datetime, day) >= '2023-04-01'
    )
    select * from base where dataselection in ('Train', 'Test');"""

    
    df = client.query(sq).to_dataframe(progress_bar_type='tqdm')
    print(f"The shape of df before duplicate drop is:\t {df.shape}")

    df = df.drop_duplicates(keep='first')

    print(f"The shape of df after duplicate drop is:\t {df.shape}")


    # Assuming your DataFrame is called 'df' and has the structure from the image
    # Replace this with your actual DataFrame loading process
    sq = f"""
    with base as 
    (select 
    a.digitalLoanAccountId, 
    FORMAT_DATE('%Y-%m', a.ln_appln_submit_datetime) Application_month,
    FORMAT_DATE('%F', DATE_TRUNC(a.ln_appln_submit_datetime, WEEK(MONDAY))) as Appl_week_start_date,
    EXTRACT(WEEK(MONDAY) FROM a.ln_appln_submit_datetime) as Appl_week_number,
    a.ln_loan_type,
    case when date_trunc(a.ln_appln_submit_datetime, day) between '2023-07-01' and '2024-06-30' then 'Train'
        when date_trunc(a.ln_appln_submit_datetime, day) >= '2024-07-01' then 'Test'
        Else 'Other' end dataselection,
    a.beta_apps_score s_apps_score,
    from 
    {a} a
    where a.ln_loan_applied_flag = 1 and ln_dl_rule_reject_flag = 0
    and date_trunc(a.ln_appln_submit_datetime, day) >= '2023-04-01'
    )
    select * from base where dataselection in ('Train', 'Test') and s_apps_score is not null;"""

    print(df.groupby(['dataselection'])['digitalLoanAccountId'].nunique())
    
    df = client.query(sq).to_dataframe(progress_bar_type='tqdm')

    # Covert s_apps_score to numeric if it's not already
    df['s_apps_score'] = pd.to_numeric(df['s_apps_score'], errors='coerce')

    # Convert Application_month to datetime if it's not already
    if df['Application_month'].dtype != 'datetime64[ns]':
        df['Application_month'] = pd.to_datetime(df['Application_month'] + '-01')

    # Separate train and test data
    train_df = df[df['dataselection'] == 'Train']
    test_df = df[df['dataselection'] == 'Test']

    # Calculate decile bins for the entire training set
    train_deciles = pd.qcut(train_df['s_apps_score'], 10, labels=False, retbins=True)
    train_decile_bins = train_deciles[1]

    # Function to calculate PSI using the pre-defined decile bins
    def calculate_psi_with_bins(data_scores, decile_bins):
        """Calculates PSI using pre-defined decile bins."""
        data_deciles = pd.cut(data_scores, bins=decile_bins, labels=False, include_lowest=True)
        distribution = pd.Series(data_deciles).value_counts().sort_index() / len(data_scores)

        # Align with training distribution
        all_bins = range(10)  # Assuming 10 deciles
        distribution_aligned = distribution.reindex(all_bins, fill_value=0)
        train_dist_aligned = pd.Series(train_deciles[0]).value_counts().sort_index() / len(train_df['s_apps_score'])
        train_dist_aligned = train_dist_aligned.reindex(all_bins, fill_value=0)

        psi_values = (distribution_aligned - train_dist_aligned) * np.log(distribution_aligned / train_dist_aligned)
        return psi_values.sum()

    # Calculate PSI for the entire training set
    train_psi = calculate_psi_with_bins(train_df['s_apps_score'], train_decile_bins)

    # Get the last month of the training set
    last_train_month = train_df['Application_month'].max()
    last_train_month_str = last_train_month.strftime('%Y-%m')

    # Your existing query and initial dataframe setup remains the same

    # Calculate monthly PSI for the test set
    monthly_psi_results = []

    # Add the train set PSI to the results (with the correct last month)
    monthly_psi_results.append({
        'Month': last_train_month_str,  # Use the last month of the training set
        'scorename': 's_apps_score',
        'DateCategory': 'Training',
        'psivalues': train_psi,
        'account_count': train_df['digitalLoanAccountId'].nunique()  # Add distinct account count
    })

    # Calculate monthly PSI for the test set
    for month in sorted(test_df['Application_month'].unique()):
        print(f"Type of month: {type(month)}")
        # month_str = month.strftime('%Y-%m')
        month_str = pd.to_datetime(month).strftime('%Y-%m')
        month_df = test_df[test_df['Application_month'] == month]
        if not month_df.empty:
            month_psi = calculate_psi_with_bins(month_df['s_apps_score'], train_decile_bins)
            monthly_psi_results.append({
                'Month': month_str,
                'scorename': 's_apps_score',
                'DateCategory': 'Monthly',
                'psivalues': month_psi,
                'account_count': month_df['digitalLoanAccountId'].nunique()  # Add distinct account count
            })

    # Create the output DataFrame
    s_apps_score_output_df = pd.DataFrame(monthly_psi_results)


    s_apps_score_output_df.rename(columns={'psivalues':'s_apps_score_psivalues'}, inplace = True)
    s_apps_score_output_df


    # sb_demo_score
    sq = f"""
    with base as 
    (select 
    a.digitalLoanAccountId, 
    FORMAT_DATE('%Y-%m', a.ln_appln_submit_datetime) Application_month,
    FORMAT_DATE('%F', DATE_TRUNC(a.ln_appln_submit_datetime, WEEK(MONDAY))) as Appl_week_start_date,
    EXTRACT(WEEK(MONDAY) FROM a.ln_appln_submit_datetime) as Appl_week_number,
    a.ln_loan_type,
    case when date_trunc(a.ln_appln_submit_datetime, day) between '2023-07-01' and '2024-06-30' then 'Train'
        when date_trunc(a.ln_appln_submit_datetime, day) >= '2024-07-01' then 'Test'
        Else 'Other' end dataselection,
    a.beta_demo_score sb_demo_score,
    from {a} a
    where a.ln_loan_applied_flag = 1 and ln_dl_rule_reject_flag = 0
    and date_trunc(a.ln_appln_submit_datetime, day) >= '2023-04-01'
    )
    select * from base where sb_demo_score is not null and dataselection in ('Train', 'Test');"""
    
    df = client.query(sq).to_dataframe(progress_bar_type='tqdm')


    # Replace this with your actual DataFrame loading process
    sq = """
    with base as 
    (select 
    a.digitalLoanAccountId, 
    FORMAT_DATE('%Y-%m', a.ln_appln_submit_datetime) Application_month,
    FORMAT_DATE('%F', DATE_TRUNC(a.ln_appln_submit_datetime, WEEK(MONDAY))) as Appl_week_start_date,
    EXTRACT(WEEK(MONDAY) FROM a.ln_appln_submit_datetime) as Appl_week_number,
    a.ln_loan_type,
    case when date_trunc(a.ln_appln_submit_datetime, day) between '2023-07-01' and '2024-06-30' then 'Train'
        when date_trunc(a.ln_appln_submit_datetime, day) >= '2024-07-01' then 'Test'
        Else 'Other' end dataselection,
    a.beta_demo_score sb_demo_score,
    from  risk_mart.sil_risk_ds_master_20230101_20250309 a
    where a.ln_loan_applied_flag = 1
    and date_trunc(a.ln_appln_submit_datetime, day) >= '2023-04-01'
    )
    select * from base where sb_demo_score is not null and dataselection in ('Train', 'Test');"""
    df = client.query(sq).to_dataframe(progress_bar_type='tqdm')

    # Covert sb_demo_score to numeric if it's not already
    df['sb_demo_score'] = pd.to_numeric(df['sb_demo_score'], errors='coerce')

    print(df.groupby(['dataselection'])['digitalLoanAccountId'].nunique())

    # Convert Application_month to datetime if it's not already
    if df['Application_month'].dtype != 'datetime64[ns]':
        df['Application_month'] = pd.to_datetime(df['Application_month'] + '-01')

    # Separate train and test data
    train_df = df[df['dataselection'] == 'Train']
    test_df = df[df['dataselection'] == 'Test']

    # Calculate decile bins for the entire training set
    train_deciles = pd.qcut(train_df['sb_demo_score'], 10, labels=False, retbins=True)
    train_decile_bins = train_deciles[1]
    print(train_decile_bins)
    # Function to calculate PSI using the pre-defined decile bins
    def calculate_psi_with_bins(data_scores, decile_bins):
        """Calculates PSI using pre-defined decile bins."""
        data_deciles = pd.cut(data_scores, bins=decile_bins, labels=False, include_lowest=True)
        distribution = pd.Series(data_deciles).value_counts().sort_index() / len(data_scores)

        # Align with training distribution
        all_bins = range(10)  # Assuming 10 deciles
        distribution_aligned = distribution.reindex(all_bins, fill_value=0)
        train_dist_aligned = pd.Series(train_deciles[0]).value_counts().sort_index() / len(train_df['sb_demo_score'])
        train_dist_aligned = train_dist_aligned.reindex(all_bins, fill_value=0)

        psi_values = (distribution_aligned - train_dist_aligned) * np.log(distribution_aligned / train_dist_aligned)
        return psi_values.sum()

    # Calculate PSI for the entire training set
    train_psi = calculate_psi_with_bins(train_df['sb_demo_score'], train_decile_bins)

    # Get the last month of the training set
    last_train_month = train_df['Application_month'].max()
    last_train_month_str = last_train_month.strftime('%Y-%m')

    # Calculate monthly PSI for the test set
    monthly_psi_results = []

    # Add the train set PSI to the results (with the correct last month)
    monthly_psi_results.append({
        'Month': last_train_month_str,  # Use the last month of the training set
        # 'loan_type': train_df['ln_loan_type'].iloc[0],  # Assuming loan_type is consistent in the train set
        'scorename': 'sb_demo_score',
        'DateCategory': 'Training',
        'psivalues': train_psi,
        'account_count': train_df['digitalLoanAccountId'].nunique()  # Add distinct account count
    })

    # Calculate monthly PSI for the test set
    for month in sorted(test_df['Application_month'].unique()):
        # month_str = month.strftime('%Y-%m')
        month_str = pd.to_datetime(month).strftime('%Y-%m')
        month_df = test_df[test_df['Application_month'] == month]
        if not month_df.empty:
            month_psi = calculate_psi_with_bins(month_df['sb_demo_score'], train_decile_bins)
            monthly_psi_results.append({
                'Month': month_str,
                # 'loan_type': month_df['ln_loan_type'].iloc[0],  # Assuming loan_type is consistent in the month
                'scorename': 'sb_demo_score',
                'DateCategory': 'Monthly',
                'psivalues': month_psi,
                'account_count': month_df['digitalLoanAccountId'].nunique()  # Add distinct account count
            })

    # Create the output DataFrame
    sb_demo_score_output_df = pd.DataFrame(monthly_psi_results)
    sb_demo_score_output_df.rename(columns={'psivalues':'sb_demo_score_psivalues'}, inplace = True)
    sb_demo_score_output_df


    # s_cic_score
    # Replace this with your actual DataFrame loading process
    sq = f"""
    with base as 
    (select 
    a.digitalLoanAccountId, 
    FORMAT_DATE('%Y-%m', a.ln_appln_submit_datetime) Application_month,
    FORMAT_DATE('%F', DATE_TRUNC(a.ln_appln_submit_datetime, WEEK(MONDAY))) as Appl_week_start_date,
    EXTRACT(WEEK(MONDAY) FROM a.ln_appln_submit_datetime) as Appl_week_number,
    a.ln_loan_type,
    case when date_trunc(a.ln_appln_submit_datetime, day) between '2023-07-01' and '2024-06-30' then 'Train'
        when date_trunc(a.ln_appln_submit_datetime, day) >= '2024-07-01' then 'Test'
        Else 'Other' end dataselection,
    a.cic_score s_cic_score,
    from  {a} a
    where a.ln_loan_applied_flag = 1 and ln_dl_rule_reject_flag = 0
    and date_trunc(a.ln_appln_submit_datetime, day) >= '2023-04-01'
    )
    select * from base 
    where s_cic_score is not null and  dataselection in ('Train', 'Test');"""
    df = client.query(sq).to_dataframe(progress_bar_type='tqdm')

    print(df.groupby(['dataselection'])['digitalLoanAccountId'].nunique())

    # Covert s_cic_score to numeric if it's not already
    df['s_cic_score'] = pd.to_numeric(df['s_cic_score'], errors='coerce')

    # Convert Application_month to datetime if it's not already
    if df['Application_month'].dtype != 'datetime64[ns]':
        df['Application_month'] = pd.to_datetime(df['Application_month'] + '-01')

    # Separate train and test data
    train_df = df[df['dataselection'] == 'Train']
    test_df = df[df['dataselection'] == 'Test']

    # Calculate decile bins for the entire training set
    train_deciles = pd.qcut(train_df['s_cic_score'], 10, labels=False, retbins=True)
    train_decile_bins = train_deciles[1]
    print(train_decile_bins)
    # Function to calculate PSI using the pre-defined decile bins
    def calculate_psi_with_bins(data_scores, decile_bins):
        """Calculates PSI using pre-defined decile bins."""
        data_deciles = pd.cut(data_scores, bins=decile_bins, labels=False, include_lowest=True)
        distribution = pd.Series(data_deciles).value_counts().sort_index() / len(data_scores)
        
        # Align with training distribution
        all_bins = range(10)  # Assuming 10 deciles
        distribution_aligned = distribution.reindex(all_bins, fill_value=0)
        train_dist_aligned = pd.Series(train_deciles[0]).value_counts().sort_index() / len(train_df['s_cic_score'])
        train_dist_aligned = train_dist_aligned.reindex(all_bins, fill_value=0)
        
        # print(f"distribution_aligned-{distribution_aligned}")
        # print(f"train_dist_aligned - {train_dist_aligned}")

        psi_values = (distribution_aligned - train_dist_aligned) * np.log(distribution_aligned / train_dist_aligned)
        return psi_values.sum()

    # Calculate PSI for the entire training set
    train_psi = calculate_psi_with_bins(train_df['s_cic_score'], train_decile_bins)

    # Get the last month of the training set
    last_train_month = train_df['Application_month'].max()
    last_train_month_str = last_train_month.strftime('%Y-%m')

    # Calculate monthly PSI for the test set
    monthly_psi_results = []

    # Add the train set PSI to the results (with the correct last month)
    monthly_psi_results.append({
        'Month': last_train_month_str,  # Use the last month of the training set
        # 'loan_type': train_df['ln_loan_type'].iloc[0],  # Assuming loan_type is consistent in the train set
        'scorename': 's_cic_score',
        'DateCategory': 'Training',
        'psivalues': train_psi,
        'account_count': train_df['digitalLoanAccountId'].nunique()  # Add distinct account count
    })

    # Calculate monthly PSI for the test set
    for month in sorted(test_df['Application_month'].unique()):
        # month_str = month.strftime('%Y-%m')
        month_str = pd.to_datetime(month).strftime('%Y-%m')
        month_df = test_df[test_df['Application_month'] == month]
        if not month_df.empty:
            month_psi = calculate_psi_with_bins(month_df['s_cic_score'], train_decile_bins)
            monthly_psi_results.append({
                'Month': month_str,
                # 'loan_type': month_df['ln_loan_type'].iloc[0],  # Assuming loan_type is consistent in the month
                'scorename': 's_cic_score',
                'DateCategory': 'Monthly',
                'psivalues': month_psi,
                'account_count': month_df['digitalLoanAccountId'].nunique()  # Add distinct account count
            })

    # Create the output DataFrame
    s_cic_score_output_df = pd.DataFrame(monthly_psi_results)
    s_cic_score_output_df.rename(columns={'psivalues':'s_cic_score_psivalues'}, inplace = True)
    s_cic_score_output_df


    # checking CIC PSI with period testing from Jan 2025 to Feb 2, 2025
    # Replace this with your actual DataFrame loading process
    sq = f"""
    with base as 
    (select 
    a.digitalLoanAccountId, 
    FORMAT_DATE('%Y-%m', a.ln_appln_submit_datetime) Application_month,
    ---FORMAT_DATE('%F', DATE_TRUNC(a.ln_appln_submit_datetime, WEEK(MONDAY))) as Appl_week_start_date,
    ---- EXTRACT(WEEK(MONDAY) FROM a.ln_appln_submit_datetime) as Appl_week_number,
    ---a.ln_loan_type,
    case when date_trunc(a.ln_appln_submit_datetime, day) between '2023-07-01' and '2024-06-30' then 'Train'
        when date_trunc(a.ln_appln_submit_datetime, day) >= '2024-07-01' then 'Test'
        Else 'Other' end dataselection,
    a.cic_score s_cic_score,
    from {a}  a
    where a.ln_loan_applied_flag = 1 and ln_dl_rule_reject_flag = 0
    and date_trunc(a.ln_appln_submit_datetime, day) >= '2023-07-01'
    )
    select * from base where dataselection in ('Train', 'Test') and s_cic_score is not null;"""
    df = client.query(sq).to_dataframe(progress_bar_type='tqdm')

    # convert s_cic_score to numeric if it's not already
    df['s_cic_score'] = pd.to_numeric(df['s_cic_score'], errors='coerce')

    # Convert Application_month to datetime if it's not already
    if df['Application_month'].dtype != 'datetime64[ns]':
        df['Application_month'] = pd.to_datetime(df['Application_month'] + '-01')

    # Separate train and test data
    train_df = df[df['dataselection'] == 'Train']
    test_df = df[df['dataselection'] == 'Test']

    # Function to calculate PSI between two periods
    def calculate_psi(expected_array, actual_array, bins=10):
        """
        Calculate PSI for two arrays
        
        Parameters:
        -----------
        expected_array : numpy array of expected/training values
        actual_array : numpy array of actual/test values
        bins : number of bins to create
        
        Returns:
        --------
        psi_value : float, the calculated PSI value
        bin_details : DataFrame with binning details
        """
        # Create bins based on the expected array
        quantiles = np.linspace(0, 1, bins+1)
        bin_edges = np.quantile(expected_array, quantiles)
        
        # Ensure bin edges are unique (handle duplicates if they exist)
        bin_edges = np.unique(bin_edges)
        if len(bin_edges) < bins + 1:
            # Add small increments to duplicate values
            temp_edges = np.sort(np.unique(expected_array))
            if len(temp_edges) >= bins + 1:
                bin_edges = np.quantile(temp_edges, quantiles)
            else:
                # If not enough unique values, use min-max range divided into bins
                bin_edges = np.linspace(min(expected_array), max(expected_array), bins+1)
        
        # Create bins for both arrays
        expected_counts, _ = np.histogram(expected_array, bins=bin_edges)
        actual_counts, _ = np.histogram(actual_array, bins=bin_edges)
        
        # Calculate percentages
        expected_percents = expected_counts / len(expected_array) * 100
        actual_percents = actual_counts / len(actual_array) * 100
        
        # Calculate differences and PSI components
        diff = actual_percents - expected_percents
        
        # Safe division and log calculation (avoiding div by zero)
        ratio = np.divide(actual_percents, expected_percents, 
                        out=np.ones_like(actual_percents), 
                        where=expected_percents!=0)
        ln_ratio = np.log(ratio, out=np.zeros_like(ratio), where=ratio>0)
        
        # Calculate PSI components and total
        psi_components = diff / 100 * ln_ratio
        psi_value = np.sum(psi_components)
        
        # Create detailed results DataFrame
        bin_details = pd.DataFrame({
            'Bins': [f"{i+1}" for i in range(len(expected_counts))],
            '# Train': expected_counts,
            '# Train %': expected_percents,
            '# Test': actual_counts,
            '# Test %': actual_percents,
            'A-B': diff,
            'ln(A/B)': ln_ratio,
            'PSI': psi_components * 100
        })
        
        bin_details.loc['Grand Total'] = [
            '', sum(expected_counts), 100.0, sum(actual_counts), 100.0, '', '', psi_value * 100
        ]
        
        return psi_value, bin_details

    # Calculate monthly PSI as in your original code
    def calculate_monthly_psi():
        # Calculate decile bins for the entire training set
        train_deciles = pd.qcut(train_df['s_cic_score'], 10, labels=False, retbins=True)
        train_decile_bins = train_deciles[1]
        
        # Get the last month of the training set
        last_train_month = train_df['Application_month'].max()
        last_train_month_str = last_train_month.strftime('%Y-%m')
        
        # Calculate monthly PSI for the test set
        monthly_psi_results = []
        
        # Add the train set PSI to the results (with the correct last month)
        monthly_psi_results.append({
            'Month': last_train_month_str,
            'scorename': 's_cic_score',
            'DateCategory': 'Training',
            'psivalues': 0.0  # PSI against itself is 0
        })
        
        # Calculate monthly PSI for the test set
        for month in sorted(test_df['Application_month'].unique()):
            # month_str = month.strftime('%Y-%m')
            month_str = pd.to_datetime(month).strftime('%Y-%m')
            month_df = test_df[test_df['Application_month'] == month]
            
            if not month_df.empty:
                # Calculate PSI using our function
                month_psi, _ = calculate_psi(train_df['s_cic_score'].values, month_df['s_cic_score'].values)
                
                monthly_psi_results.append({
                    'Month': month_str,
                    'scorename': 's_cic_score',
                    'DateCategory': 'Monthly',
                    'psivalues': month_psi
                })
        
        # Create the output DataFrame
        monthly_psi_df = pd.DataFrame(monthly_psi_results)
        monthly_psi_df.rename(columns={'psivalues': 's_cic_score_psivalues'}, inplace=True)
        
        return monthly_psi_df

    # Question - hardcoded dates
    # Calculate PSI between two specific periods (as shown in the image)
    def calculate_period_psi():
        # Define the periods matching the image
        train_period = train_df  # Already defined as 2023-07 to 2024-06
        
        # Filter test data for Jan-Feb 2025 - using datetime objects to avoid the February 29 issue
        jan_2025 = pd.Timestamp('2025-01-01')
        feb_2025 = pd.Timestamp('2025-02-28')  # Using Feb 28 instead of Feb 29
        
        test_period = test_df[(test_df['Application_month'] >= jan_2025) & 
                            (test_df['Application_month'] <= feb_2025)]
        
        # Calculate PSI between periods
        period_psi, psi_details = calculate_psi(train_period['s_cic_score'].values, 
                                            test_period['s_cic_score'].values,
                                            bins=10)
        
        print("PSI between 2023-07 to 2024-06 and 2025-01 to 2025-02:")
        print(f"Overall PSI: {period_psi:.6f}")
        
        return period_psi, psi_details

    # Run both calculations
    print("Calculating monthly PSI values...")
    monthly_psi_results = calculate_monthly_psi()
    print(monthly_psi_results)

    print("\nCalculating period PSI (matching the image)...")
    period_psi, psi_details = calculate_period_psi()
    print("\nDetailed PSI calculation by bin:")
    print(psi_details)

    # Question - do we need this block of code?
    sq = f"""
    with base as 
    (select 
    a.digitalLoanAccountId, 
    FORMAT_DATE('%Y-%m', a.ln_appln_submit_datetime) Application_month,
    case when date_trunc(a.ln_appln_submit_datetime, day) between '2023-07-01' and '2024-06-30' then 'Train'
        when date_trunc(a.ln_appln_submit_datetime, day) >= '2024-07-01' then 'Test'
        Else 'Other' end dataselection,
    a.cic_score s_cic_score,
    from {a}  a
    where a.ln_loan_applied_flag = 1 and ln_dl_rule_reject_flag = 0
    and date_trunc(a.ln_appln_submit_datetime, day) >= '2023-07-01'
    )
    select * from base where s_cic_score is not null and dataselection in ('Train', 'Test');"""
    df = client.query(sq).to_dataframe(progress_bar_type='tqdm')

    df.groupby(['Application_month', 'dataselection'])['digitalLoanAccountId'].nunique()

    # df.to_csv("Test.csv") Changes


    # sb_stack_score
    # Assuming your DataFrame is called 'df' and has the structure from the image
    # Replace this with your actual DataFrame loading process
    sq = f"""
    with base as 
    (select 
    a.digitalLoanAccountId, 
    FORMAT_DATE('%Y-%m', a.ln_appln_submit_datetime) Application_month,
    FORMAT_DATE('%F', DATE_TRUNC(a.ln_appln_submit_datetime, WEEK(MONDAY))) as Appl_week_start_date,
    EXTRACT(WEEK(MONDAY) FROM a.ln_appln_submit_datetime) as Appl_week_number,
    a.ln_loan_type,
    case when date_trunc(a.ln_appln_submit_datetime, day) between '2023-07-01' and '2024-06-30' then 'Train'
        when date_trunc(a.ln_appln_submit_datetime, day) >= '2024-07-01' then 'Test'
        Else 'Other' end dataselection,
    a.beta_stack_score sb_stack_score,
    from {a} a
    where a.ln_loan_applied_flag = 1 and ln_dl_rule_reject_flag = 0
    and date_trunc(a.ln_appln_submit_datetime, day) >= '2023-04-01'
    )
    select * from base where sb_stack_score is not null and dataselection in ('Train', 'Test');"""
    df = client.query(sq).to_dataframe(progress_bar_type='tqdm')
    print(df.groupby(['dataselection'])['digitalLoanAccountId'].nunique())

    # convert sb_stack_score to numeric if it's not already
    df['sb_stack_score'] = pd.to_numeric(df['sb_stack_score'], errors='coerce')

    # Convert Application_month to datetime if it's not already
    if df['Application_month'].dtype != 'datetime64[ns]':
        df['Application_month'] = pd.to_datetime(df['Application_month'] + '-01')

    # Separate train and test data
    train_df = df[df['dataselection'] == 'Train']
    test_df = df[df['dataselection'] == 'Test']

    # Calculate decile bins for the entire training set
    train_deciles = pd.qcut(train_df['sb_stack_score'], 10, labels=False, retbins=True)
    train_decile_bins = train_deciles[1]
    print(train_decile_bins)
    # Function to calculate PSI using the pre-defined decile bins
    def calculate_psi_with_bins(data_scores, decile_bins):
        """Calculates PSI using pre-defined decile bins."""
        data_deciles = pd.cut(data_scores, bins=decile_bins, labels=False, include_lowest=True)
        distribution = pd.Series(data_deciles).value_counts().sort_index() / len(data_scores)

        # Align with training distribution
        all_bins = range(10)  # Assuming 10 deciles
        distribution_aligned = distribution.reindex(all_bins, fill_value=0)
        train_dist_aligned = pd.Series(train_deciles[0]).value_counts().sort_index() / len(train_df['sb_stack_score'])
        train_dist_aligned = train_dist_aligned.reindex(all_bins, fill_value=0)

        psi_values = (distribution_aligned - train_dist_aligned) * np.log(distribution_aligned / train_dist_aligned)
        return psi_values.sum()

    # Calculate PSI for the entire training set
    train_psi = calculate_psi_with_bins(train_df['sb_stack_score'], train_decile_bins)

    # Get the last month of the training set
    last_train_month = train_df['Application_month'].max()
    last_train_month_str = last_train_month.strftime('%Y-%m')

    # Calculate monthly PSI for the test set
    monthly_psi_results = []

    # Add the train set PSI to the results (with the correct last month)
    monthly_psi_results.append({
        'Month': last_train_month_str,  # Use the last month of the training set
        # 'loan_type': train_df['ln_loan_type'].iloc[0],  # Assuming loan_type is consistent in the train set
        'scorename': 'sb_stack_score',
        'DateCategory': 'Training',
        'psivalues': train_psi,
        'account_count': train_df['digitalLoanAccountId'].nunique()  # Add distinct account count
        
    })

    # Calculate monthly PSI for the test set
    for month in sorted(test_df['Application_month'].unique()):
        # month_str = month.strftime('%Y-%m')
        month_str = pd.to_datetime(month).strftime('%Y-%m')
        month_df = test_df[test_df['Application_month'] == month]
        if not month_df.empty:
            month_psi = calculate_psi_with_bins(month_df['sb_stack_score'], train_decile_bins)
            monthly_psi_results.append({
                'Month': month_str,
                # 'loan_type': month_df['ln_loan_type'].iloc[0],  # Assuming loan_type is consistent in the month
                'scorename': 'sb_stack_score',
                'DateCategory': 'Monthly',
                'psivalues': month_psi,
                'account_count': month_df['digitalLoanAccountId'].nunique()  # Add distinct account count
            })

    # Create the output DataFrame
    sb_stack_score_output_df = pd.DataFrame(monthly_psi_results)
    sb_stack_score_output_df.rename(columns={'psivalues':'sb_stack_score_psivalues'}, inplace = True)
    sb_stack_score_output_df


    # sa_stack_score
    # Assuming your DataFrame is called 'df' and has the structure from the image
    # Replace this with your actual DataFrame loading process
    sq = f"""
    with base as 
    (select 
    a.digitalLoanAccountId, 
    FORMAT_DATE('%Y-%m', a.ln_appln_submit_datetime) Application_month,
    FORMAT_DATE('%F', DATE_TRUNC(a.ln_appln_submit_datetime, WEEK(MONDAY))) as Appl_week_start_date,
    EXTRACT(WEEK(MONDAY) FROM a.ln_appln_submit_datetime) as Appl_week_number,
    a.ln_loan_type,
    case when date_trunc(a.ln_appln_submit_datetime, day) between '2023-07-01' and '2024-06-30' then 'Train'
        when date_trunc(a.ln_appln_submit_datetime, day) >= '2024-07-01' then 'Test'
        Else 'Other' end dataselection,
    a.alpha_stack_score sa_stack_score,
    from {a}  a
    where a.ln_loan_applied_flag = 1  and ln_dl_rule_reject_flag = 0
    and date_trunc(a.ln_appln_submit_datetime, day) >= '2023-04-01'
    )
    select * from base where sa_stack_score is not null and dataselection in ('Train', 'Test');"""
    df = client.query(sq).to_dataframe(progress_bar_type='tqdm')
    print(df.groupby(['dataselection'])['digitalLoanAccountId'].nunique())
    # Covert sa_stack_score to numeric if it's not already
    df['sa_stack_score'] = pd.to_numeric(df['sa_stack_score'], errors='coerce')

    # Convert Application_month to datetime if it's not already
    if df['Application_month'].dtype != 'datetime64[ns]':
        df['Application_month'] = pd.to_datetime(df['Application_month'] + '-01')

    # Separate train and test data
    train_df = df[df['dataselection'] == 'Train']
    test_df = df[df['dataselection'] == 'Test']

    # Calculate decile bins for the entire training set
    train_deciles = pd.qcut(train_df['sa_stack_score'], 10, labels=False, retbins=True)
    train_decile_bins = train_deciles[1]
    print(train_decile_bins)
    # Function to calculate PSI using the pre-defined decile bins
    def calculate_psi_with_bins(data_scores, decile_bins):
        """Calculates PSI using pre-defined decile bins."""
        data_deciles = pd.cut(data_scores, bins=decile_bins, labels=False, include_lowest=True)
        distribution = pd.Series(data_deciles).value_counts().sort_index() / len(data_scores)

        # Align with training distribution
        all_bins = range(10)  # Assuming 10 deciles
        distribution_aligned = distribution.reindex(all_bins, fill_value=0)
        train_dist_aligned = pd.Series(train_deciles[0]).value_counts().sort_index() / len(train_df['sa_stack_score'])
        train_dist_aligned = train_dist_aligned.reindex(all_bins, fill_value=0)

        psi_values = (distribution_aligned - train_dist_aligned) * np.log(distribution_aligned / train_dist_aligned)
        return psi_values.sum()

    # Calculate PSI for the entire training set
    train_psi = calculate_psi_with_bins(train_df['sa_stack_score'], train_decile_bins)

    # Get the last month of the training set
    last_train_month = train_df['Application_month'].max()
    last_train_month_str = last_train_month.strftime('%Y-%m')

    # Calculate monthly PSI for the test set
    monthly_psi_results = []

    # Add the train set PSI to the results (with the correct last month)
    monthly_psi_results.append({
        'Month': last_train_month_str,  # Use the last month of the training set
        # 'loan_type': train_df['ln_loan_type'].iloc[0],  # Assuming loan_type is consistent in the train set
        'scorename': 'sa_stack_score',
        'DateCategory': 'Training',
        'psivalues': train_psi,
        'account_count': train_df['digitalLoanAccountId'].nunique()  # Add distinct account count
    })

    # Calculate monthly PSI for the test set
    for month in sorted(test_df['Application_month'].unique()):
        # month_str = month.strftime('%Y-%m')
        month_str = pd.to_datetime(month).strftime('%Y-%m')
        month_df = test_df[test_df['Application_month'] == month]
        if not month_df.empty:
            month_psi = calculate_psi_with_bins(month_df['sa_stack_score'], train_decile_bins)
            monthly_psi_results.append({
                'Month': month_str,
                # 'loan_type': month_df['ln_loan_type'].iloc[0],  # Assuming loan_type is consistent in the month
                'scorename': 'sa_stack_score',
                'DateCategory': 'Monthly',
                'psivalues': month_psi,
                'account_count': month_df['digitalLoanAccountId'].nunique()  # Add distinct account count
            })

    # Create the output DataFrame
    sa_stack_score_output_df = pd.DataFrame(monthly_psi_results)
    sa_stack_score_output_df.rename(columns={'psivalues':'sa_stack_score_psivalues'}, inplace = True)
    sa_stack_score_output_df


    # c_credo_score_output_df
    # Assuming your DataFrame is called 'df' and has the structure from the image
    # Replace this with your actual DataFrame loading process
    sq = f"""
    with base as 
    (select 
    a.digitalLoanAccountId, 
    FORMAT_DATE('%Y-%m', a.ln_appln_submit_datetime) Application_month,
    FORMAT_DATE('%F', DATE_TRUNC(a.ln_appln_submit_datetime, WEEK(MONDAY))) as Appl_week_start_date,
    EXTRACT(WEEK(MONDAY) FROM a.ln_appln_submit_datetime) as Appl_week_number,
    a.ln_loan_type,
    case when date_trunc(a.ln_appln_submit_datetime, day) between '2023-07-01' and '2024-06-30' then 'Train'
        when date_trunc(a.ln_appln_submit_datetime, day) >= '2024-07-01' then 'Test'
        Else 'Other' end dataselection,
    a.credo_quick_score c_credo_score,
    from {a}  a
    where a.ln_loan_applied_flag = 1  and ln_dl_rule_reject_flag = 0
    and date_trunc(a.ln_appln_submit_datetime, day) >= '2023-07-01'
    )
    select * from base where c_credo_score is not null and dataselection in ('Train', 'Test');"""
    df = client.query(sq).to_dataframe(progress_bar_type='tqdm')
    print(df.groupby(['dataselection'])['digitalLoanAccountId'].nunique())

    # Convert c_credo_score to numeric if it's not already
    df['c_credo_score'] = pd.to_numeric(df['c_credo_score'], errors='coerce')

    # Convert Application_month to datetime if it's not already
    if df['Application_month'].dtype != 'datetime64[ns]':
        df['Application_month'] = pd.to_datetime(df['Application_month'] + '-01')

    # Separate train and test data
    train_df = df[df['dataselection'] == 'Train']
    test_df = df[df['dataselection'] == 'Test']

    # Calculate decile bins for the entire training set
    train_deciles = pd.qcut(train_df['c_credo_score'], 10, labels=False, retbins=True)
    train_decile_bins = train_deciles[1]
    print(train_decile_bins)
    # Function to calculate PSI using the pre-defined decile bins
    def calculate_psi_with_bins(data_scores, decile_bins):
        """Calculates PSI using pre-defined decile bins."""
        data_deciles = pd.cut(data_scores, bins=decile_bins, labels=False, include_lowest=True)
        distribution = pd.Series(data_deciles).value_counts().sort_index() / len(data_scores)

        # Align with training distribution
        all_bins = range(10)  # Assuming 10 deciles
        distribution_aligned = distribution.reindex(all_bins, fill_value=0)
        train_dist_aligned = pd.Series(train_deciles[0]).value_counts().sort_index() / len(train_df['c_credo_score'])
        train_dist_aligned = train_dist_aligned.reindex(all_bins, fill_value=0)

        psi_values = (distribution_aligned - train_dist_aligned) * np.log(distribution_aligned / train_dist_aligned)
        return psi_values.sum()

    # Calculate PSI for the entire training set
    train_psi = calculate_psi_with_bins(train_df['c_credo_score'], train_decile_bins)

    # Get the last month of the training set
    last_train_month = train_df['Application_month'].max()
    last_train_month_str = last_train_month.strftime('%Y-%m')

    # Calculate monthly PSI for the test set
    monthly_psi_results = []

    # Add the train set PSI to the results (with the correct last month)
    monthly_psi_results.append({
        'Month': last_train_month_str,  # Use the last month of the training set
        # 'loan_type': train_df['ln_loan_type'].iloc[0],  # Assuming loan_type is consistent in the train set
        'scorename': 'c_credo_score',
        'DateCategory': 'Training',
        'psivalues': train_psi,
        'account_count': train_df['digitalLoanAccountId'].nunique()  # Add distinct account count
    })

    # Calculate monthly PSI for the test set
    for month in sorted(test_df['Application_month'].unique()):
        # month_str = month.strftime('%Y-%m')
        month_str = pd.to_datetime(month).strftime('%Y-%m')
        month_df = test_df[test_df['Application_month'] == month]
        if not month_df.empty:
            month_psi = calculate_psi_with_bins(month_df['c_credo_score'], train_decile_bins)
            monthly_psi_results.append({
                'Month': month_str,
                # 'loan_type': month_df['ln_loan_type'].iloc[0],  # Assuming loan_type is consistent in the month
                'scorename': 'c_credo_score',
                'DateCategory': 'Monthly',
                'psivalues': month_psi,
                'account_count': month_df['digitalLoanAccountId'].nunique()  # Add distinct account count
            })

    # Create the output DataFrame
    c_credo_score_output_df = pd.DataFrame(monthly_psi_results)
    c_credo_score_output_df.rename(columns={'psivalues':'c_credo_score_psivalues'}, inplace = True)

    c_credo_score_output_df



    # s_credo_score
    # Assuming your DataFrame is called 'df' and has the structure from the image
    # Replace this with your actual DataFrame loading process
    sq = f"""
    with base as 
    (select 
    a.digitalLoanAccountId, 
    FORMAT_DATE('%Y-%m', a.ln_appln_submit_datetime) Application_month,
    FORMAT_DATE('%F', DATE_TRUNC(a.ln_appln_submit_datetime, WEEK(MONDAY))) as Appl_week_start_date,
    EXTRACT(WEEK(MONDAY) FROM a.ln_appln_submit_datetime) as Appl_week_number,
    a.ln_loan_type,
    case when date_trunc(a.ln_appln_submit_datetime, day) between '2023-07-01' and '2024-06-30' then 'Train'
        when date_trunc(a.ln_appln_submit_datetime, day) >= '2024-07-01' then 'Test'
        Else 'Other' end dataselection,
    a.credo_sil_score s_credo_score,
    from {a}  a
    where a.ln_loan_applied_flag = 1  and ln_dl_rule_reject_flag = 0
    and date_trunc(a.ln_appln_submit_datetime, day) >= '2023-07-01'
    )
    select * from base where s_credo_score is not null and dataselection in ('Train', 'Test');"""
    df = client.query(sq).to_dataframe(progress_bar_type='tqdm')
    print(df.groupby(['dataselection'])['digitalLoanAccountId'].nunique())

    # Convert s_credo_score to numeric if it's not already
    df['s_credo_score'] = pd.to_numeric(df['s_credo_score'], errors='coerce')

    # Convert Application_month to datetime if it's not already
    if df['Application_month'].dtype != 'datetime64[ns]':
        df['Application_month'] = pd.to_datetime(df['Application_month'] + '-01')

    # Separate train and test data
    train_df = df[df['dataselection'] == 'Train']
    test_df = df[df['dataselection'] == 'Test']

    # Calculate decile bins for the entire training set
    train_deciles = pd.qcut(train_df['s_credo_score'], 10, labels=False, retbins=True)
    train_decile_bins = train_deciles[1]
    print(train_decile_bins)
    # Function to calculate PSI using the pre-defined decile bins
    def calculate_psi_with_bins(data_scores, decile_bins):
        """Calculates PSI using pre-defined decile bins."""
        data_deciles = pd.cut(data_scores, bins=decile_bins, labels=False, include_lowest=True)
        distribution = pd.Series(data_deciles).value_counts().sort_index() / len(data_scores)

        # Align with training distribution
        all_bins = range(10)  # Assuming 10 deciles
        distribution_aligned = distribution.reindex(all_bins, fill_value=0)
        train_dist_aligned = pd.Series(train_deciles[0]).value_counts().sort_index() / len(train_df['s_credo_score'])
        train_dist_aligned = train_dist_aligned.reindex(all_bins, fill_value=0)

        psi_values = (distribution_aligned - train_dist_aligned) * np.log(distribution_aligned / train_dist_aligned)
        return psi_values.sum()

    # Calculate PSI for the entire training set
    train_psi = calculate_psi_with_bins(train_df['s_credo_score'], train_decile_bins)

    # Get the last month of the training set
    last_train_month = train_df['Application_month'].max()
    last_train_month_str = last_train_month.strftime('%Y-%m')

    # Calculate monthly PSI for the test set
    monthly_psi_results = []

    # Add the train set PSI to the results (with the correct last month)
    monthly_psi_results.append({
        'Month': last_train_month_str,  # Use the last month of the training set
        # 'loan_type': train_df['ln_loan_type'].iloc[0],  # Assuming loan_type is consistent in the train set
        'scorename': 's_credo_score',
        'DateCategory': 'Training',
        'psivalues': train_psi,
        'account_count': train_df['digitalLoanAccountId'].nunique()  # Add distinct account count
    })

    # Calculate monthly PSI for the test set
    for month in sorted(test_df['Application_month'].unique()):
        # month_str = month.strftime('%Y-%m')
        month_str = pd.to_datetime(month).strftime('%Y-%m')
        month_df = test_df[test_df['Application_month'] == month]
        if not month_df.empty:
            month_psi = calculate_psi_with_bins(month_df['s_credo_score'], train_decile_bins)
            monthly_psi_results.append({
                'Month': month_str,
                # 'loan_type': month_df['ln_loan_type'].iloc[0],  # Assuming loan_type is consistent in the month
                'scorename': 's_credo_score',
                'DateCategory': 'Monthly',
                'psivalues': month_psi,
                'account_count': month_df['digitalLoanAccountId'].nunique()  # Add distinct account count
            })

    # Create the output DataFrame
    s_credo_score_output_df = pd.DataFrame(monthly_psi_results)
    s_credo_score_output_df.rename(columns= {'psivalues':'s_credo_score_psivalues'}, inplace = True)
    s_credo_score_output_df


    # fu_credo_score
    # Assuming your DataFrame is called 'df' and has the structure from the image
    # Replace this with your actual DataFrame loading process
    sq = f"""
    with base as 
    (select 
    a.digitalLoanAccountId, 
    FORMAT_DATE('%Y-%m', a.ln_appln_submit_datetime) Application_month,
    FORMAT_DATE('%F', DATE_TRUNC(a.ln_appln_submit_datetime, WEEK(MONDAY))) as Appl_week_start_date,
    EXTRACT(WEEK(MONDAY) FROM a.ln_appln_submit_datetime) as Appl_week_number,
    a.ln_loan_type,
    case when date_trunc(a.ln_appln_submit_datetime, day) between '2023-07-01' and '2024-06-30' then 'Train'
        when date_trunc(a.ln_appln_submit_datetime, day) >= '2024-07-01' then 'Test'
        Else 'Other' end dataselection,
    a.credo_flex_score fu_credo_score,
    from {a} a
    where a.ln_loan_applied_flag = 1  and ln_dl_rule_reject_flag = 0
    and date_trunc(a.ln_appln_submit_datetime, day) >= '2023-07-01'
    )
    select * from base where fu_credo_score is not null and dataselection in ('Train', 'Test');"""
    df = client.query(sq).to_dataframe(progress_bar_type='tqdm')
    print(df.groupby(['dataselection'])['digitalLoanAccountId'].nunique())

    # Convert fu_credo_score to numeric if it's not already
    df['fu_credo_score'] = pd.to_numeric(df['fu_credo_score'], errors='coerce')

    # Convert Application_month to datetime if it's not already
    if df['Application_month'].dtype != 'datetime64[ns]':
        df['Application_month'] = pd.to_datetime(df['Application_month'] + '-01')

    # Separate train and test data
    train_df = df[df['dataselection'] == 'Train']
    test_df = df[df['dataselection'] == 'Test']

    # Calculate decile bins for the entire training set
    train_deciles = pd.qcut(train_df['fu_credo_score'], 10, labels=False, retbins=True)
    train_decile_bins = train_deciles[1]
    print(train_decile_bins)
    # Function to calculate PSI using the pre-defined decile bins
    def calculate_psi_with_bins(data_scores, decile_bins):
        """Calculates PSI using pre-defined decile bins."""
        data_deciles = pd.cut(data_scores, bins=decile_bins, labels=False, include_lowest=True)
        distribution = pd.Series(data_deciles).value_counts().sort_index() / len(data_scores)

        # Align with training distribution
        all_bins = range(10)  # Assuming 10 deciles
        distribution_aligned = distribution.reindex(all_bins, fill_value=0)
        train_dist_aligned = pd.Series(train_deciles[0]).value_counts().sort_index() / len(train_df['fu_credo_score'])
        train_dist_aligned = train_dist_aligned.reindex(all_bins, fill_value=0)

        psi_values = (distribution_aligned - train_dist_aligned) * np.log(distribution_aligned / train_dist_aligned)
        return psi_values.sum()

    # Calculate PSI for the entire training set
    train_psi = calculate_psi_with_bins(train_df['fu_credo_score'], train_decile_bins)

    # Get the last month of the training set
    last_train_month = train_df['Application_month'].max()
    last_train_month_str = last_train_month.strftime('%Y-%m')

    # Calculate monthly PSI for the test set
    monthly_psi_results = []

    # Add the train set PSI to the results (with the correct last month)
    monthly_psi_results.append({
        'Month': last_train_month_str,  # Use the last month of the training set
        # 'loan_type': train_df['ln_loan_type'].iloc[0],  # Assuming loan_type is consistent in the train set
        'scorename': 'fu_credo_score',
        'DateCategory': 'Training',
        'psivalues': train_psi,
        'account_count': train_df['digitalLoanAccountId'].nunique()  # Add distinct account count
    })

    # Calculate monthly PSI for the test set
    for month in sorted(test_df['Application_month'].unique()):
        # month_str = month.strftime('%Y-%m')
        month_str = pd.to_datetime(month).strftime('%Y-%m')
        month_df = test_df[test_df['Application_month'] == month]
        if not month_df.empty:
            month_psi = calculate_psi_with_bins(month_df['fu_credo_score'], train_decile_bins)
            monthly_psi_results.append({
                'Month': month_str,
                # 'loan_type': month_df['ln_loan_type'].iloc[0],  # Assuming loan_type is consistent in the month
                'scorename': 'fu_credo_score',
                'DateCategory': 'Monthly',
                'psivalues': month_psi,
                'account_count': month_df['digitalLoanAccountId'].nunique()  # Add distinct account count
            })

    # Create the output DataFrame
    fu_credo_score_output_df = pd.DataFrame(monthly_psi_results)
    fu_credo_score_output_df.rename(columns={'psivalues':'fu_credo_score_psivalues'}, inplace = True)
    fu_credo_score_output_df


    # r_credo_score
    # Assuming your DataFrame is called 'df' and has the structure from the image
    # Replace this with your actual DataFrame loading process
    sq = f"""
    with base as 
    (select 
    a.digitalLoanAccountId, 
    FORMAT_DATE('%Y-%m', a.ln_appln_submit_datetime) Application_month,
    FORMAT_DATE('%F', DATE_TRUNC(a.ln_appln_submit_datetime, WEEK(MONDAY))) as Appl_week_start_date,
    EXTRACT(WEEK(MONDAY) FROM a.ln_appln_submit_datetime) as Appl_week_number,
    a.ln_loan_type,
    case when date_trunc(a.ln_appln_submit_datetime, day) between '2023-07-01' and '2024-06-30' then 'Train'
        when date_trunc(a.ln_appln_submit_datetime, day) >= '2024-07-01' then 'Test'
        Else 'Other' end dataselection,
    a.credo_reloan_score r_credo_score,
    from  {a} a
    where a.ln_loan_applied_flag = 1  and ln_dl_rule_reject_flag = 0
    and date_trunc(a.ln_appln_submit_datetime, day) >= '2023-07-01'
    )
    select * from base where r_credo_score is not null and dataselection in ('Train', 'Test');"""
    df = client.query(sq).to_dataframe(progress_bar_type='tqdm')
    print(df.groupby(['dataselection'])['digitalLoanAccountId'].nunique())

    # Convert r_credo_score to numeric if it's not already
    df['r_credo_score'] = pd.to_numeric(df['r_credo_score'], errors='coerce')

    # Convert Application_month to datetime if it's not already
    if df['Application_month'].dtype != 'datetime64[ns]':
        df['Application_month'] = pd.to_datetime(df['Application_month'] + '-01')

    # Separate train and test data
    train_df = df[df['dataselection'] == 'Train']
    test_df = df[df['dataselection'] == 'Test']

    # Calculate decile bins for the entire training set
    train_deciles = pd.qcut(train_df['r_credo_score'], 10, labels=False, retbins=True)
    train_decile_bins = train_deciles[1]
    print(train_decile_bins)
    # Function to calculate PSI using the pre-defined decile bins
    def calculate_psi_with_bins(data_scores, decile_bins):
        """Calculates PSI using pre-defined decile bins."""
        data_deciles = pd.cut(data_scores, bins=decile_bins, labels=False, include_lowest=True)
        distribution = pd.Series(data_deciles).value_counts().sort_index() / len(data_scores)

        # Align with training distribution
        all_bins = range(10)  # Assuming 10 deciles
        distribution_aligned = distribution.reindex(all_bins, fill_value=0)
        train_dist_aligned = pd.Series(train_deciles[0]).value_counts().sort_index() / len(train_df['r_credo_score'])
        train_dist_aligned = train_dist_aligned.reindex(all_bins, fill_value=0)

        psi_values = (distribution_aligned - train_dist_aligned) * np.log(distribution_aligned / train_dist_aligned)
        return psi_values.sum()

    # Calculate PSI for the entire training set
    train_psi = calculate_psi_with_bins(train_df['r_credo_score'], train_decile_bins)

    # Get the last month of the training set
    last_train_month = train_df['Application_month'].max()
    last_train_month_str = last_train_month.strftime('%Y-%m')

    # Calculate monthly PSI for the test set
    monthly_psi_results = []

    # Add the train set PSI to the results (with the correct last month)
    monthly_psi_results.append({
        'Month': last_train_month_str,  # Use the last month of the training set
        # 'loan_type': train_df['ln_loan_type'].iloc[0],  # Assuming loan_type is consistent in the train set
        'scorename': 'r_credo_score',
        'DateCategory': 'Training',
        'psivalues': train_psi,
        'account_count': train_df['digitalLoanAccountId'].nunique()  # Add distinct account count
    })

    # Calculate monthly PSI for the test set
    for month in sorted(test_df['Application_month'].unique()):
        # month_str = month.strftime('%Y-%m')
        month_str = pd.to_datetime(month).strftime('%Y-%m')
        month_df = test_df[test_df['Application_month'] == month]
        if not month_df.empty:
            month_psi = calculate_psi_with_bins(month_df['r_credo_score'], train_decile_bins)
            monthly_psi_results.append({
                'Month': month_str,
                # 'loan_type': month_df['ln_loan_type'].iloc[0],  # Assuming loan_type is consistent in the month
                'scorename': 'r_credo_score',
                'DateCategory': 'Monthly',
                'psivalues': month_psi,
                'account_count': month_df['digitalLoanAccountId'].nunique()  # Add distinct account count
            })

    # Create the output DataFrame
    r_credo_score_output_df = pd.DataFrame(monthly_psi_results)
    r_credo_score_output_df.rename(columns={'psivalues':'r_credo_score_psivalues'}, inplace = True)
    r_credo_score_output_df


    # gen_credo_score
    # Assuming your DataFrame is called 'df' and has the structure from the image
    # Replace this with your actual DataFrame loading process
    sq = f"""
    with base as 
    (select 
    a.digitalLoanAccountId, 
    FORMAT_DATE('%Y-%m', a.ln_appln_submit_datetime) Application_month,
    FORMAT_DATE('%F', DATE_TRUNC(a.ln_appln_submit_datetime, WEEK(MONDAY))) as Appl_week_start_date,
    EXTRACT(WEEK(MONDAY) FROM a.ln_appln_submit_datetime) as Appl_week_number,
    a.ln_loan_type,
    case when date_trunc(a.ln_appln_submit_datetime, day) between '2023-07-01' and '2024-06-30' then 'Train'
        when date_trunc(a.ln_appln_submit_datetime, day) >= '2024-07-01' then 'Test'
        Else 'Other' end dataselection,
    a.credo_gen_score gen_credo_score,
    from {a}  a
    where a.ln_loan_applied_flag = 1  and ln_dl_rule_reject_flag = 0
    and date_trunc(a.ln_appln_submit_datetime, day) >= '2023-04-01'
    )
    select * from base where gen_credo_score is not null and dataselection in ('Train', 'Test');"""
    df = client.query(sq).to_dataframe(progress_bar_type='tqdm')
    print(df.groupby(['dataselection'])['digitalLoanAccountId'].nunique())

    # Convert gen_credo_score to numeric if it's not already
    df['gen_credo_score'] = pd.to_numeric(df['gen_credo_score'], errors='coerce')

    # Convert Application_month to datetime if it's not already
    if df['Application_month'].dtype != 'datetime64[ns]':
        df['Application_month'] = pd.to_datetime(df['Application_month'] + '-01')

    # Separate train and test data
    train_df = df[df['dataselection'] == 'Train']
    test_df = df[df['dataselection'] == 'Test']

    # Calculate decile bins for the entire training set
    train_deciles = pd.qcut(train_df['gen_credo_score'], 10, labels=False, retbins=True)
    train_decile_bins = train_deciles[1]
    print(train_decile_bins)
    # Function to calculate PSI using the pre-defined decile bins
    def calculate_psi_with_bins(data_scores, decile_bins):
        """Calculates PSI using pre-defined decile bins."""
        data_deciles = pd.cut(data_scores, bins=decile_bins, labels=False, include_lowest=True)
        distribution = pd.Series(data_deciles).value_counts().sort_index() / len(data_scores)

        # Align with training distribution
        all_bins = range(10)  # Assuming 10 deciles
        distribution_aligned = distribution.reindex(all_bins, fill_value=0)
        train_dist_aligned = pd.Series(train_deciles[0]).value_counts().sort_index() / len(train_df['gen_credo_score'])
        train_dist_aligned = train_dist_aligned.reindex(all_bins, fill_value=0)

        psi_values = (distribution_aligned - train_dist_aligned) * np.log(distribution_aligned / train_dist_aligned)
        return psi_values.sum()

    # Calculate PSI for the entire training set
    train_psi = calculate_psi_with_bins(train_df['gen_credo_score'], train_decile_bins)

    # Get the last month of the training set
    last_train_month = train_df['Application_month'].max()
    last_train_month_str = last_train_month.strftime('%Y-%m')

    # Calculate monthly PSI for the test set
    monthly_psi_results = []

    # Add the train set PSI to the results (with the correct last month)
    monthly_psi_results.append({
        'Month': last_train_month_str,  # Use the last month of the training set
        # 'loan_type': train_df['ln_loan_type'].iloc[0],  # Assuming loan_type is consistent in the train set
        'scorename': 'gen_credo_score',
        'DateCategory': 'Training',
        'psivalues': train_psi,
        'account_count': train_df['digitalLoanAccountId'].nunique()  # Add distinct account count
    })

    # Calculate monthly PSI for the test set
    for month in sorted(test_df['Application_month'].unique()):
        # month_str = month.strftime('%Y-%m')
        month_str = pd.to_datetime(month).strftime('%Y-%m')
        month_df = test_df[test_df['Application_month'] == month]
        if not month_df.empty:
            month_psi = calculate_psi_with_bins(month_df['gen_credo_score'], train_decile_bins)
            monthly_psi_results.append({
                'Month': month_str,
                # 'loan_type': month_df['ln_loan_type'].iloc[0],  # Assuming loan_type is consistent in the month
                'scorename': 'gen_credo_score',
                'DateCategory': 'Monthly',
                'psivalues': month_psi,
                'account_count': month_df['digitalLoanAccountId'].nunique()  # Add distinct account count
            })

    # Create the output DataFrame
    gen_credo_score_output_df = pd.DataFrame(monthly_psi_results)
    gen_credo_score_output_df.rename(columns={'psivalues':'gen_credo_score_psivalues'}, inplace = True)
    gen_credo_score_output_df



    # Combining DataFrames
    def concatenate_dataframes(dataframe_list):
        """
        Concatenates a list of Pandas DataFrames into a single DataFrame.

        Args:
            dataframe_list: A list of Pandas DataFrames to concatenate.

        Returns:
            A single concatenated Pandas DataFrame, or None if the input list is empty.
        """
        if not dataframe_list:
            return None  # Return None if the list is empty

        try:
            concatenated_df = pd.concat(dataframe_list, ignore_index=True)
            return concatenated_df
        except Exception as e:
            print(f"An error occurred during concatenation: {e}")
            return None


    dataframe_list = [
        s_apps_score_output_df,
        sb_demo_score_output_df,
        s_cic_score_output_df,
        sb_stack_score_output_df,
        sa_stack_score_output_df,
        c_credo_score_output_df,
        s_credo_score_output_df,
        fu_credo_score_output_df,
        r_credo_score_output_df,
        gen_credo_score_output_df,
    ]

    concatenated_result = concatenate_dataframes(dataframe_list)

    if concatenated_result is not None:
        print(concatenated_result)
    else:
        print("Concatenation failed or the input list was empty.")

    concatenated_result.dtypes

    sq = """drop table if exists prj-prod-dataplatform.dap_ds_poweruser_playground.Model_Psi;""" # Changes

    query_job = client.query(sq)
    query_job.result()



    # Define your table schema
    table_schema = [
        bigquery.SchemaField('Month', 'STRING'),
        bigquery.SchemaField('scorename', 'STRING'),
        bigquery.SchemaField('DateCategory', 'STRING'),
        bigquery.SchemaField('s_apps_score_psivalues', 'FLOAT64'),
        bigquery.SchemaField('account_count', 'INT64'),
        bigquery.SchemaField('sb_demo_score_psivalues', 'FLOAT64'),
        bigquery.SchemaField('s_cic_score_psivalues', 'FLOAT64'),
        bigquery.SchemaField('sb_stack_score_psivalues', 'FLOAT64'),
        bigquery.SchemaField('sa_stack_score_psivalues', 'FLOAT64'),
        bigquery.SchemaField('c_credo_score_psivalues', 'FLOAT64'),
        bigquery.SchemaField('s_credo_score_psivalue', 'FLOAT64'),
        bigquery.SchemaField('fu_credo_score_psivalues', 'FLOAT64'),
        bigquery.SchemaField('r_credo_score_psivalues', 'FLOAT64'),
        bigquery.SchemaField('gen_credo_score_psivalues', 'FLOAT64'),
    
    ]

    # Create your BigQuery table
    table_id = 'prj-prod-dataplatform.dap_ds_poweruser_playground.Model_Psi' # Changes
    table = bigquery.Table(table_id, schema=table_schema)
    table = client.create_table(table)

    # Load your DataFrame into BigQuery
    job_config = bigquery.LoadJobConfig(
        write_disposition='WRITE_TRUNCATE'
    )

    load_job = client.load_table_from_dataframe(
        concatenated_result, table_id, job_config=job_config
    )

    load_job.result()
