WITH cleaned AS (
  SELECT
    customerId,digitalLoanAccountId,prediction,start_time,end_time,modelDisplayName,modelVersionId,

    REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature
  FROM `prj-prod-dataplatform.audit_balance.ml_model_run_details`
  WHERE modelDisplayName = 'Alpha - CIC-SIL-Model')
SELECT
  customerId,digitalLoanAccountId,prediction,start_time,end_time,modelDisplayName,modelVersionId,
  JSON_VALUE(calcFeature, "$.cic_Personal_Loans_granted_contracts_amt_24M") AS cic_Personal_Loans_granted_contracts_amt_24M,
  JSON_VALUE(calcFeature, "$.cic_days_since_last_inquiry") AS cic_days_since_last_inquiry,
  JSON_VALUE(calcFeature, "$.cic_cnt_active_contracts") AS cic_cnt_active_contracts,
  JSON_VALUE(calcFeature, "$.cic_vel_contract_nongranted_cnt_12on24") AS cic_vel_contract_nongranted_cnt_12on24,
  JSON_VALUE(calcFeature, "$.cic_max_amt_granted_24M") AS cic_max_amt_granted_24M,
  JSON_VALUE(calcFeature, "$.cic_zero_non_granted_ever_flag") AS cic_zero_non_granted_ever_flag,
  JSON_VALUE(calcFeature, "$.cic_tot_active_contracts_util") AS cic_tot_active_contracts_util,
  JSON_VALUE(calcFeature, "$.cic_vel_contract_granted_amt_12on24") AS cic_vel_contract_granted_amt_12on24,
  JSON_VALUE(calcFeature, "$.cic_zero_granted_ever_flag") AS cic_zero_granted_ever_flag
FROM cleaned