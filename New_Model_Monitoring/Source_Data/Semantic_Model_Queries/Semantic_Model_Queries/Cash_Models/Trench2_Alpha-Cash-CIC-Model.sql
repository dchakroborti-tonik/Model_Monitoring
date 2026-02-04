WITH parsed as (
  select customerId, digitalLoanAccountId,modelDisplayName,modelVersionId,start_time,end_time,prediction,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeatures,
--REPLACE(REPLACE(prediction, "'", '"'), "None", "null") AS prediction_clean
FROM `prj-prod-dataplatform.audit_balance.ml_model_run_details`
where modelDisplayName = 'Alpha-Cash-CIC-Model'),

latest_request as (
select * from parsed
QUALIFY ROW_NUMBER() OVER (PARTITION BY customerId, digitalLoanAccountId,modelDisplayName ORDER BY start_time DESC ) = 1),

model_run as (
select customerId,digitalLoanAccountId,modelName, publish_time,requestPayload as requestPayload_clean
--REPLACE(REPLACE(requestPayload, "'", '"'), "None", "null") AS requestPayload_clean
from `prj-prod-dataplatform.audit_balance.ml_request_details` 
WHERE modelName = 'Alpha-Cash-Model-response'
QUALIFY ROW_NUMBER() OVER (PARTITION BY customerId, digitalLoanAccountId,modelName ORDER BY publish_time DESC ) = 1)

select * from (
  select 
 r.customerId,
 r.digitalLoanAccountId,
 r.prediction,
 r.start_time,
 r.end_time,
 r.modelDisplayName,
 r.modelVersionId,

  REGEXP_EXTRACT(m.requestPayload_clean, r"osType[:=]['\"]?([^'\"]+)['\"]?") AS osType,
  REGEXP_EXTRACT(m.requestPayload_clean, r"loanType[:=]['\"]?([^'\"]+)['\"]?") AS loanType,
  REGEXP_EXTRACT(m.requestPayload_clean, r"trenchCategory[:=]['\"]?([^'\"]+)['\"]?") AS trenchCategory,
   SAFE_CAST(REGEXP_EXTRACT(m.requestPayload_clean, r"aStackScore[:= ]([0-9\.]+)") AS FLOAT64) AS aStackScore,
  SAFE_CAST(REGEXP_EXTRACT(m.requestPayload_clean, r"aCicScore[:= ]([0-9\.]+)") AS FLOAT64) AS aCicScore,
  --  Alpha CIC Score Model Features for Trench 2
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
  JSON_VALUE(r.calcFeatures, "$.cic_flg_zero_granted_ever") AS cic_flg_zero_granted_ever
FROM latest_request r
left join model_run m
on r.digitalLoanAccountId = m.digitalLoanAccountId 
 )
where trenchCategory = 'Trench 2'

