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
  SAFE_CAST(JSON_VALUE(r.calcFeatures, "$.ln_appln_submit_datetime") AS TIMESTAMP) AS ln_appln_submit_datetime
  
FROM latest_request r
left join model_run m
on r.digitalLoanAccountId = m.digitalLoanAccountId 
)
where trenchCategory = 'Trench 2'

