WITH parsed as (
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
 r.prediction,
 r.start_time,
 r.end_time,
 r.modelDisplayName,
 r.modelVersionId,

  REGEXP_EXTRACT(m.requestPayload_clean, r"osType[:=]['\"]?([^'\"]+)['\"]?") AS osType,
  REGEXP_EXTRACT(m.requestPayload_clean, r"loanType[:=]['\"]?([^'\"]+)['\"]?") AS loanType,
  REGEXP_EXTRACT(m.requestPayload_clean, r"trenchCategory[:=]['\"]?([^'\"]+)['\"]?") AS trenchCategory,

  SAFE_CAST(JSON_VALUE(r.calcFeatures, "$.apps_score") AS FLOAT64) AS  apps_score,
 SAFE_CAST(JSON_VALUE(r.calcFeatures, "$.c_demo_score") AS FLOAT64) AS  c_demo_score,
 SAFE_CAST(JSON_VALUE(r.calcFeatures, "$.c_credo_score") AS FLOAT64) AS  c_credo_score,
 SAFE_CAST(JSON_VALUE(r.calcFeatures, "$.c_tx_score") AS FLOAT64) AS  c_tx_score,
 SAFE_CAST(JSON_VALUE(r.calcFeatures, "$.ca_cic_score") AS FLOAT64) AS  ca_cic_score,

FROM latest_request r
left join model_run m
on r.digitalLoanAccountId = m.digitalLoanAccountId 
) where trenchCategory = 'Trench 2'


