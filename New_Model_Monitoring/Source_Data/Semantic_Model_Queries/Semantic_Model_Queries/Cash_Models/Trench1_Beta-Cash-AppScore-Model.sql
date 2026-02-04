WITH parsed as (
  select customerId, digitalLoanAccountId,modelDisplayName,modelVersionId,start_time,end_time,prediction,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeatures,
REPLACE(REPLACE(prediction, "'", '"'), "None", "null") AS prediction_clean
FROM `prj-prod-dataplatform.audit_balance.ml_model_run_details`
where modelDisplayName = 'Beta-Cash-AppScore-Model'),

latest_request as (
select * from parsed
QUALIFY ROW_NUMBER() OVER (PARTITION BY customerId, digitalLoanAccountId,modelDisplayName ORDER BY start_time DESC ) = 1),

model_run as (
select customerId,digitalLoanAccountId,modelName, publish_time,
REPLACE(REPLACE(requestPayload, "'", '"'), "None", "null") AS requestPayload_clean
from `prj-prod-dataplatform.audit_balance.ml_request_details` 
WHERE modelName = 'Beta-Cash-Model-response'
QUALIFY ROW_NUMBER() OVER (PARTITION BY customerId, digitalLoanAccountId,modelName ORDER BY publish_time DESC ) = 1)

select * from (
  select 
 r.customerId,
 r.digitalLoanAccountId,
 --r.prediction,
 r.start_time,
 r.end_time,
 r.modelDisplayName,
 r.modelVersionId,

  JSON_VALUE(m.requestPayload_clean, "$.predictions.trenchCategory") AS trenchCategory,
  JSON_VALUE(m.requestPayload_clean, "$.predictions.loanType") AS loanType,
  --  Beta App Score Model Features for all Trench 1
  SAFE_CAST(JSON_VALUE(r.prediction_clean, "$.combined_score") AS Float64) AS combined_score,
  SAFE_CAST(JSON_VALUE(r.calcFeatures, "$.app_cnt_health_and_fitness_ever") AS INT64) AS app_cnt_health_and_fitness_ever,
  SAFE_CAST(JSON_VALUE(r.calcFeatures, "$.app_cnt_shopping_ever") AS FLOAT64) AS app_cnt_shopping_ever,
  SAFE_CAST(JSON_VALUE(r.calcFeatures, "$.app_cnt_crypto_ever") AS INT64) AS app_cnt_crypto_ever,
  JSON_VALUE(r.calcFeatures, "$.app_cnt_driver_ever") AS app_cnt_driver_ever,
  JSON_VALUE(r.calcFeatures, "$.app_cnt_payday_180d") AS app_cnt_payday_180d,
  JSON_VALUE(r.calcFeatures, "$.app_cnt_gambling_180d") AS app_cnt_gambling_180d,
  JSON_VALUE(r.calcFeatures, "$.app_avg_time_bw_installed_mins_3d") AS app_avg_time_bw_installed_mins_3d,
  JSON_VALUE(r.calcFeatures, "$.app_median_time_bw_installed_mins_3d") AS app_median_time_bw_installed_mins_3d,
  JSON_VALUE(r.calcFeatures, "$.app_avg_time_bw_installed_mins_ever") AS app_avg_time_bw_installed_mins_ever,
  JSON_VALUE(r.calcFeatures, "$.app_median_time_bw_installed_mins_ever") AS app_median_time_bw_installed_mins_ever,  
FROM latest_request r
left join model_run m
on r.digitalLoanAccountId = m.digitalLoanAccountId 
)
where trenchCategory = 'Trench 1'

