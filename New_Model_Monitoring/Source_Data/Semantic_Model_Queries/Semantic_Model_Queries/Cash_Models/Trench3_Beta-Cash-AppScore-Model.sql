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
  --  Beta App Score Model Features for Trench 3
  SAFE_CAST(JSON_VALUE(r.prediction_clean, "$.combined_score") AS Float64) AS combined_score,
  SAFE_CAST(JSON_VALUE(r.calcFeatures, "$.app_cnt_health_and_fitness_ever") AS INT64) AS app_cnt_health_and_fitness_ever,
  SAFE_CAST(JSON_VALUE(r.calcFeatures, "$.app_cnt_productivity_ever") AS FLOAT64) AS app_cnt_productivity_ever,
  SAFE_CAST(JSON_VALUE(r.calcFeatures, "$.app_cnt_rated_for_18plus_ever") AS INT64) AS app_cnt_rated_for_18plus_ever,
  JSON_VALUE(r.calcFeatures, "$.app_cnt_books_and_reference_ever") AS app_cnt_books_and_reference_ever,
  JSON_VALUE(r.calcFeatures, "$.app_cnt_gaming_180d") AS app_cnt_gaming_180d,
  JSON_VALUE(r.calcFeatures, "$.app_cnt_absence_tag_365d") AS app_cnt_absence_tag_365d,
  JSON_VALUE(r.calcFeatures, "$.app_last_payday_install_to_apply_days") AS app_last_payday_install_to_apply_days,
--  Beta App Score Model Binned Features for Trench 3
  JSON_VALUE(r.calcFeatures, "$.app_cnt_absence_tag_365d_binned") AS app_cnt_absence_tag_365d_binned,
  JSON_VALUE(r.calcFeatures, "$.app_cnt_gaming_180d_binned") AS app_cnt_gaming_180d_binned,
  JSON_VALUE(r.calcFeatures, "$.app_cnt_productivity_ever_binned") AS app_cnt_productivity_ever_binned,
  JSON_VALUE(r.calcFeatures, "$.app_cnt_rated_for_18plus_ever_binned") AS app_cnt_rated_for_18plus_ever_binned,
  JSON_VALUE(r.calcFeatures, "$.app_cnt_health_and_fitness_ever_binned") AS app_cnt_health_and_fitness_ever_binned,
  JSON_VALUE(r.calcFeatures, "$.app_cnt_books_and_reference_ever_binned") AS app_cnt_books_and_reference_ever_binned,
  JSON_VALUE(r.calcFeatures, "$.app_last_payday_install_to_apply_days_binned") AS app_last_payday_install_to_apply_days_binned,
  JSON_VALUE(r.calcFeatures, "$.ln_user_type") AS ln_user_type,  
FROM latest_request r
left join model_run m
on r.digitalLoanAccountId = m.digitalLoanAccountId 
)
where trenchCategory = 'Trench 3'

