WITH parsed as (
  select customerId, digitalLoanAccountId,modelDisplayName,modelVersionId,start_time,end_time,prediction,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeatures,
--REPLACE(REPLACE(prediction, "'", '"'), "None", "null") AS prediction_clean
FROM `prj-prod-dataplatform.audit_balance.ml_model_run_details`
where modelDisplayName = 'Beta-Cash-Stack-Model'),

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
 r.prediction,
 r.start_time,
 r.end_time,
 r.modelDisplayName,
 r.modelVersionId,


 JSON_VALUE(m.requestPayload_clean, "$.predictions.trenchCategory") AS trenchCategory,

  SAFE_CAST(JSON_VALUE(r.calcFeatures, "$.demo_score") AS FLOAT64) AS  demo_score,
 SAFE_CAST(JSON_VALUE(r.calcFeatures, "$.credo_score") AS FLOAT64) AS  credo_score,
 SAFE_CAST(JSON_VALUE(r.calcFeatures, "$.trx_score") AS FLOAT64) AS  trx_score,
 SAFE_CAST(JSON_VALUE(r.calcFeatures, "$.app_score") AS FLOAT64) AS  app_score

FROM latest_request r
left join model_run m
on r.digitalLoanAccountId = m.digitalLoanAccountId 
  )
 where trenchCategory = 'Trench 1'

