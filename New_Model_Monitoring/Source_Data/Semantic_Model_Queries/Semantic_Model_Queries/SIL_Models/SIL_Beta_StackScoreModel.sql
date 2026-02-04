WITH cleaned AS (
  SELECT
    customerId,digitalLoanAccountId,prediction,start_time,end_time,modelDisplayName,modelVersionId,

    REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature
  FROM `prj-prod-dataplatform.audit_balance.ml_model_run_details`
  WHERE modelDisplayName = 'Beta - StackScoreModel')
SELECT
  customerId,digitalLoanAccountId,prediction,start_time,end_time,modelDisplayName,modelVersionId,
  JSON_VALUE(calcFeature, "$.s_apps_score") AS s_apps_score,
  JSON_VALUE(calcFeature, "$.s_credo_score") AS s_credo_score,
  JSON_VALUE(calcFeature, "$.sb_demo_score") AS sb_demo_score
FROM cleaned
