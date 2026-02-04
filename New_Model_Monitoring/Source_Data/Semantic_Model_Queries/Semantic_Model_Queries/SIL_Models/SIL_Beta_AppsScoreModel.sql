WITH cleaned AS (
  SELECT
    customerId,digitalLoanAccountId,prediction,start_time,end_time,modelDisplayName,modelVersionId,

    REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature,
    REPLACE(REPLACE(prediction, "'", '"'), "None", "null") AS prediction_clean
  FROM `prj-prod-dataplatform.audit_balance.ml_model_run_details`
  WHERE modelDisplayName = 'Beta - AppsScoreModel')
SELECT
  customerId,digitalLoanAccountId,prediction,start_time,end_time,modelDisplayName,modelVersionId,
  safe_cast(JSON_VALUE(prediction_clean, "$.combined_score") AS float64) as combined_score,
  JSON_VALUE(calcFeature, "$.app_cnt_rated_for_3plus_ever") AS app_cnt_rated_for_3plus_ever,
  JSON_VALUE(calcFeature, "$.app_cnt_education_ever") AS app_cnt_education_ever,
  JSON_VALUE(calcFeature, "$.app_cnt_business_ever") AS app_cnt_business_ever,
  JSON_VALUE(calcFeature, "$.app_cnt_music_and_audio_ever") AS app_cnt_music_and_audio_ever,
  JSON_VALUE(calcFeature, "$.app_cnt_travel_and_local_ever") AS app_cnt_travel_and_local_ever,
  JSON_VALUE(calcFeature, "$.app_cnt_finance_7d") AS app_cnt_finance_7d,
  JSON_VALUE(calcFeature, "$.app_cnt_competitors_30d") AS app_cnt_competitors_30d,
  JSON_VALUE(calcFeature, "$.app_cnt_finance_30d") AS app_cnt_finance_30d,
  JSON_VALUE(calcFeature, "$.app_cnt_absence_tag_30d") AS app_cnt_absence_tag_30d,
  JSON_VALUE(calcFeature, "$.app_cnt_absence_tag_90d") AS app_cnt_absence_tag_90d,
  JSON_VALUE(calcFeature, "$.app_cnt_finance_90d") AS app_cnt_finance_90d,
  JSON_VALUE(calcFeature, "$.app_cnt_competitors_90d") AS app_cnt_competitors_90d,
  JSON_VALUE(calcFeature, "$.app_cnt_payday_90d") AS app_cnt_payday_90d,
  JSON_VALUE(calcFeature, "$.app_avg_time_bw_installed_mins_30d") AS app_avg_time_bw_installed_mins_30d,
  JSON_VALUE(calcFeature, "$.app_median_time_bw_installed_mins_30d") AS app_median_time_bw_installed_mins_30d,
  JSON_VALUE(calcFeature, "$.app_first_competitors_install_to_apply_days") AS app_first_competitors_install_to_apply_days,
  JSON_VALUE(calcFeature, "$.app_first_payday_install_to_apply_days") AS app_first_payday_install_to_apply_days,
  JSON_VALUE(calcFeature, "$.app_vel_finance_30_over_365") AS app_vel_finance_30_over_365
FROM cleaned
