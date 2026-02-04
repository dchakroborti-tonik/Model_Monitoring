WITH cleaned AS (
  SELECT
  customerId,digitalLoanAccountId,prediction,start_time,end_time,modelDisplayName,modelVersionId,
REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature_cleaned
  FROM `prj-prod-dataplatform.audit_balance.ml_model_run_details`
  WHERE modelDisplayName = 'Beta - DemoScoreModel')
SELECT
  customerId,  digitalLoanAccountId,start_time, prediction,
  JSON_VALUE(calcFeature_cleaned, "$.beta_de_ln_vas_opted_flag") AS beta_de_ln_vas_opted_flag,
  JSON_VALUE(calcFeature_cleaned, "$.beta_de_ln_doc_type_rolled") AS beta_de_ln_doc_type_rolled,
  JSON_VALUE(calcFeature_cleaned, "$.beta_de_ln_marital_status") AS beta_de_ln_marital_status,
  JSON_VALUE(calcFeature_cleaned, "$.beta_de_ln_age_bin") AS beta_de_ln_age_bin,
  JSON_VALUE(calcFeature_cleaned, "$.beta_de_ln_province_bin") AS beta_de_ln_province_bin,
  JSON_VALUE(calcFeature_cleaned, "$.beta_de_ln_ref2_type") AS beta_de_ln_ref2_type,
  JSON_VALUE(calcFeature_cleaned, "$.beta_de_ln_education_level") AS beta_de_ln_education_level,
  JSON_VALUE(calcFeature_cleaned, "$.beta_de_ln_ref1_type") AS beta_de_ln_ref1_type,
  JSON_VALUE(calcFeature_cleaned, "$.beta_de_ln_industry_new_bin") AS beta_de_ln_industry_new_bin,
  JSON_VALUE(calcFeature_cleaned, "$.beta_de_ln_appln_day_of_week") AS beta_de_ln_appln_day_of_week,
  JSON_VALUE(calcFeature_cleaned, "$.beta_de_onb_name_email_match_score") AS beta_de_onb_name_email_match_score,
  JSON_VALUE(calcFeature_cleaned, "$.beta_de_ln_employment_type_new_bin") AS beta_de_ln_employment_type_new_bin,
  JSON_VALUE(calcFeature_cleaned, "$.beta_de_ln_telconame") AS beta_de_ln_telconame,
  JSON_VALUE(calcFeature_cleaned, "$.beta_de_time_bw_onb_loan_appln_mins") AS beta_de_time_bw_onb_loan_appln_mins,
  JSON_VALUE(calcFeature_cleaned, "$.beta_de_ln_source_of_funds_new_bin") AS beta_de_ln_source_of_funds_new_bin,
  JSON_VALUE(calcFeature_cleaned, "$.beta_de_ln_brand_bin") AS beta_de_ln_brand_bin,
  JSON_VALUE(calcFeature_cleaned, "$.beta_de_ln_email_primary_domain") AS beta_de_ln_email_primary_domain
FROM cleaned