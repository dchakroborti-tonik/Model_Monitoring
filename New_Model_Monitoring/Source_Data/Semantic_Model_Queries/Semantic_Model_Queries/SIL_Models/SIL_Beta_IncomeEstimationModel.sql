WITH cleaned AS (
  SELECT
    customerId,digitalLoanAccountId,prediction,start_time,end_time,modelDisplayName,modelVersionId,

    REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature
  FROM `prj-prod-dataplatform.audit_balance.ml_model_run_details`
  WHERE modelDisplayName = 'Beta - IncomeEstimationModel')
SELECT
  customerId,digitalLoanAccountId,prediction,start_time,end_time,modelDisplayName,modelVersionId,
  JSON_VALUE(calcFeature, "$.inc_beta_ln_loan_type") AS inc_beta_ln_loan_type,
  JSON_VALUE(calcFeature, "$.inc_beta_ln_education_level") AS inc_beta_ln_education_level,
  JSON_VALUE(calcFeature, "$.inc_beta_ln_employment_type_new") AS inc_beta_ln_employment_type_new,
  JSON_VALUE(calcFeature, "$.inc_beta_ln_industry_new") AS inc_beta_ln_industry_new,
  JSON_VALUE(calcFeature, "$.inc_beta_ln_age") AS inc_beta_ln_age,
  JSON_VALUE(calcFeature, "$.inc_beta_ln_brand") AS inc_beta_ln_brand,
  JSON_VALUE(calcFeature, "$.inc_beta_ln_city") AS inc_beta_ln_city,
  JSON_VALUE(calcFeature, "$.inc_beta_ln_purpose") AS inc_beta_ln_purpose,
  JSON_VALUE(calcFeature, "$.inc_beta_ln_osversion_bin") AS inc_beta_ln_osversion_bin,
  JSON_VALUE(calcFeature, "$.inc_beta_ln_postal_code") AS inc_beta_ln_postal_code,
  JSON_VALUE(calcFeature, "$.inc_beta_ln_gender") AS inc_beta_ln_gender,
  JSON_VALUE(calcFeature, "$.inc_beta_ln_doc_type_rolled") AS inc_beta_ln_doc_type_rolled,
  JSON_VALUE(calcFeature, "$.inc_beta_ln_cnt_dependents") AS inc_beta_ln_cnt_dependents,
  JSON_VALUE(calcFeature, "$.inc_beta_ln_source_of_funds_new") AS inc_beta_ln_source_of_funds_new,
  JSON_VALUE(calcFeature, "$.inc_beta_ln_marital_status_new") AS inc_beta_ln_marital_status_new,
  JSON_VALUE(calcFeature, "$.inc_beta_encoded_company_name_grouped") AS inc_beta_encoded_company_name_grouped
FROM cleaned