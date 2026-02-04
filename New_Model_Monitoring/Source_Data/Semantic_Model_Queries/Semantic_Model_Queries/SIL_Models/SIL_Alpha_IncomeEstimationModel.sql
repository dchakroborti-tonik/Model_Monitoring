WITH cleaned AS (
  SELECT
    customerId,digitalLoanAccountId,prediction,start_time,end_time,modelDisplayName,modelVersionId,

    REPLACE(REPLACE(calcFeature, "'", '"'), "None", "null") AS calcFeature
  FROM `prj-prod-dataplatform.audit_balance.ml_model_run_details`
  WHERE modelDisplayName = 'Alpha  - IncomeEstimationModel')
SELECT
  customerId,digitalLoanAccountId,prediction,start_time,end_time,modelDisplayName,modelVersionId,
  JSON_VALUE(calcFeature, "$.inc_alpha_cic_credit_avg_credit_limit") AS inc_alpha_cic_credit_avg_credit_limit,
  JSON_VALUE(calcFeature, "$.inc_alpha_cic_max_active_contracts_amt") AS inc_alpha_cic_max_active_contracts_amt,
  JSON_VALUE(calcFeature, "$.inc_alpha_ln_age") AS inc_alpha_ln_age,
  JSON_VALUE(calcFeature, "$.inc_alpha_doc_type_rolled") AS inc_alpha_doc_type_rolled,
  JSON_VALUE(calcFeature, "$.inc_alpha_ln_brand") AS inc_alpha_ln_brand,
  JSON_VALUE(calcFeature, "$.inc_alpha_ln_city") AS inc_alpha_ln_city,
  JSON_VALUE(calcFeature, "$.inc_alpha_ln_cnt_dependents") AS inc_alpha_ln_cnt_dependents,
  JSON_VALUE(calcFeature, "$.inc_alpha_ln_education_level") AS inc_alpha_ln_education_level,
  JSON_VALUE(calcFeature, "$.inc_alpha_ln_employment_type_new") AS inc_alpha_ln_employment_type_new,
  JSON_VALUE(calcFeature, "$.inc_alpha_ln_gender") AS inc_alpha_ln_gender,
  JSON_VALUE(calcFeature, "$.inc_alpha_ln_industry_new") AS inc_alpha_ln_industry_new,
  JSON_VALUE(calcFeature, "$.inc_alpha_ln_loan_prod_type") AS inc_alpha_ln_loan_prod_type,
  JSON_VALUE(calcFeature, "$.inc_alpha_ln_marital_status_new") AS inc_alpha_ln_marital_status_new,
  JSON_VALUE(calcFeature, "$.inc_alpha_ln_nature_of_work_new") AS inc_alpha_ln_nature_of_work_new,
  JSON_VALUE(calcFeature, "$.inc_alpha_ln_osversion_bin") AS inc_alpha_ln_osversion_bin,
  JSON_VALUE(calcFeature, "$.inc_alpha_ln_purpose") AS inc_alpha_ln_purpose,
  JSON_VALUE(calcFeature, "$.inc_alpha_ln_source_of_funds_new") AS inc_alpha_ln_source_of_funds_new,
  JSON_VALUE(calcFeature, "$.inc_alpha_encoded_company_name_grouped") AS inc_alpha_encoded_company_name_grouped
FROM cleaned