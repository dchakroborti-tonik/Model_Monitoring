SELECT
  customer_id,
  observation_date,
  trench,
  s_t3g_is_android,
  s_t3g_de_score,
  s_t3_tx_score,
  s_t3g_de_score_norm,
  s_t3_tx_score_norm,
  s_t3g_stack_score,
  -- demo_model_details,
  -- trx_model_details,
  -- stack_model_details,

  -- Flattened model details
  demo_model.model_type AS demo_model_type,
  demo_model.version AS demo_model_version,
  demo_model.date AS demo_model_date,
  demo_model.file_path AS demo_model_file_path,
  demo_model.file_name AS demo_model_file_name,

  trx_model.model_type AS trx_model_type,
  trx_model.version AS trx_model_version,
  trx_model.date AS trx_model_date,
  trx_model.file_path AS trx_model_file_path,
  trx_model.file_name AS trx_model_file_name,

  stack_model.model_type AS stack_model_type,
  stack_model.version AS stack_model_version,
  stack_model.date AS stack_model_date,
  stack_model.file_path AS stack_model_file_path,
  stack_model.file_name AS stack_model_file_name,

  -- Flatten s_t3_gamma_demo_feature
  JSON_VALUE(s_t3_gamma_demo_feature, '$.s_t3g_de_ln_age_observation_date') AS s_t3g_de_ln_age_observation_date,
  JSON_VALUE(s_t3_gamma_demo_feature, '$.s_t3g_de_ln_self_dec_income') AS s_t3g_de_ln_self_dec_income,
  JSON_VALUE(s_t3_gamma_demo_feature, '$.s_t3g_de_ln_vas_opted_flag') AS s_t3g_de_ln_vas_opted_flag,
  JSON_VALUE(s_t3_gamma_demo_feature, '$.s_t3g_de_ln_industry_new_cat_bin') AS s_t3g_de_ln_industry_new_cat_bin,
  JSON_VALUE(s_t3_gamma_demo_feature, '$.s_t3g_de_ln_province_cat') AS s_t3g_de_ln_province_cat,
  JSON_VALUE(s_t3_gamma_demo_feature, '$.s_t3g_de_ln_doc_type_rolled') AS s_t3g_de_ln_doc_type_rolled,
  JSON_VALUE(s_t3_gamma_demo_feature, '$.s_t3g_de_ln_marital_status') AS s_t3g_de_ln_marital_status,
  JSON_VALUE(s_t3_gamma_demo_feature, '$.s_t3g_de_ln_telconame') AS s_t3g_de_ln_telconame,
  JSON_VALUE(s_t3_gamma_demo_feature, '$.s_t3g_de_ln_brand_cat') AS s_t3g_de_ln_brand_cat,
  JSON_VALUE(s_t3_gamma_demo_feature, '$.s_t3g_de_application_Is_Weekend') AS s_t3g_de_application_Is_Weekend,
  JSON_VALUE(s_t3_gamma_demo_feature, '$.s_t3g_de_ln_education_level') AS s_t3g_de_ln_education_level,
  JSON_VALUE(s_t3_gamma_demo_feature, '$.s_t3g_de_ln_source_funds_new_cat_bin') AS s_t3g_de_ln_source_funds_new_cat_bin,
  JSON_VALUE(s_t3_gamma_demo_feature, '$.s_t3g_de_flg_has_alternate_mob_no') AS s_t3g_de_flg_has_alternate_mob_no,
  JSON_VALUE(s_t3_gamma_demo_feature, '$.s_t3g_de_ln_age_observation_date_WOE') AS s_t3g_de_ln_age_observation_date_WOE,
  JSON_VALUE(s_t3_gamma_demo_feature, '$.s_t3g_de_ln_self_dec_income_WOE') AS s_t3g_de_ln_self_dec_income_WOE,
  JSON_VALUE(s_t3_gamma_demo_feature, '$.s_t3g_de_ln_vas_opted_flag_WOE') AS s_t3g_de_ln_vas_opted_flag_WOE,
  JSON_VALUE(s_t3_gamma_demo_feature, '$.s_t3g_de_ln_industry_new_cat_bin_WOE') AS s_t3g_de_ln_industry_new_cat_bin_WOE,
  JSON_VALUE(s_t3_gamma_demo_feature, '$.s_t3g_de_ln_province_cat_WOE') AS s_t3g_de_ln_province_cat_WOE,
  JSON_VALUE(s_t3_gamma_demo_feature, '$.s_t3g_de_ln_doc_type_rolled_WOE') AS s_t3g_de_ln_doc_type_rolled_WOE,
  JSON_VALUE(s_t3_gamma_demo_feature, '$.s_t3g_de_ln_marital_status_WOE') AS s_t3g_de_ln_marital_status_WOE,
  JSON_VALUE(s_t3_gamma_demo_feature, '$.s_t3g_de_ln_telconame_WOE') AS s_t3g_de_ln_telconame_WOE,
  JSON_VALUE(s_t3_gamma_demo_feature, '$.s_t3g_de_ln_brand_cat_WOE') AS s_t3g_de_ln_brand_cat_WOE,
  JSON_VALUE(s_t3_gamma_demo_feature, '$.s_t3g_de_application_Is_Weekend_WOE') AS s_t3g_de_application_Is_Weekend_WOE,
  JSON_VALUE(s_t3_gamma_demo_feature, '$.s_t3g_de_ln_education_level_WOE') AS s_t3g_de_ln_education_level_WOE,
  JSON_VALUE(s_t3_gamma_demo_feature, '$.s_t3g_de_ln_source_funds_new_cat_bin_WOE') AS s_t3g_de_ln_source_funds_new_cat_bin_WOE,
  JSON_VALUE(s_t3_gamma_demo_feature, '$.s_t3g_de_flg_has_alternate_mob_no_WOE') AS s_t3g_de_flg_has_alternate_mob_no_WOE,

  -- Flatten s_t3_gamma_transaction_feature
  JSON_VALUE(s_t3_gamma_transaction_feature, '$.s_t3_tx_time_since_last_applied_loan_application_time') AS s_t3_tx_time_since_last_applied_loan_application_time,
  JSON_VALUE(s_t3_gamma_transaction_feature, '$.s_t3_tx_first_applied_loan_amount') AS s_t3_tx_first_applied_loan_amount,
  JSON_VALUE(s_t3_gamma_transaction_feature, '$.s_t3_tx_first_applied_loan_tenor') AS s_t3_tx_first_applied_loan_tenor,
  JSON_VALUE(s_t3_gamma_transaction_feature, '$.s_t3_tx_max_ever_dpd') AS s_t3_tx_max_ever_dpd,
  JSON_VALUE(s_t3_gamma_transaction_feature, '$.s_t3_tx_cnt_installments_paid_tot_with_dpd') AS s_t3_tx_cnt_installments_paid_tot_with_dpd,
  JSON_VALUE(s_t3_gamma_transaction_feature, '$.s_t3_tx_cnt_jira_tickets_created_bin') AS s_t3_tx_cnt_jira_tickets_created_bin,
  JSON_VALUE(s_t3_gamma_transaction_feature, '$.s_t3_tx_last_applied_loan_type_bin') AS s_t3_tx_last_applied_loan_type_bin,
  JSON_VALUE(s_t3_gamma_transaction_feature, '$.s_t3_tx_amt_cash_in_total') AS s_t3_tx_amt_cash_in_total,
  JSON_VALUE(s_t3_gamma_transaction_feature, '$.s_t3_tx_min_age_completed_loans') AS s_t3_tx_min_age_completed_loans,
  JSON_VALUE(s_t3_gamma_transaction_feature, '$.s_t3_tx_first_product_bin') AS s_t3_tx_first_product_bin,
  JSON_VALUE(s_t3_gamma_transaction_feature, '$.s_t3_tx_meng_no_of_logins') AS s_t3_tx_meng_no_of_logins,
  JSON_VALUE(s_t3_gamma_transaction_feature, '$.s_t3_tx_dob_observation_date') AS s_t3_tx_dob_observation_date,
  JSON_VALUE(s_t3_gamma_transaction_feature, '$.s_t3_tx_avg_days_bt_trans') AS s_t3_tx_avg_days_bt_trans,
  JSON_VALUE(s_t3_gamma_transaction_feature, '$.s_t3_tx_cs_contactable_last_90d_cnt') AS s_t3_tx_cs_contactable_last_90d_cnt

FROM `prj-prod-dataplatform.ml_model_prediction.sil_behavioral_score_master`,
UNNEST([STRUCT(
    -- Parent struct holds all three models
    STRUCT(
      SPLIT(REPLACE(REPLACE(demo_model_details, '{', ''), '}', ''), ':')[OFFSET(0)] AS model_type,
      SPLIT(REPLACE(REPLACE(demo_model_details, '{', ''), '}', ''), ':')[OFFSET(1)] AS version,
      SPLIT(REPLACE(REPLACE(demo_model_details, '{', ''), '}', ''), ':')[OFFSET(2)] AS date,
      SPLIT(REPLACE(REPLACE(demo_model_details, '{', ''), '}', ''), ':')[OFFSET(3)] AS file_path,
      SPLIT(REPLACE(REPLACE(demo_model_details, '{', ''), '}', ''), ':')[OFFSET(4)] AS file_name
    ) AS demo_model,
    STRUCT(
      SPLIT(REPLACE(REPLACE(trx_model_details, '{', ''), '}', ''), ':')[OFFSET(0)] AS model_type,
      SPLIT(REPLACE(REPLACE(trx_model_details, '{', ''), '}', ''), ':')[OFFSET(1)] AS version,
      SPLIT(REPLACE(REPLACE(trx_model_details, '{', ''), '}', ''), ':')[OFFSET(2)] AS date,
      SPLIT(REPLACE(REPLACE(trx_model_details, '{', ''), '}', ''), ':')[OFFSET(3)] AS file_path,
      SPLIT(REPLACE(REPLACE(trx_model_details, '{', ''), '}', ''), ':')[OFFSET(4)] AS file_name
    ) AS trx_model,
    STRUCT(
      SPLIT(REPLACE(REPLACE(stack_model_details, '{', ''), '}', ''), ':')[OFFSET(0)] AS model_type,
      SPLIT(REPLACE(REPLACE(stack_model_details, '{', ''), '}', ''), ':')[OFFSET(1)] AS version,
      SPLIT(REPLACE(REPLACE(stack_model_details, '{', ''), '}', ''), ':')[OFFSET(2)] AS date,
      SPLIT(REPLACE(REPLACE(stack_model_details, '{', ''), '}', ''), ':')[OFFSET(3)] AS file_path,
      SPLIT(REPLACE(REPLACE(stack_model_details, '{', ''), '}', ''), ':')[OFFSET(4)] AS file_name
    ) AS stack_model
  )]) AS model_info;