SELECT
  customer_id,
  observation_date,
  trench,
  c_t3g_is_android,
  c_t3g_de_score,
  c_t3_tx_score,
  c_t3g_de_score_norm,
  c_t3_tx_score_norm,
  c_t3g_stack_score,

  -- Flattened demo features
  demo_features.c_t3g_de_ln_ref1_type,
  demo_features.c_t3g_de_ln_self_dec_income,
  demo_features.c_t3g_de_ln_age_observation_date,
  demo_features.c_t3g_de_ln_vas_opted_flag,
  demo_features.c_t3g_de_ln_industry_new_cat_bin,
  demo_features.c_t3g_de_ln_brand_cat,
  demo_features.c_t3g_de_kyc_gender,
  demo_features.c_t3g_de_email_primary_domain,
  demo_features.c_t3g_de_ln_doc_type_rolled,
  demo_features.c_t3g_de_name_email_match_score,

  -- Flattened transaction features
  trx_features.c_t3_tx_cnt_installments_paid_tot_with_dpd,
  trx_features.c_t3_tx_time_since_last_applied_loan_application_time,
  trx_features.c_t3_tx_last_applied_loan_decision,
  trx_features.c_t3_tx_min_age_completed_loans,
  trx_features.c_t3_tx_dob_observation_date,
  trx_features.c_t3_tx_cnt_jira_tickets_created_bin,
  trx_features.c_t3_tx_max_ever_dpd,
  trx_features.c_t3_tx_amt_cash_in_total,
  trx_features.c_t3_tx_last_applied_loan_type_bin,
  trx_features.c_t3_tx_cnt_completed_loans,
  trx_features.c_t3_tx_meng_no_of_logins,
  trx_features.c_t3_tx_last_applied_loan_tenor,
  trx_features.c_t3_tx_med_days_bt_cash_out_trans,
  trx_features.c_t3_tx_avg_days_bt_cash_in_trans,

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
  stack_model.file_name AS stack_model_file_name

FROM
  `prj-prod-dataplatform.ml_model_prediction.cash_behavioral_score_master`,
  UNNEST([STRUCT(
    SAFE.PARSE_JSON(c_t3_gamma_demo_feature) AS demo_features,
    SAFE.PARSE_JSON(c_t3_gamma_transaction_feature) AS trx_features,
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
  )])
;