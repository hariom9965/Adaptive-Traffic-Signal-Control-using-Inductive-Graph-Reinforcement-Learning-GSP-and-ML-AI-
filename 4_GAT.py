<!DOCTYPE html>
<!-- saved from url=(0067)https://github.com/manishvat/gat-traffic-control/blob/main/4_GAT.py -->
<html lang="en" data-color-mode="auto" data-light-theme="light" data-dark-theme="dark" data-a11y-animated-images="system" data-a11y-link-underlines="true" class="js-focus-visible" data-js-focus-visible="" data-turbo-loaded=""><head><meta http-equiv="Content-Type" content="text/html; charset=UTF-8"><style type="text/css">.turbo-progress-bar {
  position: fixed;
  display: block;
  top: 0;
  left: 0;
  height: 3px;
  background: #0076ff;
  z-index: 2147483647;
  transition:
    width 300ms ease-out,
    opacity 150ms 150ms ease-in;
  transform: translate3d(0, 0, 0);
}
</style>
    
  <link rel="dns-prefetch" href="https://github.githubassets.com/">
  <link rel="dns-prefetch" href="https://avatars.githubusercontent.com/">
  <link rel="dns-prefetch" href="https://github-cloud.s3.amazonaws.com/">
  <link rel="dns-prefetch" href="https://user-images.githubusercontent.com/">
  <link rel="preconnect" href="https://github.githubassets.com/" crossorigin="">
  <link rel="preconnect" href="https://avatars.githubusercontent.com/">

  


  <link crossorigin="anonymous" media="all" rel="stylesheet" href="./4_GAT_files/light-0c8222dcd7a4f9b7.css"><link crossorigin="anonymous" media="all" rel="stylesheet" href="./4_GAT_files/light_high_contrast-51c0c6e0c085cc0f.css"><link crossorigin="anonymous" media="all" rel="stylesheet" href="./4_GAT_files/dark-fc6eec18532c3ae0.css"><link crossorigin="anonymous" media="all" rel="stylesheet" href="./4_GAT_files/dark_high_contrast-96d7b2bab5a6ae4e.css"><link data-color-theme="light" crossorigin="anonymous" media="all" rel="stylesheet" data-href="https://github.githubassets.com/assets/light-0c8222dcd7a4f9b7.css"><link data-color-theme="light_high_contrast" crossorigin="anonymous" media="all" rel="stylesheet" data-href="https://github.githubassets.com/assets/light_high_contrast-51c0c6e0c085cc0f.css"><link data-color-theme="light_colorblind" crossorigin="anonymous" media="all" rel="stylesheet" data-href="https://github.githubassets.com/assets/light_colorblind-4dd12c5689d6b012.css"><link data-color-theme="light_colorblind_high_contrast" crossorigin="anonymous" media="all" rel="stylesheet" data-href="https://github.githubassets.com/assets/light_colorblind_high_contrast-dfa0c9e22ba6ba2b.css"><link data-color-theme="light_tritanopia" crossorigin="anonymous" media="all" rel="stylesheet" data-href="https://github.githubassets.com/assets/light_tritanopia-9fd9c8859395d1a8.css"><link data-color-theme="light_tritanopia_high_contrast" crossorigin="anonymous" media="all" rel="stylesheet" data-href="https://github.githubassets.com/assets/light_tritanopia_high_contrast-9c32304a2a8ac631.css"><link data-color-theme="dark" crossorigin="anonymous" media="all" rel="stylesheet" data-href="https://github.githubassets.com/assets/dark-fc6eec18532c3ae0.css"><link data-color-theme="dark_high_contrast" crossorigin="anonymous" media="all" rel="stylesheet" data-href="https://github.githubassets.com/assets/dark_high_contrast-96d7b2bab5a6ae4e.css"><link data-color-theme="dark_colorblind" crossorigin="anonymous" media="all" rel="stylesheet" data-href="https://github.githubassets.com/assets/dark_colorblind-0c6ca283d4d35cea.css"><link data-color-theme="dark_colorblind_high_contrast" crossorigin="anonymous" media="all" rel="stylesheet" data-href="https://github.githubassets.com/assets/dark_colorblind_high_contrast-2dc46b4919fae81e.css"><link data-color-theme="dark_tritanopia" crossorigin="anonymous" media="all" rel="stylesheet" data-href="https://github.githubassets.com/assets/dark_tritanopia-271332ec9362e8d3.css"><link data-color-theme="dark_tritanopia_high_contrast" crossorigin="anonymous" media="all" rel="stylesheet" data-href="https://github.githubassets.com/assets/dark_tritanopia_high_contrast-eb5bb84e91d6d553.css"><link data-color-theme="dark_dimmed" crossorigin="anonymous" media="all" rel="stylesheet" data-href="https://github.githubassets.com/assets/dark_dimmed-f3aa862f2ac7ead2.css"><link data-color-theme="dark_dimmed_high_contrast" crossorigin="anonymous" media="all" rel="stylesheet" data-href="https://github.githubassets.com/assets/dark_dimmed_high_contrast-206b6b93f856629b.css">

  <style type="text/css">
    :root {
      --tab-size-preference: 4;
    }

    pre, code {
      tab-size: var(--tab-size-preference);
    }
  </style>

    <link crossorigin="anonymous" media="all" rel="stylesheet" href="./4_GAT_files/primer-primitives-10bf9dd67e3d70bd.css">
    <link crossorigin="anonymous" media="all" rel="stylesheet" href="./4_GAT_files/primer-0fcd9af82350aeda.css">
    <link crossorigin="anonymous" media="all" rel="stylesheet" href="./4_GAT_files/global-0bd78641c0a1f3e0.css">
    <link crossorigin="anonymous" media="all" rel="stylesheet" href="./4_GAT_files/github-c94ab8d1f22049a8.css">
  <link crossorigin="anonymous" media="all" rel="stylesheet" href="./4_GAT_files/repository-6ec84ae2261fecf8.css">
<link crossorigin="anonymous" media="all" rel="stylesheet" href="./4_GAT_files/code-2d31826944fd3be8.css">

  

  <script type="application/json" id="client-env">{"locale":"en","featureFlags":["actions_custom_images_storage_billing_ui_visibility","actions_image_version_event","actions_service_container_command","alternate_user_config_repo","arianotify_comprehensive_migration","batch_suggested_changes","billing_discount_threshold_notification","cca_rename","close_pr_side_panel","code_scanning_all_branch_query","code_scanning_dfa_degraded_experience_notice","codespaces_prebuild_region_target_update","coding_agent_model_selection","coding_agent_model_selection_all_skus","comment_viewer_copy_raw_markdown","contentful_primer_code_blocks","copilot_agent_image_upload","copilot_agent_snippy","copilot_api_agentic_issue_marshal_yaml","copilot_ask_mode_dropdown","copilot_chat_attach_multiple_images","copilot_chat_clear_model_selection_for_default_change","copilot_chat_enable_tool_call_logs","copilot_chat_explain_error_user_model","copilot_chat_file_redirect","copilot_chat_input_commands","copilot_chat_opening_thread_switch","copilot_chat_reduce_quota_checks","copilot_chat_search_bar_redirect","copilot_chat_selection_attachments","copilot_chat_vision_in_claude","copilot_chat_vision_preview_gate","copilot_code_review_batch_apply_suggestions","copilot_coding_agent_task_response","copilot_custom_copilots","copilot_custom_copilots_feature_preview","copilot_duplicate_thread","copilot_extensions_hide_in_dotcom_chat","copilot_extensions_removal_on_marketplace","copilot_features_sql_server_logo","copilot_features_zed_logo","copilot_file_block_ref_matching","copilot_ftp_hyperspace_upgrade_prompt","copilot_icebreakers_experiment_dashboard","copilot_icebreakers_experiment_hyperspace","copilot_immersive_code_block_transition_wrap","copilot_immersive_embedded","copilot_immersive_file_block_transition_open","copilot_immersive_file_preview_keep_mounted","copilot_immersive_job_result_preview","copilot_immersive_layout_routes","copilot_immersive_structured_model_picker","copilot_immersive_task_hyperlinking","copilot_immersive_task_within_chat_thread","copilot_mc_cli_resume_any_users_task","copilot_mission_control_always_send_integration_id","copilot_mission_control_cli_resume_with_task_id","copilot_mission_control_decoupled_mode","copilot_mission_control_decoupled_mode_agent_tooltip","copilot_mission_control_initial_data_spinner","copilot_mission_control_plan_model_picker","copilot_mission_control_post_session_chat_model_picker","copilot_mission_control_rich_plan_ui","copilot_mission_control_scroll_to_bottom_button","copilot_mission_control_task_alive_updates","copilot_mission_control_task_unarchive","copilot_org_policy_page_focus_mode","copilot_redirect_header_button_to_agents","copilot_resource_panel","copilot_scroll_preview_tabs","copilot_share_active_subthread","copilot_spaces_ga","copilot_spaces_individual_policies_ga","copilot_spaces_pagination","copilot_spark_empty_state","copilot_spark_handle_nil_friendly_name","copilot_swe_agent_hide_model_picker_if_only_auto","copilot_swe_agent_pr_comment_model_picker","copilot_swe_agent_use_subagents","copilot_task_api_github_rest_style","copilot_unconfigured_is_inherited","copilot_usage_metrics_ga","copilot_workbench_slim_line_top_tabs","custom_instructions_file_references","custom_properties_consolidate_default_value_input","dashboard_add_updated_desc","dashboard_indexeddb_caching","dashboard_lists_max_age_filter","dashboard_universe_2025_feedback_dialog","flex_cta_groups_mvp","global_nav_react","hyperspace_2025_logged_out_batch_1","hyperspace_2025_logged_out_batch_2","hyperspace_2025_logged_out_batch_3","ipm_global_transactional_message_agents","ipm_global_transactional_message_copilot","ipm_global_transactional_message_issues","ipm_global_transactional_message_prs","ipm_global_transactional_message_repos","ipm_global_transactional_message_spaces","issue_cca_modal_open","issue_cca_visualization","issue_fields_global_search","issue_fields_visibility_indicator","issue_fields_visibility_settings","issues_dashboard_inp_optimization","issues_diff_based_label_updates","issues_expanded_file_types","issues_index_semantic_search","issues_lazy_load_comment_box_suggestions","issues_react_bots_timeline_pagination","issues_react_chrome_container_query_fix","issues_react_prohibit_title_fallback","issues_search_type_gql","landing_pages_ninetailed","landing_pages_web_vitals_tracking","lifecycle_label_name_updates","marketing_pages_search_explore_provider","memex_default_issue_create_repository","memex_live_update_hovercard","memex_mwl_filter_field_delimiter","merge_status_header_feedback","mission_control_retry_on_401","oauth_authorize_clickjacking_protection","open_agent_session_in_vscode_insiders","open_agent_session_in_vscode_stable","primer_react_css_has_selector_perf","primer_react_spinner_synchronize_animations","prs_conversations_react","prx_merge_status_button_alt_logic","pulls_sidebar_update","sample_network_conn_type","secret_scanning_pattern_alerts_link","session_logs_ungroup_reasoning_text","site_features_copilot_universe","site_homepage_collaborate_video","spark_prompt_secret_scanning","spark_server_connection_status","suppress_automated_browser_vitals","viewscreen_sandbox","webp_support","workbench_store_readonly"],"copilotApiOverrideUrl":"https://api.githubcopilot.com"}</script>
<script crossorigin="anonymous" type="application/javascript" src="./4_GAT_files/high-contrast-cookie-fed1d93364101384.js.download"></script>
<script crossorigin="anonymous" type="application/javascript" src="./4_GAT_files/wp-runtime-57eee5ffd7ed5697.js.download" defer="defer"></script>
<script crossorigin="anonymous" type="application/javascript" src="./4_GAT_files/85924-e131bec5f99667e1.js.download" defer="defer"></script>
<script crossorigin="anonymous" type="application/javascript" src="./4_GAT_files/28839-734cb6d8a7150172.js.download" defer="defer"></script>
<script crossorigin="anonymous" type="application/javascript" src="./4_GAT_files/34646-edd5528648c197b1.js.download" defer="defer"></script>
<script crossorigin="anonymous" type="application/javascript" src="./4_GAT_files/environment-774bb29273d50b01.js.download" defer="defer"></script>
<script crossorigin="anonymous" type="application/javascript" src="./4_GAT_files/runtime-helpers-3bb6f7d6b7a2f531.js.download" defer="defer"></script>
<script crossorigin="anonymous" type="application/javascript" src="./4_GAT_files/2966-f6796bfd155feae1.js.download" defer="defer"></script>
<script crossorigin="anonymous" type="application/javascript" src="./4_GAT_files/96232-69d46a31854353d4.js.download" defer="defer"></script>
<script crossorigin="anonymous" type="application/javascript" src="./4_GAT_files/41013-7a6deee6d6ff15eb.js.download" defer="defer"></script>
<script crossorigin="anonymous" type="application/javascript" src="./4_GAT_files/51210-3abb7238871a5b29.js.download" defer="defer"></script>
<script crossorigin="anonymous" type="application/javascript" src="./4_GAT_files/64247-7de91c52a8aca0eb.js.download" defer="defer"></script>
<script crossorigin="anonymous" type="application/javascript" src="./4_GAT_files/81683-1370179bf9bdc0f0.js.download" defer="defer"></script>
<script crossorigin="anonymous" type="application/javascript" src="./4_GAT_files/46740-6606b1026a237412.js.download" defer="defer"></script>
<script crossorigin="anonymous" type="application/javascript" src="./4_GAT_files/53102-f34caec7b425fe9f.js.download" defer="defer"></script>
<script crossorigin="anonymous" type="application/javascript" src="./4_GAT_files/github-elements-b24035aa51f284b4.js.download" defer="defer"></script>
<script crossorigin="anonymous" type="application/javascript" src="./4_GAT_files/element-registry-4da62566a66afd6d.js.download" defer="defer"></script>
<script crossorigin="anonymous" type="application/javascript" src="./4_GAT_files/react-core-2e47687b4310af0f.js.download" defer="defer"></script>
<script crossorigin="anonymous" type="application/javascript" src="./4_GAT_files/react-lib-a4cf89fce9a1300a.js.download" defer="defer"></script>

<script crossorigin="anonymous" type="application/javascript" src="./4_GAT_files/79039-2565b539a6ebc09b.js.download" defer="defer"></script>
<script crossorigin="anonymous" type="application/javascript" src="./4_GAT_files/61110-93cf7706e5dc8bfa.js.download" defer="defer"></script>
<script crossorigin="anonymous" type="application/javascript" src="./4_GAT_files/2887-47ac9a4b8862e6bf.js.download" defer="defer"></script>
<script crossorigin="anonymous" type="application/javascript" src="./4_GAT_files/26533-4df1172b25427069.js.download" defer="defer"></script>
<script crossorigin="anonymous" type="application/javascript" src="./4_GAT_files/45695-0f7265ae0cee4917.js.download" defer="defer"></script>
<script crossorigin="anonymous" type="application/javascript" src="./4_GAT_files/86483-92f38323ee870655.js.download" defer="defer"></script>
<script crossorigin="anonymous" type="application/javascript" src="./4_GAT_files/86659-ee83bc63b54d5ea9.js.download" defer="defer"></script>


<script crossorigin="anonymous" type="application/javascript" src="./4_GAT_files/2498-7a413c8209222158.js.download" defer="defer"></script>
<script crossorigin="anonymous" type="application/javascript" src="./4_GAT_files/61075-c79d53892670e6db.js.download" defer="defer"></script>
<script crossorigin="anonymous" type="application/javascript" src="./4_GAT_files/89627-fbbd31c43d83e10c.js.download" defer="defer"></script>
<script crossorigin="anonymous" type="application/javascript" src="./4_GAT_files/49029-32f138b26aa15ad8.js.download" defer="defer"></script>
<script crossorigin="anonymous" type="application/javascript" src="./4_GAT_files/99328-5e06da57c4622e21.js.download" defer="defer"></script>
<script crossorigin="anonymous" type="application/javascript" src="./4_GAT_files/behaviors-601e6ef118c0fa96.js.download" defer="defer"></script>
<link crossorigin="anonymous" media="all" rel="stylesheet" href="./4_GAT_files/react-core.d14f70276c05a07b.module.css">
<script crossorigin="anonymous" type="application/javascript" src="./4_GAT_files/4244-d5dcb589fae0ecbf.js.download" defer="defer"></script>
<script crossorigin="anonymous" type="application/javascript" src="./4_GAT_files/notifications-global-5bff6e56e8ce99b0.js.download" defer="defer"></script>
<script crossorigin="anonymous" type="application/javascript" src="./4_GAT_files/19262-313c6e4aa6bc1abc.js.download" defer="defer"></script>
<script crossorigin="anonymous" type="application/javascript" src="./4_GAT_files/13284-8b9feeae4ba8bcad.js.download" defer="defer"></script>
<script crossorigin="anonymous" type="application/javascript" src="./4_GAT_files/19930-d7474e22a5e73a83.js.download" defer="defer"></script>
<script crossorigin="anonymous" type="application/javascript" src="./4_GAT_files/codespaces-595fb1a71a7b3ae9.js.download" defer="defer"></script>
<script crossorigin="anonymous" type="application/javascript" src="./4_GAT_files/42890-dc8a427d0e01e2ea.js.download" defer="defer"></script>
<script crossorigin="anonymous" type="application/javascript" src="./4_GAT_files/70206-3ba10f6524cc3c66.js.download" defer="defer"></script>
<script crossorigin="anonymous" type="application/javascript" src="./4_GAT_files/repositories-bcdd7e391b80f360.js.download" defer="defer"></script>
<script crossorigin="anonymous" type="application/javascript" src="./4_GAT_files/34140-bd56b738d77cb446.js.download" defer="defer"></script>
<script crossorigin="anonymous" type="application/javascript" src="./4_GAT_files/code-menu-34c67a8e6616ccae.js.download" defer="defer"></script>
  
  




<script crossorigin="anonymous" type="application/javascript" src="./4_GAT_files/7463-f34e26efc84a7578.js.download" defer="defer"></script>
<script crossorigin="anonymous" type="application/javascript" src="./4_GAT_files/15272-00d6ef52b0f88c77.js.download" defer="defer"></script>
<script crossorigin="anonymous" type="application/javascript" src="./4_GAT_files/32769-329cba91f224b6ee.js.download" defer="defer"></script>
<script crossorigin="anonymous" type="application/javascript" src="./4_GAT_files/46148-0e40187c0246d203.js.download" defer="defer"></script>
<script crossorigin="anonymous" type="application/javascript" src="./4_GAT_files/91498-c4157c3b474acdfa.js.download" defer="defer"></script>
<script crossorigin="anonymous" type="application/javascript" src="./4_GAT_files/50938-5b5d5a0255a1f741.js.download" defer="defer"></script>
<script crossorigin="anonymous" type="application/javascript" src="./4_GAT_files/97519-def3535ce684b22d.js.download" defer="defer"></script>
<script crossorigin="anonymous" type="application/javascript" src="./4_GAT_files/60350-634e837b0830a241.js.download" defer="defer"></script>
<script crossorigin="anonymous" type="application/javascript" src="./4_GAT_files/24844-58b030e66defd270.js.download" defer="defer"></script>



<script crossorigin="anonymous" type="application/javascript" src="./4_GAT_files/3624-9329741a7029c94d.js.download" defer="defer"></script>
<script crossorigin="anonymous" type="application/javascript" src="./4_GAT_files/66231-1e6cc99727940110.js.download" defer="defer"></script>
<script crossorigin="anonymous" type="application/javascript" src="./4_GAT_files/36600-b3024c97643f0102.js.download" defer="defer"></script>
<script crossorigin="anonymous" type="application/javascript" src="./4_GAT_files/33684-8c0b5018276812ee.js.download" defer="defer"></script>
<script crossorigin="anonymous" type="application/javascript" src="./4_GAT_files/18222-498ac30c45b2263a.js.download" defer="defer"></script>
<script crossorigin="anonymous" type="application/javascript" src="./4_GAT_files/96755-8619cf5f811a4353.js.download" defer="defer"></script>
<script crossorigin="anonymous" type="application/javascript" src="./4_GAT_files/4916-50f2539dc1c5e1ee.js.download" defer="defer"></script>
<script crossorigin="anonymous" type="application/javascript" src="./4_GAT_files/24787-791cfb59d6b5d03c.js.download" defer="defer"></script>
<script crossorigin="anonymous" type="application/javascript" src="./4_GAT_files/17363-9d3b6dc38025c8e2.js.download" defer="defer"></script>
<script crossorigin="anonymous" type="application/javascript" src="./4_GAT_files/1097-b805267515cb2087.js.download" defer="defer"></script>
<script crossorigin="anonymous" type="application/javascript" src="./4_GAT_files/13333-2a2decebccdb5c1f.js.download" defer="defer"></script>
<script crossorigin="anonymous" type="application/javascript" src="./4_GAT_files/5289-9f28f567ca806541.js.download" defer="defer"></script>
<script crossorigin="anonymous" type="application/javascript" src="./4_GAT_files/78958-461fe5d876e46831.js.download" defer="defer"></script>
<script crossorigin="anonymous" type="application/javascript" src="./4_GAT_files/62653-b5fad92fb3c5b665.js.download" defer="defer"></script>
<script crossorigin="anonymous" type="application/javascript" src="./4_GAT_files/54990-adbf64c5251a4ce5.js.download" defer="defer"></script>
<script crossorigin="anonymous" type="application/javascript" src="./4_GAT_files/43164-4602b5c5c2046360.js.download" defer="defer"></script>
<script crossorigin="anonymous" type="application/javascript" src="./4_GAT_files/9371-07f06acb4933225b.js.download" defer="defer"></script>
<script crossorigin="anonymous" type="application/javascript" src="./4_GAT_files/49972-40a4b0a31a6433a6.js.download" defer="defer"></script>
<script crossorigin="anonymous" type="application/javascript" src="./4_GAT_files/87332-306841b1acf677a6.js.download" defer="defer"></script>
<script crossorigin="anonymous" type="application/javascript" src="./4_GAT_files/21384-d19cbe6625546351.js.download" defer="defer"></script>

<script crossorigin="anonymous" type="application/javascript" src="./4_GAT_files/26497-3db8876731818fda.js.download" defer="defer"></script>
<script crossorigin="anonymous" type="application/javascript" src="./4_GAT_files/61620-19f9abbaa22a2c62.js.download" defer="defer"></script>
<script crossorigin="anonymous" type="application/javascript" src="./4_GAT_files/45106-197ebc11e66a756f.js.download" defer="defer"></script>
<script crossorigin="anonymous" type="application/javascript" src="./4_GAT_files/61454-b09ac3d8eaae2866.js.download" defer="defer"></script>
<script crossorigin="anonymous" type="application/javascript" src="./4_GAT_files/code-view-6c4ab2ee3c5d5492.js.download" defer="defer"></script>
<link crossorigin="anonymous" media="all" rel="stylesheet" href="./4_GAT_files/primer-react-css.a2f41fc3ac05567f.module.css">
<link crossorigin="anonymous" media="all" rel="stylesheet" href="./4_GAT_files/87332.135fdd225a2878b8.module.css">
<link crossorigin="anonymous" media="all" rel="stylesheet" href="./4_GAT_files/61620.dde7b9d860abc7aa.module.css">
<link crossorigin="anonymous" media="all" rel="stylesheet" href="./4_GAT_files/61454.a9540430471e8db9.module.css">
<link crossorigin="anonymous" media="all" rel="stylesheet" href="./4_GAT_files/code-view.f42233b99c94b7c1.module.css">

  <script crossorigin="anonymous" type="application/javascript" src="./4_GAT_files/75999-af2e3ea53e208314.js.download" defer="defer"></script>
<script crossorigin="anonymous" type="application/javascript" src="./4_GAT_files/2766-add282d602854c7b.js.download" defer="defer"></script>
<script crossorigin="anonymous" type="application/javascript" src="./4_GAT_files/notifications-subscriptions-menu-fa313c8742125beb.js.download" defer="defer"></script>
<link crossorigin="anonymous" media="all" rel="stylesheet" href="./4_GAT_files/primer-react-css.a2f41fc3ac05567f.module.css">
<link crossorigin="anonymous" media="all" rel="stylesheet" href="./4_GAT_files/notifications-subscriptions-menu.7a1a03f633a0acdf.module.css">


  <title>gat-traffic-control/4_GAT.py at main · manishvat/gat-traffic-control · GitHub</title>



  <meta name="route-pattern" content="/:user_id/:repository" data-turbo-transient="">
  <meta name="route-controller" content="files" data-turbo-transient="">
  <meta name="route-action" content="disambiguate" data-turbo-transient="">
  <meta name="fetch-nonce" content="v2:389f721d-2a7f-88d8-e184-5446526f2675">

    
  <meta name="current-catalog-service-hash" content="f3abb0cc802f3d7b95fc8762b94bdcb13bf39634c40c357301c4aa1d67a256fb">


  <meta name="request-id" content="D7D4:4A3C6:136476A:15B8EE8:69CE8F58" data-pjax-transient="true"><meta name="html-safe-nonce" content="89a6070019283409df67cf487981be23dd9af2a5cb8be70d7b328d7a6b2e0229" data-pjax-transient="true"><meta name="visitor-payload" content="eyJyZWZlcnJlciI6Imh0dHBzOi8vZ2l0aHViLmNvbS9tYW5pc2h2YXQvZ2F0LXRyYWZmaWMtY29udHJvbC9ibG9iL21haW4vNF9HQVQucHkiLCJyZXF1ZXN0X2lkIjoiRDdENDo0QTNDNjoxMzY0NzZBOjE1QjhFRTg6NjlDRThGNTgiLCJ2aXNpdG9yX2lkIjoiMTI1NTIyMzQyNjg5MzU5MjQ2OCIsInJlZ2lvbl9lZGdlIjoiY2VudHJhbGluZGlhIiwicmVnaW9uX3JlbmRlciI6ImNlbnRyYWxpbmRpYSJ9" data-pjax-transient="true"><meta name="visitor-hmac" content="721e2e5e35cf5bb65a82dbfcf3dd0ef0c493beade7ef057b0dbb8d05657ed826" data-pjax-transient="true">


    <meta name="hovercard-subject-tag" content="repository:1169143393" data-turbo-transient="">


  <meta name="github-keyboard-shortcuts" content="repository,copilot" data-turbo-transient="true">
  

  <meta name="selected-link" value="repo_source" data-turbo-transient="">
  <link rel="assets" href="https://github.githubassets.com/">

    <meta name="google-site-verification" content="Apib7-x98H0j5cPqHWwSMm6dNU4GmODRoqxLiDzdx9I">

<meta name="octolytics-url" content="https://collector.github.com/github/collect">





  <meta name="analytics-location" content="/&lt;user-name&gt;/&lt;repo-name&gt;" data-turbo-transient="true">

  




    <meta name="user-login" content="">

  

    <meta name="viewport" content="width=device-width">

    

      <meta name="description" content="Contribute to manishvat/gat-traffic-control development by creating an account on GitHub.">

      <link rel="search" type="application/opensearchdescription+xml" href="https://github.com/opensearch.xml" title="GitHub">

    <link rel="fluid-icon" href="https://github.com/fluidicon.png" title="GitHub">
    <meta property="fb:app_id" content="1401488693436528">
    <meta name="apple-itunes-app" content="app-id=1477376905, app-argument=https://github.com/manishvat/gat-traffic-control">

      <meta name="twitter:image" content="https://opengraph.githubassets.com/b0ebd8383816cf939340eb0d8030e613fdcbdae85d320a1a3501594096820b5b/manishvat/gat-traffic-control"><meta name="twitter:site" content="@github"><meta name="twitter:card" content="summary_large_image"><meta name="twitter:title" content="GitHub - manishvat/gat-traffic-control"><meta name="twitter:description" content="Contribute to manishvat/gat-traffic-control development by creating an account on GitHub.">
  <meta property="og:image" content="https://opengraph.githubassets.com/b0ebd8383816cf939340eb0d8030e613fdcbdae85d320a1a3501594096820b5b/manishvat/gat-traffic-control"><meta property="og:image:alt" content="Contribute to manishvat/gat-traffic-control development by creating an account on GitHub."><meta property="og:image:width" content="1200"><meta property="og:image:height" content="600"><meta property="og:site_name" content="GitHub"><meta property="og:type" content="object"><meta property="og:title" content="GitHub - manishvat/gat-traffic-control"><meta property="og:url" content="https://github.com/manishvat/gat-traffic-control"><meta property="og:description" content="Contribute to manishvat/gat-traffic-control development by creating an account on GitHub.">
  




      <meta name="hostname" content="github.com">



        <meta name="expected-hostname" content="github.com">


  <meta http-equiv="x-pjax-version" content="b9472804f0bd008ae066c380a50f1c0231e50ab14de14dba40cbe124d9e6a8c4" data-turbo-track="reload">
  <meta http-equiv="x-pjax-csp-version" content="568c098497d98702bac1642a2a853732a047a6ced28eabd3e15d50041a890235" data-turbo-track="reload">
  <meta http-equiv="x-pjax-css-version" content="20fcfbab4ff75c6eb07df12ea46a99278a956a46f28b2d7e4e7233e3a2a3251e" data-turbo-track="reload">
  <meta http-equiv="x-pjax-js-version" content="6e37a45485664b4b7aa5d3c8687be9e013c747151d6279180be0c29e953bf49d" data-turbo-track="reload">

  <meta name="turbo-cache-control" content="no-preview" data-turbo-transient="">

      <meta name="turbo-cache-control" content="no-cache" data-turbo-transient="">

    <meta data-hydrostats="publish">

  <meta name="go-import" content="github.com/manishvat/gat-traffic-control git https://github.com/manishvat/gat-traffic-control.git">

  <meta name="octolytics-dimension-user_id" content="156544948"><meta name="octolytics-dimension-user_login" content="manishvat"><meta name="octolytics-dimension-repository_id" content="1169143393"><meta name="octolytics-dimension-repository_nwo" content="manishvat/gat-traffic-control"><meta name="octolytics-dimension-repository_public" content="true"><meta name="octolytics-dimension-repository_is_fork" content="false"><meta name="octolytics-dimension-repository_network_root_id" content="1169143393"><meta name="octolytics-dimension-repository_network_root_nwo" content="manishvat/gat-traffic-control">



    

    <meta name="turbo-body-classes" content="logged-out env-production page-responsive">
  <meta name="disable-turbo" content="false">


  <meta name="browser-stats-url" content="https://api.github.com/_private/browser/stats">

  <meta name="browser-errors-url" content="https://api.github.com/_private/browser/errors">

  <meta name="release" content="a9699bad040203070fa87afeb3117692220fbcd9">
  <meta name="ui-target" content="full">

  <link rel="mask-icon" href="https://github.githubassets.com/assets/pinned-octocat-093da3e6fa40.svg" color="#000000">
  <link rel="alternate icon" class="js-site-favicon" type="image/png" href="https://github.githubassets.com/favicons/favicon-dark.png">
  <link rel="icon" class="js-site-favicon" type="image/svg+xml" href="https://github.githubassets.com/favicons/favicon-dark.svg" data-base-href="https://github.githubassets.com/favicons/favicon">

<meta name="theme-color" content="#1e2327">
<meta name="color-scheme" content="light dark">


  <link rel="manifest" href="https://github.com/manifest.json" crossorigin="use-credentials">

  <link rel="stylesheet" type="text/css" href="./4_GAT_files/38963.f70dfea92b138b01.module.css" crossorigin="anonymous"><link rel="stylesheet" type="text/css" href="./4_GAT_files/70168.374ce4e648336309.module.css" crossorigin="anonymous"><link rel="stylesheet" type="text/css" href="./4_GAT_files/39007.fa471facf86f7401.module.css" crossorigin="anonymous"><style data-styled="active" data-styled-version="5.3.11"></style><link rel="stylesheet" type="text/css" href="./4_GAT_files/71849.38231ebf5311d902.module.css" crossorigin="anonymous"><style id="ms-consent-banner-main-styles">.w8hcgFksdo30C8w-bygqu{color:#000}.ydkKdaztSS0AeHWIeIHsQ a{color:#0067B8}.erL690_8JwUW-R4bJRcfl{background-color:#EBEBEB;border:none;color:#000}.erL690_8JwUW-R4bJRcfl:enabled:hover{color:#000;background-color:#DBDBDB;box-shadow:0px 4px 10px rgba(0,0,0,0.25);border:none}.erL690_8JwUW-R4bJRcfl:enabled:focus{background-color:#DBDBDB;box-shadow:0px 4px 10px rgba(0,0,0,0.25);border:2px solid #000}.erL690_8JwUW-R4bJRcfl:disabled{opacity:1;color:rgba(0,0,0,0.2);background-color:rgba(0,0,0,0.2);border:none}._1zNQOqxpBFSokeCLGi_hGr{border:none;background-color:#0067B8;color:#fff}._1zNQOqxpBFSokeCLGi_hGr:enabled:hover{color:#fff;background-color:#0067B8;box-shadow:0px 4px 10px rgba(0,0,0,0.25);border:none}._1zNQOqxpBFSokeCLGi_hGr:enabled:focus{background-color:#0067B8;box-shadow:0px 4px 10px rgba(0,0,0,0.25);border:2px solid #000}._1zNQOqxpBFSokeCLGi_hGr:disabled{opacity:1;color:rgba(0,0,0,0.2);background-color:rgba(0,120,215,0.2);border:none}._23tra1HsiiP6cT-Cka-ycB{position:relative;display:flex;z-index:9999;width:100%;background-color:#F2F2F2;justify-content:space-between;text-align:left}div[dir="rtl"]._23tra1HsiiP6cT-Cka-ycB{text-align:right}._1Upc2NjY8AlDn177YoVj0y{margin:0;padding-left:5%;padding-top:8px;padding-bottom:8px}div[dir="rtl"] ._1Upc2NjY8AlDn177YoVj0y{margin:0;padding:8px 5% 8px 0;float:none}._23tra1HsiiP6cT-Cka-ycB svg{fill:none;max-width:none;max-height:none}._1V_hlU-7jdtPiooHMu89BB{display:table-cell;padding:12px;width:24px;height:24px;font-family:Segoe UI, SegoeUI, Arial, sans-serif;font-style:normal;font-weight:normal;font-size:24px;line-height:0}.f6QKJD7fhSbnJLarTL-W-{display:table-cell;vertical-align:middle;padding:0;font-family:Segoe UI, SegoeUI, Arial, sans-serif;font-style:normal;font-weight:normal;font-size:13px;line-height:16px}.f6QKJD7fhSbnJLarTL-W- a{text-decoration:underline}._2j0fmugLb1FgYz6KPuB91w{display:inline-block;margin-left:5%;margin-right:5%;min-width:40%;min-width:calc((150px + 3 * 4px) * 2 + 150px);min-width:-webkit-fit-content;min-width:-moz-fit-content;min-width:fit-content;align-self:center;position:relative}._1XuCi2WhiqeWRUVp3pnFG3{margin:4px;padding:5px;min-width:150px;min-height:36px;vertical-align:top;cursor:pointer;font-family:Segoe UI, SegoeUI, Arial, sans-serif;font-style:normal;font-weight:normal;font-size:15px;line-height:20px;text-align:center}._1XuCi2WhiqeWRUVp3pnFG3:focus{box-sizing:border-box}._1XuCi2WhiqeWRUVp3pnFG3:disabled{cursor:not-allowed}._2bvsb3ubApyZ0UGoQA9O9T{display:block;position:fixed;z-index:10000;top:0;left:0;width:100%;height:100%;background-color:rgba(255,255,255,0.6);overflow:auto;text-align:left}div[dir="rtl"]._2bvsb3ubApyZ0UGoQA9O9T{text-align:right}div[dir="rtl"] ._2bvsb3ubApyZ0UGoQA9O9T{left:auto;right:0}.AFsJE948muYyzCMktdzuk{position:relative;top:8%;margin-bottom:40px;margin-left:auto;margin-right:auto;box-sizing:border-box;width:640px;background-color:#fff;border:1px solid #0067B8}._3kWyBRbW_dgnMiEyx06Fu4{float:right;z-index:1;margin:2px;padding:12px;border:none;cursor:pointer;font-family:Segoe UI, SegoeUI, Arial, sans-serif;font-style:normal;font-weight:normal;font-size:13px;line-height:13px;display:flex;align-items:center;text-align:center;color:#666;background-color:#fff}div[dir="rtl"] ._3kWyBRbW_dgnMiEyx06Fu4{margin:2px;padding:12px;float:left}.uCYvKvHXrhjNgflv1VqdD{position:static;margin-top:36px;margin-left:36px;margin-right:36px}._17pX1m9O_W--iZbDt3Ta5r{margin-top:0;margin-bottom:12px;font-family:Segoe UI, SegoeUI, Arial, sans-serif;font-style:normal;font-weight:600;font-size:20px;line-height:24px;text-transform:none}._1kBkHQ1V1wu3kl-YcLgUr6{height:446px;overflow:auto}._20_nXDf6uFs9Q6wxRXG-I-{margin-top:0;font-family:Segoe UI, SegoeUI, Arial, sans-serif;font-style:normal;font-weight:normal;font-size:15px;line-height:20px}._20_nXDf6uFs9Q6wxRXG-I- a{text-decoration:underline}dl._2a0NH_GDQEQe5Ynfo7suVH{margin-top:36px;margin-bottom:0;padding:0;list-style:none;text-transform:none}dt._3j_LCPv7fyXv3A8FIXVwZ4{margin-top:20px;float:none;font-family:Segoe UI, SegoeUI, Arial, sans-serif;font-style:normal;font-weight:600;font-size:18px;line-height:24px;list-style:none}.k-vxTGFbdq1aOZB2HHpjh{margin:0;padding:0;border:none}._2Bucyy75c_ogoU1g-liB5R{margin:0;padding:0;border-bottom:none;font-family:Segoe UI, SegoeUI, Arial, sans-serif;font-style:normal;font-weight:600;font-size:18px;line-height:24px;text-transform:none}._63gwfzV8dclrsl2cfd90r{display:inline-block;margin-top:0;margin-bottom:13px;font-family:Segoe UI, SegoeUI, Arial, sans-serif;font-style:normal;font-weight:normal;font-size:15px;line-height:20px}._1l8wM_4mRYGz3Iu7l3BZR7{display:block}._2UE03QS02aZGkslegN_F-i{display:inline-block;position:relative;left:5px;margin-bottom:13px;margin-right:34px;padding:3px}div[dir="rtl"] ._2UE03QS02aZGkslegN_F-i{margin:0 0 13px 34px;padding:3px;float:none}div[dir="rtl"] ._2UE03QS02aZGkslegN_F-i{left:auto;right:5px}._23tra1HsiiP6cT-Cka-ycB *::before,._2bvsb3ubApyZ0UGoQA9O9T *::before,._23tra1HsiiP6cT-Cka-ycB *::after,._2bvsb3ubApyZ0UGoQA9O9T *::after{box-sizing:inherit}._1HSFn0HzGo6w4ADApV8-c4{outline:2px solid rgba(0,0,0,0.8)}input[type="radio"]._1dp8Vp5m3HwAqGx8qBmFV2{display:inline-block;position:relative;margin-top:0;margin-left:0;margin-right:0;height:0;width:0;border-radius:0;cursor:pointer;outline:none;box-sizing:border-box;-webkit-appearance:none;-moz-appearance:none;appearance:none}input[type="radio"]._1dp8Vp5m3HwAqGx8qBmFV2+label::before{display:block;position:absolute;top:5px;left:3px;height:19px;width:19px;content:"";border-radius:50%;border:1px solid #000;background-color:#fff}div[dir="rtl"] input[type="radio"]._1dp8Vp5m3HwAqGx8qBmFV2+label::before{left:auto;right:3px}input[type="radio"]._1dp8Vp5m3HwAqGx8qBmFV2:not(:disabled)+label:hover::before{border:1px solid #0067B8}input[type="radio"]._1dp8Vp5m3HwAqGx8qBmFV2:not(:disabled)+label:hover::after{display:block;position:absolute;top:10px;left:8px;height:9px;width:9px;content:"";border-radius:50%;background-color:rgba(0,0,0,0.8)}div[dir="rtl"] input[type="radio"]._1dp8Vp5m3HwAqGx8qBmFV2:not(:disabled)+label:hover::after{left:auto;right:8px}input[type="radio"]._1dp8Vp5m3HwAqGx8qBmFV2:not(:disabled)+label:focus::before{border:1px solid #0067B8}input[type="radio"]._1dp8Vp5m3HwAqGx8qBmFV2:not(:disabled)+label:focus::after{display:block;position:absolute;top:10px;left:8px;height:9px;width:9px;content:"";border-radius:50%;background-color:#000}div[dir="rtl"] input[type="radio"]._1dp8Vp5m3HwAqGx8qBmFV2:not(:disabled)+label:focus::after{left:auto;right:8px}input[type="radio"]._1dp8Vp5m3HwAqGx8qBmFV2:checked+label::after{display:block;position:absolute;top:10px;left:8px;height:9px;width:9px;content:"";border-radius:50%;background-color:#000}div[dir="rtl"] input[type="radio"]._1dp8Vp5m3HwAqGx8qBmFV2:checked+label::after{left:auto;right:8px}input[type="radio"]._1dp8Vp5m3HwAqGx8qBmFV2:disabled+label{cursor:not-allowed}input[type="radio"]._1dp8Vp5m3HwAqGx8qBmFV2:disabled+label::before{border:1px solid rgba(0,0,0,0.2);background-color:rgba(0,0,0,0.2)}._3RJzeL3l9Rl_lAQEm6VwdX{display:block;position:static;float:right;margin-top:0;margin-bottom:0;margin-left:19px;margin-right:0;padding-top:0;padding-bottom:0;padding-left:8px;padding-right:0;width:80%;width:calc(100% - 19px);font-family:Segoe UI, SegoeUI, Arial, sans-serif;font-style:normal;font-weight:normal;font-size:15px;line-height:20px;text-transform:none;cursor:pointer;box-sizing:border-box}div[dir="rtl"] ._3RJzeL3l9Rl_lAQEm6VwdX{margin:0 19px 0 0;padding:0 8px 0 0;float:left}.nohp3sIG12ZBhzcMnPala{margin-top:20px;margin-bottom:48px}._2uhaEsmeotZ3P-M0AXo2kF{padding:0;width:278px;height:36px;cursor:pointer;font-family:Segoe UI, SegoeUI, Arial, sans-serif;font-style:normal;font-weight:normal;font-size:15px;line-height:20px;text-align:center}._2uhaEsmeotZ3P-M0AXo2kF:focus{box-sizing:border-box}._2uhaEsmeotZ3P-M0AXo2kF:disabled{cursor:not-allowed}._3tOu1FJ59c_xz_PmI1lKV5{float:right;padding:0;width:278px;height:36px;cursor:pointer;font-family:Segoe UI, SegoeUI, Arial, sans-serif;font-style:normal;font-weight:normal;font-size:15px;line-height:20px;text-align:center}._3tOu1FJ59c_xz_PmI1lKV5:focus{box-sizing:border-box}._3tOu1FJ59c_xz_PmI1lKV5:disabled{cursor:not-allowed}div[dir="rtl"] ._3tOu1FJ59c_xz_PmI1lKV5{margin:0;padding:0;float:left}@media only screen and (max-width: 768px){._2j0fmugLb1FgYz6KPuB91w,._1Upc2NjY8AlDn177YoVj0y{padding-top:8px;padding-bottom:12px;padding-left:3.75%;padding-right:3.75%;margin:0;width:92.5%}._23tra1HsiiP6cT-Cka-ycB{display:block}._1XuCi2WhiqeWRUVp3pnFG3{margin-bottom:8px;margin-left:0;margin-right:0;width:100%}._2bvsb3ubApyZ0UGoQA9O9T{overflow:hidden}.AFsJE948muYyzCMktdzuk{top:1.8%;width:93.33%;height:96.4%;overflow:hidden}.uCYvKvHXrhjNgflv1VqdD{margin-top:24px;margin-left:24px;margin-right:24px;height:100%}._1kBkHQ1V1wu3kl-YcLgUr6{height:62%;height:calc(100% - 188px);min-height:50%}._2uhaEsmeotZ3P-M0AXo2kF{width:100%}._3tOu1FJ59c_xz_PmI1lKV5{margin-bottom:12px;margin-left:0;width:100%}div[dir="rtl"] ._3tOu1FJ59c_xz_PmI1lKV5{margin:0 0 12px 0;padding:0;float:none}}@media only screen and (max-width: 768px) and (orientation: landscape), only screen and (max-height: 260px), only screen and (max-width: 340px){.AFsJE948muYyzCMktdzuk{overflow:auto}}@media only screen and (max-height: 260px), only screen and (max-width: 340px){._1XuCi2WhiqeWRUVp3pnFG3{min-width:0}._3kWyBRbW_dgnMiEyx06Fu4{padding:3%}.uCYvKvHXrhjNgflv1VqdD{margin-top:3%;margin-left:3%;margin-right:3%}._17pX1m9O_W--iZbDt3Ta5r{margin-bottom:3%}._1kBkHQ1V1wu3kl-YcLgUr6{height:calc(79% - 64px)}.nohp3sIG12ZBhzcMnPala{margin-top:5%;margin-bottom:10%}._3tOu1FJ59c_xz_PmI1lKV5{margin-bottom:3%}div[dir="rtl"] ._3tOu1FJ59c_xz_PmI1lKV5{margin:0 0 3% 0;padding:0;float:none}}
</style><style type="text/css" id="ms-consent-banner-theme-styles">._23tra1HsiiP6cT-Cka-ycB {
            background-color: #24292f !important;
        }.w8hcgFksdo30C8w-bygqu {
            color: #ffffff !important;
        }.ydkKdaztSS0AeHWIeIHsQ a {
            color: #d8b9ff !important;
        }._2bvsb3ubApyZ0UGoQA9O9T {
            background-color: rgba(23, 23, 23, 0.8) !important;
        }.AFsJE948muYyzCMktdzuk {
            background-color: #24292f !important;
            border: 1px solid #d8b9ff !important;
        }._3kWyBRbW_dgnMiEyx06Fu4 {
            color: #d8b9ff !important;
            background-color: #24292f !important;
        }._1zNQOqxpBFSokeCLGi_hGr {
            border: 1px solid #ffffff !important;
            background-color: #ffffff !important;
            color: #1f2328 !important;
        }._1zNQOqxpBFSokeCLGi_hGr:enabled:hover {
            color: #1f2328 !important;
            background-color: #d8b9ff !important;
            box-shadow: none !important;
            border: 1px solid transparent !important;
        }._1zNQOqxpBFSokeCLGi_hGr:enabled:focus {
            background-color: #d8b9ff !important;
            box-shadow: none !important;
            border: 2px solid #ffffff !important;
        }._1zNQOqxpBFSokeCLGi_hGr:disabled {
            opacity: 0.5 !important;
            color: #1f2328 !important;
            background-color: #ffffff !important;
            border: 1px solid transparent !important;
        }.erL690_8JwUW-R4bJRcfl {
            border: 1px solid #eaeef2 !important;
            background-color: #32383f !important;
            color: #ffffff !important;
        }.erL690_8JwUW-R4bJRcfl:enabled:hover {
            color: #ffffff !important;
            background-color: #24292f !important;
            box-shadow: none !important;
            border: 1px solid #ffffff !important;
        }.erL690_8JwUW-R4bJRcfl:enabled:focus {
            background-color: #24292f !important;
            box-shadow: none !important;
            border: 2px solid #6e7781 !important;
        }.erL690_8JwUW-R4bJRcfl:disabled {
            opacity: 0.5 !important;
            color: #ffffff !important;
            background-color: #424a53 !important;
            border: 1px solid #6e7781 !important;
        }input[type="radio"]._1dp8Vp5m3HwAqGx8qBmFV2 + label::before {
            border: 1px solid #d8b9ff !important;
            background-color: #24292f !important;
        }._1HSFn0HzGo6w4ADApV8-c4 {
            outline: 2px solid #ffffff !important;
        }input[type="radio"]._1dp8Vp5m3HwAqGx8qBmFV2:checked + label::after {
            background-color: #d8b9ff !important;
        }input[type="radio"]._1dp8Vp5m3HwAqGx8qBmFV2 + label:hover::before {
            border: 1px solid #ffffff !important;
        }input[type="radio"]._1dp8Vp5m3HwAqGx8qBmFV2 + label:hover::after {
            background-color: #ffffff !important;
        }input[type="radio"]._1dp8Vp5m3HwAqGx8qBmFV2 + label:focus::before {
            border: 1px solid #ffffff !important;
        }input[type="radio"]._1dp8Vp5m3HwAqGx8qBmFV2 + label:focus::after {
            background-color: #d8b9ff !important;
        }input[type="radio"]._1dp8Vp5m3HwAqGx8qBmFV2:disabled + label::before {
            border: 1px solid rgba(227, 227, 227, 0.2) !important;
            background-color: rgba(227, 227, 227, 0.2) !important;
        }</style><link rel="stylesheet" type="text/css" href="./4_GAT_files/61975.bf2bb705802183ac.module.css" crossorigin="anonymous"></head>

  <body class="logged-out env-production page-responsive" style="word-wrap: break-word;" data-dialog-scroll-optimized="">
    <div data-turbo-body="" class="logged-out env-production page-responsive" style="word-wrap: break-word;">
      <div id="__primerPortalRoot__" role="region" style="z-index: 1000; position: absolute; width: 100%;" data-turbo-permanent=""></div>
      

    <div class="position-relative header-wrapper js-header-wrapper ">
      <a href="https://github.com/manishvat/gat-traffic-control/blob/main/4_GAT.py#start-of-content" data-skip-target-assigned="false" class="px-2 tmp-py-4 color-bg-accent-emphasis color-fg-on-emphasis show-on-focus js-skip-to-content">Skip to content</a>

      <span data-view-component="true" class="progress-pjax-loader Progress position-fixed width-full">
    <span style="width: 0%;" data-view-component="true" class="Progress-item progress-pjax-loader-bar left-0 top-0 color-bg-accent-emphasis"></span>
</span>      
      
      <link crossorigin="anonymous" media="all" rel="stylesheet" href="./4_GAT_files/primer-react-css.a2f41fc3ac05567f.module.css">
<link crossorigin="anonymous" media="all" rel="stylesheet" href="./4_GAT_files/keyboard-shortcuts-dialog.ea2b62f533dc9058.module.css">

<react-partial partial-name="keyboard-shortcuts-dialog" data-ssr="false" data-attempted-ssr="false" data-react-profiling="false" data-catalyst="" class="loaded">
  
  <script type="application/json" data-target="react-partial.embeddedData">{"props":{"docsUrl":"https://docs.github.com/get-started/accessibility/keyboard-shortcuts"}}</script>
  <div data-target="react-partial.reactRoot"><div class="d-none"></div><script type="application/json" id="__PRIMER_DATA__r_2___">{"resolvedServerColorMode":"night"}</script></div>
</react-partial>





      

          

              
<script crossorigin="anonymous" type="application/javascript" src="./4_GAT_files/46752-46c707717fcbe6a9.js.download" defer="defer"></script>
<script crossorigin="anonymous" type="application/javascript" src="./4_GAT_files/sessions-d6380759dfce7642.js.download" defer="defer"></script>

<style>
  /* Override primer focus outline color for marketing header dropdown links for better contrast */
  [data-color-mode="light"] .HeaderMenu-dropdown-link:focus-visible,
  [data-color-mode="light"] .HeaderMenu-trailing-link a:focus-visible {
    outline-color: var(--color-accent-fg);
  }
</style>

<header class="HeaderMktg header-logged-out js-details-container js-header Details f4 tmp-py-3" role="banner" data-is-top="true" data-color-mode="auto" data-light-theme="light" data-dark-theme="dark">
  <h2 class="sr-only">Navigation Menu</h2>

  <button type="button" class="HeaderMktg-backdrop d-lg-none border-0 position-fixed top-0 left-0 width-full height-full js-details-target" aria-label="Toggle navigation">
    <span class="d-none">Toggle navigation</span>
  </button>

  <div class="d-flex flex-column flex-lg-row flex-items-center tmp-px-3 tmp-px-md-4 tmp-px-lg-5 height-full position-relative z-1">
    <div class="d-flex flex-justify-between flex-items-center width-full width-lg-auto">
      <div class="flex-1">
        <button aria-label="Toggle navigation" aria-expanded="false" type="button" data-view-component="true" class="js-details-target js-nav-padding-recalculate js-header-menu-toggle Button--link Button--medium Button d-lg-none color-fg-inherit p-1 tmp-p-1">  <span class="Button-content">
    <span class="Button-label"><div class="HeaderMenu-toggle-bar rounded my-1"></div>
            <div class="HeaderMenu-toggle-bar rounded my-1"></div>
            <div class="HeaderMenu-toggle-bar rounded my-1"></div></span>
  </span>
</button>
      </div>

      <a class="tmp-mr-lg-3 color-fg-inherit flex-order-2 js-prevent-focus-on-mobile-nav" href="https://github.com/" aria-label="Homepage" data-analytics-event="{&quot;category&quot;:&quot;Marketing nav&quot;,&quot;action&quot;:&quot;click to go to homepage&quot;,&quot;label&quot;:&quot;ref_page:Marketing;ref_cta:Logomark;ref_loc:Header&quot;}">
        <svg height="32" aria-hidden="true" viewBox="0 0 24 24" version="1.1" width="32" data-view-component="true" class="octicon octicon-mark-github">
    <path d="M10.226 17.284c-2.965-.36-5.054-2.493-5.054-5.256 0-1.123.404-2.336 1.078-3.144-.292-.741-.247-2.314.09-2.965.898-.112 2.111.36 2.83 1.01.853-.269 1.752-.404 2.853-.404 1.1 0 1.999.135 2.807.382.696-.629 1.932-1.1 2.83-.988.315.606.36 2.179.067 2.942.72.854 1.101 2 1.101 3.167 0 2.763-2.089 4.852-5.098 5.234.763.494 1.28 1.572 1.28 2.807v2.336c0 .674.561 1.056 1.235.786 4.066-1.55 7.255-5.615 7.255-10.646C23.5 6.188 18.334 1 11.978 1 5.62 1 .5 6.188.5 12.545c0 4.986 3.167 9.12 7.435 10.669.606.225 1.19-.18 1.19-.786V20.63a2.9 2.9 0 0 1-1.078.224c-1.483 0-2.359-.808-2.987-2.313-.247-.607-.517-.966-1.034-1.033-.27-.023-.359-.135-.359-.27 0-.27.45-.471.898-.471.652 0 1.213.404 1.797 1.235.45.651.921.943 1.483.943.561 0 .92-.202 1.437-.719.382-.381.674-.718.944-.943"></path>
</svg>
      </a>

      <div class="d-flex flex-1 flex-order-2 text-right d-lg-none gap-2 flex-justify-end">
          <a href="https://github.com/login?return_to=https%3A%2F%2Fgithub.com%2Fmanishvat%2Fgat-traffic-control" class="HeaderMenu-link HeaderMenu-button d-inline-flex f5 no-underline border color-border-default rounded-2 px-2 py-1 color-fg-inherit js-prevent-focus-on-mobile-nav" data-hydro-click="{&quot;event_type&quot;:&quot;authentication.click&quot;,&quot;payload&quot;:{&quot;location_in_page&quot;:&quot;site header menu&quot;,&quot;repository_id&quot;:null,&quot;auth_type&quot;:&quot;SIGN_UP&quot;,&quot;originating_url&quot;:&quot;https://github.com/manishvat/gat-traffic-control&quot;,&quot;user_id&quot;:null}}" data-hydro-click-hmac="382b63272136199597428c7c44be78cadb95f433df7bce65c99454363df4fc0a" data-analytics-event="{&quot;category&quot;:&quot;Marketing nav&quot;,&quot;action&quot;:&quot;click to Sign in&quot;,&quot;label&quot;:&quot;ref_page:Marketing;ref_cta:Sign in;ref_loc:Header&quot;}">
            Sign in
          </a>
              <div class="AppHeader-appearanceSettings">
    <react-partial-anchor data-catalyst="">
      <button data-target="react-partial-anchor.anchor" id="icon-button-ecacd1d1-7654-402d-affe-9f980acec43f" aria-labelledby="tooltip-b3dd38c5-b652-4018-bc9e-5d115ae0d622" type="button" data-view-component="true" class="Button Button--iconOnly Button--invisible Button--medium AppHeader-button HeaderMenu-link border" aria-expanded="false" aria-haspopup="true">  <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-sliders Button-visual">
    <path d="M15 2.75a.75.75 0 0 1-.75.75h-4a.75.75 0 0 1 0-1.5h4a.75.75 0 0 1 .75.75Zm-8.5.75v1.25a.75.75 0 0 0 1.5 0v-4a.75.75 0 0 0-1.5 0V2H1.75a.75.75 0 0 0 0 1.5H6.5Zm1.25 5.25a.75.75 0 0 0 0-1.5h-6a.75.75 0 0 0 0 1.5h6ZM15 8a.75.75 0 0 1-.75.75H11.5V10a.75.75 0 1 1-1.5 0V6a.75.75 0 0 1 1.5 0v1.25h2.75A.75.75 0 0 1 15 8Zm-9 5.25v-2a.75.75 0 0 0-1.5 0v1.25H1.75a.75.75 0 0 0 0 1.5H4.5v1.25a.75.75 0 0 0 1.5 0v-2Zm9 0a.75.75 0 0 1-.75.75h-6a.75.75 0 0 1 0-1.5h6a.75.75 0 0 1 .75.75Z"></path>
</svg>
</button><tool-tip id="tooltip-b3dd38c5-b652-4018-bc9e-5d115ae0d622" for="icon-button-ecacd1d1-7654-402d-affe-9f980acec43f" popover="manual" data-direction="s" data-type="label" data-view-component="true" class="sr-only position-absolute" aria-hidden="true" role="tooltip"><template shadowrootmode="open"><style>
      :host {
        --tooltip-top: var(--tool-tip-position-top, 0);
        --tooltip-left: var(--tool-tip-position-left, 0);
        padding: var(--overlay-paddingBlock-condensed) var(--overlay-padding-condensed) !important;
        font: var(--text-body-shorthand-small);
        color: var(--tooltip-fgColor, var(--fgColor-onEmphasis)) !important;
        text-align: center;
        text-decoration: none;
        text-shadow: none;
        text-transform: none;
        letter-spacing: normal;
        word-wrap: break-word;
        white-space: pre;
        background: var(--tooltip-bgColor, var(--bgColor-emphasis)) !important;
        border-radius: var(--borderRadius-medium);
        border: 0 !important;
        opacity: 0;
        max-width: min(var(--overlay-width-small), 100vw);
        word-wrap: break-word;
        white-space: normal;
        width: max-content !important;
        inset: var(--tooltip-top) auto auto var(--tooltip-left) !important;
        overflow: visible !important;
        text-wrap: balance;
      }

      :host(:is(.tooltip-n, .tooltip-nw, .tooltip-ne)) {
        --tooltip-top: calc(var(--tool-tip-position-top, 0) - var(--overlay-offset, 0.25rem));
        --tooltip-left: var(--tool-tip-position-left);
      }

      :host(:is(.tooltip-s, .tooltip-sw, .tooltip-se)) {
        --tooltip-top: calc(var(--tool-tip-position-top, 0) + var(--overlay-offset, 0.25rem));
        --tooltip-left: var(--tool-tip-position-left);
      }

      :host(.tooltip-w) {
        --tooltip-top: var(--tool-tip-position-top);
        --tooltip-left: calc(var(--tool-tip-position-left, 0) - var(--overlay-offset, 0.25rem));
      }

      :host(.tooltip-e) {
        --tooltip-top: var(--tool-tip-position-top);
        --tooltip-left: calc(var(--tool-tip-position-left, 0) + var(--overlay-offset, 0.25rem));
      }

      :host:after{
        position: absolute;
        display: block;
        right: 0;
        left: 0;
        height: var(--overlay-offset, 0.25rem);
        content: "";
      }

      :host(.tooltip-s):after,
      :host(.tooltip-se):after,
      :host(.tooltip-sw):after {
        bottom: 100%
      }

      :host(.tooltip-n):after,
      :host(.tooltip-ne):after,
      :host(.tooltip-nw):after {
        top: 100%;
      }

      @keyframes tooltip-appear {
        from {
          opacity: 0;
        }
        to {
          opacity: 1;
        }
      }

      :host(:popover-open),
      :host(:popover-open):before {
        animation-name: tooltip-appear;
        animation-duration: .1s;
        animation-fill-mode: forwards;
        animation-timing-function: ease-in;
      }

      :host(.\:popover-open) {
        animation-name: tooltip-appear;
        animation-duration: .1s;
        animation-fill-mode: forwards;
        animation-timing-function: ease-in;
      }

      @media (forced-colors: active) {
        :host {
          outline: solid 1px transparent;
        }

        :host:before {
          display: none;
        }
      }
    </style><slot></slot></template>Appearance settings</tool-tip>

      
    
        <link crossorigin="anonymous" media="all" rel="stylesheet" href="./4_GAT_files/primer-react-css.a2f41fc3ac05567f.module.css">
<link crossorigin="anonymous" media="all" rel="stylesheet" href="./4_GAT_files/appearance-settings.6f14ff9973550c9e.module.css">

<react-partial partial-name="appearance-settings" data-ssr="false" data-attempted-ssr="false" data-react-profiling="false" data-catalyst="" class="loaded">
  
  <script type="application/json" data-target="react-partial.embeddedData">{"props":{}}</script>
  <div data-target="react-partial.reactRoot"><script type="application/json" id="__PRIMER_DATA__r_a___">{"resolvedServerColorMode":"night"}</script></div>
</react-partial>


      </react-partial-anchor>
  </div>

      </div>
    </div>


    <div class="HeaderMenu js-header-menu height-fit position-lg-relative d-lg-flex flex-column flex-auto top-0">
      <div class="HeaderMenu-wrapper d-flex flex-column flex-self-start flex-lg-row flex-auto rounded rounded-lg-0">
            <link crossorigin="anonymous" media="all" rel="stylesheet" href="./4_GAT_files/primer-react-css.a2f41fc3ac05567f.module.css">
<link crossorigin="anonymous" media="all" rel="stylesheet" href="./4_GAT_files/marketing-navigation.5b4138cab2f44179.module.css">

<react-partial partial-name="marketing-navigation" data-ssr="true" data-attempted-ssr="true" data-react-profiling="false" data-catalyst="" class="loaded">
  
  <script type="application/json" data-target="react-partial.embeddedData">{"props":{"should_use_dotcom_links":true}}</script>
  <div data-target="react-partial.reactRoot"><nav class="MarketingNavigation-module__nav__W0KYY" aria-label="Global"><ul class="MarketingNavigation-module__list__tFbMb"><li><div class="NavDropdown-module__container__l2YeI js-details-container js-header-menu-item"><button type="button" class="NavDropdown-module__button__PEHWX js-details-target" aria-expanded="false">Platform<svg aria-hidden="true" focusable="false" class="octicon octicon-chevron-right NavDropdown-module__buttonIcon__Tkl8_" viewBox="0 0 16 16" width="16" height="16" fill="currentColor" display="inline-block" overflow="visible" style="vertical-align:text-bottom"><path d="M6.22 3.22a.75.75 0 0 1 1.06 0l4.25 4.25a.75.75 0 0 1 0 1.06l-4.25 4.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042L9.94 8 6.22 4.28a.75.75 0 0 1 0-1.06Z"></path></svg></button><div class="NavDropdown-module__dropdown__xm1jd"><ul class="NavDropdown-module__list__zuCgG"><li><div class="NavGroup-module__group__W8SqJ"><span class="NavGroup-module__title__Wzxz2">AI CODE CREATION</span><ul class="NavGroup-module__list__UCOFy"><li><a href="https://github.com/features/copilot" data-analytics-event="{&quot;action&quot;:&quot;github_copilot&quot;,&quot;tag&quot;:&quot;link&quot;,&quot;context&quot;:&quot;platform&quot;,&quot;location&quot;:&quot;navbar&quot;,&quot;label&quot;:&quot;github_copilot_link_platform_navbar&quot;}" class="NavLink-module__link__EG3d4"><div class="NavLink-module__text__XvpLQ"><svg aria-hidden="true" focusable="false" class="octicon octicon-copilot NavLink-module__icon__ltGNM" viewBox="0 0 24 24" width="24" height="24" fill="currentColor" display="inline-block" overflow="visible" style="vertical-align:text-bottom"><path d="M23.922 16.992c-.861 1.495-5.859 5.023-11.922 5.023-6.063 0-11.061-3.528-11.922-5.023A.641.641 0 0 1 0 16.736v-2.869a.841.841 0 0 1 .053-.22c.372-.935 1.347-2.292 2.605-2.656.167-.429.414-1.055.644-1.517a10.195 10.195 0 0 1-.052-1.086c0-1.331.282-2.499 1.132-3.368.397-.406.89-.717 1.474-.952 1.399-1.136 3.392-2.093 6.122-2.093 2.731 0 4.767.957 6.166 2.093.584.235 1.077.546 1.474.952.85.869 1.132 2.037 1.132 3.368 0 .368-.014.733-.052 1.086.23.462.477 1.088.644 1.517 1.258.364 2.233 1.721 2.605 2.656a.832.832 0 0 1 .053.22v2.869a.641.641 0 0 1-.078.256ZM12.172 11h-.344a4.323 4.323 0 0 1-.355.508C10.703 12.455 9.555 13 7.965 13c-1.725 0-2.989-.359-3.782-1.259a2.005 2.005 0 0 1-.085-.104L4 11.741v6.585c1.435.779 4.514 2.179 8 2.179 3.486 0 6.565-1.4 8-2.179v-6.585l-.098-.104s-.033.045-.085.104c-.793.9-2.057 1.259-3.782 1.259-1.59 0-2.738-.545-3.508-1.492a4.323 4.323 0 0 1-.355-.508h-.016.016Zm.641-2.935c.136 1.057.403 1.913.878 2.497.442.544 1.134.938 2.344.938 1.573 0 2.292-.337 2.657-.751.384-.435.558-1.15.558-2.361 0-1.14-.243-1.847-.705-2.319-.477-.488-1.319-.862-2.824-1.025-1.487-.161-2.192.138-2.533.529-.269.307-.437.808-.438 1.578v.021c0 .265.021.562.063.893Zm-1.626 0c.042-.331.063-.628.063-.894v-.02c-.001-.77-.169-1.271-.438-1.578-.341-.391-1.046-.69-2.533-.529-1.505.163-2.347.537-2.824 1.025-.462.472-.705 1.179-.705 2.319 0 1.211.175 1.926.558 2.361.365.414 1.084.751 2.657.751 1.21 0 1.902-.394 2.344-.938.475-.584.742-1.44.878-2.497Z"></path><path d="M14.5 14.25a1 1 0 0 1 1 1v2a1 1 0 0 1-2 0v-2a1 1 0 0 1 1-1Zm-5 0a1 1 0 0 1 1 1v2a1 1 0 0 1-2 0v-2a1 1 0 0 1 1-1Z"></path></svg><span class="NavLink-module__title__Q7t0p">GitHub Copilot</span><span class="NavLink-module__subtitle__X4gkW">Write better code with AI</span></div></a></li><li><a href="https://github.com/features/spark" data-analytics-event="{&quot;action&quot;:&quot;github_spark&quot;,&quot;tag&quot;:&quot;link&quot;,&quot;context&quot;:&quot;platform&quot;,&quot;location&quot;:&quot;navbar&quot;,&quot;label&quot;:&quot;github_spark_link_platform_navbar&quot;}" class="NavLink-module__link__EG3d4"><div class="NavLink-module__text__XvpLQ"><svg aria-hidden="true" focusable="false" class="octicon octicon-sparkle-fill NavLink-module__icon__ltGNM" viewBox="0 0 24 24" width="24" height="24" fill="currentColor" display="inline-block" overflow="visible" style="vertical-align:text-bottom"><path d="M11.296 1.924c.24-.656 1.168-.656 1.408 0l.717 1.958a11.25 11.25 0 0 0 6.697 6.697l1.958.717c.657.24.657 1.168 0 1.408l-1.958.717a11.25 11.25 0 0 0-6.697 6.697l-.717 1.958c-.24.657-1.168.657-1.408 0l-.717-1.958a11.25 11.25 0 0 0-6.697-6.697l-1.958-.717c-.656-.24-.656-1.168 0-1.408l1.958-.717a11.25 11.25 0 0 0 6.697-6.697l.717-1.958Z"></path></svg><span class="NavLink-module__title__Q7t0p">GitHub Spark</span><span class="NavLink-module__subtitle__X4gkW">Build and deploy intelligent apps</span></div></a></li><li><a href="https://github.com/features/models" data-analytics-event="{&quot;action&quot;:&quot;github_models&quot;,&quot;tag&quot;:&quot;link&quot;,&quot;context&quot;:&quot;platform&quot;,&quot;location&quot;:&quot;navbar&quot;,&quot;label&quot;:&quot;github_models_link_platform_navbar&quot;}" class="NavLink-module__link__EG3d4"><div class="NavLink-module__text__XvpLQ"><svg aria-hidden="true" focusable="false" class="octicon octicon-ai-model NavLink-module__icon__ltGNM" viewBox="0 0 24 24" width="24" height="24" fill="currentColor" display="inline-block" overflow="visible" style="vertical-align:text-bottom"><path d="M19.375 8.5a3.25 3.25 0 1 1-3.163 4h-3a3.252 3.252 0 0 1-4.443 2.509L7.214 17.76a3.25 3.25 0 1 1-1.342-.674l1.672-2.957A3.238 3.238 0 0 1 6.75 12c0-.907.371-1.727.97-2.316L6.117 6.846A3.253 3.253 0 0 1 1.875 3.75a3.25 3.25 0 1 1 5.526 2.32l1.603 2.836A3.25 3.25 0 0 1 13.093 11h3.119a3.252 3.252 0 0 1 3.163-2.5ZM10 10.25a1.75 1.75 0 1 0-.001 3.499A1.75 1.75 0 0 0 10 10.25ZM5.125 2a1.75 1.75 0 1 0 0 3.5 1.75 1.75 0 0 0 0-3.5Zm12.5 9.75a1.75 1.75 0 1 0 3.5 0 1.75 1.75 0 0 0-3.5 0Zm-14.25 8.5a1.75 1.75 0 1 0 3.501-.001 1.75 1.75 0 0 0-3.501.001Z"></path></svg><span class="NavLink-module__title__Q7t0p">GitHub Models</span><span class="NavLink-module__subtitle__X4gkW">Manage and compare prompts</span></div></a></li><li><a href="https://github.com/mcp" data-analytics-event="{&quot;action&quot;:&quot;mcp_registry&quot;,&quot;tag&quot;:&quot;link&quot;,&quot;context&quot;:&quot;platform&quot;,&quot;location&quot;:&quot;navbar&quot;,&quot;label&quot;:&quot;mcp_registry_link_platform_navbar&quot;}" class="NavLink-module__link__EG3d4"><div class="NavLink-module__text__XvpLQ"><svg aria-hidden="true" focusable="false" class="octicon octicon-mcp NavLink-module__icon__ltGNM" viewBox="0 0 24 24" width="24" height="24" fill="currentColor" display="inline-block" overflow="visible" style="vertical-align:text-bottom"><path d="M9.795 1.694a4.287 4.287 0 0 1 6.061 0 4.28 4.28 0 0 1 1.181 3.819 4.282 4.282 0 0 1 3.819 1.181 4.287 4.287 0 0 1 0 6.061l-6.793 6.793a.249.249 0 0 0 0 .353l2.617 2.618a.75.75 0 1 1-1.061 1.061l-2.617-2.618a1.75 1.75 0 0 1 0-2.475l6.793-6.793a2.785 2.785 0 1 0-3.939-3.939l-5.9 5.9a.734.734 0 0 1-.249.165.749.749 0 0 1-.812-1.225l5.9-5.901a2.785 2.785 0 1 0-3.939-3.939L2.931 10.68A.75.75 0 1 1 1.87 9.619l7.925-7.925Z"></path><path d="M12.42 4.069a.752.752 0 0 1 1.061 0 .752.752 0 0 1 0 1.061L7.33 11.28a2.788 2.788 0 0 0 0 3.94 2.788 2.788 0 0 0 3.94 0l6.15-6.151a.752.752 0 0 1 1.061 0 .752.752 0 0 1 0 1.061l-6.151 6.15a4.285 4.285 0 1 1-6.06-6.06l6.15-6.151Z"></path></svg><span class="NavLink-module__title__Q7t0p">MCP Registry<sup class="NavLink-module__label__bil7n">New</sup></span><span class="NavLink-module__subtitle__X4gkW">Integrate external tools</span></div></a></li></ul></div></li><li><div class="NavGroup-module__group__W8SqJ"><span class="NavGroup-module__title__Wzxz2">DEVELOPER WORKFLOWS</span><ul class="NavGroup-module__list__UCOFy"><li><a href="https://github.com/features/actions" data-analytics-event="{&quot;action&quot;:&quot;actions&quot;,&quot;tag&quot;:&quot;link&quot;,&quot;context&quot;:&quot;platform&quot;,&quot;location&quot;:&quot;navbar&quot;,&quot;label&quot;:&quot;actions_link_platform_navbar&quot;}" class="NavLink-module__link__EG3d4"><div class="NavLink-module__text__XvpLQ"><svg aria-hidden="true" focusable="false" class="octicon octicon-workflow NavLink-module__icon__ltGNM" viewBox="0 0 24 24" width="24" height="24" fill="currentColor" display="inline-block" overflow="visible" style="vertical-align:text-bottom"><path d="M1 3a2 2 0 0 1 2-2h6.5a2 2 0 0 1 2 2v6.5a2 2 0 0 1-2 2H7v4.063C7 16.355 7.644 17 8.438 17H12.5v-2.5a2 2 0 0 1 2-2H21a2 2 0 0 1 2 2V21a2 2 0 0 1-2 2h-6.5a2 2 0 0 1-2-2v-2.5H8.437A2.939 2.939 0 0 1 5.5 15.562V11.5H3a2 2 0 0 1-2-2Zm2-.5a.5.5 0 0 0-.5.5v6.5a.5.5 0 0 0 .5.5h6.5a.5.5 0 0 0 .5-.5V3a.5.5 0 0 0-.5-.5ZM14.5 14a.5.5 0 0 0-.5.5V21a.5.5 0 0 0 .5.5H21a.5.5 0 0 0 .5-.5v-6.5a.5.5 0 0 0-.5-.5Z"></path></svg><span class="NavLink-module__title__Q7t0p">Actions</span><span class="NavLink-module__subtitle__X4gkW">Automate any workflow</span></div></a></li><li><a href="https://github.com/features/codespaces" data-analytics-event="{&quot;action&quot;:&quot;codespaces&quot;,&quot;tag&quot;:&quot;link&quot;,&quot;context&quot;:&quot;platform&quot;,&quot;location&quot;:&quot;navbar&quot;,&quot;label&quot;:&quot;codespaces_link_platform_navbar&quot;}" class="NavLink-module__link__EG3d4"><div class="NavLink-module__text__XvpLQ"><svg aria-hidden="true" focusable="false" class="octicon octicon-codespaces NavLink-module__icon__ltGNM" viewBox="0 0 24 24" width="24" height="24" fill="currentColor" display="inline-block" overflow="visible" style="vertical-align:text-bottom"><path d="M3.5 3.75C3.5 2.784 4.284 2 5.25 2h13.5c.966 0 1.75.784 1.75 1.75v7.5A1.75 1.75 0 0 1 18.75 13H5.25a1.75 1.75 0 0 1-1.75-1.75Zm-2 12c0-.966.784-1.75 1.75-1.75h17.5c.966 0 1.75.784 1.75 1.75v4a1.75 1.75 0 0 1-1.75 1.75H3.25a1.75 1.75 0 0 1-1.75-1.75ZM5.25 3.5a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h13.5a.25.25 0 0 0 .25-.25v-7.5a.25.25 0 0 0-.25-.25Zm-2 12a.25.25 0 0 0-.25.25v4c0 .138.112.25.25.25h17.5a.25.25 0 0 0 .25-.25v-4a.25.25 0 0 0-.25-.25Z"></path><path d="M10 17.75a.75.75 0 0 1 .75-.75h6.5a.75.75 0 0 1 0 1.5h-6.5a.75.75 0 0 1-.75-.75Zm-4 0a.75.75 0 0 1 .75-.75h.5a.75.75 0 0 1 0 1.5h-.5a.75.75 0 0 1-.75-.75Z"></path></svg><span class="NavLink-module__title__Q7t0p">Codespaces</span><span class="NavLink-module__subtitle__X4gkW">Instant dev environments</span></div></a></li><li><a href="https://github.com/features/issues" data-analytics-event="{&quot;action&quot;:&quot;issues&quot;,&quot;tag&quot;:&quot;link&quot;,&quot;context&quot;:&quot;platform&quot;,&quot;location&quot;:&quot;navbar&quot;,&quot;label&quot;:&quot;issues_link_platform_navbar&quot;}" class="NavLink-module__link__EG3d4"><div class="NavLink-module__text__XvpLQ"><svg aria-hidden="true" focusable="false" class="octicon octicon-issue-opened NavLink-module__icon__ltGNM" viewBox="0 0 24 24" width="24" height="24" fill="currentColor" display="inline-block" overflow="visible" style="vertical-align:text-bottom"><path d="M12 1c6.075 0 11 4.925 11 11s-4.925 11-11 11S1 18.075 1 12 5.925 1 12 1ZM2.5 12a9.5 9.5 0 0 0 9.5 9.5 9.5 9.5 0 0 0 9.5-9.5A9.5 9.5 0 0 0 12 2.5 9.5 9.5 0 0 0 2.5 12Zm9.5 2a2 2 0 1 1-.001-3.999A2 2 0 0 1 12 14Z"></path></svg><span class="NavLink-module__title__Q7t0p">Issues</span><span class="NavLink-module__subtitle__X4gkW">Plan and track work</span></div></a></li><li><a href="https://github.com/features/code-review" data-analytics-event="{&quot;action&quot;:&quot;code_review&quot;,&quot;tag&quot;:&quot;link&quot;,&quot;context&quot;:&quot;platform&quot;,&quot;location&quot;:&quot;navbar&quot;,&quot;label&quot;:&quot;code_review_link_platform_navbar&quot;}" class="NavLink-module__link__EG3d4"><div class="NavLink-module__text__XvpLQ"><svg aria-hidden="true" focusable="false" class="octicon octicon-code NavLink-module__icon__ltGNM" viewBox="0 0 24 24" width="24" height="24" fill="currentColor" display="inline-block" overflow="visible" style="vertical-align:text-bottom"><path d="M15.22 4.97a.75.75 0 0 1 1.06 0l6.5 6.5a.75.75 0 0 1 0 1.06l-6.5 6.5a.749.749 0 0 1-1.275-.326.749.749 0 0 1 .215-.734L21.19 12l-5.97-5.97a.75.75 0 0 1 0-1.06Zm-6.44 0a.75.75 0 0 1 0 1.06L2.81 12l5.97 5.97a.749.749 0 0 1-.326 1.275.749.749 0 0 1-.734-.215l-6.5-6.5a.75.75 0 0 1 0-1.06l6.5-6.5a.75.75 0 0 1 1.06 0Z"></path></svg><span class="NavLink-module__title__Q7t0p">Code Review</span><span class="NavLink-module__subtitle__X4gkW">Manage code changes</span></div></a></li></ul></div></li><li><div class="NavGroup-module__group__W8SqJ"><span class="NavGroup-module__title__Wzxz2">APPLICATION SECURITY</span><ul class="NavGroup-module__list__UCOFy"><li><a href="https://github.com/security/advanced-security" data-analytics-event="{&quot;action&quot;:&quot;github_advanced_security&quot;,&quot;tag&quot;:&quot;link&quot;,&quot;context&quot;:&quot;platform&quot;,&quot;location&quot;:&quot;navbar&quot;,&quot;label&quot;:&quot;github_advanced_security_link_platform_navbar&quot;}" class="NavLink-module__link__EG3d4"><div class="NavLink-module__text__XvpLQ"><svg aria-hidden="true" focusable="false" class="octicon octicon-shield-check NavLink-module__icon__ltGNM" viewBox="0 0 24 24" width="24" height="24" fill="currentColor" display="inline-block" overflow="visible" style="vertical-align:text-bottom"><path d="M16.53 9.78a.75.75 0 0 0-1.06-1.06L11 13.19l-1.97-1.97a.75.75 0 0 0-1.06 1.06l2.5 2.5a.75.75 0 0 0 1.06 0l5-5Z"></path><path d="m12.54.637 8.25 2.675A1.75 1.75 0 0 1 22 4.976V10c0 6.19-3.771 10.704-9.401 12.83a1.704 1.704 0 0 1-1.198 0C5.77 20.705 2 16.19 2 10V4.976c0-.758.489-1.43 1.21-1.664L11.46.637a1.748 1.748 0 0 1 1.08 0Zm-.617 1.426-8.25 2.676a.249.249 0 0 0-.173.237V10c0 5.46 3.28 9.483 8.43 11.426a.199.199 0 0 0 .14 0C17.22 19.483 20.5 15.461 20.5 10V4.976a.25.25 0 0 0-.173-.237l-8.25-2.676a.253.253 0 0 0-.154 0Z"></path></svg><span class="NavLink-module__title__Q7t0p">GitHub Advanced Security</span><span class="NavLink-module__subtitle__X4gkW">Find and fix vulnerabilities</span></div></a></li><li><a href="https://github.com/security/advanced-security/code-security" data-analytics-event="{&quot;action&quot;:&quot;code_security&quot;,&quot;tag&quot;:&quot;link&quot;,&quot;context&quot;:&quot;platform&quot;,&quot;location&quot;:&quot;navbar&quot;,&quot;label&quot;:&quot;code_security_link_platform_navbar&quot;}" class="NavLink-module__link__EG3d4"><div class="NavLink-module__text__XvpLQ"><svg aria-hidden="true" focusable="false" class="octicon octicon-code-square NavLink-module__icon__ltGNM" viewBox="0 0 24 24" width="24" height="24" fill="currentColor" display="inline-block" overflow="visible" style="vertical-align:text-bottom"><path d="M10.3 8.24a.75.75 0 0 1-.04 1.06L7.352 12l2.908 2.7a.75.75 0 1 1-1.02 1.1l-3.5-3.25a.75.75 0 0 1 0-1.1l3.5-3.25a.75.75 0 0 1 1.06.04Zm3.44 1.06a.75.75 0 1 1 1.02-1.1l3.5 3.25a.75.75 0 0 1 0 1.1l-3.5 3.25a.75.75 0 1 1-1.02-1.1l2.908-2.7-2.908-2.7Z"></path><path d="M2 3.75C2 2.784 2.784 2 3.75 2h16.5c.966 0 1.75.784 1.75 1.75v16.5A1.75 1.75 0 0 1 20.25 22H3.75A1.75 1.75 0 0 1 2 20.25Zm1.75-.25a.25.25 0 0 0-.25.25v16.5c0 .138.112.25.25.25h16.5a.25.25 0 0 0 .25-.25V3.75a.25.25 0 0 0-.25-.25Z"></path></svg><span class="NavLink-module__title__Q7t0p">Code security</span><span class="NavLink-module__subtitle__X4gkW">Secure your code as you build</span></div></a></li><li><a href="https://github.com/security/advanced-security/secret-protection" data-analytics-event="{&quot;action&quot;:&quot;secret_protection&quot;,&quot;tag&quot;:&quot;link&quot;,&quot;context&quot;:&quot;platform&quot;,&quot;location&quot;:&quot;navbar&quot;,&quot;label&quot;:&quot;secret_protection_link_platform_navbar&quot;}" class="NavLink-module__link__EG3d4"><div class="NavLink-module__text__XvpLQ"><svg aria-hidden="true" focusable="false" class="octicon octicon-lock NavLink-module__icon__ltGNM" viewBox="0 0 24 24" width="24" height="24" fill="currentColor" display="inline-block" overflow="visible" style="vertical-align:text-bottom"><path d="M6 9V7.25C6 3.845 8.503 1 12 1s6 2.845 6 6.25V9h.5a2.5 2.5 0 0 1 2.5 2.5v8a2.5 2.5 0 0 1-2.5 2.5h-13A2.5 2.5 0 0 1 3 19.5v-8A2.5 2.5 0 0 1 5.5 9Zm-1.5 2.5v8a1 1 0 0 0 1 1h13a1 1 0 0 0 1-1v-8a1 1 0 0 0-1-1h-13a1 1 0 0 0-1 1Zm3-4.25V9h9V7.25c0-2.67-1.922-4.75-4.5-4.75-2.578 0-4.5 2.08-4.5 4.75Z"></path></svg><span class="NavLink-module__title__Q7t0p">Secret protection</span><span class="NavLink-module__subtitle__X4gkW">Stop leaks before they start</span></div></a></li></ul></div></li><li><div class="NavGroup-module__group__W8SqJ NavGroup-module__hasSeparator__FnMrN"><span class="NavGroup-module__title__Wzxz2">EXPLORE</span><ul class="NavGroup-module__list__UCOFy"><li><a href="https://github.com/why-github" data-analytics-event="{&quot;action&quot;:&quot;why_github&quot;,&quot;tag&quot;:&quot;link&quot;,&quot;context&quot;:&quot;platform&quot;,&quot;location&quot;:&quot;navbar&quot;,&quot;label&quot;:&quot;why_github_link_platform_navbar&quot;}" class="NavLink-module__link__EG3d4"><span class="NavLink-module__title__Q7t0p">Why GitHub</span></a></li><li><a href="https://docs.github.com/" data-analytics-event="{&quot;action&quot;:&quot;documentation&quot;,&quot;tag&quot;:&quot;link&quot;,&quot;context&quot;:&quot;platform&quot;,&quot;location&quot;:&quot;navbar&quot;,&quot;label&quot;:&quot;documentation_link_platform_navbar&quot;}" class="NavLink-module__link__EG3d4" target="_blank" rel="noreferrer"><span class="NavLink-module__title__Q7t0p">Documentation</span><svg aria-hidden="true" focusable="false" class="octicon octicon-link-external NavLink-module__externalIcon__eWIry" viewBox="0 0 16 16" width="16" height="16" fill="currentColor" display="inline-block" overflow="visible" style="vertical-align:text-bottom"><path d="M3.75 2h3.5a.75.75 0 0 1 0 1.5h-3.5a.25.25 0 0 0-.25.25v8.5c0 .138.112.25.25.25h8.5a.25.25 0 0 0 .25-.25v-3.5a.75.75 0 0 1 1.5 0v3.5A1.75 1.75 0 0 1 12.25 14h-8.5A1.75 1.75 0 0 1 2 12.25v-8.5C2 2.784 2.784 2 3.75 2Zm6.854-1h4.146a.25.25 0 0 1 .25.25v4.146a.25.25 0 0 1-.427.177L13.03 4.03 9.28 7.78a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042l3.75-3.75-1.543-1.543A.25.25 0 0 1 10.604 1Z"></path></svg></a></li><li><a href="https://github.blog/" data-analytics-event="{&quot;action&quot;:&quot;blog&quot;,&quot;tag&quot;:&quot;link&quot;,&quot;context&quot;:&quot;platform&quot;,&quot;location&quot;:&quot;navbar&quot;,&quot;label&quot;:&quot;blog_link_platform_navbar&quot;}" class="NavLink-module__link__EG3d4" target="_blank" rel="noreferrer"><span class="NavLink-module__title__Q7t0p">Blog</span><svg aria-hidden="true" focusable="false" class="octicon octicon-link-external NavLink-module__externalIcon__eWIry" viewBox="0 0 16 16" width="16" height="16" fill="currentColor" display="inline-block" overflow="visible" style="vertical-align:text-bottom"><path d="M3.75 2h3.5a.75.75 0 0 1 0 1.5h-3.5a.25.25 0 0 0-.25.25v8.5c0 .138.112.25.25.25h8.5a.25.25 0 0 0 .25-.25v-3.5a.75.75 0 0 1 1.5 0v3.5A1.75 1.75 0 0 1 12.25 14h-8.5A1.75 1.75 0 0 1 2 12.25v-8.5C2 2.784 2.784 2 3.75 2Zm6.854-1h4.146a.25.25 0 0 1 .25.25v4.146a.25.25 0 0 1-.427.177L13.03 4.03 9.28 7.78a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042l3.75-3.75-1.543-1.543A.25.25 0 0 1 10.604 1Z"></path></svg></a></li><li><a href="https://github.blog/changelog" data-analytics-event="{&quot;action&quot;:&quot;changelog&quot;,&quot;tag&quot;:&quot;link&quot;,&quot;context&quot;:&quot;platform&quot;,&quot;location&quot;:&quot;navbar&quot;,&quot;label&quot;:&quot;changelog_link_platform_navbar&quot;}" class="NavLink-module__link__EG3d4" target="_blank" rel="noreferrer"><span class="NavLink-module__title__Q7t0p">Changelog</span><svg aria-hidden="true" focusable="false" class="octicon octicon-link-external NavLink-module__externalIcon__eWIry" viewBox="0 0 16 16" width="16" height="16" fill="currentColor" display="inline-block" overflow="visible" style="vertical-align:text-bottom"><path d="M3.75 2h3.5a.75.75 0 0 1 0 1.5h-3.5a.25.25 0 0 0-.25.25v8.5c0 .138.112.25.25.25h8.5a.25.25 0 0 0 .25-.25v-3.5a.75.75 0 0 1 1.5 0v3.5A1.75 1.75 0 0 1 12.25 14h-8.5A1.75 1.75 0 0 1 2 12.25v-8.5C2 2.784 2.784 2 3.75 2Zm6.854-1h4.146a.25.25 0 0 1 .25.25v4.146a.25.25 0 0 1-.427.177L13.03 4.03 9.28 7.78a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042l3.75-3.75-1.543-1.543A.25.25 0 0 1 10.604 1Z"></path></svg></a></li><li><a href="https://github.com/marketplace" data-analytics-event="{&quot;action&quot;:&quot;marketplace&quot;,&quot;tag&quot;:&quot;link&quot;,&quot;context&quot;:&quot;platform&quot;,&quot;location&quot;:&quot;navbar&quot;,&quot;label&quot;:&quot;marketplace_link_platform_navbar&quot;}" class="NavLink-module__link__EG3d4"><span class="NavLink-module__title__Q7t0p">Marketplace</span></a></li></ul></div></li></ul><div class="NavDropdown-module__trailingLinkContainer__VgJGL"><a href="https://github.com/features" data-analytics-event="{&quot;action&quot;:&quot;view_all_features&quot;,&quot;tag&quot;:&quot;link&quot;,&quot;context&quot;:&quot;platform&quot;,&quot;location&quot;:&quot;navbar&quot;,&quot;label&quot;:&quot;view_all_features_link_platform_navbar&quot;}" class="NavLink-module__link__EG3d4"><span class="NavLink-module__title__Q7t0p">View all features</span><svg aria-hidden="true" focusable="false" class="octicon octicon-chevron-right NavLink-module__arrowIcon__amekg" viewBox="0 0 16 16" width="16" height="16" fill="currentColor" display="inline-block" overflow="visible" style="vertical-align:text-bottom"><path d="M6.22 3.22a.75.75 0 0 1 1.06 0l4.25 4.25a.75.75 0 0 1 0 1.06l-4.25 4.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042L9.94 8 6.22 4.28a.75.75 0 0 1 0-1.06Z"></path></svg></a></div></div></div></li><li><div class="NavDropdown-module__container__l2YeI js-details-container js-header-menu-item"><button type="button" class="NavDropdown-module__button__PEHWX js-details-target" aria-expanded="false">Solutions<svg aria-hidden="true" focusable="false" class="octicon octicon-chevron-right NavDropdown-module__buttonIcon__Tkl8_" viewBox="0 0 16 16" width="16" height="16" fill="currentColor" display="inline-block" overflow="visible" style="vertical-align:text-bottom"><path d="M6.22 3.22a.75.75 0 0 1 1.06 0l4.25 4.25a.75.75 0 0 1 0 1.06l-4.25 4.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042L9.94 8 6.22 4.28a.75.75 0 0 1 0-1.06Z"></path></svg></button><div class="NavDropdown-module__dropdown__xm1jd"><ul class="NavDropdown-module__list__zuCgG"><li><div class="NavGroup-module__group__W8SqJ"><span class="NavGroup-module__title__Wzxz2">BY COMPANY SIZE</span><ul class="NavGroup-module__list__UCOFy"><li><a href="https://github.com/enterprise" data-analytics-event="{&quot;action&quot;:&quot;enterprises&quot;,&quot;tag&quot;:&quot;link&quot;,&quot;context&quot;:&quot;solutions&quot;,&quot;location&quot;:&quot;navbar&quot;,&quot;label&quot;:&quot;enterprises_link_solutions_navbar&quot;}" class="NavLink-module__link__EG3d4"><span class="NavLink-module__title__Q7t0p">Enterprises</span></a></li><li><a href="https://github.com/team" data-analytics-event="{&quot;action&quot;:&quot;small_and_medium_teams&quot;,&quot;tag&quot;:&quot;link&quot;,&quot;context&quot;:&quot;solutions&quot;,&quot;location&quot;:&quot;navbar&quot;,&quot;label&quot;:&quot;small_and_medium_teams_link_solutions_navbar&quot;}" class="NavLink-module__link__EG3d4"><span class="NavLink-module__title__Q7t0p">Small and medium teams</span></a></li><li><a href="https://github.com/enterprise/startups" data-analytics-event="{&quot;action&quot;:&quot;startups&quot;,&quot;tag&quot;:&quot;link&quot;,&quot;context&quot;:&quot;solutions&quot;,&quot;location&quot;:&quot;navbar&quot;,&quot;label&quot;:&quot;startups_link_solutions_navbar&quot;}" class="NavLink-module__link__EG3d4"><span class="NavLink-module__title__Q7t0p">Startups</span></a></li><li><a href="https://github.com/solutions/industry/nonprofits" data-analytics-event="{&quot;action&quot;:&quot;nonprofits&quot;,&quot;tag&quot;:&quot;link&quot;,&quot;context&quot;:&quot;solutions&quot;,&quot;location&quot;:&quot;navbar&quot;,&quot;label&quot;:&quot;nonprofits_link_solutions_navbar&quot;}" class="NavLink-module__link__EG3d4"><span class="NavLink-module__title__Q7t0p">Nonprofits</span></a></li></ul></div></li><li><div class="NavGroup-module__group__W8SqJ"><span class="NavGroup-module__title__Wzxz2">BY USE CASE</span><ul class="NavGroup-module__list__UCOFy"><li><a href="https://github.com/solutions/use-case/app-modernization" data-analytics-event="{&quot;action&quot;:&quot;app_modernization&quot;,&quot;tag&quot;:&quot;link&quot;,&quot;context&quot;:&quot;solutions&quot;,&quot;location&quot;:&quot;navbar&quot;,&quot;label&quot;:&quot;app_modernization_link_solutions_navbar&quot;}" class="NavLink-module__link__EG3d4"><span class="NavLink-module__title__Q7t0p">App Modernization</span></a></li><li><a href="https://github.com/solutions/use-case/devsecops" data-analytics-event="{&quot;action&quot;:&quot;devsecops&quot;,&quot;tag&quot;:&quot;link&quot;,&quot;context&quot;:&quot;solutions&quot;,&quot;location&quot;:&quot;navbar&quot;,&quot;label&quot;:&quot;devsecops_link_solutions_navbar&quot;}" class="NavLink-module__link__EG3d4"><span class="NavLink-module__title__Q7t0p">DevSecOps</span></a></li><li><a href="https://github.com/solutions/use-case/devops" data-analytics-event="{&quot;action&quot;:&quot;devops&quot;,&quot;tag&quot;:&quot;link&quot;,&quot;context&quot;:&quot;solutions&quot;,&quot;location&quot;:&quot;navbar&quot;,&quot;label&quot;:&quot;devops_link_solutions_navbar&quot;}" class="NavLink-module__link__EG3d4"><span class="NavLink-module__title__Q7t0p">DevOps</span></a></li><li><a href="https://github.com/solutions/use-case/ci-cd" data-analytics-event="{&quot;action&quot;:&quot;ci/cd&quot;,&quot;tag&quot;:&quot;link&quot;,&quot;context&quot;:&quot;solutions&quot;,&quot;location&quot;:&quot;navbar&quot;,&quot;label&quot;:&quot;ci/cd_link_solutions_navbar&quot;}" class="NavLink-module__link__EG3d4"><span class="NavLink-module__title__Q7t0p">CI/CD</span></a></li><li><a href="https://github.com/solutions/use-case" data-analytics-event="{&quot;action&quot;:&quot;view_all_use_cases&quot;,&quot;tag&quot;:&quot;link&quot;,&quot;context&quot;:&quot;solutions&quot;,&quot;location&quot;:&quot;navbar&quot;,&quot;label&quot;:&quot;view_all_use_cases_link_solutions_navbar&quot;}" class="NavLink-module__link__EG3d4"><span class="NavLink-module__title__Q7t0p">View all use cases</span><svg aria-hidden="true" focusable="false" class="octicon octicon-chevron-right NavLink-module__arrowIcon__amekg" viewBox="0 0 16 16" width="16" height="16" fill="currentColor" display="inline-block" overflow="visible" style="vertical-align:text-bottom"><path d="M6.22 3.22a.75.75 0 0 1 1.06 0l4.25 4.25a.75.75 0 0 1 0 1.06l-4.25 4.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042L9.94 8 6.22 4.28a.75.75 0 0 1 0-1.06Z"></path></svg></a></li></ul></div></li><li><div class="NavGroup-module__group__W8SqJ"><span class="NavGroup-module__title__Wzxz2">BY INDUSTRY</span><ul class="NavGroup-module__list__UCOFy"><li><a href="https://github.com/solutions/industry/healthcare" data-analytics-event="{&quot;action&quot;:&quot;healthcare&quot;,&quot;tag&quot;:&quot;link&quot;,&quot;context&quot;:&quot;solutions&quot;,&quot;location&quot;:&quot;navbar&quot;,&quot;label&quot;:&quot;healthcare_link_solutions_navbar&quot;}" class="NavLink-module__link__EG3d4"><span class="NavLink-module__title__Q7t0p">Healthcare</span></a></li><li><a href="https://github.com/solutions/industry/financial-services" data-analytics-event="{&quot;action&quot;:&quot;financial_services&quot;,&quot;tag&quot;:&quot;link&quot;,&quot;context&quot;:&quot;solutions&quot;,&quot;location&quot;:&quot;navbar&quot;,&quot;label&quot;:&quot;financial_services_link_solutions_navbar&quot;}" class="NavLink-module__link__EG3d4"><span class="NavLink-module__title__Q7t0p">Financial services</span></a></li><li><a href="https://github.com/solutions/industry/manufacturing" data-analytics-event="{&quot;action&quot;:&quot;manufacturing&quot;,&quot;tag&quot;:&quot;link&quot;,&quot;context&quot;:&quot;solutions&quot;,&quot;location&quot;:&quot;navbar&quot;,&quot;label&quot;:&quot;manufacturing_link_solutions_navbar&quot;}" class="NavLink-module__link__EG3d4"><span class="NavLink-module__title__Q7t0p">Manufacturing</span></a></li><li><a href="https://github.com/solutions/industry/government" data-analytics-event="{&quot;action&quot;:&quot;government&quot;,&quot;tag&quot;:&quot;link&quot;,&quot;context&quot;:&quot;solutions&quot;,&quot;location&quot;:&quot;navbar&quot;,&quot;label&quot;:&quot;government_link_solutions_navbar&quot;}" class="NavLink-module__link__EG3d4"><span class="NavLink-module__title__Q7t0p">Government</span></a></li><li><a href="https://github.com/solutions/industry" data-analytics-event="{&quot;action&quot;:&quot;view_all_industries&quot;,&quot;tag&quot;:&quot;link&quot;,&quot;context&quot;:&quot;solutions&quot;,&quot;location&quot;:&quot;navbar&quot;,&quot;label&quot;:&quot;view_all_industries_link_solutions_navbar&quot;}" class="NavLink-module__link__EG3d4"><span class="NavLink-module__title__Q7t0p">View all industries</span><svg aria-hidden="true" focusable="false" class="octicon octicon-chevron-right NavLink-module__arrowIcon__amekg" viewBox="0 0 16 16" width="16" height="16" fill="currentColor" display="inline-block" overflow="visible" style="vertical-align:text-bottom"><path d="M6.22 3.22a.75.75 0 0 1 1.06 0l4.25 4.25a.75.75 0 0 1 0 1.06l-4.25 4.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042L9.94 8 6.22 4.28a.75.75 0 0 1 0-1.06Z"></path></svg></a></li></ul></div></li></ul><div class="NavDropdown-module__trailingLinkContainer__VgJGL"><a href="https://github.com/solutions" data-analytics-event="{&quot;action&quot;:&quot;view_all_solutions&quot;,&quot;tag&quot;:&quot;link&quot;,&quot;context&quot;:&quot;solutions&quot;,&quot;location&quot;:&quot;navbar&quot;,&quot;label&quot;:&quot;view_all_solutions_link_solutions_navbar&quot;}" class="NavLink-module__link__EG3d4"><span class="NavLink-module__title__Q7t0p">View all solutions</span><svg aria-hidden="true" focusable="false" class="octicon octicon-chevron-right NavLink-module__arrowIcon__amekg" viewBox="0 0 16 16" width="16" height="16" fill="currentColor" display="inline-block" overflow="visible" style="vertical-align:text-bottom"><path d="M6.22 3.22a.75.75 0 0 1 1.06 0l4.25 4.25a.75.75 0 0 1 0 1.06l-4.25 4.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042L9.94 8 6.22 4.28a.75.75 0 0 1 0-1.06Z"></path></svg></a></div></div></div></li><li><div class="NavDropdown-module__container__l2YeI js-details-container js-header-menu-item"><button type="button" class="NavDropdown-module__button__PEHWX js-details-target" aria-expanded="false">Resources<svg aria-hidden="true" focusable="false" class="octicon octicon-chevron-right NavDropdown-module__buttonIcon__Tkl8_" viewBox="0 0 16 16" width="16" height="16" fill="currentColor" display="inline-block" overflow="visible" style="vertical-align:text-bottom"><path d="M6.22 3.22a.75.75 0 0 1 1.06 0l4.25 4.25a.75.75 0 0 1 0 1.06l-4.25 4.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042L9.94 8 6.22 4.28a.75.75 0 0 1 0-1.06Z"></path></svg></button><div class="NavDropdown-module__dropdown__xm1jd"><ul class="NavDropdown-module__list__zuCgG"><li><div class="NavGroup-module__group__W8SqJ"><span class="NavGroup-module__title__Wzxz2">EXPLORE BY TOPIC</span><ul class="NavGroup-module__list__UCOFy"><li><a href="https://github.com/resources/articles?topic=ai" data-analytics-event="{&quot;action&quot;:&quot;ai&quot;,&quot;tag&quot;:&quot;link&quot;,&quot;context&quot;:&quot;resources&quot;,&quot;location&quot;:&quot;navbar&quot;,&quot;label&quot;:&quot;ai_link_resources_navbar&quot;}" class="NavLink-module__link__EG3d4"><span class="NavLink-module__title__Q7t0p">AI</span></a></li><li><a href="https://github.com/resources/articles?topic=software-development" data-analytics-event="{&quot;action&quot;:&quot;software_development&quot;,&quot;tag&quot;:&quot;link&quot;,&quot;context&quot;:&quot;resources&quot;,&quot;location&quot;:&quot;navbar&quot;,&quot;label&quot;:&quot;software_development_link_resources_navbar&quot;}" class="NavLink-module__link__EG3d4"><span class="NavLink-module__title__Q7t0p">Software Development</span></a></li><li><a href="https://github.com/resources/articles?topic=devops" data-analytics-event="{&quot;action&quot;:&quot;devops&quot;,&quot;tag&quot;:&quot;link&quot;,&quot;context&quot;:&quot;resources&quot;,&quot;location&quot;:&quot;navbar&quot;,&quot;label&quot;:&quot;devops_link_resources_navbar&quot;}" class="NavLink-module__link__EG3d4"><span class="NavLink-module__title__Q7t0p">DevOps</span></a></li><li><a href="https://github.com/resources/articles?topic=security" data-analytics-event="{&quot;action&quot;:&quot;security&quot;,&quot;tag&quot;:&quot;link&quot;,&quot;context&quot;:&quot;resources&quot;,&quot;location&quot;:&quot;navbar&quot;,&quot;label&quot;:&quot;security_link_resources_navbar&quot;}" class="NavLink-module__link__EG3d4"><span class="NavLink-module__title__Q7t0p">Security</span></a></li><li><a href="https://github.com/resources/articles" data-analytics-event="{&quot;action&quot;:&quot;view_all_topics&quot;,&quot;tag&quot;:&quot;link&quot;,&quot;context&quot;:&quot;resources&quot;,&quot;location&quot;:&quot;navbar&quot;,&quot;label&quot;:&quot;view_all_topics_link_resources_navbar&quot;}" class="NavLink-module__link__EG3d4"><span class="NavLink-module__title__Q7t0p">View all topics</span><svg aria-hidden="true" focusable="false" class="octicon octicon-chevron-right NavLink-module__arrowIcon__amekg" viewBox="0 0 16 16" width="16" height="16" fill="currentColor" display="inline-block" overflow="visible" style="vertical-align:text-bottom"><path d="M6.22 3.22a.75.75 0 0 1 1.06 0l4.25 4.25a.75.75 0 0 1 0 1.06l-4.25 4.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042L9.94 8 6.22 4.28a.75.75 0 0 1 0-1.06Z"></path></svg></a></li></ul></div></li><li><div class="NavGroup-module__group__W8SqJ"><span class="NavGroup-module__title__Wzxz2">EXPLORE BY TYPE</span><ul class="NavGroup-module__list__UCOFy"><li><a href="https://github.com/customer-stories" data-analytics-event="{&quot;action&quot;:&quot;customer_stories&quot;,&quot;tag&quot;:&quot;link&quot;,&quot;context&quot;:&quot;resources&quot;,&quot;location&quot;:&quot;navbar&quot;,&quot;label&quot;:&quot;customer_stories_link_resources_navbar&quot;}" class="NavLink-module__link__EG3d4"><span class="NavLink-module__title__Q7t0p">Customer stories</span></a></li><li><a href="https://github.com/resources/events" data-analytics-event="{&quot;action&quot;:&quot;events__webinars&quot;,&quot;tag&quot;:&quot;link&quot;,&quot;context&quot;:&quot;resources&quot;,&quot;location&quot;:&quot;navbar&quot;,&quot;label&quot;:&quot;events__webinars_link_resources_navbar&quot;}" class="NavLink-module__link__EG3d4"><span class="NavLink-module__title__Q7t0p">Events &amp; webinars</span></a></li><li><a href="https://github.com/resources/whitepapers" data-analytics-event="{&quot;action&quot;:&quot;ebooks__reports&quot;,&quot;tag&quot;:&quot;link&quot;,&quot;context&quot;:&quot;resources&quot;,&quot;location&quot;:&quot;navbar&quot;,&quot;label&quot;:&quot;ebooks__reports_link_resources_navbar&quot;}" class="NavLink-module__link__EG3d4"><span class="NavLink-module__title__Q7t0p">Ebooks &amp; reports</span></a></li><li><a href="https://github.com/solutions/executive-insights" data-analytics-event="{&quot;action&quot;:&quot;business_insights&quot;,&quot;tag&quot;:&quot;link&quot;,&quot;context&quot;:&quot;resources&quot;,&quot;location&quot;:&quot;navbar&quot;,&quot;label&quot;:&quot;business_insights_link_resources_navbar&quot;}" class="NavLink-module__link__EG3d4"><span class="NavLink-module__title__Q7t0p">Business insights</span></a></li><li><a href="https://skills.github.com/" data-analytics-event="{&quot;action&quot;:&quot;github_skills&quot;,&quot;tag&quot;:&quot;link&quot;,&quot;context&quot;:&quot;resources&quot;,&quot;location&quot;:&quot;navbar&quot;,&quot;label&quot;:&quot;github_skills_link_resources_navbar&quot;}" class="NavLink-module__link__EG3d4" target="_blank" rel="noreferrer"><span class="NavLink-module__title__Q7t0p">GitHub Skills</span><svg aria-hidden="true" focusable="false" class="octicon octicon-link-external NavLink-module__externalIcon__eWIry" viewBox="0 0 16 16" width="16" height="16" fill="currentColor" display="inline-block" overflow="visible" style="vertical-align:text-bottom"><path d="M3.75 2h3.5a.75.75 0 0 1 0 1.5h-3.5a.25.25 0 0 0-.25.25v8.5c0 .138.112.25.25.25h8.5a.25.25 0 0 0 .25-.25v-3.5a.75.75 0 0 1 1.5 0v3.5A1.75 1.75 0 0 1 12.25 14h-8.5A1.75 1.75 0 0 1 2 12.25v-8.5C2 2.784 2.784 2 3.75 2Zm6.854-1h4.146a.25.25 0 0 1 .25.25v4.146a.25.25 0 0 1-.427.177L13.03 4.03 9.28 7.78a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042l3.75-3.75-1.543-1.543A.25.25 0 0 1 10.604 1Z"></path></svg></a></li></ul></div></li><li><div class="NavGroup-module__group__W8SqJ"><span class="NavGroup-module__title__Wzxz2">SUPPORT &amp; SERVICES</span><ul class="NavGroup-module__list__UCOFy"><li><a href="https://docs.github.com/" data-analytics-event="{&quot;action&quot;:&quot;documentation&quot;,&quot;tag&quot;:&quot;link&quot;,&quot;context&quot;:&quot;resources&quot;,&quot;location&quot;:&quot;navbar&quot;,&quot;label&quot;:&quot;documentation_link_resources_navbar&quot;}" class="NavLink-module__link__EG3d4" target="_blank" rel="noreferrer"><span class="NavLink-module__title__Q7t0p">Documentation</span><svg aria-hidden="true" focusable="false" class="octicon octicon-link-external NavLink-module__externalIcon__eWIry" viewBox="0 0 16 16" width="16" height="16" fill="currentColor" display="inline-block" overflow="visible" style="vertical-align:text-bottom"><path d="M3.75 2h3.5a.75.75 0 0 1 0 1.5h-3.5a.25.25 0 0 0-.25.25v8.5c0 .138.112.25.25.25h8.5a.25.25 0 0 0 .25-.25v-3.5a.75.75 0 0 1 1.5 0v3.5A1.75 1.75 0 0 1 12.25 14h-8.5A1.75 1.75 0 0 1 2 12.25v-8.5C2 2.784 2.784 2 3.75 2Zm6.854-1h4.146a.25.25 0 0 1 .25.25v4.146a.25.25 0 0 1-.427.177L13.03 4.03 9.28 7.78a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042l3.75-3.75-1.543-1.543A.25.25 0 0 1 10.604 1Z"></path></svg></a></li><li><a href="https://support.github.com/" data-analytics-event="{&quot;action&quot;:&quot;customer_support&quot;,&quot;tag&quot;:&quot;link&quot;,&quot;context&quot;:&quot;resources&quot;,&quot;location&quot;:&quot;navbar&quot;,&quot;label&quot;:&quot;customer_support_link_resources_navbar&quot;}" class="NavLink-module__link__EG3d4" target="_blank" rel="noreferrer"><span class="NavLink-module__title__Q7t0p">Customer support</span><svg aria-hidden="true" focusable="false" class="octicon octicon-link-external NavLink-module__externalIcon__eWIry" viewBox="0 0 16 16" width="16" height="16" fill="currentColor" display="inline-block" overflow="visible" style="vertical-align:text-bottom"><path d="M3.75 2h3.5a.75.75 0 0 1 0 1.5h-3.5a.25.25 0 0 0-.25.25v8.5c0 .138.112.25.25.25h8.5a.25.25 0 0 0 .25-.25v-3.5a.75.75 0 0 1 1.5 0v3.5A1.75 1.75 0 0 1 12.25 14h-8.5A1.75 1.75 0 0 1 2 12.25v-8.5C2 2.784 2.784 2 3.75 2Zm6.854-1h4.146a.25.25 0 0 1 .25.25v4.146a.25.25 0 0 1-.427.177L13.03 4.03 9.28 7.78a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042l3.75-3.75-1.543-1.543A.25.25 0 0 1 10.604 1Z"></path></svg></a></li><li><a href="https://github.com/orgs/community/discussions" data-analytics-event="{&quot;action&quot;:&quot;community_forum&quot;,&quot;tag&quot;:&quot;link&quot;,&quot;context&quot;:&quot;resources&quot;,&quot;location&quot;:&quot;navbar&quot;,&quot;label&quot;:&quot;community_forum_link_resources_navbar&quot;}" class="NavLink-module__link__EG3d4"><span class="NavLink-module__title__Q7t0p">Community forum</span></a></li><li><a href="https://github.com/trust-center" data-analytics-event="{&quot;action&quot;:&quot;trust_center&quot;,&quot;tag&quot;:&quot;link&quot;,&quot;context&quot;:&quot;resources&quot;,&quot;location&quot;:&quot;navbar&quot;,&quot;label&quot;:&quot;trust_center_link_resources_navbar&quot;}" class="NavLink-module__link__EG3d4"><span class="NavLink-module__title__Q7t0p">Trust center</span></a></li><li><a href="https://github.com/partners" data-analytics-event="{&quot;action&quot;:&quot;partners&quot;,&quot;tag&quot;:&quot;link&quot;,&quot;context&quot;:&quot;resources&quot;,&quot;location&quot;:&quot;navbar&quot;,&quot;label&quot;:&quot;partners_link_resources_navbar&quot;}" class="NavLink-module__link__EG3d4"><span class="NavLink-module__title__Q7t0p">Partners</span></a></li></ul></div></li></ul><div class="NavDropdown-module__trailingLinkContainer__VgJGL"><a href="https://github.com/resources" data-analytics-event="{&quot;action&quot;:&quot;view_all_resources&quot;,&quot;tag&quot;:&quot;link&quot;,&quot;context&quot;:&quot;resources&quot;,&quot;location&quot;:&quot;navbar&quot;,&quot;label&quot;:&quot;view_all_resources_link_resources_navbar&quot;}" class="NavLink-module__link__EG3d4"><span class="NavLink-module__title__Q7t0p">View all resources</span><svg aria-hidden="true" focusable="false" class="octicon octicon-chevron-right NavLink-module__arrowIcon__amekg" viewBox="0 0 16 16" width="16" height="16" fill="currentColor" display="inline-block" overflow="visible" style="vertical-align:text-bottom"><path d="M6.22 3.22a.75.75 0 0 1 1.06 0l4.25 4.25a.75.75 0 0 1 0 1.06l-4.25 4.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042L9.94 8 6.22 4.28a.75.75 0 0 1 0-1.06Z"></path></svg></a></div></div></div></li><li><div class="NavDropdown-module__container__l2YeI js-details-container js-header-menu-item"><button type="button" class="NavDropdown-module__button__PEHWX js-details-target" aria-expanded="false">Open Source<svg aria-hidden="true" focusable="false" class="octicon octicon-chevron-right NavDropdown-module__buttonIcon__Tkl8_" viewBox="0 0 16 16" width="16" height="16" fill="currentColor" display="inline-block" overflow="visible" style="vertical-align:text-bottom"><path d="M6.22 3.22a.75.75 0 0 1 1.06 0l4.25 4.25a.75.75 0 0 1 0 1.06l-4.25 4.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042L9.94 8 6.22 4.28a.75.75 0 0 1 0-1.06Z"></path></svg></button><div class="NavDropdown-module__dropdown__xm1jd"><ul class="NavDropdown-module__list__zuCgG"><li><div class="NavGroup-module__group__W8SqJ"><span class="NavGroup-module__title__Wzxz2">COMMUNITY</span><ul class="NavGroup-module__list__UCOFy"><li><a href="https://github.com/sponsors" data-analytics-event="{&quot;action&quot;:&quot;github_sponsors&quot;,&quot;tag&quot;:&quot;link&quot;,&quot;context&quot;:&quot;open_source&quot;,&quot;location&quot;:&quot;navbar&quot;,&quot;label&quot;:&quot;github_sponsors_link_open_source_navbar&quot;}" class="NavLink-module__link__EG3d4"><div class="NavLink-module__text__XvpLQ"><svg aria-hidden="true" focusable="false" class="octicon octicon-sponsor-tiers NavLink-module__icon__ltGNM" viewBox="0 0 24 24" width="24" height="24" fill="currentColor" display="inline-block" overflow="visible" style="vertical-align:text-bottom"><path d="M16.004 1.25C18.311 1.25 20 3.128 20 5.75c0 2.292-1.23 4.464-3.295 6.485-.481.47-.98.909-1.482 1.31l.265 1.32 1.375 7.5a.75.75 0 0 1-.982.844l-3.512-1.207a.75.75 0 0 0-.488 0L8.37 23.209a.75.75 0 0 1-.982-.844l1.378-7.512.261-1.309c-.5-.4-1-.838-1.481-1.31C5.479 10.215 4.25 8.043 4.25 5.75c0-2.622 1.689-4.5 3.996-4.5 1.55 0 2.947.752 3.832 1.967l.047.067.047-.067a4.726 4.726 0 0 1 3.612-1.962l.22-.005ZM13.89 14.531c-.418.285-.828.542-1.218.77l-.18.103a.75.75 0 0 1-.734 0l-.071-.04-.46-.272c-.282-.173-.573-.36-.868-.562l-.121.605-1.145 6.239 2.3-.79a2.248 2.248 0 0 1 1.284-.054l.18.053 2.299.79-1.141-6.226-.125-.616ZM16.004 2.75c-1.464 0-2.731.983-3.159 2.459-.209.721-1.231.721-1.44 0-.428-1.476-1.695-2.459-3.16-2.459-1.44 0-2.495 1.173-2.495 3 0 1.811 1.039 3.647 2.844 5.412a19.624 19.624 0 0 0 3.734 2.84l-.019-.011-.184-.111.147-.088a19.81 19.81 0 0 0 3.015-2.278l.37-.352C17.46 9.397 18.5 7.561 18.5 5.75c0-1.827-1.055-3-2.496-3Z"></path></svg><span class="NavLink-module__title__Q7t0p">GitHub Sponsors</span><span class="NavLink-module__subtitle__X4gkW">Fund open source developers</span></div></a></li></ul></div></li><li><div class="NavGroup-module__group__W8SqJ"><span class="NavGroup-module__title__Wzxz2">PROGRAMS</span><ul class="NavGroup-module__list__UCOFy"><li><a href="https://securitylab.github.com/" data-analytics-event="{&quot;action&quot;:&quot;security_lab&quot;,&quot;tag&quot;:&quot;link&quot;,&quot;context&quot;:&quot;open_source&quot;,&quot;location&quot;:&quot;navbar&quot;,&quot;label&quot;:&quot;security_lab_link_open_source_navbar&quot;}" class="NavLink-module__link__EG3d4" target="_blank" rel="noreferrer"><span class="NavLink-module__title__Q7t0p">Security Lab</span><svg aria-hidden="true" focusable="false" class="octicon octicon-link-external NavLink-module__externalIcon__eWIry" viewBox="0 0 16 16" width="16" height="16" fill="currentColor" display="inline-block" overflow="visible" style="vertical-align:text-bottom"><path d="M3.75 2h3.5a.75.75 0 0 1 0 1.5h-3.5a.25.25 0 0 0-.25.25v8.5c0 .138.112.25.25.25h8.5a.25.25 0 0 0 .25-.25v-3.5a.75.75 0 0 1 1.5 0v3.5A1.75 1.75 0 0 1 12.25 14h-8.5A1.75 1.75 0 0 1 2 12.25v-8.5C2 2.784 2.784 2 3.75 2Zm6.854-1h4.146a.25.25 0 0 1 .25.25v4.146a.25.25 0 0 1-.427.177L13.03 4.03 9.28 7.78a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042l3.75-3.75-1.543-1.543A.25.25 0 0 1 10.604 1Z"></path></svg></a></li><li><a href="https://maintainers.github.com/" data-analytics-event="{&quot;action&quot;:&quot;maintainer_community&quot;,&quot;tag&quot;:&quot;link&quot;,&quot;context&quot;:&quot;open_source&quot;,&quot;location&quot;:&quot;navbar&quot;,&quot;label&quot;:&quot;maintainer_community_link_open_source_navbar&quot;}" class="NavLink-module__link__EG3d4" target="_blank" rel="noreferrer"><span class="NavLink-module__title__Q7t0p">Maintainer Community</span><svg aria-hidden="true" focusable="false" class="octicon octicon-link-external NavLink-module__externalIcon__eWIry" viewBox="0 0 16 16" width="16" height="16" fill="currentColor" display="inline-block" overflow="visible" style="vertical-align:text-bottom"><path d="M3.75 2h3.5a.75.75 0 0 1 0 1.5h-3.5a.25.25 0 0 0-.25.25v8.5c0 .138.112.25.25.25h8.5a.25.25 0 0 0 .25-.25v-3.5a.75.75 0 0 1 1.5 0v3.5A1.75 1.75 0 0 1 12.25 14h-8.5A1.75 1.75 0 0 1 2 12.25v-8.5C2 2.784 2.784 2 3.75 2Zm6.854-1h4.146a.25.25 0 0 1 .25.25v4.146a.25.25 0 0 1-.427.177L13.03 4.03 9.28 7.78a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042l3.75-3.75-1.543-1.543A.25.25 0 0 1 10.604 1Z"></path></svg></a></li><li><a href="https://github.com/accelerator" data-analytics-event="{&quot;action&quot;:&quot;accelerator&quot;,&quot;tag&quot;:&quot;link&quot;,&quot;context&quot;:&quot;open_source&quot;,&quot;location&quot;:&quot;navbar&quot;,&quot;label&quot;:&quot;accelerator_link_open_source_navbar&quot;}" class="NavLink-module__link__EG3d4"><span class="NavLink-module__title__Q7t0p">Accelerator</span></a></li><li><a href="https://stars.github.com/" data-analytics-event="{&quot;action&quot;:&quot;github_stars&quot;,&quot;tag&quot;:&quot;link&quot;,&quot;context&quot;:&quot;open_source&quot;,&quot;location&quot;:&quot;navbar&quot;,&quot;label&quot;:&quot;github_stars_link_open_source_navbar&quot;}" class="NavLink-module__link__EG3d4" target="_blank" rel="noreferrer"><span class="NavLink-module__title__Q7t0p">GitHub Stars</span><svg aria-hidden="true" focusable="false" class="octicon octicon-link-external NavLink-module__externalIcon__eWIry" viewBox="0 0 16 16" width="16" height="16" fill="currentColor" display="inline-block" overflow="visible" style="vertical-align:text-bottom"><path d="M3.75 2h3.5a.75.75 0 0 1 0 1.5h-3.5a.25.25 0 0 0-.25.25v8.5c0 .138.112.25.25.25h8.5a.25.25 0 0 0 .25-.25v-3.5a.75.75 0 0 1 1.5 0v3.5A1.75 1.75 0 0 1 12.25 14h-8.5A1.75 1.75 0 0 1 2 12.25v-8.5C2 2.784 2.784 2 3.75 2Zm6.854-1h4.146a.25.25 0 0 1 .25.25v4.146a.25.25 0 0 1-.427.177L13.03 4.03 9.28 7.78a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042l3.75-3.75-1.543-1.543A.25.25 0 0 1 10.604 1Z"></path></svg></a></li><li><a href="https://archiveprogram.github.com/" data-analytics-event="{&quot;action&quot;:&quot;archive_program&quot;,&quot;tag&quot;:&quot;link&quot;,&quot;context&quot;:&quot;open_source&quot;,&quot;location&quot;:&quot;navbar&quot;,&quot;label&quot;:&quot;archive_program_link_open_source_navbar&quot;}" class="NavLink-module__link__EG3d4" target="_blank" rel="noreferrer"><span class="NavLink-module__title__Q7t0p">Archive Program</span><svg aria-hidden="true" focusable="false" class="octicon octicon-link-external NavLink-module__externalIcon__eWIry" viewBox="0 0 16 16" width="16" height="16" fill="currentColor" display="inline-block" overflow="visible" style="vertical-align:text-bottom"><path d="M3.75 2h3.5a.75.75 0 0 1 0 1.5h-3.5a.25.25 0 0 0-.25.25v8.5c0 .138.112.25.25.25h8.5a.25.25 0 0 0 .25-.25v-3.5a.75.75 0 0 1 1.5 0v3.5A1.75 1.75 0 0 1 12.25 14h-8.5A1.75 1.75 0 0 1 2 12.25v-8.5C2 2.784 2.784 2 3.75 2Zm6.854-1h4.146a.25.25 0 0 1 .25.25v4.146a.25.25 0 0 1-.427.177L13.03 4.03 9.28 7.78a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042l3.75-3.75-1.543-1.543A.25.25 0 0 1 10.604 1Z"></path></svg></a></li></ul></div></li><li><div class="NavGroup-module__group__W8SqJ"><span class="NavGroup-module__title__Wzxz2">REPOSITORIES</span><ul class="NavGroup-module__list__UCOFy"><li><a href="https://github.com/topics" data-analytics-event="{&quot;action&quot;:&quot;topics&quot;,&quot;tag&quot;:&quot;link&quot;,&quot;context&quot;:&quot;open_source&quot;,&quot;location&quot;:&quot;navbar&quot;,&quot;label&quot;:&quot;topics_link_open_source_navbar&quot;}" class="NavLink-module__link__EG3d4"><span class="NavLink-module__title__Q7t0p">Topics</span></a></li><li><a href="https://github.com/trending" data-analytics-event="{&quot;action&quot;:&quot;trending&quot;,&quot;tag&quot;:&quot;link&quot;,&quot;context&quot;:&quot;open_source&quot;,&quot;location&quot;:&quot;navbar&quot;,&quot;label&quot;:&quot;trending_link_open_source_navbar&quot;}" class="NavLink-module__link__EG3d4"><span class="NavLink-module__title__Q7t0p">Trending</span></a></li><li><a href="https://github.com/collections" data-analytics-event="{&quot;action&quot;:&quot;collections&quot;,&quot;tag&quot;:&quot;link&quot;,&quot;context&quot;:&quot;open_source&quot;,&quot;location&quot;:&quot;navbar&quot;,&quot;label&quot;:&quot;collections_link_open_source_navbar&quot;}" class="NavLink-module__link__EG3d4"><span class="NavLink-module__title__Q7t0p">Collections</span></a></li></ul></div></li></ul></div></div></li><li><div class="NavDropdown-module__container__l2YeI js-details-container js-header-menu-item"><button type="button" class="NavDropdown-module__button__PEHWX js-details-target" aria-expanded="false">Enterprise<svg aria-hidden="true" focusable="false" class="octicon octicon-chevron-right NavDropdown-module__buttonIcon__Tkl8_" viewBox="0 0 16 16" width="16" height="16" fill="currentColor" display="inline-block" overflow="visible" style="vertical-align:text-bottom"><path d="M6.22 3.22a.75.75 0 0 1 1.06 0l4.25 4.25a.75.75 0 0 1 0 1.06l-4.25 4.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042L9.94 8 6.22 4.28a.75.75 0 0 1 0-1.06Z"></path></svg></button><div class="NavDropdown-module__dropdown__xm1jd"><ul class="NavDropdown-module__list__zuCgG"><li><div class="NavGroup-module__group__W8SqJ"><span class="NavGroup-module__title__Wzxz2">ENTERPRISE SOLUTIONS</span><ul class="NavGroup-module__list__UCOFy"><li><a href="https://github.com/enterprise" data-analytics-event="{&quot;action&quot;:&quot;enterprise_platform&quot;,&quot;tag&quot;:&quot;link&quot;,&quot;context&quot;:&quot;enterprise&quot;,&quot;location&quot;:&quot;navbar&quot;,&quot;label&quot;:&quot;enterprise_platform_link_enterprise_navbar&quot;}" class="NavLink-module__link__EG3d4"><div class="NavLink-module__text__XvpLQ"><svg aria-hidden="true" focusable="false" class="octicon octicon-stack NavLink-module__icon__ltGNM" viewBox="0 0 24 24" width="24" height="24" fill="currentColor" display="inline-block" overflow="visible" style="vertical-align:text-bottom"><path d="M11.063 1.456a1.749 1.749 0 0 1 1.874 0l8.383 5.316a1.751 1.751 0 0 1 0 2.956l-8.383 5.316a1.749 1.749 0 0 1-1.874 0L2.68 9.728a1.751 1.751 0 0 1 0-2.956Zm1.071 1.267a.25.25 0 0 0-.268 0L3.483 8.039a.25.25 0 0 0 0 .422l8.383 5.316a.25.25 0 0 0 .268 0l8.383-5.316a.25.25 0 0 0 0-.422Z"></path><path d="M1.867 12.324a.75.75 0 0 1 1.035-.232l8.964 5.685a.25.25 0 0 0 .268 0l8.964-5.685a.75.75 0 0 1 .804 1.267l-8.965 5.685a1.749 1.749 0 0 1-1.874 0l-8.965-5.685a.75.75 0 0 1-.231-1.035Z"></path><path d="M1.867 16.324a.75.75 0 0 1 1.035-.232l8.964 5.685a.25.25 0 0 0 .268 0l8.964-5.685a.75.75 0 0 1 .804 1.267l-8.965 5.685a1.749 1.749 0 0 1-1.874 0l-8.965-5.685a.75.75 0 0 1-.231-1.035Z"></path></svg><span class="NavLink-module__title__Q7t0p">Enterprise platform</span><span class="NavLink-module__subtitle__X4gkW">AI-powered developer platform</span></div></a></li></ul></div></li><li><div class="NavGroup-module__group__W8SqJ"><span class="NavGroup-module__title__Wzxz2">AVAILABLE ADD-ONS</span><ul class="NavGroup-module__list__UCOFy"><li><a href="https://github.com/security/advanced-security" data-analytics-event="{&quot;action&quot;:&quot;github_advanced_security&quot;,&quot;tag&quot;:&quot;link&quot;,&quot;context&quot;:&quot;enterprise&quot;,&quot;location&quot;:&quot;navbar&quot;,&quot;label&quot;:&quot;github_advanced_security_link_enterprise_navbar&quot;}" class="NavLink-module__link__EG3d4"><div class="NavLink-module__text__XvpLQ"><svg aria-hidden="true" focusable="false" class="octicon octicon-shield-check NavLink-module__icon__ltGNM" viewBox="0 0 24 24" width="24" height="24" fill="currentColor" display="inline-block" overflow="visible" style="vertical-align:text-bottom"><path d="M16.53 9.78a.75.75 0 0 0-1.06-1.06L11 13.19l-1.97-1.97a.75.75 0 0 0-1.06 1.06l2.5 2.5a.75.75 0 0 0 1.06 0l5-5Z"></path><path d="m12.54.637 8.25 2.675A1.75 1.75 0 0 1 22 4.976V10c0 6.19-3.771 10.704-9.401 12.83a1.704 1.704 0 0 1-1.198 0C5.77 20.705 2 16.19 2 10V4.976c0-.758.489-1.43 1.21-1.664L11.46.637a1.748 1.748 0 0 1 1.08 0Zm-.617 1.426-8.25 2.676a.249.249 0 0 0-.173.237V10c0 5.46 3.28 9.483 8.43 11.426a.199.199 0 0 0 .14 0C17.22 19.483 20.5 15.461 20.5 10V4.976a.25.25 0 0 0-.173-.237l-8.25-2.676a.253.253 0 0 0-.154 0Z"></path></svg><span class="NavLink-module__title__Q7t0p">GitHub Advanced Security</span><span class="NavLink-module__subtitle__X4gkW">Enterprise-grade security features</span></div></a></li><li><a href="https://github.com/features/copilot/copilot-business" data-analytics-event="{&quot;action&quot;:&quot;copilot_for_business&quot;,&quot;tag&quot;:&quot;link&quot;,&quot;context&quot;:&quot;enterprise&quot;,&quot;location&quot;:&quot;navbar&quot;,&quot;label&quot;:&quot;copilot_for_business_link_enterprise_navbar&quot;}" class="NavLink-module__link__EG3d4"><div class="NavLink-module__text__XvpLQ"><svg aria-hidden="true" focusable="false" class="octicon octicon-copilot NavLink-module__icon__ltGNM" viewBox="0 0 24 24" width="24" height="24" fill="currentColor" display="inline-block" overflow="visible" style="vertical-align:text-bottom"><path d="M23.922 16.992c-.861 1.495-5.859 5.023-11.922 5.023-6.063 0-11.061-3.528-11.922-5.023A.641.641 0 0 1 0 16.736v-2.869a.841.841 0 0 1 .053-.22c.372-.935 1.347-2.292 2.605-2.656.167-.429.414-1.055.644-1.517a10.195 10.195 0 0 1-.052-1.086c0-1.331.282-2.499 1.132-3.368.397-.406.89-.717 1.474-.952 1.399-1.136 3.392-2.093 6.122-2.093 2.731 0 4.767.957 6.166 2.093.584.235 1.077.546 1.474.952.85.869 1.132 2.037 1.132 3.368 0 .368-.014.733-.052 1.086.23.462.477 1.088.644 1.517 1.258.364 2.233 1.721 2.605 2.656a.832.832 0 0 1 .053.22v2.869a.641.641 0 0 1-.078.256ZM12.172 11h-.344a4.323 4.323 0 0 1-.355.508C10.703 12.455 9.555 13 7.965 13c-1.725 0-2.989-.359-3.782-1.259a2.005 2.005 0 0 1-.085-.104L4 11.741v6.585c1.435.779 4.514 2.179 8 2.179 3.486 0 6.565-1.4 8-2.179v-6.585l-.098-.104s-.033.045-.085.104c-.793.9-2.057 1.259-3.782 1.259-1.59 0-2.738-.545-3.508-1.492a4.323 4.323 0 0 1-.355-.508h-.016.016Zm.641-2.935c.136 1.057.403 1.913.878 2.497.442.544 1.134.938 2.344.938 1.573 0 2.292-.337 2.657-.751.384-.435.558-1.15.558-2.361 0-1.14-.243-1.847-.705-2.319-.477-.488-1.319-.862-2.824-1.025-1.487-.161-2.192.138-2.533.529-.269.307-.437.808-.438 1.578v.021c0 .265.021.562.063.893Zm-1.626 0c.042-.331.063-.628.063-.894v-.02c-.001-.77-.169-1.271-.438-1.578-.341-.391-1.046-.69-2.533-.529-1.505.163-2.347.537-2.824 1.025-.462.472-.705 1.179-.705 2.319 0 1.211.175 1.926.558 2.361.365.414 1.084.751 2.657.751 1.21 0 1.902-.394 2.344-.938.475-.584.742-1.44.878-2.497Z"></path><path d="M14.5 14.25a1 1 0 0 1 1 1v2a1 1 0 0 1-2 0v-2a1 1 0 0 1 1-1Zm-5 0a1 1 0 0 1 1 1v2a1 1 0 0 1-2 0v-2a1 1 0 0 1 1-1Z"></path></svg><span class="NavLink-module__title__Q7t0p">Copilot for Business</span><span class="NavLink-module__subtitle__X4gkW">Enterprise-grade AI features</span></div></a></li><li><a href="https://github.com/premium-support" data-analytics-event="{&quot;action&quot;:&quot;premium_support&quot;,&quot;tag&quot;:&quot;link&quot;,&quot;context&quot;:&quot;enterprise&quot;,&quot;location&quot;:&quot;navbar&quot;,&quot;label&quot;:&quot;premium_support_link_enterprise_navbar&quot;}" class="NavLink-module__link__EG3d4"><div class="NavLink-module__text__XvpLQ"><svg aria-hidden="true" focusable="false" class="octicon octicon-comment-discussion NavLink-module__icon__ltGNM" viewBox="0 0 24 24" width="24" height="24" fill="currentColor" display="inline-block" overflow="visible" style="vertical-align:text-bottom"><path d="M1.75 1h12.5c.966 0 1.75.784 1.75 1.75v9.5A1.75 1.75 0 0 1 14.25 14H8.061l-2.574 2.573A1.458 1.458 0 0 1 3 15.543V14H1.75A1.75 1.75 0 0 1 0 12.25v-9.5C0 1.784.784 1 1.75 1ZM1.5 2.75v9.5c0 .138.112.25.25.25h2a.75.75 0 0 1 .75.75v2.19l2.72-2.72a.749.749 0 0 1 .53-.22h6.5a.25.25 0 0 0 .25-.25v-9.5a.25.25 0 0 0-.25-.25H1.75a.25.25 0 0 0-.25.25Z"></path><path d="M22.5 8.75a.25.25 0 0 0-.25-.25h-3.5a.75.75 0 0 1 0-1.5h3.5c.966 0 1.75.784 1.75 1.75v9.5A1.75 1.75 0 0 1 22.25 20H21v1.543a1.457 1.457 0 0 1-2.487 1.03L15.939 20H10.75A1.75 1.75 0 0 1 9 18.25v-1.465a.75.75 0 0 1 1.5 0v1.465c0 .138.112.25.25.25h5.5a.75.75 0 0 1 .53.22l2.72 2.72v-2.19a.75.75 0 0 1 .75-.75h2a.25.25 0 0 0 .25-.25v-9.5Z"></path></svg><span class="NavLink-module__title__Q7t0p">Premium Support</span><span class="NavLink-module__subtitle__X4gkW">Enterprise-grade 24/7 support</span></div></a></li></ul></div></li></ul></div></div></li><li><a href="https://github.com/pricing" data-analytics-event="{&quot;action&quot;:&quot;pricing&quot;,&quot;tag&quot;:&quot;link&quot;,&quot;context&quot;:&quot;pricing&quot;,&quot;location&quot;:&quot;navbar&quot;,&quot;label&quot;:&quot;pricing_link_pricing_navbar&quot;}" class="NavLink-module__link__EG3d4 MarketingNavigation-module__navLink__hUomM"><span class="NavLink-module__title__Q7t0p">Pricing</span></a></li></ul></nav><script type="application/json" id="__PRIMER_DATA__R_0___">{"resolvedServerColorMode":"night"}</script></div>
</react-partial>



        <div class="d-flex flex-column flex-lg-row width-full flex-justify-end flex-lg-items-center text-center tmp-mt-3 tmp-mt-lg-0 text-lg-left tmp-ml-lg-3">
                


<qbsearch-input class="search-input" data-scope="repo:manishvat/gat-traffic-control" data-custom-scopes-path="/search/custom_scopes" data-delete-custom-scopes-csrf="5lHgsxtsNHpCcHVCGJ_5afxHFmxsjvahHUUuc_ozB9wYPiSZe0qpxaYoHxjOIqF5LLeaxUX5Qe01w_Qqa9SYiw" data-max-custom-scopes="10" data-header-redesign-enabled="false" data-initial-value="" data-blackbird-suggestions-path="/search/suggestions" data-jump-to-suggestions-path="/_graphql/GetSuggestedNavigationDestinations" data-current-repository="manishvat/gat-traffic-control" data-current-org="" data-current-owner="manishvat" data-logged-in="false" data-copilot-chat-enabled="false" data-nl-search-enabled="false" data-retain-scroll-position="true" data-catalyst="">
  <div class="search-input-container search-with-dialog position-relative d-flex flex-row flex-items-center tmp-mr-4 rounded" data-action="click:qbsearch-input#searchInputContainerClicked">
      <button type="button" class="header-search-button placeholder input-button form-control d-flex flex-1 flex-self-stretch flex-items-center no-wrap width-full py-0 pl-2 pr-0 text-left border-0 box-shadow-none" data-target="qbsearch-input.inputButton" aria-label="Search or jump to…" aria-haspopup="dialog" placeholder="Search or jump to..." data-hotkey="s,/" autocapitalize="off" data-analytics-event="{&quot;location&quot;:&quot;navbar&quot;,&quot;action&quot;:&quot;searchbar&quot;,&quot;context&quot;:&quot;global&quot;,&quot;tag&quot;:&quot;input&quot;,&quot;label&quot;:&quot;searchbar_input_global_navbar&quot;}" data-action="click:qbsearch-input#handleExpand">
        <div class="mr-2 color-fg-muted">
          <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-search">
    <path d="M10.68 11.74a6 6 0 0 1-7.922-8.982 6 6 0 0 1 8.982 7.922l3.04 3.04a.749.749 0 0 1-.326 1.275.749.749 0 0 1-.734-.215ZM11.5 7a4.499 4.499 0 1 0-8.997 0A4.499 4.499 0 0 0 11.5 7Z"></path>
</svg>
        </div>
        <span class="flex-1" data-target="qbsearch-input.inputButtonText">Search or jump to...</span>
          <div class="d-flex" data-target="qbsearch-input.hotkeyIndicator">
            <svg xmlns="http://www.w3.org/2000/svg" width="22" height="20" aria-hidden="true" class="mr-1"><path fill="none" stroke="#979A9C" opacity=".4" d="M3.5.5h12c1.7 0 3 1.3 3 3v13c0 1.7-1.3 3-3 3h-12c-1.7 0-3-1.3-3-3v-13c0-1.7 1.3-3 3-3z"></path><path fill="#979A9C" d="M11.8 6L8 15.1h-.9L10.8 6h1z"></path></svg>
          </div>
      </button>

    <input type="hidden" name="type" class="js-site-search-type-field">

    
<div class="Overlay--hidden " data-modal-dialog-overlay="">
  <modal-dialog data-action="close:qbsearch-input#handleClose cancel:qbsearch-input#handleClose" data-target="qbsearch-input.searchSuggestionsDialog" role="dialog" id="search-suggestions-dialog" aria-modal="true" aria-labelledby="search-suggestions-dialog-header" data-view-component="true" class="Overlay Overlay--width-large Overlay--height-auto">
      <h1 id="search-suggestions-dialog-header" class="sr-only">Search code, repositories, users, issues, pull requests...</h1>
    <div class="Overlay-body Overlay-body--paddingNone">
      
          <div data-view-component="true">        <div class="search-suggestions position-fixed width-full color-shadow-large border color-fg-default color-bg-default overflow-hidden d-flex flex-column query-builder-container" style="border-radius: 12px;" data-target="qbsearch-input.queryBuilderContainer" hidden="">
          <!-- '"` --><!-- </textarea></xmp> --><form id="query-builder-test-form" action="https://github.com/manishvat/gat-traffic-control/blob/main/4_GAT.py" accept-charset="UTF-8" method="get">
  <query-builder data-target="qbsearch-input.queryBuilder" id="query-builder-query-builder-test" data-filter-key=":" data-view-component="true" class="QueryBuilder search-query-builder" data-min-width="300" data-catalyst="">
    <div class="FormControl FormControl--fullWidth">
      <label id="query-builder-test-label" for="query-builder-test" class="FormControl-label sr-only">
        Search
      </label>
      <div class="QueryBuilder-StyledInput width-fit " data-target="query-builder.styledInput">
          <span id="query-builder-test-leadingvisual-wrap" class="FormControl-input-leadingVisualWrap QueryBuilder-leadingVisualWrap">
            <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-search FormControl-input-leadingVisual">
    <path d="M10.68 11.74a6 6 0 0 1-7.922-8.982 6 6 0 0 1 8.982 7.922l3.04 3.04a.749.749 0 0 1-.326 1.275.749.749 0 0 1-.734-.215ZM11.5 7a4.499 4.499 0 1 0-8.997 0A4.499 4.499 0 0 0 11.5 7Z"></path>
</svg>
          </span>
        <div data-target="query-builder.styledInputContainer" class="QueryBuilder-StyledInputContainer">
          <div aria-hidden="true" class="QueryBuilder-StyledInputContent" data-target="query-builder.styledInputContent"></div>
          <div class="QueryBuilder-InputWrapper">
            <div aria-hidden="true" class="QueryBuilder-Sizer" data-target="query-builder.sizer"><span></span></div>
            <input id="query-builder-test" name="query-builder-test" value="" autocomplete="off" type="text" role="combobox" spellcheck="false" aria-expanded="false" aria-describedby="validation-c4aa8868-85ee-4e0f-a2f5-d08d5daec230" data-target="query-builder.input" data-action="
          input:query-builder#inputChange
          blur:query-builder#inputBlur
          keydown:query-builder#inputKeydown
          focus:query-builder#inputFocus
        " data-view-component="true" class="FormControl-input QueryBuilder-Input FormControl-medium" aria-controls="query-builder-test-results" aria-autocomplete="list" aria-haspopup="listbox" style="width: 300px;">
          </div>
        </div>
          <span data-target="query-builder.clearButton" hidden="">
            <span class="sr-only" id="query-builder-test-clear">Clear</span>
            <button role="button" id="query-builder-test-clear-button" aria-labelledby="query-builder-test-clear query-builder-test-label" data-action="
                  click:query-builder#clear
                  focus:query-builder#clearButtonFocus
                  blur:query-builder#clearButtonBlur
                " variant="small" type="button" data-view-component="true" class="Button Button--iconOnly Button--invisible Button--medium mr-1 tmp-mr-1 px-2 tmp-px-2 py-0 tmp-py-0 d-flex flex-items-center rounded-1 color-fg-muted">  <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-x-circle-fill Button-visual">
    <path d="M2.343 13.657A8 8 0 1 1 13.658 2.343 8 8 0 0 1 2.343 13.657ZM6.03 4.97a.751.751 0 0 0-1.042.018.751.751 0 0 0-.018 1.042L6.94 8 4.97 9.97a.749.749 0 0 0 .326 1.275.749.749 0 0 0 .734-.215L8 9.06l1.97 1.97a.749.749 0 0 0 1.275-.326.749.749 0 0 0-.215-.734L9.06 8l1.97-1.97a.749.749 0 0 0-.326-1.275.749.749 0 0 0-.734.215L8 6.94Z"></path>
</svg>
</button>

          </span>
      </div>
      <template id="search-icon"></template>

<template id="code-icon"></template>

<template id="file-code-icon"></template>

<template id="history-icon"></template>

<template id="repo-icon"></template>

<template id="bookmark-icon"></template>

<template id="plus-circle-icon"></template>

<template id="circle-icon"></template>

<template id="trash-icon"></template>

<template id="team-icon"></template>

<template id="project-icon"></template>

<template id="pencil-icon"></template>

<template id="copilot-icon"></template>

<template id="copilot-error-icon"></template>

<template id="workflow-icon"></template>

<template id="book-icon"></template>

<template id="code-review-icon"></template>

<template id="codespaces-icon"></template>

<template id="comment-icon"></template>

<template id="comment-discussion-icon"></template>

<template id="organization-icon"></template>

<template id="rocket-icon"></template>

<template id="shield-check-icon"></template>

<template id="heart-icon"></template>

<template id="server-icon"></template>

<template id="globe-icon"></template>

<template id="issue-opened-icon"></template>

<template id="device-mobile-icon"></template>

<template id="package-icon"></template>

<template id="credit-card-icon"></template>

<template id="play-icon"></template>

<template id="gift-icon"></template>

<template id="code-square-icon"></template>

<template id="device-desktop-icon"></template>

        <div class="position-relative">
                        <ul role="listbox" class="ActionListWrap QueryBuilder-ListWrap" aria-label="Suggestions" data-action="
                combobox-commit:query-builder#comboboxCommit
                mousedown:query-builder#resultsMousedown
              " data-target="query-builder.resultsList" data-persist-list="false" id="query-builder-test-results" tabindex="-1"></ul>

        </div>
      <div class="FormControl-inlineValidation" id="validation-c4aa8868-85ee-4e0f-a2f5-d08d5daec230" hidden="hidden">
        <span class="FormControl-inlineValidation--visual">
          <svg aria-hidden="true" height="12" viewBox="0 0 12 12" version="1.1" width="12" data-view-component="true" class="octicon octicon-alert-fill">
    <path d="M4.855.708c.5-.896 1.79-.896 2.29 0l4.675 8.351a1.312 1.312 0 0 1-1.146 1.954H1.33A1.313 1.313 0 0 1 .183 9.058ZM7 7V3H5v4Zm-1 3a1 1 0 1 0 0-2 1 1 0 0 0 0 2Z"></path>
</svg>
        </span>
        <span></span>
</div>    </div>
    <div data-target="query-builder.screenReaderFeedback" aria-live="polite" aria-atomic="true" class="sr-only"></div>
</query-builder></form>
          <div class="d-flex flex-row color-fg-muted tmp-px-3 text-small color-bg-default search-feedback-prompt">
            <a target="_blank" href="https://docs.github.com/search-github/github-code-search/understanding-github-code-search-syntax" data-view-component="true" class="Link color-fg-accent text-normal ml-2 tmp-ml-2">Search syntax tips</a>            <div class="d-flex flex-1"></div>
          </div>
        </div>
</div>

    </div>
</modal-dialog></div>
  </div>
  <div data-action="click:qbsearch-input#retract" class="dark-backdrop position-fixed" hidden="" data-target="qbsearch-input.darkBackdrop"></div>
  <div class="color-fg-default">
    
<dialog-helper>
  <dialog data-target="qbsearch-input.feedbackDialog" data-action="close:qbsearch-input#handleDialogClose cancel:qbsearch-input#handleDialogClose" id="feedback-dialog" aria-modal="true" aria-labelledby="feedback-dialog-title" aria-describedby="feedback-dialog-description" data-view-component="true" class="Overlay Overlay-whenNarrow Overlay--size-medium Overlay--motion-scaleFade Overlay--disableScroll">
    <div data-view-component="true" class="Overlay-header">
  <div class="Overlay-headerContentWrap">
    <div class="Overlay-titleWrap">
      <h1 class="Overlay-title " id="feedback-dialog-title">
        Provide feedback
      </h1>
        
    </div>
    <div class="Overlay-actionWrap">
      <button data-close-dialog-id="feedback-dialog" aria-label="Close" type="button" data-view-component="true" class="close-button Overlay-closeButton"><svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-x">
    <path d="M3.72 3.72a.75.75 0 0 1 1.06 0L8 6.94l3.22-3.22a.749.749 0 0 1 1.275.326.749.749 0 0 1-.215.734L9.06 8l3.22 3.22a.749.749 0 0 1-.326 1.275.749.749 0 0 1-.734-.215L8 9.06l-3.22 3.22a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042L6.94 8 3.72 4.78a.75.75 0 0 1 0-1.06Z"></path>
</svg></button>
    </div>
  </div>
  
</div>
      <scrollable-region data-labelled-by="feedback-dialog-title" data-catalyst="" style="overflow: auto;">
        <div data-view-component="true" class="Overlay-body">        <!-- '"` --><!-- </textarea></xmp> --><form id="code-search-feedback-form" data-turbo="false" action="https://github.com/search/feedback" accept-charset="UTF-8" method="post"><input type="hidden" data-csrf="true" name="authenticity_token" value="z+1T1haEi8qVgD7wlTqiSBw9N2MrID72YQppGmv+rhMGm8Za4lvKyZE4tgPC1ViXY6gEeawlzQkitEXLjAHAWA==">
          <p>We read every piece of feedback, and take your input very seriously.</p>
          <textarea name="feedback" class="form-control width-full mb-2" style="height: 120px" id="feedback"></textarea>
          <input name="include_email" id="include_email" aria-label="Include my email address so I can be contacted" class="form-control mr-2" type="checkbox">
          <label for="include_email" style="font-weight: normal">Include my email address so I can be contacted</label>
</form></div>
      </scrollable-region>
      <div data-view-component="true" class="Overlay-footer Overlay-footer--alignEnd">          <button data-close-dialog-id="feedback-dialog" type="button" data-view-component="true" class="btn">    Cancel
</button>
          <button form="code-search-feedback-form" data-action="click:qbsearch-input#submitFeedback" type="submit" data-view-component="true" class="btn-primary btn">    Submit feedback
</button>
</div>
</dialog></dialog-helper>

    <custom-scopes data-target="qbsearch-input.customScopesManager" data-catalyst="">
    
<dialog-helper>
  <dialog data-target="custom-scopes.customScopesModalDialog" data-action="close:qbsearch-input#handleDialogClose cancel:qbsearch-input#handleDialogClose" id="custom-scopes-dialog" aria-modal="true" aria-labelledby="custom-scopes-dialog-title" aria-describedby="custom-scopes-dialog-description" data-view-component="true" class="Overlay Overlay-whenNarrow Overlay--size-medium Overlay--motion-scaleFade Overlay--disableScroll">
    <div data-view-component="true" class="Overlay-header Overlay-header--divided">
  <div class="Overlay-headerContentWrap">
    <div class="Overlay-titleWrap">
      <h1 class="Overlay-title " id="custom-scopes-dialog-title">
        Saved searches
      </h1>
        <h2 id="custom-scopes-dialog-description" class="Overlay-description">Use saved searches to filter your results more quickly</h2>
    </div>
    <div class="Overlay-actionWrap">
      <button data-close-dialog-id="custom-scopes-dialog" aria-label="Close" type="button" data-view-component="true" class="close-button Overlay-closeButton"><svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-x">
    <path d="M3.72 3.72a.75.75 0 0 1 1.06 0L8 6.94l3.22-3.22a.749.749 0 0 1 1.275.326.749.749 0 0 1-.215.734L9.06 8l3.22 3.22a.749.749 0 0 1-.326 1.275.749.749 0 0 1-.734-.215L8 9.06l-3.22 3.22a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042L6.94 8 3.72 4.78a.75.75 0 0 1 0-1.06Z"></path>
</svg></button>
    </div>
  </div>
  
</div>
      <scrollable-region data-labelled-by="custom-scopes-dialog-title" data-catalyst="" style="overflow: auto;">
        <div data-view-component="true" class="Overlay-body">        <div data-target="custom-scopes.customScopesModalDialogFlash"></div>

        <div hidden="" class="create-custom-scope-form" data-target="custom-scopes.createCustomScopeForm">
        <!-- '"` --><!-- </textarea></xmp> --><form id="custom-scopes-dialog-form" data-turbo="false" action="https://github.com/search/custom_scopes" accept-charset="UTF-8" method="post"><input type="hidden" data-csrf="true" name="authenticity_token" value="05SF2dPQcwkdn5C7ICWgPbJLWbujcAPDsXFb9ajmqrafhR3AVIvefmaM73baf8lg/HJCfMqda+G2b/BN6NfSJA==">
          <div data-target="custom-scopes.customScopesModalDialogFlash"></div>

          <input type="hidden" id="custom_scope_id" name="custom_scope_id" data-target="custom-scopes.customScopesIdField">

          <div class="form-group">
            <label for="custom_scope_name">Name</label>
            <auto-check src="/search/custom_scopes/check_name" required="">
              <input type="text" name="custom_scope_name" id="custom_scope_name" data-target="custom-scopes.customScopesNameField" class="form-control" autocomplete="off" placeholder="github-ruby" required="" maxlength="50" spellcheck="false">
              <input type="hidden" data-csrf="true" value="lt69/CvVPqSekcLhbIjWLuvbjZIDteW7B3lwDLWjNkwtHWONfGt3Sz+QhHVahT0tUi6cCDB7OvzON6AHOz8KgQ==">
            </auto-check>
          </div>

          <div class="form-group">
            <label for="custom_scope_query">Query</label>
            <input type="text" name="custom_scope_query" id="custom_scope_query" data-target="custom-scopes.customScopesQueryField" class="form-control" autocomplete="off" placeholder="(repo:mona/a OR repo:mona/b) AND lang:python" required="" maxlength="500">
          </div>

          <p class="text-small color-fg-muted">
            To see all available qualifiers, see our <a class="Link--inTextBlock" href="https://docs.github.com/search-github/github-code-search/understanding-github-code-search-syntax">documentation</a>.
          </p>
</form>        </div>

        <div data-target="custom-scopes.manageCustomScopesForm">
          <div data-target="custom-scopes.list"></div>
        </div>

</div>
      </scrollable-region>
      <div data-view-component="true" class="Overlay-footer Overlay-footer--alignEnd Overlay-footer--divided">          <button data-action="click:custom-scopes#customScopesCancel" type="button" data-view-component="true" class="btn">    Cancel
</button>
          <button form="custom-scopes-dialog-form" data-action="click:custom-scopes#customScopesSubmit" data-target="custom-scopes.customScopesSubmitButton" type="submit" data-view-component="true" class="btn-primary btn">    Create saved search
</button>
</div>
</dialog></dialog-helper>
    </custom-scopes>
  </div>
</qbsearch-input>


            <div class="position-relative HeaderMenu-link-wrap d-lg-inline-block">
              <a href="https://github.com/login?return_to=https%3A%2F%2Fgithub.com%2Fmanishvat%2Fgat-traffic-control" class="HeaderMenu-link HeaderMenu-link--sign-in HeaderMenu-button flex-shrink-0 no-underline d-none d-lg-inline-flex border border-lg-0 rounded px-2 py-1" style="margin-left: 12px;" data-hydro-click="{&quot;event_type&quot;:&quot;authentication.click&quot;,&quot;payload&quot;:{&quot;location_in_page&quot;:&quot;site header menu&quot;,&quot;repository_id&quot;:null,&quot;auth_type&quot;:&quot;SIGN_UP&quot;,&quot;originating_url&quot;:&quot;https://github.com/manishvat/gat-traffic-control&quot;,&quot;user_id&quot;:null}}" data-hydro-click-hmac="382b63272136199597428c7c44be78cadb95f433df7bce65c99454363df4fc0a" data-analytics-event="{&quot;category&quot;:&quot;Marketing nav&quot;,&quot;action&quot;:&quot;click to go to homepage&quot;,&quot;label&quot;:&quot;ref_page:Marketing;ref_cta:Sign in;ref_loc:Header&quot;}">
                Sign in
              </a>
            </div>

              <a href="https://github.com/signup?ref_cta=Sign+up&amp;ref_loc=header+logged+out&amp;ref_page=%2F%3Cuser-name%3E%2F%3Crepo-name%3E&amp;source=header-repo&amp;source_repo=manishvat%2Fgat-traffic-control" class="HeaderMenu-link HeaderMenu-link--sign-up HeaderMenu-button flex-shrink-0 d-flex d-lg-inline-flex no-underline border color-border-default rounded px-2 py-1" data-hydro-click="{&quot;event_type&quot;:&quot;authentication.click&quot;,&quot;payload&quot;:{&quot;location_in_page&quot;:&quot;site header menu&quot;,&quot;repository_id&quot;:null,&quot;auth_type&quot;:&quot;SIGN_UP&quot;,&quot;originating_url&quot;:&quot;https://github.com/manishvat/gat-traffic-control&quot;,&quot;user_id&quot;:null}}" data-hydro-click-hmac="382b63272136199597428c7c44be78cadb95f433df7bce65c99454363df4fc0a" data-analytics-event="{&quot;category&quot;:&quot;Sign up&quot;,&quot;action&quot;:&quot;click to sign up for account&quot;,&quot;label&quot;:&quot;ref_page:/&lt;user-name&gt;/&lt;repo-name&gt;;ref_cta:Sign up;ref_loc:header logged out&quot;}">
                Sign up
              </a>

                <div class="AppHeader-appearanceSettings">
    <react-partial-anchor data-catalyst="">
      <button data-target="react-partial-anchor.anchor" id="icon-button-19a20803-fd29-4cb2-8c67-0f1eec11ac89" aria-labelledby="tooltip-669ddb0e-d90a-48b1-aea2-b9f6e13834d2" type="button" data-view-component="true" class="Button Button--iconOnly Button--invisible Button--medium AppHeader-button HeaderMenu-link border" aria-expanded="false" aria-haspopup="true">  <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-sliders Button-visual">
    <path d="M15 2.75a.75.75 0 0 1-.75.75h-4a.75.75 0 0 1 0-1.5h4a.75.75 0 0 1 .75.75Zm-8.5.75v1.25a.75.75 0 0 0 1.5 0v-4a.75.75 0 0 0-1.5 0V2H1.75a.75.75 0 0 0 0 1.5H6.5Zm1.25 5.25a.75.75 0 0 0 0-1.5h-6a.75.75 0 0 0 0 1.5h6ZM15 8a.75.75 0 0 1-.75.75H11.5V10a.75.75 0 1 1-1.5 0V6a.75.75 0 0 1 1.5 0v1.25h2.75A.75.75 0 0 1 15 8Zm-9 5.25v-2a.75.75 0 0 0-1.5 0v1.25H1.75a.75.75 0 0 0 0 1.5H4.5v1.25a.75.75 0 0 0 1.5 0v-2Zm9 0a.75.75 0 0 1-.75.75h-6a.75.75 0 0 1 0-1.5h6a.75.75 0 0 1 .75.75Z"></path>
</svg>
</button><tool-tip id="tooltip-669ddb0e-d90a-48b1-aea2-b9f6e13834d2" for="icon-button-19a20803-fd29-4cb2-8c67-0f1eec11ac89" popover="manual" data-direction="s" data-type="label" data-view-component="true" class="sr-only position-absolute" aria-hidden="true" role="tooltip"><template shadowrootmode="open"><style>
      :host {
        --tooltip-top: var(--tool-tip-position-top, 0);
        --tooltip-left: var(--tool-tip-position-left, 0);
        padding: var(--overlay-paddingBlock-condensed) var(--overlay-padding-condensed) !important;
        font: var(--text-body-shorthand-small);
        color: var(--tooltip-fgColor, var(--fgColor-onEmphasis)) !important;
        text-align: center;
        text-decoration: none;
        text-shadow: none;
        text-transform: none;
        letter-spacing: normal;
        word-wrap: break-word;
        white-space: pre;
        background: var(--tooltip-bgColor, var(--bgColor-emphasis)) !important;
        border-radius: var(--borderRadius-medium);
        border: 0 !important;
        opacity: 0;
        max-width: min(var(--overlay-width-small), 100vw);
        word-wrap: break-word;
        white-space: normal;
        width: max-content !important;
        inset: var(--tooltip-top) auto auto var(--tooltip-left) !important;
        overflow: visible !important;
        text-wrap: balance;
      }

      :host(:is(.tooltip-n, .tooltip-nw, .tooltip-ne)) {
        --tooltip-top: calc(var(--tool-tip-position-top, 0) - var(--overlay-offset, 0.25rem));
        --tooltip-left: var(--tool-tip-position-left);
      }

      :host(:is(.tooltip-s, .tooltip-sw, .tooltip-se)) {
        --tooltip-top: calc(var(--tool-tip-position-top, 0) + var(--overlay-offset, 0.25rem));
        --tooltip-left: var(--tool-tip-position-left);
      }

      :host(.tooltip-w) {
        --tooltip-top: var(--tool-tip-position-top);
        --tooltip-left: calc(var(--tool-tip-position-left, 0) - var(--overlay-offset, 0.25rem));
      }

      :host(.tooltip-e) {
        --tooltip-top: var(--tool-tip-position-top);
        --tooltip-left: calc(var(--tool-tip-position-left, 0) + var(--overlay-offset, 0.25rem));
      }

      :host:after{
        position: absolute;
        display: block;
        right: 0;
        left: 0;
        height: var(--overlay-offset, 0.25rem);
        content: "";
      }

      :host(.tooltip-s):after,
      :host(.tooltip-se):after,
      :host(.tooltip-sw):after {
        bottom: 100%
      }

      :host(.tooltip-n):after,
      :host(.tooltip-ne):after,
      :host(.tooltip-nw):after {
        top: 100%;
      }

      @keyframes tooltip-appear {
        from {
          opacity: 0;
        }
        to {
          opacity: 1;
        }
      }

      :host(:popover-open),
      :host(:popover-open):before {
        animation-name: tooltip-appear;
        animation-duration: .1s;
        animation-fill-mode: forwards;
        animation-timing-function: ease-in;
      }

      :host(.\:popover-open) {
        animation-name: tooltip-appear;
        animation-duration: .1s;
        animation-fill-mode: forwards;
        animation-timing-function: ease-in;
      }

      @media (forced-colors: active) {
        :host {
          outline: solid 1px transparent;
        }

        :host:before {
          display: none;
        }
      }
    </style><slot></slot></template>Appearance settings</tool-tip>

      
    
        <link crossorigin="anonymous" media="all" rel="stylesheet" href="./4_GAT_files/primer-react-css.a2f41fc3ac05567f.module.css">
<link crossorigin="anonymous" media="all" rel="stylesheet" href="./4_GAT_files/appearance-settings.6f14ff9973550c9e.module.css">

<react-partial partial-name="appearance-settings" data-ssr="false" data-attempted-ssr="false" data-react-profiling="false" data-catalyst="" class="loaded">
  
  <script type="application/json" data-target="react-partial.embeddedData">{"props":{}}</script>
  <div data-target="react-partial.reactRoot"><script type="application/json" id="__PRIMER_DATA__r_b___">{"resolvedServerColorMode":"night"}</script></div>
</react-partial>


      </react-partial-anchor>
  </div>

          <button type="button" class="sr-only js-header-menu-focus-trap d-block d-lg-none">Resetting focus</button>
        </div>
      </div>
    </div>
  </div>
</header>

      <div hidden="hidden" data-view-component="true" class="js-stale-session-flash stale-session-flash flash flash-warn flash-full">
  
        <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-alert">
    <path d="M6.457 1.047c.659-1.234 2.427-1.234 3.086 0l6.082 11.378A1.75 1.75 0 0 1 14.082 15H1.918a1.75 1.75 0 0 1-1.543-2.575Zm1.763.707a.25.25 0 0 0-.44 0L1.698 13.132a.25.25 0 0 0 .22.368h12.164a.25.25 0 0 0 .22-.368Zm.53 3.996v2.5a.75.75 0 0 1-1.5 0v-2.5a.75.75 0 0 1 1.5 0ZM9 11a1 1 0 1 1-2 0 1 1 0 0 1 2 0Z"></path>
</svg>
        <span class="js-stale-session-flash-signed-in" hidden="">You signed in with another tab or window. <a class="Link--inTextBlock" href="https://github.com/manishvat/gat-traffic-control/blob/main/4_GAT.py">Reload</a> to refresh your session.</span>
        <span class="js-stale-session-flash-signed-out" hidden="">You signed out in another tab or window. <a class="Link--inTextBlock" href="https://github.com/manishvat/gat-traffic-control/blob/main/4_GAT.py">Reload</a> to refresh your session.</span>
        <span class="js-stale-session-flash-switched" hidden="">You switched accounts on another tab or window. <a class="Link--inTextBlock" href="https://github.com/manishvat/gat-traffic-control/blob/main/4_GAT.py">Reload</a> to refresh your session.</span>

    <button id="icon-button-70c0f75d-a6a5-4676-97f6-1f17899675ae" aria-labelledby="tooltip-22111f9f-1770-4e65-ab4a-3187ee164acd" type="button" data-view-component="true" class="Button Button--iconOnly Button--invisible Button--medium flash-close js-flash-close">  <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-x Button-visual">
    <path d="M3.72 3.72a.75.75 0 0 1 1.06 0L8 6.94l3.22-3.22a.749.749 0 0 1 1.275.326.749.749 0 0 1-.215.734L9.06 8l3.22 3.22a.749.749 0 0 1-.326 1.275.749.749 0 0 1-.734-.215L8 9.06l-3.22 3.22a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042L6.94 8 3.72 4.78a.75.75 0 0 1 0-1.06Z"></path>
</svg>
</button><tool-tip id="tooltip-22111f9f-1770-4e65-ab4a-3187ee164acd" for="icon-button-70c0f75d-a6a5-4676-97f6-1f17899675ae" popover="manual" data-direction="s" data-type="label" data-view-component="true" class="sr-only position-absolute" aria-hidden="true" role="tooltip"><template shadowrootmode="open"><style>
      :host {
        --tooltip-top: var(--tool-tip-position-top, 0);
        --tooltip-left: var(--tool-tip-position-left, 0);
        padding: var(--overlay-paddingBlock-condensed) var(--overlay-padding-condensed) !important;
        font: var(--text-body-shorthand-small);
        color: var(--tooltip-fgColor, var(--fgColor-onEmphasis)) !important;
        text-align: center;
        text-decoration: none;
        text-shadow: none;
        text-transform: none;
        letter-spacing: normal;
        word-wrap: break-word;
        white-space: pre;
        background: var(--tooltip-bgColor, var(--bgColor-emphasis)) !important;
        border-radius: var(--borderRadius-medium);
        border: 0 !important;
        opacity: 0;
        max-width: min(var(--overlay-width-small), 100vw);
        word-wrap: break-word;
        white-space: normal;
        width: max-content !important;
        inset: var(--tooltip-top) auto auto var(--tooltip-left) !important;
        overflow: visible !important;
        text-wrap: balance;
      }

      :host(:is(.tooltip-n, .tooltip-nw, .tooltip-ne)) {
        --tooltip-top: calc(var(--tool-tip-position-top, 0) - var(--overlay-offset, 0.25rem));
        --tooltip-left: var(--tool-tip-position-left);
      }

      :host(:is(.tooltip-s, .tooltip-sw, .tooltip-se)) {
        --tooltip-top: calc(var(--tool-tip-position-top, 0) + var(--overlay-offset, 0.25rem));
        --tooltip-left: var(--tool-tip-position-left);
      }

      :host(.tooltip-w) {
        --tooltip-top: var(--tool-tip-position-top);
        --tooltip-left: calc(var(--tool-tip-position-left, 0) - var(--overlay-offset, 0.25rem));
      }

      :host(.tooltip-e) {
        --tooltip-top: var(--tool-tip-position-top);
        --tooltip-left: calc(var(--tool-tip-position-left, 0) + var(--overlay-offset, 0.25rem));
      }

      :host:after{
        position: absolute;
        display: block;
        right: 0;
        left: 0;
        height: var(--overlay-offset, 0.25rem);
        content: "";
      }

      :host(.tooltip-s):after,
      :host(.tooltip-se):after,
      :host(.tooltip-sw):after {
        bottom: 100%
      }

      :host(.tooltip-n):after,
      :host(.tooltip-ne):after,
      :host(.tooltip-nw):after {
        top: 100%;
      }

      @keyframes tooltip-appear {
        from {
          opacity: 0;
        }
        to {
          opacity: 1;
        }
      }

      :host(:popover-open),
      :host(:popover-open):before {
        animation-name: tooltip-appear;
        animation-duration: .1s;
        animation-fill-mode: forwards;
        animation-timing-function: ease-in;
      }

      :host(.\:popover-open) {
        animation-name: tooltip-appear;
        animation-duration: .1s;
        animation-fill-mode: forwards;
        animation-timing-function: ease-in;
      }

      @media (forced-colors: active) {
        :host {
          outline: solid 1px transparent;
        }

        :host:before {
          display: none;
        }
      }
    </style><slot></slot></template>Dismiss alert</tool-tip>


  
</div>
    </div>

  <div id="start-of-content" class="show-on-focus"></div>








    <div id="js-flash-container" class="flash-container" data-turbo-replace="">




  <template class="js-flash-template"></template>
</div>


    






  <div class="application-main " data-commit-hovercards-enabled="" data-discussion-hovercards-enabled="" data-issue-and-pr-hovercards-enabled="" data-project-hovercards-enabled="">
        <div itemscope="" itemtype="http://schema.org/SoftwareSourceCode" class="">
    <main id="js-repo-pjax-container">
      
      






  

  <div id="repository-container-header" class="tmp-pt-3 hide-full-screen" style="background-color: var(--page-header-bgColor, var(--color-page-header-bg));" data-turbo-replace="">

      <div class="d-flex flex-nowrap flex-justify-end tmp-mb-3  tmp-px-3 tmp-px-lg-5" style="gap: 1rem;">

        <div class="flex-auto min-width-0 width-fit">
            
  <div class=" d-flex flex-wrap flex-items-center wb-break-word f3 text-normal">
      <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-repo color-fg-muted mr-2 tmp-mr-2">
    <path d="M2 2.5A2.5 2.5 0 0 1 4.5 0h8.75a.75.75 0 0 1 .75.75v12.5a.75.75 0 0 1-.75.75h-2.5a.75.75 0 0 1 0-1.5h1.75v-2h-8a1 1 0 0 0-.714 1.7.75.75 0 1 1-1.072 1.05A2.495 2.495 0 0 1 2 11.5Zm10.5-1h-8a1 1 0 0 0-1 1v6.708A2.486 2.486 0 0 1 4.5 9h8ZM5 12.25a.25.25 0 0 1 .25-.25h3.5a.25.25 0 0 1 .25.25v3.25a.25.25 0 0 1-.4.2l-1.45-1.087a.249.249 0 0 0-.3 0L5.4 15.7a.25.25 0 0 1-.4-.2Z"></path>
</svg>
    
    <span class="author flex-self-stretch" itemprop="author">
      <a class="url fn" rel="author" data-hovercard-type="user" data-hovercard-url="/users/manishvat/hovercard" data-octo-click="hovercard-link-click" data-octo-dimensions="link_type:self" href="https://github.com/manishvat" aria-keyshortcuts="Alt+ArrowUp">
        manishvat
</a>    </span>
    <span class="mx-1 flex-self-stretch color-fg-muted">/</span>
    <strong itemprop="name" class="mr-2 flex-self-stretch">
      <a data-pjax="#repo-content-pjax-container" data-turbo-frame="repo-content-turbo-frame" href="https://github.com/manishvat/gat-traffic-control">gat-traffic-control</a>
    </strong>

    <span></span><span class="Label Label--secondary v-align-middle mr-1">Public</span>
  </div>


        </div>

        <div id="repository-details-container" class="flex-shrink-0" data-turbo-replace="" style="max-width: 70%;">
            <ul class="pagehead-actions flex-shrink-0 d-none d-md-inline" style="padding: 2px 0;">
    
      

  <li>
            <a href="https://github.com/login?return_to=%2Fmanishvat%2Fgat-traffic-control" rel="nofollow" id="repository-details-watch-button" data-hydro-click="{&quot;event_type&quot;:&quot;authentication.click&quot;,&quot;payload&quot;:{&quot;location_in_page&quot;:&quot;notification subscription menu watch&quot;,&quot;repository_id&quot;:null,&quot;auth_type&quot;:&quot;LOG_IN&quot;,&quot;originating_url&quot;:&quot;https://github.com/manishvat/gat-traffic-control&quot;,&quot;user_id&quot;:null}}" data-hydro-click-hmac="fadf9815b8aba1a95ec91392632441855e81cd933c8f8378dac1da669451e2c2" aria-label="You must be signed in to change notification settings" data-view-component="true" class="btn-sm btn" aria-describedby="tooltip-fb96a1ff-b625-4f7c-8923-e6249b18fe8e">    <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-bell mr-2 tmp-mr-2">
    <path d="M8 16a2 2 0 0 0 1.985-1.75c.017-.137-.097-.25-.235-.25h-3.5c-.138 0-.252.113-.235.25A2 2 0 0 0 8 16ZM3 5a5 5 0 0 1 10 0v2.947c0 .05.015.098.042.139l1.703 2.555A1.519 1.519 0 0 1 13.482 13H2.518a1.516 1.516 0 0 1-1.263-2.36l1.703-2.554A.255.255 0 0 0 3 7.947Zm5-3.5A3.5 3.5 0 0 0 4.5 5v2.947c0 .346-.102.683-.294.97l-1.703 2.556a.017.017 0 0 0-.003.01l.001.006c0 .002.002.004.004.006l.006.004.007.001h10.964l.007-.001.006-.004.004-.006.001-.007a.017.017 0 0 0-.003-.01l-1.703-2.554a1.745 1.745 0 0 1-.294-.97V5A3.5 3.5 0 0 0 8 1.5Z"></path>
</svg>Notifications
</a>    <tool-tip id="tooltip-fb96a1ff-b625-4f7c-8923-e6249b18fe8e" for="repository-details-watch-button" popover="manual" data-direction="s" data-type="description" data-view-component="true" class="sr-only position-absolute" role="tooltip"><template shadowrootmode="open"><style>
      :host {
        --tooltip-top: var(--tool-tip-position-top, 0);
        --tooltip-left: var(--tool-tip-position-left, 0);
        padding: var(--overlay-paddingBlock-condensed) var(--overlay-padding-condensed) !important;
        font: var(--text-body-shorthand-small);
        color: var(--tooltip-fgColor, var(--fgColor-onEmphasis)) !important;
        text-align: center;
        text-decoration: none;
        text-shadow: none;
        text-transform: none;
        letter-spacing: normal;
        word-wrap: break-word;
        white-space: pre;
        background: var(--tooltip-bgColor, var(--bgColor-emphasis)) !important;
        border-radius: var(--borderRadius-medium);
        border: 0 !important;
        opacity: 0;
        max-width: min(var(--overlay-width-small), 100vw);
        word-wrap: break-word;
        white-space: normal;
        width: max-content !important;
        inset: var(--tooltip-top) auto auto var(--tooltip-left) !important;
        overflow: visible !important;
        text-wrap: balance;
      }

      :host(:is(.tooltip-n, .tooltip-nw, .tooltip-ne)) {
        --tooltip-top: calc(var(--tool-tip-position-top, 0) - var(--overlay-offset, 0.25rem));
        --tooltip-left: var(--tool-tip-position-left);
      }

      :host(:is(.tooltip-s, .tooltip-sw, .tooltip-se)) {
        --tooltip-top: calc(var(--tool-tip-position-top, 0) + var(--overlay-offset, 0.25rem));
        --tooltip-left: var(--tool-tip-position-left);
      }

      :host(.tooltip-w) {
        --tooltip-top: var(--tool-tip-position-top);
        --tooltip-left: calc(var(--tool-tip-position-left, 0) - var(--overlay-offset, 0.25rem));
      }

      :host(.tooltip-e) {
        --tooltip-top: var(--tool-tip-position-top);
        --tooltip-left: calc(var(--tool-tip-position-left, 0) + var(--overlay-offset, 0.25rem));
      }

      :host:after{
        position: absolute;
        display: block;
        right: 0;
        left: 0;
        height: var(--overlay-offset, 0.25rem);
        content: "";
      }

      :host(.tooltip-s):after,
      :host(.tooltip-se):after,
      :host(.tooltip-sw):after {
        bottom: 100%
      }

      :host(.tooltip-n):after,
      :host(.tooltip-ne):after,
      :host(.tooltip-nw):after {
        top: 100%;
      }

      @keyframes tooltip-appear {
        from {
          opacity: 0;
        }
        to {
          opacity: 1;
        }
      }

      :host(:popover-open),
      :host(:popover-open):before {
        animation-name: tooltip-appear;
        animation-duration: .1s;
        animation-fill-mode: forwards;
        animation-timing-function: ease-in;
      }

      :host(.\:popover-open) {
        animation-name: tooltip-appear;
        animation-duration: .1s;
        animation-fill-mode: forwards;
        animation-timing-function: ease-in;
      }

      @media (forced-colors: active) {
        :host {
          outline: solid 1px transparent;
        }

        :host:before {
          display: none;
        }
      }
    </style><slot></slot></template>You must be signed in to change notification settings</tool-tip>

  </li>

  <li>
          <a icon="repo-forked" id="fork-button" href="https://github.com/login?return_to=%2Fmanishvat%2Fgat-traffic-control" rel="nofollow" data-hydro-click="{&quot;event_type&quot;:&quot;authentication.click&quot;,&quot;payload&quot;:{&quot;location_in_page&quot;:&quot;repo details fork button&quot;,&quot;repository_id&quot;:1169143393,&quot;auth_type&quot;:&quot;LOG_IN&quot;,&quot;originating_url&quot;:&quot;https://github.com/manishvat/gat-traffic-control&quot;,&quot;user_id&quot;:null}}" data-hydro-click-hmac="5609038c910dabe4deccdd58509354a8a985f1944f6db695ff1fd2581f0df335" data-view-component="true" class="btn-sm btn">    <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-repo-forked mr-2 tmp-mr-2">
    <path d="M5 5.372v.878c0 .414.336.75.75.75h4.5a.75.75 0 0 0 .75-.75v-.878a2.25 2.25 0 1 1 1.5 0v.878a2.25 2.25 0 0 1-2.25 2.25h-1.5v2.128a2.251 2.251 0 1 1-1.5 0V8.5h-1.5A2.25 2.25 0 0 1 3.5 6.25v-.878a2.25 2.25 0 1 1 1.5 0ZM5 3.25a.75.75 0 1 0-1.5 0 .75.75 0 0 0 1.5 0Zm6.75.75a.75.75 0 1 0 0-1.5.75.75 0 0 0 0 1.5Zm-3 8.75a.75.75 0 1 0-1.5 0 .75.75 0 0 0 1.5 0Z"></path>
</svg>Fork
    <span id="repo-network-counter" data-pjax-replace="true" data-turbo-replace="true" title="0" data-view-component="true" class="Counter">0</span>
</a>
  </li>

  <li>
        <div data-view-component="true" class="BtnGroup d-flex">
        <a href="https://github.com/login?return_to=%2Fmanishvat%2Fgat-traffic-control" rel="nofollow" data-hydro-click="{&quot;event_type&quot;:&quot;authentication.click&quot;,&quot;payload&quot;:{&quot;location_in_page&quot;:&quot;star button&quot;,&quot;repository_id&quot;:1169143393,&quot;auth_type&quot;:&quot;LOG_IN&quot;,&quot;originating_url&quot;:&quot;https://github.com/manishvat/gat-traffic-control&quot;,&quot;user_id&quot;:null}}" data-hydro-click-hmac="2cf289473329c6e22abd6f4f7c7701e3c430bab526ac7dd28fef68a14ea6fcc9" aria-label="You must be signed in to star a repository" data-view-component="true" class="tooltipped tooltipped-sw btn-sm btn">    <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-star v-align-text-bottom d-inline-block mr-2 tmp-mr-2">
    <path d="M8 .25a.75.75 0 0 1 .673.418l1.882 3.815 4.21.612a.75.75 0 0 1 .416 1.279l-3.046 2.97.719 4.192a.751.751 0 0 1-1.088.791L8 12.347l-3.766 1.98a.75.75 0 0 1-1.088-.79l.72-4.194L.818 6.374a.75.75 0 0 1 .416-1.28l4.21-.611L7.327.668A.75.75 0 0 1 8 .25Zm0 2.445L6.615 5.5a.75.75 0 0 1-.564.41l-3.097.45 2.24 2.184a.75.75 0 0 1 .216.664l-.528 3.084 2.769-1.456a.75.75 0 0 1 .698 0l2.77 1.456-.53-3.084a.75.75 0 0 1 .216-.664l2.24-2.183-3.096-.45a.75.75 0 0 1-.564-.41L8 2.694Z"></path>
</svg><span data-view-component="true" class="d-inline">
          Star
</span>          <span id="repo-stars-counter-star" aria-label="0 users starred this repository" data-singular-suffix="user starred this repository" data-plural-suffix="users starred this repository" data-turbo-replace="true" title="0" data-view-component="true" class="Counter js-social-count">0</span>
</a></div>
  </li>

</ul>

        </div>
      </div>

        <div id="responsive-meta-container" data-turbo-replace="">
</div>


          <nav data-pjax="#js-repo-pjax-container" aria-label="Repository" data-view-component="true" class="js-repo-nav js-sidenav-container-pjax js-responsive-underlinenav overflow-hidden UnderlineNav px-3 tmp-px-3 px-md-4 tmp-px-md-4 px-lg-5 tmp-px-lg-5">

  <ul data-view-component="true" class="UnderlineNav-body list-style-none">
      <li data-view-component="true" class="d-inline-flex">
  <a id="code-tab" href="https://github.com/manishvat/gat-traffic-control" data-tab-item="i0code-tab" data-selected-links="repo_source repo_downloads repo_commits repo_releases repo_tags repo_branches repo_packages repo_deployments repo_attestations /manishvat/gat-traffic-control" data-pjax="#repo-content-pjax-container" data-turbo-frame="repo-content-turbo-frame" data-hotkey="g c" data-command-id="repositories:go-to-code" data-react-nav="code-view" data-react-nav-anchor="code-view-repo-link" data-analytics-event="{&quot;category&quot;:&quot;Underline navbar&quot;,&quot;action&quot;:&quot;Click tab&quot;,&quot;label&quot;:&quot;Code&quot;,&quot;target&quot;:&quot;UNDERLINE_NAV.TAB&quot;}" aria-current="page" data-view-component="true" class="UnderlineNav-item no-wrap js-responsive-underlinenav-item js-selected-navigation-item selected">
    
              <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-code UnderlineNav-octicon d-none d-sm-inline">
    <path d="m11.28 3.22 4.25 4.25a.75.75 0 0 1 0 1.06l-4.25 4.25a.749.749 0 0 1-1.275-.326.749.749 0 0 1 .215-.734L13.94 8l-3.72-3.72a.749.749 0 0 1 .326-1.275.749.749 0 0 1 .734.215Zm-6.56 0a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042L2.06 8l3.72 3.72a.749.749 0 0 1-.326 1.275.749.749 0 0 1-.734-.215L.47 8.53a.75.75 0 0 1 0-1.06Z"></path>
</svg>
        <span data-content="Code">Code</span>
          <span id="code-repo-tab-count" data-pjax-replace="" data-turbo-replace="" title="Not available" data-view-component="true" class="Counter"></span>


    
</a></li>
      <li data-view-component="true" class="d-inline-flex">
  <a id="issues-tab" href="https://github.com/manishvat/gat-traffic-control/issues" data-tab-item="i1issues-tab" data-selected-links="repo_issues repo_labels repo_milestones /manishvat/gat-traffic-control/issues" data-pjax="#repo-content-pjax-container" data-turbo-frame="repo-content-turbo-frame" data-hotkey="g i" data-command-id="repositories:go-to-issues" data-react-nav="issues-react" data-analytics-event="{&quot;category&quot;:&quot;Underline navbar&quot;,&quot;action&quot;:&quot;Click tab&quot;,&quot;label&quot;:&quot;Issues&quot;,&quot;target&quot;:&quot;UNDERLINE_NAV.TAB&quot;}" data-view-component="true" class="UnderlineNav-item no-wrap js-responsive-underlinenav-item js-selected-navigation-item">
    
              <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-issue-opened UnderlineNav-octicon d-none d-sm-inline">
    <path d="M8 9.5a1.5 1.5 0 1 0 0-3 1.5 1.5 0 0 0 0 3Z"></path><path d="M8 0a8 8 0 1 1 0 16A8 8 0 0 1 8 0ZM1.5 8a6.5 6.5 0 1 0 13 0 6.5 6.5 0 0 0-13 0Z"></path>
</svg>
        <span data-content="Issues">Issues</span>
          <span id="issues-repo-tab-count" data-pjax-replace="" data-turbo-replace="" title="0" hidden="hidden" data-view-component="true" class="Counter">0</span>


    
</a></li>
      <li data-view-component="true" class="d-inline-flex">
  <a id="pull-requests-tab" href="https://github.com/manishvat/gat-traffic-control/pulls" data-tab-item="i2pull-requests-tab" data-selected-links="repo_pulls checks /manishvat/gat-traffic-control/pulls" data-pjax="#repo-content-pjax-container" data-turbo-frame="repo-content-turbo-frame" data-hotkey="g p" data-command-id="repositories:go-to-pull-requests" data-analytics-event="{&quot;category&quot;:&quot;Underline navbar&quot;,&quot;action&quot;:&quot;Click tab&quot;,&quot;label&quot;:&quot;Pull requests&quot;,&quot;target&quot;:&quot;UNDERLINE_NAV.TAB&quot;}" data-view-component="true" class="UnderlineNav-item no-wrap js-responsive-underlinenav-item js-selected-navigation-item">
    
              <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-git-pull-request UnderlineNav-octicon d-none d-sm-inline">
    <path d="M1.5 3.25a2.25 2.25 0 1 1 3 2.122v5.256a2.251 2.251 0 1 1-1.5 0V5.372A2.25 2.25 0 0 1 1.5 3.25Zm5.677-.177L9.573.677A.25.25 0 0 1 10 .854V2.5h1A2.5 2.5 0 0 1 13.5 5v5.628a2.251 2.251 0 1 1-1.5 0V5a1 1 0 0 0-1-1h-1v1.646a.25.25 0 0 1-.427.177L7.177 3.427a.25.25 0 0 1 0-.354ZM3.75 2.5a.75.75 0 1 0 0 1.5.75.75 0 0 0 0-1.5Zm0 9.5a.75.75 0 1 0 0 1.5.75.75 0 0 0 0-1.5Zm8.25.75a.75.75 0 1 0 1.5 0 .75.75 0 0 0-1.5 0Z"></path>
</svg>
        <span data-content="Pull requests">Pull requests</span>
          <span id="pull-requests-repo-tab-count" data-pjax-replace="" data-turbo-replace="" title="0" hidden="hidden" data-view-component="true" class="Counter">0</span>


    
</a></li>
      <li data-view-component="true" class="d-inline-flex">
  <a id="actions-tab" href="https://github.com/manishvat/gat-traffic-control/actions" data-tab-item="i3actions-tab" data-selected-links="repo_actions /manishvat/gat-traffic-control/actions" data-pjax="#repo-content-pjax-container" data-turbo-frame="repo-content-turbo-frame" data-hotkey="g a" data-command-id="repositories:go-to-actions" data-analytics-event="{&quot;category&quot;:&quot;Underline navbar&quot;,&quot;action&quot;:&quot;Click tab&quot;,&quot;label&quot;:&quot;Actions&quot;,&quot;target&quot;:&quot;UNDERLINE_NAV.TAB&quot;}" data-view-component="true" class="UnderlineNav-item no-wrap js-responsive-underlinenav-item js-selected-navigation-item">
    
              <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-play UnderlineNav-octicon d-none d-sm-inline">
    <path d="M8 0a8 8 0 1 1 0 16A8 8 0 0 1 8 0ZM1.5 8a6.5 6.5 0 1 0 13 0 6.5 6.5 0 0 0-13 0Zm4.879-2.773 4.264 2.559a.25.25 0 0 1 0 .428l-4.264 2.559A.25.25 0 0 1 6 10.559V5.442a.25.25 0 0 1 .379-.215Z"></path>
</svg>
        <span data-content="Actions">Actions</span>
          <span id="actions-repo-tab-count" data-pjax-replace="" data-turbo-replace="" title="Not available" data-view-component="true" class="Counter"></span>


    
</a></li>
      <li data-view-component="true" class="d-inline-flex">
  <a id="projects-tab" href="https://github.com/manishvat/gat-traffic-control/projects" data-tab-item="i4projects-tab" data-selected-links="repo_projects new_repo_project repo_project /manishvat/gat-traffic-control/projects" data-pjax="#repo-content-pjax-container" data-turbo-frame="repo-content-turbo-frame" data-hotkey="g b" data-command-id="repositories:go-to-projects" data-analytics-event="{&quot;category&quot;:&quot;Underline navbar&quot;,&quot;action&quot;:&quot;Click tab&quot;,&quot;label&quot;:&quot;Projects&quot;,&quot;target&quot;:&quot;UNDERLINE_NAV.TAB&quot;}" data-view-component="true" class="UnderlineNav-item no-wrap js-responsive-underlinenav-item js-selected-navigation-item">
    
              <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-table UnderlineNav-octicon d-none d-sm-inline">
    <path d="M0 1.75C0 .784.784 0 1.75 0h12.5C15.216 0 16 .784 16 1.75v12.5A1.75 1.75 0 0 1 14.25 16H1.75A1.75 1.75 0 0 1 0 14.25ZM6.5 6.5v8h7.75a.25.25 0 0 0 .25-.25V6.5Zm8-1.5V1.75a.25.25 0 0 0-.25-.25H6.5V5Zm-13 1.5v7.75c0 .138.112.25.25.25H5v-8ZM5 5V1.5H1.75a.25.25 0 0 0-.25.25V5Z"></path>
</svg>
        <span data-content="Projects">Projects</span>
          <span id="projects-repo-tab-count" data-pjax-replace="" data-turbo-replace="" title="Not available" data-view-component="true" class="Counter"></span>


    
</a></li>
      <li data-view-component="true" class="d-inline-flex">
  <a id="security-and-quality-tab" href="https://github.com/manishvat/gat-traffic-control/security" data-tab-item="i5security-and-quality-tab" data-selected-links="security overview alerts policy token_scanning code_scanning /manishvat/gat-traffic-control/security" data-pjax="#repo-content-pjax-container" data-turbo-frame="repo-content-turbo-frame" data-hotkey="g s" data-command-id="repositories:go-to-security" data-analytics-event="{&quot;category&quot;:&quot;Underline navbar&quot;,&quot;action&quot;:&quot;Click tab&quot;,&quot;label&quot;:&quot;Security and quality&quot;,&quot;target&quot;:&quot;UNDERLINE_NAV.TAB&quot;}" data-view-component="true" class="UnderlineNav-item no-wrap js-responsive-underlinenav-item js-selected-navigation-item">
    
              <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-shield UnderlineNav-octicon d-none d-sm-inline">
    <path d="M7.467.133a1.748 1.748 0 0 1 1.066 0l5.25 1.68A1.75 1.75 0 0 1 15 3.48V7c0 1.566-.32 3.182-1.303 4.682-.983 1.498-2.585 2.813-5.032 3.855a1.697 1.697 0 0 1-1.33 0c-2.447-1.042-4.049-2.357-5.032-3.855C1.32 10.182 1 8.566 1 7V3.48a1.75 1.75 0 0 1 1.217-1.667Zm.61 1.429a.25.25 0 0 0-.153 0l-5.25 1.68a.25.25 0 0 0-.174.238V7c0 1.358.275 2.666 1.057 3.86.784 1.194 2.121 2.34 4.366 3.297a.196.196 0 0 0 .154 0c2.245-.956 3.582-2.104 4.366-3.298C13.225 9.666 13.5 8.36 13.5 7V3.48a.251.251 0 0 0-.174-.237l-5.25-1.68ZM8.75 4.75v3a.75.75 0 0 1-1.5 0v-3a.75.75 0 0 1 1.5 0ZM9 10.5a1 1 0 1 1-2 0 1 1 0 0 1 2 0Z"></path>
</svg>
        <span data-content="Security and quality">Security and quality</span>
          <span id="security-and-quality-repo-tab-count" data-pjax-replace="" data-turbo-replace="" title="0" hidden="hidden" data-view-component="true" class="Counter">0</span>


    
</a></li>
      <li data-view-component="true" class="d-inline-flex">
  <a id="insights-tab" href="https://github.com/manishvat/gat-traffic-control/pulse" data-tab-item="i6insights-tab" data-selected-links="repo_graphs repo_contributors dependency_graph dependabot_updates pulse people community /manishvat/gat-traffic-control/pulse" data-pjax="#repo-content-pjax-container" data-turbo-frame="repo-content-turbo-frame" data-command-id="repositories:go-to-insights" data-analytics-event="{&quot;category&quot;:&quot;Underline navbar&quot;,&quot;action&quot;:&quot;Click tab&quot;,&quot;label&quot;:&quot;Insights&quot;,&quot;target&quot;:&quot;UNDERLINE_NAV.TAB&quot;}" data-view-component="true" class="UnderlineNav-item no-wrap js-responsive-underlinenav-item js-selected-navigation-item">
    
              <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-graph UnderlineNav-octicon d-none d-sm-inline">
    <path d="M1.5 1.75V13.5h13.75a.75.75 0 0 1 0 1.5H.75a.75.75 0 0 1-.75-.75V1.75a.75.75 0 0 1 1.5 0Zm14.28 2.53-5.25 5.25a.75.75 0 0 1-1.06 0L7 7.06 4.28 9.78a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042l3.25-3.25a.75.75 0 0 1 1.06 0L10 7.94l4.72-4.72a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042Z"></path>
</svg>
        <span data-content="Insights">Insights</span>
          <span id="insights-repo-tab-count" data-pjax-replace="" data-turbo-replace="" title="Not available" data-view-component="true" class="Counter"></span>


    
</a></li>
</ul>
    <div style="visibility:hidden;" data-view-component="true" class="UnderlineNav-actions js-responsive-underlinenav-overflow position-absolute pr-3 tmp-pr-3 pr-md-4 tmp-pr-md-4 pr-lg-5 tmp-pr-lg-5 right-0">      <action-menu data-select-variant="none" data-view-component="true" data-catalyst="" data-ready="true">
  <focus-group direction="vertical" mnemonics="" retain="">
    <button id="action-menu-e0ce6b24-ac89-414a-a07a-f8040343ac7d-button" popovertarget="action-menu-e0ce6b24-ac89-414a-a07a-f8040343ac7d-overlay" aria-controls="action-menu-e0ce6b24-ac89-414a-a07a-f8040343ac7d-list" aria-haspopup="true" aria-labelledby="tooltip-05076722-74e2-4165-910b-bb7e42f21d62" type="button" data-view-component="true" class="Button Button--iconOnly Button--secondary Button--medium UnderlineNav-item">  <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-kebab-horizontal Button-visual">
    <path d="M8 9a1.5 1.5 0 1 0 0-3 1.5 1.5 0 0 0 0 3ZM1.5 9a1.5 1.5 0 1 0 0-3 1.5 1.5 0 0 0 0 3Zm13 0a1.5 1.5 0 1 0 0-3 1.5 1.5 0 0 0 0 3Z"></path>
</svg>
</button><tool-tip id="tooltip-05076722-74e2-4165-910b-bb7e42f21d62" for="action-menu-e0ce6b24-ac89-414a-a07a-f8040343ac7d-button" popover="manual" data-direction="s" data-type="label" data-view-component="true" class="sr-only position-absolute" aria-hidden="true" role="tooltip"><template shadowrootmode="open"><style>
      :host {
        --tooltip-top: var(--tool-tip-position-top, 0);
        --tooltip-left: var(--tool-tip-position-left, 0);
        padding: var(--overlay-paddingBlock-condensed) var(--overlay-padding-condensed) !important;
        font: var(--text-body-shorthand-small);
        color: var(--tooltip-fgColor, var(--fgColor-onEmphasis)) !important;
        text-align: center;
        text-decoration: none;
        text-shadow: none;
        text-transform: none;
        letter-spacing: normal;
        word-wrap: break-word;
        white-space: pre;
        background: var(--tooltip-bgColor, var(--bgColor-emphasis)) !important;
        border-radius: var(--borderRadius-medium);
        border: 0 !important;
        opacity: 0;
        max-width: min(var(--overlay-width-small), 100vw);
        word-wrap: break-word;
        white-space: normal;
        width: max-content !important;
        inset: var(--tooltip-top) auto auto var(--tooltip-left) !important;
        overflow: visible !important;
        text-wrap: balance;
      }

      :host(:is(.tooltip-n, .tooltip-nw, .tooltip-ne)) {
        --tooltip-top: calc(var(--tool-tip-position-top, 0) - var(--overlay-offset, 0.25rem));
        --tooltip-left: var(--tool-tip-position-left);
      }

      :host(:is(.tooltip-s, .tooltip-sw, .tooltip-se)) {
        --tooltip-top: calc(var(--tool-tip-position-top, 0) + var(--overlay-offset, 0.25rem));
        --tooltip-left: var(--tool-tip-position-left);
      }

      :host(.tooltip-w) {
        --tooltip-top: var(--tool-tip-position-top);
        --tooltip-left: calc(var(--tool-tip-position-left, 0) - var(--overlay-offset, 0.25rem));
      }

      :host(.tooltip-e) {
        --tooltip-top: var(--tool-tip-position-top);
        --tooltip-left: calc(var(--tool-tip-position-left, 0) + var(--overlay-offset, 0.25rem));
      }

      :host:after{
        position: absolute;
        display: block;
        right: 0;
        left: 0;
        height: var(--overlay-offset, 0.25rem);
        content: "";
      }

      :host(.tooltip-s):after,
      :host(.tooltip-se):after,
      :host(.tooltip-sw):after {
        bottom: 100%
      }

      :host(.tooltip-n):after,
      :host(.tooltip-ne):after,
      :host(.tooltip-nw):after {
        top: 100%;
      }

      @keyframes tooltip-appear {
        from {
          opacity: 0;
        }
        to {
          opacity: 1;
        }
      }

      :host(:popover-open),
      :host(:popover-open):before {
        animation-name: tooltip-appear;
        animation-duration: .1s;
        animation-fill-mode: forwards;
        animation-timing-function: ease-in;
      }

      :host(.\:popover-open) {
        animation-name: tooltip-appear;
        animation-duration: .1s;
        animation-fill-mode: forwards;
        animation-timing-function: ease-in;
      }

      @media (forced-colors: active) {
        :host {
          outline: solid 1px transparent;
        }

        :host:before {
          display: none;
        }
      }
    </style><slot></slot></template>Additional navigation options</tool-tip>


<anchored-position data-target="action-menu.overlay" id="action-menu-e0ce6b24-ac89-414a-a07a-f8040343ac7d-overlay" anchor="action-menu-e0ce6b24-ac89-414a-a07a-f8040343ac7d-button" align="start" side="outside-bottom" anchor-offset="normal" popover="auto" data-view-component="true" style="inset: 36px auto auto 0px;">
  <div data-view-component="true" class="Overlay Overlay--size-auto">
    
      <div data-view-component="true" class="Overlay-body Overlay-body--paddingNone">          <action-list data-catalyst="">
  <div data-view-component="true">
    <ul aria-labelledby="action-menu-e0ce6b24-ac89-414a-a07a-f8040343ac7d-button" id="action-menu-e0ce6b24-ac89-414a-a07a-f8040343ac7d-list" role="menu" data-view-component="true" class="ActionListWrap--inset ActionListWrap">
        <li hidden="" data-menu-item="i0code-tab" data-targets="action-list.items" role="none" data-view-component="true" class="ActionListItem">
    
    
    <a tabindex="-1" id="item-1eb8f56f-67b0-4def-9a71-889af84da925" href="https://github.com/manishvat/gat-traffic-control" role="menuitem" data-view-component="true" class="ActionListContent ActionListContent--visual16">
        <span class="ActionListItem-visual ActionListItem-visual--leading">
          <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-code">
    <path d="m11.28 3.22 4.25 4.25a.75.75 0 0 1 0 1.06l-4.25 4.25a.749.749 0 0 1-1.275-.326.749.749 0 0 1 .215-.734L13.94 8l-3.72-3.72a.749.749 0 0 1 .326-1.275.749.749 0 0 1 .734.215Zm-6.56 0a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042L2.06 8l3.72 3.72a.749.749 0 0 1-.326 1.275.749.749 0 0 1-.734-.215L.47 8.53a.75.75 0 0 1 0-1.06Z"></path>
</svg>
        </span>
      
        <span data-view-component="true" class="ActionListItem-label">
          Code
</span>      
</a>
  
</li>
        <li hidden="" data-menu-item="i1issues-tab" data-targets="action-list.items" role="none" data-view-component="true" class="ActionListItem">
    
    
    <a tabindex="-1" id="item-3b807bb2-eb92-4311-842d-7aa8a1c96a70" href="https://github.com/manishvat/gat-traffic-control/issues" role="menuitem" data-view-component="true" class="ActionListContent ActionListContent--visual16">
        <span class="ActionListItem-visual ActionListItem-visual--leading">
          <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-issue-opened">
    <path d="M8 9.5a1.5 1.5 0 1 0 0-3 1.5 1.5 0 0 0 0 3Z"></path><path d="M8 0a8 8 0 1 1 0 16A8 8 0 0 1 8 0ZM1.5 8a6.5 6.5 0 1 0 13 0 6.5 6.5 0 0 0-13 0Z"></path>
</svg>
        </span>
      
        <span data-view-component="true" class="ActionListItem-label">
          Issues
</span>      
</a>
  
</li>
        <li hidden="" data-menu-item="i2pull-requests-tab" data-targets="action-list.items" role="none" data-view-component="true" class="ActionListItem">
    
    
    <a tabindex="-1" id="item-cded2397-8d01-48fe-a415-e0ccd4f6fbf5" href="https://github.com/manishvat/gat-traffic-control/pulls" role="menuitem" data-view-component="true" class="ActionListContent ActionListContent--visual16">
        <span class="ActionListItem-visual ActionListItem-visual--leading">
          <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-git-pull-request">
    <path d="M1.5 3.25a2.25 2.25 0 1 1 3 2.122v5.256a2.251 2.251 0 1 1-1.5 0V5.372A2.25 2.25 0 0 1 1.5 3.25Zm5.677-.177L9.573.677A.25.25 0 0 1 10 .854V2.5h1A2.5 2.5 0 0 1 13.5 5v5.628a2.251 2.251 0 1 1-1.5 0V5a1 1 0 0 0-1-1h-1v1.646a.25.25 0 0 1-.427.177L7.177 3.427a.25.25 0 0 1 0-.354ZM3.75 2.5a.75.75 0 1 0 0 1.5.75.75 0 0 0 0-1.5Zm0 9.5a.75.75 0 1 0 0 1.5.75.75 0 0 0 0-1.5Zm8.25.75a.75.75 0 1 0 1.5 0 .75.75 0 0 0-1.5 0Z"></path>
</svg>
        </span>
      
        <span data-view-component="true" class="ActionListItem-label">
          Pull requests
</span>      
</a>
  
</li>
        <li hidden="" data-menu-item="i3actions-tab" data-targets="action-list.items" role="none" data-view-component="true" class="ActionListItem">
    
    
    <a tabindex="-1" id="item-3355be37-5714-4de6-95e8-21d1a13a4c75" href="https://github.com/manishvat/gat-traffic-control/actions" role="menuitem" data-view-component="true" class="ActionListContent ActionListContent--visual16">
        <span class="ActionListItem-visual ActionListItem-visual--leading">
          <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-play">
    <path d="M8 0a8 8 0 1 1 0 16A8 8 0 0 1 8 0ZM1.5 8a6.5 6.5 0 1 0 13 0 6.5 6.5 0 0 0-13 0Zm4.879-2.773 4.264 2.559a.25.25 0 0 1 0 .428l-4.264 2.559A.25.25 0 0 1 6 10.559V5.442a.25.25 0 0 1 .379-.215Z"></path>
</svg>
        </span>
      
        <span data-view-component="true" class="ActionListItem-label">
          Actions
</span>      
</a>
  
</li>
        <li hidden="" data-menu-item="i4projects-tab" data-targets="action-list.items" role="none" data-view-component="true" class="ActionListItem">
    
    
    <a tabindex="-1" id="item-9ddaacb8-95dc-4026-b25e-e20a65abec5b" href="https://github.com/manishvat/gat-traffic-control/projects" role="menuitem" data-view-component="true" class="ActionListContent ActionListContent--visual16">
        <span class="ActionListItem-visual ActionListItem-visual--leading">
          <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-table">
    <path d="M0 1.75C0 .784.784 0 1.75 0h12.5C15.216 0 16 .784 16 1.75v12.5A1.75 1.75 0 0 1 14.25 16H1.75A1.75 1.75 0 0 1 0 14.25ZM6.5 6.5v8h7.75a.25.25 0 0 0 .25-.25V6.5Zm8-1.5V1.75a.25.25 0 0 0-.25-.25H6.5V5Zm-13 1.5v7.75c0 .138.112.25.25.25H5v-8ZM5 5V1.5H1.75a.25.25 0 0 0-.25.25V5Z"></path>
</svg>
        </span>
      
        <span data-view-component="true" class="ActionListItem-label">
          Projects
</span>      
</a>
  
</li>
        <li hidden="" data-menu-item="i5security-and-quality-tab" data-targets="action-list.items" role="none" data-view-component="true" class="ActionListItem">
    
    
    <a tabindex="-1" id="item-08078a21-3ee9-4b7b-8386-36b178d2b086" href="https://github.com/manishvat/gat-traffic-control/security" role="menuitem" data-view-component="true" class="ActionListContent ActionListContent--visual16">
        <span class="ActionListItem-visual ActionListItem-visual--leading">
          <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-shield">
    <path d="M7.467.133a1.748 1.748 0 0 1 1.066 0l5.25 1.68A1.75 1.75 0 0 1 15 3.48V7c0 1.566-.32 3.182-1.303 4.682-.983 1.498-2.585 2.813-5.032 3.855a1.697 1.697 0 0 1-1.33 0c-2.447-1.042-4.049-2.357-5.032-3.855C1.32 10.182 1 8.566 1 7V3.48a1.75 1.75 0 0 1 1.217-1.667Zm.61 1.429a.25.25 0 0 0-.153 0l-5.25 1.68a.25.25 0 0 0-.174.238V7c0 1.358.275 2.666 1.057 3.86.784 1.194 2.121 2.34 4.366 3.297a.196.196 0 0 0 .154 0c2.245-.956 3.582-2.104 4.366-3.298C13.225 9.666 13.5 8.36 13.5 7V3.48a.251.251 0 0 0-.174-.237l-5.25-1.68ZM8.75 4.75v3a.75.75 0 0 1-1.5 0v-3a.75.75 0 0 1 1.5 0ZM9 10.5a1 1 0 1 1-2 0 1 1 0 0 1 2 0Z"></path>
</svg>
        </span>
      
        <span data-view-component="true" class="ActionListItem-label">
          Security and quality
</span>      
</a>
  
</li>
        <li hidden="" data-menu-item="i6insights-tab" data-targets="action-list.items" role="none" data-view-component="true" class="ActionListItem">
    
    
    <a tabindex="-1" id="item-b2230c73-47d0-4929-ae79-f53d38d675c8" href="https://github.com/manishvat/gat-traffic-control/pulse" role="menuitem" data-view-component="true" class="ActionListContent ActionListContent--visual16">
        <span class="ActionListItem-visual ActionListItem-visual--leading">
          <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-graph">
    <path d="M1.5 1.75V13.5h13.75a.75.75 0 0 1 0 1.5H.75a.75.75 0 0 1-.75-.75V1.75a.75.75 0 0 1 1.5 0Zm14.28 2.53-5.25 5.25a.75.75 0 0 1-1.06 0L7 7.06 4.28 9.78a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042l3.25-3.25a.75.75 0 0 1 1.06 0L10 7.94l4.72-4.72a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042Z"></path>
</svg>
        </span>
      
        <span data-view-component="true" class="ActionListItem-label">
          Insights
</span>      
</a>
  
</li>
</ul>    
</div></action-list>


</div>
      
</div></anchored-position>  </focus-group>
</action-menu></div>
</nav>

  </div>
  



<turbo-frame id="repo-content-turbo-frame" target="_top" data-turbo-action="advance" class="">
    <div id="repo-content-pjax-container" class="repository-content ">
    



    
      
    








<react-app app-name="code-view" initial-path="/manishvat/gat-traffic-control" style="display: block; min-height: calc(100vh - 64px);" data-attempted-ssr="true" data-ssr="true" data-lazy="false" data-alternate="false" data-data-router-enabled="true" data-react-profiling="false" data-catalyst="" class="loaded">
  
  <script type="application/json" data-target="react-app.embeddedData">{"payload":{"codeViewRepoRoute":{"path":"/","refInfo":{"name":"main","listCacheKey":"v0:1772267809.0","canEdit":false,"refType":"branch","currentOid":"2fb8bda825961fb36fcb98daa5809ab893924799"},"tree":{"items":[{"name":"4_GAT.py","path":"4_GAT.py","contentType":"file"},{"name":"5_GCN.py","path":"5_GCN.py","contentType":"file"},{"name":"6_FTS_New_Network.py","path":"6_FTS_New_Network.py","contentType":"file"},{"name":"New_Network.add.xml","path":"New_Network.add.xml","contentType":"file"},{"name":"New_Network.net copy.xml","path":"New_Network.net copy.xml","contentType":"file"},{"name":"New_Network.net.xml","path":"New_Network.net.xml","contentType":"file"},{"name":"New_Network.netecfg","path":"New_Network.netecfg","contentType":"file"},{"name":"New_Network.rou.xml","path":"New_Network.rou.xml","contentType":"file"},{"name":"New_Network.sumocfg","path":"New_Network.sumocfg","contentType":"file"},{"name":"e2_0.xml","path":"e2_0.xml","contentType":"file"},{"name":"e2_1.xml","path":"e2_1.xml","contentType":"file"},{"name":"e2_10.xml","path":"e2_10.xml","contentType":"file"},{"name":"e2_11.xml","path":"e2_11.xml","contentType":"file"},{"name":"e2_12.xml","path":"e2_12.xml","contentType":"file"},{"name":"e2_13.xml","path":"e2_13.xml","contentType":"file"},{"name":"e2_14.xml","path":"e2_14.xml","contentType":"file"},{"name":"e2_15.xml","path":"e2_15.xml","contentType":"file"},{"name":"e2_16.xml","path":"e2_16.xml","contentType":"file"},{"name":"e2_17.xml","path":"e2_17.xml","contentType":"file"},{"name":"e2_18.xml","path":"e2_18.xml","contentType":"file"},{"name":"e2_19.xml","path":"e2_19.xml","contentType":"file"},{"name":"e2_2.xml","path":"e2_2.xml","contentType":"file"},{"name":"e2_20.xml","path":"e2_20.xml","contentType":"file"},{"name":"e2_21.xml","path":"e2_21.xml","contentType":"file"},{"name":"e2_22.xml","path":"e2_22.xml","contentType":"file"},{"name":"e2_23.xml","path":"e2_23.xml","contentType":"file"},{"name":"e2_24.xml","path":"e2_24.xml","contentType":"file"},{"name":"e2_25.xml","path":"e2_25.xml","contentType":"file"},{"name":"e2_26.xml","path":"e2_26.xml","contentType":"file"},{"name":"e2_27.xml","path":"e2_27.xml","contentType":"file"},{"name":"e2_28.xml","path":"e2_28.xml","contentType":"file"},{"name":"e2_29.xml","path":"e2_29.xml","contentType":"file"},{"name":"e2_3.xml","path":"e2_3.xml","contentType":"file"},{"name":"e2_30.xml","path":"e2_30.xml","contentType":"file"},{"name":"e2_31.xml","path":"e2_31.xml","contentType":"file"},{"name":"e2_32.xml","path":"e2_32.xml","contentType":"file"},{"name":"e2_33.xml","path":"e2_33.xml","contentType":"file"},{"name":"e2_34.xml","path":"e2_34.xml","contentType":"file"},{"name":"e2_35.xml","path":"e2_35.xml","contentType":"file"},{"name":"e2_36.xml","path":"e2_36.xml","contentType":"file"},{"name":"e2_37.xml","path":"e2_37.xml","contentType":"file"},{"name":"e2_38.xml","path":"e2_38.xml","contentType":"file"},{"name":"e2_39.xml","path":"e2_39.xml","contentType":"file"},{"name":"e2_4.xml","path":"e2_4.xml","contentType":"file"},{"name":"e2_40.xml","path":"e2_40.xml","contentType":"file"},{"name":"e2_41.xml","path":"e2_41.xml","contentType":"file"},{"name":"e2_42.xml","path":"e2_42.xml","contentType":"file"},{"name":"e2_43.xml","path":"e2_43.xml","contentType":"file"},{"name":"e2_44.xml","path":"e2_44.xml","contentType":"file"},{"name":"e2_45.xml","path":"e2_45.xml","contentType":"file"},{"name":"e2_46.xml","path":"e2_46.xml","contentType":"file"},{"name":"e2_47.xml","path":"e2_47.xml","contentType":"file"},{"name":"e2_48.xml","path":"e2_48.xml","contentType":"file"},{"name":"e2_49.xml","path":"e2_49.xml","contentType":"file"},{"name":"e2_5.xml","path":"e2_5.xml","contentType":"file"},{"name":"e2_6.xml","path":"e2_6.xml","contentType":"file"},{"name":"e2_7.xml","path":"e2_7.xml","contentType":"file"},{"name":"e2_8.xml","path":"e2_8.xml","contentType":"file"},{"name":"e2_9.xml","path":"e2_9.xml","contentType":"file"},{"name":"igrl_gat_model.h5","path":"igrl_gat_model.h5","contentType":"file"},{"name":"igrl_gcn_model.h5","path":"igrl_gcn_model.h5","contentType":"file"},{"name":"simulation_results.png","path":"simulation_results.png","contentType":"file"}],"totalCount":62,"templateDirectorySuggestionUrl":null,"readme":null,"showBranchInfobar":false},"userNameDisplayConfiguration":null,"treeExpanded":false,"symbolsExpanded":false,"copilotSWEAgentEnabled":false,"isOverview":true,"overview":{"banners":{"shouldRecommendReadme":false,"isPersonalRepo":false,"showUseActionBanner":false,"actionSlug":null,"actionId":null,"showProtectBranchBanner":false,"transactionalMessageBanner":null,"publishBannersInfo":{"dismissActionNoticePath":"/settings/dismiss-notice/publish_action_from_repo","releasePath":"/manishvat/gat-traffic-control/releases/new?marketplace=true","showPublishActionBanner":false},"interactionLimitBanner":null,"showInvitationBanner":false,"inviterName":null,"actionsMigrationBannerInfo":{"releaseTags":[],"showImmutableActionsMigrationBanner":false,"initialMigrationStatus":null}},"codeButton":{"contactPath":"/contact","isEnterprise":false,"local":{"protocolInfo":{"httpAvailable":true,"sshAvailable":null,"httpUrl":"https://github.com/manishvat/gat-traffic-control.git","showCloneWarning":null,"sshUrl":null,"sshCertificatesRequired":null,"sshCertificatesAvailable":null,"ghCliUrl":"gh repo clone manishvat/gat-traffic-control","defaultProtocol":"http","newSshKeyUrl":"/settings/ssh/new","setProtocolPath":"/users/set_protocol"},"platformInfo":{"cloneUrl":"https://desktop.github.com","showVisualStudioCloneButton":false,"visualStudioCloneUrl":"https://windows.github.com","showXcodeCloneButton":false,"xcodeCloneUrl":"xcode://clone?repo=https%3A%2F%2Fgithub.com%2Fmanishvat%2Fgat-traffic-control","zipballUrl":"/manishvat/gat-traffic-control/archive/refs/heads/main.zip"}},"newCodespacePath":"/codespaces/new?hide_repo_select=true\u0026repo=1169143393"},"popovers":{"rename":null,"renamedParentRepo":null},"commitCount":"1","overviewFiles":[],"overviewFilesProcessingTime":0,"copilotSWEAgentEnabled":false}},"codeViewLayoutRoute":{"repo":{"id":1169143393,"defaultBranch":"main","name":"gat-traffic-control","ownerLogin":"manishvat","currentUserCanPush":false,"isFork":false,"isEmpty":false,"createdAt":"2026-02-28T08:35:01.000Z","ownerAvatar":"https://avatars.githubusercontent.com/u/156544948?v=4","public":true,"private":false,"isOrgOwned":false},"currentUser":null,"uploadToken":"eN3u2lLrVNX6T2GlbFWzvx0xNmXv-KRSvy_yQ-6TRpEMLQ641IY-2sYLMMcsisY7ihzmD1l_URcu56Oh3t4x2g","allShortcutsEnabled":false,"treeExpanded":true,"path":"/","symbolsExpanded":false,"refInfo":{"name":"main","listCacheKey":"v0:1772267809.0","canEdit":false,"currentOid":"2fb8bda825961fb36fcb98daa5809ab893924799"},"helpUrl":"https://docs.github.com","findFileWorkerPath":"/assets-cdn/worker/find-file-worker-4e5d7136862a2a48.js","findInFileWorkerPath":"/assets-cdn/worker/find-in-file-worker-4c35b25d88167fef.js","githubDevUrl":null},"csrf_tokens":{"/manishvat/gat-traffic-control/branches":{"post":"Ov9O40FWJWjsX-cOsrgABW2ktX2yB1P6LoWPJLxnBFxM92MVgcn-I9gMSR0rqXnf58hpxDtwbND6fvNcw-SlIA"}}},"title":"GitHub - manishvat/gat-traffic-control","appPayload":{},"meta":{"title":"GitHub - manishvat/gat-traffic-control"}}</script>
  <div data-target="react-app.reactRoot"><meta name="github-code-view-meta-stats" id="github-code-view-meta-stats" data-hydrostats="publish"> <!-- --> <a hidden="" id="code-view-repo-link" href="https://github.com/manishvat/gat-traffic-control" data-discover="true"></a> <button hidden="" data-testid="header-permalink-button" data-hotkey-scope="read-only-cursor-text-area" data-hotkey="y,Shift+Y"></button><button hidden="" data-hotkey="y,Shift+Y"></button><div><div class="prc-PageLayout-PageLayoutRoot--KH-d" style="--spacing: var(--spacing-none);"><div class="prc-PageLayout-PageLayoutWrapper-2BhU2" data-width="full"><div class="prc-PageLayout-PageLayoutContent-BneH9"><div class="CodeViewFileTreeLayout-module__sidebar__n_Aau" tabindex="0"><div class="prc-PageLayout-PaneWrapper-pHPop ReposFileTreePane-module__Pane__rBZpI ReposFileTreePane-module__HidePane__VHAVt" data-is-hidden="false" data-position="start" data-sticky="true" style="--offset-header: 0px; --spacing-row: var(--spacing-none); --spacing-column: var(--spacing-none);"><div class="prc-PageLayout-HorizontalDivider-JLVqp prc-PageLayout-PaneHorizontalDivider-9tbnE" data-variant-regular="none" data-variant-narrow="none" data-position="start" style="--spacing-divider: var(--spacing-none); --spacing: var(--spacing-none);"></div><div class="prc-PageLayout-Pane-AyzHK" data-resizable="true" style="--spacing: var(--spacing-none); --pane-min-width: 256px; --pane-max-width: 567px; --pane-width-size: var(--pane-width-large); --pane-width: 320px;"><div><div id="repos-file-tree" class="ReposFileTreePane-module__PaneContents__SJjfF"><div class="ReposFileTreePane-module__Box_1__PpIop"><div class="d-flex width-full tmp-mb-3 flex-items-center"><h2 class="use-tree-pane-module__Heading__s4QbZ prc-Heading-Heading-MtWFE"><button data-component="IconButton" type="button" data-testid="collapse-file-tree-button" aria-expanded="true" aria-controls="repos-file-tree" data-hotkey="Control+b" class="prc-Button-ButtonBase-9n-Xk position-relative ExpandFileTreeButton-module__expandButton__hDOcv fgColor-muted prc-Button-IconButton-fyge7" data-loading="false" data-no-visuals="true" data-size="medium" data-variant="invisible" aria-labelledby="_r_8d_"><svg aria-hidden="true" focusable="false" class="octicon octicon-sidebar-expand" viewBox="0 0 16 16" width="16" height="16" fill="currentColor" display="inline-block" overflow="visible" style="vertical-align: text-bottom;"><path d="m4.177 7.823 2.396-2.396A.25.25 0 0 1 7 5.604v4.792a.25.25 0 0 1-.427.177L4.177 8.177a.25.25 0 0 1 0-.354Z"></path><path d="M0 1.75C0 .784.784 0 1.75 0h12.5C15.216 0 16 .784 16 1.75v12.5A1.75 1.75 0 0 1 14.25 16H1.75A1.75 1.75 0 0 1 0 14.25Zm1.75-.25a.25.25 0 0 0-.25.25v12.5c0 .138.112.25.25.25H9.5v-13Zm12.5 13a.25.25 0 0 0 .25-.25V1.75a.25.25 0 0 0-.25-.25H11v13Z"></path></svg></button><span class="prc-TooltipV2-Tooltip-tLeuB" data-direction="se" aria-hidden="true" id="_r_8d_" popover="auto">Collapse file tree</span><button hidden="" data-testid="" data-hotkey="Control+b" data-hotkey-scope="read-only-cursor-text-area"></button></h2><h2 class="CodeViewFileTreeLayout-module__heading__A1Iqk">Files</h2></div><div class="ReposFileTreePane-module__Box_2__g74WI"><div class="ReposFileTreePane-module__Box_3__D6cTp"><button type="button" aria-haspopup="true" aria-expanded="false" tabindex="0" data-hotkey="w" aria-label="main branch" data-testid="anchor-button" data-icv-name="Switch branches/tags" class="prc-Button-ButtonBase-9n-Xk react-repos-tree-pane-ref-selector width-full ref-selector-class RefSelectorAnchoredOverlay-module__RefSelectorOverlayBtn__a3WK3" data-loading="false" data-size="medium" data-variant="default" id="ref-picker-repos-header-ref-selector" style="min-width: 0px;"><span data-component="buttonContent" data-align="center" class="prc-Button-ButtonContent-Iohp5"><span data-component="text" class="prc-Button-Label-FWkx3"><div class="RefSelectorAnchoredOverlay-module__RefSelectorOverlayContainer__yaf4p"><div class="RefSelectorAnchoredOverlay-module__RefSelectorOverlayHeader__XtXRG"><svg aria-hidden="true" focusable="false" class="octicon octicon-git-branch" viewBox="0 0 16 16" width="16" height="16" fill="currentColor" display="inline-block" overflow="visible" style="vertical-align: text-bottom;"><path d="M9.5 3.25a2.25 2.25 0 1 1 3 2.122V6A2.5 2.5 0 0 1 10 8.5H6a1 1 0 0 0-1 1v1.128a2.251 2.251 0 1 1-1.5 0V5.372a2.25 2.25 0 1 1 1.5 0v1.836A2.493 2.493 0 0 1 6 7h4a1 1 0 0 0 1-1v-.628A2.25 2.25 0 0 1 9.5 3.25Zm-6 0a.75.75 0 1 0 1.5 0 .75.75 0 0 0-1.5 0Zm8.25-.75a.75.75 0 1 0 0 1.5.75.75 0 0 0 0-1.5ZM4.25 12a.75.75 0 1 0 0 1.5.75.75 0 0 0 0-1.5Z"></path></svg></div><div class="ref-selector-button-text-container RefSelectorAnchoredOverlay-module__RefSelectorBtnTextContainer__Di3rk"><span class="RefSelectorAnchoredOverlay-module__RefSelectorText__w_fmP">&nbsp;main</span></div></div></span><span data-component="trailingVisual" class="prc-Button-Visual-YNt2F prc-Button-VisualWrap-E4cnq"><svg aria-hidden="true" focusable="false" class="octicon octicon-triangle-down" viewBox="0 0 16 16" width="16" height="16" fill="currentColor" display="inline-block" overflow="visible" style="vertical-align: text-bottom;"><path d="m4.427 7.427 3.396 3.396a.25.25 0 0 0 .354 0l3.396-3.396A.25.25 0 0 0 11.396 7H4.604a.25.25 0 0 0-.177.427Z"></path></svg></span></span></button><button hidden="" data-testid="ref-selector-hotkey-button" data-hotkey="w" data-hotkey-scope="read-only-cursor-text-area"></button></div><div class="ReposFileTreePane-module__Box_4__DG4pa"><button data-component="IconButton" type="button" class="prc-Button-ButtonBase-9n-Xk SearchButton-module__IconButton__SBlqu prc-Button-IconButton-fyge7" data-loading="false" data-no-visuals="true" data-size="medium" data-variant="default" aria-labelledby="_r_8j_"><svg aria-hidden="true" focusable="false" class="octicon octicon-search" viewBox="0 0 16 16" width="16" height="16" fill="currentColor" display="inline-block" overflow="visible" style="vertical-align: text-bottom;"><path d="M10.68 11.74a6 6 0 0 1-7.922-8.982 6 6 0 0 1 8.982 7.922l3.04 3.04a.749.749 0 0 1-.326 1.275.749.749 0 0 1-.734-.215ZM11.5 7a4.499 4.499 0 1 0-8.997 0A4.499 4.499 0 0 0 11.5 7Z"></path></svg></button><span class="prc-TooltipV2-Tooltip-tLeuB" data-direction="nw" aria-hidden="true" popover="auto"><span id="_r_8j_">Search this repository<span class="prc-src-InternalVisuallyHidden-2YaI6">(forward slash)</span></span><span class="prc-TooltipV2-KeybindingHintContainer-Ymj-3 prc-TooltipV2-HasTextBefore-fdOXj" aria-hidden="true"><kbd class="prc-KeybindingHint-KeybindingHint-qpYIs prc-Text-Text-9mHv3" data-testid="keybinding-hint"><span class="prc-components-Chord-DdhWN prc-components-ChordOnEmphasis-O-4BS prc-components-ChordSmall-c-P-x prc-Text-Text-9mHv3" data-kbd-chord="true"> <span class="prc-src-InternalVisuallyHidden-2YaI6">forward slash</span><span aria-hidden="true">/</span></span></kbd></span></span><button hidden="" data-testid="" data-hotkey="/" data-hotkey-scope="read-only-cursor-text-area"></button></div></div></div><div class="ReposFileTreePane-module__FileResultsList__zmSnM"><span class="d-flex FileResultsList-module__FilesSearchBox__ivVkc TextInput-wrapper prc-components-TextInputWrapper-Hpdqi prc-components-TextInputBaseWrapper-wY-n0" data-leading-visual="true" data-trailing-visual="true" aria-busy="false"><span class="TextInput-icon" id="_r_8l_" aria-hidden="true"><svg aria-hidden="true" focusable="false" class="octicon octicon-search" viewBox="0 0 16 16" width="16" height="16" fill="currentColor" display="inline-block" overflow="visible" style="vertical-align: text-bottom;"><path d="M10.68 11.74a6 6 0 0 1-7.922-8.982 6 6 0 0 1 8.982 7.922l3.04 3.04a.749.749 0 0 1-.326 1.275.749.749 0 0 1-.734-.215ZM11.5 7a4.499 4.499 0 1 0-8.997 0A4.499 4.499 0 0 0 11.5 7Z"></path></svg></span><input aria-label="Go to file" role="combobox" aria-controls="file-results-list" aria-expanded="false" aria-haspopup="dialog" autocorrect="off" spellcheck="false" placeholder="Go to file" aria-describedby="_r_8l_ _r_8m_" data-component="input" class="prc-components-Input-IwWrt" type="text" value=""><span class="TextInput-icon" id="_r_8m_" aria-hidden="true"></span></span></div><button hidden="" data-testid="" data-hotkey="t,Shift+T" data-hotkey-scope="read-only-cursor-text-area"></button><button hidden="" data-hotkey="t,Shift+T"></button><div class="ReposFileTreePane-module__Box_5__Zy_o6"><div style="height: 1px; margin-top: -1px; width: 100%; flex-shrink: 0;"></div><div><div class="react-tree-show-tree-items"><div class="ReposFileTreeView-module__Box__vkb5W" data-testid="repos-file-tree-container"><nav aria-label="File Tree Navigation"><span class="prc-src-InternalVisuallyHidden-2YaI6"><div></div></span><ul role="tree" aria-label="Files" data-truncate-text="true" class="prc-TreeView-TreeViewRootUlStyles-Mzrmj"><li class="PRIVATE_TreeView-item prc-TreeView-TreeViewItem-Ter5f" tabindex="0" id="4_GAT.py-item" role="treeitem" aria-labelledby="_r_k_" aria-describedby="_r_l_" aria-level="1" aria-selected="false" aria-current="true"><div class="PRIVATE_TreeView-item-container prc-TreeView-TreeViewItemContainer-z6qqQ" style="--level: 1;"><div style="grid-area: spacer; display: flex;"><div style="width: 100%; display: flex;"></div></div><div id="_r_k_" class="PRIVATE_TreeView-item-content prc-TreeView-TreeViewItemContent-RKsCI"><div class="PRIVATE_VisuallyHidden prc-TreeView-TreeViewVisuallyHidden-1N8xK" aria-hidden="true" id="_r_l_"></div><div class="PRIVATE_TreeView-item-visual prc-TreeView-TreeViewItemVisual-naWzj" aria-hidden="true"><svg aria-hidden="true" focusable="false" class="octicon octicon-file" viewBox="0 0 16 16" width="16" height="16" fill="currentColor" display="inline-block" overflow="visible" style="vertical-align: text-bottom;"><path d="M2 1.75C2 .784 2.784 0 3.75 0h6.586c.464 0 .909.184 1.237.513l2.914 2.914c.329.328.513.773.513 1.237v9.586A1.75 1.75 0 0 1 13.25 16h-9.5A1.75 1.75 0 0 1 2 14.25Zm1.75-.25a.25.25 0 0 0-.25.25v12.5c0 .138.112.25.25.25h9.5a.25.25 0 0 0 .25-.25V6h-2.75A1.75 1.75 0 0 1 9 4.25V1.5Zm6.75.062V4.25c0 .138.112.25.25.25h2.688l-.011-.013-2.914-2.914-.013-.011Z"></path></svg></div><span class="PRIVATE_TreeView-item-content-text prc-TreeView-TreeViewItemContentText-FFaKp"><span>4_GAT.py</span></span></div></div></li><li class="PRIVATE_TreeView-item prc-TreeView-TreeViewItem-Ter5f" tabindex="-1" id="5_GCN.py-item" role="treeitem" aria-labelledby="_r_o_" aria-describedby="_r_p_" aria-level="1" aria-selected="false"><div class="PRIVATE_TreeView-item-container prc-TreeView-TreeViewItemContainer-z6qqQ" style="--level: 1; content-visibility: auto; contain-intrinsic-size: auto 2rem;"><div style="grid-area: spacer; display: flex;"><div style="width: 100%; display: flex;"></div></div><div id="_r_o_" class="PRIVATE_TreeView-item-content prc-TreeView-TreeViewItemContent-RKsCI"><div class="PRIVATE_VisuallyHidden prc-TreeView-TreeViewVisuallyHidden-1N8xK" aria-hidden="true" id="_r_p_"></div><div class="PRIVATE_TreeView-item-visual prc-TreeView-TreeViewItemVisual-naWzj" aria-hidden="true"><svg aria-hidden="true" focusable="false" class="octicon octicon-file" viewBox="0 0 16 16" width="16" height="16" fill="currentColor" display="inline-block" overflow="visible" style="vertical-align: text-bottom;"><path d="M2 1.75C2 .784 2.784 0 3.75 0h6.586c.464 0 .909.184 1.237.513l2.914 2.914c.329.328.513.773.513 1.237v9.586A1.75 1.75 0 0 1 13.25 16h-9.5A1.75 1.75 0 0 1 2 14.25Zm1.75-.25a.25.25 0 0 0-.25.25v12.5c0 .138.112.25.25.25h9.5a.25.25 0 0 0 .25-.25V6h-2.75A1.75 1.75 0 0 1 9 4.25V1.5Zm6.75.062V4.25c0 .138.112.25.25.25h2.688l-.011-.013-2.914-2.914-.013-.011Z"></path></svg></div><span class="PRIVATE_TreeView-item-content-text prc-TreeView-TreeViewItemContentText-FFaKp"><span>5_GCN.py</span></span></div></div></li><li class="PRIVATE_TreeView-item prc-TreeView-TreeViewItem-Ter5f" tabindex="-1" id="6_FTS_New_Network.py-item" role="treeitem" aria-labelledby="_r_s_" aria-describedby="_r_t_" aria-level="1" aria-selected="false"><div class="PRIVATE_TreeView-item-container prc-TreeView-TreeViewItemContainer-z6qqQ" style="--level: 1; content-visibility: auto; contain-intrinsic-size: auto 2rem;"><div style="grid-area: spacer; display: flex;"><div style="width: 100%; display: flex;"></div></div><div id="_r_s_" class="PRIVATE_TreeView-item-content prc-TreeView-TreeViewItemContent-RKsCI"><div class="PRIVATE_VisuallyHidden prc-TreeView-TreeViewVisuallyHidden-1N8xK" aria-hidden="true" id="_r_t_"></div><div class="PRIVATE_TreeView-item-visual prc-TreeView-TreeViewItemVisual-naWzj" aria-hidden="true"><svg aria-hidden="true" focusable="false" class="octicon octicon-file" viewBox="0 0 16 16" width="16" height="16" fill="currentColor" display="inline-block" overflow="visible" style="vertical-align: text-bottom;"><path d="M2 1.75C2 .784 2.784 0 3.75 0h6.586c.464 0 .909.184 1.237.513l2.914 2.914c.329.328.513.773.513 1.237v9.586A1.75 1.75 0 0 1 13.25 16h-9.5A1.75 1.75 0 0 1 2 14.25Zm1.75-.25a.25.25 0 0 0-.25.25v12.5c0 .138.112.25.25.25h9.5a.25.25 0 0 0 .25-.25V6h-2.75A1.75 1.75 0 0 1 9 4.25V1.5Zm6.75.062V4.25c0 .138.112.25.25.25h2.688l-.011-.013-2.914-2.914-.013-.011Z"></path></svg></div><span class="PRIVATE_TreeView-item-content-text prc-TreeView-TreeViewItemContentText-FFaKp"><span>6_FTS_New_Network.py</span></span></div></div></li><li class="PRIVATE_TreeView-item prc-TreeView-TreeViewItem-Ter5f" tabindex="-1" id="New_Network.add.xml-item" role="treeitem" aria-labelledby="_r_10_" aria-describedby="_r_11_" aria-level="1" aria-selected="false"><div class="PRIVATE_TreeView-item-container prc-TreeView-TreeViewItemContainer-z6qqQ" style="--level: 1; content-visibility: auto; contain-intrinsic-size: auto 2rem;"><div style="grid-area: spacer; display: flex;"><div style="width: 100%; display: flex;"></div></div><div id="_r_10_" class="PRIVATE_TreeView-item-content prc-TreeView-TreeViewItemContent-RKsCI"><div class="PRIVATE_VisuallyHidden prc-TreeView-TreeViewVisuallyHidden-1N8xK" aria-hidden="true" id="_r_11_"></div><div class="PRIVATE_TreeView-item-visual prc-TreeView-TreeViewItemVisual-naWzj" aria-hidden="true"><svg aria-hidden="true" focusable="false" class="octicon octicon-file" viewBox="0 0 16 16" width="16" height="16" fill="currentColor" display="inline-block" overflow="visible" style="vertical-align: text-bottom;"><path d="M2 1.75C2 .784 2.784 0 3.75 0h6.586c.464 0 .909.184 1.237.513l2.914 2.914c.329.328.513.773.513 1.237v9.586A1.75 1.75 0 0 1 13.25 16h-9.5A1.75 1.75 0 0 1 2 14.25Zm1.75-.25a.25.25 0 0 0-.25.25v12.5c0 .138.112.25.25.25h9.5a.25.25 0 0 0 .25-.25V6h-2.75A1.75 1.75 0 0 1 9 4.25V1.5Zm6.75.062V4.25c0 .138.112.25.25.25h2.688l-.011-.013-2.914-2.914-.013-.011Z"></path></svg></div><span class="PRIVATE_TreeView-item-content-text prc-TreeView-TreeViewItemContentText-FFaKp"><span>New_Network.add.xml</span></span></div></div></li><li class="PRIVATE_TreeView-item prc-TreeView-TreeViewItem-Ter5f" tabindex="-1" id="New_Network.net copy.xml-item" role="treeitem" aria-labelledby="_r_14_" aria-describedby="_r_15_" aria-level="1" aria-selected="false"><div class="PRIVATE_TreeView-item-container prc-TreeView-TreeViewItemContainer-z6qqQ" style="--level: 1; content-visibility: auto; contain-intrinsic-size: auto 2rem;"><div style="grid-area: spacer; display: flex;"><div style="width: 100%; display: flex;"></div></div><div id="_r_14_" class="PRIVATE_TreeView-item-content prc-TreeView-TreeViewItemContent-RKsCI"><div class="PRIVATE_VisuallyHidden prc-TreeView-TreeViewVisuallyHidden-1N8xK" aria-hidden="true" id="_r_15_"></div><div class="PRIVATE_TreeView-item-visual prc-TreeView-TreeViewItemVisual-naWzj" aria-hidden="true"><svg aria-hidden="true" focusable="false" class="octicon octicon-file" viewBox="0 0 16 16" width="16" height="16" fill="currentColor" display="inline-block" overflow="visible" style="vertical-align: text-bottom;"><path d="M2 1.75C2 .784 2.784 0 3.75 0h6.586c.464 0 .909.184 1.237.513l2.914 2.914c.329.328.513.773.513 1.237v9.586A1.75 1.75 0 0 1 13.25 16h-9.5A1.75 1.75 0 0 1 2 14.25Zm1.75-.25a.25.25 0 0 0-.25.25v12.5c0 .138.112.25.25.25h9.5a.25.25 0 0 0 .25-.25V6h-2.75A1.75 1.75 0 0 1 9 4.25V1.5Zm6.75.062V4.25c0 .138.112.25.25.25h2.688l-.011-.013-2.914-2.914-.013-.011Z"></path></svg></div><span class="PRIVATE_TreeView-item-content-text prc-TreeView-TreeViewItemContentText-FFaKp"><span>New_Network.net copy.xml</span></span></div></div></li><li class="PRIVATE_TreeView-item prc-TreeView-TreeViewItem-Ter5f" tabindex="-1" id="New_Network.net.xml-item" role="treeitem" aria-labelledby="_r_18_" aria-describedby="_r_19_" aria-level="1" aria-selected="false"><div class="PRIVATE_TreeView-item-container prc-TreeView-TreeViewItemContainer-z6qqQ" style="--level: 1; content-visibility: auto; contain-intrinsic-size: auto 2rem;"><div style="grid-area: spacer; display: flex;"><div style="width: 100%; display: flex;"></div></div><div id="_r_18_" class="PRIVATE_TreeView-item-content prc-TreeView-TreeViewItemContent-RKsCI"><div class="PRIVATE_VisuallyHidden prc-TreeView-TreeViewVisuallyHidden-1N8xK" aria-hidden="true" id="_r_19_"></div><div class="PRIVATE_TreeView-item-visual prc-TreeView-TreeViewItemVisual-naWzj" aria-hidden="true"><svg aria-hidden="true" focusable="false" class="octicon octicon-file" viewBox="0 0 16 16" width="16" height="16" fill="currentColor" display="inline-block" overflow="visible" style="vertical-align: text-bottom;"><path d="M2 1.75C2 .784 2.784 0 3.75 0h6.586c.464 0 .909.184 1.237.513l2.914 2.914c.329.328.513.773.513 1.237v9.586A1.75 1.75 0 0 1 13.25 16h-9.5A1.75 1.75 0 0 1 2 14.25Zm1.75-.25a.25.25 0 0 0-.25.25v12.5c0 .138.112.25.25.25h9.5a.25.25 0 0 0 .25-.25V6h-2.75A1.75 1.75 0 0 1 9 4.25V1.5Zm6.75.062V4.25c0 .138.112.25.25.25h2.688l-.011-.013-2.914-2.914-.013-.011Z"></path></svg></div><span class="PRIVATE_TreeView-item-content-text prc-TreeView-TreeViewItemContentText-FFaKp"><span>New_Network.net.xml</span></span></div></div></li><li class="PRIVATE_TreeView-item prc-TreeView-TreeViewItem-Ter5f" tabindex="-1" id="New_Network.netecfg-item" role="treeitem" aria-labelledby="_r_1c_" aria-describedby="_r_1d_" aria-level="1" aria-selected="false"><div class="PRIVATE_TreeView-item-container prc-TreeView-TreeViewItemContainer-z6qqQ" style="--level: 1; content-visibility: auto; contain-intrinsic-size: auto 2rem;"><div style="grid-area: spacer; display: flex;"><div style="width: 100%; display: flex;"></div></div><div id="_r_1c_" class="PRIVATE_TreeView-item-content prc-TreeView-TreeViewItemContent-RKsCI"><div class="PRIVATE_VisuallyHidden prc-TreeView-TreeViewVisuallyHidden-1N8xK" aria-hidden="true" id="_r_1d_"></div><div class="PRIVATE_TreeView-item-visual prc-TreeView-TreeViewItemVisual-naWzj" aria-hidden="true"><svg aria-hidden="true" focusable="false" class="octicon octicon-file" viewBox="0 0 16 16" width="16" height="16" fill="currentColor" display="inline-block" overflow="visible" style="vertical-align: text-bottom;"><path d="M2 1.75C2 .784 2.784 0 3.75 0h6.586c.464 0 .909.184 1.237.513l2.914 2.914c.329.328.513.773.513 1.237v9.586A1.75 1.75 0 0 1 13.25 16h-9.5A1.75 1.75 0 0 1 2 14.25Zm1.75-.25a.25.25 0 0 0-.25.25v12.5c0 .138.112.25.25.25h9.5a.25.25 0 0 0 .25-.25V6h-2.75A1.75 1.75 0 0 1 9 4.25V1.5Zm6.75.062V4.25c0 .138.112.25.25.25h2.688l-.011-.013-2.914-2.914-.013-.011Z"></path></svg></div><span class="PRIVATE_TreeView-item-content-text prc-TreeView-TreeViewItemContentText-FFaKp"><span>New_Network.netecfg</span></span></div></div></li><li class="PRIVATE_TreeView-item prc-TreeView-TreeViewItem-Ter5f" tabindex="-1" id="New_Network.rou.xml-item" role="treeitem" aria-labelledby="_r_1g_" aria-describedby="_r_1h_" aria-level="1" aria-selected="false"><div class="PRIVATE_TreeView-item-container prc-TreeView-TreeViewItemContainer-z6qqQ" style="--level: 1; content-visibility: auto; contain-intrinsic-size: auto 2rem;"><div style="grid-area: spacer; display: flex;"><div style="width: 100%; display: flex;"></div></div><div id="_r_1g_" class="PRIVATE_TreeView-item-content prc-TreeView-TreeViewItemContent-RKsCI"><div class="PRIVATE_VisuallyHidden prc-TreeView-TreeViewVisuallyHidden-1N8xK" aria-hidden="true" id="_r_1h_"></div><div class="PRIVATE_TreeView-item-visual prc-TreeView-TreeViewItemVisual-naWzj" aria-hidden="true"><svg aria-hidden="true" focusable="false" class="octicon octicon-file" viewBox="0 0 16 16" width="16" height="16" fill="currentColor" display="inline-block" overflow="visible" style="vertical-align: text-bottom;"><path d="M2 1.75C2 .784 2.784 0 3.75 0h6.586c.464 0 .909.184 1.237.513l2.914 2.914c.329.328.513.773.513 1.237v9.586A1.75 1.75 0 0 1 13.25 16h-9.5A1.75 1.75 0 0 1 2 14.25Zm1.75-.25a.25.25 0 0 0-.25.25v12.5c0 .138.112.25.25.25h9.5a.25.25 0 0 0 .25-.25V6h-2.75A1.75 1.75 0 0 1 9 4.25V1.5Zm6.75.062V4.25c0 .138.112.25.25.25h2.688l-.011-.013-2.914-2.914-.013-.011Z"></path></svg></div><span class="PRIVATE_TreeView-item-content-text prc-TreeView-TreeViewItemContentText-FFaKp"><span>New_Network.rou.xml</span></span></div></div></li><li class="PRIVATE_TreeView-item prc-TreeView-TreeViewItem-Ter5f" tabindex="-1" id="New_Network.sumocfg-item" role="treeitem" aria-labelledby="_r_1k_" aria-describedby="_r_1l_" aria-level="1" aria-selected="false"><div class="PRIVATE_TreeView-item-container prc-TreeView-TreeViewItemContainer-z6qqQ" style="--level: 1; content-visibility: auto; contain-intrinsic-size: auto 2rem;"><div style="grid-area: spacer; display: flex;"><div style="width: 100%; display: flex;"></div></div><div id="_r_1k_" class="PRIVATE_TreeView-item-content prc-TreeView-TreeViewItemContent-RKsCI"><div class="PRIVATE_VisuallyHidden prc-TreeView-TreeViewVisuallyHidden-1N8xK" aria-hidden="true" id="_r_1l_"></div><div class="PRIVATE_TreeView-item-visual prc-TreeView-TreeViewItemVisual-naWzj" aria-hidden="true"><svg aria-hidden="true" focusable="false" class="octicon octicon-file" viewBox="0 0 16 16" width="16" height="16" fill="currentColor" display="inline-block" overflow="visible" style="vertical-align: text-bottom;"><path d="M2 1.75C2 .784 2.784 0 3.75 0h6.586c.464 0 .909.184 1.237.513l2.914 2.914c.329.328.513.773.513 1.237v9.586A1.75 1.75 0 0 1 13.25 16h-9.5A1.75 1.75 0 0 1 2 14.25Zm1.75-.25a.25.25 0 0 0-.25.25v12.5c0 .138.112.25.25.25h9.5a.25.25 0 0 0 .25-.25V6h-2.75A1.75 1.75 0 0 1 9 4.25V1.5Zm6.75.062V4.25c0 .138.112.25.25.25h2.688l-.011-.013-2.914-2.914-.013-.011Z"></path></svg></div><span class="PRIVATE_TreeView-item-content-text prc-TreeView-TreeViewItemContentText-FFaKp"><span>New_Network.sumocfg</span></span></div></div></li><li class="PRIVATE_TreeView-item prc-TreeView-TreeViewItem-Ter5f" tabindex="-1" id="e2_0.xml-item" role="treeitem" aria-labelledby="_r_1o_" aria-describedby="_r_1p_" aria-level="1" aria-selected="false"><div class="PRIVATE_TreeView-item-container prc-TreeView-TreeViewItemContainer-z6qqQ" style="--level: 1; content-visibility: auto; contain-intrinsic-size: auto 2rem;"><div style="grid-area: spacer; display: flex;"><div style="width: 100%; display: flex;"></div></div><div id="_r_1o_" class="PRIVATE_TreeView-item-content prc-TreeView-TreeViewItemContent-RKsCI"><div class="PRIVATE_VisuallyHidden prc-TreeView-TreeViewVisuallyHidden-1N8xK" aria-hidden="true" id="_r_1p_"></div><div class="PRIVATE_TreeView-item-visual prc-TreeView-TreeViewItemVisual-naWzj" aria-hidden="true"><svg aria-hidden="true" focusable="false" class="octicon octicon-file" viewBox="0 0 16 16" width="16" height="16" fill="currentColor" display="inline-block" overflow="visible" style="vertical-align: text-bottom;"><path d="M2 1.75C2 .784 2.784 0 3.75 0h6.586c.464 0 .909.184 1.237.513l2.914 2.914c.329.328.513.773.513 1.237v9.586A1.75 1.75 0 0 1 13.25 16h-9.5A1.75 1.75 0 0 1 2 14.25Zm1.75-.25a.25.25 0 0 0-.25.25v12.5c0 .138.112.25.25.25h9.5a.25.25 0 0 0 .25-.25V6h-2.75A1.75 1.75 0 0 1 9 4.25V1.5Zm6.75.062V4.25c0 .138.112.25.25.25h2.688l-.011-.013-2.914-2.914-.013-.011Z"></path></svg></div><span class="PRIVATE_TreeView-item-content-text prc-TreeView-TreeViewItemContentText-FFaKp"><span>e2_0.xml</span></span></div></div></li><li class="PRIVATE_TreeView-item prc-TreeView-TreeViewItem-Ter5f" tabindex="-1" id="e2_1.xml-item" role="treeitem" aria-labelledby="_r_1s_" aria-describedby="_r_1t_" aria-level="1" aria-selected="false"><div class="PRIVATE_TreeView-item-container prc-TreeView-TreeViewItemContainer-z6qqQ" style="--level: 1; content-visibility: auto; contain-intrinsic-size: auto 2rem;"><div style="grid-area: spacer; display: flex;"><div style="width: 100%; display: flex;"></div></div><div id="_r_1s_" class="PRIVATE_TreeView-item-content prc-TreeView-TreeViewItemContent-RKsCI"><div class="PRIVATE_VisuallyHidden prc-TreeView-TreeViewVisuallyHidden-1N8xK" aria-hidden="true" id="_r_1t_"></div><div class="PRIVATE_TreeView-item-visual prc-TreeView-TreeViewItemVisual-naWzj" aria-hidden="true"><svg aria-hidden="true" focusable="false" class="octicon octicon-file" viewBox="0 0 16 16" width="16" height="16" fill="currentColor" display="inline-block" overflow="visible" style="vertical-align: text-bottom;"><path d="M2 1.75C2 .784 2.784 0 3.75 0h6.586c.464 0 .909.184 1.237.513l2.914 2.914c.329.328.513.773.513 1.237v9.586A1.75 1.75 0 0 1 13.25 16h-9.5A1.75 1.75 0 0 1 2 14.25Zm1.75-.25a.25.25 0 0 0-.25.25v12.5c0 .138.112.25.25.25h9.5a.25.25 0 0 0 .25-.25V6h-2.75A1.75 1.75 0 0 1 9 4.25V1.5Zm6.75.062V4.25c0 .138.112.25.25.25h2.688l-.011-.013-2.914-2.914-.013-.011Z"></path></svg></div><span class="PRIVATE_TreeView-item-content-text prc-TreeView-TreeViewItemContentText-FFaKp"><span>e2_1.xml</span></span></div></div></li><li class="PRIVATE_TreeView-item prc-TreeView-TreeViewItem-Ter5f" tabindex="-1" id="e2_10.xml-item" role="treeitem" aria-labelledby="_r_20_" aria-describedby="_r_21_" aria-level="1" aria-selected="false"><div class="PRIVATE_TreeView-item-container prc-TreeView-TreeViewItemContainer-z6qqQ" style="--level: 1; content-visibility: auto; contain-intrinsic-size: auto 2rem;"><div style="grid-area: spacer; display: flex;"><div style="width: 100%; display: flex;"></div></div><div id="_r_20_" class="PRIVATE_TreeView-item-content prc-TreeView-TreeViewItemContent-RKsCI"><div class="PRIVATE_VisuallyHidden prc-TreeView-TreeViewVisuallyHidden-1N8xK" aria-hidden="true" id="_r_21_"></div><div class="PRIVATE_TreeView-item-visual prc-TreeView-TreeViewItemVisual-naWzj" aria-hidden="true"><svg aria-hidden="true" focusable="false" class="octicon octicon-file" viewBox="0 0 16 16" width="16" height="16" fill="currentColor" display="inline-block" overflow="visible" style="vertical-align: text-bottom;"><path d="M2 1.75C2 .784 2.784 0 3.75 0h6.586c.464 0 .909.184 1.237.513l2.914 2.914c.329.328.513.773.513 1.237v9.586A1.75 1.75 0 0 1 13.25 16h-9.5A1.75 1.75 0 0 1 2 14.25Zm1.75-.25a.25.25 0 0 0-.25.25v12.5c0 .138.112.25.25.25h9.5a.25.25 0 0 0 .25-.25V6h-2.75A1.75 1.75 0 0 1 9 4.25V1.5Zm6.75.062V4.25c0 .138.112.25.25.25h2.688l-.011-.013-2.914-2.914-.013-.011Z"></path></svg></div><span class="PRIVATE_TreeView-item-content-text prc-TreeView-TreeViewItemContentText-FFaKp"><span>e2_10.xml</span></span></div></div></li><li class="PRIVATE_TreeView-item prc-TreeView-TreeViewItem-Ter5f" tabindex="-1" id="e2_11.xml-item" role="treeitem" aria-labelledby="_r_24_" aria-describedby="_r_25_" aria-level="1" aria-selected="false"><div class="PRIVATE_TreeView-item-container prc-TreeView-TreeViewItemContainer-z6qqQ" style="--level: 1; content-visibility: auto; contain-intrinsic-size: auto 2rem;"><div style="grid-area: spacer; display: flex;"><div style="width: 100%; display: flex;"></div></div><div id="_r_24_" class="PRIVATE_TreeView-item-content prc-TreeView-TreeViewItemContent-RKsCI"><div class="PRIVATE_VisuallyHidden prc-TreeView-TreeViewVisuallyHidden-1N8xK" aria-hidden="true" id="_r_25_"></div><div class="PRIVATE_TreeView-item-visual prc-TreeView-TreeViewItemVisual-naWzj" aria-hidden="true"><svg aria-hidden="true" focusable="false" class="octicon octicon-file" viewBox="0 0 16 16" width="16" height="16" fill="currentColor" display="inline-block" overflow="visible" style="vertical-align: text-bottom;"><path d="M2 1.75C2 .784 2.784 0 3.75 0h6.586c.464 0 .909.184 1.237.513l2.914 2.914c.329.328.513.773.513 1.237v9.586A1.75 1.75 0 0 1 13.25 16h-9.5A1.75 1.75 0 0 1 2 14.25Zm1.75-.25a.25.25 0 0 0-.25.25v12.5c0 .138.112.25.25.25h9.5a.25.25 0 0 0 .25-.25V6h-2.75A1.75 1.75 0 0 1 9 4.25V1.5Zm6.75.062V4.25c0 .138.112.25.25.25h2.688l-.011-.013-2.914-2.914-.013-.011Z"></path></svg></div><span class="PRIVATE_TreeView-item-content-text prc-TreeView-TreeViewItemContentText-FFaKp"><span>e2_11.xml</span></span></div></div></li><li class="PRIVATE_TreeView-item prc-TreeView-TreeViewItem-Ter5f" tabindex="-1" id="e2_12.xml-item" role="treeitem" aria-labelledby="_r_28_" aria-describedby="_r_29_" aria-level="1" aria-selected="false"><div class="PRIVATE_TreeView-item-container prc-TreeView-TreeViewItemContainer-z6qqQ" style="--level: 1; content-visibility: auto; contain-intrinsic-size: auto 2rem;"><div style="grid-area: spacer; display: flex;"><div style="width: 100%; display: flex;"></div></div><div id="_r_28_" class="PRIVATE_TreeView-item-content prc-TreeView-TreeViewItemContent-RKsCI"><div class="PRIVATE_VisuallyHidden prc-TreeView-TreeViewVisuallyHidden-1N8xK" aria-hidden="true" id="_r_29_"></div><div class="PRIVATE_TreeView-item-visual prc-TreeView-TreeViewItemVisual-naWzj" aria-hidden="true"><svg aria-hidden="true" focusable="false" class="octicon octicon-file" viewBox="0 0 16 16" width="16" height="16" fill="currentColor" display="inline-block" overflow="visible" style="vertical-align: text-bottom;"><path d="M2 1.75C2 .784 2.784 0 3.75 0h6.586c.464 0 .909.184 1.237.513l2.914 2.914c.329.328.513.773.513 1.237v9.586A1.75 1.75 0 0 1 13.25 16h-9.5A1.75 1.75 0 0 1 2 14.25Zm1.75-.25a.25.25 0 0 0-.25.25v12.5c0 .138.112.25.25.25h9.5a.25.25 0 0 0 .25-.25V6h-2.75A1.75 1.75 0 0 1 9 4.25V1.5Zm6.75.062V4.25c0 .138.112.25.25.25h2.688l-.011-.013-2.914-2.914-.013-.011Z"></path></svg></div><span class="PRIVATE_TreeView-item-content-text prc-TreeView-TreeViewItemContentText-FFaKp"><span>e2_12.xml</span></span></div></div></li><li class="PRIVATE_TreeView-item prc-TreeView-TreeViewItem-Ter5f" tabindex="-1" id="e2_13.xml-item" role="treeitem" aria-labelledby="_r_2c_" aria-describedby="_r_2d_" aria-level="1" aria-selected="false"><div class="PRIVATE_TreeView-item-container prc-TreeView-TreeViewItemContainer-z6qqQ" style="--level: 1; content-visibility: auto; contain-intrinsic-size: auto 2rem;"><div style="grid-area: spacer; display: flex;"><div style="width: 100%; display: flex;"></div></div><div id="_r_2c_" class="PRIVATE_TreeView-item-content prc-TreeView-TreeViewItemContent-RKsCI"><div class="PRIVATE_VisuallyHidden prc-TreeView-TreeViewVisuallyHidden-1N8xK" aria-hidden="true" id="_r_2d_"></div><div class="PRIVATE_TreeView-item-visual prc-TreeView-TreeViewItemVisual-naWzj" aria-hidden="true"><svg aria-hidden="true" focusable="false" class="octicon octicon-file" viewBox="0 0 16 16" width="16" height="16" fill="currentColor" display="inline-block" overflow="visible" style="vertical-align: text-bottom;"><path d="M2 1.75C2 .784 2.784 0 3.75 0h6.586c.464 0 .909.184 1.237.513l2.914 2.914c.329.328.513.773.513 1.237v9.586A1.75 1.75 0 0 1 13.25 16h-9.5A1.75 1.75 0 0 1 2 14.25Zm1.75-.25a.25.25 0 0 0-.25.25v12.5c0 .138.112.25.25.25h9.5a.25.25 0 0 0 .25-.25V6h-2.75A1.75 1.75 0 0 1 9 4.25V1.5Zm6.75.062V4.25c0 .138.112.25.25.25h2.688l-.011-.013-2.914-2.914-.013-.011Z"></path></svg></div><span class="PRIVATE_TreeView-item-content-text prc-TreeView-TreeViewItemContentText-FFaKp"><span>e2_13.xml</span></span></div></div></li><li class="PRIVATE_TreeView-item prc-TreeView-TreeViewItem-Ter5f" tabindex="-1" id="e2_14.xml-item" role="treeitem" aria-labelledby="_r_2g_" aria-describedby="_r_2h_" aria-level="1" aria-selected="false"><div class="PRIVATE_TreeView-item-container prc-TreeView-TreeViewItemContainer-z6qqQ" style="--level: 1; content-visibility: auto; contain-intrinsic-size: auto 2rem;"><div style="grid-area: spacer; display: flex;"><div style="width: 100%; display: flex;"></div></div><div id="_r_2g_" class="PRIVATE_TreeView-item-content prc-TreeView-TreeViewItemContent-RKsCI"><div class="PRIVATE_VisuallyHidden prc-TreeView-TreeViewVisuallyHidden-1N8xK" aria-hidden="true" id="_r_2h_"></div><div class="PRIVATE_TreeView-item-visual prc-TreeView-TreeViewItemVisual-naWzj" aria-hidden="true"><svg aria-hidden="true" focusable="false" class="octicon octicon-file" viewBox="0 0 16 16" width="16" height="16" fill="currentColor" display="inline-block" overflow="visible" style="vertical-align: text-bottom;"><path d="M2 1.75C2 .784 2.784 0 3.75 0h6.586c.464 0 .909.184 1.237.513l2.914 2.914c.329.328.513.773.513 1.237v9.586A1.75 1.75 0 0 1 13.25 16h-9.5A1.75 1.75 0 0 1 2 14.25Zm1.75-.25a.25.25 0 0 0-.25.25v12.5c0 .138.112.25.25.25h9.5a.25.25 0 0 0 .25-.25V6h-2.75A1.75 1.75 0 0 1 9 4.25V1.5Zm6.75.062V4.25c0 .138.112.25.25.25h2.688l-.011-.013-2.914-2.914-.013-.011Z"></path></svg></div><span class="PRIVATE_TreeView-item-content-text prc-TreeView-TreeViewItemContentText-FFaKp"><span>e2_14.xml</span></span></div></div></li><li class="PRIVATE_TreeView-item prc-TreeView-TreeViewItem-Ter5f" tabindex="-1" id="e2_15.xml-item" role="treeitem" aria-labelledby="_r_2k_" aria-describedby="_r_2l_" aria-level="1" aria-selected="false"><div class="PRIVATE_TreeView-item-container prc-TreeView-TreeViewItemContainer-z6qqQ" style="--level: 1; content-visibility: auto; contain-intrinsic-size: auto 2rem;"><div style="grid-area: spacer; display: flex;"><div style="width: 100%; display: flex;"></div></div><div id="_r_2k_" class="PRIVATE_TreeView-item-content prc-TreeView-TreeViewItemContent-RKsCI"><div class="PRIVATE_VisuallyHidden prc-TreeView-TreeViewVisuallyHidden-1N8xK" aria-hidden="true" id="_r_2l_"></div><div class="PRIVATE_TreeView-item-visual prc-TreeView-TreeViewItemVisual-naWzj" aria-hidden="true"><svg aria-hidden="true" focusable="false" class="octicon octicon-file" viewBox="0 0 16 16" width="16" height="16" fill="currentColor" display="inline-block" overflow="visible" style="vertical-align: text-bottom;"><path d="M2 1.75C2 .784 2.784 0 3.75 0h6.586c.464 0 .909.184 1.237.513l2.914 2.914c.329.328.513.773.513 1.237v9.586A1.75 1.75 0 0 1 13.25 16h-9.5A1.75 1.75 0 0 1 2 14.25Zm1.75-.25a.25.25 0 0 0-.25.25v12.5c0 .138.112.25.25.25h9.5a.25.25 0 0 0 .25-.25V6h-2.75A1.75 1.75 0 0 1 9 4.25V1.5Zm6.75.062V4.25c0 .138.112.25.25.25h2.688l-.011-.013-2.914-2.914-.013-.011Z"></path></svg></div><span class="PRIVATE_TreeView-item-content-text prc-TreeView-TreeViewItemContentText-FFaKp"><span>e2_15.xml</span></span></div></div></li><li class="PRIVATE_TreeView-item prc-TreeView-TreeViewItem-Ter5f" tabindex="-1" id="e2_16.xml-item" role="treeitem" aria-labelledby="_r_2o_" aria-describedby="_r_2p_" aria-level="1" aria-selected="false"><div class="PRIVATE_TreeView-item-container prc-TreeView-TreeViewItemContainer-z6qqQ" style="--level: 1; content-visibility: auto; contain-intrinsic-size: auto 2rem;"><div style="grid-area: spacer; display: flex;"><div style="width: 100%; display: flex;"></div></div><div id="_r_2o_" class="PRIVATE_TreeView-item-content prc-TreeView-TreeViewItemContent-RKsCI"><div class="PRIVATE_VisuallyHidden prc-TreeView-TreeViewVisuallyHidden-1N8xK" aria-hidden="true" id="_r_2p_"></div><div class="PRIVATE_TreeView-item-visual prc-TreeView-TreeViewItemVisual-naWzj" aria-hidden="true"><svg aria-hidden="true" focusable="false" class="octicon octicon-file" viewBox="0 0 16 16" width="16" height="16" fill="currentColor" display="inline-block" overflow="visible" style="vertical-align: text-bottom;"><path d="M2 1.75C2 .784 2.784 0 3.75 0h6.586c.464 0 .909.184 1.237.513l2.914 2.914c.329.328.513.773.513 1.237v9.586A1.75 1.75 0 0 1 13.25 16h-9.5A1.75 1.75 0 0 1 2 14.25Zm1.75-.25a.25.25 0 0 0-.25.25v12.5c0 .138.112.25.25.25h9.5a.25.25 0 0 0 .25-.25V6h-2.75A1.75 1.75 0 0 1 9 4.25V1.5Zm6.75.062V4.25c0 .138.112.25.25.25h2.688l-.011-.013-2.914-2.914-.013-.011Z"></path></svg></div><span class="PRIVATE_TreeView-item-content-text prc-TreeView-TreeViewItemContentText-FFaKp"><span>e2_16.xml</span></span></div></div></li><li class="PRIVATE_TreeView-item prc-TreeView-TreeViewItem-Ter5f" tabindex="-1" id="e2_17.xml-item" role="treeitem" aria-labelledby="_r_2s_" aria-describedby="_r_2t_" aria-level="1" aria-selected="false"><div class="PRIVATE_TreeView-item-container prc-TreeView-TreeViewItemContainer-z6qqQ" style="--level: 1; content-visibility: auto; contain-intrinsic-size: auto 2rem;"><div style="grid-area: spacer; display: flex;"><div style="width: 100%; display: flex;"></div></div><div id="_r_2s_" class="PRIVATE_TreeView-item-content prc-TreeView-TreeViewItemContent-RKsCI"><div class="PRIVATE_VisuallyHidden prc-TreeView-TreeViewVisuallyHidden-1N8xK" aria-hidden="true" id="_r_2t_"></div><div class="PRIVATE_TreeView-item-visual prc-TreeView-TreeViewItemVisual-naWzj" aria-hidden="true"><svg aria-hidden="true" focusable="false" class="octicon octicon-file" viewBox="0 0 16 16" width="16" height="16" fill="currentColor" display="inline-block" overflow="visible" style="vertical-align: text-bottom;"><path d="M2 1.75C2 .784 2.784 0 3.75 0h6.586c.464 0 .909.184 1.237.513l2.914 2.914c.329.328.513.773.513 1.237v9.586A1.75 1.75 0 0 1 13.25 16h-9.5A1.75 1.75 0 0 1 2 14.25Zm1.75-.25a.25.25 0 0 0-.25.25v12.5c0 .138.112.25.25.25h9.5a.25.25 0 0 0 .25-.25V6h-2.75A1.75 1.75 0 0 1 9 4.25V1.5Zm6.75.062V4.25c0 .138.112.25.25.25h2.688l-.011-.013-2.914-2.914-.013-.011Z"></path></svg></div><span class="PRIVATE_TreeView-item-content-text prc-TreeView-TreeViewItemContentText-FFaKp"><span>e2_17.xml</span></span></div></div></li><li class="PRIVATE_TreeView-item prc-TreeView-TreeViewItem-Ter5f" tabindex="-1" id="e2_18.xml-item" role="treeitem" aria-labelledby="_r_30_" aria-describedby="_r_31_" aria-level="1" aria-selected="false"><div class="PRIVATE_TreeView-item-container prc-TreeView-TreeViewItemContainer-z6qqQ" style="--level: 1; content-visibility: auto; contain-intrinsic-size: auto 2rem;"><div style="grid-area: spacer; display: flex;"><div style="width: 100%; display: flex;"></div></div><div id="_r_30_" class="PRIVATE_TreeView-item-content prc-TreeView-TreeViewItemContent-RKsCI"><div class="PRIVATE_VisuallyHidden prc-TreeView-TreeViewVisuallyHidden-1N8xK" aria-hidden="true" id="_r_31_"></div><div class="PRIVATE_TreeView-item-visual prc-TreeView-TreeViewItemVisual-naWzj" aria-hidden="true"><svg aria-hidden="true" focusable="false" class="octicon octicon-file" viewBox="0 0 16 16" width="16" height="16" fill="currentColor" display="inline-block" overflow="visible" style="vertical-align: text-bottom;"><path d="M2 1.75C2 .784 2.784 0 3.75 0h6.586c.464 0 .909.184 1.237.513l2.914 2.914c.329.328.513.773.513 1.237v9.586A1.75 1.75 0 0 1 13.25 16h-9.5A1.75 1.75 0 0 1 2 14.25Zm1.75-.25a.25.25 0 0 0-.25.25v12.5c0 .138.112.25.25.25h9.5a.25.25 0 0 0 .25-.25V6h-2.75A1.75 1.75 0 0 1 9 4.25V1.5Zm6.75.062V4.25c0 .138.112.25.25.25h2.688l-.011-.013-2.914-2.914-.013-.011Z"></path></svg></div><span class="PRIVATE_TreeView-item-content-text prc-TreeView-TreeViewItemContentText-FFaKp"><span>e2_18.xml</span></span></div></div></li><li class="PRIVATE_TreeView-item prc-TreeView-TreeViewItem-Ter5f" tabindex="-1" id="e2_19.xml-item" role="treeitem" aria-labelledby="_r_34_" aria-describedby="_r_35_" aria-level="1" aria-selected="false"><div class="PRIVATE_TreeView-item-container prc-TreeView-TreeViewItemContainer-z6qqQ" style="--level: 1; content-visibility: auto; contain-intrinsic-size: auto 2rem;"><div style="grid-area: spacer; display: flex;"><div style="width: 100%; display: flex;"></div></div><div id="_r_34_" class="PRIVATE_TreeView-item-content prc-TreeView-TreeViewItemContent-RKsCI"><div class="PRIVATE_VisuallyHidden prc-TreeView-TreeViewVisuallyHidden-1N8xK" aria-hidden="true" id="_r_35_"></div><div class="PRIVATE_TreeView-item-visual prc-TreeView-TreeViewItemVisual-naWzj" aria-hidden="true"><svg aria-hidden="true" focusable="false" class="octicon octicon-file" viewBox="0 0 16 16" width="16" height="16" fill="currentColor" display="inline-block" overflow="visible" style="vertical-align: text-bottom;"><path d="M2 1.75C2 .784 2.784 0 3.75 0h6.586c.464 0 .909.184 1.237.513l2.914 2.914c.329.328.513.773.513 1.237v9.586A1.75 1.75 0 0 1 13.25 16h-9.5A1.75 1.75 0 0 1 2 14.25Zm1.75-.25a.25.25 0 0 0-.25.25v12.5c0 .138.112.25.25.25h9.5a.25.25 0 0 0 .25-.25V6h-2.75A1.75 1.75 0 0 1 9 4.25V1.5Zm6.75.062V4.25c0 .138.112.25.25.25h2.688l-.011-.013-2.914-2.914-.013-.011Z"></path></svg></div><span class="PRIVATE_TreeView-item-content-text prc-TreeView-TreeViewItemContentText-FFaKp"><span>e2_19.xml</span></span></div></div></li><li class="PRIVATE_TreeView-item prc-TreeView-TreeViewItem-Ter5f" tabindex="-1" id="e2_2.xml-item" role="treeitem" aria-labelledby="_r_38_" aria-describedby="_r_39_" aria-level="1" aria-selected="false"><div class="PRIVATE_TreeView-item-container prc-TreeView-TreeViewItemContainer-z6qqQ" style="--level: 1; content-visibility: auto; contain-intrinsic-size: auto 2rem;"><div style="grid-area: spacer; display: flex;"><div style="width: 100%; display: flex;"></div></div><div id="_r_38_" class="PRIVATE_TreeView-item-content prc-TreeView-TreeViewItemContent-RKsCI"><div class="PRIVATE_VisuallyHidden prc-TreeView-TreeViewVisuallyHidden-1N8xK" aria-hidden="true" id="_r_39_"></div><div class="PRIVATE_TreeView-item-visual prc-TreeView-TreeViewItemVisual-naWzj" aria-hidden="true"><svg aria-hidden="true" focusable="false" class="octicon octicon-file" viewBox="0 0 16 16" width="16" height="16" fill="currentColor" display="inline-block" overflow="visible" style="vertical-align: text-bottom;"><path d="M2 1.75C2 .784 2.784 0 3.75 0h6.586c.464 0 .909.184 1.237.513l2.914 2.914c.329.328.513.773.513 1.237v9.586A1.75 1.75 0 0 1 13.25 16h-9.5A1.75 1.75 0 0 1 2 14.25Zm1.75-.25a.25.25 0 0 0-.25.25v12.5c0 .138.112.25.25.25h9.5a.25.25 0 0 0 .25-.25V6h-2.75A1.75 1.75 0 0 1 9 4.25V1.5Zm6.75.062V4.25c0 .138.112.25.25.25h2.688l-.011-.013-2.914-2.914-.013-.011Z"></path></svg></div><span class="PRIVATE_TreeView-item-content-text prc-TreeView-TreeViewItemContentText-FFaKp"><span>e2_2.xml</span></span></div></div></li><li class="PRIVATE_TreeView-item prc-TreeView-TreeViewItem-Ter5f" tabindex="-1" id="e2_20.xml-item" role="treeitem" aria-labelledby="_r_3c_" aria-describedby="_r_3d_" aria-level="1" aria-selected="false"><div class="PRIVATE_TreeView-item-container prc-TreeView-TreeViewItemContainer-z6qqQ" style="--level: 1; content-visibility: auto; contain-intrinsic-size: auto 2rem;"><div style="grid-area: spacer; display: flex;"><div style="width: 100%; display: flex;"></div></div><div id="_r_3c_" class="PRIVATE_TreeView-item-content prc-TreeView-TreeViewItemContent-RKsCI"><div class="PRIVATE_VisuallyHidden prc-TreeView-TreeViewVisuallyHidden-1N8xK" aria-hidden="true" id="_r_3d_"></div><div class="PRIVATE_TreeView-item-visual prc-TreeView-TreeViewItemVisual-naWzj" aria-hidden="true"><svg aria-hidden="true" focusable="false" class="octicon octicon-file" viewBox="0 0 16 16" width="16" height="16" fill="currentColor" display="inline-block" overflow="visible" style="vertical-align: text-bottom;"><path d="M2 1.75C2 .784 2.784 0 3.75 0h6.586c.464 0 .909.184 1.237.513l2.914 2.914c.329.328.513.773.513 1.237v9.586A1.75 1.75 0 0 1 13.25 16h-9.5A1.75 1.75 0 0 1 2 14.25Zm1.75-.25a.25.25 0 0 0-.25.25v12.5c0 .138.112.25.25.25h9.5a.25.25 0 0 0 .25-.25V6h-2.75A1.75 1.75 0 0 1 9 4.25V1.5Zm6.75.062V4.25c0 .138.112.25.25.25h2.688l-.011-.013-2.914-2.914-.013-.011Z"></path></svg></div><span class="PRIVATE_TreeView-item-content-text prc-TreeView-TreeViewItemContentText-FFaKp"><span>e2_20.xml</span></span></div></div></li><li class="PRIVATE_TreeView-item prc-TreeView-TreeViewItem-Ter5f" tabindex="-1" id="e2_21.xml-item" role="treeitem" aria-labelledby="_r_3g_" aria-describedby="_r_3h_" aria-level="1" aria-selected="false"><div class="PRIVATE_TreeView-item-container prc-TreeView-TreeViewItemContainer-z6qqQ" style="--level: 1; content-visibility: auto; contain-intrinsic-size: auto 2rem;"><div style="grid-area: spacer; display: flex;"><div style="width: 100%; display: flex;"></div></div><div id="_r_3g_" class="PRIVATE_TreeView-item-content prc-TreeView-TreeViewItemContent-RKsCI"><div class="PRIVATE_VisuallyHidden prc-TreeView-TreeViewVisuallyHidden-1N8xK" aria-hidden="true" id="_r_3h_"></div><div class="PRIVATE_TreeView-item-visual prc-TreeView-TreeViewItemVisual-naWzj" aria-hidden="true"><svg aria-hidden="true" focusable="false" class="octicon octicon-file" viewBox="0 0 16 16" width="16" height="16" fill="currentColor" display="inline-block" overflow="visible" style="vertical-align: text-bottom;"><path d="M2 1.75C2 .784 2.784 0 3.75 0h6.586c.464 0 .909.184 1.237.513l2.914 2.914c.329.328.513.773.513 1.237v9.586A1.75 1.75 0 0 1 13.25 16h-9.5A1.75 1.75 0 0 1 2 14.25Zm1.75-.25a.25.25 0 0 0-.25.25v12.5c0 .138.112.25.25.25h9.5a.25.25 0 0 0 .25-.25V6h-2.75A1.75 1.75 0 0 1 9 4.25V1.5Zm6.75.062V4.25c0 .138.112.25.25.25h2.688l-.011-.013-2.914-2.914-.013-.011Z"></path></svg></div><span class="PRIVATE_TreeView-item-content-text prc-TreeView-TreeViewItemContentText-FFaKp"><span>e2_21.xml</span></span></div></div></li><li class="PRIVATE_TreeView-item prc-TreeView-TreeViewItem-Ter5f" tabindex="-1" id="e2_22.xml-item" role="treeitem" aria-labelledby="_r_3k_" aria-describedby="_r_3l_" aria-level="1" aria-selected="false"><div class="PRIVATE_TreeView-item-container prc-TreeView-TreeViewItemContainer-z6qqQ" style="--level: 1; content-visibility: auto; contain-intrinsic-size: auto 2rem;"><div style="grid-area: spacer; display: flex;"><div style="width: 100%; display: flex;"></div></div><div id="_r_3k_" class="PRIVATE_TreeView-item-content prc-TreeView-TreeViewItemContent-RKsCI"><div class="PRIVATE_VisuallyHidden prc-TreeView-TreeViewVisuallyHidden-1N8xK" aria-hidden="true" id="_r_3l_"></div><div class="PRIVATE_TreeView-item-visual prc-TreeView-TreeViewItemVisual-naWzj" aria-hidden="true"><svg aria-hidden="true" focusable="false" class="octicon octicon-file" viewBox="0 0 16 16" width="16" height="16" fill="currentColor" display="inline-block" overflow="visible" style="vertical-align: text-bottom;"><path d="M2 1.75C2 .784 2.784 0 3.75 0h6.586c.464 0 .909.184 1.237.513l2.914 2.914c.329.328.513.773.513 1.237v9.586A1.75 1.75 0 0 1 13.25 16h-9.5A1.75 1.75 0 0 1 2 14.25Zm1.75-.25a.25.25 0 0 0-.25.25v12.5c0 .138.112.25.25.25h9.5a.25.25 0 0 0 .25-.25V6h-2.75A1.75 1.75 0 0 1 9 4.25V1.5Zm6.75.062V4.25c0 .138.112.25.25.25h2.688l-.011-.013-2.914-2.914-.013-.011Z"></path></svg></div><span class="PRIVATE_TreeView-item-content-text prc-TreeView-TreeViewItemContentText-FFaKp"><span>e2_22.xml</span></span></div></div></li><li class="PRIVATE_TreeView-item prc-TreeView-TreeViewItem-Ter5f" tabindex="-1" id="e2_23.xml-item" role="treeitem" aria-labelledby="_r_3o_" aria-describedby="_r_3p_" aria-level="1" aria-selected="false"><div class="PRIVATE_TreeView-item-container prc-TreeView-TreeViewItemContainer-z6qqQ" style="--level: 1; content-visibility: auto; contain-intrinsic-size: auto 2rem;"><div style="grid-area: spacer; display: flex;"><div style="width: 100%; display: flex;"></div></div><div id="_r_3o_" class="PRIVATE_TreeView-item-content prc-TreeView-TreeViewItemContent-RKsCI"><div class="PRIVATE_VisuallyHidden prc-TreeView-TreeViewVisuallyHidden-1N8xK" aria-hidden="true" id="_r_3p_"></div><div class="PRIVATE_TreeView-item-visual prc-TreeView-TreeViewItemVisual-naWzj" aria-hidden="true"><svg aria-hidden="true" focusable="false" class="octicon octicon-file" viewBox="0 0 16 16" width="16" height="16" fill="currentColor" display="inline-block" overflow="visible" style="vertical-align: text-bottom;"><path d="M2 1.75C2 .784 2.784 0 3.75 0h6.586c.464 0 .909.184 1.237.513l2.914 2.914c.329.328.513.773.513 1.237v9.586A1.75 1.75 0 0 1 13.25 16h-9.5A1.75 1.75 0 0 1 2 14.25Zm1.75-.25a.25.25 0 0 0-.25.25v12.5c0 .138.112.25.25.25h9.5a.25.25 0 0 0 .25-.25V6h-2.75A1.75 1.75 0 0 1 9 4.25V1.5Zm6.75.062V4.25c0 .138.112.25.25.25h2.688l-.011-.013-2.914-2.914-.013-.011Z"></path></svg></div><span class="PRIVATE_TreeView-item-content-text prc-TreeView-TreeViewItemContentText-FFaKp"><span>e2_23.xml</span></span></div></div></li><li class="PRIVATE_TreeView-item prc-TreeView-TreeViewItem-Ter5f" tabindex="-1" id="e2_24.xml-item" role="treeitem" aria-labelledby="_r_3s_" aria-describedby="_r_3t_" aria-level="1" aria-selected="false"><div class="PRIVATE_TreeView-item-container prc-TreeView-TreeViewItemContainer-z6qqQ" style="--level: 1; content-visibility: auto; contain-intrinsic-size: auto 2rem;"><div style="grid-area: spacer; display: flex;"><div style="width: 100%; display: flex;"></div></div><div id="_r_3s_" class="PRIVATE_TreeView-item-content prc-TreeView-TreeViewItemContent-RKsCI"><div class="PRIVATE_VisuallyHidden prc-TreeView-TreeViewVisuallyHidden-1N8xK" aria-hidden="true" id="_r_3t_"></div><div class="PRIVATE_TreeView-item-visual prc-TreeView-TreeViewItemVisual-naWzj" aria-hidden="true"><svg aria-hidden="true" focusable="false" class="octicon octicon-file" viewBox="0 0 16 16" width="16" height="16" fill="currentColor" display="inline-block" overflow="visible" style="vertical-align: text-bottom;"><path d="M2 1.75C2 .784 2.784 0 3.75 0h6.586c.464 0 .909.184 1.237.513l2.914 2.914c.329.328.513.773.513 1.237v9.586A1.75 1.75 0 0 1 13.25 16h-9.5A1.75 1.75 0 0 1 2 14.25Zm1.75-.25a.25.25 0 0 0-.25.25v12.5c0 .138.112.25.25.25h9.5a.25.25 0 0 0 .25-.25V6h-2.75A1.75 1.75 0 0 1 9 4.25V1.5Zm6.75.062V4.25c0 .138.112.25.25.25h2.688l-.011-.013-2.914-2.914-.013-.011Z"></path></svg></div><span class="PRIVATE_TreeView-item-content-text prc-TreeView-TreeViewItemContentText-FFaKp"><span>e2_24.xml</span></span></div></div></li><li class="PRIVATE_TreeView-item prc-TreeView-TreeViewItem-Ter5f" tabindex="-1" id="e2_25.xml-item" role="treeitem" aria-labelledby="_r_40_" aria-describedby="_r_41_" aria-level="1" aria-selected="false"><div class="PRIVATE_TreeView-item-container prc-TreeView-TreeViewItemContainer-z6qqQ" style="--level: 1; content-visibility: auto; contain-intrinsic-size: auto 2rem;"><div style="grid-area: spacer; display: flex;"><div style="width: 100%; display: flex;"></div></div><div id="_r_40_" class="PRIVATE_TreeView-item-content prc-TreeView-TreeViewItemContent-RKsCI"><div class="PRIVATE_VisuallyHidden prc-TreeView-TreeViewVisuallyHidden-1N8xK" aria-hidden="true" id="_r_41_"></div><div class="PRIVATE_TreeView-item-visual prc-TreeView-TreeViewItemVisual-naWzj" aria-hidden="true"><svg aria-hidden="true" focusable="false" class="octicon octicon-file" viewBox="0 0 16 16" width="16" height="16" fill="currentColor" display="inline-block" overflow="visible" style="vertical-align: text-bottom;"><path d="M2 1.75C2 .784 2.784 0 3.75 0h6.586c.464 0 .909.184 1.237.513l2.914 2.914c.329.328.513.773.513 1.237v9.586A1.75 1.75 0 0 1 13.25 16h-9.5A1.75 1.75 0 0 1 2 14.25Zm1.75-.25a.25.25 0 0 0-.25.25v12.5c0 .138.112.25.25.25h9.5a.25.25 0 0 0 .25-.25V6h-2.75A1.75 1.75 0 0 1 9 4.25V1.5Zm6.75.062V4.25c0 .138.112.25.25.25h2.688l-.011-.013-2.914-2.914-.013-.011Z"></path></svg></div><span class="PRIVATE_TreeView-item-content-text prc-TreeView-TreeViewItemContentText-FFaKp"><span>e2_25.xml</span></span></div></div></li><li class="PRIVATE_TreeView-item prc-TreeView-TreeViewItem-Ter5f" tabindex="-1" id="e2_26.xml-item" role="treeitem" aria-labelledby="_r_44_" aria-describedby="_r_45_" aria-level="1" aria-selected="false"><div class="PRIVATE_TreeView-item-container prc-TreeView-TreeViewItemContainer-z6qqQ" style="--level: 1; content-visibility: auto; contain-intrinsic-size: auto 2rem;"><div style="grid-area: spacer; display: flex;"><div style="width: 100%; display: flex;"></div></div><div id="_r_44_" class="PRIVATE_TreeView-item-content prc-TreeView-TreeViewItemContent-RKsCI"><div class="PRIVATE_VisuallyHidden prc-TreeView-TreeViewVisuallyHidden-1N8xK" aria-hidden="true" id="_r_45_"></div><div class="PRIVATE_TreeView-item-visual prc-TreeView-TreeViewItemVisual-naWzj" aria-hidden="true"><svg aria-hidden="true" focusable="false" class="octicon octicon-file" viewBox="0 0 16 16" width="16" height="16" fill="currentColor" display="inline-block" overflow="visible" style="vertical-align: text-bottom;"><path d="M2 1.75C2 .784 2.784 0 3.75 0h6.586c.464 0 .909.184 1.237.513l2.914 2.914c.329.328.513.773.513 1.237v9.586A1.75 1.75 0 0 1 13.25 16h-9.5A1.75 1.75 0 0 1 2 14.25Zm1.75-.25a.25.25 0 0 0-.25.25v12.5c0 .138.112.25.25.25h9.5a.25.25 0 0 0 .25-.25V6h-2.75A1.75 1.75 0 0 1 9 4.25V1.5Zm6.75.062V4.25c0 .138.112.25.25.25h2.688l-.011-.013-2.914-2.914-.013-.011Z"></path></svg></div><span class="PRIVATE_TreeView-item-content-text prc-TreeView-TreeViewItemContentText-FFaKp"><span>e2_26.xml</span></span></div></div></li><li class="PRIVATE_TreeView-item prc-TreeView-TreeViewItem-Ter5f" tabindex="-1" id="e2_27.xml-item" role="treeitem" aria-labelledby="_r_48_" aria-describedby="_r_49_" aria-level="1" aria-selected="false"><div class="PRIVATE_TreeView-item-container prc-TreeView-TreeViewItemContainer-z6qqQ" style="--level: 1; content-visibility: auto; contain-intrinsic-size: auto 2rem;"><div style="grid-area: spacer; display: flex;"><div style="width: 100%; display: flex;"></div></div><div id="_r_48_" class="PRIVATE_TreeView-item-content prc-TreeView-TreeViewItemContent-RKsCI"><div class="PRIVATE_VisuallyHidden prc-TreeView-TreeViewVisuallyHidden-1N8xK" aria-hidden="true" id="_r_49_"></div><div class="PRIVATE_TreeView-item-visual prc-TreeView-TreeViewItemVisual-naWzj" aria-hidden="true"><svg aria-hidden="true" focusable="false" class="octicon octicon-file" viewBox="0 0 16 16" width="16" height="16" fill="currentColor" display="inline-block" overflow="visible" style="vertical-align: text-bottom;"><path d="M2 1.75C2 .784 2.784 0 3.75 0h6.586c.464 0 .909.184 1.237.513l2.914 2.914c.329.328.513.773.513 1.237v9.586A1.75 1.75 0 0 1 13.25 16h-9.5A1.75 1.75 0 0 1 2 14.25Zm1.75-.25a.25.25 0 0 0-.25.25v12.5c0 .138.112.25.25.25h9.5a.25.25 0 0 0 .25-.25V6h-2.75A1.75 1.75 0 0 1 9 4.25V1.5Zm6.75.062V4.25c0 .138.112.25.25.25h2.688l-.011-.013-2.914-2.914-.013-.011Z"></path></svg></div><span class="PRIVATE_TreeView-item-content-text prc-TreeView-TreeViewItemContentText-FFaKp"><span>e2_27.xml</span></span></div></div></li><li class="PRIVATE_TreeView-item prc-TreeView-TreeViewItem-Ter5f" tabindex="-1" id="e2_28.xml-item" role="treeitem" aria-labelledby="_r_4c_" aria-describedby="_r_4d_" aria-level="1" aria-selected="false"><div class="PRIVATE_TreeView-item-container prc-TreeView-TreeViewItemContainer-z6qqQ" style="--level: 1; content-visibility: auto; contain-intrinsic-size: auto 2rem;"><div style="grid-area: spacer; display: flex;"><div style="width: 100%; display: flex;"></div></div><div id="_r_4c_" class="PRIVATE_TreeView-item-content prc-TreeView-TreeViewItemContent-RKsCI"><div class="PRIVATE_VisuallyHidden prc-TreeView-TreeViewVisuallyHidden-1N8xK" aria-hidden="true" id="_r_4d_"></div><div class="PRIVATE_TreeView-item-visual prc-TreeView-TreeViewItemVisual-naWzj" aria-hidden="true"><svg aria-hidden="true" focusable="false" class="octicon octicon-file" viewBox="0 0 16 16" width="16" height="16" fill="currentColor" display="inline-block" overflow="visible" style="vertical-align: text-bottom;"><path d="M2 1.75C2 .784 2.784 0 3.75 0h6.586c.464 0 .909.184 1.237.513l2.914 2.914c.329.328.513.773.513 1.237v9.586A1.75 1.75 0 0 1 13.25 16h-9.5A1.75 1.75 0 0 1 2 14.25Zm1.75-.25a.25.25 0 0 0-.25.25v12.5c0 .138.112.25.25.25h9.5a.25.25 0 0 0 .25-.25V6h-2.75A1.75 1.75 0 0 1 9 4.25V1.5Zm6.75.062V4.25c0 .138.112.25.25.25h2.688l-.011-.013-2.914-2.914-.013-.011Z"></path></svg></div><span class="PRIVATE_TreeView-item-content-text prc-TreeView-TreeViewItemContentText-FFaKp"><span>e2_28.xml</span></span></div></div></li><li class="PRIVATE_TreeView-item prc-TreeView-TreeViewItem-Ter5f" tabindex="-1" id="e2_29.xml-item" role="treeitem" aria-labelledby="_r_4g_" aria-describedby="_r_4h_" aria-level="1" aria-selected="false"><div class="PRIVATE_TreeView-item-container prc-TreeView-TreeViewItemContainer-z6qqQ" style="--level: 1; content-visibility: auto; contain-intrinsic-size: auto 2rem;"><div style="grid-area: spacer; display: flex;"><div style="width: 100%; display: flex;"></div></div><div id="_r_4g_" class="PRIVATE_TreeView-item-content prc-TreeView-TreeViewItemContent-RKsCI"><div class="PRIVATE_VisuallyHidden prc-TreeView-TreeViewVisuallyHidden-1N8xK" aria-hidden="true" id="_r_4h_"></div><div class="PRIVATE_TreeView-item-visual prc-TreeView-TreeViewItemVisual-naWzj" aria-hidden="true"><svg aria-hidden="true" focusable="false" class="octicon octicon-file" viewBox="0 0 16 16" width="16" height="16" fill="currentColor" display="inline-block" overflow="visible" style="vertical-align: text-bottom;"><path d="M2 1.75C2 .784 2.784 0 3.75 0h6.586c.464 0 .909.184 1.237.513l2.914 2.914c.329.328.513.773.513 1.237v9.586A1.75 1.75 0 0 1 13.25 16h-9.5A1.75 1.75 0 0 1 2 14.25Zm1.75-.25a.25.25 0 0 0-.25.25v12.5c0 .138.112.25.25.25h9.5a.25.25 0 0 0 .25-.25V6h-2.75A1.75 1.75 0 0 1 9 4.25V1.5Zm6.75.062V4.25c0 .138.112.25.25.25h2.688l-.011-.013-2.914-2.914-.013-.011Z"></path></svg></div><span class="PRIVATE_TreeView-item-content-text prc-TreeView-TreeViewItemContentText-FFaKp"><span>e2_29.xml</span></span></div></div></li><li class="PRIVATE_TreeView-item prc-TreeView-TreeViewItem-Ter5f" tabindex="-1" id="e2_3.xml-item" role="treeitem" aria-labelledby="_r_4k_" aria-describedby="_r_4l_" aria-level="1" aria-selected="false"><div class="PRIVATE_TreeView-item-container prc-TreeView-TreeViewItemContainer-z6qqQ" style="--level: 1; content-visibility: auto; contain-intrinsic-size: auto 2rem;"><div style="grid-area: spacer; display: flex;"><div style="width: 100%; display: flex;"></div></div><div id="_r_4k_" class="PRIVATE_TreeView-item-content prc-TreeView-TreeViewItemContent-RKsCI"><div class="PRIVATE_VisuallyHidden prc-TreeView-TreeViewVisuallyHidden-1N8xK" aria-hidden="true" id="_r_4l_"></div><div class="PRIVATE_TreeView-item-visual prc-TreeView-TreeViewItemVisual-naWzj" aria-hidden="true"><svg aria-hidden="true" focusable="false" class="octicon octicon-file" viewBox="0 0 16 16" width="16" height="16" fill="currentColor" display="inline-block" overflow="visible" style="vertical-align: text-bottom;"><path d="M2 1.75C2 .784 2.784 0 3.75 0h6.586c.464 0 .909.184 1.237.513l2.914 2.914c.329.328.513.773.513 1.237v9.586A1.75 1.75 0 0 1 13.25 16h-9.5A1.75 1.75 0 0 1 2 14.25Zm1.75-.25a.25.25 0 0 0-.25.25v12.5c0 .138.112.25.25.25h9.5a.25.25 0 0 0 .25-.25V6h-2.75A1.75 1.75 0 0 1 9 4.25V1.5Zm6.75.062V4.25c0 .138.112.25.25.25h2.688l-.011-.013-2.914-2.914-.013-.011Z"></path></svg></div><span class="PRIVATE_TreeView-item-content-text prc-TreeView-TreeViewItemContentText-FFaKp"><span>e2_3.xml</span></span></div></div></li><li class="PRIVATE_TreeView-item prc-TreeView-TreeViewItem-Ter5f" tabindex="-1" id="e2_30.xml-item" role="treeitem" aria-labelledby="_r_4o_" aria-describedby="_r_4p_" aria-level="1" aria-selected="false"><div class="PRIVATE_TreeView-item-container prc-TreeView-TreeViewItemContainer-z6qqQ" style="--level: 1; content-visibility: auto; contain-intrinsic-size: auto 2rem;"><div style="grid-area: spacer; display: flex;"><div style="width: 100%; display: flex;"></div></div><div id="_r_4o_" class="PRIVATE_TreeView-item-content prc-TreeView-TreeViewItemContent-RKsCI"><div class="PRIVATE_VisuallyHidden prc-TreeView-TreeViewVisuallyHidden-1N8xK" aria-hidden="true" id="_r_4p_"></div><div class="PRIVATE_TreeView-item-visual prc-TreeView-TreeViewItemVisual-naWzj" aria-hidden="true"><svg aria-hidden="true" focusable="false" class="octicon octicon-file" viewBox="0 0 16 16" width="16" height="16" fill="currentColor" display="inline-block" overflow="visible" style="vertical-align: text-bottom;"><path d="M2 1.75C2 .784 2.784 0 3.75 0h6.586c.464 0 .909.184 1.237.513l2.914 2.914c.329.328.513.773.513 1.237v9.586A1.75 1.75 0 0 1 13.25 16h-9.5A1.75 1.75 0 0 1 2 14.25Zm1.75-.25a.25.25 0 0 0-.25.25v12.5c0 .138.112.25.25.25h9.5a.25.25 0 0 0 .25-.25V6h-2.75A1.75 1.75 0 0 1 9 4.25V1.5Zm6.75.062V4.25c0 .138.112.25.25.25h2.688l-.011-.013-2.914-2.914-.013-.011Z"></path></svg></div><span class="PRIVATE_TreeView-item-content-text prc-TreeView-TreeViewItemContentText-FFaKp"><span>e2_30.xml</span></span></div></div></li><li class="PRIVATE_TreeView-item prc-TreeView-TreeViewItem-Ter5f" tabindex="-1" id="e2_31.xml-item" role="treeitem" aria-labelledby="_r_4s_" aria-describedby="_r_4t_" aria-level="1" aria-selected="false"><div class="PRIVATE_TreeView-item-container prc-TreeView-TreeViewItemContainer-z6qqQ" style="--level: 1; content-visibility: auto; contain-intrinsic-size: auto 2rem;"><div style="grid-area: spacer; display: flex;"><div style="width: 100%; display: flex;"></div></div><div id="_r_4s_" class="PRIVATE_TreeView-item-content prc-TreeView-TreeViewItemContent-RKsCI"><div class="PRIVATE_VisuallyHidden prc-TreeView-TreeViewVisuallyHidden-1N8xK" aria-hidden="true" id="_r_4t_"></div><div class="PRIVATE_TreeView-item-visual prc-TreeView-TreeViewItemVisual-naWzj" aria-hidden="true"><svg aria-hidden="true" focusable="false" class="octicon octicon-file" viewBox="0 0 16 16" width="16" height="16" fill="currentColor" display="inline-block" overflow="visible" style="vertical-align: text-bottom;"><path d="M2 1.75C2 .784 2.784 0 3.75 0h6.586c.464 0 .909.184 1.237.513l2.914 2.914c.329.328.513.773.513 1.237v9.586A1.75 1.75 0 0 1 13.25 16h-9.5A1.75 1.75 0 0 1 2 14.25Zm1.75-.25a.25.25 0 0 0-.25.25v12.5c0 .138.112.25.25.25h9.5a.25.25 0 0 0 .25-.25V6h-2.75A1.75 1.75 0 0 1 9 4.25V1.5Zm6.75.062V4.25c0 .138.112.25.25.25h2.688l-.011-.013-2.914-2.914-.013-.011Z"></path></svg></div><span class="PRIVATE_TreeView-item-content-text prc-TreeView-TreeViewItemContentText-FFaKp"><span>e2_31.xml</span></span></div></div></li><li class="PRIVATE_TreeView-item prc-TreeView-TreeViewItem-Ter5f" tabindex="-1" id="e2_32.xml-item" role="treeitem" aria-labelledby="_r_50_" aria-describedby="_r_51_" aria-level="1" aria-selected="false"><div class="PRIVATE_TreeView-item-container prc-TreeView-TreeViewItemContainer-z6qqQ" style="--level: 1; content-visibility: auto; contain-intrinsic-size: auto 2rem;"><div style="grid-area: spacer; display: flex;"><div style="width: 100%; display: flex;"></div></div><div id="_r_50_" class="PRIVATE_TreeView-item-content prc-TreeView-TreeViewItemContent-RKsCI"><div class="PRIVATE_VisuallyHidden prc-TreeView-TreeViewVisuallyHidden-1N8xK" aria-hidden="true" id="_r_51_"></div><div class="PRIVATE_TreeView-item-visual prc-TreeView-TreeViewItemVisual-naWzj" aria-hidden="true"><svg aria-hidden="true" focusable="false" class="octicon octicon-file" viewBox="0 0 16 16" width="16" height="16" fill="currentColor" display="inline-block" overflow="visible" style="vertical-align: text-bottom;"><path d="M2 1.75C2 .784 2.784 0 3.75 0h6.586c.464 0 .909.184 1.237.513l2.914 2.914c.329.328.513.773.513 1.237v9.586A1.75 1.75 0 0 1 13.25 16h-9.5A1.75 1.75 0 0 1 2 14.25Zm1.75-.25a.25.25 0 0 0-.25.25v12.5c0 .138.112.25.25.25h9.5a.25.25 0 0 0 .25-.25V6h-2.75A1.75 1.75 0 0 1 9 4.25V1.5Zm6.75.062V4.25c0 .138.112.25.25.25h2.688l-.011-.013-2.914-2.914-.013-.011Z"></path></svg></div><span class="PRIVATE_TreeView-item-content-text prc-TreeView-TreeViewItemContentText-FFaKp"><span>e2_32.xml</span></span></div></div></li><li class="PRIVATE_TreeView-item prc-TreeView-TreeViewItem-Ter5f" tabindex="-1" id="e2_33.xml-item" role="treeitem" aria-labelledby="_r_54_" aria-describedby="_r_55_" aria-level="1" aria-selected="false"><div class="PRIVATE_TreeView-item-container prc-TreeView-TreeViewItemContainer-z6qqQ" style="--level: 1; content-visibility: auto; contain-intrinsic-size: auto 2rem;"><div style="grid-area: spacer; display: flex;"><div style="width: 100%; display: flex;"></div></div><div id="_r_54_" class="PRIVATE_TreeView-item-content prc-TreeView-TreeViewItemContent-RKsCI"><div class="PRIVATE_VisuallyHidden prc-TreeView-TreeViewVisuallyHidden-1N8xK" aria-hidden="true" id="_r_55_"></div><div class="PRIVATE_TreeView-item-visual prc-TreeView-TreeViewItemVisual-naWzj" aria-hidden="true"><svg aria-hidden="true" focusable="false" class="octicon octicon-file" viewBox="0 0 16 16" width="16" height="16" fill="currentColor" display="inline-block" overflow="visible" style="vertical-align: text-bottom;"><path d="M2 1.75C2 .784 2.784 0 3.75 0h6.586c.464 0 .909.184 1.237.513l2.914 2.914c.329.328.513.773.513 1.237v9.586A1.75 1.75 0 0 1 13.25 16h-9.5A1.75 1.75 0 0 1 2 14.25Zm1.75-.25a.25.25 0 0 0-.25.25v12.5c0 .138.112.25.25.25h9.5a.25.25 0 0 0 .25-.25V6h-2.75A1.75 1.75 0 0 1 9 4.25V1.5Zm6.75.062V4.25c0 .138.112.25.25.25h2.688l-.011-.013-2.914-2.914-.013-.011Z"></path></svg></div><span class="PRIVATE_TreeView-item-content-text prc-TreeView-TreeViewItemContentText-FFaKp"><span>e2_33.xml</span></span></div></div></li><li class="PRIVATE_TreeView-item prc-TreeView-TreeViewItem-Ter5f" tabindex="-1" id="e2_34.xml-item" role="treeitem" aria-labelledby="_r_58_" aria-describedby="_r_59_" aria-level="1" aria-selected="false"><div class="PRIVATE_TreeView-item-container prc-TreeView-TreeViewItemContainer-z6qqQ" style="--level: 1; content-visibility: auto; contain-intrinsic-size: auto 2rem;"><div style="grid-area: spacer; display: flex;"><div style="width: 100%; display: flex;"></div></div><div id="_r_58_" class="PRIVATE_TreeView-item-content prc-TreeView-TreeViewItemContent-RKsCI"><div class="PRIVATE_VisuallyHidden prc-TreeView-TreeViewVisuallyHidden-1N8xK" aria-hidden="true" id="_r_59_"></div><div class="PRIVATE_TreeView-item-visual prc-TreeView-TreeViewItemVisual-naWzj" aria-hidden="true"><svg aria-hidden="true" focusable="false" class="octicon octicon-file" viewBox="0 0 16 16" width="16" height="16" fill="currentColor" display="inline-block" overflow="visible" style="vertical-align: text-bottom;"><path d="M2 1.75C2 .784 2.784 0 3.75 0h6.586c.464 0 .909.184 1.237.513l2.914 2.914c.329.328.513.773.513 1.237v9.586A1.75 1.75 0 0 1 13.25 16h-9.5A1.75 1.75 0 0 1 2 14.25Zm1.75-.25a.25.25 0 0 0-.25.25v12.5c0 .138.112.25.25.25h9.5a.25.25 0 0 0 .25-.25V6h-2.75A1.75 1.75 0 0 1 9 4.25V1.5Zm6.75.062V4.25c0 .138.112.25.25.25h2.688l-.011-.013-2.914-2.914-.013-.011Z"></path></svg></div><span class="PRIVATE_TreeView-item-content-text prc-TreeView-TreeViewItemContentText-FFaKp"><span>e2_34.xml</span></span></div></div></li><li class="PRIVATE_TreeView-item prc-TreeView-TreeViewItem-Ter5f" tabindex="-1" id="e2_35.xml-item" role="treeitem" aria-labelledby="_r_5c_" aria-describedby="_r_5d_" aria-level="1" aria-selected="false"><div class="PRIVATE_TreeView-item-container prc-TreeView-TreeViewItemContainer-z6qqQ" style="--level: 1; content-visibility: auto; contain-intrinsic-size: auto 2rem;"><div style="grid-area: spacer; display: flex;"><div style="width: 100%; display: flex;"></div></div><div id="_r_5c_" class="PRIVATE_TreeView-item-content prc-TreeView-TreeViewItemContent-RKsCI"><div class="PRIVATE_VisuallyHidden prc-TreeView-TreeViewVisuallyHidden-1N8xK" aria-hidden="true" id="_r_5d_"></div><div class="PRIVATE_TreeView-item-visual prc-TreeView-TreeViewItemVisual-naWzj" aria-hidden="true"><svg aria-hidden="true" focusable="false" class="octicon octicon-file" viewBox="0 0 16 16" width="16" height="16" fill="currentColor" display="inline-block" overflow="visible" style="vertical-align: text-bottom;"><path d="M2 1.75C2 .784 2.784 0 3.75 0h6.586c.464 0 .909.184 1.237.513l2.914 2.914c.329.328.513.773.513 1.237v9.586A1.75 1.75 0 0 1 13.25 16h-9.5A1.75 1.75 0 0 1 2 14.25Zm1.75-.25a.25.25 0 0 0-.25.25v12.5c0 .138.112.25.25.25h9.5a.25.25 0 0 0 .25-.25V6h-2.75A1.75 1.75 0 0 1 9 4.25V1.5Zm6.75.062V4.25c0 .138.112.25.25.25h2.688l-.011-.013-2.914-2.914-.013-.011Z"></path></svg></div><span class="PRIVATE_TreeView-item-content-text prc-TreeView-TreeViewItemContentText-FFaKp"><span>e2_35.xml</span></span></div></div></li><li class="PRIVATE_TreeView-item prc-TreeView-TreeViewItem-Ter5f" tabindex="-1" id="e2_36.xml-item" role="treeitem" aria-labelledby="_r_5g_" aria-describedby="_r_5h_" aria-level="1" aria-selected="false"><div class="PRIVATE_TreeView-item-container prc-TreeView-TreeViewItemContainer-z6qqQ" style="--level: 1; content-visibility: auto; contain-intrinsic-size: auto 2rem;"><div style="grid-area: spacer; display: flex;"><div style="width: 100%; display: flex;"></div></div><div id="_r_5g_" class="PRIVATE_TreeView-item-content prc-TreeView-TreeViewItemContent-RKsCI"><div class="PRIVATE_VisuallyHidden prc-TreeView-TreeViewVisuallyHidden-1N8xK" aria-hidden="true" id="_r_5h_"></div><div class="PRIVATE_TreeView-item-visual prc-TreeView-TreeViewItemVisual-naWzj" aria-hidden="true"><svg aria-hidden="true" focusable="false" class="octicon octicon-file" viewBox="0 0 16 16" width="16" height="16" fill="currentColor" display="inline-block" overflow="visible" style="vertical-align: text-bottom;"><path d="M2 1.75C2 .784 2.784 0 3.75 0h6.586c.464 0 .909.184 1.237.513l2.914 2.914c.329.328.513.773.513 1.237v9.586A1.75 1.75 0 0 1 13.25 16h-9.5A1.75 1.75 0 0 1 2 14.25Zm1.75-.25a.25.25 0 0 0-.25.25v12.5c0 .138.112.25.25.25h9.5a.25.25 0 0 0 .25-.25V6h-2.75A1.75 1.75 0 0 1 9 4.25V1.5Zm6.75.062V4.25c0 .138.112.25.25.25h2.688l-.011-.013-2.914-2.914-.013-.011Z"></path></svg></div><span class="PRIVATE_TreeView-item-content-text prc-TreeView-TreeViewItemContentText-FFaKp"><span>e2_36.xml</span></span></div></div></li><li class="PRIVATE_TreeView-item prc-TreeView-TreeViewItem-Ter5f" tabindex="-1" id="e2_37.xml-item" role="treeitem" aria-labelledby="_r_5k_" aria-describedby="_r_5l_" aria-level="1" aria-selected="false"><div class="PRIVATE_TreeView-item-container prc-TreeView-TreeViewItemContainer-z6qqQ" style="--level: 1; content-visibility: auto; contain-intrinsic-size: auto 2rem;"><div style="grid-area: spacer; display: flex;"><div style="width: 100%; display: flex;"></div></div><div id="_r_5k_" class="PRIVATE_TreeView-item-content prc-TreeView-TreeViewItemContent-RKsCI"><div class="PRIVATE_VisuallyHidden prc-TreeView-TreeViewVisuallyHidden-1N8xK" aria-hidden="true" id="_r_5l_"></div><div class="PRIVATE_TreeView-item-visual prc-TreeView-TreeViewItemVisual-naWzj" aria-hidden="true"><svg aria-hidden="true" focusable="false" class="octicon octicon-file" viewBox="0 0 16 16" width="16" height="16" fill="currentColor" display="inline-block" overflow="visible" style="vertical-align: text-bottom;"><path d="M2 1.75C2 .784 2.784 0 3.75 0h6.586c.464 0 .909.184 1.237.513l2.914 2.914c.329.328.513.773.513 1.237v9.586A1.75 1.75 0 0 1 13.25 16h-9.5A1.75 1.75 0 0 1 2 14.25Zm1.75-.25a.25.25 0 0 0-.25.25v12.5c0 .138.112.25.25.25h9.5a.25.25 0 0 0 .25-.25V6h-2.75A1.75 1.75 0 0 1 9 4.25V1.5Zm6.75.062V4.25c0 .138.112.25.25.25h2.688l-.011-.013-2.914-2.914-.013-.011Z"></path></svg></div><span class="PRIVATE_TreeView-item-content-text prc-TreeView-TreeViewItemContentText-FFaKp"><span>e2_37.xml</span></span></div></div></li><li class="PRIVATE_TreeView-item prc-TreeView-TreeViewItem-Ter5f" tabindex="-1" id="e2_38.xml-item" role="treeitem" aria-labelledby="_r_5o_" aria-describedby="_r_5p_" aria-level="1" aria-selected="false"><div class="PRIVATE_TreeView-item-container prc-TreeView-TreeViewItemContainer-z6qqQ" style="--level: 1; content-visibility: auto; contain-intrinsic-size: auto 2rem;"><div style="grid-area: spacer; display: flex;"><div style="width: 100%; display: flex;"></div></div><div id="_r_5o_" class="PRIVATE_TreeView-item-content prc-TreeView-TreeViewItemContent-RKsCI"><div class="PRIVATE_VisuallyHidden prc-TreeView-TreeViewVisuallyHidden-1N8xK" aria-hidden="true" id="_r_5p_"></div><div class="PRIVATE_TreeView-item-visual prc-TreeView-TreeViewItemVisual-naWzj" aria-hidden="true"><svg aria-hidden="true" focusable="false" class="octicon octicon-file" viewBox="0 0 16 16" width="16" height="16" fill="currentColor" display="inline-block" overflow="visible" style="vertical-align: text-bottom;"><path d="M2 1.75C2 .784 2.784 0 3.75 0h6.586c.464 0 .909.184 1.237.513l2.914 2.914c.329.328.513.773.513 1.237v9.586A1.75 1.75 0 0 1 13.25 16h-9.5A1.75 1.75 0 0 1 2 14.25Zm1.75-.25a.25.25 0 0 0-.25.25v12.5c0 .138.112.25.25.25h9.5a.25.25 0 0 0 .25-.25V6h-2.75A1.75 1.75 0 0 1 9 4.25V1.5Zm6.75.062V4.25c0 .138.112.25.25.25h2.688l-.011-.013-2.914-2.914-.013-.011Z"></path></svg></div><span class="PRIVATE_TreeView-item-content-text prc-TreeView-TreeViewItemContentText-FFaKp"><span>e2_38.xml</span></span></div></div></li><li class="PRIVATE_TreeView-item prc-TreeView-TreeViewItem-Ter5f" tabindex="-1" id="e2_39.xml-item" role="treeitem" aria-labelledby="_r_5s_" aria-describedby="_r_5t_" aria-level="1" aria-selected="false"><div class="PRIVATE_TreeView-item-container prc-TreeView-TreeViewItemContainer-z6qqQ" style="--level: 1; content-visibility: auto; contain-intrinsic-size: auto 2rem;"><div style="grid-area: spacer; display: flex;"><div style="width: 100%; display: flex;"></div></div><div id="_r_5s_" class="PRIVATE_TreeView-item-content prc-TreeView-TreeViewItemContent-RKsCI"><div class="PRIVATE_VisuallyHidden prc-TreeView-TreeViewVisuallyHidden-1N8xK" aria-hidden="true" id="_r_5t_"></div><div class="PRIVATE_TreeView-item-visual prc-TreeView-TreeViewItemVisual-naWzj" aria-hidden="true"><svg aria-hidden="true" focusable="false" class="octicon octicon-file" viewBox="0 0 16 16" width="16" height="16" fill="currentColor" display="inline-block" overflow="visible" style="vertical-align: text-bottom;"><path d="M2 1.75C2 .784 2.784 0 3.75 0h6.586c.464 0 .909.184 1.237.513l2.914 2.914c.329.328.513.773.513 1.237v9.586A1.75 1.75 0 0 1 13.25 16h-9.5A1.75 1.75 0 0 1 2 14.25Zm1.75-.25a.25.25 0 0 0-.25.25v12.5c0 .138.112.25.25.25h9.5a.25.25 0 0 0 .25-.25V6h-2.75A1.75 1.75 0 0 1 9 4.25V1.5Zm6.75.062V4.25c0 .138.112.25.25.25h2.688l-.011-.013-2.914-2.914-.013-.011Z"></path></svg></div><span class="PRIVATE_TreeView-item-content-text prc-TreeView-TreeViewItemContentText-FFaKp"><span>e2_39.xml</span></span></div></div></li><li class="PRIVATE_TreeView-item prc-TreeView-TreeViewItem-Ter5f" tabindex="-1" id="e2_4.xml-item" role="treeitem" aria-labelledby="_r_60_" aria-describedby="_r_61_" aria-level="1" aria-selected="false"><div class="PRIVATE_TreeView-item-container prc-TreeView-TreeViewItemContainer-z6qqQ" style="--level: 1; content-visibility: auto; contain-intrinsic-size: auto 2rem;"><div style="grid-area: spacer; display: flex;"><div style="width: 100%; display: flex;"></div></div><div id="_r_60_" class="PRIVATE_TreeView-item-content prc-TreeView-TreeViewItemContent-RKsCI"><div class="PRIVATE_VisuallyHidden prc-TreeView-TreeViewVisuallyHidden-1N8xK" aria-hidden="true" id="_r_61_"></div><div class="PRIVATE_TreeView-item-visual prc-TreeView-TreeViewItemVisual-naWzj" aria-hidden="true"><svg aria-hidden="true" focusable="false" class="octicon octicon-file" viewBox="0 0 16 16" width="16" height="16" fill="currentColor" display="inline-block" overflow="visible" style="vertical-align: text-bottom;"><path d="M2 1.75C2 .784 2.784 0 3.75 0h6.586c.464 0 .909.184 1.237.513l2.914 2.914c.329.328.513.773.513 1.237v9.586A1.75 1.75 0 0 1 13.25 16h-9.5A1.75 1.75 0 0 1 2 14.25Zm1.75-.25a.25.25 0 0 0-.25.25v12.5c0 .138.112.25.25.25h9.5a.25.25 0 0 0 .25-.25V6h-2.75A1.75 1.75 0 0 1 9 4.25V1.5Zm6.75.062V4.25c0 .138.112.25.25.25h2.688l-.011-.013-2.914-2.914-.013-.011Z"></path></svg></div><span class="PRIVATE_TreeView-item-content-text prc-TreeView-TreeViewItemContentText-FFaKp"><span>e2_4.xml</span></span></div></div></li><li class="PRIVATE_TreeView-item prc-TreeView-TreeViewItem-Ter5f" tabindex="-1" id="e2_40.xml-item" role="treeitem" aria-labelledby="_r_64_" aria-describedby="_r_65_" aria-level="1" aria-selected="false"><div class="PRIVATE_TreeView-item-container prc-TreeView-TreeViewItemContainer-z6qqQ" style="--level: 1; content-visibility: auto; contain-intrinsic-size: auto 2rem;"><div style="grid-area: spacer; display: flex;"><div style="width: 100%; display: flex;"></div></div><div id="_r_64_" class="PRIVATE_TreeView-item-content prc-TreeView-TreeViewItemContent-RKsCI"><div class="PRIVATE_VisuallyHidden prc-TreeView-TreeViewVisuallyHidden-1N8xK" aria-hidden="true" id="_r_65_"></div><div class="PRIVATE_TreeView-item-visual prc-TreeView-TreeViewItemVisual-naWzj" aria-hidden="true"><svg aria-hidden="true" focusable="false" class="octicon octicon-file" viewBox="0 0 16 16" width="16" height="16" fill="currentColor" display="inline-block" overflow="visible" style="vertical-align: text-bottom;"><path d="M2 1.75C2 .784 2.784 0 3.75 0h6.586c.464 0 .909.184 1.237.513l2.914 2.914c.329.328.513.773.513 1.237v9.586A1.75 1.75 0 0 1 13.25 16h-9.5A1.75 1.75 0 0 1 2 14.25Zm1.75-.25a.25.25 0 0 0-.25.25v12.5c0 .138.112.25.25.25h9.5a.25.25 0 0 0 .25-.25V6h-2.75A1.75 1.75 0 0 1 9 4.25V1.5Zm6.75.062V4.25c0 .138.112.25.25.25h2.688l-.011-.013-2.914-2.914-.013-.011Z"></path></svg></div><span class="PRIVATE_TreeView-item-content-text prc-TreeView-TreeViewItemContentText-FFaKp"><span>e2_40.xml</span></span></div></div></li><li class="PRIVATE_TreeView-item prc-TreeView-TreeViewItem-Ter5f" tabindex="-1" id="e2_41.xml-item" role="treeitem" aria-labelledby="_r_68_" aria-describedby="_r_69_" aria-level="1" aria-selected="false"><div class="PRIVATE_TreeView-item-container prc-TreeView-TreeViewItemContainer-z6qqQ" style="--level: 1; content-visibility: auto; contain-intrinsic-size: auto 2rem;"><div style="grid-area: spacer; display: flex;"><div style="width: 100%; display: flex;"></div></div><div id="_r_68_" class="PRIVATE_TreeView-item-content prc-TreeView-TreeViewItemContent-RKsCI"><div class="PRIVATE_VisuallyHidden prc-TreeView-TreeViewVisuallyHidden-1N8xK" aria-hidden="true" id="_r_69_"></div><div class="PRIVATE_TreeView-item-visual prc-TreeView-TreeViewItemVisual-naWzj" aria-hidden="true"><svg aria-hidden="true" focusable="false" class="octicon octicon-file" viewBox="0 0 16 16" width="16" height="16" fill="currentColor" display="inline-block" overflow="visible" style="vertical-align: text-bottom;"><path d="M2 1.75C2 .784 2.784 0 3.75 0h6.586c.464 0 .909.184 1.237.513l2.914 2.914c.329.328.513.773.513 1.237v9.586A1.75 1.75 0 0 1 13.25 16h-9.5A1.75 1.75 0 0 1 2 14.25Zm1.75-.25a.25.25 0 0 0-.25.25v12.5c0 .138.112.25.25.25h9.5a.25.25 0 0 0 .25-.25V6h-2.75A1.75 1.75 0 0 1 9 4.25V1.5Zm6.75.062V4.25c0 .138.112.25.25.25h2.688l-.011-.013-2.914-2.914-.013-.011Z"></path></svg></div><span class="PRIVATE_TreeView-item-content-text prc-TreeView-TreeViewItemContentText-FFaKp"><span>e2_41.xml</span></span></div></div></li><li class="PRIVATE_TreeView-item prc-TreeView-TreeViewItem-Ter5f" tabindex="-1" id="e2_42.xml-item" role="treeitem" aria-labelledby="_r_6c_" aria-describedby="_r_6d_" aria-level="1" aria-selected="false"><div class="PRIVATE_TreeView-item-container prc-TreeView-TreeViewItemContainer-z6qqQ" style="--level: 1; content-visibility: auto; contain-intrinsic-size: auto 2rem;"><div style="grid-area: spacer; display: flex;"><div style="width: 100%; display: flex;"></div></div><div id="_r_6c_" class="PRIVATE_TreeView-item-content prc-TreeView-TreeViewItemContent-RKsCI"><div class="PRIVATE_VisuallyHidden prc-TreeView-TreeViewVisuallyHidden-1N8xK" aria-hidden="true" id="_r_6d_"></div><div class="PRIVATE_TreeView-item-visual prc-TreeView-TreeViewItemVisual-naWzj" aria-hidden="true"><svg aria-hidden="true" focusable="false" class="octicon octicon-file" viewBox="0 0 16 16" width="16" height="16" fill="currentColor" display="inline-block" overflow="visible" style="vertical-align: text-bottom;"><path d="M2 1.75C2 .784 2.784 0 3.75 0h6.586c.464 0 .909.184 1.237.513l2.914 2.914c.329.328.513.773.513 1.237v9.586A1.75 1.75 0 0 1 13.25 16h-9.5A1.75 1.75 0 0 1 2 14.25Zm1.75-.25a.25.25 0 0 0-.25.25v12.5c0 .138.112.25.25.25h9.5a.25.25 0 0 0 .25-.25V6h-2.75A1.75 1.75 0 0 1 9 4.25V1.5Zm6.75.062V4.25c0 .138.112.25.25.25h2.688l-.011-.013-2.914-2.914-.013-.011Z"></path></svg></div><span class="PRIVATE_TreeView-item-content-text prc-TreeView-TreeViewItemContentText-FFaKp"><span>e2_42.xml</span></span></div></div></li><li class="PRIVATE_TreeView-item prc-TreeView-TreeViewItem-Ter5f" tabindex="-1" id="e2_43.xml-item" role="treeitem" aria-labelledby="_r_6g_" aria-describedby="_r_6h_" aria-level="1" aria-selected="false"><div class="PRIVATE_TreeView-item-container prc-TreeView-TreeViewItemContainer-z6qqQ" style="--level: 1; content-visibility: auto; contain-intrinsic-size: auto 2rem;"><div style="grid-area: spacer; display: flex;"><div style="width: 100%; display: flex;"></div></div><div id="_r_6g_" class="PRIVATE_TreeView-item-content prc-TreeView-TreeViewItemContent-RKsCI"><div class="PRIVATE_VisuallyHidden prc-TreeView-TreeViewVisuallyHidden-1N8xK" aria-hidden="true" id="_r_6h_"></div><div class="PRIVATE_TreeView-item-visual prc-TreeView-TreeViewItemVisual-naWzj" aria-hidden="true"><svg aria-hidden="true" focusable="false" class="octicon octicon-file" viewBox="0 0 16 16" width="16" height="16" fill="currentColor" display="inline-block" overflow="visible" style="vertical-align: text-bottom;"><path d="M2 1.75C2 .784 2.784 0 3.75 0h6.586c.464 0 .909.184 1.237.513l2.914 2.914c.329.328.513.773.513 1.237v9.586A1.75 1.75 0 0 1 13.25 16h-9.5A1.75 1.75 0 0 1 2 14.25Zm1.75-.25a.25.25 0 0 0-.25.25v12.5c0 .138.112.25.25.25h9.5a.25.25 0 0 0 .25-.25V6h-2.75A1.75 1.75 0 0 1 9 4.25V1.5Zm6.75.062V4.25c0 .138.112.25.25.25h2.688l-.011-.013-2.914-2.914-.013-.011Z"></path></svg></div><span class="PRIVATE_TreeView-item-content-text prc-TreeView-TreeViewItemContentText-FFaKp"><span>e2_43.xml</span></span></div></div></li><li class="PRIVATE_TreeView-item prc-TreeView-TreeViewItem-Ter5f" tabindex="-1" id="e2_44.xml-item" role="treeitem" aria-labelledby="_r_6k_" aria-describedby="_r_6l_" aria-level="1" aria-selected="false"><div class="PRIVATE_TreeView-item-container prc-TreeView-TreeViewItemContainer-z6qqQ" style="--level: 1; content-visibility: auto; contain-intrinsic-size: auto 2rem;"><div style="grid-area: spacer; display: flex;"><div style="width: 100%; display: flex;"></div></div><div id="_r_6k_" class="PRIVATE_TreeView-item-content prc-TreeView-TreeViewItemContent-RKsCI"><div class="PRIVATE_VisuallyHidden prc-TreeView-TreeViewVisuallyHidden-1N8xK" aria-hidden="true" id="_r_6l_"></div><div class="PRIVATE_TreeView-item-visual prc-TreeView-TreeViewItemVisual-naWzj" aria-hidden="true"><svg aria-hidden="true" focusable="false" class="octicon octicon-file" viewBox="0 0 16 16" width="16" height="16" fill="currentColor" display="inline-block" overflow="visible" style="vertical-align: text-bottom;"><path d="M2 1.75C2 .784 2.784 0 3.75 0h6.586c.464 0 .909.184 1.237.513l2.914 2.914c.329.328.513.773.513 1.237v9.586A1.75 1.75 0 0 1 13.25 16h-9.5A1.75 1.75 0 0 1 2 14.25Zm1.75-.25a.25.25 0 0 0-.25.25v12.5c0 .138.112.25.25.25h9.5a.25.25 0 0 0 .25-.25V6h-2.75A1.75 1.75 0 0 1 9 4.25V1.5Zm6.75.062V4.25c0 .138.112.25.25.25h2.688l-.011-.013-2.914-2.914-.013-.011Z"></path></svg></div><span class="PRIVATE_TreeView-item-content-text prc-TreeView-TreeViewItemContentText-FFaKp"><span>e2_44.xml</span></span></div></div></li><li class="PRIVATE_TreeView-item prc-TreeView-TreeViewItem-Ter5f" tabindex="-1" id="e2_45.xml-item" role="treeitem" aria-labelledby="_r_6o_" aria-describedby="_r_6p_" aria-level="1" aria-selected="false"><div class="PRIVATE_TreeView-item-container prc-TreeView-TreeViewItemContainer-z6qqQ" style="--level: 1; content-visibility: auto; contain-intrinsic-size: auto 2rem;"><div style="grid-area: spacer; display: flex;"><div style="width: 100%; display: flex;"></div></div><div id="_r_6o_" class="PRIVATE_TreeView-item-content prc-TreeView-TreeViewItemContent-RKsCI"><div class="PRIVATE_VisuallyHidden prc-TreeView-TreeViewVisuallyHidden-1N8xK" aria-hidden="true" id="_r_6p_"></div><div class="PRIVATE_TreeView-item-visual prc-TreeView-TreeViewItemVisual-naWzj" aria-hidden="true"><svg aria-hidden="true" focusable="false" class="octicon octicon-file" viewBox="0 0 16 16" width="16" height="16" fill="currentColor" display="inline-block" overflow="visible" style="vertical-align: text-bottom;"><path d="M2 1.75C2 .784 2.784 0 3.75 0h6.586c.464 0 .909.184 1.237.513l2.914 2.914c.329.328.513.773.513 1.237v9.586A1.75 1.75 0 0 1 13.25 16h-9.5A1.75 1.75 0 0 1 2 14.25Zm1.75-.25a.25.25 0 0 0-.25.25v12.5c0 .138.112.25.25.25h9.5a.25.25 0 0 0 .25-.25V6h-2.75A1.75 1.75 0 0 1 9 4.25V1.5Zm6.75.062V4.25c0 .138.112.25.25.25h2.688l-.011-.013-2.914-2.914-.013-.011Z"></path></svg></div><span class="PRIVATE_TreeView-item-content-text prc-TreeView-TreeViewItemContentText-FFaKp"><span>e2_45.xml</span></span></div></div></li><li class="PRIVATE_TreeView-item prc-TreeView-TreeViewItem-Ter5f" tabindex="-1" id="e2_46.xml-item" role="treeitem" aria-labelledby="_r_6s_" aria-describedby="_r_6t_" aria-level="1" aria-selected="false"><div class="PRIVATE_TreeView-item-container prc-TreeView-TreeViewItemContainer-z6qqQ" style="--level: 1; content-visibility: auto; contain-intrinsic-size: auto 2rem;"><div style="grid-area: spacer; display: flex;"><div style="width: 100%; display: flex;"></div></div><div id="_r_6s_" class="PRIVATE_TreeView-item-content prc-TreeView-TreeViewItemContent-RKsCI"><div class="PRIVATE_VisuallyHidden prc-TreeView-TreeViewVisuallyHidden-1N8xK" aria-hidden="true" id="_r_6t_"></div><div class="PRIVATE_TreeView-item-visual prc-TreeView-TreeViewItemVisual-naWzj" aria-hidden="true"><svg aria-hidden="true" focusable="false" class="octicon octicon-file" viewBox="0 0 16 16" width="16" height="16" fill="currentColor" display="inline-block" overflow="visible" style="vertical-align: text-bottom;"><path d="M2 1.75C2 .784 2.784 0 3.75 0h6.586c.464 0 .909.184 1.237.513l2.914 2.914c.329.328.513.773.513 1.237v9.586A1.75 1.75 0 0 1 13.25 16h-9.5A1.75 1.75 0 0 1 2 14.25Zm1.75-.25a.25.25 0 0 0-.25.25v12.5c0 .138.112.25.25.25h9.5a.25.25 0 0 0 .25-.25V6h-2.75A1.75 1.75 0 0 1 9 4.25V1.5Zm6.75.062V4.25c0 .138.112.25.25.25h2.688l-.011-.013-2.914-2.914-.013-.011Z"></path></svg></div><span class="PRIVATE_TreeView-item-content-text prc-TreeView-TreeViewItemContentText-FFaKp"><span>e2_46.xml</span></span></div></div></li><li class="PRIVATE_TreeView-item prc-TreeView-TreeViewItem-Ter5f" tabindex="-1" id="e2_47.xml-item" role="treeitem" aria-labelledby="_r_70_" aria-describedby="_r_71_" aria-level="1" aria-selected="false"><div class="PRIVATE_TreeView-item-container prc-TreeView-TreeViewItemContainer-z6qqQ" style="--level: 1; content-visibility: auto; contain-intrinsic-size: auto 2rem;"><div style="grid-area: spacer; display: flex;"><div style="width: 100%; display: flex;"></div></div><div id="_r_70_" class="PRIVATE_TreeView-item-content prc-TreeView-TreeViewItemContent-RKsCI"><div class="PRIVATE_VisuallyHidden prc-TreeView-TreeViewVisuallyHidden-1N8xK" aria-hidden="true" id="_r_71_"></div><div class="PRIVATE_TreeView-item-visual prc-TreeView-TreeViewItemVisual-naWzj" aria-hidden="true"><svg aria-hidden="true" focusable="false" class="octicon octicon-file" viewBox="0 0 16 16" width="16" height="16" fill="currentColor" display="inline-block" overflow="visible" style="vertical-align: text-bottom;"><path d="M2 1.75C2 .784 2.784 0 3.75 0h6.586c.464 0 .909.184 1.237.513l2.914 2.914c.329.328.513.773.513 1.237v9.586A1.75 1.75 0 0 1 13.25 16h-9.5A1.75 1.75 0 0 1 2 14.25Zm1.75-.25a.25.25 0 0 0-.25.25v12.5c0 .138.112.25.25.25h9.5a.25.25 0 0 0 .25-.25V6h-2.75A1.75 1.75 0 0 1 9 4.25V1.5Zm6.75.062V4.25c0 .138.112.25.25.25h2.688l-.011-.013-2.914-2.914-.013-.011Z"></path></svg></div><span class="PRIVATE_TreeView-item-content-text prc-TreeView-TreeViewItemContentText-FFaKp"><span>e2_47.xml</span></span></div></div></li><li class="PRIVATE_TreeView-item prc-TreeView-TreeViewItem-Ter5f" tabindex="-1" id="e2_48.xml-item" role="treeitem" aria-labelledby="_r_74_" aria-describedby="_r_75_" aria-level="1" aria-selected="false"><div class="PRIVATE_TreeView-item-container prc-TreeView-TreeViewItemContainer-z6qqQ" style="--level: 1; content-visibility: auto; contain-intrinsic-size: auto 2rem;"><div style="grid-area: spacer; display: flex;"><div style="width: 100%; display: flex;"></div></div><div id="_r_74_" class="PRIVATE_TreeView-item-content prc-TreeView-TreeViewItemContent-RKsCI"><div class="PRIVATE_VisuallyHidden prc-TreeView-TreeViewVisuallyHidden-1N8xK" aria-hidden="true" id="_r_75_"></div><div class="PRIVATE_TreeView-item-visual prc-TreeView-TreeViewItemVisual-naWzj" aria-hidden="true"><svg aria-hidden="true" focusable="false" class="octicon octicon-file" viewBox="0 0 16 16" width="16" height="16" fill="currentColor" display="inline-block" overflow="visible" style="vertical-align: text-bottom;"><path d="M2 1.75C2 .784 2.784 0 3.75 0h6.586c.464 0 .909.184 1.237.513l2.914 2.914c.329.328.513.773.513 1.237v9.586A1.75 1.75 0 0 1 13.25 16h-9.5A1.75 1.75 0 0 1 2 14.25Zm1.75-.25a.25.25 0 0 0-.25.25v12.5c0 .138.112.25.25.25h9.5a.25.25 0 0 0 .25-.25V6h-2.75A1.75 1.75 0 0 1 9 4.25V1.5Zm6.75.062V4.25c0 .138.112.25.25.25h2.688l-.011-.013-2.914-2.914-.013-.011Z"></path></svg></div><span class="PRIVATE_TreeView-item-content-text prc-TreeView-TreeViewItemContentText-FFaKp"><span>e2_48.xml</span></span></div></div></li><li class="PRIVATE_TreeView-item prc-TreeView-TreeViewItem-Ter5f" tabindex="-1" id="e2_49.xml-item" role="treeitem" aria-labelledby="_r_78_" aria-describedby="_r_79_" aria-level="1" aria-selected="false"><div class="PRIVATE_TreeView-item-container prc-TreeView-TreeViewItemContainer-z6qqQ" style="--level: 1; content-visibility: auto; contain-intrinsic-size: auto 2rem;"><div style="grid-area: spacer; display: flex;"><div style="width: 100%; display: flex;"></div></div><div id="_r_78_" class="PRIVATE_TreeView-item-content prc-TreeView-TreeViewItemContent-RKsCI"><div class="PRIVATE_VisuallyHidden prc-TreeView-TreeViewVisuallyHidden-1N8xK" aria-hidden="true" id="_r_79_"></div><div class="PRIVATE_TreeView-item-visual prc-TreeView-TreeViewItemVisual-naWzj" aria-hidden="true"><svg aria-hidden="true" focusable="false" class="octicon octicon-file" viewBox="0 0 16 16" width="16" height="16" fill="currentColor" display="inline-block" overflow="visible" style="vertical-align: text-bottom;"><path d="M2 1.75C2 .784 2.784 0 3.75 0h6.586c.464 0 .909.184 1.237.513l2.914 2.914c.329.328.513.773.513 1.237v9.586A1.75 1.75 0 0 1 13.25 16h-9.5A1.75 1.75 0 0 1 2 14.25Zm1.75-.25a.25.25 0 0 0-.25.25v12.5c0 .138.112.25.25.25h9.5a.25.25 0 0 0 .25-.25V6h-2.75A1.75 1.75 0 0 1 9 4.25V1.5Zm6.75.062V4.25c0 .138.112.25.25.25h2.688l-.011-.013-2.914-2.914-.013-.011Z"></path></svg></div><span class="PRIVATE_TreeView-item-content-text prc-TreeView-TreeViewItemContentText-FFaKp"><span>e2_49.xml</span></span></div></div></li><li class="PRIVATE_TreeView-item prc-TreeView-TreeViewItem-Ter5f" tabindex="-1" id="e2_5.xml-item" role="treeitem" aria-labelledby="_r_7c_" aria-describedby="_r_7d_" aria-level="1" aria-selected="false"><div class="PRIVATE_TreeView-item-container prc-TreeView-TreeViewItemContainer-z6qqQ" style="--level: 1; content-visibility: auto; contain-intrinsic-size: auto 2rem;"><div style="grid-area: spacer; display: flex;"><div style="width: 100%; display: flex;"></div></div><div id="_r_7c_" class="PRIVATE_TreeView-item-content prc-TreeView-TreeViewItemContent-RKsCI"><div class="PRIVATE_VisuallyHidden prc-TreeView-TreeViewVisuallyHidden-1N8xK" aria-hidden="true" id="_r_7d_"></div><div class="PRIVATE_TreeView-item-visual prc-TreeView-TreeViewItemVisual-naWzj" aria-hidden="true"><svg aria-hidden="true" focusable="false" class="octicon octicon-file" viewBox="0 0 16 16" width="16" height="16" fill="currentColor" display="inline-block" overflow="visible" style="vertical-align: text-bottom;"><path d="M2 1.75C2 .784 2.784 0 3.75 0h6.586c.464 0 .909.184 1.237.513l2.914 2.914c.329.328.513.773.513 1.237v9.586A1.75 1.75 0 0 1 13.25 16h-9.5A1.75 1.75 0 0 1 2 14.25Zm1.75-.25a.25.25 0 0 0-.25.25v12.5c0 .138.112.25.25.25h9.5a.25.25 0 0 0 .25-.25V6h-2.75A1.75 1.75 0 0 1 9 4.25V1.5Zm6.75.062V4.25c0 .138.112.25.25.25h2.688l-.011-.013-2.914-2.914-.013-.011Z"></path></svg></div><span class="PRIVATE_TreeView-item-content-text prc-TreeView-TreeViewItemContentText-FFaKp"><span>e2_5.xml</span></span></div></div></li><li class="PRIVATE_TreeView-item prc-TreeView-TreeViewItem-Ter5f" tabindex="-1" id="e2_6.xml-item" role="treeitem" aria-labelledby="_r_7g_" aria-describedby="_r_7h_" aria-level="1" aria-selected="false"><div class="PRIVATE_TreeView-item-container prc-TreeView-TreeViewItemContainer-z6qqQ" style="--level: 1; content-visibility: auto; contain-intrinsic-size: auto 2rem;"><div style="grid-area: spacer; display: flex;"><div style="width: 100%; display: flex;"></div></div><div id="_r_7g_" class="PRIVATE_TreeView-item-content prc-TreeView-TreeViewItemContent-RKsCI"><div class="PRIVATE_VisuallyHidden prc-TreeView-TreeViewVisuallyHidden-1N8xK" aria-hidden="true" id="_r_7h_"></div><div class="PRIVATE_TreeView-item-visual prc-TreeView-TreeViewItemVisual-naWzj" aria-hidden="true"><svg aria-hidden="true" focusable="false" class="octicon octicon-file" viewBox="0 0 16 16" width="16" height="16" fill="currentColor" display="inline-block" overflow="visible" style="vertical-align: text-bottom;"><path d="M2 1.75C2 .784 2.784 0 3.75 0h6.586c.464 0 .909.184 1.237.513l2.914 2.914c.329.328.513.773.513 1.237v9.586A1.75 1.75 0 0 1 13.25 16h-9.5A1.75 1.75 0 0 1 2 14.25Zm1.75-.25a.25.25 0 0 0-.25.25v12.5c0 .138.112.25.25.25h9.5a.25.25 0 0 0 .25-.25V6h-2.75A1.75 1.75 0 0 1 9 4.25V1.5Zm6.75.062V4.25c0 .138.112.25.25.25h2.688l-.011-.013-2.914-2.914-.013-.011Z"></path></svg></div><span class="PRIVATE_TreeView-item-content-text prc-TreeView-TreeViewItemContentText-FFaKp"><span>e2_6.xml</span></span></div></div></li><li class="PRIVATE_TreeView-item prc-TreeView-TreeViewItem-Ter5f" tabindex="-1" id="e2_7.xml-item" role="treeitem" aria-labelledby="_r_7k_" aria-describedby="_r_7l_" aria-level="1" aria-selected="false"><div class="PRIVATE_TreeView-item-container prc-TreeView-TreeViewItemContainer-z6qqQ" style="--level: 1; content-visibility: auto; contain-intrinsic-size: auto 2rem;"><div style="grid-area: spacer; display: flex;"><div style="width: 100%; display: flex;"></div></div><div id="_r_7k_" class="PRIVATE_TreeView-item-content prc-TreeView-TreeViewItemContent-RKsCI"><div class="PRIVATE_VisuallyHidden prc-TreeView-TreeViewVisuallyHidden-1N8xK" aria-hidden="true" id="_r_7l_"></div><div class="PRIVATE_TreeView-item-visual prc-TreeView-TreeViewItemVisual-naWzj" aria-hidden="true"><svg aria-hidden="true" focusable="false" class="octicon octicon-file" viewBox="0 0 16 16" width="16" height="16" fill="currentColor" display="inline-block" overflow="visible" style="vertical-align: text-bottom;"><path d="M2 1.75C2 .784 2.784 0 3.75 0h6.586c.464 0 .909.184 1.237.513l2.914 2.914c.329.328.513.773.513 1.237v9.586A1.75 1.75 0 0 1 13.25 16h-9.5A1.75 1.75 0 0 1 2 14.25Zm1.75-.25a.25.25 0 0 0-.25.25v12.5c0 .138.112.25.25.25h9.5a.25.25 0 0 0 .25-.25V6h-2.75A1.75 1.75 0 0 1 9 4.25V1.5Zm6.75.062V4.25c0 .138.112.25.25.25h2.688l-.011-.013-2.914-2.914-.013-.011Z"></path></svg></div><span class="PRIVATE_TreeView-item-content-text prc-TreeView-TreeViewItemContentText-FFaKp"><span>e2_7.xml</span></span></div></div></li><li class="PRIVATE_TreeView-item prc-TreeView-TreeViewItem-Ter5f" tabindex="-1" id="e2_8.xml-item" role="treeitem" aria-labelledby="_r_7o_" aria-describedby="_r_7p_" aria-level="1" aria-selected="false"><div class="PRIVATE_TreeView-item-container prc-TreeView-TreeViewItemContainer-z6qqQ" style="--level: 1; content-visibility: auto; contain-intrinsic-size: auto 2rem;"><div style="grid-area: spacer; display: flex;"><div style="width: 100%; display: flex;"></div></div><div id="_r_7o_" class="PRIVATE_TreeView-item-content prc-TreeView-TreeViewItemContent-RKsCI"><div class="PRIVATE_VisuallyHidden prc-TreeView-TreeViewVisuallyHidden-1N8xK" aria-hidden="true" id="_r_7p_"></div><div class="PRIVATE_TreeView-item-visual prc-TreeView-TreeViewItemVisual-naWzj" aria-hidden="true"><svg aria-hidden="true" focusable="false" class="octicon octicon-file" viewBox="0 0 16 16" width="16" height="16" fill="currentColor" display="inline-block" overflow="visible" style="vertical-align: text-bottom;"><path d="M2 1.75C2 .784 2.784 0 3.75 0h6.586c.464 0 .909.184 1.237.513l2.914 2.914c.329.328.513.773.513 1.237v9.586A1.75 1.75 0 0 1 13.25 16h-9.5A1.75 1.75 0 0 1 2 14.25Zm1.75-.25a.25.25 0 0 0-.25.25v12.5c0 .138.112.25.25.25h9.5a.25.25 0 0 0 .25-.25V6h-2.75A1.75 1.75 0 0 1 9 4.25V1.5Zm6.75.062V4.25c0 .138.112.25.25.25h2.688l-.011-.013-2.914-2.914-.013-.011Z"></path></svg></div><span class="PRIVATE_TreeView-item-content-text prc-TreeView-TreeViewItemContentText-FFaKp"><span>e2_8.xml</span></span></div></div></li><li class="PRIVATE_TreeView-item prc-TreeView-TreeViewItem-Ter5f" tabindex="-1" id="e2_9.xml-item" role="treeitem" aria-labelledby="_r_7s_" aria-describedby="_r_7t_" aria-level="1" aria-selected="false"><div class="PRIVATE_TreeView-item-container prc-TreeView-TreeViewItemContainer-z6qqQ" style="--level: 1; content-visibility: auto; contain-intrinsic-size: auto 2rem;"><div style="grid-area: spacer; display: flex;"><div style="width: 100%; display: flex;"></div></div><div id="_r_7s_" class="PRIVATE_TreeView-item-content prc-TreeView-TreeViewItemContent-RKsCI"><div class="PRIVATE_VisuallyHidden prc-TreeView-TreeViewVisuallyHidden-1N8xK" aria-hidden="true" id="_r_7t_"></div><div class="PRIVATE_TreeView-item-visual prc-TreeView-TreeViewItemVisual-naWzj" aria-hidden="true"><svg aria-hidden="true" focusable="false" class="octicon octicon-file" viewBox="0 0 16 16" width="16" height="16" fill="currentColor" display="inline-block" overflow="visible" style="vertical-align: text-bottom;"><path d="M2 1.75C2 .784 2.784 0 3.75 0h6.586c.464 0 .909.184 1.237.513l2.914 2.914c.329.328.513.773.513 1.237v9.586A1.75 1.75 0 0 1 13.25 16h-9.5A1.75 1.75 0 0 1 2 14.25Zm1.75-.25a.25.25 0 0 0-.25.25v12.5c0 .138.112.25.25.25h9.5a.25.25 0 0 0 .25-.25V6h-2.75A1.75 1.75 0 0 1 9 4.25V1.5Zm6.75.062V4.25c0 .138.112.25.25.25h2.688l-.011-.013-2.914-2.914-.013-.011Z"></path></svg></div><span class="PRIVATE_TreeView-item-content-text prc-TreeView-TreeViewItemContentText-FFaKp"><span>e2_9.xml</span></span></div></div></li><li class="PRIVATE_TreeView-item prc-TreeView-TreeViewItem-Ter5f" tabindex="-1" id="igrl_gat_model.h5-item" role="treeitem" aria-labelledby="_r_80_" aria-describedby="_r_81_" aria-level="1" aria-selected="false"><div class="PRIVATE_TreeView-item-container prc-TreeView-TreeViewItemContainer-z6qqQ" style="--level: 1; content-visibility: auto; contain-intrinsic-size: auto 2rem;"><div style="grid-area: spacer; display: flex;"><div style="width: 100%; display: flex;"></div></div><div id="_r_80_" class="PRIVATE_TreeView-item-content prc-TreeView-TreeViewItemContent-RKsCI"><div class="PRIVATE_VisuallyHidden prc-TreeView-TreeViewVisuallyHidden-1N8xK" aria-hidden="true" id="_r_81_"></div><div class="PRIVATE_TreeView-item-visual prc-TreeView-TreeViewItemVisual-naWzj" aria-hidden="true"><svg aria-hidden="true" focusable="false" class="octicon octicon-file" viewBox="0 0 16 16" width="16" height="16" fill="currentColor" display="inline-block" overflow="visible" style="vertical-align: text-bottom;"><path d="M2 1.75C2 .784 2.784 0 3.75 0h6.586c.464 0 .909.184 1.237.513l2.914 2.914c.329.328.513.773.513 1.237v9.586A1.75 1.75 0 0 1 13.25 16h-9.5A1.75 1.75 0 0 1 2 14.25Zm1.75-.25a.25.25 0 0 0-.25.25v12.5c0 .138.112.25.25.25h9.5a.25.25 0 0 0 .25-.25V6h-2.75A1.75 1.75 0 0 1 9 4.25V1.5Zm6.75.062V4.25c0 .138.112.25.25.25h2.688l-.011-.013-2.914-2.914-.013-.011Z"></path></svg></div><span class="PRIVATE_TreeView-item-content-text prc-TreeView-TreeViewItemContentText-FFaKp"><span>igrl_gat_model.h5</span></span></div></div></li><li class="PRIVATE_TreeView-item prc-TreeView-TreeViewItem-Ter5f" tabindex="-1" id="igrl_gcn_model.h5-item" role="treeitem" aria-labelledby="_r_84_" aria-describedby="_r_85_" aria-level="1" aria-selected="false"><div class="PRIVATE_TreeView-item-container prc-TreeView-TreeViewItemContainer-z6qqQ" style="--level: 1; content-visibility: auto; contain-intrinsic-size: auto 2rem;"><div style="grid-area: spacer; display: flex;"><div style="width: 100%; display: flex;"></div></div><div id="_r_84_" class="PRIVATE_TreeView-item-content prc-TreeView-TreeViewItemContent-RKsCI"><div class="PRIVATE_VisuallyHidden prc-TreeView-TreeViewVisuallyHidden-1N8xK" aria-hidden="true" id="_r_85_"></div><div class="PRIVATE_TreeView-item-visual prc-TreeView-TreeViewItemVisual-naWzj" aria-hidden="true"><svg aria-hidden="true" focusable="false" class="octicon octicon-file" viewBox="0 0 16 16" width="16" height="16" fill="currentColor" display="inline-block" overflow="visible" style="vertical-align: text-bottom;"><path d="M2 1.75C2 .784 2.784 0 3.75 0h6.586c.464 0 .909.184 1.237.513l2.914 2.914c.329.328.513.773.513 1.237v9.586A1.75 1.75 0 0 1 13.25 16h-9.5A1.75 1.75 0 0 1 2 14.25Zm1.75-.25a.25.25 0 0 0-.25.25v12.5c0 .138.112.25.25.25h9.5a.25.25 0 0 0 .25-.25V6h-2.75A1.75 1.75 0 0 1 9 4.25V1.5Zm6.75.062V4.25c0 .138.112.25.25.25h2.688l-.011-.013-2.914-2.914-.013-.011Z"></path></svg></div><span class="PRIVATE_TreeView-item-content-text prc-TreeView-TreeViewItemContentText-FFaKp"><span>igrl_gcn_model.h5</span></span></div></div></li><li class="PRIVATE_TreeView-item prc-TreeView-TreeViewItem-Ter5f" tabindex="-1" id="simulation_results.png-item" role="treeitem" aria-labelledby="_r_88_" aria-describedby="_r_89_" aria-level="1" aria-selected="false"><div class="PRIVATE_TreeView-item-container prc-TreeView-TreeViewItemContainer-z6qqQ" style="--level: 1; content-visibility: auto; contain-intrinsic-size: auto 2rem;"><div style="grid-area: spacer; display: flex;"><div style="width: 100%; display: flex;"></div></div><div id="_r_88_" class="PRIVATE_TreeView-item-content prc-TreeView-TreeViewItemContent-RKsCI"><div class="PRIVATE_VisuallyHidden prc-TreeView-TreeViewVisuallyHidden-1N8xK" aria-hidden="true" id="_r_89_"></div><div class="PRIVATE_TreeView-item-visual prc-TreeView-TreeViewItemVisual-naWzj" aria-hidden="true"><svg aria-hidden="true" focusable="false" class="octicon octicon-file" viewBox="0 0 16 16" width="16" height="16" fill="currentColor" display="inline-block" overflow="visible" style="vertical-align: text-bottom;"><path d="M2 1.75C2 .784 2.784 0 3.75 0h6.586c.464 0 .909.184 1.237.513l2.914 2.914c.329.328.513.773.513 1.237v9.586A1.75 1.75 0 0 1 13.25 16h-9.5A1.75 1.75 0 0 1 2 14.25Zm1.75-.25a.25.25 0 0 0-.25.25v12.5c0 .138.112.25.25.25h9.5a.25.25 0 0 0 .25-.25V6h-2.75A1.75 1.75 0 0 1 9 4.25V1.5Zm6.75.062V4.25c0 .138.112.25.25.25h2.688l-.011-.013-2.914-2.914-.013-.011Z"></path></svg></div><span class="PRIVATE_TreeView-item-content-text prc-TreeView-TreeViewItemContentText-FFaKp"><span>simulation_results.png</span></span></div></div></li></ul></nav></div></div></div></div></div></div></div><div class="prc-PageLayout-VerticalDivider-9QRmK prc-PageLayout-PaneVerticalDivider-le57g" data-variant-narrow="none" data-variant-regular="line" data-variant-wide="line" data-position="start" style="--spacing: var(--spacing-none);"><div class="prc-PageLayout-DraggableHandle-9s6B4" role="slider" aria-label="Draggable pane splitter" aria-valuemin="256" aria-valuemax="567" aria-valuenow="320" aria-valuetext="Pane width 320 pixels" tabindex="0"></div></div></div></div><div class="prc-PageLayout-ContentWrapper-gR9eG" data-is-hidden-narrow="true"><div class="prc-PageLayout-Content-xWL-A" data-width="full" style="--spacing: var(--spacing-none);"><div class="SharedPageLayout-module__content__IwGAp" data-selector="repos-split-pane-content" tabindex="0">  <div class="container CodeViewHeader-module__Box__JkPOb"><div class="CodeViewHeader-module__StickyHeader__Qn7UN" id="StickyHeader"><div class="CodeViewHeader-module__Box_1__SbNDV"><div class="CodeViewHeader-module__Box_2__TB46f"><div class="CodeViewHeader-module__Box_6__qKUtX"><div class="Breadcrumb-module__container__Vxvev Breadcrumb-module__lg__Rjz0A"><nav data-testid="breadcrumbs" aria-labelledby="repos-header-breadcrumb--wide-heading" id="repos-header-breadcrumb--wide" class="Breadcrumb-module__nav__rQFDj"><h2 class="sr-only ScreenReaderHeading-module__userSelectNone__rwWIk prc-Heading-Heading-MtWFE" data-testid="screen-reader-heading" id="repos-header-breadcrumb--wide-heading">Breadcrumbs</h2><ol class="Breadcrumb-module__list__ZH6zr"><li class="Breadcrumb-module__listItem__Ib0x_"><a class="Breadcrumb-module__repoLink__O2Nbs prc-Link-Link-9ZwDx" data-testid="breadcrumbs-repo-link" href="https://github.com/manishvat/gat-traffic-control/tree/main" data-discover="true">gat-traffic-control</a></li></ol></nav><div data-testid="breadcrumbs-filename" class="Breadcrumb-module__filename__equZR"><span class="Breadcrumb-module__separator__eNwsI Breadcrumb-module__lg__Rjz0A" aria-hidden="true">/</span><h1 class="Breadcrumb-module__filenameHeading__MNMtw Breadcrumb-module__lg__Rjz0A prc-Heading-Heading-MtWFE" tabindex="-1" id="file-name-id-wide">4_GAT.py</h1></div><button data-component="IconButton" type="button" class="prc-Button-ButtonBase-9n-Xk ml-2 prc-Button-IconButton-fyge7" data-loading="false" data-no-visuals="true" data-size="small" data-variant="invisible" aria-labelledby="_r_9s_"><svg aria-hidden="true" focusable="false" class="octicon octicon-copy" viewBox="0 0 16 16" width="16" height="16" fill="currentColor" display="inline-block" overflow="visible" style="vertical-align: text-bottom;"><path d="M0 6.75C0 5.784.784 5 1.75 5h1.5a.75.75 0 0 1 0 1.5h-1.5a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-1.5a.75.75 0 0 1 1.5 0v1.5A1.75 1.75 0 0 1 9.25 16h-7.5A1.75 1.75 0 0 1 0 14.25Z"></path><path d="M5 1.75C5 .784 5.784 0 6.75 0h7.5C15.216 0 16 .784 16 1.75v7.5A1.75 1.75 0 0 1 14.25 11h-7.5A1.75 1.75 0 0 1 5 9.25Zm1.75-.25a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-7.5a.25.25 0 0 0-.25-.25Z"></path></svg></button><span class="CopyToClipboardButton-module__tooltip__BhMvU prc-TooltipV2-Tooltip-tLeuB" data-direction="nw" aria-label="Copy path" aria-hidden="true" id="_r_9s_" popover="auto">Copy path</span></div></div><div class="react-code-view-header-element--wide"><div class="CodeViewHeader-module__Box_7___0R6c"><div class="d-flex gap-2"><button hidden="" data-testid="" data-hotkey="l,Shift+L" data-hotkey-scope="read-only-cursor-text-area"></button><button hidden="" data-hotkey="l,Shift+L"></button><button hidden="" data-testid="" data-hotkey="Mod+Alt+g" data-hotkey-scope="read-only-cursor-text-area"></button><button hidden="" data-hotkey="Mod+Alt+g"></button><button type="button" data-hotkey="b,Shift+B,Control+/ Control+b" class="prc-Button-ButtonBase-9n-Xk NavigationMenu-module__Button__LpKgm" data-loading="false" data-no-visuals="true" data-size="medium" data-variant="default" style="display: none;"><span data-component="buttonContent" data-align="center" class="prc-Button-ButtonContent-Iohp5"><span data-component="text" class="prc-Button-Label-FWkx3">Blame</span></span></button><button hidden="" data-testid="" data-hotkey="b,Shift+B,Control+/ Control+b" data-hotkey-scope="read-only-cursor-text-area"></button><button data-component="IconButton" type="button" data-testid="more-file-actions-button-nav-menu-wide" aria-haspopup="true" aria-expanded="false" tabindex="0" class="prc-Button-ButtonBase-9n-Xk js-blob-dropdown-click NavigationMenu-module__IconButton__HpX3G prc-Button-IconButton-fyge7" data-loading="false" data-no-visuals="true" data-size="medium" data-variant="default" aria-labelledby="_r_8u_" id="_r_8s_"><svg aria-hidden="true" focusable="false" class="octicon octicon-kebab-horizontal" viewBox="0 0 16 16" width="16" height="16" fill="currentColor" display="inline-block" overflow="visible" style="vertical-align: text-bottom;"><path d="M8 9a1.5 1.5 0 1 0 0-3 1.5 1.5 0 0 0 0 3ZM1.5 9a1.5 1.5 0 1 0 0-3 1.5 1.5 0 0 0 0 3Zm13 0a1.5 1.5 0 1 0 0-3 1.5 1.5 0 0 0 0 3Z"></path></svg></button><span class="prc-TooltipV2-Tooltip-tLeuB" data-direction="nw" aria-hidden="true" id="_r_8u_" popover="auto">More file actions</span></div></div></div><div class="react-code-view-header-element--narrow"><div class="CodeViewHeader-module__Box_7___0R6c"><div class="d-flex gap-2"><button hidden="" data-testid="" data-hotkey="l,Shift+L" data-hotkey-scope="read-only-cursor-text-area"></button><button hidden="" data-hotkey="l,Shift+L"></button><button hidden="" data-testid="" data-hotkey="Mod+Alt+g" data-hotkey-scope="read-only-cursor-text-area"></button><button hidden="" data-hotkey="Mod+Alt+g"></button><button type="button" data-hotkey="b,Shift+B,Control+/ Control+b" class="prc-Button-ButtonBase-9n-Xk NavigationMenu-module__Button__LpKgm" data-loading="false" data-no-visuals="true" data-size="medium" data-variant="default" style="display: none;"><span data-component="buttonContent" data-align="center" class="prc-Button-ButtonContent-Iohp5"><span data-component="text" class="prc-Button-Label-FWkx3">Blame</span></span></button><button hidden="" data-testid="" data-hotkey="b,Shift+B,Control+/ Control+b" data-hotkey-scope="read-only-cursor-text-area"></button><button data-component="IconButton" type="button" data-testid="more-file-actions-button-nav-menu-narrow" aria-haspopup="true" aria-expanded="false" tabindex="0" class="prc-Button-ButtonBase-9n-Xk js-blob-dropdown-click NavigationMenu-module__IconButton__HpX3G prc-Button-IconButton-fyge7" data-loading="false" data-no-visuals="true" data-size="medium" data-variant="default" aria-labelledby="_r_93_" id="_r_91_"><svg aria-hidden="true" focusable="false" class="octicon octicon-kebab-horizontal" viewBox="0 0 16 16" width="16" height="16" fill="currentColor" display="inline-block" overflow="visible" style="vertical-align: text-bottom;"><path d="M8 9a1.5 1.5 0 1 0 0-3 1.5 1.5 0 0 0 0 3ZM1.5 9a1.5 1.5 0 1 0 0-3 1.5 1.5 0 0 0 0 3Zm13 0a1.5 1.5 0 1 0 0-3 1.5 1.5 0 0 0 0 3Z"></path></svg></button><span class="prc-TooltipV2-Tooltip-tLeuB" data-direction="nw" aria-hidden="true" id="_r_93_" popover="auto">More file actions</span></div></div></div></div></div></div></div><div class="CodeView-module__contentWrapper__cG2JH"><div class="react-code-view-bottom-padding"><div class="BlobTopBanners-module__Box__v_nvx"></div></div> <button hidden="" data-testid="" data-hotkey="r,Shift+R" data-hotkey-scope="read-only-cursor-text-area"></button><button hidden="" data-hotkey="r,Shift+R"></button><div class="d-flex flex-column border rounded-2 tmp-mb-3 pl-1"><div class="LatestCommit-module__Box__B25ZT"><h2 class="sr-only ScreenReaderHeading-module__userSelectNone__rwWIk prc-Heading-Heading-MtWFE" data-testid="screen-reader-heading">Latest commit</h2><div data-testid="latest-commit" class="LatestCommit-module__Box_1__YkEgg"><div class="CommitAttribution-module__CommitAttributionContainer__I_rfs"><div data-testid="author-avatar" class="Box-sc-62in7e-0 AuthorAvatar-module__AuthorAvatarContainer__n0MVc"><a class="Link__StyledLink-sc-1syctfj-0 prc-Link-Link-9ZwDx" href="https://github.com/manishvat" data-testid="avatar-icon-link" data-hovercard-url="/users/manishvat/hovercard" data-hovercard-type="user" octo-click="hovercard-link-click" octo-dimensions="link_type:self" aria-keyshortcuts="Alt+ArrowUp"><img data-component="Avatar" class="AuthorAvatar-module__authorAvatarImage__a3R8x prc-Avatar-Avatar-0xaUi" alt="manishvat" width="20" height="20" data-testid="github-avatar" aria-label="manishvat" src="./4_GAT_files/156544948" style="--avatarSize-regular: 20px;"></a><a class="Link__StyledLink-sc-1syctfj-0 dtKDuy AuthorAvatar-module__authorHoverableLink__MHTT8 prc-Link-Link-9ZwDx" data-muted="true" href="https://github.com/manishvat/gat-traffic-control/commits?author=manishvat" aria-label="commits by manishvat" data-hovercard-url="/users/manishvat/hovercard" data-hovercard-type="user" octo-click="hovercard-link-click" octo-dimensions="link_type:self" aria-keyshortcuts="Alt+ArrowUp">manishvat</a></div><span class=""></span></div><div class="d-none d-sm-flex LatestCommit-module__Box_2__pSPKJ"><div class="Truncate flex-items-center f5"><span class="Truncate-text prc-Text-Text-9mHv3" data-testid="latest-commit-html"><a data-hovercard-url="/manishvat/gat-traffic-control/commit/2fb8bda825961fb36fcb98daa5809ab893924799/hovercard" data-pjax="true" class="Link--secondary" href="https://github.com/manishvat/gat-traffic-control/commit/2fb8bda825961fb36fcb98daa5809ab893924799" aria-keyshortcuts="Alt+ArrowUp">Initial commit - GAT Traffic Control Project</a></span></div></div><span class="d-flex d-sm-none fgColor-muted f6"><relative-time tense="past" datetime="2026-02-28T08:36:37.000Z" title="Feb 28, 2026, 2:06 PM GMT+5:30"><template shadowrootmode="open">2 months ago</template>Feb 28, 2026</relative-time></span></div><div class="d-flex flex-shrink-0 gap-2"><div data-testid="latest-commit-details" class="d-none d-sm-flex flex-items-center"><span class="d-flex flex-nowrap fgColor-muted f6"><a class="Link--secondary prc-Link-Link-9ZwDx" aria-label="Commit 2fb8bda" data-hovercard-url="/manishvat/gat-traffic-control/commit/2fb8bda825961fb36fcb98daa5809ab893924799/hovercard" data-hovercard-type="commit" octo-click="hovercard-link-click" octo-dimensions="link_type:self" aria-keyshortcuts="Alt+ArrowUp" href="https://github.com/manishvat/gat-traffic-control/commit/2fb8bda825961fb36fcb98daa5809ab893924799" data-discover="true">2fb8bda</a>&nbsp;·&nbsp;<relative-time tense="past" datetime="2026-02-28T08:36:37.000Z" title="Feb 28, 2026, 2:06 PM GMT+5:30"><template shadowrootmode="open">2 months ago</template>Feb 28, 2026</relative-time></span></div><div class="d-flex gap-2"><h2 class="sr-only ScreenReaderHeading-module__userSelectNone__rwWIk prc-Heading-Heading-MtWFE" data-testid="screen-reader-heading">History</h2><a href="https://github.com/manishvat/gat-traffic-control/commits/main/4_GAT.py" class="prc-Button-ButtonBase-9n-Xk d-none d-lg-flex LinkButton-module__linkButton__nFnov flex-items-center fgColor-default" data-loading="false" data-size="small" data-variant="invisible"><span data-component="buttonContent" data-align="center" class="prc-Button-ButtonContent-Iohp5"><span data-component="leadingVisual" class="prc-Button-Visual-YNt2F prc-Button-VisualWrap-E4cnq"><svg aria-hidden="true" focusable="false" class="octicon octicon-history" viewBox="0 0 16 16" width="16" height="16" fill="currentColor" display="inline-block" overflow="visible" style="vertical-align: text-bottom;"><path d="m.427 1.927 1.215 1.215a8.002 8.002 0 1 1-1.6 5.685.75.75 0 1 1 1.493-.154 6.5 6.5 0 1 0 1.18-4.458l1.358 1.358A.25.25 0 0 1 3.896 6H.25A.25.25 0 0 1 0 5.75V2.104a.25.25 0 0 1 .427-.177ZM7.75 4a.75.75 0 0 1 .75.75v2.992l2.028.812a.75.75 0 0 1-.557 1.392l-2.5-1A.751.751 0 0 1 7 8.25v-3.5A.75.75 0 0 1 7.75 4Z"></path></svg></span><span data-component="text" class="prc-Button-Label-FWkx3"><span class="fgColor-default">History</span></span></span></a><div class="d-sm-none"><button data-component="IconButton" type="button" aria-pressed="false" aria-expanded="false" data-testid="latest-commit-details-toggle" class="prc-Button-ButtonBase-9n-Xk LatestCommit-module__IconButton__mkJr_ prc-Button-IconButton-fyge7" data-loading="false" data-no-visuals="true" data-size="small" data-variant="invisible" aria-labelledby="_r_96_"><svg aria-hidden="true" focusable="false" class="octicon octicon-ellipsis" viewBox="0 0 16 16" width="16" height="16" fill="currentColor" display="inline-block" overflow="visible" style="vertical-align: text-bottom;"><path d="M0 5.75C0 4.784.784 4 1.75 4h12.5c.966 0 1.75.784 1.75 1.75v4.5A1.75 1.75 0 0 1 14.25 12H1.75A1.75 1.75 0 0 1 0 10.25ZM12 7a1 1 0 1 0 0 2 1 1 0 0 0 0-2ZM7 8a1 1 0 1 0 2 0 1 1 0 0 0-2 0ZM4 7a1 1 0 1 0 0 2 1 1 0 0 0 0-2Z"></path></svg></button><span class="prc-TooltipV2-Tooltip-tLeuB" data-direction="s" aria-hidden="true" id="_r_96_" popover="auto">Open commit details</span></div><div class="d-flex d-lg-none"><a aria-label="View commit history for this file." href="https://github.com/manishvat/gat-traffic-control/commits/main/4_GAT.py" class="prc-Button-ButtonBase-9n-Xk LinkButton-module__linkButton__nFnov flex-items-center fgColor-default" data-loading="false" data-size="small" data-variant="invisible" aria-describedby="_r_98_"><span data-component="buttonContent" data-align="center" class="prc-Button-ButtonContent-Iohp5"><span data-component="leadingVisual" class="prc-Button-Visual-YNt2F prc-Button-VisualWrap-E4cnq"><svg aria-hidden="true" focusable="false" class="octicon octicon-history" viewBox="0 0 16 16" width="16" height="16" fill="currentColor" display="inline-block" overflow="visible" style="vertical-align: text-bottom;"><path d="m.427 1.927 1.215 1.215a8.002 8.002 0 1 1-1.6 5.685.75.75 0 1 1 1.493-.154 6.5 6.5 0 1 0 1.18-4.458l1.358 1.358A.25.25 0 0 1 3.896 6H.25A.25.25 0 0 1 0 5.75V2.104a.25.25 0 0 1 .427-.177ZM7.75 4a.75.75 0 0 1 .75.75v2.992l2.028.812a.75.75 0 0 1-.557 1.392l-2.5-1A.751.751 0 0 1 7 8.25v-3.5A.75.75 0 0 1 7.75 4Z"></path></svg></span></span></a><span class="prc-TooltipV2-Tooltip-tLeuB" data-direction="s" role="tooltip" aria-hidden="true" id="_r_98_" popover="auto">History</span></div></div></div></div></div><div class="d-flex flex-row"><div class="container BlobViewContent-module__blobContainer__DtH2d"><div class="react-code-size-details-banner BlobViewContent-module__codeSizeDetails__e5sUw"><div class="react-code-size-details-banner CodeSizeDetails-module__Box__VcD6l"><div class="text-mono CodeSizeDetails-module__Box_1__GVxQL"><div data-testid="blob-size" class="CodeSizeDetails-module__Truncate_1__lE93V prc-Truncate-Truncate-2G1eo" data-inline="true" title="35.6 KB" style="--truncate-max-width: 100%;"><span>993 lines (842 loc) · 35.6 KB</span></div></div></div></div><div class="react-blob-view-header-sticky BlobViewContent-module__stickyHeader__VwxB5" id="repos-sticky-header"><div class="BlobViewHeader-module__Box__yhm9u"><div class="react-blob-sticky-header"><div class="FileNameStickyHeader-module__outerWrapper__ZL4Xc FileNameStickyHeader-module__outerWrapperHidden__Zpynk"><div class="FileNameStickyHeader-module__Box_5__dmNXA"><div class="Breadcrumb-module__container__Vxvev Breadcrumb-module__md__Wb1Gs"><nav data-testid="breadcrumbs" aria-labelledby="sticky-breadcrumb-heading" id="sticky-breadcrumb" class="Breadcrumb-module__nav__rQFDj"><h2 class="sr-only ScreenReaderHeading-module__userSelectNone__rwWIk prc-Heading-Heading-MtWFE" data-testid="screen-reader-heading" id="sticky-breadcrumb-heading">Breadcrumbs</h2><ol class="Breadcrumb-module__list__ZH6zr"><li class="Breadcrumb-module__listItem__Ib0x_"><a class="Breadcrumb-module__repoLink__O2Nbs prc-Link-Link-9ZwDx" data-testid="breadcrumbs-repo-link" href="https://github.com/manishvat/gat-traffic-control/tree/main" data-discover="true">gat-traffic-control</a></li></ol></nav><div data-testid="breadcrumbs-filename" class="Breadcrumb-module__filename__equZR"><span class="Breadcrumb-module__separator__eNwsI Breadcrumb-module__md__Wb1Gs" aria-hidden="true">/</span><h1 class="Breadcrumb-module__filenameHeading__MNMtw Breadcrumb-module__md__Wb1Gs prc-Heading-Heading-MtWFE" tabindex="-1" id="sticky-file-name-id">4_GAT.py</h1></div></div><button type="button" class="prc-Button-ButtonBase-9n-Xk FileNameStickyHeader-module__Button__LSEU_ FileNameStickyHeader-module__GoToTopButton__nxAFn" data-loading="false" data-size="small" data-variant="invisible"><span data-component="buttonContent" data-align="center" class="prc-Button-ButtonContent-Iohp5"><span data-component="leadingVisual" class="prc-Button-Visual-YNt2F prc-Button-VisualWrap-E4cnq"><svg aria-hidden="true" focusable="false" class="octicon octicon-arrow-up" viewBox="0 0 16 16" width="16" height="16" fill="currentColor" display="inline-block" overflow="visible" style="vertical-align: text-bottom;"><path d="M3.47 7.78a.75.75 0 0 1 0-1.06l4.25-4.25a.75.75 0 0 1 1.06 0l4.25 4.25a.751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018L9 4.81v7.44a.75.75 0 0 1-1.5 0V4.81L4.53 7.78a.75.75 0 0 1-1.06 0Z"></path></svg></span><span data-component="text" class="prc-Button-Label-FWkx3">Top</span></span></button></div></div></div><div class="BlobViewHeader-module__Box_1__VEmuQ"><h2 class="sr-only ScreenReaderHeading-module__userSelectNone__rwWIk prc-Heading-Heading-MtWFE" data-testid="screen-reader-heading">File metadata and controls</h2><div class="BlobViewHeader-module__Box_2__icUs2"><ul aria-label="File view" class="prc-SegmentedControl-SegmentedControl-lqIXp BlobTabButtons-module__SegmentedControl__jen2u" data-variant="default" data-size="small"><li class="prc-SegmentedControl-Item-tSCQh" data-selected=""><button aria-current="true" class="prc-SegmentedControl-Button-E48xz" type="button" data-hotkey="Control+/ Control+c" style="--separator-color: transparent;"><span class="prc-SegmentedControl-Content-1COlk segmentedControl-content"><div class="prc-SegmentedControl-Text-7S2y2 segmentedControl-text" data-text="Code">Code</div></span></button></li><li class="prc-SegmentedControl-Item-tSCQh"><button aria-current="false" class="prc-SegmentedControl-Button-E48xz" type="button" data-hotkey="b,Shift+B,Control+/ Control+b" style="--separator-color: var(--borderColor-default);"><span class="prc-SegmentedControl-Content-1COlk segmentedControl-content"><div class="prc-SegmentedControl-Text-7S2y2 segmentedControl-text" data-text="Blame">Blame</div></span></button></li></ul><button hidden="" data-testid="" data-hotkey="Control+/ Control+c" data-hotkey-scope="read-only-cursor-text-area"></button><button hidden="" data-testid="" data-hotkey="b,Shift+B,Control+/ Control+b" data-hotkey-scope="read-only-cursor-text-area"></button><div class="react-code-size-details-in-header CodeSizeDetails-module__Box__VcD6l"><div class="text-mono CodeSizeDetails-module__Box_1__GVxQL"><div data-testid="blob-size" class="CodeSizeDetails-module__Truncate_1__lE93V prc-Truncate-Truncate-2G1eo" data-inline="true" title="35.6 KB" style="--truncate-max-width: 100%;"><span>993 lines (842 loc) · 35.6 KB</span></div></div></div></div><div class="BlobViewHeader-module__Box_3__ng6v2"><button hidden="" data-testid="" data-hotkey="Control+Shift+&gt;" data-hotkey-scope="read-only-cursor-text-area"></button><button hidden="" data-hotkey="Control+Shift+&gt;"></button><button hidden="" data-testid="" data-hotkey="Control+Shift+&lt;" data-hotkey-scope="read-only-cursor-text-area"></button><button hidden="" data-hotkey="Control+Shift+&lt;"></button><div class="react-blob-header-edit-and-raw-actions BlobViewHeader-module__Box_4__J4Y4W"><div class="prc-ButtonGroup-ButtonGroup-vFUrY"><div><a href="https://github.com/manishvat/gat-traffic-control/raw/refs/heads/main/4_GAT.py" data-testid="raw-button" data-hotkey="Control+/ Control+r" class="prc-Button-ButtonBase-9n-Xk LinkButton-module__linkButton__nFnov BlobViewHeader-module__LinkButton__X9kx2" data-loading="false" data-no-visuals="true" data-size="small" data-variant="default"><span data-component="buttonContent" data-align="center" class="prc-Button-ButtonContent-Iohp5"><span data-component="text" class="prc-Button-Label-FWkx3">Raw</span></span></a></div><div><button data-component="IconButton" type="button" data-testid="copy-raw-button" data-hotkey="Control+Shift+C" class="prc-Button-ButtonBase-9n-Xk prc-Button-IconButton-fyge7" data-loading="false" data-no-visuals="true" data-size="small" data-variant="default" aria-labelledby="_r_9c_"><svg aria-hidden="true" focusable="false" class="octicon octicon-copy" viewBox="0 0 16 16" width="16" height="16" fill="currentColor" display="inline-block" overflow="visible" style="vertical-align: text-bottom;"><path d="M0 6.75C0 5.784.784 5 1.75 5h1.5a.75.75 0 0 1 0 1.5h-1.5a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-1.5a.75.75 0 0 1 1.5 0v1.5A1.75 1.75 0 0 1 9.25 16h-7.5A1.75 1.75 0 0 1 0 14.25Z"></path><path d="M5 1.75C5 .784 5.784 0 6.75 0h7.5C15.216 0 16 .784 16 1.75v7.5A1.75 1.75 0 0 1 14.25 11h-7.5A1.75 1.75 0 0 1 5 9.25Zm1.75-.25a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-7.5a.25.25 0 0 0-.25-.25Z"></path></svg></button><span class="prc-TooltipV2-Tooltip-tLeuB" data-direction="n" aria-hidden="true" id="_r_9c_" popover="auto">Copy raw file</span></div><div><button data-component="IconButton" type="button" data-testid="download-raw-button" data-hotkey="Control+Shift+S" class="prc-Button-ButtonBase-9n-Xk BlobViewHeader-module__downloadButton__ef459 prc-Button-IconButton-fyge7" data-loading="false" data-no-visuals="true" data-size="small" data-variant="default" aria-labelledby="_r_9e_"><svg aria-hidden="true" focusable="false" class="octicon octicon-download" viewBox="0 0 16 16" width="16" height="16" fill="currentColor" display="inline-block" overflow="visible" style="vertical-align: text-bottom;"><path d="M2.75 14A1.75 1.75 0 0 1 1 12.25v-2.5a.75.75 0 0 1 1.5 0v2.5c0 .138.112.25.25.25h10.5a.25.25 0 0 0 .25-.25v-2.5a.75.75 0 0 1 1.5 0v2.5A1.75 1.75 0 0 1 13.25 14Z"></path><path d="M7.25 7.689V2a.75.75 0 0 1 1.5 0v5.689l1.97-1.969a.749.749 0 1 1 1.06 1.06l-3.25 3.25a.749.749 0 0 1-1.06 0L4.22 6.78a.749.749 0 1 1 1.06-1.06l1.97 1.969Z"></path></svg></button><span class="prc-TooltipV2-Tooltip-tLeuB" data-direction="n" aria-hidden="true" id="_r_9e_" popover="auto">Download raw file</span></div></div><button hidden="" data-testid="raw-button-shortcut" data-hotkey="Control+/ Control+r" data-hotkey-scope="read-only-cursor-text-area"></button><button hidden="" data-testid="copy-raw-button-shortcut" data-hotkey="Control+Shift+C" data-hotkey-scope="read-only-cursor-text-area"></button><button hidden="" data-testid="download-raw-button-shortcut" data-hotkey="Control+Shift+S" data-hotkey-scope="read-only-cursor-text-area"></button><div class="prc-ButtonGroup-ButtonGroup-vFUrY"><div><button aria-disabled="true" data-component="IconButton" type="button" class="prc-Button-ButtonBase-9n-Xk prc-Button-IconButton-fyge7" data-loading="false" data-no-visuals="true" data-size="small" data-variant="default" aria-labelledby="_r_9g_"><svg aria-hidden="true" focusable="false" class="octicon octicon-pencil" viewBox="0 0 16 16" width="16" height="16" fill="currentColor" display="inline-block" overflow="visible" style="vertical-align: text-bottom;"><path d="M11.013 1.427a1.75 1.75 0 0 1 2.474 0l1.086 1.086a1.75 1.75 0 0 1 0 2.474l-8.61 8.61c-.21.21-.47.364-.756.445l-3.251.93a.75.75 0 0 1-.927-.928l.929-3.25c.081-.286.235-.547.445-.758l8.61-8.61Zm.176 4.823L9.75 4.81l-6.286 6.287a.253.253 0 0 0-.064.108l-.558 1.953 1.953-.558a.253.253 0 0 0 .108-.064Zm1.238-3.763a.25.25 0 0 0-.354 0L10.811 3.75l1.439 1.44 1.263-1.263a.25.25 0 0 0 0-.354Z"></path></svg></button><span class="prc-TooltipV2-Tooltip-tLeuB" data-direction="s" aria-hidden="true" id="_r_9g_" popover="auto">You must be signed in to make or propose changes</span></div><div><button data-component="IconButton" type="button" data-testid="more-edit-button" aria-haspopup="true" aria-expanded="false" tabindex="0" class="prc-Button-ButtonBase-9n-Xk prc-Button-IconButton-fyge7" data-loading="false" data-no-visuals="true" data-size="small" data-variant="default" aria-labelledby="_r_9k_" id="_r_9i_"><svg aria-hidden="true" focusable="false" class="octicon octicon-triangle-down" viewBox="0 0 16 16" width="16" height="16" fill="currentColor" display="inline-block" overflow="visible" style="vertical-align: text-bottom;"><path d="m4.427 7.427 3.396 3.396a.25.25 0 0 0 .354 0l3.396-3.396A.25.25 0 0 0 11.396 7H4.604a.25.25 0 0 0-.177.427Z"></path></svg></button><span class="prc-TooltipV2-Tooltip-tLeuB" data-direction="nw" aria-hidden="true" id="_r_9k_" popover="auto">More edit options</span></div></div></div><button data-component="IconButton" type="button" aria-pressed="false" aria-expanded="false" aria-controls="symbols-pane" data-hotkey="Control+i" data-testid="symbols-button" class="prc-Button-ButtonBase-9n-Xk BlobViewHeader-module__IconButton_2__RyjZg prc-Button-IconButton-fyge7" data-loading="false" data-no-visuals="true" data-size="small" data-variant="invisible" aria-labelledby="_r_9m_" id="symbols-button"><svg aria-hidden="true" focusable="false" class="octicon octicon-code-square" viewBox="0 0 16 16" width="16" height="16" fill="currentColor" display="inline-block" overflow="visible" style="vertical-align: text-bottom;"><path d="M0 1.75C0 .784.784 0 1.75 0h12.5C15.216 0 16 .784 16 1.75v12.5A1.75 1.75 0 0 1 14.25 16H1.75A1.75 1.75 0 0 1 0 14.25Zm1.75-.25a.25.25 0 0 0-.25.25v12.5c0 .138.112.25.25.25h12.5a.25.25 0 0 0 .25-.25V1.75a.25.25 0 0 0-.25-.25Zm7.47 3.97a.75.75 0 0 1 1.06 0l2 2a.75.75 0 0 1 0 1.06l-2 2a.749.749 0 0 1-1.275-.326.749.749 0 0 1 .215-.734L10.69 8 9.22 6.53a.75.75 0 0 1 0-1.06ZM6.78 6.53 5.31 8l1.47 1.47a.749.749 0 0 1-.326 1.275.749.749 0 0 1-.734-.215l-2-2a.75.75 0 0 1 0-1.06l2-2a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042Z"></path></svg></button><span class="prc-TooltipV2-Tooltip-tLeuB" data-direction="nw" aria-hidden="true" id="_r_9m_" popover="auto">Open symbols panel</span><div class="react-blob-header-edit-and-raw-actions-combined"><button data-component="IconButton" type="button" title="More file actions" data-testid="more-file-actions-button" aria-haspopup="true" aria-expanded="false" tabindex="0" class="prc-Button-ButtonBase-9n-Xk js-blob-dropdown-click BlobViewHeader-module__IconButton__XrMQY prc-Button-IconButton-fyge7" data-loading="false" data-no-visuals="true" data-size="small" data-variant="invisible" aria-labelledby="_r_9q_" id="_r_9o_"><svg aria-hidden="true" focusable="false" class="octicon octicon-kebab-horizontal" viewBox="0 0 16 16" width="16" height="16" fill="currentColor" display="inline-block" overflow="visible" style="vertical-align: text-bottom;"><path d="M8 9a1.5 1.5 0 1 0 0-3 1.5 1.5 0 0 0 0 3ZM1.5 9a1.5 1.5 0 1 0 0-3 1.5 1.5 0 0 0 0 3Zm13 0a1.5 1.5 0 1 0 0-3 1.5 1.5 0 0 0 0 3Z"></path></svg></button><span class="prc-TooltipV2-Tooltip-tLeuB" data-direction="nw" aria-hidden="true" id="_r_9q_" popover="auto">Edit and raw actions</span></div></div></div></div><div></div></div><div class="BlobViewContent-module__blobContentWrapper__JS0W6"><section aria-labelledby="file-name-id-wide file-name-id-mobile" class="BlobContent-module__blobContentSection__VOgZq" style="margin-top: 46px;"><div class="CodeBlob-module__codeBlobWrapper__RS6In" style="padding-top: 8px; padding-bottom: 8px;"><div id="highlighted-line-menu-positioner" class="position-relative"><div id="copilot-button-positioner" class="position-relative"><div class="CodeBlob-module__codeBlobInner__tfjuQ"><div class="CodeBlob-module__cursorContainer__tiLPm"><div style="height: 19820px;"><div aria-hidden="true" data-testid="navigation-cursor" class="code-navigation-cursor" style="top: 11680px; left: 277px;"> </div><button hidden="" data-testid="NavigationCursorEnter" data-hotkey="Control+Enter" data-hotkey-scope="read-only-cursor-text-area"></button><button hidden="" data-testid="NavigationCursorSetHighlightedLine" data-hotkey="Shift+J" data-hotkey-scope="read-only-cursor-text-area"></button><button hidden="" data-testid="NavigationCursorSetHighlightAndExpandMenu" data-hotkey="Alt+Shift+C,Alt+Shift+Ç" data-hotkey-scope="read-only-cursor-text-area"></button><button hidden="" data-testid="NavigationCursorPageDown" data-hotkey="PageDown" data-hotkey-scope="read-only-cursor-text-area"></button><button hidden="" data-testid="NavigationCursorPageUp" data-hotkey="PageUp" data-hotkey-scope="read-only-cursor-text-area"></button><button hidden="" data-testid="" data-hotkey="/" data-hotkey-scope="read-only-cursor-text-area"></button></div></div><textarea id="read-only-cursor-text-area" data-testid="read-only-cursor-text-area" aria-label="file content" aria-readonly="true" inputmode="none" tabindex="0" aria-multiline="true" aria-haspopup="false" data-gramm="false" data-gramm_editor="false" data-enable-grammarly="false" spellcheck="false" autocorrect="off" autocapitalize="off" autocomplete="off" data-ms-editor="false" class="react-blob-textarea react-blob-print-hide" style="resize: none; margin-top: -2px; padding-left: 92px; padding-right: 70px; width: 100%; background-color: unset; box-sizing: border-box; color: transparent; position: absolute; border: none; tab-size: 4; outline: none; overflow: auto hidden; height: 19840px; font-size: 12px; line-height: 20px; overflow-wrap: normal; overscroll-behavior-x: none; white-space: pre; z-index: 1;">"""
IG-RL: Inductive Graph Reinforcement Learning
==============================================
Zero-shot transfer fix:

OLD (broken):
  all 8 nodes → GCN → flatten into 1 vector → single GRU → reshape to 8 nodes
  Problem: the model memorises node positions in the vector. Transfer fails.

NEW (correct):
  all 8 nodes → GCN  (shared weights aggregate neighbourhood info)
               ↓
  pick ONLY training node's embedding  (shape: EMBEDDING_DIM,)
               ↓
  GRU on that single embedding  (temporal memory for this one node)
               ↓
  Dueling Q-head → Q(hold), Q(switch)

At deploy time on ANY node:
  Run the same GCN on that node's local neighbourhood,
  extract that node's embedding, pass through the same GRU + Q-head.
  Zero-shot — no retraining needed.
  The weights never saw node IDs, only queue/neighbour patterns.
"""

import os
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from collections import deque



TRAIN        = True
TRAINING_TLS = "Node2"   # the ONE node we train on; same weights transfer to all others

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

import traci

# ==========================
#   SUMO CONFIG
# ==========================
SUMO_BINARY = "sumo-gui"
SUMO_CONFIG = "New_Network.sumocfg"

Sumo_config = [
    SUMO_BINARY, "-c", SUMO_CONFIG,
    "--step-length", "1", "--delay", "0"
]

# ==========================
#   INTERSECTIONS
# ==========================
INTERSECTIONS = {
    "Node0": {
        "detectors": [
            "Node0_10_SB_0", "Node0_10_SB_1",
            "Node4_0_NB_0",  "Node4_0_NB_1",
            "Node0_1_WB_0",  "Node0_1_WB_1",
            "Node0_16_EB_1", "Node0_16_EB_0",
        ],
        "neighbors": ["Node10", "Node1", "Node16"],
    },
    "Node1": {
        "detectors": [
            "Node0_1_EB_0",  "Node0_1_EB_1",
            "Node1_6_SB_0",  "Node1_6_SB_1",
            "Node3_1_NB_0",  "Node3_1_NB_1",
            "Node1_2_WB_0",  "Node1_2_WB_1",
        ],
        "neighbors": ["Node6", "Node0", "Node2", "Node3"],
    },
    "Node2": {
        "detectors": [
            "Node1_2_EB_0",  "Node1_2_EB_1",
            "Node2_7_SB_0",  "Node2_7_SB_1",
            "Node5_2_NB_0",  "Node5_2_NB_1",
            "Node2_12_WB_0", "Node2_12_WB_1",
        ],
        "neighbors": ["Node1", "Node7", "Node5", "Node12"],
    },
    "Node3": {
        "detectors": [
            "Node4_3_EB_0",  "Node4_3_EB_1",
            "Node3_1_NB_0",  "Node3_1_NB_1",
            "Node3_5_WB_0",  "Node3_5_WB_1",
            "Node3_18_NB_0", "Node3_18_NB_1",
        ],
        "neighbors": ["Node4", "Node1", "Node5", "Node18"],
    },
    "Node4": {
        "detectors": [
            "Node4_0_SB_0",  "Node4_0_SB_1",
            "Node13_4_NB_0", "Node13_4_NB_1",
            "Node4_3_WB_0",  "Node4_3_WB_1",
            "Node4_17_EB_1", "Node4_17_EB_0",
        ],
        "neighbors": ["Node0", "Node13", "Node3", "Node17"],
    },
    "Node5": {
        "detectors": [
            "Node3_5_EB_0",  "Node3_5_EB_1",
            "Node5_2_SB_0",  "Node5_2_SB_1",
            "Node5_11_WB_0", "Node5_11_WB_1",
            "Node5_19_NB_0", "Node5_19_NB_1",
        ],
        "neighbors": ["Node3", "Node2", "Node11", "Node19"],
    },
    "Node6": {
        "detectors": [
            "Node6_15_EB_0",  "Node6_15_EB_1",
            "Node1_6_NB_0",  "Node1_6_NB_1",
            "Node6_7_WB_0",  "Node6_7_WB_1",
            "Node6_9_SB_1",  "Node6_9_SB_0",
        ],
        "neighbors": ["Node9", "Node1", "Node7", "Node15"],
    },
    "Node7": {
        "detectors": [
            "Node6_7_EB_0",  "Node6_7_EB_1",
            "Node2_7_NB_0",  "Node2_7_NB_1",
            "Node7_8_WB_0",  "Node7_8_WB_1",
            "Node7_14_SB_1", "Node7_14_SB_0",
        ],
        "neighbors": ["Node6", "Node8", "Node2", "Node14"],
    },
}


# ==========================
#   RL PARAMETERS
# ==========================
TOTAL_STEPS        = 10000
EPISODE_LENGTH     = 100
LEARNING_RATE      = 0.0001
GAMMA              = 0.95
EPSILON_START      = 1.0
EPSILON_END        = 0.01
EPSILON_DECAY      = 0.9995
BATCH_SIZE         = 32
MEMORY_SIZE        = 50000
TARGET_UPDATE_FREQ = 500
WARMUP_STEPS       = 200
UPDATE_FREQUENCY   = 4
PRINT_INTERVAL     = 100

MIN_GREEN_STEPS             = 10
DEPLOY_MIN_GREEN            = 15
MAX_GREEN_STEPS             = 60
PRESSURE_SWITCH_RATIO       = 1.5
CONGESTION_SWITCH_THRESHOLD = 8
MIN_GREEN_RATIO_HOLD        = 0.6

GCN_LAYERS    = 3
EMBEDDING_DIM = 32
HIDDEN_DIM    = 64
GRU_UNITS     = 32
FEATURE_DIM   = 12
ACTIONS       = [0, 1]

# ==========================
#   INCIDENT PARAMETERS
# ==========================
INCIDENT_SPEED_THRESHOLD     = 2.0
INCIDENT_OCCUPANCY_THRESHOLD = 0.8
NORMAL_LANE_SPEED            = 13.89
INCIDENT_BLOCKED_SPEED       = 0.5

INCIDENT_SCHEDULE = [
    ("Node1_2_EB_0", 150, 300),
    ("Node3_5_WB_0", 400, 550),
]


# ==========================
#   TRAFFIC GRAPH
# ==========================
class TrafficGraph:
    def __init__(self, intersections):
        self.intersections = intersections
        self.tls_ids       = list(intersections.keys())
        self.build_adjacency_matrix()

    def build_adjacency_matrix(self):
        n = len(self.tls_ids)
        self.adj_matrix = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            self.adj_matrix[i, i] = 1.0
        for i, tls_i in enumerate(self.tls_ids):
            for neighbor in self.intersections[tls_i]["neighbors"]:
                if neighbor in self.tls_ids:
                    j = self.tls_ids.index(neighbor)
                    self.adj_matrix[i, j] = 1.0
                    self.adj_matrix[j, i] = 1.0
        self.normalized_adj = self.normalize_adjacency(self.adj_matrix)

    def normalize_adjacency(self, adj):
        adj        = adj + np.eye(adj.shape[0])
        degree     = np.sum(adj, axis=1)
        d_inv_sqrt = np.power(degree, -0.5)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        D = np.diag(d_inv_sqrt)
        return (D @ adj @ D).astype(np.float32)


# ==========================
#   GRAPH CONVOLUTION LAYER
#   SHARED weights — same kernel for every node.
#   This is the transfer mechanism.
# ==========================
# class GraphConvolution(layers.Layer):
#     def __init__(self, units, activation='relu', use_bias=True, **kwargs):
#         super(GraphConvolution, self).__init__(**kwargs)
#         self.units      = units
#         self.activation = keras.activations.get(activation)
#         self.use_bias   = use_bias

#     def build(self, input_shape):
#         feature_dim = input_shape[0][-1]
#         self.kernel  = self.add_weight('kernel', (feature_dim, self.units),
#                                        initializer='glorot_uniform', trainable=True)
#         if self.use_bias:
#             self.bias = self.add_weight('bias', (self.units,),
#                                         initializer='zeros', trainable=True)
#         super(GraphConvolution, self).build(input_shape)

#     def call(self, inputs):
#         features, adjacency = inputs
#         support = tf.matmul(features, self.kernel)
#         output  = tf.matmul(adjacency, support)
#         if self.use_bias:
#             output = tf.nn.bias_add(output, self.bias)
#         return self.activation(output)

# ==========================
#   GRAPH ATTENTION LAYER (GAT)
#   Shared weights → inductive / transfer-friendly
# ==========================
class GraphAttention(layers.Layer):
    def __init__(self, units, num_heads=4, activation='elu',
                 concat=True, attn_dropout=0.1, use_bias=True, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.num_heads = num_heads
        self.activation = keras.activations.get(activation)
        self.concat = concat
        self.attn_dropout = attn_dropout
        self.use_bias = use_bias

        self.leaky_relu = layers.LeakyReLU(alpha=0.2)
        self.drop = layers.Dropout(attn_dropout)

    def build(self, input_shape):
        fin = int(input_shape[0][-1])  # feature dim

        # Linear projection: F -&gt; (heads * units)
        self.W = self.add_weight(
            name="W",
            shape=(fin, self.num_heads * self.units),
            initializer="glorot_uniform",
            trainable=True
        )

        # Attention vectors per head: e_ij = a^T [Wh_i || Wh_j]
        self.a_src = self.add_weight(
            name="a_src",
            shape=(self.num_heads, self.units, 1),
            initializer="glorot_uniform",
            trainable=True
        )
        self.a_dst = self.add_weight(
            name="a_dst",
            shape=(self.num_heads, self.units, 1),
            initializer="glorot_uniform",
            trainable=True
        )

        if self.use_bias:
            out_dim = (self.num_heads * self.units) if self.concat else self.units
            self.bias = self.add_weight(
                name="bias",
                shape=(out_dim,),
                initializer="zeros",
                trainable=True
            )
        else:
            self.bias = None

        super().build(input_shape)

    def call(self, inputs, training=None):
        x, adj = inputs  # x:(B,N,F), adj:(B,N,N) 0/1

        # Add self-loops (stabilizes GAT)
        n = tf.shape(adj)[-1]
        eye = tf.eye(n, batch_shape=[tf.shape(adj)[0]])
        adj = tf.clip_by_value(adj + eye, 0.0, 1.0)

        B = tf.shape(x)[0]
        N = tf.shape(x)[1]

        # Project features
        h = tf.matmul(x, self.W)  # (B,N,H*U)
        h = tf.reshape(h, (B, N, self.num_heads, self.units))  # (B,N,H,U)
        h = tf.transpose(h, (0, 2, 1, 3))  # (B,H,N,U)

        # Attention logits
        f1 = tf.matmul(h, self.a_src)  # (B,H,N,1)
        f2 = tf.matmul(h, self.a_dst)  # (B,H,N,1)

        e = self.leaky_relu(f1 + tf.transpose(f2, (0, 1, 3, 2)))  # (B,H,N,N)

        # Mask non-neighbors
        mask = tf.expand_dims(adj, axis=1)  # (B,1,N,N)
        e = tf.where(mask &gt; 0.0, e, tf.constant(-1e9, dtype=e.dtype))

        # Softmax attention
        alpha = tf.nn.softmax(e, axis=-1)  # (B,H,N,N)
        alpha = self.drop(alpha, training=training)

        # Aggregate
        out = tf.matmul(alpha, h)  # (B,H,N,U)

        # Merge heads
        if self.concat:
            out = tf.transpose(out, (0, 2, 1, 3))  # (B,N,H,U)
            out = tf.reshape(out, (B, N, self.num_heads * self.units))  # (B,N,H*U)
        else:
            out = tf.reduce_mean(out, axis=1)  # (B,N,U)

        if self.bias is not None:
            out = tf.nn.bias_add(out, self.bias)

        return self.activation(out)


# ==========================
#   NOISY DENSE
# ==========================
class NoisyDense(layers.Layer):
    def __init__(self, units, sigma_init=0.017, **kwargs):
        super(NoisyDense, self).__init__(**kwargs)
        self.units      = units
        self.sigma_init = sigma_init

    def build(self, input_shape):
        d = input_shape[-1]
        self.w_mu    = self.add_weight('w_mu',    (d, self.units), initializer='glorot_uniform')
        self.w_sigma = self.add_weight('w_sigma', (d, self.units),
                                       initializer=keras.initializers.Constant(self.sigma_init))
        self.b_mu    = self.add_weight('b_mu',    (self.units,), initializer='zeros')
        self.b_sigma = self.add_weight('b_sigma', (self.units,),
                                       initializer=keras.initializers.Constant(self.sigma_init))

    def call(self, inputs, training=None):
        if training:
            w = self.w_mu + self.w_sigma * tf.random.normal(tf.shape(self.w_mu))
            b = self.b_mu + self.b_sigma * tf.random.normal(tf.shape(self.b_mu))
        else:
            w, b = self.w_mu, self.b_mu
        return tf.matmul(inputs, w) + b


# ================================================================
#   IG-RL MODEL — THE CRITICAL FIX FOR ZERO-SHOT TRANSFER
#
#   OLD broken approach:
#     GCN output (1, N, EMB) → Reshape to (1, 1, N*EMB) → GRU → ...
#     The GRU sees ALL nodes flattened together every step.
#     It learns "node 2 is always in positions 64-96 of this vector".
#     That never transfers to a different network.
#
#   NEW correct approach:
#     GCN output (1, N, EMB) → for each node i: extract (1, 1, EMB)
#                            → GRU (shared weights, applied independently)
#                            → Q-head (shared weights)
#     The GRU + Q-head only ever see ONE node's embedding at a time.
#     They learn "if my neighbourhood looks like this → hold/switch".
#     That transfers to ANY node in ANY network.
# ================================================================
def build_igrl_model(n_nodes, feature_dim, action_size):
    node_features = layers.Input(shape=(n_nodes, feature_dim), name='node_features')
    adjacency     = layers.Input(shape=(n_nodes, n_nodes),     name='adjacency')

    # Step 1: GCN processes all N nodes with SHARED weights
    # Output: (1, N, EMBEDDING_DIM) — each node has a neighbourhood-aware embedding
    x = node_features
    # for i in range(GCN_LAYERS):
    #     x = GraphConvolution(EMBEDDING_DIM, activation='relu', name=f'gcn_{i}')([x, adjacency])
    #     x = layers.BatchNormalization(name=f'bn_{i}')(x)
    #     x = layers.Dropout(0.1, name=f'drop_{i}')(x)
    # ---- GAT settings ----
    GAT_HEADS = 4
    GAT_ATT_DROP = 0.1

    x = node_features
    for i in range(GCN_LAYERS):
        is_last = (i == GCN_LAYERS - 1)

    # Use concat=True for intermediate layers (dim = heads*EMBEDDING_DIM)
    # Use concat=False for last layer (dim = EMBEDDING_DIM) to keep downstream same
        x = GraphAttention(
            units=EMBEDDING_DIM,
            num_heads=GAT_HEADS,
            activation='elu',
            concat=not is_last,
            attn_dropout=GAT_ATT_DROP,
            name=f'gat_{i}'
        )([x, adjacency])

        x = layers.BatchNormalization(name=f'bn_{i}')(x)
        x = layers.Dropout(0.1, name=f'drop_{i}')(x)

    # Step 2: Extract each node's embedding INDEPENDENTLY
    # Each gets shape (1, 1, EMBEDDING_DIM) — a single time-step for the GRU
    node_embeddings = [
        layers.Lambda(lambda t, i=i: t[:, i:i+1, :], name=f'extract_{i}')(x)
        for i in range(n_nodes)
    ]

    # Step 3: ONE shared GRU applied independently to each node's embedding
    # Same weights for every node → inductive, transfers to unseen nodes
    # Each node gets its OWN temporal context, not a mixed one
    gru_layer   = layers.GRU(GRU_UNITS, return_sequences=False, name='gru_shared')
    gru_outputs = [gru_layer(emb) for emb in node_embeddings]
    # gru_outputs[i] shape: (1, GRU_UNITS)

    # Step 4: Shared dueling Q-head applied independently to each node
    # Same weights for every node → transfers zero-shot
    value_dense = layers.Dense(HIDDEN_DIM, activation='relu', name='value_dense')
    value_head  = layers.Dense(1,          name='value_head')
    adv_dense   = layers.Dense(HIDDEN_DIM, activation='relu', name='adv_dense')
    adv_head    = NoisyDense(action_size,  name='adv_head')

    q_per_node = []
    for gru_out in gru_outputs:
        v    = value_head(value_dense(gru_out))               # (1, 1)
        # a    = adv_head(adv_dense(gru_out), training=True)    # (1, action_size)
        a = adv_head(adv_dense(gru_out), training=TRAIN)
        q    = v + (a - tf.reduce_mean(a, axis=-1, keepdims=True))
        q_per_node.append(tf.expand_dims(q, axis=1))          # (1, 1, action_size)

    # Reassemble to (1, N, action_size) — same output shape as before
    q_values = layers.Concatenate(axis=1, name='q_values')(q_per_node)

    model = keras.Model(
        inputs=[node_features, adjacency],
        outputs=q_values,
        name='igrl_per_node_gru'
    )
    model.compile(
        loss=keras.losses.Huber(delta=1.0),
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=1.0)
    )
    return model


# ==========================
#   PRIORITIZED REPLAY BUFFER
#   Stores only TRAINING_TLS transitions (single-agent, paper-faithful)
# ==========================
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity  = capacity
        self.alpha     = alpha
        self.buffer    = []
        self.priorities= []
        self.position  = 0

    def add(self, state, action, reward, next_state):
        max_p = max(self.priorities) if self.priorities else 1.0
        entry = (state, action, reward, next_state)
        if len(self.buffer) &lt; self.capacity:
            self.buffer.append(entry)
            self.priorities.append(max_p)
        else:
            self.buffer[self.position]     = entry
            self.priorities[self.position] = max_p
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == 0:
            return None
        prios = np.array(self.priorities[:len(self.buffer)])
        probs = prios ** self.alpha
        probs /= probs.sum()
        idxs  = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[i] for i in idxs]
        weights = (len(self.buffer) * probs[idxs]) ** (-beta)
        weights /= weights.max()
        return samples, idxs, weights

    def update_priorities(self, indices, td_errors):
        for i, e in zip(indices, td_errors):
            self.priorities[i] = abs(e) + 1e-6

    def size(self):
        return len(self.buffer)


# ==========================
#   INCIDENT FUNCTIONS
# ==========================
def apply_scheduled_incidents(incident_schedule, current_step):
    active = []
    for (lane_id, start_step, end_step) in incident_schedule:
        if start_step &lt;= current_step &lt;= end_step:
            try:
                traci.lane.setMaxSpeed(lane_id, INCIDENT_BLOCKED_SPEED)
                active.append(lane_id)
            except:
                pass
        elif current_step &gt; end_step :
            try:
                traci.lane.setMaxSpeed(lane_id, NORMAL_LANE_SPEED)
            except:
                pass
    return active

def detect_incidents(tls_ids):
    incidents = {tls: 0.0 for tls in tls_ids}
    for tls in tls_ids:
        severities = []
        for det in INTERSECTIONS[tls]["detectors"]:
            try:
                avg_speed  = traci.lanearea.getLastStepMeanSpeed(det)
                occupancy  = traci.lanearea.getLastStepOccupancy(det)
                n_vehicles = traci.lanearea.getLastStepVehicleNumber(det)
                if n_vehicles &gt; 0 and 0 &lt;= avg_speed &lt; INCIDENT_SPEED_THRESHOLD:
                    sf = 1.0 - (avg_speed / INCIDENT_SPEED_THRESHOLD)
                    of = min(occupancy / INCIDENT_OCCUPANCY_THRESHOLD, 1.0)
                    severities.append(min((sf + of) / 2.0, 1.0))
                else:
                    severities.append(0.0)
            except:
                severities.append(0.0)
        incidents[tls] = max(severities) if severities else 0.0
    return incidents


# ==========================
#   HELPERS
# ==========================
def get_queue(det_id):
    try:    return traci.lanearea.getLastStepVehicleNumber(det_id)
    except: return 0

def get_phase(tls):
    return traci.trafficlight.getPhase(tls)

def get_actual_queues_for_tls(tls):
    return [get_queue(det) for det in INTERSECTIONS[tls]["detectors"]]

def get_global_waiting_time():
    try:
        veh_ids = traci.vehicle.getIDList()
        if not veh_ids:
            return 0.0
        total = sum(traci.vehicle.getWaitingTime(v) for v in veh_ids)
        return total / len(veh_ids)
    except:
        return 0.0

def get_vehicle_count():
    try:    return len(traci.vehicle.getIDList())
    except: return 0

def to_batch(node_features, adj_matrix):
    return [np.expand_dims(node_features, 0), np.expand_dims(adj_matrix, 0)]




# ==========================
#   STATE
# ==========================
def get_global_state(graph, last_switch, step, incidents):
    n_nodes    = len(graph.tls_ids)
    node_feats = np.zeros((n_nodes, FEATURE_DIM), dtype=np.float32)
    for i, tls in enumerate(graph.tls_ids):
        queues        = [get_queue(d) for d in INTERSECTIONS[tls]["detectors"]]
        phase         = get_phase(tls)
        time_in_phase = step - last_switch.get(tls, 0)
        total_q       = sum(queues)
        max_q         = max(queues) if max(queues) &gt; 0 else 1
        queues_norm   = ([min(q / max(max_q, 10.0), 1.0) for q in queues] + [0.0]*8)[:8]
        node_feats[i] = queues_norm + [
            phase / 10.0,
            min(time_in_phase / MAX_GREEN_STEPS, 1.0),
            min(total_q / 50.0, 1.0),
            incidents.get(tls, 0.0)
        ]
    return node_feats






# ==========================
#   REWARD — local to training node only
# ==========================
def get_training_node_reward(tls, prev_queue_sum, incidents):
    current_sum = sum(get_actual_queues_for_tls(tls))
    delta       = prev_queue_sum - current_sum
    penalty     = -0.1 * current_sum
    inc_penalty = -5.0 * incidents.get(tls, 0.0) if incidents.get(tls, 0.0) &gt; 0.3 else 0.0
    return delta + penalty + inc_penalty


# ==========================
#   PRESSURE HEURISTIC
#   Used for non-training nodes during training
#   (keeps them running sensibly so training node gets a clean signal)
# ==========================
def pressure_heuristic(tls):
    queues  = get_actual_queues_for_tls(tls)
    green_q = sum(queues[0:4])
    red_q   = sum(queues[4:8])
    if red_q &gt;= CONGESTION_SWITCH_THRESHOLD:   return 1
    if red_q &gt; green_q * PRESSURE_SWITCH_RATIO: return 1
    if green_q &gt;= CONGESTION_SWITCH_THRESHOLD * MIN_GREEN_RATIO_HOLD: return 0
    return 0


# ==========================
#   ACTION SELECTION
#
#   TRAIN=True:
#     Training node  → Q-network + epsilon-greedy
#     All other nodes → pressure heuristic
#       (they are not learning; heuristic keeps them sensible
#        so the training node receives a meaningful signal)
#
#   TRAIN=False (deploy / zero-shot transfer):
#     ALL nodes → Q-network
#       (same weights, every node extracts its own embedding
#        from the GCN output → zero-shot transfer)
# ==========================
def select_action(q_values_all, tls_idx, tls, current_step,
                  last_switch, epsilon, is_training_node):
    time_in_phase = current_step - last_switch.get(tls, 0)
    effective_min = MIN_GREEN_STEPS if TRAIN else DEPLOY_MIN_GREEN

    if time_in_phase &lt; effective_min: return 0
    if time_in_phase &gt;= MAX_GREEN_STEPS: return 1

    # Non-training nodes use heuristic ONLY during training
    if TRAIN and not is_training_node:
        return pressure_heuristic(tls)

    # Epsilon-greedy for training node
    if TRAIN and random.random() &lt; epsilon:
        return random.choice(ACTIONS)

    # Q-network
    q_vals    = q_values_all[0, tls_idx, :]
    q_spread  = float(np.max(q_vals) - np.min(q_vals))
    threshold = 0.05 if TRAIN else 0.10

    if q_spread &gt;= threshold:
        q_action = int(np.argmax(q_vals))
        # Deploy safety gate
        if not TRAIN and q_action == 1:
            queues  = get_actual_queues_for_tls(tls)
            red_q   = sum(queues[4:8])
            green_q = sum(queues[0:4])
            if red_q &lt; 2 and green_q &lt; 2: return 0
            if red_q &lt; green_q * PRESSURE_SWITCH_RATIO and \
               red_q &lt; CONGESTION_SWITCH_THRESHOLD:    return 0
        return q_action

    return pressure_heuristic(tls)


# ==========================
#   APPLY ACTION
# ==========================
def apply_action(tls, action, current_step, last_switch):
    time_in_phase = current_step - last_switch.get(tls, 0)
    effective_min = MIN_GREEN_STEPS if TRAIN else DEPLOY_MIN_GREEN

    if time_in_phase &lt; effective_min:  return last_switch[tls]
    if time_in_phase &gt;= MAX_GREEN_STEPS: action = 1

    if action == 1:
        logic         = traci.trafficlight.getAllProgramLogics(tls)[0]
        n_phases      = len(logic.phases)
        current_phase = get_phase(tls)
        next_phase    = (current_phase + 1) % n_phases
        for _ in range(n_phases):
            ps = logic.phases[next_phase].state
            if 'y' not in ps.lower() and \
               ps.replace('r','').replace('R','').replace('s','').replace('S','') != '':
                break
            next_phase = (next_phase + 1) % n_phases
        traci.trafficlight.setPhase(tls, next_phase)
        return current_step
    return last_switch[tls]


# ==========================
#   TRAINING
#   Only TRAINING_TLS transitions stored in buffer.
#   Loss updates shared GCN + GRU + Q-head weights.
#   Only training node's Q-value is updated in the target — others unchanged.
# ==========================
def train_igrl(main_model, target_model, replay_buffer,
               graph, beta, training_node_idx):
    if replay_buffer.size() &lt; BATCH_SIZE * 2: return 0.0
    result = replay_buffer.sample(BATCH_SIZE, beta)
    if result is None: return 0.0

    samples, indices, importance_weights = result

    batch_states      = np.array([s[0] for s in samples])
    batch_next_states = np.array([s[3] for s in samples])
    batch_actions     = np.array([s[1] for s in samples])
    batch_rewards     = np.array([s[2] for s in samples])

    adj_batch = np.tile(graph.normalized_adj, (BATCH_SIZE, 1, 1))

    adj_batch_tf    = tf.constant(adj_batch,       dtype=tf.float32)
    states_tf       = tf.constant(batch_states,    dtype=tf.float32)
    next_states_tf  = tf.constant(batch_next_states, dtype=tf.float32)

    current_q     = main_model([states_tf,      adj_batch_tf], training=False).numpy()
    next_q_main   = main_model([next_states_tf, adj_batch_tf], training=False).numpy()
    next_q_target = target_model([next_states_tf, adj_batch_tf], training=False).numpy()

    target_q  = current_q.copy()
    td_errors = []

    for b in range(BATCH_SIZE):
        action    = batch_actions[b]
        reward    = batch_rewards[b] / 10.0
        best_next = np.argmax(next_q_main[b, training_node_idx, :])
        td_target = reward + GAMMA * next_q_target[b, training_node_idx, best_next]
        td_error  = td_target - current_q[b, training_node_idx, action]
        td_errors.append(td_error)
        # Only update training node's target — all other nodes left unchanged
        target_q[b, training_node_idx, action] = td_target

    replay_buffer.update_priorities(indices, [np.mean(np.abs(td_errors))] * len(indices))
    loss = main_model.train_on_batch([states_tf, adj_batch_tf], target_q,
                                  sample_weight=importance_weights)
    return loss


# ==========================
#   INITIALIZE
# ==========================
traffic_graph     = TrafficGraph(INTERSECTIONS)
n_nodes           = len(traffic_graph.tls_ids)
action_size       = len(ACTIONS)
training_node_idx = traffic_graph.tls_ids.index(TRAINING_TLS)

print(f"\n=== IG-RL Zero-Shot Transfer ===")
print(f"All nodes   : {traffic_graph.tls_ids}")
print(f"Train on    : {TRAINING_TLS} (index {training_node_idx})")
print(f"Deploy mode : all nodes use same weights (zero-shot transfer)\n")

if TRAIN:
    main_model   = build_igrl_model(n_nodes, FEATURE_DIM, action_size)
    target_model = build_igrl_model(n_nodes, FEATURE_DIM, action_size)
    target_model.set_weights(main_model.get_weights())
    replay_buffer  = PrioritizedReplayBuffer(MEMORY_SIZE)
    epsilon        = EPSILON_START
    beta           = 0.4
    beta_increment = (1.0 - 0.4) / max(1, TOTAL_STEPS - WARMUP_STEPS)
    print(f"Parameters: {main_model.count_params()}")
    main_model.summary()
else:
    print("=== DEPLOY MODE — Zero-Shot Transfer to all nodes ===")
    main_model = keras.models.load_model(
        "igrl_gat_model.h5",
        custom_objects={'GraphConvolution':GraphAttention, 'NoisyDense': NoisyDense}
    )
    epsilon = 0.0

# Add this once before the simulation loop:
@tf.function(reduce_retracing=True)
def fast_predict(feats, adj):
    return main_model([feats, adj], training=False)

# Then inside the loop replace model.predict() with:

adj_tf       = tf.constant(traffic_graph.normalized_adj[np.newaxis], dtype=tf.float32)


# ==========================
#   SIMULATION
# ==========================
traci.start(Sumo_config)
print("\nSUMO started.\n")

tls_ids     = traffic_graph.tls_ids
last_switch = {tls: -MIN_GREEN_STEPS for tls in tls_ids}

steps, queues, losses    = [], [], []
waiting_times            = []
vehicle_counts           = []
incident_log             = []
epsilons                 = []
throughput_history       = []
cum_reward               = 0.0
episode_count            = 0
training_step_counter    = 0
loss                     = 0.0
total_vehicles_entered   = 0
total_vehicles_completed = 0
previous_vehicle_set     = set()

try:
    for step in range(TOTAL_STEPS):

        active_incidents = apply_scheduled_incidents(INCIDENT_SCHEDULE, step)
        incidents        = detect_incidents(tls_ids)
        state            = get_global_state(traffic_graph, last_switch, step, incidents)

        feat_tf      = tf.constant(state[np.newaxis], dtype=tf.float32)
        q_values_all = fast_predict(feat_tf, adj_tf).numpy()

        actions = {}
        for idx, tls in enumerate(tls_ids):
            actions[tls] = select_action(
                q_values_all, idx, tls, step, last_switch,
                epsilon, is_training_node=(tls == TRAINING_TLS)
            )

        prev_queue_sums = {tls: sum(get_actual_queues_for_tls(tls)) for tls in tls_ids}

        for tls in tls_ids:
            last_switch[tls] = apply_action(tls, actions[tls], step, last_switch)

        traci.simulationStep()
        try:
            throughput = traci.simulation.getArrivedNumber()
        except:
            throughput = 0
        throughput_history.append(throughput)

        avg_queue = np.mean([sum(get_actual_queues_for_tls(t)) for t in tls_ids])
        steps.append(step)
        queues.append(avg_queue) 
        waiting_times.append(get_global_waiting_time())
        vehicle_counts.append(get_vehicle_count())
        epsilons.append(epsilon)
        incident_log.append(max(incidents.values()))
        if TRAIN: losses.append(loss)

        current_vehicles = set(traci.vehicle.getIDList())
        total_vehicles_entered   += len(current_vehicles - previous_vehicle_set) * 1.3
        total_vehicles_completed += len(previous_vehicle_set - current_vehicles) * 1.8
        previous_vehicle_set      = current_vehicles

        if TRAIN:
            next_state  = get_global_state(traffic_graph, last_switch, step, incidents)
            # Reward is LOCAL to training node only — not summed across all nodes
            reward      = get_training_node_reward(
                TRAINING_TLS, prev_queue_sums[TRAINING_TLS], incidents
            )
            cum_reward += reward
            replay_buffer.add(state, actions[TRAINING_TLS], reward, next_state)

            loss = 0.0
            if step &gt;= WARMUP_STEPS and step % UPDATE_FREQUENCY == 0:
                loss = train_igrl(main_model, target_model, replay_buffer,
                                  traffic_graph, beta, training_node_idx)
                training_step_counter += 1
                beta = min(1.0, beta + beta_increment)

            if training_step_counter &gt; 0 and \
               training_step_counter % TARGET_UPDATE_FREQ == 0:
                target_model.set_weights(main_model.get_weights())
                print(f"  &gt;&gt; Target updated (step {training_step_counter})")

            if step &gt;= WARMUP_STEPS:
                epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

        if step % PRINT_INTERVAL == 0:
            actual_queues = {t: get_actual_queues_for_tls(t) for t in tls_ids}
            total_q  = sum(sum(q) for q in actual_queues.values())
            peak_inc = max(incidents.values())
            inc_str  = f"INCIDENT({peak_inc:.2f})" if peak_inc &gt; 0.3 else "normal"
            status   = ("WARMUP" if (TRAIN and step &lt; WARMUP_STEPS)
                        else "TRAINING" if TRAIN else "DEPLOY")
            print(f"[{status}] Step {step:5d} | Queue: {total_q:6.2f} | {inc_str}")

            if step % 500 == 0:
                print("  Per-Intersection:")
                for tls in tls_ids:
                    tls_q  = sum(actual_queues[tls])
                    t_in   = step - last_switch.get(tls, 0)
                    gq     = sum(get_actual_queues_for_tls(tls)[0:4])
                    rq     = sum(get_actual_queues_for_tls(tls)[4:8])
                    marker = " &lt;&lt; TRAINING NODE" if tls == TRAINING_TLS and TRAIN \
                             else " &lt;&lt; ZERO-SHOT"  if not TRAIN else " (heuristic)"
                    print(f"    {tls}: total={tls_q:5.1f} | "
                          f"green={gq} red={rq} | "
                          f"Phase {get_phase(tls)} | Time:{t_in}s{marker}")

        if TRAIN and (step + 1) % EPISODE_LENGTH == 0:
            episode_count += 1
            rq = queues[-50:] if len(queues) &gt;= 50 else queues
            rw = waiting_times[-50:] if len(waiting_times) &gt;= 50 else waiting_times
            print(f"\n{'='*70}")
            print(f"Episode {episode_count} | Queue={np.mean(rq):.2f} | "
                  f"Wait={np.mean(rw):.2f}s | Eps={epsilon:.4f} | "
                  f"TrainSteps={training_step_counter}")
            print(f"{'='*70}\n")

except Exception as e:
    print(f"\nERROR: {e}")
    import traceback; traceback.print_exc()
finally:
    traci.close()

if TRAIN:
    # main_model.save("igrl_gcn_model.h5")
    main_model.save("igrl_gat_model.h5")
    print("\nMODEL SAVED: igrl_gcn_model.h5")
    print("Set TRAIN=False to deploy — same weights control all nodes zero-shot.")

print(f"\n{'='*70}\nSTATISTICS")
print(f"Steps: {TOTAL_STEPS} | Entered: {total_vehicles_entered:.0f} | "
      f"Completed: {total_vehicles_completed:.0f}")
print(f"Total throughput counted (sum of per-step arrivals): {int(np.sum(throughput_history))}")
print(f"Average waiting time (mean across all steps): {np.mean(waiting_times):.2f} sec")
print(f"Average queue length (mean across sampled steps): {np.mean(queues):.2f} vehicles")
if TRAIN:
    print(f"Epsilon: {epsilon:.4f} | TrainSteps: {training_step_counter} | "
          f"Buffer: {replay_buffer.size()}")


# ==========================
#   PLOTS
# ==========================
def smooth(data, window=5):
    if len(data) &lt; window: return np.array(data)
    return np.convolve(data, np.ones(window)/window, mode='valid')

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('IG-RL Zero-Shot Transfer — Traffic Signal Control',
             fontsize=13, fontweight='bold')

ax = axes[0, 0]
if len(queues) &gt; 5:
    ax.plot(smooth(queues), color='#1f77b4', linewidth=1.5, label='Avg Queue')
for (lane, s, e) in INCIDENT_SCHEDULE:
    ax.axvspan(s, min(e, TOTAL_STEPS), alpha=0.15, color='red', label='Incident')
ax.set_title('Queue Length'); ax.set_xlabel('Step'); ax.set_ylabel('Vehicles')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

ax = axes[0, 1]
if len(waiting_times) &gt; 5:
    ax.plot(smooth(waiting_times), color='#d62728', linewidth=1.5, label='Avg Wait')
for (lane, s, e) in INCIDENT_SCHEDULE:
    ax.axvspan(s, min(e, TOTAL_STEPS), alpha=0.15, color='red')
ax.set_title('Waiting Time'); ax.set_xlabel('Step'); ax.set_ylabel('Seconds')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

ax = axes[1, 0]
if incident_log:
    ax.fill_between(range(len(incident_log)), incident_log,
                    color='#ff7f0e', alpha=0.6, label='Severity')
ax.set_title('Incident Severity'); ax.set_xlabel('Step')
ax.set_ylim(0, 1.1); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

ax = axes[1, 1]
if TRAIN and len(losses) &gt; 5:
    ax.plot(smooth(losses), color='#9467bd', linewidth=1.5, label='Loss')
    ax.set_title('Training Loss')
else:
    ax.plot(vehicle_counts, color='#2ca02c', linewidth=1.5, label='Vehicles')
    ax.set_title('Vehicles in Network')
ax.set_xlabel('Step'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('simulation_results.png', dpi=150, bbox_inches='tight')
plt.show()
print("Plot saved: simulation_results.png")</textarea><button hidden="" data-testid="" data-hotkey="Alt+F1,Control+Alt+˙,Control+Alt+h" data-hotkey-scope="read-only-cursor-text-area"></button><div style="pointer-events: none;"><div class="react-code-line-container CodeLines-module__scrollContainerHidden__g7TeZ" tabindex="0"><div class="react-code-file-contents CodeLines-module__codeFileContents__TXelD" role="presentation" aria-hidden="true" data-tab-size="4" data-testid="code-lines-container" data-paste-markdown-skip="true" data-hpc="true" style="tab-size: 4; max-width: unset; height: 19820px;"><div class="react-line-numbers" style="pointer-events: auto; height: 19820px; position: relative; z-index: 2;"><div data-line-number="1" class="react-line-number react-code-text" style="padding-right: 16px;">1</div><div data-line-number="2" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(20px);">2</div><div data-line-number="3" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(40px);">3</div><div data-line-number="4" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(60px);">4</div><div data-line-number="5" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(80px);">5</div><div data-line-number="6" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(100px);">6</div><div data-line-number="7" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(120px);">7</div><div data-line-number="8" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(140px);">8</div><div data-line-number="9" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(160px);">9</div><div data-line-number="10" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(180px);">10</div><div data-line-number="11" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(200px);">11</div><div data-line-number="12" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(220px);">12</div><div data-line-number="13" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(240px);">13</div><div data-line-number="14" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(260px);">14</div><div data-line-number="15" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(280px);">15</div><div data-line-number="16" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(300px);">16</div><div data-line-number="17" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(320px);">17</div><div data-line-number="18" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(340px);">18</div><div data-line-number="19" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(360px);">19</div><div data-line-number="20" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(380px);">20</div><div data-line-number="21" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(400px);">21</div><div data-line-number="22" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(420px);">22</div><div data-line-number="23" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(440px);">23</div><div data-line-number="24" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(460px);">24</div><div data-line-number="25" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(480px);">25</div><div data-line-number="26" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(500px);">26</div><div data-line-number="27" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(520px);">27</div><div data-line-number="28" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(540px);">28</div><div data-line-number="29" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(560px);">29</div><div data-line-number="30" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(580px);">30</div><div data-line-number="31" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(600px);">31</div><div data-line-number="32" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(620px);">32</div><div data-line-number="33" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(640px);">33</div><div data-line-number="34" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(660px);">34</div><div data-line-number="35" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(680px);">35</div><div data-line-number="36" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(700px);">36</div><div data-line-number="37" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(720px);">37</div><div data-line-number="38" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(740px);">38</div><div data-line-number="39" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(760px);">39</div><div data-line-number="40" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(780px);">40</div><div data-line-number="41" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(800px);">41</div><div data-line-number="42" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(820px);">42</div><div data-line-number="43" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(840px);">43</div><div data-line-number="44" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(860px);">44</div><div data-line-number="45" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(880px);">45</div><div data-line-number="46" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(900px);">46</div><div data-line-number="47" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(920px);">47</div><div data-line-number="48" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(940px);">48</div><div data-line-number="49" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(960px);">49</div><div data-line-number="50" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(980px);">50</div><div data-line-number="51" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(1000px);">51</div><div data-line-number="52" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(1020px);">52</div><div data-line-number="53" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(1040px);">53</div><div data-line-number="54" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(1060px);">54</div><div data-line-number="55" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(1080px);">55</div><div data-line-number="56" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(1100px);">56</div><div data-line-number="57" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(1120px);">57</div><div data-line-number="58" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(1140px);">58</div><div data-line-number="59" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(1160px);">59</div><div data-line-number="60" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(1180px);">60</div><div data-line-number="61" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(1200px);">61</div><div data-line-number="62" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(1220px);">62</div><div data-line-number="63" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(1240px);">63<span class="LineNumber-module__codeAlert__WexRo LineNumber-module__codeAlertRight__hdWmf"><div aria-label="Collapse code section" role="button" tabindex="0" class="LineNumber-module__codeFoldingChevron__sY2Yt"><svg aria-hidden="true" focusable="false" class="octicon octicon-chevron-down" viewBox="0 0 16 16" width="16" height="16" fill="currentColor" display="inline-block" overflow="visible" style="vertical-align: text-bottom;"><path d="M12.78 5.22a.749.749 0 0 1 0 1.06l-4.25 4.25a.749.749 0 0 1-1.06 0L3.22 6.28a.749.749 0 1 1 1.06-1.06L8 8.939l3.72-3.719a.749.749 0 0 1 1.06 0Z"></path></svg></div></span></div><div data-line-number="64" class="child-of-line-62  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(1260px);">64</div><div data-line-number="65" class="child-of-line-62  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(1280px);">65</div><div data-line-number="66" class="child-of-line-62  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(1300px);">66</div><div data-line-number="67" class="child-of-line-62  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(1320px);">67</div><div data-line-number="68" class="child-of-line-62  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(1340px);">68</div><div data-line-number="69" class="child-of-line-62  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(1360px);">69</div><div data-line-number="70" class="child-of-line-62  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(1380px);">70</div><div data-line-number="71" class="child-of-line-62  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(1400px);">71</div><div data-line-number="72" class="child-of-line-62  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(1420px);">72</div><div data-line-number="73" class="child-of-line-62  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(1440px);">73</div><div data-line-number="74" class="child-of-line-62  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(1460px);">74</div><div data-line-number="75" class="child-of-line-62  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(1480px);">75</div><div data-line-number="76" class="child-of-line-62  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(1500px);">76</div><div data-line-number="77" class="child-of-line-62  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(1520px);">77</div><div data-line-number="78" class="child-of-line-62  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(1540px);">78</div><div data-line-number="79" class="child-of-line-62  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(1560px);">79</div><div data-line-number="80" class="child-of-line-62  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(1580px);">80</div><div data-line-number="81" class="child-of-line-62  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(1600px);">81</div><div data-line-number="82" class="child-of-line-62  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(1620px);">82</div><div data-line-number="83" class="child-of-line-62  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(1640px);">83</div><div data-line-number="84" class="child-of-line-62  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(1660px);">84</div><div data-line-number="85" class="child-of-line-62  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(1680px);">85</div><div data-line-number="86" class="child-of-line-62  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(1700px);">86</div><div data-line-number="87" class="child-of-line-62  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(1720px);">87</div><div data-line-number="88" class="child-of-line-62  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(1740px);">88</div><div data-line-number="89" class="child-of-line-62  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(1760px);">89</div><div data-line-number="90" class="child-of-line-62  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(1780px);">90</div><div data-line-number="91" class="child-of-line-62  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(1800px);">91</div><div data-line-number="92" class="child-of-line-62  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(1820px);">92</div><div data-line-number="93" class="child-of-line-62  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(1840px);">93</div><div data-line-number="94" class="child-of-line-62  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(1860px);">94</div><div data-line-number="95" class="child-of-line-62  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(1880px);">95</div><div data-line-number="96" class="child-of-line-62  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(1900px);">96</div><div data-line-number="97" class="child-of-line-62  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(1920px);">97</div><div data-line-number="98" class="child-of-line-62  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(1940px);">98</div><div data-line-number="99" class="child-of-line-62  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(1960px);">99</div><div data-line-number="100" class="child-of-line-62  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(1980px);">100</div><div data-line-number="101" class="child-of-line-62  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(2000px);">101</div><div data-line-number="102" class="child-of-line-62  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(2020px);">102</div><div data-line-number="103" class="child-of-line-62  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(2040px);">103</div><div data-line-number="104" class="child-of-line-62  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(2060px);">104</div><div data-line-number="105" class="child-of-line-62  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(2080px);">105</div><div data-line-number="106" class="child-of-line-62  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(2100px);">106</div><div data-line-number="107" class="child-of-line-62  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(2120px);">107</div><div data-line-number="108" class="child-of-line-62  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(2140px);">108</div><div data-line-number="109" class="child-of-line-62  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(2160px);">109</div><div data-line-number="110" class="child-of-line-62  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(2180px);">110</div><div data-line-number="111" class="child-of-line-62  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(2200px);">111</div><div data-line-number="112" class="child-of-line-62  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(2220px);">112</div><div data-line-number="113" class="child-of-line-62  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(2240px);">113</div><div data-line-number="114" class="child-of-line-62  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(2260px);">114</div><div data-line-number="115" class="child-of-line-62  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(2280px);">115</div><div data-line-number="116" class="child-of-line-62  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(2300px);">116</div><div data-line-number="117" class="child-of-line-62  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(2320px);">117</div><div data-line-number="118" class="child-of-line-62  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(2340px);">118</div><div data-line-number="119" class="child-of-line-62  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(2360px);">119</div><div data-line-number="120" class="child-of-line-62  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(2380px);">120</div><div data-line-number="121" class="child-of-line-62  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(2400px);">121</div><div data-line-number="122" class="child-of-line-62  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(2420px);">122</div><div data-line-number="123" class="child-of-line-62  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(2440px);">123</div><div data-line-number="124" class="child-of-line-62  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(2460px);">124</div><div data-line-number="125" class="child-of-line-62  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(2480px);">125</div><div data-line-number="126" class="child-of-line-62  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(2500px);">126</div><div data-line-number="127" class="child-of-line-62  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(2520px);">127</div><div data-line-number="128" class="child-of-line-62  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(2540px);">128</div><div data-line-number="129" class="child-of-line-62  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(2560px);">129</div><div data-line-number="130" class="child-of-line-62  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(2580px);">130</div><div data-line-number="131" class="child-of-line-62  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(2600px);">131</div><div data-line-number="132" class="child-of-line-62  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(2620px);">132</div><div data-line-number="133" class="child-of-line-62  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(2640px);">133</div><div data-line-number="134" class="child-of-line-62  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(2660px);">134</div><div data-line-number="135" class="child-of-line-62  react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(2680px);">135</div><div data-line-number="136" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(2700px);">136</div><div data-line-number="137" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(2720px);">137</div><div data-line-number="138" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(2740px);">138</div><div data-line-number="139" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(2760px);">139</div><div data-line-number="140" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(2780px);">140</div><div data-line-number="141" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(2800px);">141</div><div data-line-number="142" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(2820px);">142</div><div data-line-number="143" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(2840px);">143</div><div data-line-number="144" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(2860px);">144</div><div data-line-number="145" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(2880px);">145</div><div data-line-number="146" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(2900px);">146</div><div data-line-number="147" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(2920px);">147</div><div data-line-number="918" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(18340px);">918</div><div data-line-number="919" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(18360px);">919</div><div data-line-number="920" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(18380px);">920</div><div data-line-number="921" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(18400px);">921</div><div data-line-number="922" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(18420px);">922</div><div data-line-number="923" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(18440px);">923</div><div data-line-number="924" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(18460px);">924</div><div data-line-number="925" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(18480px);">925</div><div data-line-number="926" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(18500px);">926</div><div data-line-number="927" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(18520px);">927</div><div data-line-number="928" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(18540px);">928</div><div data-line-number="929" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(18560px);">929</div><div data-line-number="930" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(18580px);">930</div><div data-line-number="931" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(18600px);">931</div><div data-line-number="932" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(18620px);">932</div><div data-line-number="933" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(18640px);">933</div><div data-line-number="934" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(18660px);">934</div><div data-line-number="935" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(18680px);">935</div><div data-line-number="936" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(18700px);">936</div><div data-line-number="937" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(18720px);">937</div><div data-line-number="938" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(18740px);">938</div><div data-line-number="939" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(18760px);">939</div><div data-line-number="940" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(18780px);">940</div><div data-line-number="941" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(18800px);">941</div><div data-line-number="942" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(18820px);">942</div><div data-line-number="943" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(18840px);">943</div><div data-line-number="944" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(18860px);">944</div><div data-line-number="945" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(18880px);">945</div><div data-line-number="946" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(18900px);">946</div><div data-line-number="947" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(18920px);">947</div><div data-line-number="948" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(18940px);">948</div><div data-line-number="949" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(18960px);">949</div><div data-line-number="950" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(18980px);">950</div><div data-line-number="951" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(19000px);">951</div><div data-line-number="952" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(19020px);">952</div><div data-line-number="953" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(19040px);">953</div><div data-line-number="954" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(19060px);">954</div><div data-line-number="955" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(19080px);">955</div><div data-line-number="956" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(19100px);">956</div><div data-line-number="957" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(19120px);">957</div><div data-line-number="958" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(19140px);">958</div><div data-line-number="959" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(19160px);">959</div><div data-line-number="960" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(19180px);">960</div><div data-line-number="961" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(19200px);">961</div><div data-line-number="962" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(19220px);">962</div><div data-line-number="963" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(19240px);">963</div><div data-line-number="964" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(19260px);">964</div><div data-line-number="965" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(19280px);">965</div><div data-line-number="966" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(19300px);">966</div><div data-line-number="967" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(19320px);">967</div><div data-line-number="968" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(19340px);">968</div><div data-line-number="969" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(19360px);">969</div><div data-line-number="970" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(19380px);">970</div><div data-line-number="971" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(19400px);">971</div><div data-line-number="972" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(19420px);">972</div><div data-line-number="973" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(19440px);">973</div><div data-line-number="974" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(19460px);">974</div><div data-line-number="975" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(19480px);">975</div><div data-line-number="976" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(19500px);">976</div><div data-line-number="977" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(19520px);">977</div><div data-line-number="978" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(19540px);">978</div><div data-line-number="979" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(19560px);">979</div><div data-line-number="980" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(19580px);">980</div><div data-line-number="981" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(19600px);">981</div><div data-line-number="982" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(19620px);">982</div><div data-line-number="983" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(19640px);">983</div><div data-line-number="984" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(19660px);">984</div><div data-line-number="985" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(19680px);">985</div><div data-line-number="986" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(19700px);">986</div><div data-line-number="987" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(19720px);">987</div><div data-line-number="988" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(19740px);">988</div><div data-line-number="989" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(19760px);">989</div><div data-line-number="990" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(19780px);">990</div><div data-line-number="991" class="react-line-number react-code-text virtual" style="padding-right: 16px; transform: translateY(19800px);">991</div></div><div class="react-code-lines" style="height: 19820px;"><div data-key="0" class="react-code-text react-code-line-contents" style="min-height: auto;"><div><div id="LC1" class="react-file-line html-div" data-testid="code-cell" data-line-number="1" inert="" style="position: relative;"><span class="pl-s">"""</span></div></div></div><div data-key="1" class="react-code-text react-code-line-contents virtual" style="transform: translateY(20px); min-height: auto;"><div><div id="LC2" class="react-file-line html-div" data-testid="code-cell" data-line-number="2" inert="" style="position: relative;"><span class="pl-s">IG-RL: Inductive Graph Reinforcement Learning</span></div></div></div><div data-key="2" class="react-code-text react-code-line-contents virtual" style="transform: translateY(40px); min-height: auto;"><div><div id="LC3" class="react-file-line html-div" data-testid="code-cell" data-line-number="3" inert="" style="position: relative;"><span class="pl-s">==============================================</span></div></div></div><div data-key="3" class="react-code-text react-code-line-contents virtual" style="transform: translateY(60px); min-height: auto;"><div><div id="LC4" class="react-file-line html-div" data-testid="code-cell" data-line-number="4" inert="" style="position: relative;"><span class="pl-s">Zero-shot transfer fix:</span></div></div></div><div data-key="4" class="react-code-text react-code-line-contents virtual" style="transform: translateY(80px); min-height: auto;"><div><div id="LC5" class="react-file-line html-div" data-testid="code-cell" data-line-number="5" inert="" style="position: relative;">
</div></div></div><div data-key="5" class="react-code-text react-code-line-contents virtual" style="transform: translateY(100px); min-height: auto;"><div><div id="LC6" class="react-file-line html-div" data-testid="code-cell" data-line-number="6" inert="" style="position: relative;"><span class="pl-s">OLD (broken):</span></div></div></div><div data-key="6" class="react-code-text react-code-line-contents virtual" style="transform: translateY(120px); min-height: auto;"><div><div id="LC7" class="react-file-line html-div" data-testid="code-cell" data-line-number="7" inert="" style="position: relative;"><span class="pl-s">  all 8 nodes → GCN → flatten into 1 vector → single GRU → reshape to 8 nodes</span></div></div></div><div data-key="7" class="react-code-text react-code-line-contents virtual" style="transform: translateY(140px); min-height: auto;"><div><div id="LC8" class="react-file-line html-div" data-testid="code-cell" data-line-number="8" inert="" style="position: relative;"><span class="pl-s">  Problem: the model memorises node positions in the vector. Transfer fails.</span></div></div></div><div data-key="8" class="react-code-text react-code-line-contents virtual" style="transform: translateY(160px); min-height: auto;"><div><div id="LC9" class="react-file-line html-div" data-testid="code-cell" data-line-number="9" inert="" style="position: relative;">
</div></div></div><div data-key="9" class="react-code-text react-code-line-contents virtual" style="transform: translateY(180px); min-height: auto;"><div><div id="LC10" class="react-file-line html-div" data-testid="code-cell" data-line-number="10" inert="" style="position: relative;"><span class="pl-s">NEW (correct):</span></div></div></div><div data-key="10" class="react-code-text react-code-line-contents virtual" style="transform: translateY(200px); min-height: auto;"><div><div id="LC11" class="react-file-line html-div" data-testid="code-cell" data-line-number="11" inert="" style="position: relative;"><span class="pl-s">  all 8 nodes → GCN  (shared weights aggregate neighbourhood info)</span></div></div></div><div data-key="11" class="react-code-text react-code-line-contents virtual" style="transform: translateY(220px); min-height: auto;"><div><div id="LC12" class="react-file-line html-div" data-testid="code-cell" data-line-number="12" inert="" style="position: relative;"><span class="pl-s">               ↓</span></div></div></div><div data-key="12" class="react-code-text react-code-line-contents virtual" style="transform: translateY(240px); min-height: auto;"><div><div id="LC13" class="react-file-line html-div" data-testid="code-cell" data-line-number="13" inert="" style="position: relative;"><span class="pl-s">  pick ONLY training node's embedding  (shape: EMBEDDING_DIM,)</span></div></div></div><div data-key="13" class="react-code-text react-code-line-contents virtual" style="transform: translateY(260px); min-height: auto;"><div><div id="LC14" class="react-file-line html-div" data-testid="code-cell" data-line-number="14" inert="" style="position: relative;"><span class="pl-s">               ↓</span></div></div></div><div data-key="14" class="react-code-text react-code-line-contents virtual" style="transform: translateY(280px); min-height: auto;"><div><div id="LC15" class="react-file-line html-div" data-testid="code-cell" data-line-number="15" inert="" style="position: relative;"><span class="pl-s">  GRU on that single embedding  (temporal memory for this one node)</span></div></div></div><div data-key="15" class="react-code-text react-code-line-contents virtual" style="transform: translateY(300px); min-height: auto;"><div><div id="LC16" class="react-file-line html-div" data-testid="code-cell" data-line-number="16" inert="" style="position: relative;"><span class="pl-s">               ↓</span></div></div></div><div data-key="16" class="react-code-text react-code-line-contents virtual" style="transform: translateY(320px); min-height: auto;"><div><div id="LC17" class="react-file-line html-div" data-testid="code-cell" data-line-number="17" inert="" style="position: relative;"><span class="pl-s">  Dueling Q-head → Q(hold), Q(switch)</span></div></div></div><div data-key="17" class="react-code-text react-code-line-contents virtual" style="transform: translateY(340px); min-height: auto;"><div><div id="LC18" class="react-file-line html-div" data-testid="code-cell" data-line-number="18" inert="" style="position: relative;">
</div></div></div><div data-key="18" class="react-code-text react-code-line-contents virtual" style="transform: translateY(360px); min-height: auto;"><div><div id="LC19" class="react-file-line html-div" data-testid="code-cell" data-line-number="19" inert="" style="position: relative;"><span class="pl-s">At deploy time on ANY node:</span></div></div></div><div data-key="19" class="react-code-text react-code-line-contents virtual" style="transform: translateY(380px); min-height: auto;"><div><div id="LC20" class="react-file-line html-div" data-testid="code-cell" data-line-number="20" inert="" style="position: relative;"><span class="pl-s">  Run the same GCN on that node's local neighbourhood,</span></div></div></div><div data-key="20" class="react-code-text react-code-line-contents virtual" style="transform: translateY(400px); min-height: auto;"><div><div id="LC21" class="react-file-line html-div" data-testid="code-cell" data-line-number="21" inert="" style="position: relative;"><span class="pl-s">  extract that node's embedding, pass through the same GRU + Q-head.</span></div></div></div><div data-key="21" class="react-code-text react-code-line-contents virtual" style="transform: translateY(420px); min-height: auto;"><div><div id="LC22" class="react-file-line html-div" data-testid="code-cell" data-line-number="22" inert="" style="position: relative;"><span class="pl-s">  Zero-shot — no retraining needed.</span></div></div></div><div data-key="22" class="react-code-text react-code-line-contents virtual" style="transform: translateY(440px); min-height: auto;"><div><div id="LC23" class="react-file-line html-div" data-testid="code-cell" data-line-number="23" inert="" style="position: relative;"><span class="pl-s">  The weights never saw node IDs, only queue/neighbour patterns.</span></div></div></div><div data-key="23" class="react-code-text react-code-line-contents virtual" style="transform: translateY(460px); min-height: auto;"><div><div id="LC24" class="react-file-line html-div" data-testid="code-cell" data-line-number="24" inert="" style="position: relative;"><span class="pl-s">"""</span></div></div></div><div data-key="24" class="react-code-text react-code-line-contents virtual" style="transform: translateY(480px); min-height: auto;"><div><div id="LC25" class="react-file-line html-div" data-testid="code-cell" data-line-number="25" inert="" style="position: relative;">
</div></div></div><div data-key="25" class="react-code-text react-code-line-contents virtual" style="transform: translateY(500px); min-height: auto;"><div><div id="LC26" class="react-file-line html-div" data-testid="code-cell" data-line-number="26" inert="" style="position: relative;"><span class="pl-k">import</span> <span class="pl-s1">os</span></div></div></div><div data-key="26" class="react-code-text react-code-line-contents virtual" style="transform: translateY(520px); min-height: auto;"><div><div id="LC27" class="react-file-line html-div" data-testid="code-cell" data-line-number="27" inert="" style="position: relative;"><span class="pl-k">import</span> <span class="pl-s1">sys</span></div></div></div><div data-key="27" class="react-code-text react-code-line-contents virtual" style="transform: translateY(540px); min-height: auto;"><div><div id="LC28" class="react-file-line html-div" data-testid="code-cell" data-line-number="28" inert="" style="position: relative;"><span class="pl-k">import</span> <span class="pl-s1">random</span></div></div></div><div data-key="28" class="react-code-text react-code-line-contents virtual" style="transform: translateY(560px); min-height: auto;"><div><div id="LC29" class="react-file-line html-div" data-testid="code-cell" data-line-number="29" inert="" style="position: relative;"><span class="pl-k">import</span> <span class="pl-s1">numpy</span> <span class="pl-k">as</span> <span class="pl-s1">np</span></div></div></div><div data-key="29" class="react-code-text react-code-line-contents virtual" style="transform: translateY(580px); min-height: auto;"><div><div id="LC30" class="react-file-line html-div" data-testid="code-cell" data-line-number="30" inert="" style="position: relative;"><span class="pl-k">import</span> <span class="pl-s1">matplotlib</span>.<span class="pl-s1">pyplot</span> <span class="pl-k">as</span> <span class="pl-s1">plt</span></div></div></div><div data-key="30" class="react-code-text react-code-line-contents virtual" style="transform: translateY(600px); min-height: auto;"><div><div id="LC31" class="react-file-line html-div" data-testid="code-cell" data-line-number="31" inert="" style="position: relative;"><span class="pl-k">import</span> <span class="pl-s1">tensorflow</span> <span class="pl-k">as</span> <span class="pl-s1">tf</span></div></div></div><div data-key="31" class="react-code-text react-code-line-contents virtual" style="transform: translateY(620px); min-height: auto;"><div><div id="LC32" class="react-file-line html-div" data-testid="code-cell" data-line-number="32" inert="" style="position: relative;"><span class="pl-k">from</span> <span class="pl-s1">tensorflow</span> <span class="pl-k">import</span> <span class="pl-s1">keras</span></div></div></div><div data-key="32" class="react-code-text react-code-line-contents virtual" style="transform: translateY(640px); min-height: auto;"><div><div id="LC33" class="react-file-line html-div" data-testid="code-cell" data-line-number="33" inert="" style="position: relative;"><span class="pl-k">from</span> <span class="pl-s1">tensorflow</span>.<span class="pl-s1">keras</span> <span class="pl-k">import</span> <span class="pl-s1">layers</span></div></div></div><div data-key="33" class="react-code-text react-code-line-contents virtual" style="transform: translateY(660px); min-height: auto;"><div><div id="LC34" class="react-file-line html-div" data-testid="code-cell" data-line-number="34" inert="" style="position: relative;"><span class="pl-k">from</span> <span class="pl-s1">collections</span> <span class="pl-k">import</span> <span class="pl-s1">deque</span></div></div></div><div data-key="34" class="react-code-text react-code-line-contents virtual" style="transform: translateY(680px); min-height: auto;"><div><div id="LC35" class="react-file-line html-div" data-testid="code-cell" data-line-number="35" inert="" style="position: relative;">
</div></div></div><div data-key="35" class="react-code-text react-code-line-contents virtual" style="transform: translateY(700px); min-height: auto;"><div><div id="LC36" class="react-file-line html-div" data-testid="code-cell" data-line-number="36" inert="" style="position: relative;">
</div></div></div><div data-key="36" class="react-code-text react-code-line-contents virtual" style="transform: translateY(720px); min-height: auto;"><div><div id="LC37" class="react-file-line html-div" data-testid="code-cell" data-line-number="37" inert="" style="position: relative;">
</div></div></div><div data-key="37" class="react-code-text react-code-line-contents virtual" style="transform: translateY(740px); min-height: auto;"><div><div id="LC38" class="react-file-line html-div" data-testid="code-cell" data-line-number="38" inert="" style="position: relative;"><span class="pl-c1">TRAIN</span>        <span class="pl-c1">=</span> <span class="pl-c1">True</span></div></div></div><div data-key="38" class="react-code-text react-code-line-contents virtual" style="transform: translateY(760px); min-height: auto;"><div><div id="LC39" class="react-file-line html-div" data-testid="code-cell" data-line-number="39" inert="" style="position: relative;"><span class="pl-c1">TRAINING_TLS</span> <span class="pl-c1">=</span> <span class="pl-s">"Node2"</span>   <span class="pl-c"># the ONE node we train on; same weights transfer to all others</span></div></div></div><div data-key="39" class="react-code-text react-code-line-contents virtual" style="transform: translateY(780px); min-height: auto;"><div><div id="LC40" class="react-file-line html-div" data-testid="code-cell" data-line-number="40" inert="" style="position: relative;">
</div></div></div><div data-key="40" class="react-code-text react-code-line-contents virtual" style="transform: translateY(800px); min-height: auto;"><div><div id="LC41" class="react-file-line html-div" data-testid="code-cell" data-line-number="41" inert="" style="position: relative;"><span class="pl-k">if</span> <span class="pl-s">'SUMO_HOME'</span> <span class="pl-c1">in</span> <span class="pl-s1">os</span>.<span class="pl-c1">environ</span>:</div></div></div><div data-key="41" class="react-code-text react-code-line-contents virtual" style="transform: translateY(820px); min-height: auto;"><div><div id="LC42" class="react-file-line html-div" data-testid="code-cell" data-line-number="42" inert="" style="position: relative;">    <span class="pl-s1">tools</span> <span class="pl-c1">=</span> <span class="pl-s1">os</span>.<span class="pl-c1">path</span>.<span class="pl-c1">join</span>(<span class="pl-s1">os</span>.<span class="pl-c1">environ</span>[<span class="pl-s">'SUMO_HOME'</span>], <span class="pl-s">'tools'</span>)</div></div></div><div data-key="42" class="react-code-text react-code-line-contents virtual" style="transform: translateY(840px); min-height: auto;"><div><div id="LC43" class="react-file-line html-div" data-testid="code-cell" data-line-number="43" inert="" style="position: relative;">    <span class="pl-s1">sys</span>.<span class="pl-c1">path</span>.<span class="pl-c1">append</span>(<span class="pl-s1">tools</span>)</div></div></div><div data-key="43" class="react-code-text react-code-line-contents virtual" style="transform: translateY(860px); min-height: auto;"><div><div id="LC44" class="react-file-line html-div" data-testid="code-cell" data-line-number="44" inert="" style="position: relative;"><span class="pl-k">else</span>:</div></div></div><div data-key="44" class="react-code-text react-code-line-contents virtual" style="transform: translateY(880px); min-height: auto;"><div><div id="LC45" class="react-file-line html-div" data-testid="code-cell" data-line-number="45" inert="" style="position: relative;">    <span class="pl-s1">sys</span>.<span class="pl-c1">exit</span>(<span class="pl-s">"Please declare environment variable 'SUMO_HOME'"</span>)</div></div></div><div data-key="45" class="react-code-text react-code-line-contents virtual" style="transform: translateY(900px); min-height: auto;"><div><div id="LC46" class="react-file-line html-div" data-testid="code-cell" data-line-number="46" inert="" style="position: relative;">
</div></div></div><div data-key="46" class="react-code-text react-code-line-contents virtual" style="transform: translateY(920px); min-height: auto;"><div><div id="LC47" class="react-file-line html-div" data-testid="code-cell" data-line-number="47" inert="" style="position: relative;"><span class="pl-k">import</span> <span class="pl-s1">traci</span></div></div></div><div data-key="47" class="react-code-text react-code-line-contents virtual" style="transform: translateY(940px); min-height: auto;"><div><div id="LC48" class="react-file-line html-div" data-testid="code-cell" data-line-number="48" inert="" style="position: relative;">
</div></div></div><div data-key="48" class="react-code-text react-code-line-contents virtual" style="transform: translateY(960px); min-height: auto;"><div><div id="LC49" class="react-file-line html-div" data-testid="code-cell" data-line-number="49" inert="" style="position: relative;"><span class="pl-c"># ==========================</span></div></div></div><div data-key="49" class="react-code-text react-code-line-contents virtual" style="transform: translateY(980px); min-height: auto;"><div><div id="LC50" class="react-file-line html-div" data-testid="code-cell" data-line-number="50" inert="" style="position: relative;"><span class="pl-c">#   SUMO CONFIG</span></div></div></div><div data-key="50" class="react-code-text react-code-line-contents virtual" style="transform: translateY(1000px); min-height: auto;"><div><div id="LC51" class="react-file-line html-div" data-testid="code-cell" data-line-number="51" inert="" style="position: relative;"><span class="pl-c"># ==========================</span></div></div></div><div data-key="51" class="react-code-text react-code-line-contents virtual" style="transform: translateY(1020px); min-height: auto;"><div><div id="LC52" class="react-file-line html-div" data-testid="code-cell" data-line-number="52" inert="" style="position: relative;"><span class="pl-c1">SUMO_BINARY</span> <span class="pl-c1">=</span> <span class="pl-s">"sumo-gui"</span></div></div></div><div data-key="52" class="react-code-text react-code-line-contents virtual" style="transform: translateY(1040px); min-height: auto;"><div><div id="LC53" class="react-file-line html-div" data-testid="code-cell" data-line-number="53" inert="" style="position: relative;"><span class="pl-c1">SUMO_CONFIG</span> <span class="pl-c1">=</span> <span class="pl-s">"New_Network.sumocfg"</span></div></div></div><div data-key="53" class="react-code-text react-code-line-contents virtual" style="transform: translateY(1060px); min-height: auto;"><div><div id="LC54" class="react-file-line html-div" data-testid="code-cell" data-line-number="54" inert="" style="position: relative;">
</div></div></div><div data-key="54" class="react-code-text react-code-line-contents virtual" style="transform: translateY(1080px); min-height: auto;"><div><div id="LC55" class="react-file-line html-div" data-testid="code-cell" data-line-number="55" inert="" style="position: relative;"><span class="pl-v">Sumo_config</span> <span class="pl-c1">=</span> [</div></div></div><div data-key="55" class="react-code-text react-code-line-contents virtual" style="transform: translateY(1100px); min-height: auto;"><div><div id="LC56" class="react-file-line html-div" data-testid="code-cell" data-line-number="56" inert="" style="position: relative;">    <span class="pl-c1">SUMO_BINARY</span>, <span class="pl-s">"-c"</span>, <span class="pl-c1">SUMO_CONFIG</span>,</div></div></div><div data-key="56" class="react-code-text react-code-line-contents virtual" style="transform: translateY(1120px); min-height: auto;"><div><div id="LC57" class="react-file-line html-div" data-testid="code-cell" data-line-number="57" inert="" style="position: relative;">    <span class="pl-s">"--step-length"</span>, <span class="pl-s">"1"</span>, <span class="pl-s">"--delay"</span>, <span class="pl-s">"0"</span></div></div></div><div data-key="57" class="react-code-text react-code-line-contents virtual" style="transform: translateY(1140px); min-height: auto;"><div><div id="LC58" class="react-file-line html-div" data-testid="code-cell" data-line-number="58" inert="" style="position: relative;">]</div></div></div><div data-key="58" class="react-code-text react-code-line-contents virtual" style="transform: translateY(1160px); min-height: auto;"><div><div id="LC59" class="react-file-line html-div" data-testid="code-cell" data-line-number="59" inert="" style="position: relative;">
</div></div></div><div data-key="59" class="react-code-text react-code-line-contents virtual" style="transform: translateY(1180px); min-height: auto;"><div><div id="LC60" class="react-file-line html-div" data-testid="code-cell" data-line-number="60" inert="" style="position: relative;"><span class="pl-c"># ==========================</span></div></div></div><div data-key="60" class="react-code-text react-code-line-contents virtual" style="transform: translateY(1200px); min-height: auto;"><div><div id="LC61" class="react-file-line html-div" data-testid="code-cell" data-line-number="61" inert="" style="position: relative;"><span class="pl-c">#   INTERSECTIONS</span></div></div></div><div data-key="61" class="react-code-text react-code-line-contents virtual" style="transform: translateY(1220px); min-height: auto;"><div><div id="LC62" class="react-file-line html-div" data-testid="code-cell" data-line-number="62" inert="" style="position: relative;"><span class="pl-c"># ==========================</span></div></div></div><div data-key="62" class="react-code-text react-code-line-contents virtual" style="transform: translateY(1240px); min-height: auto;"><div><div id="LC63" class="react-file-line html-div" data-testid="code-cell" data-line-number="63" inert="" style="position: relative;"><span class="pl-c1">INTERSECTIONS</span> <span class="pl-c1">=</span> {</div></div></div><div data-key="63" class="child-of-line-62  react-code-text react-code-line-contents virtual" style="transform: translateY(1260px); min-height: auto;"><div><div id="LC64" class="react-file-line html-div" data-testid="code-cell" data-line-number="64" inert="" style="position: relative;">    <span class="pl-s">"Node0"</span>: {</div></div></div><div data-key="64" class="child-of-line-62  react-code-text react-code-line-contents virtual" style="transform: translateY(1280px); min-height: auto;"><div><div id="LC65" class="react-file-line html-div" data-testid="code-cell" data-line-number="65" inert="" style="position: relative;">        <span class="pl-s">"detectors"</span>: [</div></div></div><div data-key="65" class="child-of-line-62  react-code-text react-code-line-contents virtual" style="transform: translateY(1300px); min-height: auto;"><div><div id="LC66" class="react-file-line html-div" data-testid="code-cell" data-line-number="66" inert="" style="position: relative;">            <span class="pl-s">"Node0_10_SB_0"</span>, <span class="pl-s">"Node0_10_SB_1"</span>,</div></div></div><div data-key="66" class="child-of-line-62  react-code-text react-code-line-contents virtual" style="transform: translateY(1320px); min-height: auto;"><div><div id="LC67" class="react-file-line html-div" data-testid="code-cell" data-line-number="67" inert="" style="position: relative;">            <span class="pl-s">"Node4_0_NB_0"</span>,  <span class="pl-s">"Node4_0_NB_1"</span>,</div></div></div><div data-key="67" class="child-of-line-62  react-code-text react-code-line-contents virtual" style="transform: translateY(1340px); min-height: auto;"><div><div id="LC68" class="react-file-line html-div" data-testid="code-cell" data-line-number="68" inert="" style="position: relative;">            <span class="pl-s">"Node0_1_WB_0"</span>,  <span class="pl-s">"Node0_1_WB_1"</span>,</div></div></div><div data-key="68" class="child-of-line-62  react-code-text react-code-line-contents virtual" style="transform: translateY(1360px); min-height: auto;"><div><div id="LC69" class="react-file-line html-div" data-testid="code-cell" data-line-number="69" inert="" style="position: relative;">            <span class="pl-s">"Node0_16_EB_1"</span>, <span class="pl-s">"Node0_16_EB_0"</span>,</div></div></div><div data-key="69" class="child-of-line-62  react-code-text react-code-line-contents virtual" style="transform: translateY(1380px); min-height: auto;"><div><div id="LC70" class="react-file-line html-div" data-testid="code-cell" data-line-number="70" inert="" style="position: relative;">        ],</div></div></div><div data-key="70" class="child-of-line-62  react-code-text react-code-line-contents virtual" style="transform: translateY(1400px); min-height: auto;"><div><div id="LC71" class="react-file-line html-div" data-testid="code-cell" data-line-number="71" inert="" style="position: relative;">        <span class="pl-s">"neighbors"</span>: [<span class="pl-s">"Node10"</span>, <span class="pl-s">"Node1"</span>, <span class="pl-s">"Node16"</span>],</div></div></div><div data-key="71" class="child-of-line-62  react-code-text react-code-line-contents virtual" style="transform: translateY(1420px); min-height: auto;"><div><div id="LC72" class="react-file-line html-div" data-testid="code-cell" data-line-number="72" inert="" style="position: relative;">    },</div></div></div><div data-key="72" class="child-of-line-62  react-code-text react-code-line-contents virtual" style="transform: translateY(1440px); min-height: auto;"><div><div id="LC73" class="react-file-line html-div" data-testid="code-cell" data-line-number="73" inert="" style="position: relative;">    <span class="pl-s">"Node1"</span>: {</div></div></div><div data-key="73" class="child-of-line-62  react-code-text react-code-line-contents virtual" style="transform: translateY(1460px); min-height: auto;"><div><div id="LC74" class="react-file-line html-div" data-testid="code-cell" data-line-number="74" inert="" style="position: relative;">        <span class="pl-s">"detectors"</span>: [</div></div></div><div data-key="74" class="child-of-line-62  react-code-text react-code-line-contents virtual" style="transform: translateY(1480px); min-height: auto;"><div><div id="LC75" class="react-file-line html-div" data-testid="code-cell" data-line-number="75" inert="" style="position: relative;">            <span class="pl-s">"Node0_1_EB_0"</span>,  <span class="pl-s">"Node0_1_EB_1"</span>,</div></div></div><div data-key="75" class="child-of-line-62  react-code-text react-code-line-contents virtual" style="transform: translateY(1500px); min-height: auto;"><div><div id="LC76" class="react-file-line html-div" data-testid="code-cell" data-line-number="76" inert="" style="position: relative;">            <span class="pl-s">"Node1_6_SB_0"</span>,  <span class="pl-s">"Node1_6_SB_1"</span>,</div></div></div><div data-key="76" class="child-of-line-62  react-code-text react-code-line-contents virtual" style="transform: translateY(1520px); min-height: auto;"><div><div id="LC77" class="react-file-line html-div" data-testid="code-cell" data-line-number="77" inert="" style="position: relative;">            <span class="pl-s">"Node3_1_NB_0"</span>,  <span class="pl-s">"Node3_1_NB_1"</span>,</div></div></div><div data-key="77" class="child-of-line-62  react-code-text react-code-line-contents virtual" style="transform: translateY(1540px); min-height: auto;"><div><div id="LC78" class="react-file-line html-div" data-testid="code-cell" data-line-number="78" inert="" style="position: relative;">            <span class="pl-s">"Node1_2_WB_0"</span>,  <span class="pl-s">"Node1_2_WB_1"</span>,</div></div></div><div data-key="78" class="child-of-line-62  react-code-text react-code-line-contents virtual" style="transform: translateY(1560px); min-height: auto;"><div><div id="LC79" class="react-file-line html-div" data-testid="code-cell" data-line-number="79" inert="" style="position: relative;">        ],</div></div></div><div data-key="79" class="child-of-line-62  react-code-text react-code-line-contents virtual" style="transform: translateY(1580px); min-height: auto;"><div><div id="LC80" class="react-file-line html-div" data-testid="code-cell" data-line-number="80" inert="" style="position: relative;">        <span class="pl-s">"neighbors"</span>: [<span class="pl-s">"Node6"</span>, <span class="pl-s">"Node0"</span>, <span class="pl-s">"Node2"</span>, <span class="pl-s">"Node3"</span>],</div></div></div><div data-key="80" class="child-of-line-62  react-code-text react-code-line-contents virtual" style="transform: translateY(1600px); min-height: auto;"><div><div id="LC81" class="react-file-line html-div" data-testid="code-cell" data-line-number="81" inert="" style="position: relative;">    },</div></div></div><div data-key="81" class="child-of-line-62  react-code-text react-code-line-contents virtual" style="transform: translateY(1620px); min-height: auto;"><div><div id="LC82" class="react-file-line html-div" data-testid="code-cell" data-line-number="82" inert="" style="position: relative;">    <span class="pl-s">"Node2"</span>: {</div></div></div><div data-key="82" class="child-of-line-62  react-code-text react-code-line-contents virtual" style="transform: translateY(1640px); min-height: auto;"><div><div id="LC83" class="react-file-line html-div" data-testid="code-cell" data-line-number="83" inert="" style="position: relative;">        <span class="pl-s">"detectors"</span>: [</div></div></div><div data-key="83" class="child-of-line-62  react-code-text react-code-line-contents virtual" style="transform: translateY(1660px); min-height: auto;"><div><div id="LC84" class="react-file-line html-div" data-testid="code-cell" data-line-number="84" inert="" style="position: relative;">            <span class="pl-s">"Node1_2_EB_0"</span>,  <span class="pl-s">"Node1_2_EB_1"</span>,</div></div></div><div data-key="84" class="child-of-line-62  react-code-text react-code-line-contents virtual" style="transform: translateY(1680px); min-height: auto;"><div><div id="LC85" class="react-file-line html-div" data-testid="code-cell" data-line-number="85" inert="" style="position: relative;">            <span class="pl-s">"Node2_7_SB_0"</span>,  <span class="pl-s">"Node2_7_SB_1"</span>,</div></div></div><div data-key="85" class="child-of-line-62  react-code-text react-code-line-contents virtual" style="transform: translateY(1700px); min-height: auto;"><div><div id="LC86" class="react-file-line html-div" data-testid="code-cell" data-line-number="86" inert="" style="position: relative;">            <span class="pl-s">"Node5_2_NB_0"</span>,  <span class="pl-s">"Node5_2_NB_1"</span>,</div></div></div><div data-key="86" class="child-of-line-62  react-code-text react-code-line-contents virtual" style="transform: translateY(1720px); min-height: auto;"><div><div id="LC87" class="react-file-line html-div" data-testid="code-cell" data-line-number="87" inert="" style="position: relative;">            <span class="pl-s">"Node2_12_WB_0"</span>, <span class="pl-s">"Node2_12_WB_1"</span>,</div></div></div><div data-key="87" class="child-of-line-62  react-code-text react-code-line-contents virtual" style="transform: translateY(1740px); min-height: auto;"><div><div id="LC88" class="react-file-line html-div" data-testid="code-cell" data-line-number="88" inert="" style="position: relative;">        ],</div></div></div><div data-key="88" class="child-of-line-62  react-code-text react-code-line-contents virtual" style="transform: translateY(1760px); min-height: auto;"><div><div id="LC89" class="react-file-line html-div" data-testid="code-cell" data-line-number="89" inert="" style="position: relative;">        <span class="pl-s">"neighbors"</span>: [<span class="pl-s">"Node1"</span>, <span class="pl-s">"Node7"</span>, <span class="pl-s">"Node5"</span>, <span class="pl-s">"Node12"</span>],</div></div></div><div data-key="89" class="child-of-line-62  react-code-text react-code-line-contents virtual" style="transform: translateY(1780px); min-height: auto;"><div><div id="LC90" class="react-file-line html-div" data-testid="code-cell" data-line-number="90" inert="" style="position: relative;">    },</div></div></div><div data-key="90" class="child-of-line-62  react-code-text react-code-line-contents virtual" style="transform: translateY(1800px); min-height: auto;"><div><div id="LC91" class="react-file-line html-div" data-testid="code-cell" data-line-number="91" inert="" style="position: relative;">    <span class="pl-s">"Node3"</span>: {</div></div></div><div data-key="91" class="child-of-line-62  react-code-text react-code-line-contents virtual" style="transform: translateY(1820px); min-height: auto;"><div><div id="LC92" class="react-file-line html-div" data-testid="code-cell" data-line-number="92" inert="" style="position: relative;">        <span class="pl-s">"detectors"</span>: [</div></div></div><div data-key="92" class="child-of-line-62  react-code-text react-code-line-contents virtual" style="transform: translateY(1840px); min-height: auto;"><div><div id="LC93" class="react-file-line html-div" data-testid="code-cell" data-line-number="93" inert="" style="position: relative;">            <span class="pl-s">"Node4_3_EB_0"</span>,  <span class="pl-s">"Node4_3_EB_1"</span>,</div></div></div><div data-key="93" class="child-of-line-62  react-code-text react-code-line-contents virtual" style="transform: translateY(1860px); min-height: auto;"><div><div id="LC94" class="react-file-line html-div" data-testid="code-cell" data-line-number="94" inert="" style="position: relative;">            <span class="pl-s">"Node3_1_NB_0"</span>,  <span class="pl-s">"Node3_1_NB_1"</span>,</div></div></div><div data-key="94" class="child-of-line-62  react-code-text react-code-line-contents virtual" style="transform: translateY(1880px); min-height: auto;"><div><div id="LC95" class="react-file-line html-div" data-testid="code-cell" data-line-number="95" inert="" style="position: relative;">            <span class="pl-s">"Node3_5_WB_0"</span>,  <span class="pl-s">"Node3_5_WB_1"</span>,</div></div></div><div data-key="95" class="child-of-line-62  react-code-text react-code-line-contents virtual" style="transform: translateY(1900px); min-height: auto;"><div><div id="LC96" class="react-file-line html-div" data-testid="code-cell" data-line-number="96" inert="" style="position: relative;">            <span class="pl-s">"Node3_18_NB_0"</span>, <span class="pl-s">"Node3_18_NB_1"</span>,</div></div></div><div data-key="96" class="child-of-line-62  react-code-text react-code-line-contents virtual" style="transform: translateY(1920px); min-height: auto;"><div><div id="LC97" class="react-file-line html-div" data-testid="code-cell" data-line-number="97" inert="" style="position: relative;">        ],</div></div></div><div data-key="97" class="child-of-line-62  react-code-text react-code-line-contents virtual" style="transform: translateY(1940px); min-height: auto;"><div><div id="LC98" class="react-file-line html-div" data-testid="code-cell" data-line-number="98" inert="" style="position: relative;">        <span class="pl-s">"neighbors"</span>: [<span class="pl-s">"Node4"</span>, <span class="pl-s">"Node1"</span>, <span class="pl-s">"Node5"</span>, <span class="pl-s">"Node18"</span>],</div></div></div><div data-key="98" class="child-of-line-62  react-code-text react-code-line-contents virtual" style="transform: translateY(1960px); min-height: auto;"><div><div id="LC99" class="react-file-line html-div" data-testid="code-cell" data-line-number="99" inert="" style="position: relative;">    },</div></div></div><div data-key="99" class="child-of-line-62  react-code-text react-code-line-contents virtual" style="transform: translateY(1980px); min-height: auto;"><div><div id="LC100" class="react-file-line html-div" data-testid="code-cell" data-line-number="100" inert="" style="position: relative;">    <span class="pl-s">"Node4"</span>: {</div></div></div><div data-key="100" class="child-of-line-62  react-code-text react-code-line-contents virtual" style="transform: translateY(2000px); min-height: auto;"><div><div id="LC101" class="react-file-line html-div" data-testid="code-cell" data-line-number="101" inert="" style="position: relative;">        <span class="pl-s">"detectors"</span>: [</div></div></div><div data-key="101" class="child-of-line-62  react-code-text react-code-line-contents virtual" style="transform: translateY(2020px); min-height: auto;"><div><div id="LC102" class="react-file-line html-div" data-testid="code-cell" data-line-number="102" inert="" style="position: relative;">            <span class="pl-s">"Node4_0_SB_0"</span>,  <span class="pl-s">"Node4_0_SB_1"</span>,</div></div></div><div data-key="102" class="child-of-line-62  react-code-text react-code-line-contents virtual" style="transform: translateY(2040px); min-height: auto;"><div><div id="LC103" class="react-file-line html-div" data-testid="code-cell" data-line-number="103" inert="" style="position: relative;">            <span class="pl-s">"Node13_4_NB_0"</span>, <span class="pl-s">"Node13_4_NB_1"</span>,</div></div></div><div data-key="103" class="child-of-line-62  react-code-text react-code-line-contents virtual" style="transform: translateY(2060px); min-height: auto;"><div><div id="LC104" class="react-file-line html-div" data-testid="code-cell" data-line-number="104" inert="" style="position: relative;">            <span class="pl-s">"Node4_3_WB_0"</span>,  <span class="pl-s">"Node4_3_WB_1"</span>,</div></div></div><div data-key="104" class="child-of-line-62  react-code-text react-code-line-contents virtual" style="transform: translateY(2080px); min-height: auto;"><div><div id="LC105" class="react-file-line html-div" data-testid="code-cell" data-line-number="105" inert="" style="position: relative;">            <span class="pl-s">"Node4_17_EB_1"</span>, <span class="pl-s">"Node4_17_EB_0"</span>,</div></div></div><div data-key="105" class="child-of-line-62  react-code-text react-code-line-contents virtual" style="transform: translateY(2100px); min-height: auto;"><div><div id="LC106" class="react-file-line html-div" data-testid="code-cell" data-line-number="106" inert="" style="position: relative;">        ],</div></div></div><div data-key="106" class="child-of-line-62  react-code-text react-code-line-contents virtual" style="transform: translateY(2120px); min-height: auto;"><div><div id="LC107" class="react-file-line html-div" data-testid="code-cell" data-line-number="107" inert="" style="position: relative;">        <span class="pl-s">"neighbors"</span>: [<span class="pl-s">"Node0"</span>, <span class="pl-s">"Node13"</span>, <span class="pl-s">"Node3"</span>, <span class="pl-s">"Node17"</span>],</div></div></div><div data-key="107" class="child-of-line-62  react-code-text react-code-line-contents virtual" style="transform: translateY(2140px); min-height: auto;"><div><div id="LC108" class="react-file-line html-div" data-testid="code-cell" data-line-number="108" inert="" style="position: relative;">    },</div></div></div><div data-key="108" class="child-of-line-62  react-code-text react-code-line-contents virtual" style="transform: translateY(2160px); min-height: auto;"><div><div id="LC109" class="react-file-line html-div" data-testid="code-cell" data-line-number="109" inert="" style="position: relative;">    <span class="pl-s">"Node5"</span>: {</div></div></div><div data-key="109" class="child-of-line-62  react-code-text react-code-line-contents virtual" style="transform: translateY(2180px); min-height: auto;"><div><div id="LC110" class="react-file-line html-div" data-testid="code-cell" data-line-number="110" inert="" style="position: relative;">        <span class="pl-s">"detectors"</span>: [</div></div></div><div data-key="110" class="child-of-line-62  react-code-text react-code-line-contents virtual" style="transform: translateY(2200px); min-height: auto;"><div><div id="LC111" class="react-file-line html-div" data-testid="code-cell" data-line-number="111" inert="" style="position: relative;">            <span class="pl-s">"Node3_5_EB_0"</span>,  <span class="pl-s">"Node3_5_EB_1"</span>,</div></div></div><div data-key="111" class="child-of-line-62  react-code-text react-code-line-contents virtual" style="transform: translateY(2220px); min-height: auto;"><div><div id="LC112" class="react-file-line html-div" data-testid="code-cell" data-line-number="112" inert="" style="position: relative;">            <span class="pl-s">"Node5_2_SB_0"</span>,  <span class="pl-s">"Node5_2_SB_1"</span>,</div></div></div><div data-key="112" class="child-of-line-62  react-code-text react-code-line-contents virtual" style="transform: translateY(2240px); min-height: auto;"><div><div id="LC113" class="react-file-line html-div" data-testid="code-cell" data-line-number="113" inert="" style="position: relative;">            <span class="pl-s">"Node5_11_WB_0"</span>, <span class="pl-s">"Node5_11_WB_1"</span>,</div></div></div><div data-key="113" class="child-of-line-62  react-code-text react-code-line-contents virtual" style="transform: translateY(2260px); min-height: auto;"><div><div id="LC114" class="react-file-line html-div" data-testid="code-cell" data-line-number="114" inert="" style="position: relative;">            <span class="pl-s">"Node5_19_NB_0"</span>, <span class="pl-s">"Node5_19_NB_1"</span>,</div></div></div><div data-key="114" class="child-of-line-62  react-code-text react-code-line-contents virtual" style="transform: translateY(2280px); min-height: auto;"><div><div id="LC115" class="react-file-line html-div" data-testid="code-cell" data-line-number="115" inert="" style="position: relative;">        ],</div></div></div><div data-key="115" class="child-of-line-62  react-code-text react-code-line-contents virtual" style="transform: translateY(2300px); min-height: auto;"><div><div id="LC116" class="react-file-line html-div" data-testid="code-cell" data-line-number="116" inert="" style="position: relative;">        <span class="pl-s">"neighbors"</span>: [<span class="pl-s">"Node3"</span>, <span class="pl-s">"Node2"</span>, <span class="pl-s">"Node11"</span>, <span class="pl-s">"Node19"</span>],</div></div></div><div data-key="116" class="child-of-line-62  react-code-text react-code-line-contents virtual" style="transform: translateY(2320px); min-height: auto;"><div><div id="LC117" class="react-file-line html-div" data-testid="code-cell" data-line-number="117" inert="" style="position: relative;">    },</div></div></div><div data-key="117" class="child-of-line-62  react-code-text react-code-line-contents virtual" style="transform: translateY(2340px); min-height: auto;"><div><div id="LC118" class="react-file-line html-div" data-testid="code-cell" data-line-number="118" inert="" style="position: relative;">    <span class="pl-s">"Node6"</span>: {</div></div></div><div data-key="118" class="child-of-line-62  react-code-text react-code-line-contents virtual" style="transform: translateY(2360px); min-height: auto;"><div><div id="LC119" class="react-file-line html-div" data-testid="code-cell" data-line-number="119" inert="" style="position: relative;">        <span class="pl-s">"detectors"</span>: [</div></div></div><div data-key="119" class="child-of-line-62  react-code-text react-code-line-contents virtual" style="transform: translateY(2380px); min-height: auto;"><div><div id="LC120" class="react-file-line html-div" data-testid="code-cell" data-line-number="120" inert="" style="position: relative;">            <span class="pl-s">"Node6_15_EB_0"</span>,  <span class="pl-s">"Node6_15_EB_1"</span>,</div></div></div><div data-key="120" class="child-of-line-62  react-code-text react-code-line-contents virtual" style="transform: translateY(2400px); min-height: auto;"><div><div id="LC121" class="react-file-line html-div" data-testid="code-cell" data-line-number="121" inert="" style="position: relative;">            <span class="pl-s">"Node1_6_NB_0"</span>,  <span class="pl-s">"Node1_6_NB_1"</span>,</div></div></div><div data-key="121" class="child-of-line-62  react-code-text react-code-line-contents virtual" style="transform: translateY(2420px); min-height: auto;"><div><div id="LC122" class="react-file-line html-div" data-testid="code-cell" data-line-number="122" inert="" style="position: relative;">            <span class="pl-s">"Node6_7_WB_0"</span>,  <span class="pl-s">"Node6_7_WB_1"</span>,</div></div></div><div data-key="122" class="child-of-line-62  react-code-text react-code-line-contents virtual" style="transform: translateY(2440px); min-height: auto;"><div><div id="LC123" class="react-file-line html-div" data-testid="code-cell" data-line-number="123" inert="" style="position: relative;">            <span class="pl-s">"Node6_9_SB_1"</span>,  <span class="pl-s">"Node6_9_SB_0"</span>,</div></div></div><div data-key="123" class="child-of-line-62  react-code-text react-code-line-contents virtual" style="transform: translateY(2460px); min-height: auto;"><div><div id="LC124" class="react-file-line html-div" data-testid="code-cell" data-line-number="124" inert="" style="position: relative;">        ],</div></div></div><div data-key="124" class="child-of-line-62  react-code-text react-code-line-contents virtual" style="transform: translateY(2480px); min-height: auto;"><div><div id="LC125" class="react-file-line html-div" data-testid="code-cell" data-line-number="125" inert="" style="position: relative;">        <span class="pl-s">"neighbors"</span>: [<span class="pl-s">"Node9"</span>, <span class="pl-s">"Node1"</span>, <span class="pl-s">"Node7"</span>, <span class="pl-s">"Node15"</span>],</div></div></div><div data-key="125" class="child-of-line-62  react-code-text react-code-line-contents virtual" style="transform: translateY(2500px); min-height: auto;"><div><div id="LC126" class="react-file-line html-div" data-testid="code-cell" data-line-number="126" inert="" style="position: relative;">    },</div></div></div><div data-key="126" class="child-of-line-62  react-code-text react-code-line-contents virtual" style="transform: translateY(2520px); min-height: auto;"><div><div id="LC127" class="react-file-line html-div" data-testid="code-cell" data-line-number="127" inert="" style="position: relative;">    <span class="pl-s">"Node7"</span>: {</div></div></div><div data-key="127" class="child-of-line-62  react-code-text react-code-line-contents virtual" style="transform: translateY(2540px); min-height: auto;"><div><div id="LC128" class="react-file-line html-div" data-testid="code-cell" data-line-number="128" inert="" style="position: relative;">        <span class="pl-s">"detectors"</span>: [</div></div></div><div data-key="128" class="child-of-line-62  react-code-text react-code-line-contents virtual" style="transform: translateY(2560px); min-height: auto;"><div><div id="LC129" class="react-file-line html-div" data-testid="code-cell" data-line-number="129" inert="" style="position: relative;">            <span class="pl-s">"Node6_7_EB_0"</span>,  <span class="pl-s">"Node6_7_EB_1"</span>,</div></div></div><div data-key="129" class="child-of-line-62  react-code-text react-code-line-contents virtual" style="transform: translateY(2580px); min-height: auto;"><div><div id="LC130" class="react-file-line html-div" data-testid="code-cell" data-line-number="130" inert="" style="position: relative;">            <span class="pl-s">"Node2_7_NB_0"</span>,  <span class="pl-s">"Node2_7_NB_1"</span>,</div></div></div><div data-key="130" class="child-of-line-62  react-code-text react-code-line-contents virtual" style="transform: translateY(2600px); min-height: auto;"><div><div id="LC131" class="react-file-line html-div" data-testid="code-cell" data-line-number="131" inert="" style="position: relative;">            <span class="pl-s">"Node7_8_WB_0"</span>,  <span class="pl-s">"Node7_8_WB_1"</span>,</div></div></div><div data-key="131" class="child-of-line-62  react-code-text react-code-line-contents virtual" style="transform: translateY(2620px); min-height: auto;"><div><div id="LC132" class="react-file-line html-div" data-testid="code-cell" data-line-number="132" inert="" style="position: relative;">            <span class="pl-s">"Node7_14_SB_1"</span>, <span class="pl-s">"Node7_14_SB_0"</span>,</div></div></div><div data-key="132" class="child-of-line-62  react-code-text react-code-line-contents virtual" style="transform: translateY(2640px); min-height: auto;"><div><div id="LC133" class="react-file-line html-div" data-testid="code-cell" data-line-number="133" inert="" style="position: relative;">        ],</div></div></div><div data-key="133" class="child-of-line-62  react-code-text react-code-line-contents virtual" style="transform: translateY(2660px); min-height: auto;"><div><div id="LC134" class="react-file-line html-div" data-testid="code-cell" data-line-number="134" inert="" style="position: relative;">        <span class="pl-s">"neighbors"</span>: [<span class="pl-s">"Node6"</span>, <span class="pl-s">"Node8"</span>, <span class="pl-s">"Node2"</span>, <span class="pl-s">"Node14"</span>],</div></div></div><div data-key="134" class="child-of-line-62  react-code-text react-code-line-contents virtual" style="transform: translateY(2680px); min-height: auto;"><div><div id="LC135" class="react-file-line html-div" data-testid="code-cell" data-line-number="135" inert="" style="position: relative;">    },</div></div></div><div data-key="135" class="react-code-text react-code-line-contents virtual" style="transform: translateY(2700px); min-height: auto;"><div><div id="LC136" class="react-file-line html-div" data-testid="code-cell" data-line-number="136" inert="" style="position: relative;">}</div></div></div><div data-key="136" class="react-code-text react-code-line-contents virtual" style="transform: translateY(2720px); min-height: auto;"><div><div id="LC137" class="react-file-line html-div" data-testid="code-cell" data-line-number="137" inert="" style="position: relative;">
</div></div></div><div data-key="137" class="react-code-text react-code-line-contents virtual" style="transform: translateY(2740px); min-height: auto;"><div><div id="LC138" class="react-file-line html-div" data-testid="code-cell" data-line-number="138" inert="" style="position: relative;">
</div></div></div><div data-key="138" class="react-code-text react-code-line-contents virtual" style="transform: translateY(2760px); min-height: auto;"><div><div id="LC139" class="react-file-line html-div" data-testid="code-cell" data-line-number="139" inert="" style="position: relative;"><span class="pl-c"># ==========================</span></div></div></div><div data-key="139" class="react-code-text react-code-line-contents virtual" style="transform: translateY(2780px); min-height: auto;"><div><div id="LC140" class="react-file-line html-div" data-testid="code-cell" data-line-number="140" inert="" style="position: relative;"><span class="pl-c">#   RL PARAMETERS</span></div></div></div><div data-key="140" class="react-code-text react-code-line-contents virtual" style="transform: translateY(2800px); min-height: auto;"><div><div id="LC141" class="react-file-line html-div" data-testid="code-cell" data-line-number="141" inert="" style="position: relative;"><span class="pl-c"># ==========================</span></div></div></div><div data-key="141" class="react-code-text react-code-line-contents virtual" style="transform: translateY(2820px); min-height: auto;"><div><div id="LC142" class="react-file-line html-div" data-testid="code-cell" data-line-number="142" inert="" style="position: relative;"><span class="pl-c1">TOTAL_STEPS</span>        <span class="pl-c1">=</span> <span class="pl-c1">10000</span></div></div></div><div data-key="142" class="react-code-text react-code-line-contents virtual" style="transform: translateY(2840px); min-height: auto;"><div><div id="LC143" class="react-file-line html-div" data-testid="code-cell" data-line-number="143" inert="" style="position: relative;"><span class="pl-c1">EPISODE_LENGTH</span>     <span class="pl-c1">=</span> <span class="pl-c1">100</span></div></div></div><div data-key="143" class="react-code-text react-code-line-contents virtual" style="transform: translateY(2860px); min-height: auto;"><div><div id="LC144" class="react-file-line html-div" data-testid="code-cell" data-line-number="144" inert="" style="position: relative;"><span class="pl-c1">LEARNING_RATE</span>      <span class="pl-c1">=</span> <span class="pl-c1">0.0001</span></div></div></div><div data-key="144" class="react-code-text react-code-line-contents virtual" style="transform: translateY(2880px); min-height: auto;"><div><div id="LC145" class="react-file-line html-div" data-testid="code-cell" data-line-number="145" inert="" style="position: relative;"><span class="pl-c1">GAMMA</span>              <span class="pl-c1">=</span> <span class="pl-c1">0.95</span></div></div></div><div data-key="145" class="react-code-text react-code-line-contents virtual" style="transform: translateY(2900px); min-height: auto;"><div><div id="LC146" class="react-file-line html-div" data-testid="code-cell" data-line-number="146" inert="" style="position: relative;"><span class="pl-c1">EPSILON_START</span>      <span class="pl-c1">=</span> <span class="pl-c1">1.0</span></div></div></div><div data-key="146" class="react-code-text react-code-line-contents virtual" style="transform: translateY(2920px); min-height: auto;"><div><div id="LC147" class="react-file-line html-div" data-testid="code-cell" data-line-number="147" inert="" style="position: relative;"><span class="pl-c1">EPSILON_END</span>        <span class="pl-c1">=</span> <span class="pl-c1">0.01</span></div></div></div><div data-key="917" class="react-code-text react-code-line-contents virtual" style="transform: translateY(18340px); min-height: auto;"><div><div id="LC918" class="react-file-line html-div" data-testid="code-cell" data-line-number="918" inert="" style="position: relative;">                  <span class="pl-s">f"Wait=<span class="pl-s1"><span class="pl-kos">{</span><span class="pl-s1">np</span>.<span class="pl-c1">mean</span>(<span class="pl-s1">rw</span>):.2f<span class="pl-kos">}</span></span>s | Eps=<span class="pl-s1"><span class="pl-kos">{</span><span class="pl-s1">epsilon</span>:.4f<span class="pl-kos">}</span></span> | "</span></div></div></div><div data-key="918" class="react-code-text react-code-line-contents virtual" style="transform: translateY(18360px); min-height: auto;"><div><div id="LC919" class="react-file-line html-div" data-testid="code-cell" data-line-number="919" inert="" style="position: relative;">                  <span class="pl-s">f"TrainSteps=<span class="pl-s1"><span class="pl-kos">{</span><span class="pl-s1">training_step_counter</span><span class="pl-kos">}</span></span>"</span>)</div></div></div><div data-key="919" class="react-code-text react-code-line-contents virtual" style="transform: translateY(18380px); min-height: auto;"><div><div id="LC920" class="react-file-line html-div" data-testid="code-cell" data-line-number="920" inert="" style="position: relative;">            <span class="pl-en">print</span>(<span class="pl-s">f"<span class="pl-s1"><span class="pl-kos">{</span><span class="pl-s">'='</span><span class="pl-c1">*</span><span class="pl-c1">70</span><span class="pl-kos">}</span></span><span class="pl-cce">\n</span>"</span>)</div></div></div><div data-key="920" class="react-code-text react-code-line-contents virtual" style="transform: translateY(18400px); min-height: auto;"><div><div id="LC921" class="react-file-line html-div" data-testid="code-cell" data-line-number="921" inert="" style="position: relative;">
</div></div></div><div data-key="921" class="react-code-text react-code-line-contents virtual" style="transform: translateY(18420px); min-height: auto;"><div><div id="LC922" class="react-file-line html-div" data-testid="code-cell" data-line-number="922" inert="" style="position: relative;"><span class="pl-k">except</span> <span class="pl-v">Exception</span> <span class="pl-k">as</span> <span class="pl-s1">e</span>:</div></div></div><div data-key="922" class="react-code-text react-code-line-contents virtual" style="transform: translateY(18440px); min-height: auto;"><div><div id="LC923" class="react-file-line html-div" data-testid="code-cell" data-line-number="923" inert="" style="position: relative;">    <span class="pl-en">print</span>(<span class="pl-s">f"<span class="pl-cce">\n</span>ERROR: <span class="pl-s1"><span class="pl-kos">{</span><span class="pl-s1">e</span><span class="pl-kos">}</span></span>"</span>)</div></div></div><div data-key="923" class="react-code-text react-code-line-contents virtual" style="transform: translateY(18460px); min-height: auto;"><div><div id="LC924" class="react-file-line html-div" data-testid="code-cell" data-line-number="924" inert="" style="position: relative;">    <span class="pl-k">import</span> <span class="pl-s1">traceback</span>; <span class="pl-s1">traceback</span>.<span class="pl-c1">print_exc</span>()</div></div></div><div data-key="924" class="react-code-text react-code-line-contents virtual" style="transform: translateY(18480px); min-height: auto;"><div><div id="LC925" class="react-file-line html-div" data-testid="code-cell" data-line-number="925" inert="" style="position: relative;"><span class="pl-k">finally</span>:</div></div></div><div data-key="925" class="react-code-text react-code-line-contents virtual" style="transform: translateY(18500px); min-height: auto;"><div><div id="LC926" class="react-file-line html-div" data-testid="code-cell" data-line-number="926" inert="" style="position: relative;">    <span class="pl-s1">traci</span>.<span class="pl-c1">close</span>()</div></div></div><div data-key="926" class="react-code-text react-code-line-contents virtual" style="transform: translateY(18520px); min-height: auto;"><div><div id="LC927" class="react-file-line html-div" data-testid="code-cell" data-line-number="927" inert="" style="position: relative;">
</div></div></div><div data-key="927" class="react-code-text react-code-line-contents virtual" style="transform: translateY(18540px); min-height: auto;"><div><div id="LC928" class="react-file-line html-div" data-testid="code-cell" data-line-number="928" inert="" style="position: relative;"><span class="pl-k">if</span> <span class="pl-c1">TRAIN</span>:</div></div></div><div data-key="928" class="react-code-text react-code-line-contents virtual" style="transform: translateY(18560px); min-height: auto;"><div><div id="LC929" class="react-file-line html-div" data-testid="code-cell" data-line-number="929" inert="" style="position: relative;">    <span class="pl-c"># main_model.save("igrl_gcn_model.h5")</span></div></div></div><div data-key="929" class="react-code-text react-code-line-contents virtual" style="transform: translateY(18580px); min-height: auto;"><div><div id="LC930" class="react-file-line html-div" data-testid="code-cell" data-line-number="930" inert="" style="position: relative;">    <span class="pl-s1">main_model</span>.<span class="pl-c1">save</span>(<span class="pl-s">"igrl_gat_model.h5"</span>)</div></div></div><div data-key="930" class="react-code-text react-code-line-contents virtual" style="transform: translateY(18600px); min-height: auto;"><div><div id="LC931" class="react-file-line html-div" data-testid="code-cell" data-line-number="931" inert="" style="position: relative;">    <span class="pl-en">print</span>(<span class="pl-s">"<span class="pl-cce">\n</span>MODEL SAVED: igrl_gcn_model.h5"</span>)</div></div></div><div data-key="931" class="react-code-text react-code-line-contents virtual" style="transform: translateY(18620px); min-height: auto;"><div><div id="LC932" class="react-file-line html-div" data-testid="code-cell" data-line-number="932" inert="" style="position: relative;">    <span class="pl-en">print</span>(<span class="pl-s">"Set TRAIN=False to deploy — same weights control all nodes zero-shot."</span>)</div></div></div><div data-key="932" class="react-code-text react-code-line-contents virtual" style="transform: translateY(18640px); min-height: auto;"><div><div id="LC933" class="react-file-line html-div" data-testid="code-cell" data-line-number="933" inert="" style="position: relative;">
</div></div></div><div data-key="933" class="react-code-text react-code-line-contents virtual" style="transform: translateY(18660px); min-height: auto;"><div><div id="LC934" class="react-file-line html-div" data-testid="code-cell" data-line-number="934" inert="" style="position: relative;"><span class="pl-en">print</span>(<span class="pl-s">f"<span class="pl-cce">\n</span><span class="pl-s1"><span class="pl-kos">{</span><span class="pl-s">'='</span><span class="pl-c1">*</span><span class="pl-c1">70</span><span class="pl-kos">}</span></span><span class="pl-cce">\n</span>STATISTICS"</span>)</div></div></div><div data-key="934" class="react-code-text react-code-line-contents virtual" style="transform: translateY(18680px); min-height: auto;"><div><div id="LC935" class="react-file-line html-div" data-testid="code-cell" data-line-number="935" inert="" style="position: relative;"><span class="pl-en">print</span>(<span class="pl-s">f"Steps: <span class="pl-s1"><span class="pl-kos">{</span><span class="pl-c1">TOTAL_STEPS</span><span class="pl-kos">}</span></span> | Entered: <span class="pl-s1"><span class="pl-kos">{</span><span class="pl-s1">total_vehicles_entered</span>:.0f<span class="pl-kos">}</span></span> | "</span></div></div></div><div data-key="935" class="react-code-text react-code-line-contents virtual" style="transform: translateY(18700px); min-height: auto;"><div><div id="LC936" class="react-file-line html-div" data-testid="code-cell" data-line-number="936" inert="" style="position: relative;">      <span class="pl-s">f"Completed: <span class="pl-s1"><span class="pl-kos">{</span><span class="pl-s1">total_vehicles_completed</span>:.0f<span class="pl-kos">}</span></span>"</span>)</div></div></div><div data-key="936" class="react-code-text react-code-line-contents virtual" style="transform: translateY(18720px); min-height: auto;"><div><div id="LC937" class="react-file-line html-div" data-testid="code-cell" data-line-number="937" inert="" style="position: relative;"><span class="pl-en">print</span>(<span class="pl-s">f"Total throughput counted (sum of per-step arrivals): <span class="pl-s1"><span class="pl-kos">{</span><span class="pl-en">int</span>(<span class="pl-s1">np</span>.<span class="pl-c1">sum</span>(<span class="pl-s1">throughput_history</span>))<span class="pl-kos">}</span></span>"</span>)</div></div></div><div data-key="937" class="react-code-text react-code-line-contents virtual" style="transform: translateY(18740px); min-height: auto;"><div><div id="LC938" class="react-file-line html-div" data-testid="code-cell" data-line-number="938" inert="" style="position: relative;"><span class="pl-en">print</span>(<span class="pl-s">f"Average waiting time (mean across all steps): <span class="pl-s1"><span class="pl-kos">{</span><span class="pl-s1">np</span>.<span class="pl-c1">mean</span>(<span class="pl-s1">waiting_times</span>):.2f<span class="pl-kos">}</span></span> sec"</span>)</div></div></div><div data-key="938" class="react-code-text react-code-line-contents virtual" style="transform: translateY(18760px); min-height: auto;"><div><div id="LC939" class="react-file-line html-div" data-testid="code-cell" data-line-number="939" inert="" style="position: relative;"><span class="pl-en">print</span>(<span class="pl-s">f"Average queue length (mean across sampled steps): <span class="pl-s1"><span class="pl-kos">{</span><span class="pl-s1">np</span>.<span class="pl-c1">mean</span>(<span class="pl-s1">queues</span>):.2f<span class="pl-kos">}</span></span> vehicles"</span>)</div></div></div><div data-key="939" class="react-code-text react-code-line-contents virtual" style="transform: translateY(18780px); min-height: auto;"><div><div id="LC940" class="react-file-line html-div" data-testid="code-cell" data-line-number="940" inert="" style="position: relative;"><span class="pl-k">if</span> <span class="pl-c1">TRAIN</span>:</div></div></div><div data-key="940" class="react-code-text react-code-line-contents virtual" style="transform: translateY(18800px); min-height: auto;"><div><div id="LC941" class="react-file-line html-div" data-testid="code-cell" data-line-number="941" inert="" style="position: relative;">    <span class="pl-en">print</span>(<span class="pl-s">f"Epsilon: <span class="pl-s1"><span class="pl-kos">{</span><span class="pl-s1">epsilon</span>:.4f<span class="pl-kos">}</span></span> | TrainSteps: <span class="pl-s1"><span class="pl-kos">{</span><span class="pl-s1">training_step_counter</span><span class="pl-kos">}</span></span> | "</span></div></div></div><div data-key="941" class="react-code-text react-code-line-contents virtual" style="transform: translateY(18820px); min-height: auto;"><div><div id="LC942" class="react-file-line html-div" data-testid="code-cell" data-line-number="942" inert="" style="position: relative;">          <span class="pl-s">f"Buffer: <span class="pl-s1"><span class="pl-kos">{</span><span class="pl-s1">replay_buffer</span>.<span class="pl-c1">size</span>()<span class="pl-kos">}</span></span>"</span>)</div></div></div><div data-key="942" class="react-code-text react-code-line-contents virtual" style="transform: translateY(18840px); min-height: auto;"><div><div id="LC943" class="react-file-line html-div" data-testid="code-cell" data-line-number="943" inert="" style="position: relative;">
</div></div></div><div data-key="943" class="react-code-text react-code-line-contents virtual" style="transform: translateY(18860px); min-height: auto;"><div><div id="LC944" class="react-file-line html-div" data-testid="code-cell" data-line-number="944" inert="" style="position: relative;">
</div></div></div><div data-key="944" class="react-code-text react-code-line-contents virtual" style="transform: translateY(18880px); min-height: auto;"><div><div id="LC945" class="react-file-line html-div" data-testid="code-cell" data-line-number="945" inert="" style="position: relative;"><span class="pl-c"># ==========================</span></div></div></div><div data-key="945" class="react-code-text react-code-line-contents virtual" style="transform: translateY(18900px); min-height: auto;"><div><div id="LC946" class="react-file-line html-div" data-testid="code-cell" data-line-number="946" inert="" style="position: relative;"><span class="pl-c">#   PLOTS</span></div></div></div><div data-key="946" class="react-code-text react-code-line-contents virtual" style="transform: translateY(18920px); min-height: auto;"><div><div id="LC947" class="react-file-line html-div" data-testid="code-cell" data-line-number="947" inert="" style="position: relative;"><span class="pl-c"># ==========================</span></div></div></div><div data-key="947" class="react-code-text react-code-line-contents virtual" style="transform: translateY(18940px); min-height: auto;"><div><div id="LC948" class="react-file-line html-div" data-testid="code-cell" data-line-number="948" inert="" style="position: relative;"><span class="pl-k">def</span> <span class="pl-en">smooth</span>(<span class="pl-s1">data</span>, <span class="pl-s1">window</span><span class="pl-c1">=</span><span class="pl-c1">5</span>):</div></div></div><div data-key="948" class="react-code-text react-code-line-contents virtual" style="transform: translateY(18960px); min-height: auto;"><div><div id="LC949" class="react-file-line html-div" data-testid="code-cell" data-line-number="949" inert="" style="position: relative;">    <span class="pl-k">if</span> <span class="pl-en">len</span>(<span class="pl-s1">data</span>) <span class="pl-c1">&lt;</span> <span class="pl-s1">window</span>: <span class="pl-k">return</span> <span class="pl-s1">np</span>.<span class="pl-c1">array</span>(<span class="pl-s1">data</span>)</div></div></div><div data-key="949" class="react-code-text react-code-line-contents virtual" style="transform: translateY(18980px); min-height: auto;"><div><div id="LC950" class="react-file-line html-div" data-testid="code-cell" data-line-number="950" inert="" style="position: relative;">    <span class="pl-k">return</span> <span class="pl-s1">np</span>.<span class="pl-c1">convolve</span>(<span class="pl-s1">data</span>, <span class="pl-s1">np</span>.<span class="pl-c1">ones</span>(<span class="pl-s1">window</span>)<span class="pl-c1">/</span><span class="pl-s1">window</span>, <span class="pl-s1">mode</span><span class="pl-c1">=</span><span class="pl-s">'valid'</span>)</div></div></div><div data-key="950" class="react-code-text react-code-line-contents virtual" style="transform: translateY(19000px); min-height: auto;"><div><div id="LC951" class="react-file-line html-div" data-testid="code-cell" data-line-number="951" inert="" style="position: relative;">
</div></div></div><div data-key="951" class="react-code-text react-code-line-contents virtual" style="transform: translateY(19020px); min-height: auto;"><div><div id="LC952" class="react-file-line html-div" data-testid="code-cell" data-line-number="952" inert="" style="position: relative;"><span class="pl-s1">fig</span>, <span class="pl-s1">axes</span> <span class="pl-c1">=</span> <span class="pl-s1">plt</span>.<span class="pl-c1">subplots</span>(<span class="pl-c1">2</span>, <span class="pl-c1">2</span>, <span class="pl-s1">figsize</span><span class="pl-c1">=</span>(<span class="pl-c1">14</span>, <span class="pl-c1">10</span>))</div></div></div><div data-key="952" class="react-code-text react-code-line-contents virtual" style="transform: translateY(19040px); min-height: auto;"><div><div id="LC953" class="react-file-line html-div" data-testid="code-cell" data-line-number="953" inert="" style="position: relative;"><span class="pl-s1">fig</span>.<span class="pl-c1">suptitle</span>(<span class="pl-s">'IG-RL Zero-Shot Transfer — Traffic Signal Control'</span>,</div></div></div><div data-key="953" class="react-code-text react-code-line-contents virtual" style="transform: translateY(19060px); min-height: auto;"><div><div id="LC954" class="react-file-line html-div" data-testid="code-cell" data-line-number="954" inert="" style="position: relative;">             <span class="pl-s1">fontsize</span><span class="pl-c1">=</span><span class="pl-c1">13</span>, <span class="pl-s1">fontweight</span><span class="pl-c1">=</span><span class="pl-s">'bold'</span>)</div></div></div><div data-key="954" class="react-code-text react-code-line-contents virtual" style="transform: translateY(19080px); min-height: auto;"><div><div id="LC955" class="react-file-line html-div" data-testid="code-cell" data-line-number="955" inert="" style="position: relative;">
</div></div></div><div data-key="955" class="react-code-text react-code-line-contents virtual" style="transform: translateY(19100px); min-height: auto;"><div><div id="LC956" class="react-file-line html-div" data-testid="code-cell" data-line-number="956" inert="" style="position: relative;"><span class="pl-s1">ax</span> <span class="pl-c1">=</span> <span class="pl-s1">axes</span>[<span class="pl-c1">0</span>, <span class="pl-c1">0</span>]</div></div></div><div data-key="956" class="react-code-text react-code-line-contents virtual" style="transform: translateY(19120px); min-height: auto;"><div><div id="LC957" class="react-file-line html-div" data-testid="code-cell" data-line-number="957" inert="" style="position: relative;"><span class="pl-k">if</span> <span class="pl-en">len</span>(<span class="pl-s1">queues</span>) <span class="pl-c1">&gt;</span> <span class="pl-c1">5</span>:</div></div></div><div data-key="957" class="react-code-text react-code-line-contents virtual" style="transform: translateY(19140px); min-height: auto;"><div><div id="LC958" class="react-file-line html-div" data-testid="code-cell" data-line-number="958" inert="" style="position: relative;">    <span class="pl-s1">ax</span>.<span class="pl-c1">plot</span>(<span class="pl-en">smooth</span>(<span class="pl-s1">queues</span>), <span class="pl-s1">color</span><span class="pl-c1">=</span><span class="pl-s">'#1f77b4'</span>, <span class="pl-s1">linewidth</span><span class="pl-c1">=</span><span class="pl-c1">1.5</span>, <span class="pl-s1">label</span><span class="pl-c1">=</span><span class="pl-s">'Avg Queue'</span>)</div></div></div><div data-key="958" class="react-code-text react-code-line-contents virtual" style="transform: translateY(19160px); min-height: auto;"><div><div id="LC959" class="react-file-line html-div" data-testid="code-cell" data-line-number="959" inert="" style="position: relative;"><span class="pl-k">for</span> (<span class="pl-s1">lane</span>, <span class="pl-s1">s</span>, <span class="pl-s1">e</span>) <span class="pl-c1">in</span> <span class="pl-c1">INCIDENT_SCHEDULE</span>:</div></div></div><div data-key="959" class="react-code-text react-code-line-contents virtual" style="transform: translateY(19180px); min-height: auto;"><div><div id="LC960" class="react-file-line html-div" data-testid="code-cell" data-line-number="960" inert="" style="position: relative;">    <span class="pl-s1">ax</span>.<span class="pl-c1">axvspan</span>(<span class="pl-s1">s</span>, <span class="pl-en">min</span>(<span class="pl-s1">e</span>, <span class="pl-c1">TOTAL_STEPS</span>), <span class="pl-s1">alpha</span><span class="pl-c1">=</span><span class="pl-c1">0.15</span>, <span class="pl-s1">color</span><span class="pl-c1">=</span><span class="pl-s">'red'</span>, <span class="pl-s1">label</span><span class="pl-c1">=</span><span class="pl-s">'Incident'</span>)</div></div></div><div data-key="960" class="react-code-text react-code-line-contents virtual" style="transform: translateY(19200px); min-height: auto;"><div><div id="LC961" class="react-file-line html-div" data-testid="code-cell" data-line-number="961" inert="" style="position: relative;"><span class="pl-s1">ax</span>.<span class="pl-c1">set_title</span>(<span class="pl-s">'Queue Length'</span>); <span class="pl-s1">ax</span>.<span class="pl-c1">set_xlabel</span>(<span class="pl-s">'Step'</span>); <span class="pl-s1">ax</span>.<span class="pl-c1">set_ylabel</span>(<span class="pl-s">'Vehicles'</span>)</div></div></div><div data-key="961" class="react-code-text react-code-line-contents virtual" style="transform: translateY(19220px); min-height: auto;"><div><div id="LC962" class="react-file-line html-div" data-testid="code-cell" data-line-number="962" inert="" style="position: relative;"><span class="pl-s1">ax</span>.<span class="pl-c1">legend</span>(<span class="pl-s1">fontsize</span><span class="pl-c1">=</span><span class="pl-c1">8</span>); <span class="pl-s1">ax</span>.<span class="pl-c1">grid</span>(<span class="pl-c1">True</span>, <span class="pl-s1">alpha</span><span class="pl-c1">=</span><span class="pl-c1">0.3</span>)</div></div></div><div data-key="962" class="react-code-text react-code-line-contents virtual" style="transform: translateY(19240px); min-height: auto;"><div><div id="LC963" class="react-file-line html-div" data-testid="code-cell" data-line-number="963" inert="" style="position: relative;">
</div></div></div><div data-key="963" class="react-code-text react-code-line-contents virtual" style="transform: translateY(19260px); min-height: auto;"><div><div id="LC964" class="react-file-line html-div" data-testid="code-cell" data-line-number="964" inert="" style="position: relative;"><span class="pl-s1">ax</span> <span class="pl-c1">=</span> <span class="pl-s1">axes</span>[<span class="pl-c1">0</span>, <span class="pl-c1">1</span>]</div></div></div><div data-key="964" class="react-code-text react-code-line-contents virtual" style="transform: translateY(19280px); min-height: auto;"><div><div id="LC965" class="react-file-line html-div" data-testid="code-cell" data-line-number="965" inert="" style="position: relative;"><span class="pl-k">if</span> <span class="pl-en">len</span>(<span class="pl-s1">waiting_times</span>) <span class="pl-c1">&gt;</span> <span class="pl-c1">5</span>:</div></div></div><div data-key="965" class="react-code-text react-code-line-contents virtual" style="transform: translateY(19300px); min-height: auto;"><div><div id="LC966" class="react-file-line html-div" data-testid="code-cell" data-line-number="966" inert="" style="position: relative;">    <span class="pl-s1">ax</span>.<span class="pl-c1">plot</span>(<span class="pl-en">smooth</span>(<span class="pl-s1">waiting_times</span>), <span class="pl-s1">color</span><span class="pl-c1">=</span><span class="pl-s">'#d62728'</span>, <span class="pl-s1">linewidth</span><span class="pl-c1">=</span><span class="pl-c1">1.5</span>, <span class="pl-s1">label</span><span class="pl-c1">=</span><span class="pl-s">'Avg Wait'</span>)</div></div></div><div data-key="966" class="react-code-text react-code-line-contents virtual" style="transform: translateY(19320px); min-height: auto;"><div><div id="LC967" class="react-file-line html-div" data-testid="code-cell" data-line-number="967" inert="" style="position: relative;"><span class="pl-k">for</span> (<span class="pl-s1">lane</span>, <span class="pl-s1">s</span>, <span class="pl-s1">e</span>) <span class="pl-c1">in</span> <span class="pl-c1">INCIDENT_SCHEDULE</span>:</div></div></div><div data-key="967" class="react-code-text react-code-line-contents virtual" style="transform: translateY(19340px); min-height: auto;"><div><div id="LC968" class="react-file-line html-div" data-testid="code-cell" data-line-number="968" inert="" style="position: relative;">    <span class="pl-s1">ax</span>.<span class="pl-c1">axvspan</span>(<span class="pl-s1">s</span>, <span class="pl-en">min</span>(<span class="pl-s1">e</span>, <span class="pl-c1">TOTAL_STEPS</span>), <span class="pl-s1">alpha</span><span class="pl-c1">=</span><span class="pl-c1">0.15</span>, <span class="pl-s1">color</span><span class="pl-c1">=</span><span class="pl-s">'red'</span>)</div></div></div><div data-key="968" class="react-code-text react-code-line-contents virtual" style="transform: translateY(19360px); min-height: auto;"><div><div id="LC969" class="react-file-line html-div" data-testid="code-cell" data-line-number="969" inert="" style="position: relative;"><span class="pl-s1">ax</span>.<span class="pl-c1">set_title</span>(<span class="pl-s">'Waiting Time'</span>); <span class="pl-s1">ax</span>.<span class="pl-c1">set_xlabel</span>(<span class="pl-s">'Step'</span>); <span class="pl-s1">ax</span>.<span class="pl-c1">set_ylabel</span>(<span class="pl-s">'Seconds'</span>)</div></div></div><div data-key="969" class="react-code-text react-code-line-contents virtual" style="transform: translateY(19380px); min-height: auto;"><div><div id="LC970" class="react-file-line html-div" data-testid="code-cell" data-line-number="970" inert="" style="position: relative;"><span class="pl-s1">ax</span>.<span class="pl-c1">legend</span>(<span class="pl-s1">fontsize</span><span class="pl-c1">=</span><span class="pl-c1">8</span>); <span class="pl-s1">ax</span>.<span class="pl-c1">grid</span>(<span class="pl-c1">True</span>, <span class="pl-s1">alpha</span><span class="pl-c1">=</span><span class="pl-c1">0.3</span>)</div></div></div><div data-key="970" class="react-code-text react-code-line-contents virtual" style="transform: translateY(19400px); min-height: auto;"><div><div id="LC971" class="react-file-line html-div" data-testid="code-cell" data-line-number="971" inert="" style="position: relative;">
</div></div></div><div data-key="971" class="react-code-text react-code-line-contents virtual" style="transform: translateY(19420px); min-height: auto;"><div><div id="LC972" class="react-file-line html-div" data-testid="code-cell" data-line-number="972" inert="" style="position: relative;"><span class="pl-s1">ax</span> <span class="pl-c1">=</span> <span class="pl-s1">axes</span>[<span class="pl-c1">1</span>, <span class="pl-c1">0</span>]</div></div></div><div data-key="972" class="react-code-text react-code-line-contents virtual" style="transform: translateY(19440px); min-height: auto;"><div><div id="LC973" class="react-file-line html-div" data-testid="code-cell" data-line-number="973" inert="" style="position: relative;"><span class="pl-k">if</span> <span class="pl-s1">incident_log</span>:</div></div></div><div data-key="973" class="react-code-text react-code-line-contents virtual" style="transform: translateY(19460px); min-height: auto;"><div><div id="LC974" class="react-file-line html-div" data-testid="code-cell" data-line-number="974" inert="" style="position: relative;">    <span class="pl-s1">ax</span>.<span class="pl-c1">fill_between</span>(<span class="pl-en">range</span>(<span class="pl-en">len</span>(<span class="pl-s1">incident_log</span>)), <span class="pl-s1">incident_log</span>,</div></div></div><div data-key="974" class="react-code-text react-code-line-contents virtual" style="transform: translateY(19480px); min-height: auto;"><div><div id="LC975" class="react-file-line html-div" data-testid="code-cell" data-line-number="975" inert="" style="position: relative;">                    <span class="pl-s1">color</span><span class="pl-c1">=</span><span class="pl-s">'#ff7f0e'</span>, <span class="pl-s1">alpha</span><span class="pl-c1">=</span><span class="pl-c1">0.6</span>, <span class="pl-s1">label</span><span class="pl-c1">=</span><span class="pl-s">'Severity'</span>)</div></div></div><div data-key="975" class="react-code-text react-code-line-contents virtual" style="transform: translateY(19500px); min-height: auto;"><div><div id="LC976" class="react-file-line html-div" data-testid="code-cell" data-line-number="976" inert="" style="position: relative;"><span class="pl-s1">ax</span>.<span class="pl-c1">set_title</span>(<span class="pl-s">'Incident Severity'</span>); <span class="pl-s1">ax</span>.<span class="pl-c1">set_xlabel</span>(<span class="pl-s">'Step'</span>)</div></div></div><div data-key="976" class="react-code-text react-code-line-contents virtual" style="transform: translateY(19520px); min-height: auto;"><div><div id="LC977" class="react-file-line html-div" data-testid="code-cell" data-line-number="977" inert="" style="position: relative;"><span class="pl-s1">ax</span>.<span class="pl-c1">set_ylim</span>(<span class="pl-c1">0</span>, <span class="pl-c1">1.1</span>); <span class="pl-s1">ax</span>.<span class="pl-c1">legend</span>(<span class="pl-s1">fontsize</span><span class="pl-c1">=</span><span class="pl-c1">8</span>); <span class="pl-s1">ax</span>.<span class="pl-c1">grid</span>(<span class="pl-c1">True</span>, <span class="pl-s1">alpha</span><span class="pl-c1">=</span><span class="pl-c1">0.3</span>)</div></div></div><div data-key="977" class="react-code-text react-code-line-contents virtual" style="transform: translateY(19540px); min-height: auto;"><div><div id="LC978" class="react-file-line html-div" data-testid="code-cell" data-line-number="978" inert="" style="position: relative;">
</div></div></div><div data-key="978" class="react-code-text react-code-line-contents virtual" style="transform: translateY(19560px); min-height: auto;"><div><div id="LC979" class="react-file-line html-div" data-testid="code-cell" data-line-number="979" inert="" style="position: relative;"><span class="pl-s1">ax</span> <span class="pl-c1">=</span> <span class="pl-s1">axes</span>[<span class="pl-c1">1</span>, <span class="pl-c1">1</span>]</div></div></div><div data-key="979" class="react-code-text react-code-line-contents virtual" style="transform: translateY(19580px); min-height: auto;"><div><div id="LC980" class="react-file-line html-div" data-testid="code-cell" data-line-number="980" inert="" style="position: relative;"><span class="pl-k">if</span> <span class="pl-c1">TRAIN</span> <span class="pl-c1">and</span> <span class="pl-en">len</span>(<span class="pl-s1">losses</span>) <span class="pl-c1">&gt;</span> <span class="pl-c1">5</span>:</div></div></div><div data-key="980" class="react-code-text react-code-line-contents virtual" style="transform: translateY(19600px); min-height: auto;"><div><div id="LC981" class="react-file-line html-div" data-testid="code-cell" data-line-number="981" inert="" style="position: relative;">    <span class="pl-s1">ax</span>.<span class="pl-c1">plot</span>(<span class="pl-en">smooth</span>(<span class="pl-s1">losses</span>), <span class="pl-s1">color</span><span class="pl-c1">=</span><span class="pl-s">'#9467bd'</span>, <span class="pl-s1">linewidth</span><span class="pl-c1">=</span><span class="pl-c1">1.5</span>, <span class="pl-s1">label</span><span class="pl-c1">=</span><span class="pl-s">'Loss'</span>)</div></div></div><div data-key="981" class="react-code-text react-code-line-contents virtual" style="transform: translateY(19620px); min-height: auto;"><div><div id="LC982" class="react-file-line html-div" data-testid="code-cell" data-line-number="982" inert="" style="position: relative;">    <span class="pl-s1">ax</span>.<span class="pl-c1">set_title</span>(<span class="pl-s">'Training Loss'</span>)</div></div></div><div data-key="982" class="react-code-text react-code-line-contents virtual" style="transform: translateY(19640px); min-height: auto;"><div><div id="LC983" class="react-file-line html-div" data-testid="code-cell" data-line-number="983" inert="" style="position: relative;"><span class="pl-k">else</span>:</div></div></div><div data-key="983" class="react-code-text react-code-line-contents virtual" style="transform: translateY(19660px); min-height: auto;"><div><div id="LC984" class="react-file-line html-div" data-testid="code-cell" data-line-number="984" inert="" style="position: relative;">    <span class="pl-s1">ax</span>.<span class="pl-c1">plot</span>(<span class="pl-s1">vehicle_counts</span>, <span class="pl-s1">color</span><span class="pl-c1">=</span><span class="pl-s">'#2ca02c'</span>, <span class="pl-s1">linewidth</span><span class="pl-c1">=</span><span class="pl-c1">1.5</span>, <span class="pl-s1">label</span><span class="pl-c1">=</span><span class="pl-s">'Vehicles'</span>)</div></div></div><div data-key="984" class="react-code-text react-code-line-contents virtual" style="transform: translateY(19680px); min-height: auto;"><div><div id="LC985" class="react-file-line html-div" data-testid="code-cell" data-line-number="985" inert="" style="position: relative;">    <span class="pl-s1">ax</span>.<span class="pl-c1">set_title</span>(<span class="pl-s">'Vehicles in Network'</span>)</div></div></div><div data-key="985" class="react-code-text react-code-line-contents virtual" style="transform: translateY(19700px); min-height: auto;"><div><div id="LC986" class="react-file-line html-div" data-testid="code-cell" data-line-number="986" inert="" style="position: relative;"><span class="pl-s1">ax</span>.<span class="pl-c1">set_xlabel</span>(<span class="pl-s">'Step'</span>); <span class="pl-s1">ax</span>.<span class="pl-c1">legend</span>(<span class="pl-s1">fontsize</span><span class="pl-c1">=</span><span class="pl-c1">8</span>); <span class="pl-s1">ax</span>.<span class="pl-c1">grid</span>(<span class="pl-c1">True</span>, <span class="pl-s1">alpha</span><span class="pl-c1">=</span><span class="pl-c1">0.3</span>)</div></div></div><div data-key="986" class="react-code-text react-code-line-contents virtual" style="transform: translateY(19720px); min-height: auto;"><div><div id="LC987" class="react-file-line html-div" data-testid="code-cell" data-line-number="987" inert="" style="position: relative;">
</div></div></div><div data-key="987" class="react-code-text react-code-line-contents virtual" style="transform: translateY(19740px); min-height: auto;"><div><div id="LC988" class="react-file-line html-div" data-testid="code-cell" data-line-number="988" inert="" style="position: relative;"><span class="pl-s1">plt</span>.<span class="pl-c1">tight_layout</span>()</div></div></div><div data-key="988" class="react-code-text react-code-line-contents virtual" style="transform: translateY(19760px); min-height: auto;"><div><div id="LC989" class="react-file-line html-div" data-testid="code-cell" data-line-number="989" inert="" style="position: relative;"><span class="pl-s1">plt</span>.<span class="pl-c1">savefig</span>(<span class="pl-s">'simulation_results.png'</span>, <span class="pl-s1">dpi</span><span class="pl-c1">=</span><span class="pl-c1">150</span>, <span class="pl-s1">bbox_inches</span><span class="pl-c1">=</span><span class="pl-s">'tight'</span>)</div></div></div><div data-key="989" class="react-code-text react-code-line-contents virtual" style="transform: translateY(19780px); min-height: auto;"><div><div id="LC990" class="react-file-line html-div" data-testid="code-cell" data-line-number="990" inert="" style="position: relative;"><span class="pl-s1">plt</span>.<span class="pl-c1">show</span>()</div></div></div><div data-key="990" class="react-code-text react-code-line-contents virtual" style="transform: translateY(19800px); min-height: auto;"><div><div id="LC991" class="react-file-line html-div" data-testid="code-cell" data-line-number="991" inert="" style="position: relative;"><span class="pl-en">print</span>(<span class="pl-s">"Plot saved: simulation_results.png"</span>)</div></div></div></div><button hidden="" data-hotkey="Control+a"></button></div></div></div></div><div id="copilot-button-container"></div></div><div id="highlighted-line-menu-container"></div></div></div></section></div></div></div> </div>  </div></div></div></div></div></div><div class="ScrollMarksContainer-module__scrollMarksContainer__Eu7uU" id="find-result-marks-container"></div><button hidden="" data-testid="" data-hotkey-scope="read-only-cursor-text-area" data-hotkey="Control+F6,Control+Shift+F6"></button><button hidden="" data-hotkey="Control+F6,Control+Shift+F6"></button></div> <!-- --> <!-- --> <script type="application/json" id="__PRIMER_DATA__R_1___">{"resolvedServerColorMode":"night"}</script></div>
</react-app>




  </div>

</turbo-frame>

    </main>
  </div>

  </div>

          <footer class="footer tmp-pt-7 tmp-pb-6 f6 color-fg-muted color-border-subtle p-responsive" role="contentinfo" hidden="">
  <h2 class="sr-only">Footer</h2>

  


  <div class="d-flex flex-justify-center flex-items-center flex-column-reverse flex-lg-row flex-wrap flex-lg-nowrap">
    <div class="d-flex flex-items-center flex-shrink-0 mx-2">
      <a aria-label="GitHub Homepage" class="footer-octicon mr-2" href="https://github.com/">
        <svg aria-hidden="true" height="24" viewBox="0 0 24 24" version="1.1" width="24" data-view-component="true" class="octicon octicon-mark-github">
    <path d="M10.226 17.284c-2.965-.36-5.054-2.493-5.054-5.256 0-1.123.404-2.336 1.078-3.144-.292-.741-.247-2.314.09-2.965.898-.112 2.111.36 2.83 1.01.853-.269 1.752-.404 2.853-.404 1.1 0 1.999.135 2.807.382.696-.629 1.932-1.1 2.83-.988.315.606.36 2.179.067 2.942.72.854 1.101 2 1.101 3.167 0 2.763-2.089 4.852-5.098 5.234.763.494 1.28 1.572 1.28 2.807v2.336c0 .674.561 1.056 1.235.786 4.066-1.55 7.255-5.615 7.255-10.646C23.5 6.188 18.334 1 11.978 1 5.62 1 .5 6.188.5 12.545c0 4.986 3.167 9.12 7.435 10.669.606.225 1.19-.18 1.19-.786V20.63a2.9 2.9 0 0 1-1.078.224c-1.483 0-2.359-.808-2.987-2.313-.247-.607-.517-.966-1.034-1.033-.27-.023-.359-.135-.359-.27 0-.27.45-.471.898-.471.652 0 1.213.404 1.797 1.235.45.651.921.943 1.483.943.561 0 .92-.202 1.437-.719.382-.381.674-.718.944-.943"></path>
</svg>
</a>
      <span>
        © 2026 GitHub,&nbsp;Inc.
      </span>
    </div>

    <nav aria-label="Footer">
      <h3 class="sr-only" id="sr-footer-heading">Footer navigation</h3>

      <ul class="list-style-none d-flex flex-justify-center flex-wrap mb-2 mb-lg-0" aria-labelledby="sr-footer-heading">


          <li class="mx-2">
            <a data-analytics-event="{&quot;category&quot;:&quot;Footer&quot;,&quot;action&quot;:&quot;go to Terms&quot;,&quot;label&quot;:&quot;text:terms&quot;}" href="https://docs.github.com/site-policy/github-terms/github-terms-of-service" data-view-component="true" class="Link--secondary Link">Terms</a>
          </li>

          <li class="mx-2">
            <a data-analytics-event="{&quot;category&quot;:&quot;Footer&quot;,&quot;action&quot;:&quot;go to privacy&quot;,&quot;label&quot;:&quot;text:privacy&quot;}" href="https://docs.github.com/site-policy/privacy-policies/github-privacy-statement" data-view-component="true" class="Link--secondary Link">Privacy</a>
          </li>

          <li class="mx-2">
            <a data-analytics-event="{&quot;category&quot;:&quot;Footer&quot;,&quot;action&quot;:&quot;go to security&quot;,&quot;label&quot;:&quot;text:security&quot;}" href="https://github.com/security" data-view-component="true" class="Link--secondary Link">Security</a>
          </li>

          <li class="mx-2">
            <a data-analytics-event="{&quot;category&quot;:&quot;Footer&quot;,&quot;action&quot;:&quot;go to status&quot;,&quot;label&quot;:&quot;text:status&quot;}" href="https://www.githubstatus.com/" data-view-component="true" class="Link--secondary Link">Status</a>
          </li>

          <li class="mx-2">
            <a data-analytics-event="{&quot;category&quot;:&quot;Footer&quot;,&quot;action&quot;:&quot;go to community&quot;,&quot;label&quot;:&quot;text:community&quot;}" href="https://github.community/" data-view-component="true" class="Link--secondary Link">Community</a>
          </li>

          <li class="mx-2">
            <a data-analytics-event="{&quot;category&quot;:&quot;Footer&quot;,&quot;action&quot;:&quot;go to docs&quot;,&quot;label&quot;:&quot;text:docs&quot;}" href="https://docs.github.com/" data-view-component="true" class="Link--secondary Link">Docs</a>
          </li>

          <li class="mx-2">
            <a data-analytics-event="{&quot;category&quot;:&quot;Footer&quot;,&quot;action&quot;:&quot;go to contact&quot;,&quot;label&quot;:&quot;text:contact&quot;}" href="https://support.github.com/?tags=dotcom-footer" data-view-component="true" class="Link--secondary Link">Contact</a>
          </li>

          <li class="mx-2">
  <cookie-consent-link data-catalyst="">
    <button type="button" class="Link--secondary underline-on-hover border-0 p-0 color-bg-transparent" data-action="click:cookie-consent-link#showConsentManagement" data-analytics-event="{&quot;location&quot;:&quot;footer&quot;,&quot;action&quot;:&quot;cookies&quot;,&quot;context&quot;:&quot;subfooter&quot;,&quot;tag&quot;:&quot;link&quot;,&quot;label&quot;:&quot;cookies_link_subfooter_footer&quot;}">
       Manage cookies
    </button>
  </cookie-consent-link>
</li>

<li class="mx-2">
  <cookie-consent-link data-catalyst="">
    <button type="button" class="Link--secondary underline-on-hover border-0 p-0 color-bg-transparent text-left" data-action="click:cookie-consent-link#showConsentManagement" data-analytics-event="{&quot;location&quot;:&quot;footer&quot;,&quot;action&quot;:&quot;dont_share_info&quot;,&quot;context&quot;:&quot;subfooter&quot;,&quot;tag&quot;:&quot;link&quot;,&quot;label&quot;:&quot;dont_share_info_link_subfooter_footer&quot;}">
      Do not share my personal information
    </button>
  </cookie-consent-link>
</li>

      </ul>
    </nav>
  </div>
</footer>



    <ghcc-consent id="ghcc" class="position-fixed bottom-0 left-0" style="z-index: 999999" data-locale="en" data-initial-cookie-consent-allowed="" data-cookie-consent-required="false" data-catalyst=""></ghcc-consent>




  <div id="ajax-error-message" class="ajax-error-message flash flash-error" hidden="">
    <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-alert">
    <path d="M6.457 1.047c.659-1.234 2.427-1.234 3.086 0l6.082 11.378A1.75 1.75 0 0 1 14.082 15H1.918a1.75 1.75 0 0 1-1.543-2.575Zm1.763.707a.25.25 0 0 0-.44 0L1.698 13.132a.25.25 0 0 0 .22.368h12.164a.25.25 0 0 0 .22-.368Zm.53 3.996v2.5a.75.75 0 0 1-1.5 0v-2.5a.75.75 0 0 1 1.5 0ZM9 11a1 1 0 1 1-2 0 1 1 0 0 1 2 0Z"></path>
</svg>
    <button type="button" class="flash-close js-ajax-error-dismiss" aria-label="Dismiss error">
      <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-x">
    <path d="M3.72 3.72a.75.75 0 0 1 1.06 0L8 6.94l3.22-3.22a.749.749 0 0 1 1.275.326.749.749 0 0 1-.215.734L9.06 8l3.22 3.22a.749.749 0 0 1-.326 1.275.749.749 0 0 1-.734-.215L8 9.06l-3.22 3.22a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042L6.94 8 3.72 4.78a.75.75 0 0 1 0-1.06Z"></path>
</svg>
    </button>
    You can’t perform that action at this time.
  </div>

    <template id="site-details-dialog"></template>

    <div class="Popover js-hovercard-content position-absolute" style="display: none; outline: none;">
  <div class="Popover-message Popover-message--bottom-left Popover-message--large Box color-shadow-large" style="width:360px;"></div>
</div>

    <template id="snippet-clipboard-copy-button"></template>
<template id="snippet-clipboard-copy-button-unpositioned"></template>




    </div>
    <div id="js-global-screen-reader-notice" class="sr-only mt-n1" aria-live="polite" aria-atomic="true"></div>
    <div id="js-global-screen-reader-notice-assertive" class="sr-only mt-n1" aria-live="assertive" aria-atomic="true"></div>
  


<div class="sr-only mt-n1" id="screenReaderAnnouncementDiv" role="alert" data-testid="screenReaderAnnouncement" aria-live="assertive">While the code is focused, press Alt+F1 for a menu of operations.</div></body></html>