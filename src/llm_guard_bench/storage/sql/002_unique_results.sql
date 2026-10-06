-- Enforce one result per session, attack, and model.
CREATE UNIQUE INDEX ux_test_results_session_attack_model
    ON test_results (session_id, attack_id, model_name);
