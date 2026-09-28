-- =============================================================================
-- LLM-Guard-Bench: Initial Schema Migration
-- Version: 001
-- Applied by: DatabaseManager.initialize() on first startup.
-- Safe to re-run: all statements use IF NOT EXISTS.
-- =============================================================================

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

-- ---------------------------------------------------------------------------
-- Primary results table
-- v3.0: target_model_name → model_name, attack_category → category
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS test_results (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,

    session_id          TEXT    NOT NULL
                                REFERENCES sessions(session_id) ON DELETE CASCADE,

    timestamp           TEXT    NOT NULL,
    -- ISO 8601 with milliseconds: "2025-05-25T14:23:01.456Z"

    model_name          TEXT    NOT NULL,
    -- Exact model identifier string used as the test target.
    -- Examples: "gpt-4o", "llama3.2:3b", "claude-3-5-sonnet-20241022"

    attack_id           TEXT    NOT NULL
                                REFERENCES attack_definitions(attack_id),

    category            TEXT    NOT NULL
                                CHECK (category IN (
                                    'DAN',
                                    'ROLEPLAY_EXPLOIT',
                                    'PROMPT_INJECTION',
                                    'SYSTEM_PROMPT_LEAKAGE',
                                    'DO_NOT_REFUSE'
                                )),
    -- v3.0: renamed from attack_category to category

    adversarial_prompt  TEXT    NOT NULL,
    -- Full adversarial prompt string sent to the target model verbatim.

    system_prompt       TEXT,
    -- NULL when no system prompt is configured for this attack definition.

    raw_llm_response    TEXT    NOT NULL    DEFAULT '',
    -- Verbatim response text from the target model.
    -- Empty string (never NULL) when the response could not be obtained.

    evaluation_status   TEXT    NOT NULL
                                CHECK (evaluation_status IN (
                                    'PASSED',
                                    'VULNERABLE',
                                    'FAILED',
                                    'EVAL_ERROR',
                                    'TIMEOUT',
                                    'SKIPPED'
                                )),
    -- PASSED     = target model correctly refused the adversarial prompt
    -- VULNERABLE = target model was successfully jailbroken
    -- FAILED     = judge ran but returned output other than PASSED/VULNERABLE
    -- EVAL_ERROR = Python/network exception in pipeline or evaluator
    -- TIMEOUT    = target model did not respond within REQUEST_TIMEOUT_SECONDS
    -- SKIPPED    = token budget pre-flight blocked execution
    -- NOTE: AMBIGUOUS is intentionally absent — it is an internal sentinel
    --       and must never be written to the database.

    evaluation_stage    TEXT    NOT NULL
                                CHECK (evaluation_stage IN (
                                    'STAGE_1_KEYWORD',
                                    'STAGE_2_JUDGE',
                                    'PRE_FLIGHT'
                                )),
    -- PRE_FLIGHT: evaluation never started (SKIPPED, TIMEOUT, EVAL_ERROR at adapter level)
    -- STAGE_1_KEYWORD: Stage 1 was decisive (PASSED or VULNERABLE)
    -- STAGE_2_JUDGE: Stage 1 was AMBIGUOUS; judge LLM was invoked

    judge_verdict       TEXT
                                CHECK (judge_verdict IS NULL OR
                                       judge_verdict IN ('PASSED', 'VULNERABLE')),
    -- NULL when evaluation_stage = STAGE_1_KEYWORD or judge produced FAILED/EVAL_ERROR

    judge_parse_error   INTEGER NOT NULL    DEFAULT 0
                                CHECK (judge_parse_error IN (0, 1)),
    -- 1 (true) when judge returned text other than exactly "PASSED" or "VULNERABLE"

    execution_time_ms   INTEGER NOT NULL    DEFAULT 0,
    -- Wall-clock ms for the target LLM API call only (not including evaluation).
    -- 0 for SKIPPED, TIMEOUT, EVAL_ERROR records where no complete call occurred.

    total_time_ms       INTEGER NOT NULL    DEFAULT 0,
    -- Wall-clock ms for the entire pipeline task: API call + evaluation + DB write.

    prompt_tokens       INTEGER,
    -- Input token count from the API response usage object. NULL when unavailable.

    completion_tokens   INTEGER,
    -- Output token count from the API response usage object. NULL when unavailable.

    http_status_code    INTEGER,
    -- Raw HTTP status code from the target LLM API. NULL on timeout or pre-flight skip.

    error_message       TEXT
    -- NULL on success. Exception class name + message on any failure.
);

-- ---------------------------------------------------------------------------
-- Session metadata table
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS sessions (
    session_id          TEXT    PRIMARY KEY,
    started_at          TEXT    NOT NULL,
    finished_at         TEXT,
    -- NULL until session completes. If NULL in a live DB, session crashed mid-run.

    config_snapshot     TEXT    NOT NULL,
    -- JSON blob: exact settings.py values at session start time.

    total_tests         INTEGER NOT NULL    DEFAULT 0,
    passed_count        INTEGER NOT NULL    DEFAULT 0,
    vulnerable_count    INTEGER NOT NULL    DEFAULT 0,
    failed_count        INTEGER NOT NULL    DEFAULT 0,
    eval_error_count    INTEGER NOT NULL    DEFAULT 0,
    timeout_count       INTEGER NOT NULL    DEFAULT 0,
    skipped_count       INTEGER NOT NULL    DEFAULT 0
);

-- ---------------------------------------------------------------------------
-- Attack definitions table
-- Populated from config/prompts.json at the start of each session.
-- Uses INSERT OR REPLACE so edits to prompts.json propagate automatically.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS attack_definitions (
    attack_id           TEXT    PRIMARY KEY,
    category            TEXT    NOT NULL
                                CHECK (category IN (
                                    'DAN',
                                    'ROLEPLAY_EXPLOIT',
                                    'PROMPT_INJECTION',
                                    'SYSTEM_PROMPT_LEAKAGE',
                                    'DO_NOT_REFUSE'
                                )),
    attack_name         TEXT    NOT NULL,
    description         TEXT    NOT NULL,
    adversarial_prompt  TEXT    NOT NULL,
    system_prompt       TEXT,
    expected_behavior   TEXT    NOT NULL    DEFAULT 'REFUSAL'
                                CHECK (expected_behavior = 'REFUSAL'),
    severity            TEXT    NOT NULL
                                CHECK (severity IN ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL')),
    tags                TEXT    NOT NULL
    -- JSON array stored as TEXT: '["jailbreak", "social_engineering"]'
);

-- ---------------------------------------------------------------------------
-- Query performance indexes
-- ---------------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_results_session
    ON test_results (session_id);

CREATE INDEX IF NOT EXISTS idx_results_model
    ON test_results (model_name);

CREATE INDEX IF NOT EXISTS idx_results_category
    ON test_results (category);

CREATE INDEX IF NOT EXISTS idx_results_status
    ON test_results (evaluation_status);

CREATE INDEX IF NOT EXISTS idx_results_model_status
    ON test_results (model_name, evaluation_status);
-- Used by: compute_attack_success_rate GROUP BY model_name

CREATE INDEX IF NOT EXISTS idx_results_model_category_status
    ON test_results (model_name, category, evaluation_status);
-- Used by: per-category ASR breakdown aggregation
