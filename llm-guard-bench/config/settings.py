"""
Configuration Settings for LLM Guard Bench

Handles environment-specific configuration, database paths, API endpoints,
and runtime constants. All sensitive values use os.getenv() with safe defaults
to prevent credential exposure in version control.

Security Policy:
- NO hardcoded API keys or credentials
- All secrets loaded from environment variables (.env file)
- .env is GITIGNORED and never committed
- Safe defaults provided for non-secret configuration

Module Constants:
- PROJECT_ROOT: Absolute path to project root directory
- DB_PATH: SQLite database file path
- RESULTS_DIR: Output directory for reports and artifacts
- API endpoints and timeout configurations
"""

from pathlib import Path
import os
from typing import Final

# ============================================================================
# PROJECT STRUCTURE
# ============================================================================
PROJECT_ROOT: Final[Path] = Path(__file__).parent.parent
"""Absolute path to the llm-guard-bench project root directory."""

RESULTS_DIR: Final[Path] = PROJECT_ROOT / "results"
"""Output directory for benchmark results, reports, and session artifacts."""

DB_PATH: Final[Path] = PROJECT_ROOT / "results" / "guard_bench.db"
"""Path to the SQLite database file for storing test results."""

# Ensure results directory exists (created at startup)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# DATABASE CONFIGURATION
# ============================================================================
DATABASE_PATH: Final[Path] = DB_PATH
"""Alias for DB_PATH for explicit database configuration."""

# ============================================================================
# DEFAULT TIMEOUT & CONCURRENCY SETTINGS
# ============================================================================
DEFAULT_TIMEOUT_SECONDS: Final[float] = 180.0
"""
Default timeout for API requests and model inference (seconds).
Applies to both Ollama and Groq API calls.
"""

DEFAULT_CONCURRENCY: Final[int] = 1
"""
Default maximum concurrent benchmark executions.
Prevents overwhelming target or judge models.
"""

DEFAULT_JUDGE_MODEL: Final[str] = os.getenv(
    "JUDGE_MODEL_NAME",
    "openai/gpt-oss-20b",
)
"""Default judge model for evaluation (Groq endpoint)."""

# ============================================================================
# OLLAMA CONFIGURATION
# ============================================================================
DEFAULT_OLLAMA_BASE_URL: Final[str] = os.getenv(
    "OLLAMA_BASE_URL",
    "http://host.docker.internal:11434"
)
"""
Default Ollama service URL. Supports:
- Local: http://localhost:11434
- Docker host (from container): http://host.docker.internal:11434
- Docker Compose: http://ollama:11434
- Remote: https://remote-host:11434
"""

OLLAMA_ENDPOINT: Final[str] = os.getenv(
    "OLLAMA_ENDPOINT",
    DEFAULT_OLLAMA_BASE_URL
)
"""
Runtime Ollama endpoint, resolved from OLLAMA_ENDPOINT environment variable.
Falls back to DEFAULT_OLLAMA_BASE_URL if not set.
"""

# ============================================================================
# API ENDPOINTS
# ============================================================================
GROQ_API_ENDPOINT: Final[str] = "https://api.groq.com/openai/v1"
"""Official Groq API endpoint for OpenAI-compatible chat completions."""

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================
LOG_LEVEL: Final[str] = os.getenv("LOG_LEVEL", "INFO")
"""
Logging level: DEBUG, INFO, WARNING, ERROR, or CRITICAL.
Resolved from LOG_LEVEL environment variable.
"""

# ============================================================================
# SECURITY: API KEY MANAGEMENT
# ============================================================================
GROQ_API_KEY: Final[str | None] = os.getenv("GROQ_API_KEY")
"""
Groq API key for judge model inference.

Security Note:
- Loaded from GROQ_API_KEY environment variable only
- Never hardcoded in source code
- .env file is GITIGNORED
- Verify: git check-ignore -v .env

If not set, Groq adapter initialization will fail with clear error message.
"""

OPENAI_API_KEY: Final[str | None] = os.getenv("OPENAI_API_KEY")
"""OpenAI API key (optional, for alternative judge model provider)."""

ANTHROPIC_API_KEY: Final[str | None] = os.getenv("ANTHROPIC_API_KEY")
"""Anthropic API key (optional, for Claude models as judge)."""

# ============================================================================
# MODEL PROVIDER CONFIGURATION
# ============================================================================
TARGET_PROVIDER: Final[str] = os.getenv("TARGET_PROVIDER", "ollama")
"""
Target model provider: which service runs the target LLM being evaluated.
Options: ollama (local), groq (API), openai (API), anthropic (API)
"""

JUDGE_PROVIDER: Final[str] = os.getenv("JUDGE_PROVIDER", "groq")
"""
Judge model provider: which service evaluates target responses.
Options: groq (recommended), openai, anthropic
"""

# ============================================================================
# REQUEST TIMEOUTS
# ============================================================================
REQUEST_TIMEOUT: Final[float] = float(os.getenv("REQUEST_TIMEOUT", "180"))
"""
Request timeout in seconds for API calls to external services.
Applies to both target model and judge model API requests.
"""

# ============================================================================
# CONCURRENCY LIMITS
# ============================================================================
CONCURRENCY_LIMIT: Final[int] = int(os.getenv("CONCURRENCY_LIMIT", "1"))
"""
Maximum number of concurrent benchmark executions.
Higher values = faster execution but higher resource usage.
Reasonable range: 1-16 (depends on target and judge model capacity)
"""

# ============================================================================
# VALIDATION & STARTUP CHECKS
# ============================================================================
def validate_configuration() -> None:
    """
    Validate critical configuration at startup.

    Raises:
        ValueError: If required configuration is missing or invalid.
    """
    if not PROJECT_ROOT.exists():
        raise ValueError(f"Project root does not exist: {PROJECT_ROOT}")

    if not RESULTS_DIR.exists():
        try:
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise ValueError(f"Cannot create results directory: {e}")

    # Validate provider selections
    valid_target_providers = {"ollama", "groq", "openai", "anthropic"}
    if TARGET_PROVIDER not in valid_target_providers:
        raise ValueError(
            f"Invalid TARGET_PROVIDER '{TARGET_PROVIDER}'. "
            f"Must be one of: {valid_target_providers}"
        )

    valid_judge_providers = {"ollama", "groq", "openai", "anthropic"}
    if JUDGE_PROVIDER not in valid_judge_providers:
        raise ValueError(
            f"Invalid JUDGE_PROVIDER '{JUDGE_PROVIDER}'. "
            f"Must be one of: {valid_judge_providers}"
        )

    # Validate numeric settings
    if REQUEST_TIMEOUT <= 0:
        raise ValueError(f"REQUEST_TIMEOUT must be positive, got {REQUEST_TIMEOUT}")

    if CONCURRENCY_LIMIT < 1:
        raise ValueError(f"CONCURRENCY_LIMIT must be >= 1, got {CONCURRENCY_LIMIT}")


# Validate configuration on module import
validate_configuration()

