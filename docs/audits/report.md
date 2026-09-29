# LLM Guard Bench — Code Quality & Architecture Audit

**Date:** 2026-09-28  
**Reviewed checkout:** `eafte/llm-guard-bench`, commit `2901d0d03064788cf8d626beed7cff49f6eb4cbc`  
**Scope:** All Python source, SQL migration, Docker/Compose configuration, launch scripts, repository metadata, documentation, and tracked-artifact inventory. Source references below are relative to the inner `llm-guard-bench/` directory unless explicitly prefixed `root:`. Line numbers refer to this commit, not future edits.

## 1. Executive assessment

**Verdict: a useful small-scale prototype, not yet a trustworthy research benchmark or million-record execution framework.** The biggest blocker is measurement validity, followed by data durability and unbounded memory—not missing cloud providers.

The project already has useful foundations:

- An abstract async provider interface and dependency-injected target/judge adapters.
- Pydantic result records and parameterized SQL inserts.
- SQLite WAL initialization, JSONL export, and explicit infrastructure-error statuses.
- Network timeouts, some transient-failure retries, and database cleanup in `finally`.
- Chart generation and synchronous SQLite reads offloaded to threads.
- A small, understandable attack fixture and documentation of the current limited experiment scale.

However, results can currently report successful defenses without evidence, persistence failures can be hidden, the installed CLI does not work, and every major stage retains dataset-sized collections. Descriptions such as “production-ready” and “BULLETPROOF” are not supported by the implementation or test coverage.

### Priority definitions

- **P0:** fix before publishing new security conclusions or claiming reliable benchmark execution.
- **P1:** fix before large datasets, unattended operation, or a public package/container release.
- **P2:** maintainability, portability, and portfolio credibility improvements.

These are project-risk priorities, not CVSS vulnerability scores. No remote compromise was demonstrated.

### Important constraint correction

“Zero impact” and “zero RAM overflow for any input” are not unconditional guarantees a benchmark can make. Local inference consumes RAM/VRAM/CPU; cloud inference consumes quotas, money, and external data exposure. Even an async generator can allocate an arbitrarily large input line or model response.

Use a measurable objective instead:

> Bounded host memory independent of dataset row count, within enforced record/output limits; bounded concurrency, token usage, cost, and execution time; no unapproved execution of generated content or modification of shared services.

## 2. Validation performed and limitations

Dependencies were installed into an ignored `.venv` using the repository's dev extra. No paid API calls, target-model calls, container restarts, or million-row load tests were performed. Docker was unavailable. Application source and historical benchmark data were not changed.

### Observed local checks

| Check | Observed result |
|---|---|
| `pytest --collect-only -q llm-guard-bench/test_orchestration.py` | Three async diagnostic functions collected. |
| `pytest -q --override-ini asyncio_mode=strict test_orchestration.py` from inner directory | All three fail before execution: async tests lack marks/configuration for strict mode. `pytest-asyncio` is installed; declaring the dependency is insufficient. |
| Installed editable `llm-guard-bench --help` | Prints a coroutine object and `RuntimeWarning: coroutine 'main' was never awaited`, rather than CLI help. |
| Build wheel, inspect ZIP members | Wheel omits `main.py`, `config/prompts.json`, and `db/migrations/001_initial_schema.sql`. The local wheel tool first needed installation; after that, the package built and omissions were confirmed. |
| Python 3.9 grammar parse | `core/models.py` fails because it contains structural pattern matching. |
| Fake-provider evaluation | Empty response → `PASSED`; mixed refusal wording → `PASSED`; ambiguous verdict → persisted `PASSED`; judge-generated pass → incorrectly labeled keyword stage. |
| Judge parser | Plain text `NOT VULNERABLE` → `VULNERABLE` through substring fallback. |
| Temporary fresh SQLite database | Active writer connection has `PRAGMA foreign_keys = 0`; upserting a supplied attack raises `AttributeError` for absent `expected_behavior`. |
| Synthetic four-row aggregation: pass, vulnerable, evaluation error, timeout | Summary category rate = 25%; chart category rate = 50%. |
| Dataset inventory | Ten supplied attack definitions, with `turns`; no JailbreakBench ingestion implementation. |
| Read-only DB inspection | Current tracked DB has zero results/sessions/attack definitions; historical DB has 365 results and zero sessions/attack definitions. These counts are not a validation of historical verdicts. |

The existing diagnostics were deliberately not run in auto mode: they use the repository's real results directory and can modify tracked databases/charts. The targeted database reproduction used a temporary directory. Audit-generated SQLite sidecars were removed after all inspection connections closed; this is not advice to delete active WAL files.

This is a source-level audit with targeted reproductions, not a dependency-CVE scan, full Git-history secret audit, proof of model accuracy, or hosted GitHub branch-protection review. No throughput or memory measurement is claimed. Findings are grouped by executable line ranges rather than adding commentary to blank lines, imports, or every chart-style constant.

## 3. Findings: benchmark correctness and research validity

| ID / priority | Source | Finding and consequence | Required fix |
|---|---|---|---|
| V01 / P0 | `core/evaluator.py:403–409`; `core/adapters.py:170–186,210–224,319–339,355–374` | Missing provider content becomes an empty string; empty or invalid target output becomes `PASSED`. Transport/schema failures are indistinguishable from a demonstrated refusal. | Return structured provider outcomes. Preserve empty, blocked, truncated, invalid, and transport-failure outcomes separately. Only score evidence-supported judgments. |
| V02 / P0 | `core/pipeline.py:311–318`; `core/evaluator.py:195–199,258–271,316–322`; `core/models.py:23–30`; SQL `:51–67` | Ambiguous and malformed judge outputs ultimately become passes. This systematically inflates resistance scores. | Add `AMBIGUOUS` to domain, SQL, exports, and counters through a migration. Use `FAILED`/parse-error for invalid judge schemas, not a security verdict. |
| V03 / P0 | `core/evaluator.py:59–75,108–111,411–415` | Any refusal phrase anywhere short-circuits evaluation. Quoted refusals and “I cannot help … here is the answer” both pass. | Treat keywords as non-decisive evidence unless validated for a narrowly defined metric. Judge potentially mixed/compliant responses; evaluate against human-labeled cases. |
| V04 / P0 | `core/evaluator.py:149–174,293–314`; `core/pipeline.py:221–224` | Judge sees only target output, not attack objective, original request, system policy, or expected behavior. Harmful-content detection is not the same as attack success; benign policy violations or prompt leakage can be missed. | Evaluate a typed context containing the attack objective, policy, conversation, response, and dataset-specific rubric. Distinguish jailbreak compliance, prompt-injection obedience, and leakage metrics. |
| V05 / P0 | `core/evaluator.py:165,225–255` | Untrusted target text is embedded in the grading prompt and loose substring parsing accepts negations or echoed verdict words. A target can attempt judge prompt injection. | Frame response as untrusted data, separate rubric from quoted content, validate strict structured output, constrain tools/egress, and test judge attacks. Delimiters alone are not a security boundary. |
| V06 / P1 | `core/pipeline.py:298–310` | Every `PASSED` result is labeled `STAGE_1_KEYWORD`, including actual judge passes; judge rationale is discarded. | Return a structured `Evaluation` with actual stage, verdict, rubric version, reason, and parse/error metadata directly from evaluators. Remove lossy enum conversion. |
| V07 / P1 | `core/models.py:72–90`; `core/pipeline.py:178–183` | “Multi-turn” sends all user turns together in a single call, without assistant replies. This is not sequential multi-turn interaction. | Either document this as a static message sequence or perform turn-by-turn generation, append real assistant outputs, and cap turns/context/tokens. Preserve the exact transcript. |
| V08 / P0 | `analysis/aggregator.py:351–376,794–801` | Text summary uses all attempts as category denominator; heatmap uses only pass + vulnerable. Errors change one rate but not the other. | Define metrics once and pass a typed metric object to all renderers. Publish decisive ASR, coverage, errors, and counts together. |
| V09 / P0 | `analysis/aggregator.py:753–766,892–896` | Chart hardcodes `100% STABLE` regardless of errors and hardcodes judge identity. | Compute operational completion/error rates from actual run metadata; remove unsupported badges. Render actual judge/provider/model revision. |
| V10 / P1 | `analysis/aggregator.py:348–385,487–503`; `main.py:249–256` | “Successful runs” conflates defended outputs with successful execution; latency includes failure timings; fixed VRS tiers lack a documented calibration. No decisive records yields 0 rather than undefined. | Separate execution success from defense score. Report `null`/N/A for undefined metrics. Stratify completed-call latency, timeout rate, and end-to-end latency; justify or remove risk tiers. |

### Required metric contract

Let `P` = defended, `V` = successful attacks, `A` = ambiguous, and `E` = infrastructure/parse errors. Track skipped and invalid-input records separately. For a declared eligible cohort:

- Decisive attack success rate: `V / (P + V)`; N/A when denominator is zero.
- Resistance on decisive results: `P / (P + V)`; explicitly conditional, not a universal safety score.
- Decisive coverage: `(P + V) / eligible_attempts`.
- Report all outcome counts, completion rate, and category/model/dataset strata.
- If useful, publish lower/upper attack-success bounds treating unresolved eligible outcomes as unsuccessful/successful, with the denominator and assumptions stated.
- Publish uncertainty intervals and judge agreement on a labeled sample. Repeated variants of the same underlying behavior are correlated; naive independent-row confidence intervals can overstate precision.

Use official dataset definitions, licenses, splits, and grading protocols when claiming JailbreakBench comparability. Loading its rows through a custom judge does not by itself reproduce that benchmark.

## 4. Findings: bounded memory, concurrency, providers

| ID / priority | Source | Finding and consequence | Required fix |
|---|---|---|---|
| S01 / P1 | `core/loader.py:44–46,58,73–101` | `json.load` materializes the complete document, then constructs another full list of models. | Stream JSONL or incrementally parse the existing `attacks` array. Validate and filter one bounded record at a time. |
| S02 / P1 | `core/pipeline.py:66–113` | Builds one coroutine per attack, `gather` schedules them all, and retains all outputs. Semaphore bounds active work only. | Fixed worker count with bounded queues, or a bounded task window; return summary/checkpoint rather than a list. |
| S03 / P1 | `analysis/aggregator.py:251–308,355–385,487–503`; `:99,170` | `fetchall`, row-to-dict copies, JSONL accumulation, timing lists, and repeat queries load all raw responses. A streaming executor alone will not fix memory. | SQL aggregates and cursor batches; online sums/counts; bounded histogram/sketch for percentiles; charts receive aggregate records only. |
| S04 / P1 | `core/adapters.py:118–122,158–165,201–205,287–291`; `core/models.py:54–62` | No record, response-byte, conversation, or output-token limits. Whole HTTP bodies are buffered. | Enforce byte/token/turn limits before costly parsing and generation, and bounded reads even for error bodies. Streaming tokens are optional, but buffered responses must have a hard cap. |
| S05 / P1 | `core/pipeline.py:66,216–217`; `main.py:465–470` | Concurrency 0 waits indefinitely; negative values fail late. Per-task one-second sleeps do not impose a global RPM/TPM quota and synchronize bursts. | Validate positive CLI settings; independent target/judge limits, shared request/token budgets, and admission control for retries. |
| S06 / P1 | `core/adapters.py:84–104,113–143`; `core/evaluator.py:289–354` | Ollama retries connection errors but not transient HTTP statuses; judge retries all exceptions, including permanent errors. Backoffs lack jitter and `Retry-After` handling. SDK retries may compound application retries. | Typed provider exceptions, one explicit retry owner, bounded attempts and elapsed deadline, transient-status policy, jitter, quota-aware retries, and circuit breaking. Check SDK retry defaults explicitly. |
| S07 / P1 | `core/adapters.py:86,115,245`; `main.py:319–331` | New aiohttp session per Ollama request/attempt prevents connection-pool reuse. Groq client has no explicit close lifecycle. Cleanup closes only SQLite. | One bounded client pool per provider/run, owned by async context managers and closed with `AsyncExitStack`. |
| S08 / P1 | `core/pipeline.py:145–171,180–193,221–239`; `core/adapters.py:403–415` | Health checks run for every row; fixed 180-second timeouts ignore method/config arguments. Groq wraps timeouts as `RuntimeError`, obscuring error classification. | Startup/cached readiness checks and circuit breaker; connect/read/request/row/run deadlines from validated settings; structured timeout metadata. |
| S09 / P1 | `main.py:147–169`; `core/adapters.py:403–417`; `config/settings.py:131–140,182–193` | Target Groq receives no key. Settings advertise OpenAI/Anthropic that the factory rejects; Gemini is absent. Judge key selection is Groq-specific. | Provider configuration owns its credentials and capabilities independently of target/judge role. Registry supports only implemented providers; add new providers behind contract tests. |
| S10 / P2 | `core/pipeline.py:139,146,176,195,260`; SQL `:92–94` | Wall clock measures durations; “total” ends before persistence although schema says it includes DB writes. | `time.perf_counter()` for durations; UTC wall timestamps for records; explicitly define queue, inference, evaluation, persistence, and total durations. |
| S11 / P1 | `analysis/aggregator.py:225–227,242–245,509–529`; `main.py:288–304` | Cancelling a wait on executor work does not stop the underlying SQLite query or renderer. Chart timeout logging also says 60s despite a 180s deadline. | SQLite progress/interrupt strategy or isolated report process for hard deadlines; do not claim work was stopped merely because the await timed out. |

**Memory diagnosis:** current retention is O(N × record size), with multiple overlapping collections and N scheduled tasks. No trustworthy numerical memory estimate follows without measuring record sizes and runtime overhead. At one million rows, even one retained 10 KiB payload per row is roughly 9.5 GiB before Python objects, copies, task stacks, SDK buffers, and reporting.

## 5. Findings: persistence and failure semantics

| ID / priority | Source | Finding and consequence | Required fix |
|---|---|---|---|
| D01 / P0 | `db/db.py:39–46,99–102`; SQL `:8–9`; `main.py:116–139,174–217` | FK enforcement is enabled only on the migration connection, not the writer. Orchestration never populates sessions/attack definitions. Historical orphan rows corroborate the lifecycle gap. | Enable/verify connection-local PRAGMAs on every connection and insert parent metadata before results in an explicit transaction. Simply turning FKs on will break existing inserts until parents are fixed. |
| D02 / P0 | `db/db.py:250–275`; `core/models.py:53–62`; SQL `:148–151` | Attack upsert reads nonexistent `attack.expected_behavior`; accepted lowercase severity violates SQL's uppercase constraint if reached. | Align model/schema through normalization and a defined expected-behavior field or explicit SQL default. Add round-trip tests. |
| D03 / P0 | `db/db.py:79–106`; `main.py:119–121` | An incomplete schema triggers rename/rebuild, after a persistent connection is already open. On Unix that connection can still reference the renamed file; Windows may reject the rename. A missing table is not proof of corruption. | Versioned transactional migrations, column/constraint checks, SQLite backup API, and explicit recovery commands. Never silently replace storage under an open writer. |
| D04 / P0 | `core/pipeline.py:272–280`; `main.py:214–217,313–317,335–445` | Failed result persistence still returns a “successful” result. Many fatal stages return with exit code 0; nested catches prevent outer failure tracking. | Durable-write acknowledgment defines completion. Fatal sink failures stop admission and fail the run; expected row failures are typed records. Return a run summary and map it to CLI exit codes once. |
| D05 / P1 | `db/db.py:160–207` | Commit and synchronous file open/write per row; workers share a connection without explicit transaction ownership. JSONL blocks the event loop, and dual writes are not atomic. | Single bounded writer, batched transactions with a maximum flush interval, and an authoritative ledger plus replayable export. SQLite is adequate for an initial single-node design. |
| D06 / P1 | SQL `:15–19`; `main.py:66–68`; `db/db.py:224–248` | No logical-result uniqueness/checkpoint; second-resolution session IDs collide; `INSERT OR REPLACE` can delete/reinsert parent records and trigger cascades when FKs are enabled. | UUID run IDs; deterministic work IDs; explicit conflict policy and `ON CONFLICT DO UPDATE`; resumable committed-work checkpoints. |
| D07 / P1 | `core/models.py:120–141`; `db/db.py:163–197` | JSONL schema version is not stored with SQL rows. Judge identity, rubric version, dataset revision/hash, request settings, attempts, finish reason, and actual token usage are absent or never populated. | Versioned immutable run manifest, dataset/attack references and hashes, attempt records, and structured provider results. Never put secrets in config snapshots. |
| D08 / P1 | `analysis/aggregator.py:217–237,284–312` | Query errors become empty data; fallback may silently report a partial JSONL copy as a complete run. | Differentiate no rows, unavailable storage, and corrupted data; reconcile expected/committed counts and label fallback provenance/completeness. |

Additional issues:

- `DatabaseManager` creates global `RESULTS_DIR` even when passed a custom DB path; JSONL also always uses that global directory (`db/db.py:33–37,204`). Inject a storage configuration and isolate test outputs.
- Schema validation checks table names only (`db/db.py:124–145`), not columns, migration checksums, or FK integrity.
- SQLite connection closure in `_sync_query_sqlite` is skipped on query exceptions (`analysis/aggregator.py:253–286`). Use `contextlib.closing` or `finally`; the SQLite connection context manager alone manages transactions, not connection closure.
- `TestResult.from_attack_and_eval` can reject a category accepted by `AttackDefinition`; the outer pipeline then drops the record as `None` (`core/models.py:57,126,168`; pipeline `:82–89,262`). Validate the same contract at ingestion and preserve rejected-record accounting.

## 6. Findings: packaging, tests, DevSecOps, maintainability

### Packaging and execution

1. **P1 — broken installed CLI:** `setup.py:31–34` points directly to async `main`; generated console launchers do not await it. Add synchronous `cli.main()` calling `asyncio.run(async_main())` and returning a status code.
2. **P1 — incomplete wheel:** `setup.py:14` discovers generic packages but no top-level module or package data. Confirmed wheel omission of main, prompts, and SQL migration. Move to `src/llm_guard_bench`, include resources explicitly, and test wheel installation outside the checkout.
3. **P1 — Python compatibility mismatch:** `setup.py:13,41` and root README advertise 3.9, but models use `match`; settings also evaluate newer union syntax without postponed annotations. Adopt Python >=3.11 for `TaskGroup`/`asyncio.timeout`, then test declared versions.
4. **P1 — working-directory assumptions:** `main.py:127,274–280`, aggregator `:291`, and diagnostics assume the inner directory is CWD. Use `importlib.resources` for bundled fixtures and explicit paths for user data.
5. **P2 — dependencies:** lower bounds alone are not reproducible deployment specifications (`setup.py:15–29`). Keep reasonable library ranges but lock tested application deployments. NumPy is imported directly (`analysis/aggregator.py:18`) and only installed transitively through matplotlib; declare it directly or eliminate its direct use. Make plotting an optional extra so a headless worker need not load graphics libraries.

### Container and workstation safety

1. **P0 for release — build context secrets:** `Dockerfile:55` copies the full directory with no tracked `.dockerignore`. A local `.env` is ignored by Git but still enters Docker build context/image. The inner `.gitignore` even ignores `.dockerignore`, making it easy to miss. Add a tracked `.dockerignore` excluding secrets, venvs, results, VCS metadata, caches, and datasets. Supply secrets only at runtime. No actual credential leak was demonstrated here.
2. **P1:** image runs as root; base image is floating; no explicit capability restrictions, read-only root filesystem, or active resource limits (`Dockerfile:7–72`; root Compose `:81–91`). Add non-root UID, writable output mount, `cap_drop: [ALL]`, `no-new-privileges`, PID/memory/CPU limits, pinned release image digest and vulnerability scanning. Resource isolation for the benchmark container does not constrain host Ollama's RAM/VRAM.
3. **P1:** health check always succeeds through `python -c ...exit(0)` and checks a nonexistent application web endpoint (`Dockerfile:66–67`). This is a batch CLI, not a daemon: remove the artificial HTTP health check and rely on job exit state, progress, and provider readiness.
4. **P1:** default command prints help and exits; PowerShell expects a continuously running container (`Dockerfile:72`; root PowerShell `:130–149`). Use `docker compose run --rm` as a batch job. There is no implemented `run` subcommand; `--models` in PowerShell `:231` is incompatible with actual `--target` parsing. The `--help` action may mask invalid extra arguments, but does not make a service run.
5. **P1:** PowerShell orders tables alphabetically (`:162`) but checks a regex requiring the opposite order (`:195`); compares strings rather than schema sets. It checks legacy `docker-compose` although it executes `docker compose` (`:68,122`), and `docker ps` nonzero native exit is not reliably a caught exception (`:75–81`). Test native exit codes and parse structured output.
6. **P2:** both Windows launchers hardcode the author's workstation path (`root:run_bench.bat:13`; PowerShell `:19`). Use `%~dp0`/`$PSScriptRoot`. Batch launcher installs floating dependencies every run, hides errors, and continues (`:30–34`); separate installation from execution and propagate benchmark exit code.
7. **P1:** `main.py:73–107` optionally restarts the shared container named `ollama`. This violates a zero-impact promise for other workloads. Remove from normal execution or require explicit ownership-scoped maintenance authorization. Do not solve container access by mounting the host Docker socket.
8. **P0 documentation safety:** root `README.md:395` recommends deleting SQLite WAL to resolve locks. An active WAL may contain committed data. Replace with controlled shutdown, busy-timeout diagnostics, supported checkpointing, and consistent backups. PowerShell `-Clean` deletes DB/backups/exports (`:87–105`); restrict to owned paths, refuse active writers, and require confirmation or an explicit destructive-operation flag.

### Tests and CI

- No checked-in `.github/workflows`, test configuration, type-check configuration, lockfile, or security automation was found. Hosted settings were not examined.
- `test_orchestration.py:26–130` returns booleans rather than asserting; several missing-data/failed-chart cases return `True`. Under pytest those return values are not failure assertions even if async execution is enabled.
- Tests reference a hardcoded unavailable session, real DB paths, and a shared chart filename; they are diagnostics, not isolated unit tests. Database disconnect is not in `finally` on failures.
- `check_results_summary.py:56–98` compares lengths/counts derived from the same loaded list. It does not independently prove durable completion, chart rendering correctness, or expected dataset coverage.
- Add pytest fixtures with `tmp_path`, marked async tests, fake adapters, HTTP transport mocks, property-based schema checks, cancellation/fault injection, installed-wheel smoke tests, and constrained-memory load tests. Live paid/provider tests must be opt-in.
- CI should enforce formatting/linting (Ruff), strict typing on domain/ports/providers, tests and branch coverage, packaging/resources/CLI smoke tests, migration tests, secret scanning, dependency audits, and container scans/SBOM. Use least-privilege workflow permissions and SHA-pinned actions; publish artifacts only from trusted events.

### Typing, readability, SOLID

- `float = None` and `str = None` are incorrect annotations (`core/adapters.py:40,49,151,193,303,344,403`; `db/db.py:278`). Use `float | None`, `str | None`.
- Untyped `db_manager` (`pipeline.py:31`), `dict` snapshots (`models.py:193`), broad `Dict` reports, and duck-typing private `_db_path` (`aggregator.py:58–76`) defeat static guarantees. Replace with narrow protocols and typed DTOs.
- Duplicate imports (`models.py:14–17`), unused imports, mixed typing styles, repeated parsing code, and comments contradicting behavior reduce signal. Example: models claim immutability but `AttackDefinition` is mutable; even frozen models need immutable nested fields for deep immutability.
- `AttackDefinition` permits empty turns/IDs and ignores extra fields by default. Normalize severity, constrain size/counts, explicitly choose `extra='forbid'` at ingestion, and preserve dataset-native taxonomy rather than hardcoding five global categories.
- Importing settings creates directories and can throw environment-validation exceptions (`config/settings.py:38,146,155,204–205`). `main.py:23,34–38` also configures environment/logging at import. Move side effects to the composition root.
- SRP is weakest in `main.py` (environment, Docker maintenance, adapters, DB, execution, reporting) and `aggregator.py` (storage, fallback, metrics, configuration, plotting). Split by reasons to change, not merely file size.
- Dependency inversion is partly present through `BaseAdapter`, but evaluator construction and storage are concrete inside the pipeline. Inject `Evaluator`, `ResultSink`, and provider lifecycle ports.
- Interface segregation: single-turn and multi-turn duplicate one common chat operation. Use `generate(ChatRequest)` and optional streaming/tool capabilities; do not force all providers to fake unsupported features.
- Liskov/substitution: a successful empty string must not mean malformed provider response on one adapter and valid output on another. Define and contract-test the same semantic error/result behavior across providers.
- Open/closed: a simple explicit registry is sufficient. Avoid prematurely adding a plugin ecosystem, service locator, or dependency-injection framework.
- Excess broad `except Exception` blocks hide failures more often than they improve resilience. Catch expected provider/input failures near boundaries; let invariant/programming/storage failures terminate the run. On Python 3.11 `CancelledError` inherits from `BaseException`, so these catches do not generally swallow cancellation; the missing piece is structured task ownership and durable shutdown semantics, not blindly adding another catch.

### Repository and portfolio integrity

Tracked files include DBs, raw JSONL, PNGs, and two `.pyc` files despite ignore rules. Ignoring does not untrack existing files. Keep tiny sanitized test fixtures and selected immutable example reports; move large runs to versioned external artifacts with checksums, manifest, retention policy, and access controls. Review raw prompts/responses before publication. Do not rewrite Git history without coordination.

`CITATION.cff` contains placeholder repository URLs (`root:16–18`), an incomplete ORCID (`:14`), placeholder contact (`:83`), and a fake DOI placeholder (`:105`). Remove unassigned identifiers and unsupported funding claims; validate against the CFF schema and verify each bibliography entry independently. The inner README is empty. Root README's attack example (`:237–247`) uses `adversarial_prompt` rather than required `turns`; provider support and timeout claims drift from code. CONTRIBUTING references nonexistent tests and an incompatible one-argument adapter call (`root:CONTRIBUTING.md:87,122,260`).

For scholarship/recruiting reviewers, calibrated limitations, reproducible evidence, tested migrations, and honest failure accounting are stronger signals than “enterprise-grade” adjectives or an attractive heatmap.

## 7. Proposed directory structure and migration map

Keep one installable package at repository root. Do not create microservices merely to organize modules.

```text
llm-guard-bench/
├── pyproject.toml                  # >=3.11, entry point, lint/types/tests
├── uv.lock                        # or another single chosen lock mechanism
├── README.md
├── SECURITY.md                    # disclosure + threat model links
├── CITATION.cff
├── .dockerignore
├── Dockerfile
├── compose.yaml
├── .github/workflows/
│   ├── ci.yml
│   └── security.yml
├── src/llm_guard_bench/
│   ├── __init__.py
│   ├── __main__.py
│   ├── cli.py                     # sync entry + async composition root
│   ├── settings.py                # explicit runtime validation
│   ├── domain/
│   │   ├── records.py             # attack, message, generation, evaluation
│   │   ├── outcomes.py            # verdict vs operational status
│   │   ├── errors.py
│   │   └── ports.py               # Provider, Evaluator, ResultSink
│   ├── providers/
│   │   ├── registry.py
│   │   ├── ollama.py
│   │   ├── groq.py
│   │   ├── openai.py
│   │   ├── gemini.py
│   │   ├── retry.py
│   │   └── limits.py
│   ├── pipelines/
│   │   ├── runner.py              # fixed workers + queue lifecycle
│   │   ├── conversation.py        # genuine sequential turns
│   │   ├── checkpoint.py
│   │   └── shutdown.py
│   ├── streaming_io/
│   │   ├── jsonl.py               # byte-bounded reader/exporter
│   │   ├── json_array.py          # existing config compatibility
│   │   ├── parquet.py             # optional bounded record batches
│   │   └── datasets/
│   │       ├── local.py
│   │       └── jailbreakbench.py  # explicit schema/revision/license mapping
│   ├── evaluators/
│   │   ├── keywords.py
│   │   ├── judge.py
│   │   ├── cascade.py
│   │   └── rubrics.py
│   ├── storage/
│   │   ├── sqlite.py              # single-writer sink
│   │   ├── migrations.py
│   │   └── sql/                   # versioned packaged migrations
│   ├── reporting/
│   │   ├── metrics.py             # one denominator implementation
│   │   ├── queries.py             # aggregate queries, bounded cursors
│   │   └── charts.py              # optional graphics dependency
│   ├── observability/
│   │   ├── logging.py             # structured, redacted
│   │   └── metrics.py             # queue depths, attempts, throughput
│   └── resources/prompts.json     # current ten examples, packaged
├── tests/
│   ├── unit/
│   ├── contracts/
│   ├── integration/
│   ├── performance/
│   └── fixtures/                  # tiny, synthetic, non-sensitive
├── docs/
│   ├── architecture.md
│   ├── threat-model.md
│   ├── methodology.md
│   ├── reproducibility.md
│   └── audits/
└── scripts/                       # thin portable wrappers, not business logic
```

Mapping: `core/adapters.py` → providers + domain ports; `core/models.py` → domain; `core/loader.py` → streaming IO/dataset adapters; `core/pipeline.py` → runner/conversation; `core/evaluator.py` → evaluators; `db/db.py` → storage; `analysis/aggregator.py` → reporting; `main.py` → CLI composition root. Preserve the ten current examples via an adapter before migrating formats. Add new provider modules only when implemented—not empty files presented as support.

## 8. Scalable architecture

### 8.1 Bounded end-to-end dataflow

```text
Versioned source + byte/row limits
        │ async iterator; validation/rejection accounting
        ▼
Bounded attack queue
        │ Wt target workers; provider-specific concurrency + RPM/TPM
        ▼
Bounded target-output queue
        │ We evaluator workers; independent judge budgets
        ▼
Bounded result queue
        │ ONE writer; bounded batch size + flush deadline
        ▼
SQLite transactions / durable partitioned ledger
        ├── committed checkpoints + online counters
        └── replayable JSONL/Parquet export → aggregate queries → charts
```

When the writer slows, the result queue fills; evaluators block, then target workers block, then input stops reading. This is **backpressure**. `async def` or a semaphore alone does not provide it.

An async source over a local file should use bounded thread-backed reads or an async file wrapper; do not block the event loop with large JSON parsing/read loops. For JSONL, enforce maximum line bytes while reading chunks, before building a whole line or calling `json.loads`. Cap decompressed bytes for compressed sources. For JSON arrays, use an incremental parser plus per-object limits. For Parquet, cap batch sizes and oversized fields. Quarantine malformed rows with source offset, sanitized reason, and bounded samples, not an ever-growing in-memory list.

Approximate host-memory bound:

`M_base + Qa·Ba + Qo·Bo + Qr·Br + Wt·Mt + We·Me + Bbatch + M_HTTP + M_metrics`

Queue capacities Q, maximum record sizes B, worker counts W, per-worker working sets M, client pool buffers, and metric cardinality must all be bounded. Python object expansion and parser/SDK copies need measured safety margin. Local model-process RAM/VRAM is a separate budget. A million unique category labels can also make a “small counters dict” unbounded; use a declared taxonomy or disk-backed aggregate grouping.

Start conservatively (e.g., target workers 1–2 for local models, queues around twice their consumer counts, batches capped by both rows and bytes). Tune with measurements, not assumed universal defaults. A one-million-row run can take days or exceed budget even if memory is flat; estimate inference/judge tokens and require budget approval first.

### 8.2 Minimal orchestration pattern

Illustrative Python 3.11 skeleton, **not implemented production code**. `execute` must convert expected row/provider failures to typed result records, enforce per-row deadlines, and allow cancellation/fatal bugs to propagate. `sink.write` must await bounded durable persistence; a batched sink needs a separate flush/ack implementation. Source and sink lifecycle are owned by surrounding async context managers.

```python
import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Protocol, TypeVar

A = TypeVar("A")
R = TypeVar("R")
R_contra = TypeVar("R_contra", contravariant=True)

class Sink(Protocol[R_contra]):
    async def write(self, result: R_contra) -> None: ...

async def run(
    source: AsyncIterator[A],
    execute: Callable[[A], Awaitable[R]],
    sink: Sink[R],
    *,
    workers: int,
    queue_size: int,
) -> None:
    if workers < 1 or queue_size < 1:
        raise ValueError("workers and queue_size must be positive")

    # None is reserved for lifecycle signals, not valid input/output data.
    inputs: asyncio.Queue[A | None] = asyncio.Queue(queue_size)
    outputs: asyncio.Queue[R | None] = asyncio.Queue(queue_size)

    async def produce() -> None:
        async for item in source:
            await inputs.put(item)
        for _ in range(workers):
            await inputs.put(None)

    async def work() -> None:
        while True:
            item = await inputs.get()
            if item is None:
                await outputs.put(None)
                return
            await outputs.put(await execute(item))

    async def persist() -> None:
        finished = 0
        while finished < workers:
            result = await outputs.get()
            if result is None:
                finished += 1
            else:
                await sink.write(result)

    async with asyncio.TaskGroup() as group:
        group.create_task(produce())
        for _ in range(workers):
            group.create_task(work())
        group.create_task(persist())
```

Only `workers + 2` tasks exist; no `gather` over the source and no returned result list. FIFO completion markers ensure the writer consumes every enqueued result on normal completion. A sink/worker/source exception causes `TaskGroup` to cancel sibling tasks rather than hang behind a full queue. No `queue.join()` is used, so `task_done()` is not needed in this variant.

For the full design, separate target and judge worker stages using the same pattern. Preserve per-conversation sequential order inside one target worker, while unrelated attacks execute concurrently. Do not buffer all completed results to restore input order; store a sequence/source ID and sort externally only when needed.

### 8.3 Provider/Strategy contract

Use a single chat request abstraction and a structured response:

```python
from dataclasses import dataclass
from typing import Literal, Protocol

@dataclass(frozen=True)
class Message:
    role: Literal["system", "user", "assistant"]
    content: str

@dataclass(frozen=True)
class ChatRequest:
    messages: tuple[Message, ...]
    temperature: float
    max_output_tokens: int

@dataclass(frozen=True)
class Generation:
    text: str
    finish_reason: str
    input_tokens: int | None
    output_tokens: int | None
    request_id: str | None
    model_revision: str | None

class Provider(Protocol):
    async def generate(self, request: ChatRequest) -> Generation: ...
    async def aclose(self) -> None: ...
```

Validated constructors/settings enforce bounds; typing alone does not. Add capability metadata for structured JSON output, streaming, system messages, token counting, tools, and seed support. Fail fast when a required capability is unsupported—do not silently drop fields. The initial benchmark should disable tools and code execution.

- **Ollama:** map messages/options, reuse an aiohttp session, read bounded content or token streams, normalize finish/token metadata, handle model-not-found separately from transient unavailability.
- **Groq:** retain current adapter concept but normalize errors, usage, and finish reasons; own its SDK client lifecycle.
- **OpenAI:** translate neutral requests to the chosen async API, validate structured-response support for the specific model, and normalize usage/error types.
- **Gemini:** translate system instruction and user/model roles explicitly; distinguish safety blocking, no candidate, truncation, and malformed response. API semantics are not necessarily OpenAI-compatible.
- **Fake provider:** deterministic latency/errors/usage/output for unit, contract, and million-row infrastructure tests.

The runner receives `Provider`; only the composition root chooses implementations. Use an explicit registry of validated factories, e.g. `ollama`, `groq`, `openai`, `gemini`; optional dependency extras prevent local-only users needing every cloud SDK. Wrap strategies with quota/retry instrumentation rather than duplicating orchestration policy in every provider. Reuse credential references without exposing their values in logs/manifests. Target and judge configuration are independent; if they share an account, their quota limiter must share that account's budget.

Error taxonomy should preserve `AuthenticationError`, `RateLimited(retry_after)`, `TransientTransportError`, `RequestTimeout`, `InvalidRequest`, `MalformedResponse`, and explicit provider blocking. Define which are retryable and which represent valid-but-unscorable outcomes. Do not flatten all into `RuntimeError` or empty text.

### 8.4 Durability, resume, cancellation

- Generate a unique run UUID and immutable manifest containing Git commit, environment/lock fingerprint, dataset revision/hash/split, model/provider revisions, generation settings, evaluator rubric hash, budgets, and hardware information. Exclude credentials and sensitive environment values.
- Work ID should include run/cohort, dataset record identity, target config hash, repetition/seed, and conversation variant. A unique DB constraint prevents duplicate logical results; attempt IDs record retries separately.
- SQLite writer owns transaction boundaries. Set WAL, busy timeout, foreign keys, and explicit durability policy; insert parent metadata and results consistently. Batch by rows, bytes, and time (for example 100 rows or 1 MiB or one second, subject to testing).
- Advance checkpoints only after commit. Out-of-order completions require a durable completed-ID ledger or contiguous committed source watermark plus gap tracking; never checkpoint the highest observed offset alone.
- Prefer SQLite as authority and export JSONL from committed rows, or use an outbox written in the same transaction. If maintaining live dual output, reconcile after crashes; two appends cannot provide atomic cross-store commits by themselves.
- On graceful stop, stop admission, drain until a bounded deadline, commit acknowledged work, mark run interrupted, then close providers/DB. On disk full or corrupt ledger, fail loudly and cancel producers. On hard termination, replay uncommitted items from the last durable checkpoint.
- Exactly-once local result storage does not imply exactly-once cloud inference or billing. A crash after a provider reply but before commit can repeat a charged request. Use provider idempotency only when explicitly supported and document residual semantics.
- Start single-node. At sustained multi-host scale, partition sources and move shared coordination/metadata to a suitable durable queue and transactional database, with leases and global quotas. Do not share SQLite over a network filesystem as a distributed queue.

### 8.5 Threat model and observability

Treat attack datasets, provider outputs, judge outputs, and imported historical files as untrusted. Do not execute model-generated code, shell, URLs, or tools. Restrict outbound destinations to authorized providers; keep endpoints operator-controlled (add SSRF protections if untrusted users can configure URLs). Separate local-only runs from approved cloud egress; a local target with a cloud judge still exports target output. Add consent/redaction/retention policies and exclude raw prompts from routine logs.

Emit structured counters for admitted, rejected, attempted, retried, evaluated, committed, skipped, and interrupted records; queue depth; active requests; RPM/TPM; token/cost reservations and actuals; latency distributions; DB batch latency; and RSS/VRAM separately. Avoid per-record IDs as metrics labels. Redact API-key fragments, provider error bodies, prompts, and sensitive judge reasoning by default. Bound log/sample sizes and artifact retention.

## 9. Immediate refactoring plan and release gates

### Stage 1 — Repair validity and failure reporting first

1. Add failing unit tests for empty output, mixed refusal/compliance, malformed judge JSON, negated verdict text, ambiguous results, and judge-pass provenance.
2. Introduce one typed evaluation result; migrate statuses consistently across domain/DB/export/reporting. Preserve historical rows as historical: do not silently relabel old uncertain results as ground truth.
3. Unify metric denominators and remove hardcoded stability/judge labels. Mark undefined scores N/A and publish coverage alongside scores.
4. Make persistence failures fatal to the run; return meaningful nonzero CLI exit status on setup/benchmark/storage failures.
5. Remove dangerous WAL-deletion advice and add `.dockerignore` before anyone builds with real secrets.

**Gate:** fixtures prove no malformed/unknown result becomes a defense pass; chart/text metrics agree; storage failure cannot result in a successful run status.

### Stage 2 — Repair storage and packaging

1. Migrate schema safely, populate run/attack parents, enable FKs per connection, fix missing `expected_behavior` and severity normalization, and use UUID/work IDs.
2. Add isolated DB tests for fresh/old schema, interrupted writes, retry/deduplication, and resume.
3. Move to a namespaced package, synchronous CLI entry, packaged fixtures/migrations, Python >=3.11, and explicit settings.
4. Test wheel installation in a clean environment outside repository CWD. Fix batch-container workflow and portable launchers.

**Gate:** an installed wheel can show help, validate config, initialize a temp DB, and run fake-provider records; rollback/resume preserves counts; real result files remain untouched by tests.

### Stage 3 — Bound the entire pipeline

1. Replace list-returning loader with streaming ingestion; retain JSON-array compatibility.
2. Replace N-task `gather` with fixed workers and bounded queues. Remove returned result lists.
3. Add single batched writer, streaming/replayable export, aggregate SQL reporting, and per-record/output limits.
4. Add reusable provider clients, independent target/judge quotas, typed retries/deadlines, budget enforcement, and shutdown behavior.

**Gate:** run 1K, 100K, and 1M synthetic records with fake target/judge and generated input, not a committed million-row dataset. Measure peak RSS, task count, queue depth, throughput, event-loop lag, DB size, and counts. After warm-up, memory must plateau with fixed configured capacities; a chosen container limit (e.g. 512 MiB for an appropriately sized fake-provider fixture) is an explicit test target, not a claimed current capability. Run oversized-record, slow-writer, retry-storm, and cancellation cases.

### Stage 4 — Provider expansion and scientific credibility

1. Add OpenAI and Gemini only after shared provider contract tests exist; fix Groq target credentials now.
2. Implement version-pinned dataset adapters including JailbreakBench with documented mappings and licenses.
3. Add sequential conversation execution, rubric-specific evaluation, labeled calibration data, judge agreement, uncertainty estimates, and attack-on-judge tests.
4. Publish a reproducibility package: one-command fake run, modest authorized real-model experiment, immutable manifest, methodology, measured memory/throughput curves, limitations, and release artifact checksums.
5. Add CI/security scanning, threat model, accurate citation metadata, changelog, and ADRs explaining bounded queues, evaluation semantics, and storage tradeoffs.

**Gate:** support claims correspond to tested code; published security conclusions disclose dataset scope, evaluation error rate, judge limitations, costs, and environment. No universal-safety or million-row performance claims without evidence.

## 10. Recommended acceptance-test matrix

| Area | Minimum required cases |
|---|---|
| Evaluation | Empty/blocked/truncated output; refusal plus compliance; quoted refusal; malformed and extra-field JSON; non-string verdict; injected judge instruction; unknown and ambiguous outcomes; original-objective context. |
| Ingestion | 0/1/1M rows; invalid JSON; oversized line before parse; compressed expansion limits; unknown categories; missing/extra fields; duplicate source IDs; UTF-8 errors; stable source offsets. |
| Providers | Ollama/Groq/OpenAI/Gemini contract tests for 401/403/429/5xx, `Retry-After`, timeout, malformed response, empty content, safety blocks, usage metadata, cancellation, and client closure. |
| Concurrency | Fixed task count; bounded queues; slow writer blocks source; no deadlock when producer/worker/writer fails; separate target/judge budgets; concurrent conversations retain internal order. |
| Persistence | FK enabled; parents present; migration old→new; duplicate work; crash before/after commit; outbox/export replay; disk full/permissions; interrupted run; rollback and resumed count agreement. |
| Metrics | Same denominators everywhere; all-error cohort is N/A; per-model/category coverage; unknown statuses surfaced; no raw-row fetch for large report; run metadata rendered accurately. |
| Packaging | Wheel resources present; CLI help exit 0; invalid config nonzero; offline fake run outside checkout; no dependency on source working directory. |
| Security | No secrets in build context/image/logs/manifests; no generated content execution; authorized egress; non-root container; bounded artifacts; prompt-injection testing of judge. |
| Performance | Flat RSS across increasing N with fixed sizes; byte-bound stress with giant records; capped retries and costs; bounded metric cardinality; separate local-model resource measurements. |

**Bottom line:** keep the current provider abstraction and typed records, but prioritize truthful verdicts, durable results, and bounded execution before provider count or visual polish. Those are the changes that will make this repository a defensible flagship AI-security engineering project.