#!/usr/bin/env python3
"""Print and validate the result counts used by the security audit chart."""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

from analysis.aggregator import ResultsAggregator


DEFAULT_DB_PATH = Path(__file__).parent / "results" / "guard_bench.db"


def load_session_id(db_path: Path, requested_session: str | None) -> str:
    """Return the requested session or the most recently recorded session."""
    if requested_session:
        return requested_session

    with sqlite3.connect(db_path) as connection:
        row = connection.execute(
            """
            SELECT session_id
            FROM test_results
            ORDER BY timestamp DESC
            LIMIT 1
            """
        ).fetchone()

    if not row:
        raise RuntimeError(f"No results found in {db_path}")
    return row[0]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check database counts against the chart summary aggregation"
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB_PATH,
        help=f"SQLite database path (default: {DEFAULT_DB_PATH})",
    )
    parser.add_argument(
        "--session-id",
        help="Session to inspect (default: most recently recorded session)",
    )
    args = parser.parse_args()

    if not args.db.exists():
        raise FileNotFoundError(f"Results database not found: {args.db}")

    session_id = load_session_id(args.db, args.session_id)
    aggregator = ResultsAggregator(args.db)
    results = aggregator._sync_query_sqlite(session_id)
    metrics = aggregator._compute_metrics(results, session_id)
    category_data = aggregator._aggregate_by_category(results)

    total_records = len(results)
    status_counts = metrics["status_counts"]
    counted_statuses = sum(status_counts.values())

    print(f"Database: {args.db}")
    print(f"Session: {session_id}")
    print(f"Total records: {total_records}")

    print("\nBreakdown by evaluation_status:")
    for status, count in status_counts.items():
        print(f"  {status}: {count}")

    print("\nBreakdown by category:")
    for category, counts in sorted(category_data.items()):
        valid = counts["PASSED"] + counts["VULNERABLE"]
        no_data = "NO DATA (all errors)" if valid == 0 else "data available"
        print(
            f"  {category}: total={counts['total']}, "
            f"passed={counts['PASSED']}, vulnerable={counts['VULNERABLE']}, "
            f"errors={counts['errors']} [{no_data}]"
        )

    print("\nSummary chart values:")
    print(f"  TOTAL: {metrics['total_runs']}")
    for status, count in status_counts.items():
        print(f"  {status}: {count}")

    checks = {
        "database total equals metrics total": total_records == metrics["total_runs"],
        "status counts sum to total": counted_statuses == total_records,
        "chart TOTAL equals database total": metrics["total_runs"] == total_records,
        "all configured categories are represented": set(
            aggregator._load_configured_categories()
        ).issubset(category_data),
    }

    print("\nSanity checks:")
    for description, passed in checks.items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {description}")

    if not all(checks.values()):
        raise AssertionError(f"Summary sanity check failed: {checks}")

    print("\nAll summary counts match the values used by the chart.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
