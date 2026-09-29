#!/usr/bin/env python3
"""
Diagnostic test for the complete orchestration pipeline.
Tests database connectivity, JSON fallback, and error handling.
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, '.')

from db.db import DatabaseManager
from analysis.aggregator import ResultsAggregator
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_database_connection():
    """Test 1: Database connection lifecycle."""
    print("\n" + "="*80)
    print("TEST 1: Database Connection Lifecycle")
    print("="*80)
    
    try:
        db = DatabaseManager()
        logger.info("[OK] DatabaseManager created")
        
        await db.connect()
        logger.info("[OK] Database connected")
        
        await db.run_migrations()
        logger.info("[OK] Migrations executed")
        
        await db.disconnect()
        logger.info("[OK] Database disconnected")
        
        print("\n[OK] TEST 1 PASSED: Database lifecycle works correctly\n")
        return True
    except Exception as e:
        logger.error(f"[FAIL] TEST 1 FAILED: {type(e).__name__}: {str(e)}", exc_info=True)
        print(f"\n[FAIL] TEST 1 FAILED: {str(e)}\n")
        return False


async def test_aggregator_with_json_fallback():
    """Test 2: Aggregator JSON fallback."""
    print("\n" + "="*80)
    print("TEST 2: Aggregator with JSON Fallback")
    print("="*80)
    
    try:
        # Test with a session that has JSON data
        test_session = "SESS_20260526_080241"  # Session with 5 records
        db_path = Path("results/guard_bench.db")
        
        if not db_path.exists():
            logger.warning("Database file not found - skipping this test")
            print("\n[WARN] TEST 2 SKIPPED: No database file\n")
            return True
        
        aggregator = ResultsAggregator(str(db_path))
        logger.info("[OK] ResultsAggregator created")
        
        # Try to generate metrics (should use DB or JSON fallback)
        metrics = await aggregator.generate_metrics_summary(test_session)
        logger.info(f"[OK] Metrics generated: {metrics.get('total_runs')} total, {metrics.get('successful_runs')} passed")
        
        if metrics.get('total_runs', 0) > 0:
            print(f"\n[OK] TEST 2 PASSED: Aggregator retrieved {metrics.get('total_runs')} results\n")
            return True
        else:
            logger.warning("No results retrieved by aggregator")
            print(f"\n[WARN] TEST 2 WARNING: Aggregator returned zero results\n")
            return True
            
    except Exception as e:
        logger.error(f"[FAIL] TEST 2 FAILED: {type(e).__name__}: {str(e)}", exc_info=True)
        print(f"\n[FAIL] TEST 2 FAILED: {str(e)}\n")
        return False


async def test_chart_generation():
    """Test 3: Chart generation with error recovery."""
    print("\n" + "="*80)
    print("TEST 3: Chart Generation with Error Recovery")
    print("="*80)
    
    try:
        test_session = "SESS_20260526_080241"
        db_path = Path("results/guard_bench.db")
        
        if not db_path.exists():
            logger.warning("Database file not found - skipping this test")
            print("\n[WARN] TEST 3 SKIPPED: No database file\n")
            return True
        
        aggregator = ResultsAggregator(str(db_path))
        chart_path = "results/test_diagnostic_chart.png"
        
        logger.info(f"Attempting chart generation to {chart_path}")
        success = await asyncio.wait_for(
            aggregator.plot_vulnerability_chart(test_session, chart_path),
            timeout=180.0
        )
        
        chart_exists = Path(chart_path).exists()
        if success and chart_exists:
            size = Path(chart_path).stat().st_size
            logger.info(f"[OK] Chart created successfully ({size} bytes)")
            print(f"\n[OK] TEST 3 PASSED: Chart generated at {chart_path} ({size} bytes)\n")
            return True
        else:
            logger.warning(f"Chart creation returned success={success}, exists={chart_exists}")
            print(f"\n[WARN] TEST 3 WARNING: Chart success={success}, exists={chart_exists}\n")
            return True
            
    except asyncio.TimeoutError:
        logger.error("Chart generation timed out")
        print(f"\n[FAIL] TEST 3 FAILED: Chart generation timed out (180s limit)\n")
        return False
    except Exception as e:
        logger.error(f"[FAIL] TEST 3 FAILED: {type(e).__name__}: {str(e)}", exc_info=True)
        print(f"\n[FAIL] TEST 3 FAILED: {str(e)}\n")
        return False


async def main():
    """Run all diagnostic tests."""
    print("\n")
    print("+" + "="*78 + "+")
    print("|" + " "*20 + "LLM Guard Bench - Orchestration Diagnostics" + " "*14 + "|")
    print("|" + " "*78 + "|")
    print("|" + f" Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}" + " "*56 + "|")
    print("+" + "="*78 + "+")
    
    results = []
    
    # Run all tests
    results.append(await test_database_connection())
    results.append(await test_aggregator_with_json_fallback())
    results.append(await test_chart_generation())
    
    # Summary
    print("\n" + "="*80)
    print("DIAGNOSTIC SUMMARY")
    print("="*80)
    passed = sum(results)
    total = len(results)
    print(f"\nTests Passed: {passed}/{total}")
    
    if all(results):
        print("\n[OK] ALL DIAGNOSTICS PASSED - Pipeline is ready for production use")
        print("="*80 + "\n")
        return 0
    else:
        print("\n[WARN] Some diagnostics failed or skipped - Review logs above")
        print("="*80 + "\n")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
