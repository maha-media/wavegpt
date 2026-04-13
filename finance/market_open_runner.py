"""
Market Open Runner — orchestrates the full trading sequence.

Leave this running overnight. It will:
  1. Sleep until 9:25 AM ET
  2. Run live_trader.py (dry run) to verify connection + show state
  3. Sleep until 9:35 AM ET
  4. Run fill_monitor.py --execute to handle unfilled Friday orders
  5. Launch stream_trader.py for continuous regime monitoring

Usage:
    python finance/market_open_runner.py                  # sandbox, execute fills
    python finance/market_open_runner.py --dry-run        # sandbox, no execution
    python finance/market_open_runner.py --execute --live  # PRODUCTION (careful!)
"""

import argparse
import asyncio
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

ET = ZoneInfo('America/New_York')
FINANCE_DIR = Path(__file__).parent


def now_et():
    return datetime.now(ET)


def sleep_until(target_hour, target_minute):
    """Sleep until the next occurrence of HH:MM ET."""
    now = now_et()
    target = now.replace(hour=target_hour, minute=target_minute, second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=1)

    wait = (target - now).total_seconds()
    print(f"  Current time: {now.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(f"  Target:       {target.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(f"  Sleeping {wait/3600:.1f} hours ({wait:.0f}s)...")
    print(f"  Will wake at {target.strftime('%H:%M ET')}\n")

    # Sleep in chunks so Ctrl+C works and we show periodic heartbeats
    end = time.time() + wait
    heartbeat_interval = 600  # 10 minutes
    last_heartbeat = time.time()

    while time.time() < end:
        remaining = end - time.time()
        if remaining <= 0:
            break
        chunk = min(remaining, 10)  # wake every 10s to check Ctrl+C
        time.sleep(chunk)

        # Heartbeat
        if time.time() - last_heartbeat > heartbeat_interval:
            r = end - time.time()
            print(f"  [{now_et().strftime('%H:%M:%S ET')}] "
                  f"sleeping... {r/60:.0f} min remaining until {target.strftime('%H:%M ET')}")
            last_heartbeat = time.time()


def run_script(script_name, args_list, label):
    """Run a finance script and stream its output."""
    cmd = [sys.executable, '-u', str(FINANCE_DIR / script_name)] + args_list
    print(f"\n{'='*70}")
    print(f"  [{now_et().strftime('%H:%M:%S ET')}] RUNNING: {label}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"{'='*70}\n")

    result = subprocess.run(cmd, cwd=str(FINANCE_DIR))
    print(f"\n  [{now_et().strftime('%H:%M:%S ET')}] {label} finished (exit code {result.returncode})")
    return result.returncode


def run_script_background(script_name, args_list, label):
    """Launch a finance script as a background process (returns Popen)."""
    cmd = [sys.executable, '-u', str(FINANCE_DIR / script_name)] + args_list
    print(f"\n{'='*70}")
    print(f"  [{now_et().strftime('%H:%M:%S ET')}] LAUNCHING: {label}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"{'='*70}\n")

    proc = subprocess.Popen(cmd, cwd=str(FINANCE_DIR))
    return proc


def main():
    parser = argparse.ArgumentParser(description='Market Open Runner')
    parser.add_argument('--dry-run', action='store_true',
                        help='Do not execute orders (show what would happen)')
    parser.add_argument('--live', action='store_true',
                        help='Use production (not sandbox)')
    parser.add_argument('--skip-wait', action='store_true',
                        help='Skip sleeping, run everything immediately (for testing)')
    parser.add_argument('--fill-only', action='store_true',
                        help='Only run fill monitor, skip stream trader')
    args = parser.parse_args()

    dry_run = args.dry_run
    live_flag = ['--live'] if args.live else []
    execute_flag = [] if dry_run else ['--execute']

    print("=" * 70)
    print(f"  MARKET OPEN RUNNER")
    print(f"  {now_et().strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(f"  Mode: {'DRY RUN' if dry_run else 'EXECUTING'}")
    print(f"  Env:  {'PRODUCTION' if args.live else 'SANDBOX'}")
    print(f"  Skip wait: {args.skip_wait}")
    print("=" * 70)

    # --- Phase 1: Sleep until 9:25 AM ET ---
    if not args.skip_wait:
        now = now_et()
        # Check if we're already past 9:25
        market_check = now.replace(hour=9, minute=25, second=0, microsecond=0)
        if now >= market_check:
            # Already past 9:25 today
            if now.hour < 16:
                print(f"\n  Already past 9:25 ET — market is open, running immediately")
            else:
                print(f"\n  Market closed for today. Sleeping until tomorrow 9:25 ET...")
                sleep_until(9, 25)
        else:
            print(f"\n  Phase 1: Sleeping until 9:25 AM ET (pre-open connection check)...")
            sleep_until(9, 25)

    # --- Phase 2: Pre-open connection check ---
    print(f"\n  Phase 2: Pre-open connection check")
    rc = run_script('live_trader.py', live_flag, 'Pre-Open Check (dry run)')
    if rc != 0:
        print(f"\n  WARNING: live_trader dry run failed (exit {rc})")
        print(f"  Token may be expired. Check .env and re-authenticate.")
        print(f"  Continuing anyway in case it's a transient error...\n")

    # --- Phase 3: Sleep until 9:35 AM ET ---
    if not args.skip_wait:
        now = now_et()
        fill_time = now.replace(hour=9, minute=35, second=0, microsecond=0)
        if now < fill_time:
            remaining = (fill_time - now).total_seconds()
            print(f"\n  Phase 3: Waiting {remaining:.0f}s until 9:35 ET for fill monitor...")
            time.sleep(max(remaining, 0))

    # --- Phase 4: Fill monitor ---
    print(f"\n  Phase 4: Fill monitor")
    fill_args = execute_flag + live_flag
    rc = run_script('fill_monitor.py', fill_args, f'Fill Monitor ({" ".join(fill_args) or "dry run"})')
    if rc != 0:
        print(f"\n  WARNING: fill_monitor failed (exit {rc})")

    # --- Phase 5: Trading system ---
    if not args.fill_only:
        print(f"\n  Phase 5: Starting trading system...")
        stream_args = execute_flag + live_flag

        # Launch stream trader (core portfolio)
        stream_proc = run_script_background(
            'stream_trader.py', stream_args,
            f'Stream Trader ({" ".join(stream_args) or "dry run"})'
        )

        # Launch sentinel (social monitoring)
        sentinel_proc = run_script_background(
            'sentinel.py', live_flag,
            'Sentinel (social monitoring)'
        )

        # Launch speculator (autonomous spec trading)
        spec_args = execute_flag + live_flag
        spec_proc = run_script_background(
            'speculator.py', spec_args,
            f'Speculator ({" ".join(spec_args) or "dry run"})'
        )

        # Wait for all — stream_trader is the primary, others follow
        procs = [
            ('Stream Trader', stream_proc),
            ('Sentinel', sentinel_proc),
            ('Speculator', spec_proc),
        ]

        try:
            # Wait for stream_trader (the main loop) — if it exits, stop everything
            stream_proc.wait()
        except KeyboardInterrupt:
            print(f"\n  Ctrl+C — stopping all processes...")
        finally:
            for name, proc in procs:
                if proc.poll() is None:
                    proc.terminate()
                    print(f"  Terminated {name}")
    else:
        print(f"\n  Skipping stream trader (--fill-only)")

    print(f"\n  [{now_et().strftime('%H:%M:%S ET')}] Market Open Runner finished.")


if __name__ == '__main__':
    main()
