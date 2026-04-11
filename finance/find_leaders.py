"""
Find what LEADS tech. Not what correlates — what PREDICTS.

For each candidate leading indicator, test:
  1. Does it move BEFORE tech? (lead-lag at 1-5 day lags)
  2. Does it predict tech REGIME CHANGES? (fear->bull, bull->fear)
  3. How much earlier does it move? (lead time)

Leading indicator candidates:
  - Bitcoin (24/7 market, reacts first)
  - Semiconductors SMH (supply chain leads)
  - Dollar UUP (inverse risk)
  - ARKK (speculative froth indicator)
  - Credit LQD/HYG spread (institutional risk appetite)
  - China FXI/KWEB (global risk, overnight signal)
  - Yield curve slope
"""

import json
import sys
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

DATA_DIR = Path(__file__).parent / 'data'
RESULTS_DIR = Path(__file__).parent / 'training_results'

TECH7 = ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 'META', 'TSLA']

LEADERS = {
    'BTC-USD': 'Bitcoin',
    'ETH-USD': 'Ethereum',
    'UUP': 'Dollar',
    'SMH': 'Semiconductors',
    'SOXX': 'Semis (PHLX)',
    'ARKK': 'Speculative',
    'HYG': 'High Yield',
    'LQD': 'Inv Grade',
    'FXI': 'China',
    'KWEB': 'China Internet',
    '^TNX': '10yr Yield',
    '^VIX': 'VIX',
}


def main():
    # Download
    all_tickers = list(LEADERS.keys()) + TECH7
    print("Downloading data...")
    data = yf.download(all_tickers, period='5y', interval='1d', auto_adjust=True)
    closes = data['Close'].dropna(how='all')
    returns = closes.pct_change()

    # Tech equal-weight return (what we're trying to predict)
    tech_ret = returns[TECH7].mean(axis=1)
    next_tech = tech_ret.shift(-1)  # tomorrow's tech return

    T = len(tech_ret)
    print(f"  {T} trading days")

    print("\n" + "=" * 70)
    print("1. LEAD-LAG: Does indicator at t predict tech at t+N?")
    print("=" * 70)

    print(f"\n  {'Indicator':<20} {'t+1':>7} {'t+2':>7} {'t+3':>7} {'t+5':>7} {'Best Lag':>9}")
    print(f"  {'-' * 60}")

    best_leaders = []
    for sym, name in LEADERS.items():
        if sym not in returns.columns:
            continue

        corrs = {}
        for lag in [1, 2, 3, 5]:
            # indicator at time t vs tech return at t+lag
            ind = returns[sym].values[:-lag]
            tech = tech_ret.values[lag:]
            valid = ~np.isnan(ind) & ~np.isnan(tech)
            if valid.sum() > 100:
                c = np.corrcoef(ind[valid], tech[valid])[0, 1]
                corrs[lag] = float(c) if not np.isnan(c) else 0
            else:
                corrs[lag] = 0

        best_lag = max(corrs, key=lambda k: abs(corrs[k]))
        best_corr = corrs[best_lag]

        print(f"  {name:<20} {corrs.get(1,0):>+.4f} {corrs.get(2,0):>+.4f} "
              f"{corrs.get(3,0):>+.4f} {corrs.get(5,0):>+.4f} "
              f"t+{best_lag} ({best_corr:>+.4f})")

        best_leaders.append({
            'sym': sym, 'name': name,
            'corrs': corrs, 'best_lag': best_lag, 'best_corr': abs(best_corr),
        })

    # Also test: indicator's OWN momentum (multi-day) predicting tech
    print(f"\n  {'Indicator 5d mom':<20} {'t+1':>7} {'t+2':>7} {'t+3':>7} {'t+5':>7}")
    print(f"  {'-' * 50}")

    for sym, name in LEADERS.items():
        if sym not in closes.columns:
            continue
        mom5 = closes[sym].pct_change(5)
        for lag in [1, 2, 3, 5]:
            ind = mom5.values[:-lag]
            tech = tech_ret.values[lag:]
            valid = ~np.isnan(ind) & ~np.isnan(tech)
            if valid.sum() > 100:
                c = np.corrcoef(ind[valid], tech[valid])[0, 1]
                if lag == 1:
                    print(f"  {name + ' 5d':<20}", end='')
                if not np.isnan(c):
                    print(f" {c:>+.4f}", end='')
                else:
                    print(f" {'nan':>7}", end='')
        print()

    # === 2. Regime change prediction ===
    print(f"\n{'=' * 70}")
    print("2. REGIME CHANGE PREDICTION")
    print("  What moves in the 5 days BEFORE a tech crash (>3% drop)?")
    print("=" * 70)

    # Find tech crash days (equal-weight drops > 3%)
    crash_days = []
    for t in range(5, T):
        if tech_ret.iloc[t] < -0.03:
            crash_days.append(t)

    print(f"\n  {len(crash_days)} tech crash days (>3% EW drop)")

    # What were leading indicators doing 1-5 days before?
    print(f"\n  {'Indicator':<20} {'1d before':>10} {'3d before':>10} {'5d before':>10} {'Signal'}")
    print(f"  {'-' * 65}")

    for sym, name in LEADERS.items():
        if sym not in returns.columns:
            continue

        pre_1d = []
        pre_3d = []
        pre_5d = []
        for cd in crash_days:
            if cd >= 5:
                r1 = returns[sym].iloc[cd - 1]
                r3 = closes[sym].pct_change(3).iloc[cd - 1]
                r5 = closes[sym].pct_change(5).iloc[cd - 1]
                if not np.isnan(r1): pre_1d.append(r1)
                if not np.isnan(r3): pre_3d.append(r3)
                if not np.isnan(r5): pre_5d.append(r5)

        avg_1d = np.mean(pre_1d) * 100 if pre_1d else 0
        avg_3d = np.mean(pre_3d) * 100 if pre_3d else 0
        avg_5d = np.mean(pre_5d) * 100 if pre_5d else 0

        signal = ''
        if abs(avg_5d) > 1.0:
            signal = 'STRONG WARNING' if avg_5d < -1.0 else 'STRONG'
        elif abs(avg_3d) > 0.5:
            signal = 'WARNING'

        print(f"  {name:<20} {avg_1d:>+9.3f}% {avg_3d:>+9.3f}% {avg_5d:>+9.3f}% {signal}")

    # === 3. Rally prediction ===
    print(f"\n{'=' * 70}")
    print("3. RALLY PREDICTION")
    print("  What moves in the 5 days BEFORE a tech rally (>3% gain)?")
    print("=" * 70)

    rally_days = []
    for t in range(5, T):
        if tech_ret.iloc[t] > 0.03:
            rally_days.append(t)

    print(f"\n  {len(rally_days)} tech rally days (>3% EW gain)")
    print(f"\n  {'Indicator':<20} {'1d before':>10} {'3d before':>10} {'5d before':>10} {'Signal'}")
    print(f"  {'-' * 65}")

    for sym, name in LEADERS.items():
        if sym not in returns.columns:
            continue

        pre_1d = []
        pre_3d = []
        pre_5d = []
        for rd in rally_days:
            if rd >= 5:
                r1 = returns[sym].iloc[rd - 1]
                r3 = closes[sym].pct_change(3).iloc[rd - 1]
                r5 = closes[sym].pct_change(5).iloc[rd - 1]
                if not np.isnan(r1): pre_1d.append(r1)
                if not np.isnan(r3): pre_3d.append(r3)
                if not np.isnan(r5): pre_5d.append(r5)

        avg_1d = np.mean(pre_1d) * 100 if pre_1d else 0
        avg_3d = np.mean(pre_3d) * 100 if pre_3d else 0
        avg_5d = np.mean(pre_5d) * 100 if pre_5d else 0

        signal = ''
        if avg_5d < -2.0:
            signal = 'V-RECOVERY (crash then bounce)'
        elif avg_3d > 0.5:
            signal = 'MOMENTUM CONTINUATION'

        print(f"  {name:<20} {avg_1d:>+9.3f}% {avg_3d:>+9.3f}% {avg_5d:>+9.3f}% {signal}")

    # === 4. Bitcoin as overnight signal ===
    print(f"\n{'=' * 70}")
    print("4. BITCOIN AS OVERNIGHT LEADING INDICATOR")
    print("  BTC trades 24/7. Does weekend/overnight BTC predict Monday tech?")
    print("=" * 70)

    btc = returns.get('BTC-USD')
    if btc is not None:
        # Friday BTC -> Monday tech
        for dow_name, dow_filter in [('All days', lambda x: True),
                                      ('Monday tech', lambda x: x.weekday() == 0)]:
            btc_shifted = btc.shift(1)  # yesterday's BTC
            aligned = pd.DataFrame({
                'btc_yesterday': btc_shifted,
                'tech_today': tech_ret,
            }).dropna()

            if dow_name != 'All days':
                aligned = aligned[aligned.index.map(dow_filter)]

            if len(aligned) > 30:
                c = aligned['btc_yesterday'].corr(aligned['tech_today'])
                print(f"  {dow_name:<20} BTC(t-1) -> Tech(t): corr={c:>+.4f}  (n={len(aligned)})")

    # === 5. Summary: ranked leading indicators ===
    print(f"\n{'=' * 70}")
    print("5. RANKED LEADING INDICATORS")
    print("=" * 70)

    best_leaders.sort(key=lambda x: x['best_corr'], reverse=True)
    print(f"\n  {'Rank':>4} {'Indicator':<20} {'Best Lag':>9} {'|Corr|':>7} {'Direction'}")
    print(f"  {'-' * 50}")
    for rank, bl in enumerate(best_leaders):
        corr_at_best = bl['corrs'][bl['best_lag']]
        direction = 'SAME' if corr_at_best > 0 else 'INVERSE'
        print(f"  {rank+1:>4} {bl['name']:<20} t+{bl['best_lag']:<7} {bl['best_corr']:>.4f}  {direction}")

    # Save
    save_path = RESULTS_DIR / 'leading_indicators.json'
    with open(save_path, 'w') as f:
        json.dump({
            'leaders': [{k: v for k, v in bl.items()} for bl in best_leaders],
        }, f, indent=2)
    print(f"\n  Saved: {save_path}")


if __name__ == '__main__':
    main()
