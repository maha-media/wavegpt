"""
Tech 7 Money Flow Analysis.

The Mag 7 is a closed loop. Money flows between them.
This analysis maps:
  1. Lead-lag: does stock A today predict stock B tomorrow?
  2. Flow matrix: when A drops, where does the money go?
  3. Causality chains: which stocks lead the group?
  4. Correlation regime: when do they act as one vs diverge?
  5. Pair spread signals: which pairs mean-revert, which trend?

Usage:
    python finance/tech7_flows.py
"""

import json
import sys
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent / 'data'
RESULTS_DIR = Path(__file__).parent / 'training_results'

TECH7 = ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 'META', 'TSLA']


def main():
    closes = pd.read_parquet(DATA_DIR / 'tech7_closes.parquet')
    returns = closes.pct_change().dropna()

    T = len(returns)
    print("=" * 70)
    print("TECH 7 MONEY FLOW ANALYSIS")
    print(f"  {T} trading days, {closes.index[1].date()} to {closes.index[-1].date()}")
    print("=" * 70)

    # ===== 1. Lead-Lag Matrix =====
    # Does stock A's return TODAY predict stock B's return TOMORROW?
    print("\n" + "-" * 70)
    print("1. LEAD-LAG MATRIX (row leads column by 1 day)")
    print("   Read: 'Row stock today -> Column stock tomorrow'")
    print("-" * 70)

    lead_lag = np.zeros((7, 7))
    for i, s1 in enumerate(TECH7):
        for j, s2 in enumerate(TECH7):
            # s1 today vs s2 tomorrow
            r1 = returns[s1].values[:-1]
            r2 = returns[s2].values[1:]
            valid = ~np.isnan(r1) & ~np.isnan(r2)
            if valid.sum() > 50:
                lead_lag[i, j] = np.corrcoef(r1[valid], r2[valid])[0, 1]

    print(f"\n  {'':>6}", end='')
    for sym in TECH7:
        print(f" {sym:>7}", end='')
    print()
    for i, s1 in enumerate(TECH7):
        print(f"  {s1:>5}", end='')
        for j in range(7):
            val = lead_lag[i, j]
            marker = ' *' if abs(val) > 0.05 and i != j else '  '
            print(f" {val:>+.3f}{marker[1]}", end='')
        print()

    # Find strongest lead-lag relationships
    print(f"\n  Strongest lead-lag pairs (|corr| > 0.03):")
    pairs = []
    for i, s1 in enumerate(TECH7):
        for j, s2 in enumerate(TECH7):
            if i != j and abs(lead_lag[i, j]) > 0.03:
                pairs.append((s1, s2, lead_lag[i, j]))
    pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    for s1, s2, c in pairs[:15]:
        direction = "UP->UP" if c > 0 else "UP->DOWN"
        print(f"    {s1:>5} today -> {s2:>5} tomorrow: {c:>+.4f}  ({direction})")

    # ===== 2. Multi-day Lead-Lag =====
    print("\n" + "-" * 70)
    print("2. MULTI-DAY LEAD-LAG (does day t predict day t+1, t+2, t+3?)")
    print("-" * 70)

    for lag in [1, 2, 3, 5]:
        print(f"\n  Lag = {lag} days:")
        lag_pairs = []
        for i, s1 in enumerate(TECH7):
            for j, s2 in enumerate(TECH7):
                if i == j:
                    continue
                r1 = returns[s1].values[:-lag]
                r2 = returns[s2].values[lag:]
                valid = ~np.isnan(r1) & ~np.isnan(r2)
                if valid.sum() > 50:
                    c = np.corrcoef(r1[valid], r2[valid])[0, 1]
                    if abs(c) > 0.03:
                        lag_pairs.append((s1, s2, c))
        lag_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        for s1, s2, c in lag_pairs[:5]:
            print(f"    {s1} t -> {s2} t+{lag}: {c:>+.4f}")

    # ===== 3. Flow Matrix: When A drops, where does money go? =====
    print("\n" + "-" * 70)
    print("3. FLOW MATRIX — When stock A drops >1%, what happens to others NEXT DAY?")
    print("-" * 70)

    for sym_drop in TECH7:
        drop_days = returns[sym_drop] < -0.01  # dropped >1%
        n_drop = drop_days.sum()
        if n_drop < 20:
            continue

        # Get next-day returns of all other stocks after this stock drops
        next_day_rets = {}
        for sym_other in TECH7:
            next_rets = returns[sym_other].shift(-1)
            avg_next = next_rets[drop_days].mean() * 100
            next_day_rets[sym_other] = avg_next

        # Sort: where does money flow?
        sorted_others = sorted(next_day_rets.items(), key=lambda x: x[1], reverse=True)

        print(f"\n  When {sym_drop} drops >1% ({n_drop} days):")
        for sym_other, avg_ret in sorted_others:
            marker = " <-- SELF" if sym_other == sym_drop else ""
            bar = '+' * max(0, int(avg_ret * 50)) if avg_ret > 0 else '-' * max(0, int(-avg_ret * 50))
            print(f"    {sym_other}: {avg_ret:>+.3f}% next day  {bar}{marker}")

    # ===== 4. Synchronized vs Divergent Moves =====
    print("\n" + "-" * 70)
    print("4. CORRELATION REGIME — Rolling 20d correlation between ALL pairs")
    print("-" * 70)

    # Average pairwise correlation over time
    from collections import deque
    window = 20
    avg_corr_series = []
    dates = []

    for t in range(window, T):
        R = returns.iloc[t - window:t][TECH7].values
        if np.any(np.isnan(R)):
            continue
        corr = np.corrcoef(R.T)
        # Average off-diagonal correlation
        mask = ~np.eye(7, dtype=bool)
        avg_corr = corr[mask].mean()
        avg_corr_series.append(avg_corr)
        dates.append(returns.index[t])

    avg_corr = np.array(avg_corr_series)
    print(f"\n  Average pairwise correlation stats:")
    print(f"    Mean:   {avg_corr.mean():.3f}")
    print(f"    Std:    {avg_corr.std():.3f}")
    print(f"    Min:    {avg_corr.min():.3f} ({dates[avg_corr.argmin()].date()})")
    print(f"    Max:    {avg_corr.max():.3f} ({dates[avg_corr.argmax()].date()})")

    # When correlation is HIGH (all moving together) vs LOW (diverging)
    p25 = np.percentile(avg_corr, 25)
    p75 = np.percentile(avg_corr, 75)

    # Next-day returns of equal-weight portfolio in each regime
    eq_ret = returns[TECH7].mean(axis=1)
    next_eq = eq_ret.shift(-1)

    high_corr_rets = []
    low_corr_rets = []
    mid_corr_rets = []

    for i, (d, c) in enumerate(zip(dates, avg_corr)):
        if d in next_eq.index:
            r = next_eq.loc[d]
            if not np.isnan(r):
                if c > p75:
                    high_corr_rets.append(r)
                elif c < p25:
                    low_corr_rets.append(r)
                else:
                    mid_corr_rets.append(r)

    print(f"\n  Next-day equal-weight return by correlation regime:")
    print(f"    High corr (>{p75:.3f}): avg={np.mean(high_corr_rets)*100:>+.4f}%  "
          f"WR={np.mean(np.array(high_corr_rets)>0)*100:.1f}%  n={len(high_corr_rets)}")
    print(f"    Mid corr:               avg={np.mean(mid_corr_rets)*100:>+.4f}%  "
          f"WR={np.mean(np.array(mid_corr_rets)>0)*100:.1f}%  n={len(mid_corr_rets)}")
    print(f"    Low corr  (<{p25:.3f}): avg={np.mean(low_corr_rets)*100:>+.4f}%  "
          f"WR={np.mean(np.array(low_corr_rets)>0)*100:.1f}%  n={len(low_corr_rets)}")

    # ===== 5. Pair Spread Analysis =====
    print("\n" + "-" * 70)
    print("5. PAIR SPREADS — Which pairs mean-revert vs trend?")
    print("-" * 70)

    pair_results = []
    for i, s1 in enumerate(TECH7):
        for j, s2 in enumerate(TECH7):
            if i >= j:
                continue
            # Spread = log ratio
            spread = np.log(closes[s1] / closes[s2])
            spread_z = (spread - spread.rolling(50).mean()) / (spread.rolling(50).std() + 1e-8)
            spread_z = spread_z.dropna()

            # Does extreme spread predict mean reversion?
            # When spread is high (s1 overvalued vs s2), does s2 outperform next day?
            next_diff = (returns[s2] - returns[s1]).shift(-1)  # s2 - s1 next day
            aligned = pd.DataFrame({'z': spread_z, 'next_diff': next_diff}).dropna()

            if len(aligned) < 100:
                continue

            corr = aligned['z'].corr(aligned['next_diff'])
            # Positive corr = mean reversion (high spread -> s2 catches up)

            # Quintile analysis
            q1 = aligned[aligned['z'] <= aligned['z'].quantile(0.2)]['next_diff'].mean() * 100
            q5 = aligned[aligned['z'] >= aligned['z'].quantile(0.8)]['next_diff'].mean() * 100

            pair_results.append({
                'pair': f'{s1}/{s2}',
                'corr': float(corr),
                'q1_diff': float(q1),  # when s1 cheap vs s2
                'q5_diff': float(q5),  # when s1 expensive vs s2
                'spread_mean_rev': float(q5 - q1),  # positive = mean reverts
            })

    pair_results.sort(key=lambda x: abs(x['spread_mean_rev']), reverse=True)

    print(f"\n  {'Pair':<12} {'Corr':>7} {'Q1(cheap)':>10} {'Q5(rich)':>10} {'MR Spread':>10} {'Type':<15}")
    print(f"  {'-' * 65}")
    for p in pair_results:
        ptype = "MEAN-REVERT" if p['spread_mean_rev'] > 0.01 else \
                "TREND" if p['spread_mean_rev'] < -0.01 else "NEUTRAL"
        print(f"  {p['pair']:<12} {p['corr']:>+.4f} {p['q1_diff']:>+9.4f}% "
              f"{p['q5_diff']:>+9.4f}% {p['spread_mean_rev']:>+9.4f}% {ptype}")

    # ===== 6. Leadership Score: Who leads the group? =====
    print("\n" + "-" * 70)
    print("6. LEADERSHIP SCORE — Who leads, who follows?")
    print("-" * 70)

    for sym in TECH7:
        # How well does this stock's return today predict the GROUP's return tomorrow?
        group_others = [s for s in TECH7 if s != sym]
        group_ret = returns[group_others].mean(axis=1)
        next_group = group_ret.shift(-1)

        aligned = pd.DataFrame({'self': returns[sym], 'next_group': next_group}).dropna()
        leader_corr = aligned['self'].corr(aligned['next_group'])

        # How well does the GROUP today predict THIS stock tomorrow?
        next_self = returns[sym].shift(-1)
        aligned2 = pd.DataFrame({'group': group_ret, 'next_self': next_self}).dropna()
        follower_corr = aligned2['group'].corr(aligned2['next_self'])

        role = "LEADER" if leader_corr > follower_corr else "FOLLOWER"
        print(f"  {sym:<6} leads group: {leader_corr:>+.4f}  "
              f"follows group: {follower_corr:>+.4f}  -> {role}")

    # Save
    save_path = RESULTS_DIR / 'tech7_flows.json'
    with open(save_path, 'w') as f:
        json.dump({
            'lead_lag': {f'{TECH7[i]}->{TECH7[j]}': float(lead_lag[i, j])
                         for i in range(7) for j in range(7) if i != j},
            'pairs': pair_results,
        }, f, indent=2)
    print(f"\n  Saved: {save_path}")


if __name__ == '__main__':
    main()
