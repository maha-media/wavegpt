"""
Alpha Trading Playbook — the complete decision system.

Takes the sweep results and turns them into:
  1. Exact alpha thresholds (absolute values, not just percentiles)
  2. What to do at each level with exact position sizes
  3. Historical win rates and expected returns per action
  4. Day-by-day trade log showing every decision
  5. Current state: where is alpha TODAY and what should you do

Usage:
    python finance/alpha_playbook.py
"""

import json
import sys
from math import sqrt
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch

DATA_DIR = Path(__file__).parent / 'data'
RESULTS_DIR = Path(__file__).parent / 'training_results'


def main():
    # Load data
    data = torch.load(DATA_DIR / 'features_daily.pt', weights_only=False)
    features = data['features'].numpy()
    spy = data['spy'].numpy()
    timestamps = data['timestamps']
    feature_names = list(data['feature_names'])

    alpha_idx = feature_names.index('alpha')
    da_idx = feature_names.index('delta_alpha')

    alpha = features[:, alpha_idx]
    delta_alpha = features[:, da_idx]

    returns = np.zeros(len(spy))
    returns[1:] = (spy[1:] - spy[:-1]) / (spy[:-1] + 1e-8)

    # Next-day returns (what matters for trading)
    next_returns = np.zeros(len(spy))
    next_returns[:-1] = returns[1:]

    T = len(alpha)

    # Walk-forward split
    val_idx, test_idx = T, T
    for i, ts in enumerate(timestamps):
        if ts[:4] == '2025' and val_idx == T:
            val_idx = i
        if ts[:7] == '2025-07' and test_idx == T:
            test_idx = i

    # Best config from 40,500-combo sweep
    pct_lo = 40
    pct_hi = 60

    # Calibrate thresholds from TRAINING data only
    alpha_train = alpha[:val_idx]
    da_train = delta_alpha[:val_idx]

    alpha_p40 = np.percentile(alpha_train, pct_lo)
    alpha_p50 = np.percentile(alpha_train, 50)
    alpha_p60 = np.percentile(alpha_train, pct_hi)
    alpha_p80 = np.percentile(alpha_train, 80)

    da_p25 = np.percentile(da_train, 25)
    da_p75 = np.percentile(da_train, 75)

    # Absolute regime calibration
    regime_pcts = [0, 10, 25, 50, 75, 90, 100]
    regime_names = ['DEEP_CALM', 'CALM', 'NORMAL', 'ELEVATED', 'STRESS', 'CRISIS']
    regime_bounds = {}
    for i, rname in enumerate(regime_names):
        lo = float(np.percentile(alpha_train, regime_pcts[i]))
        hi = float(np.percentile(alpha_train, regime_pcts[i + 1])) if i < 5 else float('inf')
        regime_bounds[rname] = (lo, hi)

    # Risk scaler from training regime stats
    regime_ir = {}
    next_ret_train = next_returns[:val_idx]
    for rname, (lo, hi) in regime_bounds.items():
        mask = (alpha_train >= lo) & (alpha_train < hi)
        if mask.sum() > 10:
            r = next_ret_train[mask]
            regime_ir[rname] = r.mean() / (r.std() + 1e-8)

    print("=" * 70)
    print("ALPHA TRADING PLAYBOOK")
    print("=" * 70)

    # ===== SECTION 1: The Thresholds =====
    print("\n" + "=" * 70)
    print("1. ALPHA THRESHOLDS (calibrated from 2021-2024 training data)")
    print("=" * 70)
    print(f"""
  Alpha is the spectral exponent from SVD of the 17-ETF correlation matrix.
  Computed on a rolling 30-day window of daily returns.

  Training range: {alpha_train.min():.3f} to {alpha_train.max():.3f}
  Mean: {alpha_train.mean():.3f}  Std: {alpha_train.std():.3f}

  KEY THRESHOLDS:
    p40 = {alpha_p40:.4f}   (below this: contrarian long)
    p50 = {alpha_p50:.4f}   (median)
    p60 = {alpha_p60:.4f}   (above this + stable: flat / crowded)
    p80 = {alpha_p80:.4f}   (above this + rising: overextended, short)

  DELTA ALPHA THRESHOLDS (rate of change):
    p25 = {da_p25:.4f}   (falling fast: triggers contrarian long)
    p75 = {da_p75:.4f}   (rising fast: confirms overextended)
""")

    # ===== SECTION 2: The Rules =====
    print("=" * 70)
    print("2. EXACT TRADING RULES")
    print("=" * 70)

    # Classify every day
    actions = []
    for t in range(T):
        a = alpha[t]
        da = delta_alpha[t]

        # Determine absolute regime
        abs_regime = 'NORMAL'
        for rname, (lo, hi) in regime_bounds.items():
            if lo <= a < hi:
                abs_regime = rname
                break

        # Risk scaler
        ir = regime_ir.get(abs_regime, 0.0)
        risk_scaler = float(np.clip(0.6 + ir * 4.0, 0.2, 1.0))

        # Trading signal
        if a > alpha_p80 and da > da_p75:
            action = 'SHORT'
            base_size = 0.6
            signal = 'overextended'
        elif a > alpha_p50 and da >= 0:
            action = 'FLAT'
            base_size = 0.0
            signal = 'crowded'
        elif a <= alpha_p40 or da < da_p25:
            action = 'LONG'
            base_size = 0.3
            signal = 'contrarian'
        else:
            action = 'MILD_LONG'
            base_size = 0.1
            signal = 'mild_long'

        position = base_size * risk_scaler if action != 'FLAT' else 0.0
        if action == 'SHORT':
            position = -position

        actions.append({
            't': t,
            'date': timestamps[t][:10],
            'alpha': a,
            'delta_alpha': da,
            'abs_regime': abs_regime,
            'signal': signal,
            'action': action,
            'base_size': base_size,
            'risk_scaler': risk_scaler,
            'position': position,
            'next_return': next_returns[t],
            'spy_price': spy[t],
        })

    actions_df = pd.DataFrame(actions)

    # Print rules with actual numbers
    print(f"""
  RULE 1 — CONTRARIAN LONG
    When: alpha <= {alpha_p40:.4f}  OR  delta_alpha < {da_p25:.4f}
    Do:   Go LONG at 30% base * risk_scaler
    Why:  Market structure weakening = mean reversion opportunity

  RULE 2 — MILD LONG
    When: alpha in ({alpha_p40:.4f}, {alpha_p50:.4f}] and delta_alpha >= {da_p25:.4f}
    Do:   Go LONG at 10% base * risk_scaler
    Why:  Below median but not falling — small position

  RULE 3 — FLAT / CROWDED
    When: alpha > {alpha_p50:.4f} and delta_alpha >= 0
    Do:   FLAT (no position)
    Why:  Structure is high and stable — crowded trade, don't chase

  RULE 4 — SHORT / OVEREXTENDED
    When: alpha > {alpha_p80:.4f}  AND  delta_alpha > {da_p75:.4f}
    Do:   Go SHORT at 60% base * risk_scaler
    Why:  Extreme correlation + accelerating — about to snap back
""")

    # ===== SECTION 3: Historical Performance Per Action =====
    print("=" * 70)
    print("3. HISTORICAL PERFORMANCE BY ACTION")
    print("=" * 70)

    for split_name, start, end in [('TRAIN (2021-2024)', 0, val_idx),
                                     ('VAL (2025 H1)', val_idx, test_idx),
                                     ('TEST (2025 H2+)', test_idx, T - 1)]:
        df = actions_df.iloc[start:end]
        print(f"\n  [{split_name}] ({end - start} days)")
        print(f"  {'Action':<15} {'Days':>5} {'%':>5} {'AvgRet':>8} {'WinRate':>8} "
              f"{'AvgPos':>7} {'TotalPnL':>9}")
        print(f"  {'-' * 65}")

        for action in ['LONG', 'MILD_LONG', 'FLAT', 'SHORT']:
            mask = df['action'] == action
            n = mask.sum()
            if n == 0:
                continue
            sub = df[mask]
            avg_ret = sub['next_return'].mean() * 100
            wr = (sub['next_return'] > 0).mean() * 100

            # PnL from this action
            pnl = (sub['position'] * sub['next_return']).sum() * 100
            avg_pos = sub['position'].abs().mean()

            print(f"  {action:<15} {n:>5} {n/(end-start)*100:>4.0f}% "
                  f"{avg_ret:>+7.4f}% {wr:>7.1f}% {avg_pos:>6.3f} {pnl:>+8.3f}%")

        # Total
        total_pnl = (df['position'] * df['next_return']).sum() * 100
        bh = (spy[min(end, T-1)] - spy[start]) / spy[start] * 100
        print(f"  {'TOTAL':<15} {end-start:>5}       {'':>8} {'':>8} {'':>7} {total_pnl:>+8.3f}%")
        print(f"  {'B&H':<15} {'':>5}       {'':>8} {'':>8} {'':>7} {bh:>+8.3f}%")

    # ===== SECTION 4: Risk Scaler Breakdown =====
    print("\n" + "=" * 70)
    print("4. RISK SCALER BY ABSOLUTE REGIME")
    print("=" * 70)

    for rname in regime_names:
        lo, hi = regime_bounds[rname]
        ir = regime_ir.get(rname, 0)
        rs = float(np.clip(0.6 + ir * 4.0, 0.2, 1.0))
        mask = actions_df['abs_regime'] == rname
        n = mask.sum()
        if n > 0:
            avg_ret = actions_df.loc[mask, 'next_return'].mean() * 100
            std_ret = actions_df.loc[mask, 'next_return'].std() * 100
            print(f"  {rname:<12} alpha=[{lo:.3f},{hi:.3f})  {n:>4} days  "
                  f"ret={avg_ret:+.4f}%  vol={std_ret:.4f}%  IR={ir:+.3f}  "
                  f"risk_scaler={rs:.2f}")

    # ===== SECTION 5: Where Are We Today? =====
    print("\n" + "=" * 70)
    print("5. CURRENT STATE")
    print("=" * 70)

    latest = actions[-1]
    prev = actions[-2] if len(actions) > 1 else actions[-1]

    print(f"""
  Date:          {latest['date']}
  SPY:           ${latest['spy_price']:.2f}
  Alpha:         {latest['alpha']:.4f}  (p40={alpha_p40:.4f}  p50={alpha_p50:.4f}  p60={alpha_p60:.4f})
  Delta Alpha:   {latest['delta_alpha']:+.4f}  (p25={da_p25:.4f}  p75={da_p75:.4f})
  Abs Regime:    {latest['abs_regime']}
  Risk Scaler:   {latest['risk_scaler']:.2f}

  SIGNAL:        {latest['signal'].upper()}
  ACTION:        {latest['action']}
  POSITION:      {latest['position']:+.3f}  ({'%.0f%% %s' % (abs(latest['position'])*100, 'long' if latest['position'] > 0 else 'short' if latest['position'] < 0 else 'flat')})

  Yesterday:     {prev['action']} at {prev['position']:+.3f} ({prev['signal']})
  Change:        {'SAME' if latest['action'] == prev['action'] else 'CHANGED -> ' + latest['action']}
""")

    # ===== SECTION 6: Recent Trade Log =====
    print("=" * 70)
    print("6. LAST 30 DAYS — TRADE LOG")
    print("=" * 70)
    print(f"  {'Date':<12} {'Alpha':>7} {'dAlpha':>7} {'Regime':<10} {'Action':<12} "
          f"{'Pos':>6} {'SPY':>8} {'DayRet':>7}")
    print(f"  {'-' * 75}")

    for row in actions[-30:]:
        day_ret = row['next_return'] * 100
        pnl_contrib = row['position'] * row['next_return'] * 100
        print(f"  {row['date']:<12} {row['alpha']:>7.4f} {row['delta_alpha']:>+7.4f} "
              f"{row['abs_regime']:<10} {row['action']:<12} {row['position']:>+5.2f} "
              f"${row['spy_price']:>7.2f} {day_ret:>+6.3f}%")

    # ===== SECTION 7: Summary Stats =====
    print("\n" + "=" * 70)
    print("7. STRATEGY SUMMARY")
    print("=" * 70)

    # Full backtest with costs
    positions = actions_df['position'].values
    cost_frac = 3.0 / 10000
    bar_pnl = positions[:-1] * next_returns[:-1]
    costs = np.abs(np.diff(positions)) * cost_frac
    bar_pnl_net = bar_pnl - costs

    for split_name, start, end in [('Train', 0, val_idx),
                                     ('Val', val_idx, test_idx),
                                     ('Test', test_idx, T)]:
        p = bar_pnl_net[start:end-1]
        cum = np.cumsum(p)
        total = cum[-1] * 100 if len(cum) > 0 else 0
        sharpe = p.mean() / (p.std() + 1e-8) * sqrt(252)
        rm = np.maximum.accumulate(cum)
        dd = (rm - cum).max() * 100
        bh = (spy[end-1] - spy[start]) / spy[start] * 100

        n_trades = np.sum(np.abs(np.diff(positions[start:end])) > 0.01)

        print(f"  {split_name:>5}: PnL={total:>+7.2f}%  Sharpe={sharpe:>+5.2f}  "
              f"MaxDD={dd:>5.2f}%  Trades={n_trades:>3}  B&H={bh:>+7.2f}%")

    # Save playbook data
    playbook = {
        'thresholds': {
            'alpha_p40': float(alpha_p40),
            'alpha_p50': float(alpha_p50),
            'alpha_p60': float(alpha_p60),
            'alpha_p80': float(alpha_p80),
            'da_p25': float(da_p25),
            'da_p75': float(da_p75),
        },
        'rules': [
            {'name': 'CONTRARIAN_LONG', 'condition': f'alpha <= {alpha_p40:.4f} OR da < {da_p25:.4f}',
             'position': '+30% * risk_scaler'},
            {'name': 'MILD_LONG', 'condition': f'alpha in ({alpha_p40:.4f}, {alpha_p50:.4f}]',
             'position': '+10% * risk_scaler'},
            {'name': 'FLAT', 'condition': f'alpha > {alpha_p50:.4f} and da >= 0',
             'position': '0%'},
            {'name': 'SHORT', 'condition': f'alpha > {alpha_p80:.4f} AND da > {da_p75:.4f}',
             'position': '-60% * risk_scaler'},
        ],
        'regime_bounds': {k: list(v) for k, v in regime_bounds.items()},
        'regime_risk_scalers': {k: float(np.clip(0.6 + v * 4.0, 0.2, 1.0))
                                 for k, v in regime_ir.items()},
        'current_state': {
            'date': latest['date'],
            'alpha': float(latest['alpha']),
            'delta_alpha': float(latest['delta_alpha']),
            'action': latest['action'],
            'position': float(latest['position']),
            'abs_regime': latest['abs_regime'],
        },
    }

    save_path = RESULTS_DIR / 'playbook.json'
    with open(save_path, 'w') as f:
        json.dump(playbook, f, indent=2)
    print(f"\n  Saved: {save_path}")


if __name__ == '__main__':
    main()
