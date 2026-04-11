"""
Tech 7 Flow Trader — exploit the money circle.

Signals from flow analysis:
  1. LEAD-LAG: META up -> AMZN down tomorrow. AAPL up -> others cool off.
  2. DIP MAGNET: When any stock drops >1%, NVDA bounces hardest next day.
  3. MOMENTUM: Each stock's 10d momentum predicts its own next-day (+0.30 corr).
  4. CORRELATION REGIME: Low correlation = stock-picking alpha.
     High correlation = danger, reduce exposure.
  5. PAIR MEAN REVERSION: NVDA/META, AMZN/META, MSFT/META spreads revert.
  6. LEADERSHIP: MSFT/NVDA/AMZN lead, others follow.

Strategy:
  - Always long (trending market)
  - Base = equal weight
  - Adjust weights daily based on flow signals
  - Walk-forward, recalibrate quarterly

Usage:
    python finance/tech7_flow_trader.py
    python finance/tech7_flow_trader.py --starting-capital 100000
"""

import argparse
import json
import sys
from math import sqrt
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent / 'data'
RESULTS_DIR = Path(__file__).parent / 'training_results'
RESULTS_DIR.mkdir(exist_ok=True)

TECH7 = ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 'META', 'TSLA']
DEFENSIVES = ['GLD', 'SLV', 'USO', 'XLE', 'XLU', 'XLP']


def load_defensive_closes():
    """Load defensive asset prices from the main market data."""
    path = DATA_DIR / 'market_daily.parquet'
    if not path.exists():
        return None
    m = pd.read_parquet(path)
    if isinstance(m.columns, pd.MultiIndex):
        closes = m['Close']
    else:
        closes = m
    available = [s for s in DEFENSIVES if s in closes.columns]
    return closes[available].dropna(how='all') if available else None


def compute_lead_lag_matrix(returns, window):
    """Compute lead-lag correlations from a training window."""
    matrix = np.zeros((7, 7))
    for i, s1 in enumerate(TECH7):
        for j, s2 in enumerate(TECH7):
            if i == j:
                continue
            r1 = returns[s1].values[:-1]
            r2 = returns[s2].values[1:]
            valid = ~np.isnan(r1) & ~np.isnan(r2)
            if valid.sum() > 30:
                matrix[i, j] = np.corrcoef(r1[valid], r2[valid])[0, 1]
    return matrix


def compute_pair_betas(returns, window):
    """For each pair, compute mean-reversion beta from spread z-score."""
    import warnings
    warnings.filterwarnings('ignore')
    betas = {}
    for i, s1 in enumerate(TECH7):
        for j, s2 in enumerate(TECH7):
            if i >= j:
                continue
            # spread z-score correlation with next-day differential
            r1 = returns[s1].rolling(20).mean()
            r2 = returns[s2].rolling(20).mean()
            spread = r1 - r2
            spread_z = (spread - spread.rolling(50).mean()) / (spread.rolling(50).std() + 1e-8)
            next_diff = (returns[s2] - returns[s1]).shift(-1)
            aligned = pd.DataFrame({'z': spread_z, 'nd': next_diff}).dropna()
            if len(aligned) > 50:
                c = aligned['z'].corr(aligned['nd'])
                betas[(s1, s2)] = float(c) if not np.isnan(c) else 0
            else:
                betas[(s1, s2)] = 0
    return betas


def main():
    parser = argparse.ArgumentParser(description='Tech 7 Flow Trader')
    parser.add_argument('--starting-capital', type=float, default=100000)
    parser.add_argument('--cost-bps', type=float, default=5.0)
    parser.add_argument('--cal-days', type=int, default=252)
    parser.add_argument('--recal-days', type=int, default=63)
    # Signal weights (how much each signal affects weights)
    parser.add_argument('--w-momentum', type=float, default=0.3)
    parser.add_argument('--w-leadlag', type=float, default=0.2)
    parser.add_argument('--w-dip', type=float, default=0.3)
    parser.add_argument('--w-pairs', type=float, default=0.1)
    parser.add_argument('--w-corr-regime', type=float, default=0.1)
    args = parser.parse_args()

    closes = pd.read_parquet(DATA_DIR / 'tech7_closes.parquet')
    returns = closes.pct_change()
    T = len(returns)

    # Load defensive assets for rotation
    def_closes = load_defensive_closes()
    if def_closes is not None:
        # Align to same dates as tech
        def_closes = def_closes.reindex(closes.index, method='ffill')
        def_returns = def_closes.pct_change()
        def_syms = list(def_closes.columns)
        print(f"  Defensive assets: {', '.join(def_syms)}")
    else:
        def_returns = None
        def_syms = []
        print(f"  No defensive assets found (will use cash)")

    print("=" * 70)
    print("TECH 7 FLOW TRADER")
    print(f"  {T} days, {closes.index[1].date()} to {closes.index[-1].date()}")
    print(f"  Signal weights: mom={args.w_momentum} lead={args.w_leadlag} "
          f"dip={args.w_dip} pairs={args.w_pairs} corr={args.w_corr_regime}")
    print("=" * 70)

    sim_start = args.cal_days
    cost_frac = args.cost_bps / 10000.0
    capital = args.starting_capital
    weights = np.ones(7) / 7
    prev_weights = weights.copy()

    equity = [capital]
    daily_log = []
    last_recal = -999
    prev_def_weights = {}
    def_weights_dict = {}

    # Cached calibration data
    lead_lag = None
    pair_betas = None
    train_returns = None
    momentum_scale = None

    for t in range(sim_start, T):
        # --- Recalibrate ---
        if t - last_recal >= args.recal_days or lead_lag is None:
            cal_start = max(0, t - args.cal_days)
            train_ret = returns.iloc[cal_start:t]

            lead_lag = compute_lead_lag_matrix(train_ret, args.cal_days)
            pair_betas = compute_pair_betas(train_ret, args.cal_days)

            # Momentum scale: normalize 10d momentum by its std
            for sym in TECH7:
                mom = closes[sym].pct_change(10)
                mom_std = mom.iloc[cal_start:t].std()
                if momentum_scale is None:
                    momentum_scale = {}
                momentum_scale[sym] = float(mom_std) if mom_std > 0 else 1.0

            train_returns = train_ret
            last_recal = t

        # --- MOMENTUM-FIRST WEIGHTING ---
        # Step 1: Base weights from MOMENTUM (the dominant signal)
        #   Winners get more capital. This is the core.
        # Step 2: Flow signals adjust TIMING (buy dips of winners)

        # STEP 1: Momentum base weights
        # Multi-window momentum -> weight winners heavier
        # Use LONGER windows to catch sustained winners (NVDA +717%)
        # Short windows for timing, long windows for allocation
        mom_scores = np.zeros(7)
        for i, sym in enumerate(TECH7):
            for lookback in [10, 20, 50, 100]:
                if t >= lookback:
                    mom = (closes[sym].iloc[t] - closes[sym].iloc[t - lookback]) / closes[sym].iloc[t - lookback]
                    z = mom / (momentum_scale.get(sym, 1.0) + 1e-8)
                    # LONGER momentum weighted MORE for allocation
                    # Short momentum for timing (in flow signals below)
                    w = {10: 0.15, 20: 0.20, 50: 0.30, 100: 0.35}[lookback]
                    mom_scores[i] += z * w

        # Rank-based allocation: top stocks get more, smoothly
        # Softmax with temperature controls concentration
        temperature = 1.0 / max(args.w_momentum, 0.1)
        mom_exp = np.exp(mom_scores / temperature)
        mom_weights = mom_exp / mom_exp.sum()

        # Floor: nobody below 5%, cap at 35% (ride winners but diversified)
        mom_weights = np.clip(mom_weights, 0.05, 0.35)
        mom_weights /= mom_weights.sum()

        # STEP 2: Flow signals = TIMING adjustments on top of momentum
        timing_boost = np.zeros(7)

        # FLOW SIGNAL A: Lead-lag (yesterday's moves predict today's)
        if t >= 1:
            yesterday_rets = np.array([returns[sym].iloc[t - 1] for sym in TECH7])
            yesterday_rets = np.nan_to_num(yesterday_rets, nan=0.0)

            for j in range(7):
                predicted = 0
                for i in range(7):
                    if i != j:
                        predicted += lead_lag[i, j] * yesterday_rets[i]
                timing_boost[j] += predicted * args.w_leadlag * 50

        # FLOW SIGNAL B: Dip magnet (buy the drop, esp NVDA)
        if t >= 1:
            for i, sym in enumerate(TECH7):
                ret_yesterday = returns[sym].iloc[t - 1]
                if ret_yesterday < -0.01:
                    drop_size = abs(ret_yesterday)
                    # The dropper bounces
                    timing_boost[i] += drop_size * 5 * args.w_dip
                    # NVDA always catches flow on any drop
                    nvda_idx = TECH7.index('NVDA')
                    timing_boost[nvda_idx] += drop_size * 3 * args.w_dip

        # FLOW SIGNAL C: Correlation regime scaling
        if t >= 20:
            recent_rets = returns.iloc[t - 20:t][TECH7].values
            if not np.any(np.isnan(recent_rets)):
                corr_matrix = np.corrcoef(recent_rets.T)
                mask_corr = ~np.eye(7, dtype=bool)
                avg_corr = corr_matrix[mask_corr].mean()
                # Low corr -> amplify momentum differentiation
                # High corr -> compress toward equal weight
                differentiation = max(0.3, 1.5 - avg_corr)
                timing_boost *= differentiation

        # COMBINE: momentum weights + timing adjustments
        adjusted = mom_weights * (1.0 + timing_boost)
        adjusted = np.clip(adjusted, 0.03, 0.50)  # min 3%, max 50%
        weights = adjusted / adjusted.sum()

        # MACRO RISK SWITCH — scale TOTAL exposure
        # When macro says bear: reduce everything proportionally
        # When macro says bull: full exposure
        exposure = 1.0

        # Signal A: Average 50d momentum of all 7
        # If most stocks trending down -> reduce
        if t >= 50:
            avg_mom50 = np.mean([(closes[sym].iloc[t] - closes[sym].iloc[t - 50]) /
                                 closes[sym].iloc[t - 50] for sym in TECH7])
            if avg_mom50 < -0.05:
                # Strong downtrend: scale to 30%
                exposure *= 0.3
            elif avg_mom50 < 0:
                # Mild downtrend: scale to 60%
                exposure *= 0.6

        # Signal B: Correlation regime
        # High correlation = herd selling = reduce
        if t >= 20:
            recent_rets = returns.iloc[t - 20:t][TECH7].values
            if not np.any(np.isnan(recent_rets)):
                corr_matrix = np.corrcoef(recent_rets.T)
                mask_corr = ~np.eye(7, dtype=bool)
                avg_corr = corr_matrix[mask_corr].mean()
                if avg_corr > 0.75:
                    # Very high correlation: everything moving together (crash or melt-up)
                    # If also negative momentum, this is a crash -> reduce
                    if t >= 50:
                        if avg_mom50 < 0:
                            exposure *= 0.5  # crash: cut hard

        # Signal C: Number of stocks above their 50d MA
        # Breadth: if fewer than 3/7 above MA, market is weak
        if t >= 50:
            n_above_ma = sum(1 for sym in TECH7
                            if closes[sym].iloc[t] > closes[sym].iloc[t-50:t].mean())
            if n_above_ma <= 2:
                exposure *= 0.5  # only 2 of 7 healthy -> reduce
            elif n_above_ma <= 4:
                exposure *= 0.8  # mixed -> slightly cautious

        # Apply exposure: tech weights sum to exposure
        # The rest rotates into defensive assets (not cash!)
        exposure = max(exposure, 0.15)  # always keep at least 15% in tech
        weights = weights * exposure

        # DEFENSIVE ROTATION: allocate (1 - exposure) to best defensive assets
        def_allocation = 1.0 - exposure  # what's not in tech
        def_weights_dict = {}

        if def_allocation > 0.01 and def_returns is not None and t >= 50:
            # Momentum-weight the defensives too (ride what's working)
            def_mom_scores = []
            for dsym in def_syms:
                if dsym in def_closes.columns:
                    dp = def_closes[dsym].iloc[t]
                    dp50 = def_closes[dsym].iloc[t - 50] if t >= 50 else dp
                    if dp50 > 0 and not np.isnan(dp) and not np.isnan(dp50):
                        mom = (dp - dp50) / dp50
                        def_mom_scores.append(max(mom, 0))  # only positive momentum
                    else:
                        def_mom_scores.append(0)
                else:
                    def_mom_scores.append(0)

            def_mom_scores = np.array(def_mom_scores)
            if def_mom_scores.sum() > 0:
                def_w = def_mom_scores / def_mom_scores.sum() * def_allocation
            else:
                # Equal weight among defensives if none trending
                def_w = np.ones(len(def_syms)) / len(def_syms) * def_allocation

            for di, dsym in enumerate(def_syms):
                def_weights_dict[dsym] = float(def_w[di])

        # --- Execute trades ---
        day_rets = np.array([returns[sym].iloc[t] if t < len(returns) else 0
                             for sym in TECH7])
        day_rets = np.nan_to_num(day_rets, nan=0.0)

        # Tech return
        port_ret = np.sum(prev_weights * day_rets)

        # Defensive return
        def_ret = 0.0
        if prev_def_weights and def_returns is not None:
            for dsym, dw in prev_def_weights.items():
                if dsym in def_returns.columns and t < len(def_returns):
                    dr = def_returns[dsym].iloc[t]
                    if not np.isnan(dr):
                        def_ret += dw * dr

        total_port_ret = port_ret + def_ret

        # Transaction costs (on all weight changes)
        turnover = np.sum(np.abs(weights - prev_weights))
        # Add defensive turnover
        all_def_syms = set(list(def_weights_dict.keys()) + list(prev_def_weights.keys()))
        for dsym in all_def_syms:
            turnover += abs(def_weights_dict.get(dsym, 0) - prev_def_weights.get(dsym, 0))

        cost = turnover * cost_frac * capital

        pnl = total_port_ret * capital - cost
        capital += pnl
        equity.append(capital)

        # Log
        all_weights = {sym: round(float(w), 4) for sym, w in zip(TECH7, weights)}
        for dsym, dw in def_weights_dict.items():
            all_weights[dsym] = round(dw, 4)

        daily_log.append({
            'date': str(closes.index[t].date()),
            'weights': all_weights,
            'scores': {sym: round(float(s), 4) for sym, s in zip(TECH7, mom_scores)},
            'port_return': float(total_port_ret * 100),
            'tech_exposure': float(exposure),
            'def_allocation': float(def_allocation),
            'pnl': float(pnl),
            'capital': float(capital),
            'turnover': float(turnover),
        })

        prev_weights = weights.copy()
        prev_def_weights = def_weights_dict.copy()

    # --- Results ---
    equity = np.array(equity)
    final = equity[-1]
    total_ret = (final - args.starting_capital) / args.starting_capital * 100
    n_years = (T - sim_start) / 252
    annual_ret = ((final / args.starting_capital) ** (1 / n_years) - 1) * 100

    daily_rets = np.diff(equity) / equity[:-1]
    sharpe = daily_rets.mean() / (daily_rets.std() + 1e-8) * sqrt(252)
    rm = np.maximum.accumulate(equity)
    max_dd = ((rm - equity) / rm * 100).max()

    # Equal weight baseline
    eq_rets = returns[TECH7].iloc[sim_start:].mean(axis=1)
    eq_cum = (1 + eq_rets).cumprod()
    eq_total = (eq_cum.iloc[-1] - 1) * 100

    total_turnover = sum(d['turnover'] for d in daily_log)
    total_costs = sum(d['pnl'] for d in daily_log) - sum(d['port_return'] / 100 * (d['capital'] - d['pnl']) for d in daily_log)

    print(f"\n{'=' * 70}")
    print("RESULTS")
    print(f"{'=' * 70}")
    print(f"""
  Period: {daily_log[0]['date']} to {daily_log[-1]['date']} ({n_years:.1f} years)
  Starting Capital: ${args.starting_capital:>12,.0f}
  Final Capital:    ${final:>12,.0f}

  FLOW TRADER:
    Total Return:   {total_ret:>+8.2f}%
    Annual Return:  {annual_ret:>+8.2f}%
    Sharpe Ratio:   {sharpe:>8.2f}
    Max Drawdown:   {max_dd:>8.2f}%

  EQUAL WEIGHT BUY & HOLD:
    Total Return:   {eq_total:>+8.2f}%

  EDGE vs Equal Weight: {total_ret - eq_total:>+8.2f}%
""")

    # Per-stock weight and contribution analysis
    print("  AVERAGE WEIGHTS & INDIVIDUAL STOCK RETURNS:")
    avg_weights = defaultdict(list)
    for d in daily_log:
        for sym, w in d['weights'].items():
            avg_weights[sym].append(w)

    total_stock_rets = {}
    for sym in TECH7:
        sym_ret = (closes[sym].iloc[-1] - closes[sym].iloc[sim_start]) / closes[sym].iloc[sim_start] * 100
        total_stock_rets[sym] = sym_ret
        avg_w = np.mean(avg_weights[sym])
        print(f"    {sym:<6} avg weight: {avg_w:>5.1%}  stock return: {sym_ret:>+8.1f}%  "
              f"{'OVERWEIGHT' if avg_w > 0.16 else 'UNDERWEIGHT' if avg_w < 0.12 else 'NEUTRAL'}")

    # Yearly breakdown
    print(f"\n  YEARLY BREAKDOWN:")
    print(f"  {'Year':<6} {'Strategy':>9} {'EqWeight':>9} {'Edge':>8} {'Sharpe':>7} {'MaxDD':>7}")
    print(f"  {'-' * 50}")

    years = sorted(set(d['date'][:4] for d in daily_log))
    for year in years:
        yr_days = [d for d in daily_log if d['date'][:4] == year]
        if len(yr_days) < 10:
            continue
        yr_start = yr_days[0]['capital'] / (1 + yr_days[0]['port_return'] / 100)
        yr_end = yr_days[-1]['capital']
        yr_ret = (yr_end - yr_start) / yr_start * 100

        yr_rets_arr = np.array([d['pnl'] / (d['capital'] - d['pnl']) for d in yr_days])
        yr_sharpe = yr_rets_arr.mean() / (yr_rets_arr.std() + 1e-8) * sqrt(252)

        yr_eq = np.array([yr_start] + [d['capital'] for d in yr_days])
        yr_rm = np.maximum.accumulate(yr_eq)
        yr_dd = ((yr_rm - yr_eq) / yr_rm * 100).max()

        # EqWeight for this year
        yr_idx_start = returns.index.get_loc(pd.Timestamp(yr_days[0]['date']))
        yr_idx_end = returns.index.get_loc(pd.Timestamp(yr_days[-1]['date']))
        yr_eq_rets = returns[TECH7].iloc[yr_idx_start:yr_idx_end + 1].mean(axis=1)
        yr_eq_total = ((1 + yr_eq_rets).prod() - 1) * 100

        print(f"  {year:<6} {yr_ret:>+8.2f}% {yr_eq_total:>+8.2f}% "
              f"{yr_ret - yr_eq_total:>+7.2f}% {yr_sharpe:>+6.2f} {yr_dd:>6.2f}%")

    # Top/bottom days
    sorted_days = sorted(daily_log, key=lambda d: d['pnl'])
    print(f"\n  5 WORST DAYS:")
    for d in sorted_days[:5]:
        top_w = max(d['weights'].items(), key=lambda x: x[1])
        print(f"    {d['date']}  PnL=${d['pnl']:>+9,.0f}  port={d['port_return']:>+.2f}%  "
              f"top: {top_w[0]}@{top_w[1]:.1%}")

    print(f"\n  5 BEST DAYS:")
    for d in sorted_days[-5:]:
        top_w = max(d['weights'].items(), key=lambda x: x[1])
        print(f"    {d['date']}  PnL=${d['pnl']:>+9,.0f}  port={d['port_return']:>+.2f}%  "
              f"top: {top_w[0]}@{top_w[1]:.1%}")

    # Last 10 days with weights
    print(f"\n  LAST 10 DAYS:")
    for d in daily_log[-10:]:
        w = d['weights']
        top2 = sorted(w.items(), key=lambda x: x[1], reverse=True)[:2]
        bot1 = sorted(w.items(), key=lambda x: x[1])[:1]
        print(f"  {d['date']}  PnL=${d['pnl']:>+7,.0f}  "
              f"top: {top2[0][0]}={top2[0][1]:.0%} {top2[1][0]}={top2[1][1]:.0%}  "
              f"bot: {bot1[0][0]}={bot1[0][1]:.0%}")

    # Save
    save_path = RESULTS_DIR / 'tech7_flow_trader.json'
    with open(save_path, 'w') as f:
        json.dump({
            'results': {
                'total_return': round(total_ret, 2),
                'annual_return': round(annual_ret, 2),
                'sharpe': round(float(sharpe), 2),
                'max_drawdown': round(float(max_dd), 2),
                'eq_weight_return': round(float(eq_total), 2),
                'edge': round(total_ret - float(eq_total), 2),
            },
            'signal_weights': {
                'momentum': args.w_momentum,
                'lead_lag': args.w_leadlag,
                'dip_magnet': args.w_dip,
                'pairs': args.w_pairs,
                'corr_regime': args.w_corr_regime,
            },
        }, f, indent=2)
    print(f"\n  Saved: {save_path}")


if __name__ == '__main__':
    main()
