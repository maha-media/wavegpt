"""
3-Year Walk-Forward Simulation.

Realistic simulation as if running live:
  - Start with 1 year of calibration data (2021)
  - Trade from 2022 through 2026 (~3.5 years)
  - Recalibrate alpha thresholds every quarter using trailing 1yr data
  - Track every trade, daily PnL, cumulative equity curve
  - Report full performance with drawdown analysis

No lookahead bias: thresholds only use past data at each point.

Usage:
    python finance/simulate_3yr.py
    python finance/simulate_3yr.py --starting-capital 100000
    python finance/simulate_3yr.py --recal-months 6
"""

import argparse
import json
import sys
from math import sqrt
from pathlib import Path

import numpy as np
import torch

DATA_DIR = Path(__file__).parent / 'data'
RESULTS_DIR = Path(__file__).parent / 'training_results'
RESULTS_DIR.mkdir(exist_ok=True)


def calibrate_thresholds(alpha_history, da_history):
    """Compute regime thresholds from a window of past alpha values."""
    return {
        'alpha_p40': float(np.percentile(alpha_history, 40)),
        'alpha_p50': float(np.percentile(alpha_history, 50)),
        'alpha_p80': float(np.percentile(alpha_history, 80)),
        'da_p25': float(np.percentile(da_history, 25)),
        'da_p75': float(np.percentile(da_history, 75)),
        # Absolute regime bounds for risk scaler
        'regime_bounds': {
            'DEEP_CALM':  (float(np.percentile(alpha_history, 0)),
                           float(np.percentile(alpha_history, 10))),
            'CALM':       (float(np.percentile(alpha_history, 10)),
                           float(np.percentile(alpha_history, 25))),
            'NORMAL':     (float(np.percentile(alpha_history, 25)),
                           float(np.percentile(alpha_history, 50))),
            'ELEVATED':   (float(np.percentile(alpha_history, 50)),
                           float(np.percentile(alpha_history, 75))),
            'STRESS':     (float(np.percentile(alpha_history, 75)),
                           float(np.percentile(alpha_history, 90))),
            'CRISIS':     (float(np.percentile(alpha_history, 90)),
                           float('inf')),
        },
    }


def compute_risk_scaler(alpha_val, thresholds, alpha_history, returns_history):
    """Compute risk scaler from absolute regime's information ratio."""
    regime_bounds = thresholds['regime_bounds']
    abs_regime = 'NORMAL'
    for rname, (lo, hi) in regime_bounds.items():
        if lo <= alpha_val < hi:
            abs_regime = rname
            break

    # Compute IR for this regime from history
    mask = np.zeros(len(alpha_history), dtype=bool)
    lo, hi = regime_bounds[abs_regime]
    mask = (alpha_history >= lo) & (alpha_history < hi)
    if mask.sum() > 10:
        r = returns_history[mask]
        ir = r.mean() / (r.std() + 1e-8)
        risk_scaler = float(np.clip(0.6 + ir * 4.0, 0.2, 1.0))
    else:
        risk_scaler = 0.5

    return risk_scaler, abs_regime


def get_action(alpha_val, da_val, thresholds):
    """Determine trading action from alpha and delta_alpha."""
    p40 = thresholds['alpha_p40']
    p50 = thresholds['alpha_p50']
    p80 = thresholds['alpha_p80']
    da_p25 = thresholds['da_p25']
    da_p75 = thresholds['da_p75']

    if alpha_val > p80 and da_val > da_p75:
        return 'SHORT', 0.6
    elif alpha_val > p50 and da_val >= 0:
        return 'FLAT', 0.0
    elif alpha_val <= p40 or da_val < da_p25:
        return 'LONG', 0.3
    else:
        return 'MILD_LONG', 0.1


def main():
    parser = argparse.ArgumentParser(description='3-Year Walk-Forward Simulation')
    parser.add_argument('--starting-capital', type=float, default=100000)
    parser.add_argument('--cost-bps', type=float, default=3.0)
    parser.add_argument('--calibration-days', type=int, default=252,
                        help='Days of history for threshold calibration')
    parser.add_argument('--recal-days', type=int, default=63,
                        help='Recalibrate every N trading days (~quarterly)')
    args = parser.parse_args()

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

    T = len(alpha)
    returns = np.zeros(T)
    returns[1:] = (spy[1:] - spy[:-1]) / (spy[:-1] + 1e-8)

    cost_frac = args.cost_bps / 10000.0
    cal_days = args.calibration_days
    recal_days = args.recal_days

    # Find simulation start: need cal_days of history first
    sim_start = cal_days
    sim_end = T

    print("=" * 70)
    print("3-YEAR WALK-FORWARD SIMULATION")
    print("=" * 70)
    print(f"  Data: {timestamps[0][:10]} to {timestamps[-1][:10]} ({T} trading days)")
    print(f"  Calibration window: {cal_days} days (~1 year)")
    print(f"  Recalibrate every: {recal_days} days (~quarterly)")
    print(f"  Simulation: {timestamps[sim_start][:10]} to {timestamps[-1][:10]} "
          f"({sim_end - sim_start} trading days)")
    print(f"  Starting capital: ${args.starting_capital:,.0f}")
    print(f"  Transaction cost: {args.cost_bps} bps")

    # --- Run simulation ---
    capital = args.starting_capital
    position = 0.0  # fraction of capital: -1 to +1
    position_dollars = 0.0

    # Track everything
    daily_log = []
    equity_curve = [capital]
    thresholds = None
    last_recal = 0
    recal_count = 0

    for t in range(sim_start, sim_end):
        # Recalibrate if needed
        days_since_recal = t - sim_start - last_recal
        if thresholds is None or days_since_recal >= recal_days:
            cal_start = max(0, t - cal_days)
            alpha_hist = alpha[cal_start:t]
            da_hist = delta_alpha[cal_start:t]
            ret_hist = returns[cal_start:t]
            thresholds = calibrate_thresholds(alpha_hist, da_hist)
            thresholds['_ret_hist'] = ret_hist
            thresholds['_alpha_hist'] = alpha_hist
            last_recal = t - sim_start
            recal_count += 1

        # Get action
        action, base_size = get_action(alpha[t], delta_alpha[t], thresholds)

        # Compute risk scaler
        risk_scaler, abs_regime = compute_risk_scaler(
            alpha[t], thresholds,
            thresholds['_alpha_hist'], thresholds['_ret_hist']
        )

        # Target position
        if action == 'FLAT':
            target_pos = 0.0
        elif action == 'SHORT':
            target_pos = -base_size * risk_scaler
        else:
            target_pos = base_size * risk_scaler

        # Transaction cost
        pos_change = abs(target_pos - position)
        cost = pos_change * cost_frac * capital

        # Apply today's return to yesterday's position
        day_return = returns[t]
        pnl = position * day_return * capital - cost

        capital += pnl
        position = target_pos

        equity_curve.append(capital)

        daily_log.append({
            'date': timestamps[t][:10],
            'alpha': float(alpha[t]),
            'da': float(delta_alpha[t]),
            'regime': abs_regime,
            'action': action,
            'position': float(position),
            'risk_scaler': float(risk_scaler),
            'spy': float(spy[t]),
            'day_return': float(day_return * 100),
            'pnl': float(pnl),
            'capital': float(capital),
            'cost': float(cost),
        })

    equity = np.array(equity_curve)

    # --- Analysis ---
    print(f"\n  Recalibrations: {recal_count}")

    # Overall performance
    final_capital = equity[-1]
    total_return = (final_capital - args.starting_capital) / args.starting_capital * 100
    spy_start = spy[sim_start]
    spy_end = spy[-1]
    bh_return = (spy_end - spy_start) / spy_start * 100

    # Annualized
    n_years = (sim_end - sim_start) / 252
    annual_return = ((final_capital / args.starting_capital) ** (1 / n_years) - 1) * 100
    bh_annual = ((spy_end / spy_start) ** (1 / n_years) - 1) * 100

    # Sharpe
    daily_returns = np.diff(equity) / equity[:-1]
    sharpe = daily_returns.mean() / (daily_returns.std() + 1e-8) * sqrt(252)

    # Max drawdown
    running_max = np.maximum.accumulate(equity)
    drawdowns = (running_max - equity) / running_max * 100
    max_dd = drawdowns.max()
    max_dd_idx = drawdowns.argmax()

    # Drawdown duration
    in_dd = equity < running_max
    dd_starts = np.where(np.diff(in_dd.astype(int)) == 1)[0]
    dd_ends = np.where(np.diff(in_dd.astype(int)) == -1)[0]
    if len(dd_starts) > 0 and len(dd_ends) > 0:
        max_dd_duration = max(e - s for s, e in zip(dd_starts, dd_ends[:len(dd_starts)]))
    else:
        max_dd_duration = 0

    # Win rate
    log_df_returns = [d['pnl'] for d in daily_log]
    active_days = [d for d in daily_log if abs(d['position']) > 0.01]
    win_rate = sum(1 for d in active_days if d['pnl'] > 0) / max(len(active_days), 1) * 100

    # Total costs
    total_costs = sum(d['cost'] for d in daily_log)

    # Trade count
    positions = [d['position'] for d in daily_log]
    n_trades = sum(1 for i in range(1, len(positions)) if abs(positions[i] - positions[i-1]) > 0.01)

    # Action breakdown
    from collections import Counter
    action_counts = Counter(d['action'] for d in daily_log)

    print(f"\n{'=' * 70}")
    print("SIMULATION RESULTS")
    print(f"{'=' * 70}")
    print(f"""
  Period:           {timestamps[sim_start][:10]} to {timestamps[-1][:10]} ({n_years:.1f} years)
  Starting Capital: ${args.starting_capital:>12,.0f}
  Final Capital:    ${final_capital:>12,.0f}

  STRATEGY:
    Total Return:   {total_return:>+8.2f}%
    Annual Return:  {annual_return:>+8.2f}%
    Sharpe Ratio:   {sharpe:>8.2f}
    Max Drawdown:   {max_dd:>8.2f}%
    Max DD Duration:{max_dd_duration:>5} days
    Win Rate:       {win_rate:>8.1f}% (of active days)
    Total Trades:   {n_trades:>5}
    Total Costs:    ${total_costs:>10,.0f}

  BUY AND HOLD:
    Total Return:   {bh_return:>+8.2f}%
    Annual Return:  {bh_annual:>+8.2f}%

  EDGE:
    Return Edge:    {total_return - bh_return:>+8.2f}%
    Risk-Adjusted:  Strategy Sharpe {sharpe:.2f} vs ~0.7 B&H
""")

    # Action distribution
    print("  ACTION DISTRIBUTION:")
    total_days = len(daily_log)
    for action in ['LONG', 'MILD_LONG', 'FLAT', 'SHORT']:
        n = action_counts.get(action, 0)
        bar = '#' * int(n / total_days * 50)
        pnl_for_action = sum(d['pnl'] for d in daily_log if d['action'] == action)
        print(f"    {action:<12} {n:>4} days ({n/total_days*100:>4.1f}%)  "
              f"PnL: ${pnl_for_action:>+10,.0f}  {bar}")

    # Yearly breakdown
    print(f"\n  YEARLY BREAKDOWN:")
    print(f"  {'Year':<6} {'Return':>8} {'Sharpe':>7} {'MaxDD':>7} {'Trades':>7} "
          f"{'B&H':>8} {'Edge':>8}")
    print(f"  {'-' * 55}")

    years = sorted(set(d['date'][:4] for d in daily_log))
    for year in years:
        year_days = [d for d in daily_log if d['date'][:4] == year]
        if len(year_days) < 10:
            continue

        year_start_cap = year_days[0]['capital'] - year_days[0]['pnl']
        year_end_cap = year_days[-1]['capital']
        yr_ret = (year_end_cap - year_start_cap) / year_start_cap * 100

        yr_returns = np.array([(year_days[i]['capital'] - year_days[i-1]['capital']) /
                               year_days[i-1]['capital']
                               for i in range(1, len(year_days))])
        yr_sharpe = yr_returns.mean() / (yr_returns.std() + 1e-8) * sqrt(252)

        yr_equity = np.array([year_start_cap] + [d['capital'] for d in year_days])
        yr_rm = np.maximum.accumulate(yr_equity)
        yr_dd = ((yr_rm - yr_equity) / yr_rm * 100).max()

        yr_trades = sum(1 for i in range(1, len(year_days))
                        if abs(year_days[i]['position'] - year_days[i-1]['position']) > 0.01)

        # B&H for this year
        yr_spy_start = year_days[0]['spy']
        yr_spy_end = year_days[-1]['spy']
        yr_bh = (yr_spy_end - yr_spy_start) / yr_spy_start * 100

        print(f"  {year:<6} {yr_ret:>+7.2f}% {yr_sharpe:>+6.2f} {yr_dd:>6.2f}% "
              f"{yr_trades:>7} {yr_bh:>+7.2f}% {yr_ret - yr_bh:>+7.2f}%")

    # Quarterly breakdown
    print(f"\n  QUARTERLY BREAKDOWN:")
    print(f"  {'Quarter':<8} {'Return':>8} {'Sharpe':>7} {'MaxDD':>7} {'B&H':>8} {'Edge':>8}")
    print(f"  {'-' * 50}")

    quarters = sorted(set(d['date'][:4] + '-Q' + str((int(d['date'][5:7]) - 1) // 3 + 1)
                          for d in daily_log))
    for q in quarters:
        year = q[:4]
        qnum = int(q[-1])
        q_days = [d for d in daily_log
                   if d['date'][:4] == year and (int(d['date'][5:7]) - 1) // 3 + 1 == qnum]
        if len(q_days) < 5:
            continue

        q_start_cap = q_days[0]['capital'] - q_days[0]['pnl']
        q_end_cap = q_days[-1]['capital']
        q_ret = (q_end_cap - q_start_cap) / q_start_cap * 100

        q_returns = np.array([(q_days[i]['capital'] - q_days[i-1]['capital']) /
                               q_days[i-1]['capital']
                               for i in range(1, len(q_days))])
        q_sharpe = q_returns.mean() / (q_returns.std() + 1e-8) * sqrt(252) if len(q_returns) > 1 else 0

        q_equity = np.array([q_start_cap] + [d['capital'] for d in q_days])
        q_rm = np.maximum.accumulate(q_equity)
        q_dd = ((q_rm - q_equity) / q_rm * 100).max()

        q_bh = (q_days[-1]['spy'] - q_days[0]['spy']) / q_days[0]['spy'] * 100

        print(f"  {q:<8} {q_ret:>+7.2f}% {q_sharpe:>+6.2f} {q_dd:>6.2f}% "
              f"{q_bh:>+7.2f}% {q_ret - q_bh:>+7.2f}%")

    # Worst and best days
    sorted_days = sorted(daily_log, key=lambda d: d['pnl'])
    print(f"\n  5 WORST DAYS:")
    for d in sorted_days[:5]:
        print(f"    {d['date']}  PnL=${d['pnl']:>+8,.0f}  pos={d['position']:>+.2f}  "
              f"SPY={d['day_return']:>+.2f}%  action={d['action']}")

    print(f"\n  5 BEST DAYS:")
    for d in sorted_days[-5:]:
        print(f"    {d['date']}  PnL=${d['pnl']:>+8,.0f}  pos={d['position']:>+.2f}  "
              f"SPY={d['day_return']:>+.2f}%  action={d['action']}")

    # Save
    save_path = RESULTS_DIR / 'simulation_3yr.json'
    save_data = {
        'config': {
            'starting_capital': args.starting_capital,
            'cost_bps': args.cost_bps,
            'calibration_days': cal_days,
            'recal_days': recal_days,
        },
        'results': {
            'period': f"{timestamps[sim_start][:10]} to {timestamps[-1][:10]}",
            'n_years': round(n_years, 1),
            'final_capital': round(final_capital, 2),
            'total_return': round(total_return, 2),
            'annual_return': round(annual_return, 2),
            'sharpe': round(sharpe, 2),
            'max_drawdown': round(float(max_dd), 2),
            'win_rate': round(win_rate, 1),
            'n_trades': n_trades,
            'total_costs': round(total_costs, 2),
            'bh_return': round(float(bh_return), 2),
            'bh_annual': round(float(bh_annual), 2),
        },
        'equity_curve': [round(float(e), 2) for e in equity[::5]],  # every 5th point
    }
    with open(save_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\n  Saved: {save_path}")


if __name__ == '__main__':
    main()
