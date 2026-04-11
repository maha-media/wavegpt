"""
Unified Trader — Tech 7 momentum + regime rotation + defensive assets.

Two engines merged:
  ENGINE 1: Tech 7 Flow Trader (momentum-first, dip-buying, lead-lag)
  ENGINE 2: Regime Rotation (which asset class wins in this environment)

The regime decides:
  NORMAL    -> Full tech, momentum-weighted
  RISK_ON   -> Split: tech momentum + energy/commodities
  FEAR      -> Tech dip-buying (NVDA leads bounces) + SHY safety
  CRISIS    -> MAXIMUM tech (crisis = best tech buying, Sharpe 4-5)
  INFLATION -> International + small caps + gold
  RECESSION -> Commodities only (USO, SLV, XLE)

Usage:
    python finance/unified_trader.py
"""

import argparse
import json
import sys
from math import sqrt
from pathlib import Path
from collections import defaultdict, deque

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent / 'data'
RESULTS_DIR = Path(__file__).parent / 'training_results'
RESULTS_DIR.mkdir(exist_ok=True)

TECH7 = ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 'META', 'TSLA']

# Regime-specific asset pools (from regime_rotation.py discovery)
REGIME_ASSETS = {
    'NORMAL':          {'tech_pct': 0.90, 'other': ['XLK', 'XLP', 'HYG'], 'other_pct': 0.10},
    'RISK_ON':         {'tech_pct': 0.50, 'other': ['USO', 'XLE', 'XLU', 'GLD'], 'other_pct': 0.50},
    'FEAR':            {'tech_pct': 0.60, 'other': ['SHY', 'HYG', 'XLV', 'GLD'], 'other_pct': 0.40},
    'CRISIS':          {'tech_pct': 0.95, 'other': ['HYG', 'XLU'], 'other_pct': 0.05},
    'INFLATION':       {'tech_pct': 0.30, 'other': ['EFA', 'XLF', 'EEM', 'IWM', 'XLB', 'GLD'], 'other_pct': 0.70},
    'RECESSION_RISK':  {'tech_pct': 0.10, 'other': ['USO', 'SLV', 'XLE', 'GLD'], 'other_pct': 0.90},
    'UNKNOWN':         {'tech_pct': 0.60, 'other': ['GLD', 'SHY', 'XLU'], 'other_pct': 0.40},
}


def classify_regime(vix_z, credit_level, gold_z, yield_curve_level):
    """Classify current market regime from signals."""
    if np.isnan(vix_z) or np.isnan(credit_level):
        return 'UNKNOWN'
    if vix_z > 1.0 and credit_level < -0.5:
        return 'CRISIS'
    if vix_z > 0.5:
        return 'FEAR'
    if credit_level > 0.5 and vix_z < 0:
        return 'RISK_ON'
    if not np.isnan(gold_z) and gold_z > 1.0:
        return 'INFLATION'
    if not np.isnan(yield_curve_level) and yield_curve_level < -1.0:
        return 'RECESSION_RISK'
    return 'NORMAL'


def main():
    parser = argparse.ArgumentParser(description='Unified Trader')
    parser.add_argument('--starting-capital', type=float, default=100000)
    parser.add_argument('--cost-bps', type=float, default=5.0)
    parser.add_argument('--w-momentum', type=float, default=1.0)
    args = parser.parse_args()

    # Load all data
    tech_closes = pd.read_parquet(DATA_DIR / 'tech7_closes.parquet')
    universe_path = DATA_DIR / 'universe_closes.parquet'
    if not universe_path.exists():
        print("ERROR: Run regime_rotation.py first to download universe data")
        sys.exit(1)
    uni_closes = pd.read_parquet(universe_path)

    # Load leading indicators
    leaders_path = DATA_DIR / 'leaders_closes.parquet'
    if leaders_path.exists():
        leaders_closes = pd.read_parquet(leaders_path)
    else:
        leaders_closes = pd.DataFrame()

    # Align dates
    common_dates = tech_closes.index.intersection(uni_closes.index)
    tech_closes = tech_closes.loc[common_dates]
    uni_closes = uni_closes.loc[common_dates]
    if not leaders_closes.empty:
        leaders_closes = leaders_closes.reindex(common_dates, method='ffill')

    tech_returns = tech_closes.pct_change()
    uni_returns = uni_closes.pct_change()
    T = len(common_dates)

    has_leaders = not leaders_closes.empty

    print("=" * 70)
    print("UNIFIED TRADER v2 — Tech Flow + Regime + Leading Indicators")
    print(f"  {T} days, {common_dates[0].date()} to {common_dates[-1].date()}")
    print(f"  Tech: {len(TECH7)} stocks, Universe: {uni_closes.shape[1]} assets")
    if has_leaders:
        print(f"  Leading indicators: {leaders_closes.shape[1]} signals")
    print("=" * 70)

    # Precompute regime signals
    vix = uni_closes['^VIX'] if '^VIX' in uni_closes else pd.Series(0, index=common_dates)
    tlt = uni_closes.get('TLT', pd.Series(0, index=common_dates))
    shy = uni_closes.get('SHY', pd.Series(1, index=common_dates))
    hyg = uni_closes.get('HYG', pd.Series(0, index=common_dates))
    gld = uni_closes.get('GLD', pd.Series(0, index=common_dates))

    # Simulation
    cal_days = 252
    recal_days = 63       # quarterly recalibration (most stable)
    warmup_days = 252
    sim_start = warmup_days
    cost_frac = args.cost_bps / 10000.0
    capital = args.starting_capital

    equity = [capital]
    daily_log = []
    last_recal = -999
    prev_alloc = {}
    prev_regime = 'UNKNOWN'
    regime_hold_counter = 0  # how many days in current regime
    REGIME_CONFIRM_DAYS = 3  # need 3 days of agreement to switch

    # Cached training stats
    momentum_scale = {}

    for t in range(sim_start, T):
        # --- Recalibrate ---
        if t - last_recal >= recal_days or not momentum_scale:
            cal_start = max(0, t - cal_days)
            for sym in TECH7:
                mom_std = tech_closes[sym].pct_change(10).iloc[cal_start:t].std()
                momentum_scale[sym] = float(mom_std) if mom_std > 0 else 1.0
            last_recal = t

        # --- Regime classification ---
        lookback = min(50, t)
        vix_val = vix.iloc[t]
        vix_hist = vix.iloc[max(0,t-lookback):t]
        vix_z = (vix_val - vix_hist.mean()) / (vix_hist.std() + 1e-8) if vix_hist.std() > 0 else 0

        credit_ratio = hyg.iloc[t] / (tlt.iloc[t] + 1e-8) if tlt.iloc[t] > 0 else 1
        cr_hist = (hyg.iloc[max(0,t-lookback):t] / (tlt.iloc[max(0,t-lookback):t] + 1e-8))
        credit_level = (credit_ratio - cr_hist.mean()) / (cr_hist.std() + 1e-8) if cr_hist.std() > 0 else 0

        gld_val = gld.iloc[t]
        gld_hist = gld.iloc[max(0,t-lookback):t]
        gold_z = (gld_val - gld_hist.mean()) / (gld_hist.std() + 1e-8) if gld_hist.std() > 0 else 0

        curve = tlt.iloc[t] / (shy.iloc[t] + 1e-8) if shy.iloc[t] > 0 else 1
        curve_hist = (tlt.iloc[max(0,t-lookback):t] / (shy.iloc[max(0,t-lookback):t] + 1e-8))
        curve_z = (curve - curve_hist.mean()) / (curve_hist.std() + 1e-8) if curve_hist.std() > 0 else 0

        regime = classify_regime(vix_z, credit_level, gold_z, curve_z)
        regime_config = REGIME_ASSETS.get(regime, REGIME_ASSETS['NORMAL'])
        tech_pct = regime_config['tech_pct']
        other_assets = regime_config['other']
        other_pct = regime_config['other_pct']

        # --- ENGINE 3: Leading Indicator Conviction Score ---
        # This OVERRIDES regime tech_pct when leading indicators are screaming
        # Positive score = bullish for tech (increase tech_pct)
        # Negative score = bearish for tech (decrease tech_pct)
        leader_score = 0.0
        n_leader_signals = 0

        if has_leaders and t >= 10:
            def safe_mom(series, t_idx, lookback):
                """Compute momentum safely."""
                if t_idx < lookback:
                    return None
                now = series.iloc[t_idx]
                prev = series.iloc[t_idx - lookback]
                if prev > 0 and not np.isnan(now) and not np.isnan(prev):
                    return (now - prev) / prev
                return None

            # Each signal: compute z-scored momentum and weight by discovery strength
            # Weights from the correlation study (section 5 ranked leaders)
            leader_signals = []

            # ARKK 5d (corr 0.133, inverse, V-recovery king)
            m = safe_mom(leaders_closes.get('ARKK', pd.Series(dtype=float)), t, 5)
            if m is not None:
                leader_signals.append(('ARKK', -m, 4.0))  # inverse, HEAVY

            # VIX 5d (corr 0.121, same direction — VIX up = tech up at lag)
            m = safe_mom(leaders_closes.get('^VIX', pd.Series(dtype=float)), t, 5)
            if m is not None:
                leader_signals.append(('VIX', m, 3.5))  # same direction

            # HYG 5d (corr 0.179, inverse — strongest leader)
            m = safe_mom(leaders_closes.get('HYG', pd.Series(dtype=float)), t, 5)
            if m is not None:
                leader_signals.append(('HYG', -m, 5.0))  # inverse, STRONGEST

            # KWEB 5d (corr 0.163, inverse)
            m = safe_mom(leaders_closes.get('KWEB', pd.Series(dtype=float)), t, 5)
            if m is not None:
                leader_signals.append(('KWEB', -m, 3.0))

            # FXI 5d (corr 0.165, inverse)
            m = safe_mom(leaders_closes.get('FXI', pd.Series(dtype=float)), t, 5)
            if m is not None:
                leader_signals.append(('FXI', -m, 3.0))

            # SMH 5d (corr 0.107, inverse)
            m = safe_mom(leaders_closes.get('SMH', pd.Series(dtype=float)), t, 5)
            if m is not None:
                leader_signals.append(('SMH', -m, 3.0))

            # LQD 2d (corr 0.147, SAME direction)
            m = safe_mom(leaders_closes.get('LQD', pd.Series(dtype=float)), t, 2)
            if m is not None:
                leader_signals.append(('LQD', m, 3.5))  # same direction

            # UUP 5d (corr 0.119, same)
            m = safe_mom(leaders_closes.get('UUP', pd.Series(dtype=float)), t, 5)
            if m is not None:
                leader_signals.append(('UUP', m, 2.5))

            # ETH 2d (corr 0.049, inverse)
            m = safe_mom(leaders_closes.get('ETH-USD', pd.Series(dtype=float)), t, 2)
            if m is not None:
                leader_signals.append(('ETH', -m, 1.0))  # weak

            # Weighted average
            if leader_signals:
                total_weight = sum(w for _, _, w in leader_signals)
                leader_score = sum(s * w for _, s, w in leader_signals) / total_weight
                n_leader_signals = len(leader_signals)

        # APPLY leading indicator conviction to tech allocation
        # The score is in raw return units (~0.01 to 0.10 typically)
        # We want: score of 0.05 (5% move in leaders) to shift tech by ~25%
        if n_leader_signals >= 3:
            tech_adjustment = np.clip(leader_score * 5.0, -0.40, +0.40)
            tech_pct = np.clip(tech_pct + tech_adjustment, 0.10, 0.95)
            other_pct = 1.0 - tech_pct

        # --- SINGULARITY OVERRIDE ---
        # When tech is unanimously ripping, the market IS the signal.
        # Don't hedge. Don't rotate. Ride the wave.
        singularity_mode = False
        if t >= 20:
            n_positive_20d = sum(1 for sym in TECH7
                                if (tech_closes[sym].iloc[t] - tech_closes[sym].iloc[t-20]) /
                                    tech_closes[sym].iloc[t-20] > 0)
            avg_mom_20d = np.mean([(tech_closes[sym].iloc[t] - tech_closes[sym].iloc[t-20]) /
                                    tech_closes[sym].iloc[t-20] for sym in TECH7])

            # Singularity: 6+ of 7 stocks positive AND avg momentum > 5%
            if n_positive_20d >= 6 and avg_mom_20d > 0.05:
                singularity_mode = True
                tech_pct = 0.95  # all in tech
                other_pct = 0.05
            # Strong bull: 5+ positive AND avg > 2%
            elif n_positive_20d >= 5 and avg_mom_20d > 0.02:
                tech_pct = max(tech_pct, 0.80)  # at least 80% tech
                other_pct = 1.0 - tech_pct

        # --- ENGINE 1: Tech 7 momentum weights ---
        mom_scores = np.zeros(7)
        for i, sym in enumerate(TECH7):
            for lookback_w in [10, 20, 50, 100]:
                if t >= lookback_w:
                    mom = (tech_closes[sym].iloc[t] - tech_closes[sym].iloc[t - lookback_w]) / tech_closes[sym].iloc[t - lookback_w]
                    z = mom / (momentum_scale.get(sym, 1.0) + 1e-8)
                    w = {10: 0.15, 20: 0.20, 50: 0.30, 100: 0.35}[lookback_w]
                    mom_scores[i] += z * w

        temperature = 1.0 / max(args.w_momentum, 0.1)
        mom_exp = np.exp(mom_scores / temperature)
        tech_weights = mom_exp / mom_exp.sum()
        tech_weights = np.clip(tech_weights, 0.05, 0.35)
        tech_weights /= tech_weights.sum()

        # Dip boost: if any tech stock dropped >2% yesterday, boost it + NVDA
        if t >= 1:
            for i, sym in enumerate(TECH7):
                if tech_returns[sym].iloc[t - 1] < -0.02:
                    drop = abs(tech_returns[sym].iloc[t - 1])
                    tech_weights[i] *= (1 + drop * 3)
                    nvda_i = TECH7.index('NVDA')
                    tech_weights[nvda_i] *= (1 + drop * 2)
            tech_weights /= tech_weights.sum()

        # Scale tech weights by regime's tech allocation
        tech_alloc = {sym: float(w * tech_pct) for sym, w in zip(TECH7, tech_weights)}

        # --- Macro risk switch (bear market detection) ---
        # SKIP in singularity mode — the wave overrides the risk switch
        if t >= 50 and not singularity_mode:
            avg_mom50 = np.mean([(tech_closes[sym].iloc[t] - tech_closes[sym].iloc[t-50]) /
                                  tech_closes[sym].iloc[t-50] for sym in TECH7])
            n_above_ma = sum(1 for sym in TECH7
                            if tech_closes[sym].iloc[t] > tech_closes[sym].iloc[t-50:t].mean())

            if avg_mom50 < -0.05 and n_above_ma <= 2:
                for sym in tech_alloc:
                    tech_alloc[sym] *= 0.3
                other_pct = 1.0 - sum(tech_alloc.values())
            elif avg_mom50 < 0 and n_above_ma <= 4:
                for sym in tech_alloc:
                    tech_alloc[sym] *= 0.6
                other_pct = 1.0 - sum(tech_alloc.values())

        # --- ENGINE 2: Other assets momentum-weighted ---
        other_alloc = {}
        if other_pct > 0.01 and other_assets:
            other_moms = []
            for sym in other_assets:
                if sym in uni_closes.columns and t >= 50:
                    p = uni_closes[sym].iloc[t]
                    p50 = uni_closes[sym].iloc[t - 50]
                    if p50 > 0 and not np.isnan(p) and not np.isnan(p50):
                        other_moms.append(max((p - p50) / p50, 0))
                    else:
                        other_moms.append(0)
                else:
                    other_moms.append(0)

            other_moms = np.array(other_moms)
            if other_moms.sum() > 0:
                other_w = other_moms / other_moms.sum() * other_pct
            else:
                other_w = np.ones(len(other_assets)) / len(other_assets) * other_pct

            for i, sym in enumerate(other_assets):
                other_alloc[sym] = float(other_w[i])

        # --- Combined allocation ---
        alloc = {**tech_alloc, **other_alloc}

        # --- PnL ---
        port_ret = 0.0
        for sym, w in prev_alloc.items():
            if sym in tech_returns.columns:
                r = tech_returns[sym].iloc[t]
            elif sym in uni_returns.columns:
                r = uni_returns[sym].iloc[t]
            else:
                r = 0
            if not np.isnan(r):
                port_ret += w * r

        # Costs
        turnover = 0
        all_syms = set(list(alloc.keys()) + list(prev_alloc.keys()))
        for sym in all_syms:
            turnover += abs(alloc.get(sym, 0) - prev_alloc.get(sym, 0))
        cost = turnover * cost_frac * capital

        pnl = port_ret * capital - cost
        capital += pnl
        equity.append(capital)

        # Top holdings for log
        sorted_alloc = sorted(alloc.items(), key=lambda x: x[1], reverse=True)
        top3 = sorted_alloc[:3]
        tech_total = sum(v for k, v in alloc.items() if k in TECH7)
        other_total = sum(v for k, v in alloc.items() if k not in TECH7)

        daily_log.append({
            'date': str(common_dates[t].date()),
            'regime': ('SINGULARITY' if singularity_mode else regime),
            'leader_score': round(leader_score, 3),
            'tech_pct': round(tech_total * 100, 1),
            'other_pct': round(other_total * 100, 1),
            'top': [(s, round(w * 100, 1)) for s, w in top3],
            'port_return': round(port_ret * 100, 3),
            'pnl': round(pnl, 0),
            'capital': round(capital, 0),
        })

        prev_alloc = alloc

    # --- Results ---
    equity = np.array(equity)
    final = equity[-1]
    total_ret = (final - args.starting_capital) / args.starting_capital * 100
    n_years = (T - sim_start) / 252
    annual = ((final / args.starting_capital) ** (1 / n_years) - 1) * 100
    daily_rets = np.diff(equity) / equity[:-1]
    sharpe = daily_rets.mean() / (daily_rets.std() + 1e-8) * sqrt(252)
    rm = np.maximum.accumulate(equity)
    max_dd = ((rm - equity) / rm * 100).max()

    # Baselines
    eq_tech = tech_returns[TECH7].iloc[sim_start:].mean(axis=1)
    eq_tech_ret = ((1 + eq_tech).prod() - 1) * 100

    print(f"\n{'=' * 70}")
    print("UNIFIED TRADER RESULTS")
    print(f"{'=' * 70}")
    print(f"""
  Period: {daily_log[0]['date']} to {daily_log[-1]['date']} ({n_years:.1f} years)
  Starting Capital: ${args.starting_capital:>12,.0f}
  Final Capital:    ${final:>12,.0f}

  UNIFIED TRADER:
    Total Return:   {total_ret:>+8.2f}%
    Annual Return:  {annual:>+8.2f}%
    Sharpe Ratio:   {sharpe:>8.2f}
    Max Drawdown:   {max_dd:>8.2f}%

  BASELINES:
    EqWeight Tech 7: {eq_tech_ret:>+8.2f}%

  EDGE vs EqWeight Tech: {total_ret - eq_tech_ret:>+8.2f}%
""")

    # Yearly
    print("  YEARLY BREAKDOWN:")
    print(f"  {'Year':<6} {'Return':>8} {'Sharpe':>7} {'MaxDD':>7} {'EqTech':>8} {'Edge':>8} {'Regime Mix'}")
    print(f"  {'-' * 75}")

    years = sorted(set(d['date'][:4] for d in daily_log))
    for year in years:
        yr = [d for d in daily_log if d['date'][:4] == year]
        if len(yr) < 10:
            continue
        yr_start = yr[0]['capital'] / (1 + yr[0]['port_return'] / 100) if yr[0]['port_return'] != 0 else yr[0]['capital'] - yr[0]['pnl']
        yr_end = yr[-1]['capital']
        yr_ret = (yr_end - yr_start) / yr_start * 100

        yr_r = np.array([d['port_return'] / 100 for d in yr])
        yr_sharpe = yr_r.mean() / (yr_r.std() + 1e-8) * sqrt(252)

        yr_eq = np.array([yr_start] + [d['capital'] for d in yr])
        yr_dd = ((np.maximum.accumulate(yr_eq) - yr_eq) / np.maximum.accumulate(yr_eq) * 100).max()

        # EqTech
        yr_mask = tech_returns.index.year == int(year)
        yr_eq_tech = ((1 + tech_returns[TECH7][yr_mask].mean(axis=1)).prod() - 1) * 100

        # Regime mix
        regime_counts = defaultdict(int)
        for d in yr:
            regime_counts[d['regime']] += 1
        top_regimes = sorted(regime_counts.items(), key=lambda x: x[1], reverse=True)[:2]
        regime_str = ' '.join(f"{r[:4]}:{n}" for r, n in top_regimes)

        # Avg tech vs other allocation
        avg_tech = np.mean([d['tech_pct'] for d in yr])
        avg_other = np.mean([d['other_pct'] for d in yr])

        print(f"  {year:<6} {yr_ret:>+7.2f}% {yr_sharpe:>+6.2f} {yr_dd:>6.2f}% "
              f"{yr_eq_tech:>+7.2f}% {yr_ret - yr_eq_tech:>+7.2f}% "
              f"tech:{avg_tech:.0f}% def:{avg_other:.0f}% {regime_str}")

    # Last 15 days
    print(f"\n  LAST 15 DAYS:")
    print(f"  {'Date':<12} {'Regime':<12} {'Lead':>5} {'Tech%':>5} {'Def%':>5} {'Return':>7} {'Capital':>12} {'Top Holdings'}")
    print(f"  {'-' * 95}")
    for d in daily_log[-15:]:
        top_str = ' '.join(f"{s}={w:.0f}%" for s, w in d['top'][:3])
        ls = d.get('leader_score', 0)
        print(f"  {d['date']:<12} {d['regime']:<12} {ls:>+4.2f} {d['tech_pct']:>4.0f}% {d['other_pct']:>4.0f}% "
              f"{d['port_return']:>+6.2f}% ${d['capital']:>11,.0f} {top_str}")

    # Save
    save_path = RESULTS_DIR / 'unified_trader.json'
    with open(save_path, 'w') as f:
        json.dump({
            'total_return': round(total_ret, 2),
            'annual_return': round(annual, 2),
            'sharpe': round(float(sharpe), 2),
            'max_drawdown': round(float(max_dd), 2),
            'eq_tech_return': round(float(eq_tech_ret), 2),
        }, f, indent=2)
    print(f"\n  Saved: {save_path}")


if __name__ == '__main__':
    main()
