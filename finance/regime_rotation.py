"""
Full Regime Rotation — map every market environment to the best assets.

Expanded universe:
  - Tech 7 (momentum + flow)
  - Sectors: XLE, XLF, XLV, XLI, XLB, XLK, XLRE
  - Commodities: GLD, SLV, USO
  - Bonds: TLT, SHY, HYG
  - Defensive: XLU, XLP
  - Size: IWM (small cap)
  - International: EFA, EEM
  - Vol: VIX-derived signals

Signals we haven't tried:
  1. Yield curve proxy (TLT/SHY ratio = long vs short bonds)
  2. Dollar proxy (UUP or inverse: gold strength)
  3. Credit stress (HYG/TLT = junk vs treasury spread)
  4. Sector rotation momentum (which sector is leading?)
  5. Cross-asset momentum (stocks vs bonds vs commodities)
  6. VIX term structure (contango vs backwardation proxy)
  7. Small vs large (IWM/SPY ratio = risk appetite)
  8. Earnings season calendar effect

For each regime, find which assets outperform.
Then build a rotation model that shifts between them.

Usage:
    python finance/regime_rotation.py
"""

import json
import sys
from math import sqrt
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import yfinance as yf

DATA_DIR = Path(__file__).parent / 'data'
RESULTS_DIR = Path(__file__).parent / 'training_results'
RESULTS_DIR.mkdir(exist_ok=True)

# Full universe
TECH7 = ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 'META', 'TSLA']
SECTORS = ['XLE', 'XLF', 'XLV', 'XLI', 'XLB', 'XLK']
COMMODITIES = ['GLD', 'SLV', 'USO']
BONDS = ['TLT', 'SHY', 'HYG']
DEFENSIVE = ['XLU', 'XLP']
SIZE_INTL = ['IWM', 'EFA', 'EEM']
MACRO_SIGNALS = ['^VIX', '^TNX']  # VIX + 10yr yield

ALL_TRADEABLE = TECH7 + SECTORS + COMMODITIES + BONDS + DEFENSIVE + SIZE_INTL
ALL_DOWNLOAD = ALL_TRADEABLE + MACRO_SIGNALS


def download_universe():
    """Download full universe."""
    print("  Downloading full universe...")
    data = yf.download(ALL_DOWNLOAD, period='5y', interval='1d', auto_adjust=True)
    closes = data['Close'].dropna(how='all')
    print(f"  Got {len(closes)} days, {closes.shape[1]} assets")
    closes.to_parquet(DATA_DIR / 'universe_closes.parquet')
    return closes


def compute_regime_signals(closes):
    """Compute regime classification signals."""
    signals = pd.DataFrame(index=closes.index)

    # 1. Yield curve proxy: TLT/SHY = long vs short bonds
    #    Rising = curve steepening (growth), falling = flattening (recession risk)
    if 'TLT' in closes and 'SHY' in closes:
        curve = closes['TLT'] / closes['SHY']
        signals['yield_curve'] = curve.pct_change(20) * 100
        signals['yield_curve_level'] = (curve - curve.rolling(50).mean()) / (curve.rolling(50).std() + 1e-8)

    # 2. Credit stress: HYG/TLT = junk bond vs treasury
    #    Rising = risk-on (junk outperforms), falling = risk-off
    if 'HYG' in closes and 'TLT' in closes:
        credit = closes['HYG'] / closes['TLT']
        signals['credit_spread'] = credit.pct_change(20) * 100
        signals['credit_level'] = (credit - credit.rolling(50).mean()) / (credit.rolling(50).std() + 1e-8)

    # 3. Dollar proxy: GLD strength (inverse dollar)
    #    Rising gold = weak dollar = good for commodities, EM
    if 'GLD' in closes:
        signals['gold_momentum'] = closes['GLD'].pct_change(20) * 100
        signals['gold_z'] = (closes['GLD'] - closes['GLD'].rolling(50).mean()) / (closes['GLD'].rolling(50).std() + 1e-8)

    # 4. VIX level (fear gauge)
    if '^VIX' in closes:
        vix = closes['^VIX']
        signals['vix_level'] = vix
        signals['vix_z'] = (vix - vix.rolling(50).mean()) / (vix.rolling(50).std() + 1e-8)
        signals['vix_change'] = vix.pct_change(5) * 100

    # 5. Small vs Large: IWM/SPY risk appetite
    if 'IWM' in closes and 'AAPL' in closes:
        # Use tech equal weight as "large" proxy
        tech_eq = closes[TECH7].mean(axis=1) if all(s in closes for s in TECH7) else closes['AAPL']
        risk_appetite = closes['IWM'] / tech_eq
        signals['small_vs_large'] = risk_appetite.pct_change(20) * 100

    # 6. Cross-asset momentum: stocks vs bonds vs commodities
    if 'TLT' in closes and 'GLD' in closes:
        tech_mom = closes[TECH7].mean(axis=1).pct_change(20) if all(s in closes for s in TECH7) else 0
        bond_mom = closes['TLT'].pct_change(20)
        comm_mom = closes['GLD'].pct_change(20)
        if isinstance(tech_mom, pd.Series):
            signals['stocks_vs_bonds'] = (tech_mom - bond_mom) * 100
            signals['stocks_vs_commod'] = (tech_mom - comm_mom) * 100

    # 7. Sector rotation leader
    sector_mom = pd.DataFrame()
    for s in SECTORS:
        if s in closes:
            sector_mom[s] = closes[s].pct_change(20) * 100
    if not sector_mom.empty:
        signals['sector_dispersion'] = sector_mom.std(axis=1)
        valid_rows = sector_mom.notna().any(axis=1)
        signals.loc[valid_rows, 'sector_leader'] = sector_mom[valid_rows].idxmax(axis=1)
        # Best sector momentum
        signals['best_sector_mom'] = sector_mom.max(axis=1)
        signals['worst_sector_mom'] = sector_mom.min(axis=1)

    # 8. 10yr yield (rates)
    if '^TNX' in closes:
        tnx = closes['^TNX']
        signals['yield_10y'] = tnx
        signals['yield_change'] = tnx.diff(20)

    return signals


def regime_classification(signals):
    """Classify each day into a regime based on signals."""
    regimes = pd.Series(index=signals.index, dtype=str)

    for t in range(len(signals)):
        row = signals.iloc[t]

        vix_z = row.get('vix_z', 0)
        credit = row.get('credit_level', 0)
        curve = row.get('yield_curve_level', 0)
        gold_z = row.get('gold_z', 0)

        if pd.isna(vix_z) or pd.isna(credit):
            regimes.iloc[t] = 'UNKNOWN'
            continue

        # High VIX + falling credit = CRISIS
        if vix_z > 1.0 and credit < -0.5:
            regimes.iloc[t] = 'CRISIS'
        # High VIX but stable credit = FEAR (buying opportunity)
        elif vix_z > 0.5:
            regimes.iloc[t] = 'FEAR'
        # Rising credit + normal VIX = RISK_ON
        elif credit > 0.5 and vix_z < 0:
            regimes.iloc[t] = 'RISK_ON'
        # Strong gold + falling yields = INFLATION_HEDGE
        elif gold_z > 1.0:
            regimes.iloc[t] = 'INFLATION'
        # Falling curve = RECESSION_RISK
        elif curve < -1.0:
            regimes.iloc[t] = 'RECESSION_RISK'
        else:
            regimes.iloc[t] = 'NORMAL'

    return regimes


def main():
    print("=" * 70)
    print("FULL REGIME ROTATION — EXPANDED UNIVERSE")
    print(f"  {len(ALL_TRADEABLE)} tradeable assets + {len(MACRO_SIGNALS)} signal assets")
    print("=" * 70)

    # Load or download
    path = DATA_DIR / 'universe_closes.parquet'
    if path.exists():
        print("\n  Loading cached universe data...")
        closes = pd.read_parquet(path)
    else:
        closes = download_universe()

    available = [s for s in ALL_TRADEABLE if s in closes.columns]
    print(f"  Available: {len(available)} of {len(ALL_TRADEABLE)} tradeable assets")
    missing = [s for s in ALL_TRADEABLE if s not in closes.columns]
    if missing:
        print(f"  Missing: {', '.join(missing)}")

    returns = closes.pct_change()
    T = len(closes)
    print(f"  {T} trading days")

    # --- Compute regime signals ---
    print("\n  Computing regime signals...")
    signals = compute_regime_signals(closes)
    print(f"  {len(signals.columns)} signal features")

    regimes = regime_classification(signals)
    print(f"\n  Regime distribution:")
    for regime, count in regimes.value_counts().items():
        pct = count / len(regimes) * 100
        print(f"    {regime:<20} {count:>5} days ({pct:.1f}%)")

    # --- Which assets win in each regime? ---
    print("\n" + "=" * 70)
    print("ASSET PERFORMANCE BY REGIME")
    print("=" * 70)

    # Use first 3 years for discovery, last 1+ for test
    train_end = int(T * 0.75)

    for regime_name in ['RISK_ON', 'NORMAL', 'FEAR', 'CRISIS', 'INFLATION', 'RECESSION_RISK']:
        mask = (regimes == regime_name) & (regimes.index.isin(closes.index[:train_end]))
        n_days = mask.sum()
        if n_days < 20:
            continue

        print(f"\n  [{regime_name}] ({n_days} days)")

        # Next-day returns for each asset in this regime
        asset_perf = []
        for sym in available:
            next_ret = returns[sym].shift(-1)
            regime_rets = next_ret[mask].dropna()
            if len(regime_rets) < 10:
                continue
            avg = regime_rets.mean() * 100
            wr = (regime_rets > 0).mean() * 100
            sharpe = regime_rets.mean() / (regime_rets.std() + 1e-8) * sqrt(252)
            asset_perf.append({
                'sym': sym, 'avg_ret': avg, 'win_rate': wr, 'sharpe': sharpe,
            })

        asset_perf.sort(key=lambda x: x['sharpe'], reverse=True)

        print(f"  {'Rank':>4} {'Asset':<8} {'AvgRet':>8} {'WinRate':>8} {'Sharpe':>7}")
        print(f"  {'-' * 40}")
        for rank, a in enumerate(asset_perf[:10]):
            print(f"  {rank+1:>4} {a['sym']:<8} {a['avg_ret']:>+7.3f}% {a['win_rate']:>7.1f}% {a['sharpe']:>+6.2f}")

        if asset_perf:
            bottom = asset_perf[-3:]
            print(f"  ... bottom 3:")
            for a in bottom:
                print(f"       {a['sym']:<8} {a['avg_ret']:>+7.3f}% {a['win_rate']:>7.1f}% {a['sharpe']:>+6.2f}")

    # --- New signal discovery: what predicts regime transitions? ---
    print("\n" + "=" * 70)
    print("REGIME TRANSITION SIGNALS")
    print("  What predicts a shift from one regime to another?")
    print("=" * 70)

    # Look at what happens in the 5 days BEFORE a regime change
    regime_changes = []
    for t in range(5, len(regimes)):
        if regimes.iloc[t] != regimes.iloc[t-1]:
            regime_changes.append({
                'date': regimes.index[t],
                'from': regimes.iloc[t-1],
                'to': regimes.iloc[t],
            })

    print(f"\n  {len(regime_changes)} regime transitions in dataset")

    # Most common transitions
    transition_counts = defaultdict(int)
    for rc in regime_changes:
        key = f"{rc['from']} -> {rc['to']}"
        transition_counts[key] += 1

    print(f"\n  Most common transitions:")
    for trans, count in sorted(transition_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"    {trans:<35} {count:>4} times")

    # What signals predict transition INTO crisis/fear?
    print(f"\n  Signals 5 days before entering FEAR/CRISIS:")
    fear_entries = [rc for rc in regime_changes if rc['to'] in ('FEAR', 'CRISIS')]

    if fear_entries and not signals.empty:
        pre_fear_signals = []
        for rc in fear_entries:
            t_idx = signals.index.get_loc(rc['date'])
            if t_idx >= 5:
                pre = signals.iloc[t_idx - 5:t_idx].select_dtypes(include=[np.number]).mean()
                pre_fear_signals.append(pre)

        if pre_fear_signals:
            avg_pre_fear = pd.DataFrame(pre_fear_signals).mean()
            avg_normal = signals.iloc[:train_end].select_dtypes(include=[np.number]).mean()

            print(f"  {'Signal':<25} {'Pre-Fear':>10} {'Normal':>10} {'Diff':>10}")
            print(f"  {'-' * 58}")
            for col in signals.select_dtypes(include=[np.number]).columns:
                pf = avg_pre_fear.get(col, 0)
                nm = avg_normal.get(col, 0)
                if not np.isnan(pf) and not np.isnan(nm):
                    diff = pf - nm
                    if abs(diff) > 0.1:
                        print(f"  {col:<25} {pf:>+9.3f} {nm:>+9.3f} {diff:>+9.3f}")

    # --- Walk-forward simulation with regime rotation ---
    print("\n" + "=" * 70)
    print("WALK-FORWARD: REGIME-BASED ROTATION")
    print("=" * 70)

    cal_days = 252
    sim_start = cal_days
    capital = 100000.0
    equity = [capital]
    daily_log = []
    cost_frac = 5.0 / 10000

    # Build regime -> asset allocation map from training data
    # Updated every quarter
    regime_alloc = {}
    last_recal = -999

    for t in range(sim_start, T):
        current_regime = regimes.iloc[t]

        # Recalibrate allocation map
        if t - last_recal >= 63 or not regime_alloc:
            cal_start = max(0, t - cal_days)
            cal_regimes = regimes.iloc[cal_start:t]
            cal_returns = returns.iloc[cal_start:t]

            regime_alloc = {}
            for regime_name in cal_regimes.unique():
                if regime_name == 'UNKNOWN':
                    continue
                mask = cal_regimes == regime_name
                if mask.sum() < 15:
                    continue

                # Find top assets by Sharpe in this regime
                scores = {}
                for sym in available:
                    next_ret = cal_returns[sym].shift(-1)
                    r = next_ret[mask].dropna()
                    if len(r) < 10:
                        continue
                    sharpe = r.mean() / (r.std() + 1e-8) * sqrt(252)
                    scores[sym] = sharpe

                # Top 7 by Sharpe -> allocate momentum-weighted
                top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:7]
                if top:
                    # Softmax on Sharpe scores
                    vals = np.array([max(s, 0) for _, s in top])
                    if vals.sum() > 0:
                        weights = vals / vals.sum()
                    else:
                        weights = np.ones(len(top)) / len(top)
                    regime_alloc[regime_name] = {sym: float(w) for (sym, _), w in zip(top, weights)}

            last_recal = t

        # Get allocation for current regime
        if current_regime in regime_alloc:
            alloc = regime_alloc[current_regime]
        elif 'NORMAL' in regime_alloc:
            alloc = regime_alloc['NORMAL']
        else:
            alloc = {sym: 1.0 / len(available) for sym in available[:7]}

        # Portfolio return
        day_rets = {sym: returns[sym].iloc[t] if sym in returns and t < len(returns) else 0
                    for sym in alloc}
        port_ret = sum(alloc.get(sym, 0) * (day_rets.get(sym, 0) or 0) for sym in alloc)
        if np.isnan(port_ret):
            port_ret = 0

        pnl = port_ret * capital
        capital += pnl
        equity.append(capital)

        daily_log.append({
            'date': str(closes.index[t].date()),
            'regime': current_regime,
            'top_holdings': sorted(alloc.items(), key=lambda x: x[1], reverse=True)[:3],
            'port_return': float(port_ret * 100),
            'capital': float(capital),
        })

    equity = np.array(equity)
    final = equity[-1]
    total_ret = (final - 100000) / 100000 * 100
    n_years = (T - sim_start) / 252
    annual = ((final / 100000) ** (1 / n_years) - 1) * 100
    daily_rets = np.diff(equity) / equity[:-1]
    sharpe = daily_rets.mean() / (daily_rets.std() + 1e-8) * sqrt(252)
    rm = np.maximum.accumulate(equity)
    max_dd = ((rm - equity) / rm * 100).max()

    # Baselines
    eq_tech = returns[TECH7].iloc[sim_start:].mean(axis=1)
    eq_tech_ret = ((1 + eq_tech).prod() - 1) * 100
    eq_all = returns[available].iloc[sim_start:].mean(axis=1)
    eq_all_ret = ((1 + eq_all).prod() - 1) * 100

    print(f"""
  Period: {daily_log[0]['date']} to {daily_log[-1]['date']} ({n_years:.1f} years)
  Starting Capital: $100,000
  Final Capital:    ${final:,.0f}

  REGIME ROTATION:
    Total Return:   {total_ret:+.2f}%
    Annual Return:  {annual:+.2f}%
    Sharpe Ratio:   {sharpe:.2f}
    Max Drawdown:   {max_dd:.2f}%

  BASELINES:
    EqWeight Tech 7:     {eq_tech_ret:+.2f}%
    EqWeight All Assets: {eq_all_ret:+.2f}%

  EDGE vs EqWeight Tech: {total_ret - eq_tech_ret:+.2f}%
""")

    # Yearly
    print("  YEARLY BREAKDOWN:")
    print(f"  {'Year':<6} {'Strategy':>9} {'Sharpe':>7} {'MaxDD':>7} {'Regime mix'}")
    print(f"  {'-' * 65}")

    years = sorted(set(d['date'][:4] for d in daily_log))
    for year in years:
        yr_days = [d for d in daily_log if d['date'][:4] == year]
        if len(yr_days) < 10:
            continue
        yr_start = yr_days[0]['capital'] / (1 + yr_days[0]['port_return'] / 100)
        yr_end = yr_days[-1]['capital']
        yr_ret = (yr_end - yr_start) / yr_start * 100

        yr_r = np.array([d['port_return'] / 100 for d in yr_days])
        yr_sharpe = yr_r.mean() / (yr_r.std() + 1e-8) * sqrt(252)

        yr_eq = np.array([yr_start] + [d['capital'] for d in yr_days])
        yr_dd = ((np.maximum.accumulate(yr_eq) - yr_eq) / np.maximum.accumulate(yr_eq) * 100).max()

        # Regime distribution for this year
        regime_dist = defaultdict(int)
        for d in yr_days:
            regime_dist[d['regime']] += 1
        top_regimes = sorted(regime_dist.items(), key=lambda x: x[1], reverse=True)[:3]
        regime_str = ' '.join(f"{r}:{n}" for r, n in top_regimes)

        print(f"  {year:<6} {yr_ret:>+8.2f}% {yr_sharpe:>+6.2f} {yr_dd:>6.2f}% {regime_str}")

    # Last 10 days
    print(f"\n  LAST 10 DAYS:")
    for d in daily_log[-10:]:
        top = d['top_holdings']
        holdings = ' '.join(f"{s}={w:.0%}" for s, w in top[:3])
        print(f"  {d['date']}  {d['regime']:<15} {d['port_return']:>+6.2f}%  ${d['capital']:>10,.0f}  {holdings}")

    # Save
    save_path = RESULTS_DIR / 'regime_rotation.json'
    with open(save_path, 'w') as f:
        json.dump({
            'total_return': round(total_ret, 2),
            'sharpe': round(float(sharpe), 2),
            'max_drawdown': round(float(max_dd), 2),
            'regime_allocations': {k: v for k, v in regime_alloc.items()},
        }, f, indent=2)
    print(f"\n  Saved: {save_path}")


if __name__ == '__main__':
    main()
