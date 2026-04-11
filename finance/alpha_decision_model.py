"""
Alpha Decision Model: alpha is the signal, everything else is the decision matrix.

Architecture:
  1. alpha classifies regime -> base direction (long/short/flat)
  2. Each feature group produces a confidence modifier
  3. Each group's influence is CAPPED by estimated market participation
  4. Position = direction * (base_confidence + weighted group modifiers)

Feature groups and their estimated market influence:
  - spectral (alpha, regime)   : 100% — this IS market structure
  - price (momentum, vol, RSI) :  80% — most traders watch charts
  - macro (VIX, yields)        :  70% — institutional capital
  - calendar (FOMC, OpEx, QE)  :  50% — institutional flow timing
  - seasonality (month, DoW)   :  20% — "sell in May" crowd
  - weather (temp, precip)     :   5% — trader mood, indirect
  - astro (moon, planets)      :   2% — small retail following

These weights are priors, not learned. They cap how much each group
can influence the final position, preventing overfitting on noise.

Usage:
    python finance/alpha_decision_model.py
    python finance/alpha_decision_model.py --optimize
    python finance/alpha_decision_model.py --sweep-weights
"""

import argparse
import json
import sys
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd
import torch

DATA_DIR = Path(__file__).parent / 'data'
RESULTS_DIR = Path(__file__).parent / 'training_results'
RESULTS_DIR.mkdir(exist_ok=True)

PHI = (1 + sqrt(5)) / 2

# --- Feature Group Definitions ---

FEATURE_GROUPS = {
    'spectral': {
        'influence': 1.00,
        'features': [
            'alpha', 'r2', 'mode1_pct', 'mode2_pct', 'mode3_pct',
            'effective_rank', 'frac_90', 'frac_95', 'frac_99',
            'delta_alpha', 'alpha_accel',
        ],
    },
    'price': {
        'influence': 0.80,
        'features': [
            'momentum_5', 'momentum_10', 'momentum_20', 'momentum_50',
            'volatility_10', 'volatility_20', 'volatility_50',
            'up_frac_14', 'up_frac_28',
            'dist_from_high_20', 'dist_from_low_20',
            'dist_from_high_50', 'dist_from_low_50',
            'volume_ratio_20',
            'ratio_SPY_TLT', 'ratio_SPY_GLD', 'ratio_XLK_XLF', 'ratio_HYG_TLT',
            'sector_dispersion', 'sector_dispersion_20',
        ],
    },
    'macro': {
        'influence': 0.70,
        'features': [
            'macro_^VIX', 'macro_^TNX', 'macro_^TYX', 'macro_^FVX', 'macro_^IRX',
        ],
    },
    'calendar': {
        'influence': 0.50,
        'features': [
            'astro_is_quarter_end', 'astro_is_opex', 'astro_is_fomc_week',
        ],
    },
    'seasonality': {
        'influence': 0.20,
        'features': [
            'astro_season_sin', 'astro_season_cos',
            'astro_dow_sin', 'astro_dow_cos',
            'astro_month_sin', 'astro_month_cos',
            'astro_day_of_year', 'astro_year_fraction',
            'astro_day_length_hrs',
        ],
    },
    'weather': {
        'influence': 0.02,
        'features': [
            'weather_temp_max_c', 'weather_temp_min_c', 'weather_temp_avg_c',
            'weather_precipitation_mm', 'weather_snowfall_mm',
            'weather_wind_speed_max', 'weather_wind_gust_max',
            'weather_sunshine_duration_s', 'weather_radiation_sum',
            'weather_pressure_mean', 'weather_humidity_mean',
            'weather_cloud_cover_mean',
        ],
    },
    'astro': {
        'influence': 0.02,
        'features': [
            'astro_moon_phase', 'astro_moon_illum', 'astro_sun_declination',
            'astro_mercury_longitude', 'astro_venus_longitude',
            'astro_mars_longitude', 'astro_jupiter_longitude',
            'astro_saturn_longitude',
            'astro_mercury_retrograde', 'astro_venus_retrograde',
        ],
    },
}


# --- Core Model ---

class AlphaDecisionModel:
    """Alpha is the signal. Everything else is the decision matrix."""

    def __init__(self, feature_names, influence_overrides=None):
        self.feature_names = feature_names
        self.groups = {}
        self.group_stats = {}  # learned conditional statistics per group

        # Map feature names to indices and assign to groups
        for group_name, group_def in FEATURE_GROUPS.items():
            indices = []
            matched_names = []
            for fname in group_def['features']:
                if fname in feature_names:
                    indices.append(feature_names.index(fname))
                    matched_names.append(fname)

            influence = group_def['influence']
            if influence_overrides and group_name in influence_overrides:
                influence = influence_overrides[group_name]

            self.groups[group_name] = {
                'indices': indices,
                'names': matched_names,
                'influence': influence,
            }

        # Alpha index for regime classification
        self.alpha_idx = feature_names.index('alpha') if 'alpha' in feature_names else None

        # Absolute regime indices (crisis/calm spectrum from spectral analysis)
        self.regime_one_hot_indices = {}
        for rname in ['DEEP_CALM', 'CALM', 'NORMAL', 'ELEVATED', 'STRESS', 'CRISIS']:
            fname = f'regime_{rname}'
            if fname in feature_names:
                self.regime_one_hot_indices[rname] = feature_names.index(fname)

        # Absolute regime boundaries — calibrated at fit() time
        # from the actual alpha distribution of the training data
        self.regime_boundaries = None  # set in fit()

    def fit(self, features, returns, verbose=True):
        """Learn conditional statistics: for each regime x group, what predicts sizing.

        For each feature group, within each alpha regime, compute:
          - correlation of each feature with next-day return
          - mean return when feature is above/below median
          - optimal feature threshold (simple split)

        This gives us: "when alpha says long AND this feature says X, expect Y"
        """
        n_samples = features.shape[0]
        alpha = features[:, self.alpha_idx]

        # Next-bar returns (what we're trying to capture)
        next_returns = np.zeros(n_samples)
        next_returns[:-1] = returns[1:]

        # --- Absolute regime calibration (crisis/calm spectrum) ---
        # Calibrate 6 regime buckets from the TRAINING alpha distribution
        # so each bucket has meaningful sample count at this timescale
        regime_pcts = [0, 10, 25, 50, 75, 90, 100]
        regime_names = ['DEEP_CALM', 'CALM', 'NORMAL', 'ELEVATED', 'STRESS', 'CRISIS']
        boundaries = []
        for i in range(len(regime_names)):
            lo = float(np.percentile(alpha, regime_pcts[i]))
            hi = float(np.percentile(alpha, regime_pcts[i + 1]))
            if i == len(regime_names) - 1:
                hi = float('inf')
            boundaries.append((lo, hi))
        self.regime_boundaries = dict(zip(regime_names, boundaries))

        if verbose:
            print("\n  Absolute regime spectrum (calibrated to data):")
        self.abs_regime_returns = {}
        for rname, (lo, hi) in self.regime_boundaries.items():
            mask = (alpha >= lo) & (alpha < hi)
            n = mask.sum()
            if n > 5:
                avg_ret = next_returns[mask].mean()
                std_ret = next_returns[mask].std()
                self.abs_regime_returns[rname] = {
                    'count': int(n), 'mean_return': float(avg_ret),
                    'std_return': float(std_ret),
                }
                if verbose:
                    print(f"    {rname:>12} [{lo:.3f}-{hi:.3f}]: {n:>5} days ({n/n_samples*100:.1f}%)  "
                          f"avg ret: {avg_ret*100:+.4f}%  std: {std_ret*100:.4f}%")
            else:
                if verbose:
                    print(f"    {rname:>12} [{lo:.3f}-{hi:.3f}]: {n:>5} days ({n/n_samples*100:.1f}%)")

        # Adaptive regime classification using percentiles
        # Absolute thresholds don't work at daily (alpha always > 1.38)
        # Instead: where is alpha relative to its own recent history?
        self.alpha_percentiles = {
            'p20': float(np.percentile(alpha, 20)),
            'p40': float(np.percentile(alpha, 40)),
            'p60': float(np.percentile(alpha, 60)),
            'p80': float(np.percentile(alpha, 80)),
        }

        # Also compute delta_alpha percentiles for rate-of-change signal
        da_idx = self.feature_names.index('delta_alpha') if 'delta_alpha' in self.feature_names else None
        if da_idx is not None:
            delta_alpha = features[:, da_idx]
            self.da_percentiles = {
                'p25': float(np.percentile(delta_alpha, 25)),
                'p75': float(np.percentile(delta_alpha, 75)),
            }
        else:
            delta_alpha = np.zeros(n_samples)
            self.da_percentiles = {'p25': 0, 'p75': 0}

        p20 = self.alpha_percentiles['p20']
        p40 = self.alpha_percentiles['p40']
        p60 = self.alpha_percentiles['p60']
        p80 = self.alpha_percentiles['p80']
        da_p25 = self.da_percentiles['p25']
        da_p75 = self.da_percentiles['p75']

        # Regimes based on alpha percentile + rate of change
        # Mean reversion: high alpha = overextended, low alpha = opportunity
        regimes = {
            'overextended':    (alpha > p80) & (delta_alpha > da_p75),  # crowded + rising -> short
            'crowded':         (alpha > p60) & (delta_alpha >= 0),       # high + stable -> flat
            'mild_long':       (alpha > p40) & (alpha <= p60),           # middle -> mild long
            'contrarian_long': (alpha <= p40) | (delta_alpha < da_p25),  # weakening -> contrarian long
        }
        # Make mutually exclusive (priority order)
        used = np.zeros(n_samples, dtype=bool)
        for rname in ['overextended', 'crowded', 'contrarian_long', 'mild_long']:
            regimes[rname] = regimes[rname] & ~used
            used = used | regimes[rname]

        if verbose:
            print("\n  Regime distribution:")
            for rname, mask in regimes.items():
                n = mask.sum()
                avg_ret = next_returns[mask].mean() * 100 if n > 10 else 0
                print(f"    {rname:>12}: {n:>5} days ({n/n_samples*100:.1f}%)  "
                      f"avg next-day return: {avg_ret:+.4f}%")

        self.regime_stats = {}
        for rname, mask in regimes.items():
            if mask.sum() < 20:
                continue
            regime_returns = next_returns[mask]
            self.regime_stats[rname] = {
                'count': int(mask.sum()),
                'mean_return': float(regime_returns.mean()),
                'std_return': float(regime_returns.std()),
                'win_rate': float((regime_returns > 0).mean()),
            }

        # For each group, learn which features matter within each regime
        self.group_models = {}

        for group_name, group_info in self.groups.items():
            if not group_info['indices']:
                continue

            group_features = features[:, group_info['indices']]
            group_model = {'regimes': {}}

            for rname, mask in regimes.items():
                if mask.sum() < 30:
                    continue

                regime_feat = group_features[mask]
                regime_ret = next_returns[mask]

                # For each feature in this group, find the best split
                feature_scores = []
                for fi, fname in enumerate(group_info['names']):
                    col = regime_feat[:, fi]

                    # Skip if constant
                    if col.std() < 1e-10:
                        feature_scores.append({
                            'name': fname, 'corr': 0, 'split_edge': 0,
                            'above_mean': 0, 'below_mean': 0,
                        })
                        continue

                    # Correlation with returns
                    corr = np.corrcoef(col, regime_ret)[0, 1]
                    if np.isnan(corr):
                        corr = 0

                    # Median split: does above-median beat below-median?
                    median = np.median(col)
                    above = regime_ret[col > median]
                    below = regime_ret[col <= median]

                    above_mean = above.mean() if len(above) > 5 else 0
                    below_mean = below.mean() if len(below) > 5 else 0
                    split_edge = above_mean - below_mean

                    feature_scores.append({
                        'name': fname,
                        'corr': float(corr),
                        'split_edge': float(split_edge),
                        'above_mean': float(above_mean),
                        'below_mean': float(below_mean),
                        'median': float(median),
                    })

                group_model['regimes'][rname] = feature_scores

            self.group_models[group_name] = group_model

            if verbose:
                influence = group_info['influence']
                print(f"\n  {group_name} (influence cap: {influence:.0%}, "
                      f"{len(group_info['indices'])} features):")

                for rname in ['calm_long', 'normal']:
                    if rname not in group_model['regimes']:
                        continue
                    scores = group_model['regimes'][rname]
                    # Sort by absolute correlation
                    scores_sorted = sorted(scores, key=lambda s: abs(s['corr']), reverse=True)
                    top = scores_sorted[:3]
                    if top and any(abs(s['corr']) > 0.01 for s in top):
                        print(f"    [{rname}] top features:")
                        for s in top:
                            if abs(s['corr']) > 0.01:
                                direction = '+' if s['corr'] > 0 else '-'
                                print(f"      {s['name']:<30} corr={s['corr']:+.3f}  "
                                      f"split_edge={s['split_edge']*100:+.4f}%")

    def predict_position(self, features_row):
        """Given a single timestep's features, output position in [-1, 1].

        1. Alpha percentile + delta_alpha classify regime -> direction + base confidence
        2. Each group contributes a modifier, capped by influence weight
        3. Final position = direction * clamp(total_confidence, 0, 1)
        """
        alpha = features_row[self.alpha_idx]

        # Get delta_alpha
        da_idx = self.feature_names.index('delta_alpha') if 'delta_alpha' in self.feature_names else None
        delta_alpha = features_row[da_idx] if da_idx is not None else 0

        p20 = self.alpha_percentiles['p20']
        p40 = self.alpha_percentiles['p40']
        p60 = self.alpha_percentiles['p60']
        p80 = self.alpha_percentiles['p80']
        da_p25 = self.da_percentiles['p25']
        da_p75 = self.da_percentiles['p75']

        # Step 1: Percentile-based regime -> direction and base confidence
        # KEY INSIGHT: high alpha + rising = crowded/overextended -> mean reversion DOWN
        #              low/falling alpha = structure breaking -> mean reversion UP
        if alpha > p80 and delta_alpha > da_p75:
            direction = -1.0
            base_confidence = 0.6  # overextended: go short / reduce
            regime = 'overextended'
        elif alpha > p60 and delta_alpha >= 0:
            direction = 0.0
            base_confidence = 0.0  # high but stable: flat
            regime = 'crowded'
        elif alpha <= p40 or delta_alpha < da_p25:
            direction = 1.0
            base_confidence = 0.6  # structure weakening: contrarian long
            regime = 'contrarian_long'
        else:
            direction = 1.0
            base_confidence = 0.3  # middle band: mild long (market drifts up)
            regime = 'mild_long'

        # Step 1b: Absolute regime (crisis/calm) -> risk scaler
        # Determined by where alpha falls in the calibrated distribution
        abs_regime = 'NORMAL'
        if self.regime_boundaries is not None:
            for rname, (lo, hi) in self.regime_boundaries.items():
                if lo <= alpha < hi:
                    abs_regime = rname
                    break

        # Risk budget derived from training data:
        # Use the return/risk ratio of each absolute regime to set the scaler
        # Better regime stats -> more risk budget
        if hasattr(self, 'abs_regime_returns') and abs_regime in self.abs_regime_returns:
            stats = self.abs_regime_returns[abs_regime]
            # Scale by return/vol ratio (information ratio)
            if stats['std_return'] > 1e-8:
                ir = stats['mean_return'] / stats['std_return']
                # Map IR to [0.2, 1.0] risk scaler
                # IR of 0.1 -> 1.0 (great regime), IR of -0.1 -> 0.2 (bad regime)
                risk_scaler = float(np.clip(0.6 + ir * 4.0, 0.2, 1.0))
            else:
                risk_scaler = 0.5
        else:
            # Fallback: static scalers
            risk_scalers = {
                'DEEP_CALM': 0.6, 'CALM': 0.8, 'NORMAL': 1.0,
                'ELEVATED': 0.7, 'STRESS': 0.5, 'CRISIS': 0.3,
            }
            risk_scaler = risk_scalers.get(abs_regime, 0.5)

        if direction == 0:
            return 0.0, {'regime': regime, 'abs_regime': abs_regime,
                         'alpha': float(alpha),
                         'delta_alpha': float(delta_alpha),
                         'risk_scaler': risk_scaler,
                         'base': 0, 'modifiers': {}}

        # Step 2: Each group contributes a confidence modifier
        modifiers = {}
        total_modifier = 0.0

        for group_name, group_info in self.groups.items():
            if not group_info['indices']:
                continue
            if group_name not in self.group_models:
                continue
            if regime not in self.group_models[group_name]['regimes']:
                continue

            scores = self.group_models[group_name]['regimes'][regime]
            if not scores:
                continue

            # Compute this group's signal: weighted average of feature signals
            group_signal = 0.0
            n_active = 0

            for fi, score in enumerate(scores):
                if abs(score['corr']) < 0.02:  # ignore noise
                    continue
                idx = group_info['indices'][fi]
                val = features_row[idx]
                median = score['median']

                # Feature says "above median" (+1) or "below median" (-1)
                # Weighted by how predictive this feature is (correlation)
                feat_signal = np.sign(val - median) * score['corr']
                group_signal += feat_signal
                n_active += 1

            if n_active > 0:
                group_signal /= n_active  # normalize by number of features
                # Clip to [-1, 1] then scale by influence cap
                group_signal = np.clip(group_signal, -1, 1)
                capped = group_signal * group_info['influence']
                modifiers[group_name] = float(capped)
                total_modifier += capped

        # Step 3: Final position = direction * confidence * risk_scaler
        confidence = np.clip(base_confidence + total_modifier, 0.05, 1.0)
        position = direction * confidence * risk_scaler

        detail = {
            'regime': regime,
            'abs_regime': abs_regime,
            'alpha': float(alpha),
            'delta_alpha': float(delta_alpha),
            'direction': direction,
            'base_confidence': base_confidence,
            'risk_scaler': risk_scaler,
            'modifiers': modifiers,
            'total_modifier': total_modifier,
            'final_confidence': confidence,
        }

        return float(position), detail

    def backtest(self, features, returns, spy_prices, timestamps=None,
                 cost_bps=5.0, verbose=True):
        """Run full backtest."""
        n = features.shape[0]
        cost_frac = cost_bps / 10000.0

        positions = np.zeros(n)
        details = []

        for t in range(n):
            pos, detail = self.predict_position(features[t])
            positions[t] = pos
            details.append(detail)

        # PnL calculation
        next_returns = np.zeros(n)
        next_returns[:-1] = returns[1:]

        bar_pnl = positions[:-1] * next_returns[:-1]
        costs = np.abs(np.diff(positions)) * cost_frac
        bar_pnl_net = bar_pnl - costs

        cum_pnl = np.cumsum(bar_pnl_net)
        total_pnl = cum_pnl[-1] * 100

        # Sharpe
        mean_pnl = bar_pnl_net.mean()
        std_pnl = bar_pnl_net.std() + 1e-8
        sharpe = mean_pnl / std_pnl * sqrt(252)

        # Max drawdown
        running_max = np.maximum.accumulate(cum_pnl)
        drawdowns = running_max - cum_pnl
        max_dd = drawdowns.max() * 100

        # Win rate
        active_bars = bar_pnl_net[positions[:-1] != 0]
        win_rate = (active_bars > 0).mean() * 100 if len(active_bars) > 0 else 0

        # Position stats
        mean_pos = np.abs(positions).mean()
        long_pct = (positions > 0.01).mean() * 100
        short_pct = (positions < -0.01).mean() * 100
        flat_pct = 100 - long_pct - short_pct

        # Group contribution analysis
        group_contribs = {}
        for d in details:
            for gname, gmod in d['modifiers'].items():
                if gname not in group_contribs:
                    group_contribs[gname] = []
                group_contribs[gname].append(gmod)

        results = {
            'total_pnl': total_pnl,
            'sharpe': sharpe,
            'max_drawdown': max_dd,
            'win_rate': win_rate,
            'mean_position': mean_pos,
            'long_pct': long_pct,
            'short_pct': short_pct,
            'flat_pct': flat_pct,
            'total_costs': costs.sum() * 100,
            'n_bars': n,
        }

        if verbose:
            print(f"    PnL: {total_pnl:+.2f}%  Sharpe: {sharpe:.2f}  "
                  f"MaxDD: {max_dd:.2f}%  WinRate: {win_rate:.1f}%")
            print(f"    Positions: long {long_pct:.0f}% / flat {flat_pct:.0f}% / "
                  f"short {short_pct:.0f}%  avg|pos|: {mean_pos:.2f}")
            print(f"    Costs: {costs.sum()*100:.3f}%")

            if group_contribs:
                print(f"    Group contributions (avg modifier):")
                for gname, vals in sorted(group_contribs.items(),
                                          key=lambda x: abs(np.mean(x[1])), reverse=True):
                    avg = np.mean(vals)
                    influence = self.groups[gname]['influence']
                    print(f"      {gname:<15} avg={avg:+.4f}  "
                          f"(cap={influence:.0%})")

        return results, positions, cum_pnl


# --- Buy and Hold ---

def buyhold_pnl(spy_prices):
    return (spy_prices[-1] - spy_prices[0]) / spy_prices[0] * 100


# --- Main ---

def main():
    parser = argparse.ArgumentParser(description='Alpha Decision Model')
    parser.add_argument('--timescale', default='daily', choices=['daily', 'hourly', '5min'])
    parser.add_argument('--optimize', action='store_true',
                        help='Grid search over influence weights')
    parser.add_argument('--sweep-weights', action='store_true',
                        help='Sweep each group weight independently')
    args = parser.parse_args()

    print("=" * 70)
    print("ALPHA DECISION MODEL")
    print("  Alpha = signal, everything else = decision matrix")
    print("  Feature group influence capped by estimated market participation")
    print("=" * 70)

    # Load data
    path = DATA_DIR / f'features_{args.timescale}.pt'
    if not path.exists():
        print(f"ERROR: {path} not found")
        sys.exit(1)

    data = torch.load(path, weights_only=False)
    features = data['features'].numpy()
    spy = data['spy'].numpy()
    timestamps = data['timestamps']
    feature_names = data['feature_names']

    # Compute returns
    returns = np.zeros(len(spy))
    returns[1:] = (spy[1:] - spy[:-1]) / (spy[:-1] + 1e-8)

    T = features.shape[0]
    print(f"\n  Data: {T} bars x {features.shape[1]} features ({args.timescale})")

    # Walk-forward split
    val_start = '2025-01-01'
    test_start = '2025-07-01'

    val_idx = T
    test_idx = T
    for i, ts in enumerate(timestamps):
        if val_start in ts and val_idx == T:
            val_idx = i
        if test_start in ts and test_idx == T:
            test_idx = i

    if val_idx == T:
        val_idx = int(T * 0.7)
        test_idx = int(T * 0.85)

    print(f"  Split: train={val_idx} val={test_idx-val_idx} test={T-test_idx}")

    # Baselines
    print(f"\n  Buy and Hold:")
    print(f"    Train: {buyhold_pnl(spy[:val_idx]):+.2f}%")
    print(f"    Val:   {buyhold_pnl(spy[val_idx:test_idx]):+.2f}%")
    print(f"    Test:  {buyhold_pnl(spy[test_idx:]):+.2f}%")

    # --- Fit model on training data ---
    print("\n" + "=" * 70)
    print("FITTING on training data...")
    print("=" * 70)

    model = AlphaDecisionModel(feature_names)
    model.fit(features[:val_idx], returns[:val_idx], verbose=True)

    # --- Backtest all splits ---
    print("\n" + "=" * 70)
    print("BACKTEST RESULTS")
    print("=" * 70)

    all_results = {}
    for split_name, start, end in [('Train', 0, val_idx),
                                     ('Val', val_idx, test_idx),
                                     ('Test', test_idx, T)]:
        print(f"\n  [{split_name}] ({end - start} bars)")
        r, pos, cum = model.backtest(
            features[start:end], returns[start:end], spy[start:end],
            timestamps[start:end], verbose=True)
        all_results[split_name.lower()] = r

        bh = buyhold_pnl(spy[start:end])
        edge = r['total_pnl'] - bh
        print(f"    vs B&H ({bh:+.2f}%): edge = {edge:+.2f}%")

    # --- Weight sweep ---
    if args.sweep_weights:
        print("\n" + "=" * 70)
        print("INFLUENCE WEIGHT SWEEP")
        print("=" * 70)
        print("  Sweeping each group's weight while holding others fixed...")

        for group_name in FEATURE_GROUPS:
            if group_name == 'spectral':
                continue  # always 100%

            best_weight = FEATURE_GROUPS[group_name]['influence']
            best_val_sharpe = -999

            for weight in [0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 1.0]:
                overrides = {group_name: weight}
                m = AlphaDecisionModel(feature_names, influence_overrides=overrides)
                m.fit(features[:val_idx], returns[:val_idx], verbose=False)
                r, _, _ = m.backtest(features[val_idx:test_idx], returns[val_idx:test_idx],
                                     spy[val_idx:test_idx], verbose=False)
                if r['sharpe'] > best_val_sharpe:
                    best_val_sharpe = r['sharpe']
                    best_weight = weight

            print(f"  {group_name:<15} best weight: {best_weight:.2f}  "
                  f"val Sharpe: {best_val_sharpe:.2f}")

    # --- Full optimization ---
    if args.optimize:
        print("\n" + "=" * 70)
        print("GRID SEARCH OPTIMIZATION")
        print("=" * 70)

        # Coarse grid over key groups
        best_sharpe = -999
        best_config = {}

        price_weights = [0.3, 0.5, 0.8, 1.0]
        macro_weights = [0.3, 0.5, 0.7, 1.0]
        calendar_weights = [0.0, 0.2, 0.5]
        season_weights = [0.0, 0.1, 0.2]
        weather_weights = [0.0, 0.02, 0.05]
        astro_weights = [0.0, 0.01, 0.02]

        total_combos = (len(price_weights) * len(macro_weights) *
                        len(calendar_weights) * len(season_weights) *
                        len(weather_weights) * len(astro_weights))
        print(f"  {total_combos} combinations...")

        count = 0
        for pw in price_weights:
            for mw in macro_weights:
                for cw in calendar_weights:
                    for sw in season_weights:
                        for ww in weather_weights:
                            for aw in astro_weights:
                                overrides = {
                                    'price': pw, 'macro': mw,
                                    'calendar': cw, 'seasonality': sw,
                                    'weather': ww, 'astro': aw,
                                }
                                m = AlphaDecisionModel(feature_names,
                                                       influence_overrides=overrides)
                                m.fit(features[:val_idx], returns[:val_idx],
                                      verbose=False)
                                r, _, _ = m.backtest(
                                    features[val_idx:test_idx],
                                    returns[val_idx:test_idx],
                                    spy[val_idx:test_idx], verbose=False)

                                if r['sharpe'] > best_sharpe:
                                    best_sharpe = r['sharpe']
                                    best_config = overrides.copy()
                                    best_val_r = r

                                count += 1
                                if count % 500 == 0:
                                    print(f"    {count}/{total_combos}  "
                                          f"best val Sharpe: {best_sharpe:.2f}")

        print(f"\n  Best config (val Sharpe={best_sharpe:.2f}):")
        for k, v in sorted(best_config.items()):
            print(f"    {k:<15} = {v:.2f}")

        # Test with best config
        print(f"\n  Test with optimized weights:")
        m = AlphaDecisionModel(feature_names, influence_overrides=best_config)
        m.fit(features[:val_idx], returns[:val_idx], verbose=False)
        test_r, _, _ = m.backtest(features[test_idx:], returns[test_idx:],
                                   spy[test_idx:], verbose=True)
        bh = buyhold_pnl(spy[test_idx:])
        print(f"    vs B&H ({bh:+.2f}%): edge = {test_r['total_pnl'] - bh:+.2f}%")

    # Save results
    save_path = RESULTS_DIR / f'alpha_decision_{args.timescale}.json'
    with open(save_path, 'w') as f:
        json.dump({
            'results': all_results,
            'influence_weights': {g: d['influence'] for g, d in FEATURE_GROUPS.items()},
            'regime_stats': model.regime_stats if hasattr(model, 'regime_stats') else {},
        }, f, indent=2)
    print(f"\n  Saved: {save_path}")


if __name__ == '__main__':
    main()
