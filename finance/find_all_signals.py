"""
Find ALL signals in the data. Not just alpha.

For every feature:
  1. Raw correlation with next-day return
  2. Quintile analysis (non-linear relationships)
  3. Predictive power when combined with alpha regime
  4. Lagged effects (does yesterday's value predict tomorrow?)

Then: rank everything, find the real signal stack.

Usage:
    python finance/find_all_signals.py
"""

import json
import sys
from math import sqrt
from pathlib import Path

import numpy as np
import torch

DATA_DIR = Path(__file__).parent / 'data'
RESULTS_DIR = Path(__file__).parent / 'training_results'


def quintile_analysis(feature, next_returns, n_quintiles=5):
    """Split feature into quintiles, measure return in each."""
    valid = ~np.isnan(feature) & ~np.isnan(next_returns)
    f = feature[valid]
    r = next_returns[valid]
    if len(f) < n_quintiles * 10:
        return None

    thresholds = np.percentile(f, np.linspace(0, 100, n_quintiles + 1))
    quintile_returns = []
    for i in range(n_quintiles):
        lo = thresholds[i]
        hi = thresholds[i + 1] if i < n_quintiles - 1 else float('inf')
        mask = (f >= lo) & (f < hi) if i < n_quintiles - 1 else (f >= lo)
        if mask.sum() > 5:
            quintile_returns.append({
                'quintile': i + 1,
                'lo': float(lo),
                'hi': float(hi),
                'count': int(mask.sum()),
                'mean_return': float(r[mask].mean()),
                'std_return': float(r[mask].std()),
                'win_rate': float((r[mask] > 0).mean()),
            })
    return quintile_returns


def main():
    data = torch.load(DATA_DIR / 'features_daily.pt', weights_only=False)
    features = data['features'].numpy()
    spy = data['spy'].numpy()
    timestamps = data['timestamps']
    feature_names = list(data['feature_names'])

    T = features.shape[0]
    returns = np.zeros(T)
    returns[1:] = (spy[1:] - spy[:-1]) / (spy[:-1] + 1e-8)

    next_returns = np.zeros(T)
    next_returns[:-1] = returns[1:]

    # Use first 774 days as training (pre-2025)
    train_end = 774
    for i, ts in enumerate(timestamps):
        if ts[:4] == '2025':
            train_end = i
            break

    feat_train = features[:train_end]
    nr_train = next_returns[:train_end]

    print("=" * 70)
    print("SIGNAL DISCOVERY — ALL 87 FEATURES")
    print(f"  Training data: {train_end} days (pre-2025)")
    print("=" * 70)

    # ===== PHASE 1: Raw correlations =====
    print("\n" + "-" * 70)
    print("PHASE 1: Correlation with next-day SPY return")
    print("-" * 70)

    correlations = []
    for i, fname in enumerate(feature_names):
        col = feat_train[:, i]
        valid = ~np.isnan(col)
        if valid.sum() < 50:
            continue
        c = np.corrcoef(col[valid], nr_train[valid])[0, 1]
        if np.isnan(c):
            continue
        correlations.append({
            'feature': fname,
            'idx': i,
            'corr': float(c),
            'abs_corr': abs(float(c)),
        })

    correlations.sort(key=lambda x: x['abs_corr'], reverse=True)

    print(f"\n  Top 30 features by |correlation| with next-day return:")
    print(f"  {'Rank':>4} {'Feature':<35} {'Corr':>8} {'Direction':>10}")
    print(f"  {'-' * 60}")
    for rank, c in enumerate(correlations[:30]):
        direction = 'BULLISH' if c['corr'] > 0 else 'BEARISH'
        bar = '#' * int(c['abs_corr'] * 500)
        print(f"  {rank+1:>4} {c['feature']:<35} {c['corr']:>+.4f}   {direction:<10} {bar}")

    # ===== PHASE 2: Quintile analysis (non-linear) =====
    print("\n" + "-" * 70)
    print("PHASE 2: Quintile analysis (non-linear relationships)")
    print("-" * 70)

    quintile_spreads = []
    for i, fname in enumerate(feature_names):
        col = feat_train[:, i]
        qa = quintile_analysis(col, nr_train)
        if qa is None or len(qa) < 5:
            continue

        q1_ret = qa[0]['mean_return']
        q5_ret = qa[-1]['mean_return']
        spread = q5_ret - q1_ret  # top quintile minus bottom quintile
        monotonic = all(qa[j]['mean_return'] <= qa[j+1]['mean_return']
                        for j in range(len(qa)-1)) or \
                    all(qa[j]['mean_return'] >= qa[j+1]['mean_return']
                        for j in range(len(qa)-1))

        quintile_spreads.append({
            'feature': fname,
            'idx': i,
            'q1_ret': q1_ret * 100,
            'q5_ret': q5_ret * 100,
            'spread': spread * 100,
            'abs_spread': abs(spread) * 100,
            'monotonic': monotonic,
            'quintiles': qa,
        })

    quintile_spreads.sort(key=lambda x: x['abs_spread'], reverse=True)

    print(f"\n  Top 25 features by Q5-Q1 return spread:")
    print(f"  {'Rank':>4} {'Feature':<35} {'Q1 ret':>8} {'Q5 ret':>8} "
          f"{'Spread':>8} {'Mono':>5}")
    print(f"  {'-' * 75}")
    for rank, q in enumerate(quintile_spreads[:25]):
        mono = 'YES' if q['monotonic'] else 'no'
        print(f"  {rank+1:>4} {q['feature']:<35} {q['q1_ret']:>+7.3f}% "
              f"{q['q5_ret']:>+7.3f}% {q['spread']:>+7.3f}% {mono:>5}")

    # Show detailed quintile breakdown for top 5
    print(f"\n  Detailed quintile breakdown for top 5 signals:")
    for q in quintile_spreads[:5]:
        print(f"\n  {q['feature']}:")
        for qr in q['quintiles']:
            bar = '+' * max(0, int((qr['mean_return'] + 0.002) * 5000))
            print(f"    Q{qr['quintile']}: ret={qr['mean_return']*100:>+.4f}%  "
                  f"WR={qr['win_rate']*100:.1f}%  n={qr['count']:>3}  {bar}")

    # ===== PHASE 3: Lagged effects =====
    print("\n" + "-" * 70)
    print("PHASE 3: Lagged effects (does t-1 predict t+1?)")
    print("-" * 70)

    lagged_signals = []
    for lag in [1, 2, 3, 5]:
        for i, fname in enumerate(feature_names):
            col = feat_train[:, i]
            if lag >= len(col):
                continue
            lagged = col[:-lag]
            future_ret = nr_train[lag:]
            valid = ~np.isnan(lagged) & ~np.isnan(future_ret)
            if valid.sum() < 50:
                continue
            c = np.corrcoef(lagged[valid], future_ret[valid])[0, 1]
            if np.isnan(c):
                continue
            lagged_signals.append({
                'feature': fname,
                'lag': lag,
                'corr': float(c),
                'abs_corr': abs(float(c)),
            })

    lagged_signals.sort(key=lambda x: x['abs_corr'], reverse=True)

    print(f"\n  Top 20 lagged signals:")
    print(f"  {'Rank':>4} {'Feature':<35} {'Lag':>4} {'Corr':>8}")
    print(f"  {'-' * 55}")
    for rank, ls in enumerate(lagged_signals[:20]):
        print(f"  {rank+1:>4} {ls['feature']:<35} t-{ls['lag']:<2} {ls['corr']:>+.4f}")

    # ===== PHASE 4: Conditional signals (within alpha regimes) =====
    print("\n" + "-" * 70)
    print("PHASE 4: Conditional signals (features that matter IN each alpha regime)")
    print("-" * 70)

    alpha_idx = feature_names.index('alpha')
    alpha_train = feat_train[:, alpha_idx]
    p40 = np.percentile(alpha_train, 40)
    p60 = np.percentile(alpha_train, 60)

    regimes = {
        'low_alpha (contrarian long)': alpha_train <= p40,
        'mid_alpha (neutral)': (alpha_train > p40) & (alpha_train <= p60),
        'high_alpha (crowded)': alpha_train > p60,
    }

    for regime_name, regime_mask in regimes.items():
        n_regime = regime_mask.sum()
        regime_feat = feat_train[regime_mask]
        regime_ret = nr_train[regime_mask]

        conditional_corrs = []
        for i, fname in enumerate(feature_names):
            if fname in ('alpha', 'regime_int', 'delta_alpha', 'alpha_accel',
                         'n_valid'):  # skip alpha-derived
                continue
            col = regime_feat[:, i]
            valid = ~np.isnan(col)
            if valid.sum() < 30:
                continue
            c = np.corrcoef(col[valid], regime_ret[valid])[0, 1]
            if np.isnan(c):
                continue
            conditional_corrs.append({
                'feature': fname,
                'corr': float(c),
                'abs_corr': abs(float(c)),
            })

        conditional_corrs.sort(key=lambda x: x['abs_corr'], reverse=True)

        print(f"\n  [{regime_name}] ({n_regime} days, avg ret: {regime_ret.mean()*100:+.4f}%)")
        print(f"  Top 10 features within this regime:")
        for rank, cc in enumerate(conditional_corrs[:10]):
            direction = '+' if cc['corr'] > 0 else '-'
            print(f"    {rank+1:>2}. {cc['feature']:<35} corr={cc['corr']:>+.4f}")

    # ===== PHASE 5: Feature interactions =====
    print("\n" + "-" * 70)
    print("PHASE 5: Feature interactions (pairs that predict together)")
    print("-" * 70)

    # Test top signals combined
    top_feats = [c['idx'] for c in correlations[:15]]
    top_names = [c['feature'] for c in correlations[:15]]

    interaction_scores = []
    for i in range(len(top_feats)):
        for j in range(i + 1, len(top_feats)):
            fi, fj = top_feats[i], top_feats[j]
            ni, nj = top_names[i], top_names[j]

            col_i = feat_train[:, fi]
            col_j = feat_train[:, fj]
            valid = ~np.isnan(col_i) & ~np.isnan(col_j)

            if valid.sum() < 100:
                continue

            med_i = np.median(col_i[valid])
            med_j = np.median(col_j[valid])

            # 4 quadrants
            q_hh = valid & (col_i > med_i) & (col_j > med_j)
            q_hl = valid & (col_i > med_i) & (col_j <= med_j)
            q_lh = valid & (col_i <= med_i) & (col_j > med_j)
            q_ll = valid & (col_i <= med_i) & (col_j <= med_j)

            quads = {}
            for qname, qmask in [('HH', q_hh), ('HL', q_hl), ('LH', q_lh), ('LL', q_ll)]:
                if qmask.sum() > 10:
                    quads[qname] = float(nr_train[qmask].mean())

            if len(quads) == 4:
                # Interaction = spread across quadrants
                max_ret = max(quads.values())
                min_ret = min(quads.values())
                interaction = max_ret - min_ret

                interaction_scores.append({
                    'feat_1': ni,
                    'feat_2': nj,
                    'interaction': float(interaction) * 100,
                    'quadrants': {k: round(v * 100, 4) for k, v in quads.items()},
                })

    interaction_scores.sort(key=lambda x: x['interaction'], reverse=True)

    print(f"\n  Top 10 feature pair interactions:")
    print(f"  {'Rank':>4} {'Feature 1':<25} {'Feature 2':<25} {'Spread':>8}")
    print(f"  {'-' * 70}")
    for rank, ix in enumerate(interaction_scores[:10]):
        print(f"  {rank+1:>4} {ix['feat_1']:<25} {ix['feat_2']:<25} "
              f"{ix['interaction']:>+7.3f}%")
        q = ix['quadrants']
        print(f"       HH={q['HH']:>+.4f}% HL={q['HL']:>+.4f}% "
              f"LH={q['LH']:>+.4f}% LL={q['LL']:>+.4f}%")

    # ===== SUMMARY: Ranked Signal Stack =====
    print("\n" + "=" * 70)
    print("SIGNAL STACK — RANKED BY STRENGTH")
    print("=" * 70)

    # Combine all signals into one ranked list
    all_signals = []

    for c in correlations[:30]:
        all_signals.append({
            'feature': c['feature'],
            'type': 'correlation',
            'strength': c['abs_corr'],
            'direction': '+' if c['corr'] > 0 else '-',
            'detail': f"corr={c['corr']:+.4f}",
        })

    for q in quintile_spreads[:15]:
        # Only add if not already in correlations list
        if not any(s['feature'] == q['feature'] and s['type'] == 'correlation'
                   for s in all_signals):
            all_signals.append({
                'feature': q['feature'],
                'type': 'quintile',
                'strength': q['abs_spread'] / 100,
                'direction': '+' if q['spread'] > 0 else '-',
                'detail': f"Q5-Q1={q['spread']:+.3f}%",
            })

    # Deduplicate and sort
    seen = set()
    unique_signals = []
    for s in sorted(all_signals, key=lambda x: x['strength'], reverse=True):
        if s['feature'] not in seen:
            seen.add(s['feature'])
            unique_signals.append(s)

    print(f"\n  {'Rank':>4} {'Feature':<35} {'Type':<12} {'Str':>6} {'Dir':>4} {'Detail':<20}")
    print(f"  {'-' * 85}")
    for rank, s in enumerate(unique_signals[:30]):
        print(f"  {rank+1:>4} {s['feature']:<35} {s['type']:<12} "
              f"{s['strength']:.4f} {s['direction']:>4} {s['detail']}")

    # Save
    save_path = RESULTS_DIR / 'signal_discovery.json'
    with open(save_path, 'w') as f:
        json.dump({
            'correlations': correlations[:30],
            'quintile_spreads': [{k: v for k, v in q.items() if k != 'quintiles'}
                                  for q in quintile_spreads[:25]],
            'lagged': lagged_signals[:20],
            'interactions': interaction_scores[:10],
            'signal_stack': unique_signals[:30],
        }, f, indent=2)
    print(f"\n  Saved: {save_path}")


if __name__ == '__main__':
    main()
