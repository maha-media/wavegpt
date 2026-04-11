"""
Hybrid Spectral Trader: known rules + learned refinement.

The grid search found rules that work (+2.83% edge):
  - alpha > 1.38 -> go long (spectral calm = market rising)
  - trailing stop at 0.10% from recent high
  - momentum reversal (alpha falling + price up -> expect reversal)

This model doesn't rediscover those rules. Instead:
  1. Rule signals are computed and fed as privileged features
  2. A small NN learns: given the rule says X and current context,
     what's the optimal position size and when should we override?

Architecture:
  [rule_signal, context_features] -> MLP -> position_scale in [-1, 1]
  Final position = rule_combined * learned_scale * confidence

Daily timescale: 1,020 bars, 87 features. Small model (~5K params).

Usage:
    python finance/train_hybrid_trader.py
    python finance/train_hybrid_trader.py --timescale daily --seeds 50
    python finance/train_hybrid_trader.py --timescale hourly --seeds 30
"""

import argparse
import json
import sys
import time
from pathlib import Path
from math import sqrt

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DATA_DIR = Path(__file__).parent / 'data'
RESULTS_DIR = Path(__file__).parent / 'training_results'
RESULTS_DIR.mkdir(exist_ok=True)

PHI = (1 + sqrt(5)) / 2

# Rule signal feature names (must match build_features.py output)
RULE_FEATURES = [
    'rule_long_signal', 'rule_short_signal', 'rule_base_position',
    'rule_momentum_reversal', 'rule_trailing_stop', 'rule_combined',
]
REGIME_FEATURES = [
    'regime_DEEP_CALM', 'regime_CALM', 'regime_NORMAL',
    'regime_ELEVATED', 'regime_STRESS', 'regime_CRISIS',
]


# --- Model -------------------------------------------------------

class HybridTrader(nn.Module):
    """NN that refines known-good rule signals using market context.

    Two pathways:
      1. Rule pathway: rule signals -> trust gate (how much to follow the rule)
      2. Context pathway: all features (windowed) -> context embedding
      Combined -> position scale + confidence

    The rule signal provides a strong prior; the NN learns adjustments.
    """

    def __init__(self, n_features, n_rule_features, window=5, hidden=32):
        super().__init__()
        self.window = window

        # Context encoder: takes windowed features -> compact embedding
        context_in = n_features * window
        self.context_net = nn.Sequential(
            nn.LayerNorm(context_in),
            nn.Linear(context_in, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )

        # Rule trust gate: learns when to follow/override the rule
        rule_in = n_rule_features + len(REGIME_FEATURES)
        self.trust_gate = nn.Sequential(
            nn.Linear(rule_in, 16),
            nn.GELU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),  # 0 = ignore rule, 1 = fully trust rule
        )

        # Combination head: context + rule -> position adjustment
        self.position_head = nn.Sequential(
            nn.Linear(hidden + 1 + n_rule_features, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, 2),  # [position_adjust, confidence]
        )

    def forward(self, features, rule_indices, regime_indices):
        """
        Args:
            features: [B, T, D] all features
            rule_indices: list of int, indices of rule features in D
            regime_indices: list of int, indices of regime features in D

        Returns:
            positions: [B, T'] final positions in [-1, 1]
            raw_positions: [B, T'] before confidence scaling
            confidence: [B, T'] in [0, 1]
            trust: [B, T'] how much the model trusts the rule
        """
        B, T, D = features.shape
        w = self.window

        # Create windowed context: [B, T-w+1, D*w]
        windows = []
        for t in range(w - 1, T):
            windows.append(features[:, t - w + 1:t + 1, :].reshape(B, -1))
        context_in = torch.stack(windows, dim=1)  # [B, T', D*w]
        T_out = context_in.shape[1]

        # Context embedding
        context = self.context_net(context_in)  # [B, T', hidden]

        # Rule signals for the output timesteps
        rule_feats = features[:, w - 1:, :][:, :, rule_indices]  # [B, T', n_rule]
        regime_feats = features[:, w - 1:, :][:, :, regime_indices]  # [B, T', n_regime]

        # Trust gate: how much to follow the rule
        trust_in = torch.cat([rule_feats, regime_feats], dim=-1)
        trust = self.trust_gate(trust_in).squeeze(-1)  # [B, T']

        # Get the base rule position
        rule_combined_idx = rule_indices[-1]  # rule_combined is last
        rule_position = features[:, w - 1:, rule_combined_idx]  # [B, T']

        # Position head: combine context + trust + rule signals
        combo = torch.cat([context, trust.unsqueeze(-1), rule_feats], dim=-1)
        head_out = self.position_head(combo)  # [B, T', 2]

        position_adjust = torch.tanh(head_out[:, :, 0])  # [-1, 1]
        confidence = torch.sigmoid(head_out[:, :, 1])  # [0, 1]

        # Final position: blend rule signal with learned adjustment
        # trust=1 -> follow rule; trust=0 -> follow NN's own signal
        raw_positions = trust * rule_position + (1 - trust) * position_adjust
        positions = raw_positions * confidence

        return positions, raw_positions, confidence, trust


# --- Differentiable Trading Simulator -----------------------------

class DifferentiableTradingSim(nn.Module):
    """Simulates PnL from positions, fully differentiable."""

    def __init__(self, cost_bps=5.0, bars_per_day=1):
        super().__init__()
        self.cost_bps = cost_bps / 10000.0
        self.bars_per_day = bars_per_day

    def forward(self, positions, returns):
        # PnL: position at time t earns return at time t+1
        pos_shifted = positions[:, :-1]
        rets = returns[:, 1:]

        bar_pnl = pos_shifted * rets

        # Transaction costs
        position_changes = torch.abs(positions[:, 1:] - positions[:, :-1])
        costs = position_changes * self.cost_bps
        bar_pnl_net = bar_pnl - costs

        cum_pnl = torch.cumsum(bar_pnl_net, dim=1)

        # Sharpe ratio (annualized)
        mean_pnl = bar_pnl_net.mean(dim=1)
        std_pnl = bar_pnl_net.std(dim=1) + 1e-8
        annualization = sqrt(252 * self.bars_per_day)
        sharpe = mean_pnl / std_pnl * annualization

        # Max drawdown
        running_max = torch.cummax(cum_pnl, dim=1)[0]
        drawdowns = running_max - cum_pnl
        max_drawdown = drawdowns.max(dim=1)[0]

        trade_intensity = position_changes.mean(dim=1)
        mean_abs_position = positions.abs().mean(dim=1)
        total_pnl = cum_pnl[:, -1]

        with torch.no_grad():
            winning_bars = (bar_pnl_net > 0).float().mean(dim=1)

        return {
            'total_pnl': total_pnl,
            'sharpe': sharpe,
            'max_drawdown': max_drawdown,
            'trade_intensity': trade_intensity,
            'mean_abs_position': mean_abs_position,
            'cum_pnl': cum_pnl,
            'bar_pnl': bar_pnl_net,
            'win_rate': winning_bars,
            'total_costs': costs.sum(dim=1),
        }


def trading_loss(sim_results, lambda_dd=0.5, lambda_activity=0.01,
                 lambda_utilization=1.0):
    """Loss: maximize PnL, penalize drawdown and idleness."""
    loss = -sim_results['total_pnl'] * 100

    loss = loss + lambda_dd * sim_results['max_drawdown'] * 100

    utilization = sim_results['mean_abs_position']
    util_penalty = F.relu(0.3 - utilization)
    loss = loss + lambda_utilization * util_penalty * 10

    loss = loss + lambda_activity * sim_results['trade_intensity']

    return loss.mean()


# --- Data Loading -------------------------------------------------

def load_data(timescale='daily', device='cuda'):
    """Load feature tensors, find rule/regime indices, split data."""
    path = DATA_DIR / f'features_{timescale}.pt'
    if not path.exists():
        print(f"ERROR: {path} not found. Run build_features.py first.")
        sys.exit(1)

    data = torch.load(path, weights_only=False)
    features = data['features'].to(device)
    spy = data['spy'].to(device)
    timestamps = data['timestamps']
    feature_names = data['feature_names']

    # Find rule and regime feature indices
    rule_indices = []
    for rf in RULE_FEATURES:
        if rf in feature_names:
            rule_indices.append(feature_names.index(rf))
    regime_indices = []
    for rf in REGIME_FEATURES:
        if rf in feature_names:
            regime_indices.append(feature_names.index(rf))

    if not rule_indices:
        print("ERROR: No rule features found. Rebuild features with latest build_features.py.")
        sys.exit(1)

    print(f"  Found {len(rule_indices)} rule features, {len(regime_indices)} regime features")

    # Compute returns from SPY prices
    returns = torch.zeros_like(spy)
    returns[1:] = (spy[1:] - spy[:-1]) / (spy[:-1] + 1e-8)

    T = features.shape[0]
    print(f"  Loaded {timescale}: {T} bars x {features.shape[1]} features")

    # Walk-forward split
    val_start_str = '2025-01-01'
    test_start_str = '2025-07-01'

    val_idx = T
    test_idx = T
    for i, ts in enumerate(timestamps):
        if val_start_str in ts and val_idx == T:
            val_idx = i
        if test_start_str in ts and test_idx == T:
            test_idx = i

    if val_idx == T:
        val_idx = int(T * 0.7)
        test_idx = int(T * 0.85)

    print(f"  Split: train={val_idx} val={test_idx - val_idx} test={T - test_idx}")
    print(f"  Train: {timestamps[0][:10]} to {timestamps[val_idx-1][:10]}")
    print(f"  Val:   {timestamps[val_idx][:10]} to {timestamps[test_idx-1][:10]}")
    print(f"  Test:  {timestamps[test_idx][:10]} to {timestamps[-1][:10]}")

    return {
        'features': features,
        'returns': returns,
        'spy': spy,
        'timestamps': timestamps,
        'feature_names': feature_names,
        'rule_indices': rule_indices,
        'regime_indices': regime_indices,
        'train_end': val_idx,
        'val_end': test_idx,
    }


def make_sequences(features, returns, start, end, seq_len, stride=1):
    """Create overlapping sequences."""
    feat_list, ret_list = [], []
    for i in range(start, end - seq_len, stride):
        feat_list.append(features[i:i + seq_len])
        ret_list.append(returns[i:i + seq_len])

    if not feat_list:
        return None, None

    return torch.stack(feat_list), torch.stack(ret_list)


# --- Training -----------------------------------------------------

def train_one_seed(data, seed, args, device='cuda'):
    """Train a single hybrid model."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    n_features = data['features'].shape[1]
    n_rule = len(data['rule_indices'])
    seq_len = args.seq_len
    bars_per_day = {'5min': 78, 'hourly': 7, 'daily': 1}[args.timescale]

    model = HybridTrader(
        n_features=n_features,
        n_rule_features=n_rule,
        window=args.window,
        hidden=args.hidden,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())

    sim = DifferentiableTradingSim(cost_bps=args.cost_bps, bars_per_day=bars_per_day).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    rule_idx = data['rule_indices']
    regime_idx = data['regime_indices']

    # Make training sequences
    train_feat, train_ret = make_sequences(
        data['features'], data['returns'],
        0, data['train_end'], seq_len, stride=args.stride
    )
    val_feat, val_ret = make_sequences(
        data['features'], data['returns'],
        data['train_end'], data['val_end'], seq_len, stride=max(1, seq_len // 4)
    )

    if train_feat is None:
        print(f"  Seed {seed}: not enough data")
        return None

    n_train = train_feat.shape[0]

    best_val_sharpe = -float('inf')
    best_state = None
    patience_counter = 0

    for epoch in range(args.epochs):
        model.train()

        # Random batch
        if n_train > args.batch_size:
            idx = torch.randperm(n_train, device=device)[:args.batch_size]
            batch_feat = train_feat[idx]
            batch_ret = train_ret[idx]
        else:
            batch_feat = train_feat
            batch_ret = train_ret

        positions, raw_pos, confidence, trust = model(batch_feat, rule_idx, regime_idx)
        T_out = positions.shape[1]
        ret_aligned = batch_ret[:, -T_out:]

        sim_results = sim(positions, ret_aligned)
        loss = trading_loss(sim_results,
                          lambda_dd=args.lambda_dd,
                          lambda_activity=args.lambda_activity,
                          lambda_utilization=args.lambda_utilization)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # Validation
        if epoch % args.eval_every == 0 or epoch == args.epochs - 1:
            model.eval()
            with torch.no_grad():
                # Train metrics
                t_pos, _, _, t_trust = model(train_feat, rule_idx, regime_idx)
                T_out_t = t_pos.shape[1]
                t_sim = sim(t_pos, train_ret[:, -T_out_t:])
                train_sharpe = t_sim['sharpe'].mean().item()
                train_pnl = t_sim['total_pnl'].mean().item() * 100

                # Val metrics
                val_sharpe = 0
                val_pnl = 0
                v_trust_mean = 0
                if val_feat is not None:
                    v_pos, _, _, v_trust = model(val_feat, rule_idx, regime_idx)
                    T_out_v = v_pos.shape[1]
                    v_sim = sim(v_pos, val_ret[:, -T_out_v:])
                    val_sharpe = v_sim['sharpe'].mean().item()
                    val_pnl = v_sim['total_pnl'].mean().item() * 100
                    v_trust_mean = v_trust.mean().item()

                if val_sharpe > best_val_sharpe:
                    best_val_sharpe = val_sharpe
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1

                if epoch % (args.eval_every * 20) == 0:
                    marker = '  *best*' if patience_counter == 0 else ''
                    print(f"    epoch {epoch:>5}  loss={loss.item():>7.3f}  "
                          f"train: S={train_sharpe:>5.2f} P={train_pnl:>+6.2f}%  "
                          f"val: S={val_sharpe:>5.2f} P={val_pnl:>+6.2f}%  "
                          f"trust={v_trust_mean:.2f}{marker}")

                if patience_counter > args.patience:
                    print(f"    Early stopping at epoch {epoch}")
                    break

    # Load best and test
    if best_state is not None:
        model.load_state_dict(best_state)

    # Test set
    model.eval()
    test_results = None
    with torch.no_grad():
        test_feat, test_ret = make_sequences(
            data['features'], data['returns'],
            data['val_end'], len(data['features']), seq_len, stride=1
        )
        if test_feat is not None:
            test_pos, test_raw, test_conf, test_trust = model(
                test_feat, data['rule_indices'], data['regime_indices'])
            T_out_test = test_pos.shape[1]
            test_sim = sim(test_pos, test_ret[:, -T_out_test:])
            test_results = {
                'sharpe': test_sim['sharpe'].mean().item(),
                'total_pnl': test_sim['total_pnl'].mean().item() * 100,
                'max_drawdown': test_sim['max_drawdown'].mean().item() * 100,
                'win_rate': test_sim['win_rate'].mean().item() * 100,
                'trade_intensity': test_sim['trade_intensity'].mean().item(),
                'total_costs': test_sim['total_costs'].mean().item() * 100,
                'mean_position': test_pos.abs().mean().item(),
                'mean_confidence': test_conf.mean().item(),
                'mean_trust': test_trust.mean().item(),
            }

    return {
        'seed': seed,
        'best_val_sharpe': best_val_sharpe,
        'test': test_results,
        'state_dict': best_state,
        'n_params': n_params,
    }


def get_buyhold(data):
    """Buy-and-hold baseline for each split."""
    spy = data['spy']

    def pnl(start, end):
        s = spy[start:end]
        if len(s) < 2:
            return 0
        return ((s[-1] - s[0]) / s[0] * 100).item()

    return {
        'train_pnl': pnl(0, data['train_end']),
        'val_pnl': pnl(data['train_end'], data['val_end']),
        'test_pnl': pnl(data['val_end'], len(spy)),
    }


def get_rule_baseline(data, device='cuda'):
    """What the raw rule_combined signal achieves (no NN)."""
    features = data['features']
    returns = data['returns']
    rule_combined_idx = data['rule_indices'][-1]  # rule_combined

    results = {}
    for name, start, end in [('train', 0, data['train_end']),
                               ('val', data['train_end'], data['val_end']),
                               ('test', data['val_end'], len(features))]:
        rule_pos = features[start:end, rule_combined_idx]
        rets = returns[start:end]

        pos_shifted = rule_pos[:-1]
        r = rets[1:]
        bar_pnl = pos_shifted * r
        cum_pnl = torch.cumsum(bar_pnl, dim=0)

        total_pnl = cum_pnl[-1].item() * 100 if len(cum_pnl) > 0 else 0
        mean_pnl = bar_pnl.mean().item()
        std_pnl = bar_pnl.std().item() + 1e-8
        sharpe = mean_pnl / std_pnl * sqrt(252)

        running_max = torch.cummax(cum_pnl, dim=0)[0]
        max_dd = (running_max - cum_pnl).max().item() * 100

        results[name] = {
            'pnl': total_pnl,
            'sharpe': sharpe,
            'max_dd': max_dd,
        }

    return results


# --- Main ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Hybrid Spectral Trader')

    parser.add_argument('--timescale', default='daily', choices=['5min', 'hourly', 'daily'])

    # Model
    parser.add_argument('--hidden', type=int, default=32)
    parser.add_argument('--window', type=int, default=5,
                        help='Lookback window (bars)')

    # Training
    parser.add_argument('--epochs', type=int, default=5000)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-3)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--seq-len', type=int, default=60,
                        help='Sequence length (daily: 60 = ~3 months)')
    parser.add_argument('--stride', type=int, default=10)
    parser.add_argument('--patience', type=int, default=300)
    parser.add_argument('--eval-every', type=int, default=5)

    # Simulation
    parser.add_argument('--cost-bps', type=float, default=5.0)

    # Loss
    parser.add_argument('--lambda-dd', type=float, default=0.5)
    parser.add_argument('--lambda-activity', type=float, default=0.01)
    parser.add_argument('--lambda-utilization', type=float, default=1.0)

    # Ensemble
    parser.add_argument('--seeds', type=int, default=30)

    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=" * 70)
    print("HYBRID SPECTRAL TRADER -- Rules + Learned Refinement")
    print(f"  Device: {device}")
    if device == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Timescale: {args.timescale}")
    print(f"  Window: {args.window} bars, Hidden: {args.hidden}")
    print(f"  Seq len: {args.seq_len}, Stride: {args.stride}")
    print(f"  Seeds: {args.seeds}")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    data = load_data(args.timescale, device)

    # Baselines
    bh = get_buyhold(data)
    print(f"\n  Buy and Hold:")
    print(f"    Train: {bh['train_pnl']:+.2f}%  Val: {bh['val_pnl']:+.2f}%  Test: {bh['test_pnl']:+.2f}%")

    rule_bl = get_rule_baseline(data, device)
    print(f"\n  Rule baseline (no NN):")
    for name in ['train', 'val', 'test']:
        r = rule_bl[name]
        print(f"    {name.capitalize()}: PnL={r['pnl']:+.2f}%  Sharpe={r['sharpe']:.2f}  MaxDD={r['max_dd']:.2f}%")

    # Train ensemble
    print(f"\nTraining {args.seeds} hybrid models...")
    all_results = []
    t0 = time.time()

    for i in range(args.seeds):
        seed = i * 42 + 7
        print(f"\n  --- Seed {seed} ({i + 1}/{args.seeds}) ---")
        result = train_one_seed(data, seed, args, device)
        if result is not None:
            all_results.append(result)
            if result['test'] is not None:
                t = result['test']
                print(f"    TEST: Sharpe={t['sharpe']:.2f} PnL={t['total_pnl']:+.2f}% "
                      f"DD={t['max_drawdown']:.2f}% trust={t['mean_trust']:.2f}")

    elapsed = time.time() - t0
    print(f"\n  Training complete in {elapsed:.1f}s ({elapsed / 60:.1f}min)")

    if not all_results:
        print("  ERROR: no models trained")
        return

    # --- Results ---
    print(f"\n{'=' * 70}")
    print(f"ENSEMBLE RESULTS ({len(all_results)} models)")
    print(f"{'=' * 70}")

    all_results.sort(key=lambda r: r['best_val_sharpe'], reverse=True)

    print(f"\n  {'Seed':>6} {'ValS':>6} {'TestS':>6} {'TestPnL':>9} "
          f"{'MaxDD':>7} {'WR':>6} {'Trust':>6} {'|Pos|':>6} {'Params':>7}")
    print(f"  {'-' * 62}")

    test_sharpes = []
    test_pnls = []
    for r in all_results:
        t = r['test'] or {}
        ts = t.get('sharpe', 0)
        tp = t.get('total_pnl', 0)
        td = t.get('max_drawdown', 0)
        tw = t.get('win_rate', 0)
        tt = t.get('mean_trust', 0)
        mp = t.get('mean_position', 0)
        test_sharpes.append(ts)
        test_pnls.append(tp)
        print(f"  {r['seed']:>6} {r['best_val_sharpe']:>+6.2f} {ts:>+6.2f} "
              f"{tp:>+8.2f}% {td:>6.2f}% {tw:>5.1f}% {tt:>5.2f} {mp:>5.2f} {r['n_params']:>7}")

    print(f"  {'-' * 62}")
    print(f"  {'MEAN':>6} {'':>6} {np.mean(test_sharpes):>+6.2f} "
          f"{np.mean(test_pnls):>+8.2f}%")
    print(f"  {'BEST':>6} {'':>6} {max(test_sharpes):>+6.2f} "
          f"{max(test_pnls):>+8.2f}%")

    print(f"\n  Baselines:")
    print(f"    Buy and Hold test:  {bh['test_pnl']:+.2f}%")
    print(f"    Rule-only test:   {rule_bl['test']['pnl']:+.2f}%  Sharpe={rule_bl['test']['sharpe']:.2f}")
    best_pnl = max(test_pnls)
    mean_pnl = np.mean(test_pnls)
    print(f"    Best hybrid edge vs B&H:  {best_pnl - bh['test_pnl']:+.2f}%")
    print(f"    Best hybrid edge vs Rule: {best_pnl - rule_bl['test']['pnl']:+.2f}%")

    # Save
    save_data = {
        'args': vars(args),
        'buy_hold': bh,
        'rule_baseline': {k: v for k, v in rule_bl.items()},
        'n_models': len(all_results),
        'elapsed_seconds': elapsed,
        'models': [],
    }
    for r in all_results:
        save_data['models'].append({
            'seed': r['seed'],
            'val_sharpe': r['best_val_sharpe'],
            'test': r['test'],
            'n_params': r['n_params'],
        })

    results_path = RESULTS_DIR / f'hybrid_{args.timescale}.json'
    with open(results_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\n  Saved: {results_path}")

    # Save best model
    best = all_results[0]
    if best['state_dict'] is not None:
        model_path = RESULTS_DIR / f'hybrid_best_{args.timescale}.pt'
        torch.save({
            'state_dict': best['state_dict'],
            'seed': best['seed'],
            'val_sharpe': best['best_val_sharpe'],
            'test': best['test'],
            'args': vars(args),
            'feature_names': data['feature_names'],
            'rule_indices': data['rule_indices'],
            'regime_indices': data['regime_indices'],
        }, model_path)
        print(f"  Best model: {model_path}")


if __name__ == '__main__':
    main()
