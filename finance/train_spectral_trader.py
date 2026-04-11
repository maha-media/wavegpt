"""
Phase 1: GPU-Accelerated Spectral Alpha Trader

Differentiable trading simulator + neural network that learns
optimal position sizing from spectral features.

Architecture:
  Input: [T, 75] features
  -> Feature normalizer (LayerNorm)
  -> Temporal encoder (causal conv1d, sees last N bars)
  -> Trading head -> position in [-1, +1], confidence in [0, 1]
  -> Differentiable PnL simulator
  -> Loss = -Sharpe + lambda_dd * max_drawdown + lambda_activity * trade_intensity

Walk-forward: train 2021-2024, validate early 2025, test mid 2025-2026.
Multiple random seeds -> ensemble.

Usage:
    python finance/train_spectral_trader.py
    python finance/train_spectral_trader.py --epochs 5000 --seeds 50
    python finance/train_spectral_trader.py --timescale hourly
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


# --- Model -------------------------------------------------------

class SpectralTrader(nn.Module):
    """Neural network that maps spectral features to trading positions.

    Architecture:
      features [B, T, D]
        -> LayerNorm
        -> Causal Conv1D (sees last `context` bars)
        -> GELU + Conv1D
        -> Linear -> position (tanh), confidence (sigmoid)
    """

    def __init__(self, n_features, hidden=128, context=20, n_conv_layers=3):
        super().__init__()
        self.context = context

        self.norm = nn.LayerNorm(n_features)

        # Causal temporal convolutions
        layers = []
        in_ch = n_features
        for i in range(n_conv_layers):
            out_ch = hidden
            layers.append(nn.Conv1d(in_ch, out_ch, kernel_size=context if i == 0 else 3,
                                    padding=0))
            layers.append(nn.GELU())
            in_ch = out_ch
        self.conv = nn.ModuleList(layers)

        # Trading head
        self.position_head = nn.Linear(hidden, 1)   # -> tanh -> [-1, 1]
        self.confidence_head = nn.Linear(hidden, 1)  # -> sigmoid -> [0, 1]

    def forward(self, x):
        """
        Args:
            x: [B, T, D] feature tensor

        Returns:
            positions: [B, T'] where T' = T - receptive_field + 1
            confidence: [B, T']
        """
        # Normalize features
        x = self.norm(x)

        # Conv expects [B, C, T]
        h = x.transpose(1, 2)

        # Causal convolutions (no future information leaks)
        for layer in self.conv:
            h = layer(h)

        # Back to [B, T', hidden]
        h = h.transpose(1, 2)

        positions = torch.tanh(self.position_head(h)).squeeze(-1)
        confidence = torch.sigmoid(self.confidence_head(h)).squeeze(-1)

        # Scale position by confidence
        scaled_positions = positions * confidence

        return scaled_positions, positions, confidence


# --- Differentiable Trading Simulator -----------------------------

class DifferentiableTradingSim(nn.Module):
    """Simulates PnL from positions, fully differentiable.

    PnL_t = position_{t-1} * return_t
    Sharpe = mean(PnL_t) / std(PnL_t) * sqrt(252 * bars_per_day)
    Transaction costs = cost_bps * |position_t - position_{t-1}|
    """

    def __init__(self, cost_bps=5.0, bars_per_day=78):
        super().__init__()
        self.cost_bps = cost_bps / 10000.0  # basis points to fraction
        self.bars_per_day = bars_per_day

    def forward(self, positions, returns, spy_prices=None):
        """
        Args:
            positions: [B, T] position sizing in [-1, 1]
            returns: [B, T] next-bar returns (fractional)

        Returns:
            dict with PnL, Sharpe, drawdown, trade count, etc.
        """
        # PnL: position at time t earns return at time t+1
        pos_shifted = positions[:, :-1]
        rets = returns[:, 1:]

        # Per-bar PnL (fractional)
        bar_pnl = pos_shifted * rets

        # Transaction costs
        position_changes = torch.abs(positions[:, 1:] - positions[:, :-1])
        costs = position_changes * self.cost_bps
        bar_pnl_net = bar_pnl - costs

        # Cumulative PnL
        cum_pnl = torch.cumsum(bar_pnl_net, dim=1)

        # Sharpe ratio (annualized)
        mean_pnl = bar_pnl_net.mean(dim=1)
        std_pnl = bar_pnl_net.std(dim=1) + 1e-8
        annualization = sqrt(252 * self.bars_per_day)
        sharpe = mean_pnl / std_pnl * annualization

        # Max drawdown (differentiable)
        running_max = torch.cummax(cum_pnl, dim=1)[0]
        drawdowns = running_max - cum_pnl
        max_drawdown = drawdowns.max(dim=1)[0]

        # Trade activity (for regularization)
        trade_intensity = position_changes.mean(dim=1)

        # Market utilization (how much the model is in the market)
        mean_abs_position = positions.abs().mean(dim=1)

        # Total PnL
        total_pnl = cum_pnl[:, -1]

        # Win rate (non-differentiable, for logging only)
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
            'mean_pnl': mean_pnl,
            'total_costs': costs.sum(dim=1),
        }


def trading_loss(sim_results, lambda_dd=0.5, lambda_activity=0.01,
                 lambda_utilization=1.0):
    """Combined loss: maximize PnL-adjusted Sharpe, penalize idleness.

    The key insight: pure -Sharpe is degenerate (global min = do nothing).
    Instead we maximize PnL directly, with Sharpe as a risk penalty,
    and add a utilization incentive so the model must actually trade.
    """
    # Primary: maximize total PnL
    loss = -sim_results['total_pnl'] * 100  # scale up small returns

    # Risk adjustment: penalize high drawdown
    loss = loss + lambda_dd * sim_results['max_drawdown'] * 100

    # Anti-idleness: penalize NOT trading (mean |position| should be > 0.3)
    # This prevents the "do nothing" degenerate solution
    utilization = sim_results['mean_abs_position']
    target_util = 0.3
    util_penalty = F.relu(target_util - utilization)  # penalty if utilization < 0.3
    loss = loss + lambda_utilization * util_penalty * 10

    # Light penalty on excessive churning (but much less than utilization reward)
    loss = loss + lambda_activity * sim_results['trade_intensity']

    return loss.mean()


# --- Data Loading -------------------------------------------------

def load_data(timescale='5min', device='cuda'):
    """Load feature tensors and split into train/val/test."""
    path = DATA_DIR / f'features_{timescale}.pt'
    if not path.exists():
        print(f"ERROR: {path} not found. Run build_features.py first.")
        sys.exit(1)

    data = torch.load(path, weights_only=False)
    features = data['features'].to(device)
    targets = data['targets'].to(device)
    spy = data['spy'].to(device)
    timestamps = data['timestamps']
    feature_names = data['feature_names']

    # Compute returns from SPY prices
    returns = torch.zeros_like(spy)
    returns[1:] = (spy[1:] - spy[:-1]) / (spy[:-1] + 1e-8)

    T = features.shape[0]
    print(f"  Loaded {timescale}: {T} bars x {features.shape[1]} features")

    # Walk-forward split by timestamp
    val_start_str = '2025-01-01'
    test_start_str = '2025-07-01'

    val_idx = T
    test_idx = T
    for i, ts in enumerate(timestamps):
        if val_start_str in ts and val_idx == T:
            val_idx = i
        if test_start_str in ts and test_idx == T:
            test_idx = i

    # Fallback to proportional split
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
        'train_idx': val_idx,
        'val_idx': test_idx,
        'test_idx': T,
    }


def make_sequences(features, returns, spy, start, end, seq_len, stride=1):
    """Create overlapping sequences for training."""
    feat_list, ret_list, spy_list = [], [], []
    for i in range(start, end - seq_len, stride):
        feat_list.append(features[i:i+seq_len])
        ret_list.append(returns[i:i+seq_len])
        spy_list.append(spy[i:i+seq_len])

    if not feat_list:
        return None, None, None

    return (torch.stack(feat_list),
            torch.stack(ret_list),
            torch.stack(spy_list))


# --- Training -----------------------------------------------------

def train_one_seed(data, seed, args, device='cuda'):
    """Train a single model with a specific random seed."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    n_features = data['features'].shape[1]
    seq_len = args.seq_len
    bars_per_day = {'5min': 78, 'hourly': 7, 'daily': 1}[args.timescale]

    model = SpectralTrader(
        n_features=n_features,
        hidden=args.hidden,
        context=min(args.context, seq_len // 2),
        n_conv_layers=args.n_layers,
    ).to(device)

    sim = DifferentiableTradingSim(cost_bps=args.cost_bps, bars_per_day=bars_per_day).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Make training sequences
    train_feat, train_ret, train_spy = make_sequences(
        data['features'], data['returns'], data['spy'],
        0, data['train_idx'], seq_len, stride=args.stride
    )
    val_feat, val_ret, val_spy = make_sequences(
        data['features'], data['returns'], data['spy'],
        data['train_idx'], data['val_idx'], seq_len, stride=seq_len // 2
    )

    if train_feat is None:
        print(f"  Seed {seed}: not enough data for sequences")
        return None

    n_train = train_feat.shape[0]
    n_val = val_feat.shape[0] if val_feat is not None else 0

    best_val_sharpe = -float('inf')
    best_state = None
    patience_counter = 0
    history = []

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

        # Forward
        positions, raw_pos, confidence = model(batch_feat)
        T_out = positions.shape[1]
        ret_aligned = batch_ret[:, -T_out:]

        # Simulate trading
        sim_results = sim(positions, ret_aligned)

        # Loss
        loss = trading_loss(sim_results,
                          lambda_dd=args.lambda_dd,
                          lambda_activity=args.lambda_activity,
                          lambda_utilization=args.lambda_utilization)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # Periodic validation
        if epoch % args.eval_every == 0 or epoch == args.epochs - 1:
            model.eval()
            with torch.no_grad():
                train_pos, _, _ = model(train_feat)
                T_out_train = train_pos.shape[1]
                train_sim = sim(train_pos, train_ret[:, -T_out_train:])
                train_sharpe = train_sim['sharpe'].mean().item()
                train_pnl = train_sim['total_pnl'].mean().item() * 100

                val_sharpe = 0
                val_pnl = 0
                if val_feat is not None and n_val > 0:
                    val_pos, _, val_conf = model(val_feat)
                    T_out_val = val_pos.shape[1]
                    val_sim = sim(val_pos, val_ret[:, -T_out_val:])
                    val_sharpe = val_sim['sharpe'].mean().item()
                    val_pnl = val_sim['total_pnl'].mean().item() * 100

                record = {
                    'epoch': epoch,
                    'loss': loss.item(),
                    'train_sharpe': train_sharpe,
                    'train_pnl': train_pnl,
                    'val_sharpe': val_sharpe,
                    'val_pnl': val_pnl,
                }
                history.append(record)

                if val_sharpe > best_val_sharpe:
                    best_val_sharpe = val_sharpe
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1

                if epoch % (args.eval_every * 10) == 0:
                    marker = '  *best*' if patience_counter == 0 else ''
                    print(f"    epoch {epoch:>5}  loss={loss.item():>7.3f}  "
                          f"train: Sharpe={train_sharpe:>6.2f} PnL={train_pnl:>+7.2f}%  "
                          f"val: Sharpe={val_sharpe:>6.2f} PnL={val_pnl:>+7.2f}%{marker}")

                if patience_counter > args.patience:
                    print(f"    Early stopping at epoch {epoch} (patience={args.patience})")
                    break

    # Load best model for test
    if best_state is not None:
        model.load_state_dict(best_state)

    # Test set
    model.eval()
    test_results = None
    with torch.no_grad():
        test_feat, test_ret, test_spy_seq = make_sequences(
            data['features'], data['returns'], data['spy'],
            data['val_idx'], data['test_idx'], seq_len, stride=1
        )
        if test_feat is not None:
            test_pos, test_raw, test_conf = model(test_feat)
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
            }

    return {
        'seed': seed,
        'best_val_sharpe': best_val_sharpe,
        'history': history,
        'test': test_results,
        'state_dict': best_state,
        'n_params': sum(p.numel() for p in model.parameters()),
    }


def get_buyhold(data):
    """Compute buy-and-hold metrics for comparison."""
    spy = data['spy']
    train_spy = spy[:data['train_idx']]
    train_ret = (train_spy[-1] - train_spy[0]) / train_spy[0] * 100

    val_spy = spy[data['train_idx']:data['val_idx']]
    val_ret = (val_spy[-1] - val_spy[0]) / val_spy[0] * 100 if len(val_spy) > 1 else 0

    test_spy = spy[data['val_idx']:data['test_idx']]
    test_ret = (test_spy[-1] - test_spy[0]) / test_spy[0] * 100 if len(test_spy) > 1 else 0

    return {
        'train_pnl': train_ret.item(),
        'val_pnl': val_ret.item(),
        'test_pnl': test_ret.item(),
    }


# --- Main ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Train Spectral Alpha Trader')

    # Data
    parser.add_argument('--timescale', default='5min', choices=['5min', 'hourly', 'daily'])

    # Model
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--context', type=int, default=20,
                        help='Causal conv lookback (bars)')
    parser.add_argument('--n-layers', type=int, default=3)

    # Training
    parser.add_argument('--epochs', type=int, default=3000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--seq-len', type=int, default=1024,
                        help='Sequence length for training (bars)')
    parser.add_argument('--stride', type=int, default=128,
                        help='Stride between training sequences')
    parser.add_argument('--patience', type=int, default=200,
                        help='Early stopping patience (in validation checks)')
    parser.add_argument('--eval-every', type=int, default=10)

    # Simulation
    parser.add_argument('--cost-bps', type=float, default=5.0,
                        help='Transaction cost in basis points')

    # Loss weights
    parser.add_argument('--lambda-dd', type=float, default=0.5)
    parser.add_argument('--lambda-activity', type=float, default=0.01)
    parser.add_argument('--lambda-utilization', type=float, default=1.0)

    # Ensemble
    parser.add_argument('--seeds', type=int, default=20,
                        help='Number of random seeds for ensemble')

    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=" * 70)
    print("SPECTRAL ALPHA TRADER -- GPU Training")
    print(f"  Device: {device}")
    if device == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Timescale: {args.timescale}")
    print(f"  Sequence length: {args.seq_len} bars")
    print(f"  Hidden: {args.hidden}, Context: {args.context}, Layers: {args.n_layers}")
    print(f"  Transaction cost: {args.cost_bps} bps")
    print(f"  Seeds: {args.seeds}")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    data = load_data(args.timescale, device)

    # Buy and hold baseline
    bh = get_buyhold(data)
    print(f"\n  Buy and Hold baseline:")
    print(f"    Train: {bh['train_pnl']:+.2f}%")
    print(f"    Val:   {bh['val_pnl']:+.2f}%")
    print(f"    Test:  {bh['test_pnl']:+.2f}%")

    # Train ensemble
    print(f"\nTraining {args.seeds} models...")
    all_results = []
    t0 = time.time()

    for i in range(args.seeds):
        seed = i * 42 + 7
        print(f"\n  --- Seed {seed} ({i+1}/{args.seeds}) ---")
        result = train_one_seed(data, seed, args, device)
        if result is not None:
            all_results.append(result)
            if result['test'] is not None:
                t = result['test']
                print(f"    TEST: Sharpe={t['sharpe']:.2f} PnL={t['total_pnl']:+.2f}% "
                      f"DD={t['max_drawdown']:.2f}% WR={t['win_rate']:.1f}%")

    elapsed = time.time() - t0
    print(f"\n  Training complete in {elapsed:.1f}s ({elapsed/60:.1f}min)")

    if not all_results:
        print("  ERROR: no models trained successfully")
        return

    # --- Results ---------------------------------------------------

    print(f"\n{'='*70}")
    print(f"ENSEMBLE RESULTS ({len(all_results)} models)")
    print(f"{'='*70}")

    all_results.sort(key=lambda r: r['best_val_sharpe'], reverse=True)

    print(f"\n  {'Seed':>6} {'ValSharpe':>10} {'TestSharpe':>11} {'TestPnL':>9} "
          f"{'MaxDD':>8} {'WinRate':>8} {'Params':>8}")
    print(f"  {'-'*66}")

    test_sharpes = []
    test_pnls = []
    for r in all_results:
        t = r['test'] or {}
        ts = t.get('sharpe', 0)
        tp = t.get('total_pnl', 0)
        td = t.get('max_drawdown', 0)
        tw = t.get('win_rate', 0)
        test_sharpes.append(ts)
        test_pnls.append(tp)
        print(f"  {r['seed']:>6} {r['best_val_sharpe']:>+10.2f} {ts:>+11.2f} "
              f"{tp:>+8.2f}% {td:>7.2f}% {tw:>7.1f}% {r['n_params']:>8}")

    print(f"  {'-'*66}")
    print(f"  {'MEAN':>6} {'':>10} {np.mean(test_sharpes):>+11.2f} "
          f"{np.mean(test_pnls):>+8.2f}%")
    print(f"  {'BEST':>6} {'':>10} {max(test_sharpes):>+11.2f} "
          f"{max(test_pnls):>+8.2f}%")

    print(f"\n  Buy and Hold test: {bh['test_pnl']:+.2f}%")
    best_pnl = max(test_pnls)
    mean_pnl = np.mean(test_pnls)
    print(f"  Best model edge vs B&H: {best_pnl - bh['test_pnl']:+.2f}%")
    print(f"  Ensemble mean edge:     {mean_pnl - bh['test_pnl']:+.2f}%")

    # Top-5 analysis
    top_n = min(5, len(all_results))
    print(f"\n  Top-{top_n} model details:")
    for r in all_results[:top_n]:
        t = r['test'] or {}
        print(f"    Seed {r['seed']}: avg|pos|={t.get('mean_position', 0):.3f} "
              f"conf={t.get('mean_confidence', 0):.3f} "
              f"costs={t.get('total_costs', 0):.3f}%")

    # Save results
    save_data = {
        'args': vars(args),
        'buy_hold': bh,
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
            'n_epochs': len(r['history']) * args.eval_every if r['history'] else 0,
        })

    results_path = RESULTS_DIR / f'training_{args.timescale}.json'
    with open(results_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\n  Saved: {results_path}")

    # Save best model weights
    best = all_results[0]
    if best['state_dict'] is not None:
        model_path = RESULTS_DIR / f'best_model_{args.timescale}.pt'
        torch.save({
            'state_dict': best['state_dict'],
            'seed': best['seed'],
            'val_sharpe': best['best_val_sharpe'],
            'test': best['test'],
            'args': vars(args),
            'feature_names': data['feature_names'],
        }, model_path)
        print(f"  Best model: {model_path}")


if __name__ == '__main__':
    main()
