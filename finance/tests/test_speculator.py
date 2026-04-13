import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import pytest
from speculator import build_evaluation_prompt, parse_ai_decision, SpecPool, Speculator


class TestBuildPrompt:
    def test_includes_ticker(self):
        prompt = build_evaluation_prompt(
            ticker='NVDA',
            triggering_content=['$NVDA squeeze incoming'],
            velocity_data={'mentions_per_min': 5.0, 'trailing_avg': 1.2},
            price=192.50,
            momentum_5d=0.03,
            momentum_20d=0.08,
            already_held=False,
            pool_status={'positions': 2, 'available_pct': 0.60, 'daily_pnl_pct': -0.01},
        )
        assert 'NVDA' in prompt
        assert '192.50' in prompt
        assert 'squeeze' in prompt

    def test_includes_pool_constraints(self):
        prompt = build_evaluation_prompt(
            ticker='TSLA',
            triggering_content=['$TSLA moon'],
            velocity_data={},
            price=250.0,
            momentum_5d=0.05,
            momentum_20d=0.10,
            already_held=True,
            pool_status={'positions': 4, 'available_pct': 0.20, 'daily_pnl_pct': -0.03},
        )
        assert 'already held' in prompt.lower() or 'TSLA' in prompt
        assert '4' in prompt


class TestParseAiDecision:
    def test_valid_buy(self):
        raw = json.dumps({
            'action': 'buy',
            'ticker': 'NVDA',
            'conviction': 0.8,
            'reasoning': 'Short squeeze setup',
            'entry_strategy': 'market',
            'position_size_pct': 0.30,
            'stop_loss_pct': 0.08,
            'target_pct': 0.15,
            'exit_timeframe_hours': 24,
            'category': 'momentum',
        })
        decision = parse_ai_decision(raw)
        assert decision['action'] == 'buy'
        assert decision['stop_loss_pct'] == 0.08
        assert 0 < decision['position_size_pct'] <= 1.0

    def test_rejects_missing_stop_loss(self):
        raw = json.dumps({
            'action': 'buy',
            'ticker': 'NVDA',
            'conviction': 0.8,
            'position_size_pct': 0.30,
        })
        decision = parse_ai_decision(raw)
        assert decision is None

    def test_pass_action(self):
        raw = json.dumps({'action': 'pass', 'ticker': 'NVDA', 'reasoning': 'not enough signal'})
        decision = parse_ai_decision(raw)
        assert decision['action'] == 'pass'

    def test_garbage_input(self):
        decision = parse_ai_decision('this is not json at all')
        assert decision is None


class TestSpecPool:
    def test_initial_state(self):
        pool = SpecPool(total_capital=100_000, spec_pct=0.10)
        assert pool.capital == 10_000
        assert pool.position_count == 0
        assert pool.available_capital == 10_000

    def test_can_trade_under_limit(self):
        pool = SpecPool(total_capital=100_000, spec_pct=0.10)
        assert pool.can_trade() is True

    def test_cannot_trade_at_max_positions(self):
        pool = SpecPool(total_capital=100_000, spec_pct=0.10)
        pool.positions = {f'T{i}': {'shares': 1, 'entry_price': 100} for i in range(5)}
        assert pool.can_trade() is False

    def test_circuit_breaker_triggers(self):
        pool = SpecPool(total_capital=100_000, spec_pct=0.10)
        pool.daily_pnl = -600
        assert pool.circuit_breaker_tripped() is True

    def test_circuit_breaker_ok(self):
        pool = SpecPool(total_capital=100_000, spec_pct=0.10)
        pool.daily_pnl = -200
        assert pool.circuit_breaker_tripped() is False

    def test_save_and_load(self, tmp_path, monkeypatch):
        import speculator
        monkeypatch.setattr(speculator, 'POSITIONS_PATH', tmp_path / 'spec_positions.json')
        pool = SpecPool(total_capital=100_000, spec_pct=0.10)
        pool.positions = {'NVDA': {'shares': 10, 'entry_price': 200}}
        pool.daily_pnl = -150
        pool.save()

        pool2 = SpecPool(total_capital=100_000, spec_pct=0.10)
        pool2.load()
        assert 'NVDA' in pool2.positions
        assert pool2.daily_pnl == -150


class TestSpeculatorQueue:
    def test_read_queue_empty(self, tmp_path):
        spec = Speculator.__new__(Speculator)
        spec.queue_path = tmp_path / 'queue.json'
        assert spec.read_queue() == []

    def test_read_queue_with_items(self, tmp_path):
        spec = Speculator.__new__(Speculator)
        spec.queue_path = tmp_path / 'queue.json'
        spec.queue_path.write_text(json.dumps([
            {'ticker': 'NVDA', 'score': 7.0, 'text': 'moon'},
        ]))
        items = spec.read_queue()
        assert len(items) == 1
        assert items[0]['ticker'] == 'NVDA'

    def test_read_queue_clears_after_read(self, tmp_path):
        spec = Speculator.__new__(Speculator)
        spec.queue_path = tmp_path / 'queue.json'
        spec.queue_path.write_text(json.dumps([{'ticker': 'NVDA'}]))
        spec.read_queue()
        assert spec.read_queue() == []
