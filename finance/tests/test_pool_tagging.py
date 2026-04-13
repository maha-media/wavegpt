import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from stream_trader import LocalPortfolio


class TestPoolTagging:
    def test_default_pool_is_core(self):
        p = LocalPortfolio(100_000)
        p.fill_order('AAPL', 10, 150.0)
        assert p.positions['AAPL']['pool'] == 'core'

    def test_spec_pool_tag(self):
        p = LocalPortfolio(100_000)
        p.fill_order('NVDA', 5, 200.0, pool='spec')
        assert p.positions['NVDA']['pool'] == 'spec'

    def test_pool_value_by_type(self):
        p = LocalPortfolio(100_000)
        p.fill_order('AAPL', 10, 150.0, pool='core')
        p.fill_order('NVDA', 5, 200.0, pool='spec')
        prices = {'AAPL': 150.0, 'NVDA': 200.0}
        core_val = sum(
            pos['shares'] * prices.get(sym, pos['avg_cost'])
            for sym, pos in p.positions.items()
            if pos.get('pool') == 'core'
        )
        spec_val = sum(
            pos['shares'] * prices.get(sym, pos['avg_cost'])
            for sym, pos in p.positions.items()
            if pos.get('pool') == 'spec'
        )
        assert core_val == 1500.0
        assert spec_val == 1000.0

    def test_tax_events_include_pool(self):
        p = LocalPortfolio(100_000)
        p.fill_order('AAPL', 10, 150.0, pool='spec')
        p.fill_order('AAPL', -10, 140.0, pool='spec')
        assert p.tax_events[-1]['pool'] == 'spec'

    def test_existing_callers_default_to_core(self):
        """Existing code that doesn't pass pool= still works."""
        p = LocalPortfolio(100_000)
        p.fill_order('MSFT', 5, 300.0)
        p.fill_order('MSFT', -5, 310.0)
        assert p.tax_events[-1]['pool'] == 'core'
