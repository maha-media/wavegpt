import { useStream } from '../context/StreamContext'

export function TopBar() {
  const { balances, connected, mode, regime } = useStream()
  const netliq = balances?.net_liq ?? 0
  const pnl = balances?.pnl ?? 0
  const pnlPct = balances?.pnl_pct ?? 0

  return (
    <header className="topbar">
      <div className="topbar-left">
        <span className="logo">Signals</span>
        {mode === 'test' && <span className="mode-badge">TEST</span>}
        {mode === 'test' && regime && (
          <span className={`regime-pill regime-${regime.regime.toLowerCase()}`}>
            {regime.regime.replace('_', ' ')}
          </span>
        )}
        <span className={`connection-dot ${connected ? 'connected' : 'disconnected'}`} />
      </div>
      <div className="topbar-right">
        {mode === 'test' && connected && (
          <span className={`topbar-pnl ${pnl >= 0 ? 'pnl-positive' : 'pnl-negative'}`}>
            {pnl >= 0 ? '+' : ''}{pnl.toLocaleString(undefined, { maximumFractionDigits: 0 })} ({pnlPct >= 0 ? '+' : ''}{pnlPct.toFixed(2)}%)
          </span>
        )}
        <span className="net-liq">
          ${netliq.toLocaleString(undefined, { maximumFractionDigits: 0 })}
        </span>
      </div>
    </header>
  )
}
