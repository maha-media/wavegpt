import { useStream } from '../context/StreamContext'

export function RegimeBar() {
  const { regime, mode } = useStream()

  if (mode !== 'test' || !regime) return null

  return (
    <div className="panel regime-bar">
      <div className="regime-grid">
        <div className="regime-item">
          <span className="label">Regime</span>
          <span className={`value regime-value regime-${regime.regime.toLowerCase()}`}>
            {regime.regime.replace('_', ' ')}
          </span>
        </div>
        <div className="regime-item">
          <span className="label">Leader Score</span>
          <span className="value">{regime.leader_score >= 0 ? '+' : ''}{regime.leader_score.toFixed(3)}</span>
        </div>
        <div className="regime-item">
          <span className="label">Tech Weight</span>
          <span className="value">{(regime.tech_pct * 100).toFixed(1)}%</span>
        </div>
        <div className="regime-item">
          <span className="label">Ticks</span>
          <span className="value">{regime.ticks.toLocaleString()}</span>
        </div>
        <div className="regime-item">
          <span className="label">Prices</span>
          <span className="value">{regime.prices_connected}/30</span>
        </div>
        <div className="regime-item">
          <span className="label">Last Update</span>
          <span className="value">
            {regime.timestamp ? new Date(regime.timestamp).toLocaleTimeString() : '—'}
          </span>
        </div>
      </div>
    </div>
  )
}
