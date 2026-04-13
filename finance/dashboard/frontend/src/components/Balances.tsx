import { useStream } from '../context/StreamContext'

export function Balances() {
  const { balances } = useStream()
  if (!balances) return <div className="panel balances">Connecting...</div>

  return (
    <div className="panel balances">
      <h2>Account</h2>
      <div className="balance-grid">
        <div className="balance-item">
          <span className="label">Net Liquidating</span>
          <span className="value">
            ${balances.net_liq.toLocaleString(undefined, { maximumFractionDigits: 0 })}
          </span>
        </div>
        <div className="balance-item">
          <span className="label">Cash</span>
          <span className="value">
            ${balances.cash.toLocaleString(undefined, { maximumFractionDigits: 0 })}
          </span>
        </div>
        <div className="balance-item">
          <span className="label">Buying Power</span>
          <span className="value">
            ${balances.buying_power.toLocaleString(undefined, { maximumFractionDigits: 0 })}
          </span>
        </div>
      </div>
    </div>
  )
}
