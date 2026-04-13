import { useStream } from '../context/StreamContext'

function pnlClass(value: number): string {
  if (value > 0) return 'pnl-positive'
  if (value < 0) return 'pnl-negative'
  return ''
}

export function Positions() {
  const { positions, quotes, mode } = useStream()

  const enriched = positions.map(p => {
    const q = quotes[p.symbol]
    const price = q?.mid ?? p.price
    const mv = p.qty * price
    const cost = p.qty * p.avg_cost
    const pnl = mv - cost
    const pnl_pct = cost ? (pnl / cost) * 100 : 0
    return { ...p, price, market_value: mv, pnl, pnl_pct }
  })

  return (
    <div className="panel positions">
      <h2>Positions</h2>
      {enriched.length === 0 ? (
        <p className="empty">No positions</p>
      ) : (
        <table>
          <thead>
            <tr>
              <th>Symbol</th><th>Qty</th><th>Avg Cost</th><th>Price</th>
              <th>Value</th><th>P/L</th><th>P/L %</th>
              {mode === 'test' && <th>Tax</th>}
            </tr>
          </thead>
          <tbody>
            {enriched.map(p => (
              <tr key={p.symbol}>
                <td className="symbol">{p.symbol}</td>
                <td>{p.qty.toLocaleString()}</td>
                <td>${p.avg_cost.toFixed(2)}</td>
                <td>${p.price.toFixed(2)}</td>
                <td>${p.market_value.toLocaleString(undefined, { maximumFractionDigits: 0 })}</td>
                <td className={pnlClass(p.pnl)}>${p.pnl.toFixed(0)}</td>
                <td className={pnlClass(p.pnl_pct)}>{p.pnl_pct.toFixed(2)}%</td>
                {mode === 'test' && <td className="tax-status">{p.tax_status || '—'}</td>}
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  )
}
