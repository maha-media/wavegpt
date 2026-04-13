import { useEffect, useState } from 'react'
import type { Transaction } from '../types'

export function Transactions() {
  const [txns, setTxns] = useState<Transaction[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const load = () => {
      fetch('/api/transactions/')
        .then(r => r.json())
        .then(data => {
          if (Array.isArray(data)) setTxns(data)
          setLoading(false)
        })
        .catch(() => setLoading(false))
    }
    load()
    const interval = setInterval(load, 60000)
    return () => clearInterval(interval)
  }, [])

  if (loading) return <div className="panel transactions">Loading...</div>

  return (
    <div className="panel transactions">
      <h2>Recent Transactions</h2>
      {txns.length === 0 ? (
        <p className="empty">No transactions</p>
      ) : (
        <table>
          <thead>
            <tr><th>Date</th><th>Type</th><th>Symbol</th><th>Amount</th><th>Description</th></tr>
          </thead>
          <tbody>
            {txns.map(t => (
              <tr key={t.id}>
                <td>{new Date(t.date).toLocaleDateString()}</td>
                <td>{t.type}</td>
                <td className="symbol">{t.symbol}</td>
                <td>${t.amount.toFixed(2)}</td>
                <td>{t.description}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  )
}
