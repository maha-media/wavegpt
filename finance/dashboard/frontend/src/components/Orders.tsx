import { useStream } from '../context/StreamContext'

export function Orders() {
  const { orders } = useStream()

  return (
    <div className="panel orders">
      <h2>Open Orders</h2>
      {orders.length === 0 ? (
        <p className="empty">No open orders</p>
      ) : (
        <table>
          <thead>
            <tr><th>Symbol</th><th>Action</th><th>Qty</th><th>Type</th><th>Status</th></tr>
          </thead>
          <tbody>
            {orders.map(o => (
              <tr key={o.id}>
                <td className="symbol">{o.symbol}</td>
                <td>{o.action}</td>
                <td>{o.qty}</td>
                <td>{o.type}</td>
                <td>{o.status}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  )
}
