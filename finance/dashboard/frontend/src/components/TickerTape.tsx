import { useStream } from '../context/StreamContext'

export function TickerTape() {
  const { quotes } = useStream()
  const symbols = Object.keys(quotes).sort()

  if (symbols.length === 0) return null

  return (
    <footer className="ticker-tape">
      <div className="ticker-scroll">
        {symbols.map(sym => {
          const q = quotes[sym]
          return (
            <span key={sym} className="ticker-item">
              <span className="ticker-symbol">{sym}</span>
              <span className="ticker-price">{q.mid.toFixed(2)}</span>
            </span>
          )
        })}
      </div>
    </footer>
  )
}
