import { createContext, useContext, useState, type ReactNode } from 'react'
import { useSSE } from '../hooks/useSSE'
import type { Balance, Position, Quote, Order, Regime } from '../types'

interface StreamState {
  balances: Balance | null
  positions: Position[]
  quotes: Record<string, Quote>
  orders: Order[]
  connected: boolean
  mode: string
  regime: Regime | null
}

const StreamContext = createContext<StreamState>({
  balances: null,
  positions: [],
  quotes: {},
  orders: [],
  connected: false,
  mode: 'live',
  regime: null,
})

export function StreamProvider({ children }: { children: ReactNode }) {
  const [balances, setBalances] = useState<Balance | null>(null)
  const [positions, setPositions] = useState<Position[]>([])
  const [quotes, setQuotes] = useState<Record<string, Quote>>({})
  const [orders, setOrders] = useState<Order[]>([])
  const [connected, setConnected] = useState(false)
  const [mode, setMode] = useState('live')
  const [regime, setRegime] = useState<Regime | null>(null)

  useSSE('/api/stream/', {
    mode: (data: { mode: string }) => setMode(data.mode),
    balances: (data: Balance) => {
      setBalances(data)
      setConnected(true)
    },
    positions: (data: Position[]) => setPositions(data),
    quote: (data: Quote) => {
      setQuotes(prev => ({ ...prev, [data.symbol]: data }))
    },
    orders: (data: Order[]) => setOrders(data),
    regime: (data: Regime) => setRegime(data),
    heartbeat: () => setConnected(true),
  })

  return (
    <StreamContext.Provider value={{ balances, positions, quotes, orders, connected, mode, regime }}>
      {children}
    </StreamContext.Provider>
  )
}

export const useStream = () => useContext(StreamContext)
