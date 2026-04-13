export interface Balance {
  net_liq: number
  cash: number
  buying_power: number
  pnl?: number
  pnl_pct?: number
  starting_capital?: number
}

export interface Position {
  symbol: string
  qty: number
  avg_cost: number
  price: number
  market_value: number
  pnl: number
  pnl_pct: number
  tax_status?: string
}

export interface Regime {
  regime: string
  leader_score: number
  tech_pct: number
  ticks: number
  prices_connected: number
  timestamp: string
}

export interface Quote {
  symbol: string
  bid: number
  ask: number
  mid: number
}

export interface Order {
  id: string
  symbol: string
  action: string
  qty: number
  type: string
  status: string
}

export interface Transaction {
  id: string
  date: string
  type: string
  symbol: string
  amount: number
  description: string
}

export interface ChatMessage {
  role: 'user' | 'assistant'
  content: string
}
