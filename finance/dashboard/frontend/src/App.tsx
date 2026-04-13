import { useState } from 'react'
import { StreamProvider } from './context/StreamContext'
import { TopBar } from './components/TopBar'
import { Sidebar } from './components/Sidebar'
import { Balances } from './components/Balances'
import { Positions } from './components/Positions'
import { Orders } from './components/Orders'
import { Transactions } from './components/Transactions'
import { RegimeBar } from './components/RegimeBar'
import { TickerTape } from './components/TickerTape'
import './App.css'

function Dashboard() {
  const [panels, setPanels] = useState<Record<string, boolean>>({
    positions: true,
    orders: true,
    transactions: true,
  })

  const togglePanel = (key: string) => {
    setPanels(prev => ({ ...prev, [key]: !prev[key] }))
  }

  return (
    <div className="app">
      <TopBar />
      <div className="main-layout">
        <Sidebar panels={panels} onTogglePanel={togglePanel} />
        <main className="dashboard">
          <Balances />
          <RegimeBar />
          {panels.positions && <Positions />}
          <div className="panel-row">
            {panels.orders && <Orders />}
            {panels.transactions && <Transactions />}
          </div>
        </main>
      </div>
      <TickerTape />
    </div>
  )
}

export default function App() {
  return (
    <StreamProvider>
      <Dashboard />
    </StreamProvider>
  )
}
