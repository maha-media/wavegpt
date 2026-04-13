import { useState } from 'react'
import { ChatPanel } from './ChatPanel'

interface SidebarProps {
  panels: Record<string, boolean>
  onTogglePanel: (panel: string) => void
}

const PANEL_LABELS: Record<string, string> = {
  positions: 'Positions',
  orders: 'Orders',
  transactions: 'Transactions',
}

export function Sidebar({ panels, onTogglePanel }: SidebarProps) {
  const [collapsed, setCollapsed] = useState(false)

  return (
    <aside className={`sidebar ${collapsed ? 'collapsed' : ''}`}>
      <button className="sidebar-toggle" onClick={() => setCollapsed(!collapsed)}>
        {collapsed ? '\u25B6' : '\u25C0'}
      </button>
      {!collapsed && (
        <>
          <div className="sidebar-section">
            <h3>Views</h3>
            {Object.entries(PANEL_LABELS).map(([key, label]) => (
              <label key={key} className="panel-toggle">
                <input
                  type="checkbox"
                  checked={panels[key] ?? true}
                  onChange={() => onTogglePanel(key)}
                />
                {label}
              </label>
            ))}
          </div>
          <div className="sidebar-section sidebar-chat">
            <ChatPanel />
          </div>
        </>
      )}
    </aside>
  )
}
