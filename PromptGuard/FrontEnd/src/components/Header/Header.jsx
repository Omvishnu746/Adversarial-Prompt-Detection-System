import React, { useEffect, useState } from 'react'
import { fetchHealth } from '../../services/promptGuardApi'
import './Header.css'

export default function Header({ onMenuToggle, sidebarOpen }) {
  const [health, setHealth] = useState(null)

  useEffect(() => {
    fetchHealth()
      .then(setHealth)
      .catch(() => setHealth({ status: 'error' }))
  }, [])

  const isOnline = health?.status === 'ok'

  return (
    <header className="header" role="banner">
      {/* Hamburger – visible on mobile */}
      <button
        id="sidebar-toggle-btn"
        className="header-menu-btn"
        onClick={onMenuToggle}
        aria-label={sidebarOpen ? 'Close sidebar' : 'Open sidebar'}
        aria-expanded={sidebarOpen}
        aria-controls="sidebar"
      >
        <span className="hamburger-line" />
        <span className="hamburger-line" />
        <span className="hamburger-line" />
      </button>

      {/* Centre brand (mobile only – desktop shows in sidebar) */}
      <div className="header-brand" aria-hidden="true">
        <span>🛡️</span>
        <span className="header-brand-name">PromptGuard</span>
      </div>

      {/* Status pill */}
      <div className="header-status" role="status" aria-label={`Backend status: ${isOnline ? 'online' : health === null ? 'connecting' : 'offline'}`}>
        <span className={`status-dot ${isOnline ? 'dot-online' : health === null ? 'dot-connecting' : 'dot-offline'}`} aria-hidden="true" />
        {health === null && <span className="status-label">Connecting…</span>}
        {health?.status === 'ok' && (
          <span className="status-label">
            Phase {health.phase} · Online
          </span>
        )}
        {health?.status === 'error' && (
          <span className="status-label status-error">Backend offline</span>
        )}
      </div>
    </header>
  )
}
