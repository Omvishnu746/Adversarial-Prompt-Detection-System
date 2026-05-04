import React from 'react'
import { useChat } from '../../context/ChatContext'
import { formatTime } from '../../utils/formatters'
import './Sidebar.css'

export default function Sidebar({ onClose }) {
  const { sessions, activeId, newSession, setActive, deleteSession } = useChat()

  function handleNew() {
    newSession()
    if (onClose) onClose()
  }

  function handleSelect(id) {
    setActive(id)
    if (onClose) onClose()
  }

  return (
    <aside className="sidebar" aria-label="Chat history">
      {/* Logo + brand */}
      <div className="sidebar-header">
        <div className="sidebar-brand">
          <span className="sidebar-shield" aria-hidden="true">🛡️</span>
          <span className="sidebar-title">PromptGuard</span>
        </div>
        <button
          id="new-chat-btn"
          className="new-chat-btn"
          onClick={handleNew}
          title="New Chat"
          aria-label="Start new chat"
        >
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></svg>
          New Chat
        </button>
      </div>

      {/* Session list */}
      <nav className="sidebar-nav" aria-label="Chat sessions">
        {sessions.length === 0 && (
          <p className="sidebar-empty">No chats yet. Start one!</p>
        )}
        {sessions.map(session => (
          <div
            key={session.id}
            className={`sidebar-item ${session.id === activeId ? 'active' : ''}`}
            role="button"
            tabIndex={0}
            aria-current={session.id === activeId ? 'page' : undefined}
            onClick={() => handleSelect(session.id)}
            onKeyDown={e => (e.key === 'Enter' || e.key === ' ') && handleSelect(session.id)}
          >
            <div className="sidebar-item-icon" aria-hidden="true">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>
            </div>
            <div className="sidebar-item-body">
              <span className="sidebar-item-title">{session.title}</span>
              <span className="sidebar-item-time">{formatTime(session.createdAt)}</span>
            </div>
            <button
              className="sidebar-delete-btn"
              title="Delete chat"
              aria-label={`Delete chat: ${session.title}`}
              onClick={e => { e.stopPropagation(); deleteSession(session.id) }}
            >
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
            </button>
          </div>
        ))}
      </nav>

      {/* Footer */}
      <div className="sidebar-footer">
        <span className="sidebar-version">PromptGuard v0.4.0</span>
      </div>
    </aside>
  )
}
