import React, { useState, useEffect } from 'react'
import { ChatProvider, useChat } from './context/ChatContext'
import Sidebar from './components/Sidebar/Sidebar'
import Header from './components/Header/Header'
import ChatWindow from './components/ChatWindow/ChatWindow'

/**
 * Inner layout — needs access to ChatContext (provided by parent).
 * Handles sidebar toggle state and ensures a session always exists
 * before the user types their first message.
 */
function AppLayout() {
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const { activeId, newSession } = useChat()

  // Create a default session on first load
  useEffect(() => {
    if (!activeId) newSession()
  }, []) // eslint-disable-line

  function toggleSidebar() {
    setSidebarOpen(prev => !prev)
  }

  function closeSidebar() {
    setSidebarOpen(false)
  }

  return (
    <div className="app-layout">
      {/* Mobile overlay */}
      <div
        className={`sidebar-overlay ${sidebarOpen ? 'visible' : ''}`}
        onClick={closeSidebar}
        aria-hidden="true"
      />

      {/* Sidebar */}
      <Sidebar
        id="sidebar"
        className={sidebarOpen ? 'sidebar--open' : ''}
        onClose={closeSidebar}
      />

      {/* Main content */}
      <div className="app-main">
        <Header onMenuToggle={toggleSidebar} sidebarOpen={sidebarOpen} />
        <ChatWindow />
      </div>
    </div>
  )
}

export default function App() {
  return (
    <ChatProvider>
      <AppLayout />
    </ChatProvider>
  )
}
