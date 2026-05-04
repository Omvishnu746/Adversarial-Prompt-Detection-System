/**
 * PromptGuard – Chat Context
 *
 * Single source of truth for all chat sessions and the active session.
 * Uses React Context + useReducer — no extra libraries needed.
 */

import React, { createContext, useContext, useReducer, useCallback } from 'react'
import { generateId } from '../utils/formatters'

// ─── Shape ────────────────────────────────────────────────────────────────────
// Session  : { id, title, createdAt, messages: Message[] }
// Message  : { id, role: 'user'|'assistant', content, timestamp, apiResponse?, error?, isStreaming? }

const initialState = {
  sessions: [],      // Session[]
  activeId: null,    // string | null
}

// ─── Reducer ──────────────────────────────────────────────────────────────────
function chatReducer(state, action) {
  switch (action.type) {

    case 'NEW_SESSION': {
      const session = {
        id: generateId(),
        title: 'New Chat',
        createdAt: new Date().toISOString(),
        messages: [],
      }
      return {
        sessions: [session, ...state.sessions],
        activeId: session.id,
      }
    }

    case 'SET_ACTIVE': {
      return { ...state, activeId: action.id }
    }

    case 'DELETE_SESSION': {
      const sessions = state.sessions.filter(s => s.id !== action.id)
      const activeId =
        state.activeId === action.id
          ? (sessions[0]?.id ?? null)
          : state.activeId
      return { sessions, activeId }
    }

    case 'ADD_MESSAGE': {
      return {
        ...state,
        sessions: state.sessions.map(s =>
          s.id !== action.sessionId
            ? s
            : {
                ...s,
                title: s.messages.length === 0
                  ? action.message.content.slice(0, 40) + (action.message.content.length > 40 ? '…' : '')
                  : s.title,
                messages: [...s.messages, action.message],
              }
        ),
      }
    }

    case 'UPDATE_MESSAGE': {
      return {
        ...state,
        sessions: state.sessions.map(s =>
          s.id !== action.sessionId
            ? s
            : {
                ...s,
                messages: s.messages.map(m =>
                  m.id === action.messageId ? { ...m, ...action.patch } : m
                ),
              }
        ),
      }
    }

    default:
      return state
  }
}

// ─── Context ──────────────────────────────────────────────────────────────────
const ChatContext = createContext(null)

export function ChatProvider({ children }) {
  const [state, dispatch] = useReducer(chatReducer, initialState)

  const activeSession = state.sessions.find(s => s.id === state.activeId) ?? null

  const newSession = useCallback(() => {
    dispatch({ type: 'NEW_SESSION' })
  }, [])

  const setActive = useCallback((id) => {
    dispatch({ type: 'SET_ACTIVE', id })
  }, [])

  const deleteSession = useCallback((id) => {
    dispatch({ type: 'DELETE_SESSION', id })
  }, [])

  const addMessage = useCallback((sessionId, message) => {
    dispatch({ type: 'ADD_MESSAGE', sessionId, message })
  }, [])

  const updateMessage = useCallback((sessionId, messageId, patch) => {
    dispatch({ type: 'UPDATE_MESSAGE', sessionId, messageId, patch })
  }, [])

  return (
    <ChatContext.Provider value={{
      sessions: state.sessions,
      activeId: state.activeId,
      activeSession,
      newSession,
      setActive,
      deleteSession,
      addMessage,
      updateMessage,
    }}>
      {children}
    </ChatContext.Provider>
  )
}

export function useChat() {
  const ctx = useContext(ChatContext)
  if (!ctx) throw new Error('useChat must be used within <ChatProvider>')
  return ctx
}
