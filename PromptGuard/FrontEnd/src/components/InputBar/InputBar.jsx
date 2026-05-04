import React, { useRef, useEffect, useState, useCallback } from 'react'
import './InputBar.css'

const MAX_ROWS = 8

export default function InputBar({ onSend, disabled }) {
  const [value, setValue] = useState('')
  const textareaRef = useRef(null)

  // Auto-resize textarea
  useEffect(() => {
    const ta = textareaRef.current
    if (!ta) return
    ta.style.height = 'auto'
    const lineH = parseInt(getComputedStyle(ta).lineHeight, 10) || 24
    const maxH = lineH * MAX_ROWS
    ta.style.height = Math.min(ta.scrollHeight, maxH) + 'px'
  }, [value])

  const handleKeyDown = useCallback((e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      submit()
    }
  }, [value, disabled]) // eslint-disable-line

  function submit() {
    const trimmed = value.trim()
    if (!trimmed || disabled) return
    onSend(trimmed)
    setValue('')
  }

  const charCount = value.length
  const isOverLimit = charCount > 4000

  return (
    <div className="input-bar-wrap">
      {isOverLimit && (
        <p className="input-limit-warning" role="alert">
          Prompt is very long ({charCount} chars). The API accepts any length but performance may vary.
        </p>
      )}
      <div className={`input-bar ${disabled ? 'input-bar--disabled' : ''}`}>
        <textarea
          id="prompt-input"
          ref={textareaRef}
          className="input-textarea"
          placeholder="Type a prompt to analyse… (Enter to send, Shift+Enter for newline)"
          value={value}
          onChange={e => setValue(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={disabled}
          rows={1}
          aria-label="Prompt input"
          aria-multiline="true"
          autoFocus
        />
        <div className="input-controls">
          <span className={`char-count ${isOverLimit ? 'char-over' : ''}`} aria-live="polite">
            {charCount > 0 ? charCount : ''}
          </span>
          <button
            id="send-btn"
            className="send-btn"
            onClick={submit}
            disabled={disabled || !value.trim()}
            aria-label="Send prompt"
            title="Send (Enter)"
          >
            {disabled ? (
              <span className="send-spinner" aria-hidden="true" />
            ) : (
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                <line x1="22" y1="2" x2="11" y2="13"/>
                <polygon points="22 2 15 22 11 13 2 9 22 2"/>
              </svg>
            )}
          </button>
        </div>
      </div>
      <p className="input-hint">
        PromptGuard analyses prompts for adversarial intent using a 4-layer detection pipeline.
      </p>
    </div>
  )
}
