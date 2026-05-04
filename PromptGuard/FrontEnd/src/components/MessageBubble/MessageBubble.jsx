import React, { useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter'
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism'
import { formatTime, decisionMeta, formatPercent } from '../../utils/formatters'
import './MessageBubble.css'

function CopyButton({ text }) {
  const [copied, setCopied] = useState(false)
  function handleCopy() {
    navigator.clipboard.writeText(text).then(() => {
      setCopied(true)
      setTimeout(() => setCopied(false), 1800)
    })
  }
  return (
    <button className="copy-btn" onClick={handleCopy} title="Copy to clipboard" aria-label="Copy message">
      {copied
        ? <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><polyline points="20 6 9 17 4 12"/></svg>
        : <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>
      }
      {copied ? 'Copied!' : 'Copy'}
    </button>
  )
}

function RiskBar({ score }) {
  const pct = Math.round((score ?? 0) * 100)
  const color = pct >= 70 ? 'var(--color-block)' : pct >= 40 ? 'var(--color-sanitize)' : 'var(--color-allow)'
  return (
    <div className="risk-bar-wrap" aria-label={`Risk score ${pct}%`}>
      <div className="risk-bar-track">
        <div className="risk-bar-fill" style={{ width: `${pct}%`, background: color }} />
      </div>
      <span className="risk-bar-label" style={{ color }}>{pct}%</span>
    </div>
  )
}

function DecisionBadge({ decision }) {
  if (!decision) return null
  const meta = decisionMeta(decision)
  return (
    <span className={`decision-badge ${meta.colorClass}`} aria-label={`Decision: ${meta.label}`}>
      {meta.emoji} {meta.label}
    </span>
  )
}

/** Custom markdown components for react-markdown */
const markdownComponents = {
  code({ node, inline, className, children, ...props }) {
    const match = /language-(\w+)/.exec(className || '')
    return !inline && match ? (
      <SyntaxHighlighter
        style={oneDark}
        language={match[1]}
        PreTag="div"
        customStyle={{ borderRadius: '8px', fontSize: '0.82rem', margin: '8px 0' }}
        {...props}
      >
        {String(children).replace(/\n$/, '')}
      </SyntaxHighlighter>
    ) : (
      <code className="inline-code" {...props}>{children}</code>
    )
  },
  hr() {
    return <hr className="md-hr" />
  },
}

export default function MessageBubble({ message }) {
  const isUser = message.role === 'user'
  const { apiResponse, isStreaming, error, content, timestamp } = message

  return (
    <div className={`message-row ${isUser ? 'message-row--user' : 'message-row--assistant'}`}>
      {/* Avatar */}
      <div className={`message-avatar ${isUser ? 'avatar-user' : 'avatar-assistant'}`} aria-hidden="true">
        {isUser ? '👤' : '🛡️'}
      </div>

      <div className="message-body">
        {/* Role label */}
        <div className="message-meta">
          <span className="message-role">{isUser ? 'You' : 'PromptGuard'}</span>
          <span className="message-time">{formatTime(timestamp)}</span>
          {/* Decision badge in header for assistant */}
          {!isUser && apiResponse && (
            <DecisionBadge decision={apiResponse.decision} />
          )}
        </div>

        {/* Bubble */}
        <div className={`message-bubble ${isUser ? 'bubble-user' : 'bubble-assistant'} ${error ? 'bubble-error' : ''}`}>
          {/* Risk bar for assistant messages with a response */}
          {!isUser && apiResponse && !isStreaming && (
            <RiskBar score={apiResponse.risk_score} />
          )}

          {/* Content */}
          <div className="message-content">
            {isUser ? (
              <p className="user-text">{content}</p>
            ) : (
              <ReactMarkdown
                remarkPlugins={[remarkGfm]}
                components={markdownComponents}
              >
                {content || ''}
              </ReactMarkdown>
            )}
            {/* Typing cursor */}
            {isStreaming && <span className="typing-cursor" aria-hidden="true" />}
          </div>
        </div>

        {/* Actions */}
        {!isStreaming && content && (
          <div className="message-actions">
            <CopyButton text={content} />
          </div>
        )}
      </div>
    </div>
  )
}
