/**
 * PromptGuard – usePromptGuard hook
 *
 * Orchestrates sending a user prompt through the PromptGuard API,
 * adding optimistic user message, streaming typing animation for
 * the assistant reply, and committing the final API response.
 */

import { useCallback, useRef } from 'react'
import { useChat } from '../context/ChatContext'
import { checkPrompt } from '../services/promptGuardApi'
import { generateId } from '../utils/formatters'

/** Characters typed per interval tick in the typewriter animation */
const TYPING_CHUNK = 3
/** Interval (ms) between each typewriter tick */
const TYPING_INTERVAL_MS = 18

/** Emoji map for each decision outcome */
const DECISION_EMOJI = {
  ALLOW:    '🟢',
  BLOCK:    '🔴',
  SANITIZE: '🟡',
}

/**
 * Build the assistant message text from the PromptGuard API response.
 *
 * Security rule: this function must NEVER include raw numeric scores
 * (risk_score, adversarial_probability, similarity_score, logits, etc.).
 * Only decision, risk_level, and the structured explanation are surfaced.
 */
function buildAssistantContent(apiResponse) {
  const { decision, risk_level, explanation, router_result } = apiResponse
  const emoji = DECISION_EMOJI[decision] ?? '❓'

  let text = `${emoji} **Decision: ${decision}**\n\n`

  if (risk_level) {
    text += `**Risk Level:** ${risk_level}\n\n`
  }

  if (explanation?.reasons?.length) {
    text += `### ⚠️ Why this was flagged:\n`
    explanation.reasons.forEach(r => {
      text += `- **${r.category}**: ${r.message}\n`
    })
    text += '\n'
  }

  if (explanation?.suggestions?.length && decision !== 'ALLOW') {
    text += `### 💡 How to improve your prompt:\n`
    explanation.suggestions.forEach(s => {
      text += `- ${s}\n`
    })
    text += '\n'
  }

  // Only show sanitized text — never the internal reason string
  if (router_result?.sanitized_text) {
    text += `### ✂️ Sanitized Prompt:\n\`\`\`\n${router_result.sanitized_text}\n\`\`\`\n`
  }

  return text.trim()
}

export function usePromptGuard() {
  const { activeId, activeSession, newSession, addMessage, updateMessage } = useChat()
  const abortRef = useRef(null)

  const sendPrompt = useCallback(async (promptText) => {
    // Ensure there's an active session
    let sessionId = activeId
    if (!sessionId) {
      newSession()
      // newSession is sync but state updates async; we rely on the next tick
      // Instead we create the session ID ourselves then dispatch
      // This is handled by the reducer — activeId will be the new session.
      // Re-read after newSession via a small workaround: use a pending flag.
      // Simplest fix: call newSession and rely on useEffect in App to catch it.
      // We pass sessionId as a closure, but it won't be set yet.
      // Solution: generate ID here and pass it — but newSession generates its own.
      // For simplicity, if there's no session yet let App.jsx call newSession first.
      return
    }

    // ── 1. Optimistic user message ───────────────────────────────────────────
    const userMsgId = generateId()
    addMessage(sessionId, {
      id: userMsgId,
      role: 'user',
      content: promptText,
      timestamp: new Date().toISOString(),
    })

    // ── 2. Placeholder assistant message (loading) ───────────────────────────
    const assistantMsgId = generateId()
    addMessage(sessionId, {
      id: assistantMsgId,
      role: 'assistant',
      content: '',
      timestamp: new Date().toISOString(),
      isStreaming: true,
      apiResponse: null,
      error: null,
    })

    // ── 3. Call API ───────────────────────────────────────────────────────────
    let apiResponse
    try {
      apiResponse = await checkPrompt(promptText)
    } catch (err) {
      updateMessage(sessionId, assistantMsgId, {
        content: `❌ **Error:** ${err.message}`,
        isStreaming: false,
        error: err.message,
      })
      return
    }

    // ── 4. Typewriter animation ───────────────────────────────────────────────
    const fullText = buildAssistantContent(apiResponse)
    let displayed = ''
    let index = 0

    // Cancel any previous animation
    if (abortRef.current) clearInterval(abortRef.current)

    abortRef.current = setInterval(() => {
      if (index >= fullText.length) {
        clearInterval(abortRef.current)
        abortRef.current = null
        updateMessage(sessionId, assistantMsgId, {
          content: fullText,
          isStreaming: false,
          apiResponse,
        })
        return
      }
      displayed += fullText.slice(index, index + TYPING_CHUNK)
      index += TYPING_CHUNK
      updateMessage(sessionId, assistantMsgId, {
        content: displayed,
        isStreaming: true,
        apiResponse,
      })
    }, TYPING_INTERVAL_MS)
  }, [activeId, activeSession, newSession, addMessage, updateMessage])

  return { sendPrompt }
}
