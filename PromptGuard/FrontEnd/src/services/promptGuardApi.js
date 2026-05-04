/**
 * PromptGuard – API service layer
 *
 * All calls to the FastAPI backend are centralised here.
 * The base URL resolves from VITE_API_BASE_URL in .env or falls back to
 * the Vite dev proxy (empty string → relative).
 */

const BASE_URL = import.meta.env.VITE_API_BASE_URL ?? ''

/**
 * POST /api/v1/check_prompt
 *
 * @param {string} prompt – raw user prompt text
 * @returns {Promise<PromptResponse>} – structured PromptGuard response
 */
export async function checkPrompt(prompt) {
  const response = await fetch(`${BASE_URL}/api/v1/check_prompt`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ prompt }),
  })

  if (!response.ok) {
    let detail = `HTTP ${response.status}`
    try {
      const err = await response.json()
      detail = err.detail ?? JSON.stringify(err)
    } catch {
      // ignore parse errors
    }
    throw new Error(detail)
  }

  return response.json()
}

/**
 * GET /health
 *
 * @returns {Promise<HealthResponse>} – backend health/phase status
 */
export async function fetchHealth() {
  const response = await fetch(`${BASE_URL}/health`)
  if (!response.ok) throw new Error(`Health check failed: HTTP ${response.status}`)
  return response.json()
}
