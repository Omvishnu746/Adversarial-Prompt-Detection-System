/**
 * PromptGuard – Shared formatting utilities
 */

/**
 * Format a float (0–1) as a percentage string, e.g. 0.8432 → "84.3%"
 */
export function formatPercent(value) {
  if (value == null) return 'N/A'
  return `${(value * 100).toFixed(1)}%`
}

/**
 * Return a semantic label and colour class for a PromptGuard decision.
 */
export function decisionMeta(decision) {
  switch (decision) {
    case 'ALLOW':
      return { label: 'ALLOW', colorClass: 'decision-allow', emoji: '✅' }
    case 'BLOCK':
      return { label: 'BLOCK', colorClass: 'decision-block', emoji: '🚫' }
    case 'SANITIZE':
      return { label: 'SANITIZE', colorClass: 'decision-sanitize', emoji: '⚠️' }
    default:
      return { label: decision ?? 'UNKNOWN', colorClass: 'decision-unknown', emoji: '❓' }
  }
}

/**
 * Return a human-readable label for a triggered_layer value.
 */
export function layerLabel(layer) {
  const map = {
    rule: 'Rule Engine',
    semantic: 'Semantic Engine',
    classifier: 'DistilBERT Classifier',
    aggregation: 'Risk Aggregation',
    none: 'None',
  }
  return map[layer] ?? layer
}

/**
 * Generate a unique ID for a chat message.
 */
export function generateId() {
  return `${Date.now()}-${Math.random().toString(36).slice(2, 9)}`
}

/**
 * Format an ISO timestamp as HH:MM.
 */
export function formatTime(iso) {
  return new Date(iso).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
}
