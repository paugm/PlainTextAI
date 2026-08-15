import { DISPLAY_K, TOP_K } from "./config.js";

/**
 * Rank next-token logits the same way Transformers.js samples:
 * keep the top-k logits, then softmax.
 *
 * @param {ArrayLike<number>} logits
 * @param {number} chosenId
 * @param {number} [topK]
 * @param {number} [displayK]
 * @returns {{ chosenProbability: number, alternatives: Array<{ id: number, probability: number }> }}
 */
export function rankLogits(logits, chosenId, topK = TOP_K, displayK = DISPLAY_K) {
  const ranked = topKEntries(logits, topK);
  const probabilities = softmax(ranked.map((entry) => entry.logit));
  const rows = ranked.map((entry, i) => ({
    id: entry.id,
    probability: probabilities[i],
  }));
  rows.sort((a, b) => b.probability - a.probability);

  const chosen = rows.find((row) => row.id === chosenId);
  const chosenProbability = chosen ? chosen.probability : 0;

  let alternatives = rows.slice(0, displayK);
  if (chosen && !alternatives.some((row) => row.id === chosenId)) {
    alternatives = [...alternatives.slice(0, displayK - 1), chosen];
  }

  return { chosenProbability, alternatives };
}

/**
 * @param {ArrayLike<number>} logits
 * @param {number} k
 */
function topKEntries(logits, k) {
  const entries = [];
  for (let id = 0; id < logits.length; id++) {
    const logit = logits[id];
    if (Number.isFinite(logit)) {
      entries.push({ id, logit });
    }
  }
  entries.sort((a, b) => b.logit - a.logit);
  return entries.slice(0, k);
}

/** @param {number[]} values */
function softmax(values) {
  let max = -Infinity;
  for (const value of values) {
    if (value > max) {
      max = value;
    }
  }
  const out = values.map((value) => Math.exp(value - max));
  const sum = out.reduce((total, value) => total + value, 0);
  if (sum <= 0) {
    const uniform = values.length ? 1 / values.length : 0;
    return out.map(() => uniform);
  }
  return out.map((value) => value / sum);
}
