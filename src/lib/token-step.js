import { rankLogits } from "./sampling.js";

/** @typedef {import("./types.js").TokenStep} TokenStep */

/**
 * Turn a sampled token id and its logits into a walkthrough step.
 *
 * @param {{ decode: (ids: number[], options?: { skip_special_tokens?: boolean }) => string }} tokenizer
 * @param {ArrayLike<number> | null | undefined} logits
 * @param {number} tokenId
 * @returns {TokenStep | null}
 */
export function tokenStep(tokenizer, logits, tokenId) {
  const id = Number(tokenId);
  const piece = tokenizer.decode([id], { skip_special_tokens: true });
  if (!piece) {
    return null;
  }
  const ranked = logits ? rankLogits(logits, id) : { chosenProbability: 0, alternatives: [] };
  return {
    token: piece,
    tokenId: id,
    probability: ranked.chosenProbability,
    alternatives: ranked.alternatives.map((row) => ({
      token: tokenizer.decode([row.id], { skip_special_tokens: true }) || String(row.id),
      tokenId: row.id,
      probability: row.probability,
    })),
  };
}
