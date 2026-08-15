/** @typedef {import("./types.js").ContinuationOption} ContinuationOption */
/** @typedef {import("./types.js").TokenStep} TokenStep */

/**
 * Next-token options recorded after a chosen piece, if generation continued.
 *
 * @param {TokenStep[]} steps
 * @param {number} index
 * @returns {{ from: TokenStep | null, options: ContinuationOption[] }}
 */
export function continuationsFrom(steps, index) {
  const list = Array.isArray(steps) ? steps : [];
  const from = list[index] || null;
  const next = list[index + 1];
  if (!from) {
    return { from: null, options: [] };
  }
  if (!next) {
    return { from, options: [] };
  }

  const options = (next.alternatives || []).map((row) => ({
    token: row.token,
    tokenId: row.tokenId,
    probability: row.probability,
    chosen: isSamePiece(row, next),
  }));

  if (!options.some((row) => row.chosen)) {
    options.push({
      token: next.token,
      tokenId: next.tokenId,
      probability: next.probability,
      chosen: true,
    });
  }

  options.sort((a, b) => b.probability - a.probability);
  return { from, options };
}

/**
 * Keep writing through `fromIndex`, then take `option` as the next piece.
 *
 * @param {TokenStep[]} steps
 * @param {number} fromIndex
 * @param {Pick<ContinuationOption, "token" | "tokenId" | "probability">} option
 * @returns {TokenStep[]}
 */
export function branchSteps(steps, fromIndex, option) {
  const list = Array.isArray(steps) ? steps : [];
  const next = list[fromIndex + 1];
  const kept = list.slice(0, Math.max(0, fromIndex + 1));
  kept.push({
    token: option.token,
    tokenId: option.tokenId ?? 0,
    probability: option.probability,
    alternatives: next?.alternatives || [],
  });
  return kept;
}

/**
 * Opening plus kept tokens, matching what the page shows.
 *
 * @param {string} opening
 * @param {Array<Pick<TokenStep, "token">>} steps
 */
export function continuePrefix(opening, steps) {
  const head = String(opening || "").trim();
  const body = (Array.isArray(steps) ? steps : []).map((step) => step.token).join("");
  if (!head) {
    return body;
  }
  if (!body) {
    return head;
  }
  return `${head} ${body}`;
}

/**
 * @param {Pick<TokenStep, "token" | "tokenId">} row
 * @param {Pick<TokenStep, "token" | "tokenId">} step
 */
function isSamePiece(row, step) {
  if (row.tokenId != null && step.tokenId != null) {
    return Number(row.tokenId) === Number(step.tokenId);
  }
  return row.token === step.token;
}
