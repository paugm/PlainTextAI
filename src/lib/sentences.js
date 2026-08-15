// Abbreviations that end with a period but do not end a sentence.
const ABBREVIATIONS = new Set([
  "mr",
  "mrs",
  "ms",
  "dr",
  "prof",
  "sr",
  "jr",
  "vs",
  "etc",
  "st",
]);

const TRAILING_CLOSERS = /[)"'”’\]]+$/u;

/**
 * True when `text` ends on a real sentence rather than an abbreviation like Mr.
 *
 * @param {string} text
 */
export function endsWithSentence(text) {
  const trimmed = String(text)
    .replace(TRAILING_CLOSERS, "")
    .replace(/\s+$/u, "");
  if (!trimmed) {
    return false;
  }
  if (/[.]{3}$/u.test(trimmed) || /…$/u.test(trimmed)) {
    return true;
  }
  if (/[!?]$/u.test(trimmed)) {
    return true;
  }
  if (!/\.$/u.test(trimmed)) {
    return false;
  }

  const word = trimmed.match(/([^\s.!?]+)[.]$/u)?.[1] ?? "";
  if (/^\d+$/u.test(word)) {
    return true;
  }
  const letters = word.replace(/[^A-Za-z]/gu, "");
  if (letters.length <= 1) {
    return false;
  }
  return !ABBREVIATIONS.has(letters.toLowerCase());
}

/**
 * Drops a trailing fragment after the last sentence end.
 *
 * @param {string} text
 */
export function trimToLastSentence(text) {
  const value = String(text);
  if (endsWithSentence(value)) {
    return value.replace(/\s+$/u, "");
  }
  const matches = [...value.matchAll(/[.!?…]["'”’)\]]*/gu)];
  for (let i = matches.length - 1; i >= 0; i--) {
    const match = matches[i];
    const prefix = value.slice(0, (match.index ?? 0) + match[0].length);
    if (endsWithSentence(prefix)) {
      return prefix;
    }
  }
  return value;
}

/**
 * Keeps generated steps through the last sentence-ending token.
 *
 * @template {{ token: string }} T
 * @param {T[]} steps
 * @returns {{ text: string, steps: T[] }}
 */
export function clipStepsToSentence(steps) {
  const list = Array.isArray(steps) ? steps : [];
  let text = "";
  let lastGood = -1;
  for (let i = 0; i < list.length; i++) {
    text += list[i]?.token ?? "";
    if (endsWithSentence(text)) {
      lastGood = i;
    }
  }
  if (lastGood < 0) {
    return { text, steps: list };
  }
  const clipped = list.slice(0, lastGood + 1);
  return {
    text: clipped.map((step) => step.token).join(""),
    steps: clipped,
  };
}
