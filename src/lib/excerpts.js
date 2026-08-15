import {
  CHUNK_TARGET_CHARS,
  EXCERPT_COUNT,
  MAX_VOICE_CHARS,
  MIN_CHUNK_CHARS,
} from "./config.js";

/** @typedef {import("./types.js").Voice} Voice */

/**
 * Pick a handful of passages from different parts of the file.
 * The context window cannot hold a whole book, so this is what they imitate.
 *
 * @param {string} text
 * @param {{ title?: string }} [options]
 * @returns {Voice}
 */
export function prepareVoice(text, { title = "Your text" } = {}) {
  const normalized = String(text).replace(/\r\n/g, "\n").trim();
  const chunks = chunkDocument(normalized);
  const excerpts = pickSpread(chunks, EXCERPT_COUNT, MAX_VOICE_CHARS);

  return {
    title,
    wordCount: countWords(normalized),
    excerptCount: excerpts.length,
    excerptChars: excerpts.reduce((sum, excerpt) => sum + excerpt.length, 0),
    excerpts,
  };
}

/** @param {string} text */
function chunkDocument(text) {
  if (!text) {
    return [];
  }

  const paragraphs = text
    .split(/\n{2,}/)
    .map((paragraph) => paragraph.trim())
    .filter(Boolean);

  if (paragraphs.length === 0) {
    return [text.slice(0, MAX_VOICE_CHARS)];
  }

  const voiced = paragraphs.filter((paragraph) => !isWeakVoice(paragraph));
  const source = voiced.length >= 3 ? voiced : paragraphs;
  return packParagraphs(source);
}

/** @param {string[]} paragraphs */
function packParagraphs(paragraphs) {
  const chunks = [];
  let buffer = "";
  for (const paragraph of paragraphs) {
    const next = buffer ? `${buffer}\n\n${paragraph}` : paragraph;
    if (next.length >= CHUNK_TARGET_CHARS && buffer.length >= MIN_CHUNK_CHARS) {
      chunks.push(buffer);
      buffer = paragraph;
    } else {
      buffer = next;
    }
  }
  if (buffer.length >= MIN_CHUNK_CHARS) {
    chunks.push(buffer);
  } else if (chunks.length === 0 && buffer) {
    chunks.push(buffer.slice(0, MAX_VOICE_CHARS));
  }
  return chunks;
}

// Title pages, contents lists, and cast lists are a poor match for "voice".
/** @param {string} text */
function isWeakVoice(text) {
  const lines = text
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean);
  if (lines.length === 0) {
    return true;
  }
  if (text.length < 60 && lines.length <= 3) {
    return true;
  }
  if (/^SCENE\.\s/i.test(text) && text.length < 220) {
    return true;
  }

  const head = lines.slice(0, 8).join("\n");
  if (/^contents\b/i.test(lines[0]) || /\bdramatis person/i.test(head)) {
    return true;
  }
  if (lines.length <= 8 && text.length < 400 && /\bby\s+/i.test(text)) {
    return true;
  }

  const catalog = lines.filter((line) => /^(act\s+[ivxlcd]+\b|scene\s+[ivxlcd]+\.?)/i.test(line)).length;
  if (catalog >= 4) {
    return true;
  }

  const roles = lines.filter((line) => /^[A-Z][A-Z ’'-]{2,},/.test(line)).length;
  if (roles >= 3) {
    return true;
  }
  const shortLines = lines.filter((line) => line.length < 72).length;
  if (roles >= 2 && lines.length >= 4 && shortLines / lines.length >= 0.7) {
    return true;
  }

  return false;
}

/**
 * @param {string[]} chunks
 * @param {number} count
 * @param {number} budget
 */
function pickSpread(chunks, count, budget) {
  if (chunks.length === 0) {
    return [];
  }

  const n = Math.min(count, chunks.length);
  const indexes = new Set();
  if (n === 1) {
    indexes.add(0);
  } else {
    for (let i = 0; i < n; i++) {
      indexes.add(Math.round((i * (chunks.length - 1)) / (n - 1)));
    }
  }

  const picked = [...indexes].map((index) => chunks[index]);
  return fitBudget(picked, budget);
}

/**
 * @param {string[]} passages
 * @param {number} budget
 */
function fitBudget(passages, budget) {
  const out = [];
  let used = 0;
  for (const passage of passages) {
    if (used >= budget) {
      break;
    }
    const remaining = budget - used;
    if (passage.length <= remaining) {
      out.push(passage);
      used += passage.length;
      continue;
    }
    if (remaining >= MIN_CHUNK_CHARS) {
      out.push(trimToBreak(passage, remaining));
    }
    break;
  }
  return out;
}

/**
 * @param {string} text
 * @param {number} max
 */
function trimToBreak(text, max) {
  const slice = text.slice(0, max);
  const breakAt = Math.max(
    slice.lastIndexOf("\n"),
    slice.lastIndexOf(". "),
    slice.lastIndexOf("? "),
    slice.lastIndexOf("! ")
  );
  if (breakAt >= MIN_CHUNK_CHARS) {
    return slice.slice(0, breakAt + 1).trim();
  }
  return slice.trim();
}

/** @param {string} text */
function countWords(text) {
  const matches = text.match(/[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)*/g);
  return matches ? matches.length : 0;
}
