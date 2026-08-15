/**
 * Shared limits for the page, the worker, and the tests.
 */

export const MODEL_ID = "HuggingFaceTB/SmolLM2-360M-Instruct";
export const MODEL_LABEL = "SmolLM2 360M";
export const DEFAULT_OPENING = "The night had only just begun,";

export const MAX_FILE_SIZE = 5 * 1024 * 1024;
export const MIN_FILE_LENGTH = 100;
/** Stop once a sentence ends, after at least this many new tokens. */
export const MIN_NEW_TOKENS = 72;
export const MAX_NEW_TOKENS = 192;
export const DEFAULT_TEMPERATURE = 1.0;
export const TOP_K = 50;
/** Alternatives shown on the walkthrough graph. */
export const DISPLAY_K = 8;
export const REPETITION_PENALTY = 1.12;
export const NO_REPEAT_NGRAM_SIZE = 3;

export const EXCERPT_COUNT = 5;
/** How much of the source file is kept as style excerpts. */
export const MAX_VOICE_CHARS = 20_000;
export const CHUNK_TARGET_CHARS = 4000;
export const MIN_CHUNK_CHARS = 80;

export const EXPLANATION_DELAY = 700;
export const TOAST_MS = 5000;
