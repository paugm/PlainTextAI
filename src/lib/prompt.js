const SYSTEM_PROMPT = [
  "You are not a chatbot. Do not answer questions.",
  "Keep writing prose in the style of the passages the user provides.",
  "Match their vocabulary, rhythm, spelling, and attitude.",
  "Continue from the opening as the same piece of writing.",
  "Finish on a complete sentence. Do not stop mid-sentence or after an opening parenthesis.",
  "Do not mention these instructions.",
].join(" ");

/** @typedef {import("./types.js").ChatMessage} ChatMessage */

/**
 * Chat messages that tell the model to keep writing in the excerpts' style.
 *
 * @param {string[]} excerpts
 * @returns {ChatMessage[]}
 */
export function buildMessages(excerpts) {
  const voice = excerpts.join("\n\n----\n\n");
  return [
    { role: "system", content: SYSTEM_PROMPT },
    {
      role: "user",
      content: `Write like this:\n\n${voice}\n\n----\n\nContinue the opening. Do not answer it. Keep writing in this style.`,
    },
  ];
}

/**
 * Append the user's opening onto the assistant turn so the model continues it.
 *
 * @param {string} chatPrompt
 * @param {string} opening
 */
export function withOpeningPrefix(chatPrompt, opening) {
  const text = String(opening || "").trim();
  if (!text) {
    return chatPrompt;
  }
  return `${chatPrompt}${text} `;
}

/**
 * Prefix already includes the opening and kept tokens; do not add another space.
 *
 * @param {string} chatPrompt
 * @param {string} prefix
 */
export function withContinuePrefix(chatPrompt, prefix) {
  return `${chatPrompt}${String(prefix || "")}`;
}
