/**
 * Short label for a subword token in the walkthrough.
 * Leading and trailing spaces are omitted so quotes read as "twenty", not " twenty".
 *
 * @param {string | null | undefined} text
 */
export function formatToken(text) {
  if (text == null || text === "") {
    return "∅";
  }
  const { core } = splitTokenSpaces(text);
  if (/^ +$/.test(core)) {
    return "␣";
  }
  if (core === "\n" || core === "\r\n") {
    return "↵";
  }
  return core.replace(/\r\n/g, "\n").replace(/\n/g, "↵").replace(/\t/g, "⇥");
}

/**
 * Pulls edge spaces off a token for display. The original string is still the model piece.
 *
 * @param {string} token
 * @returns {{ leading: string, core: string, trailing: string }}
 */
export function splitTokenSpaces(token) {
  const leading = token.match(/^ +/)?.[0] ?? "";
  const rest = token.slice(leading.length);
  const trailing = rest.match(/ +$/)?.[0] ?? "";
  const core = rest.slice(0, rest.length - trailing.length);
  if (!core) {
    return { leading: "", core: token, trailing: "" };
  }
  return { leading, core, trailing };
}
