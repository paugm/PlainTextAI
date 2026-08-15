import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { formatToken, splitTokenSpaces } from "./tokens.js";

describe("formatToken", () => {
  it("marks spaces and newlines so they are visible in the walkthrough", () => {
    assert.equal(formatToken(" "), "␣");
    assert.equal(formatToken("\n"), "↵");
    assert.equal(formatToken("love"), "love");
    assert.equal(formatToken(""), "∅");
  });

  it("omits leading and trailing spaces in quoted labels", () => {
    assert.equal(formatToken(" twenty"), "twenty");
    assert.equal(formatToken(" one"), "one");
    assert.equal(formatToken("word "), "word");
  });
});

describe("splitTokenSpaces", () => {
  it("keeps a space-only token as the core so it still has something to show", () => {
    assert.deepEqual(splitTokenSpaces(" "), { leading: "", core: " ", trailing: "" });
  });

  it("splits a leading space off the visible word", () => {
    assert.deepEqual(splitTokenSpaces(" twenty"), {
      leading: " ",
      core: "twenty",
      trailing: "",
    });
  });
});
