import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { escapeHtml } from "./html.js";

describe("escapeHtml", () => {
  it("escapes markup", () => {
    assert.equal(escapeHtml(`<b>a&b</b>`), "&lt;b&gt;a&amp;b&lt;/b&gt;");
  });
});
