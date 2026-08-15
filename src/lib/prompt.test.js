import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { buildMessages, withContinuePrefix, withOpeningPrefix } from "./prompt.js";

describe("buildMessages", () => {
  it("asks the model to continue an opening, not answer a question", () => {
    const messages = buildMessages(["Two households, both alike in dignity"]);
    const [system, user] = messages;
    assert.ok(system);
    assert.ok(user);
    assert.equal(system.role, "system");
    assert.match(system.content, /Do not answer questions/i);
    assert.match(system.content, /complete sentence/i);
    assert.equal(user.role, "user");
    assert.match(user.content, /Two households/);
    assert.match(user.content, /Continue the opening/);
  });
});

describe("withOpeningPrefix", () => {
  it("puts the opening at the end of the assistant prompt", () => {
    assert.equal(withOpeningPrefix("<|im_start|>assistant\n", "The night had only just begun,"), "<|im_start|>assistant\nThe night had only just begun, ");
    assert.equal(withOpeningPrefix("prompt", ""), "prompt");
  });
});

describe("withContinuePrefix", () => {
  it("appends kept writing without adding another trailing space", () => {
    assert.equal(
      withContinuePrefix("<|im_start|>assistant\n", "The night had only just begun,  and the"),
      "<|im_start|>assistant\nThe night had only just begun,  and the"
    );
  });
});
