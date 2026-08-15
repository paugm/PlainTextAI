import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { describe, it } from "node:test";
import { fileURLToPath } from "node:url";
import { MAX_VOICE_CHARS } from "./config.js";
import { prepareVoice } from "./excerpts.js";

const root = join(dirname(fileURLToPath(import.meta.url)), "../..");

describe("prepareVoice", () => {
  it("picks a handful of passages under the memory budget", () => {
    const text = Array.from({ length: 80 }, (_, i) => `Paragraph ${i + 1}. ${"word ".repeat(80)}`).join("\n\n");
    const voice = prepareVoice(text, { title: "Test" });
    assert.equal(voice.title, "Test");
    assert.equal(voice.excerptCount, 5);
    assert.ok(voice.excerptChars <= MAX_VOICE_CHARS);
    assert.ok(voice.excerptChars > 12_000);
    assert.ok(voice.wordCount > 1000);
    assert.ok(voice.excerpts[0]?.includes("Paragraph 1"));
    assert.ok(voice.excerpts.at(-1)?.includes("Paragraph 80"));
  });

  it("skips Romeo and Juliet title, contents, and dramatis personae", () => {
    const text = readFileSync(join(root, "public/samples/romeo-and-juliet.txt"), "utf8");
    const voice = prepareVoice(text, { title: "Romeo and Juliet" });
    const joined = voice.excerpts.join("\n");
    assert.equal(voice.excerptCount, 5);
    assert.ok(voice.excerptChars <= MAX_VOICE_CHARS);
    assert.ok(voice.excerptChars > 15_000);
    assert.ok(voice.wordCount > 20_000);
    assert.doesNotMatch(joined, /Dramatis Person/i);
    assert.doesNotMatch(voice.excerpts[0] ?? "", /Franciscan|Prince of Verona|servant to Capulet/i);
    assert.match(voice.excerpts[0], /Two households/);
  });

  it("keeps Dracula's journal prose", () => {
    const text = readFileSync(join(root, "public/samples/dracula-chapters-i-v.txt"), "utf8");
    const voice = prepareVoice(text, { title: "Dracula" });
    assert.equal(voice.excerptCount, 5);
    assert.ok(voice.excerptChars > 15_000);
    assert.match(voice.excerpts.join("\n"), /Harker|Count|castle|journal/i);
  });
});
