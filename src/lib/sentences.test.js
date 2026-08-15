import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { clipStepsToSentence, endsWithSentence, trimToLastSentence } from "./sentences.js";

describe("endsWithSentence", () => {
  it("accepts period, question, exclamation, and ellipsis", () => {
    assert.equal(endsWithSentence("The night had only just begun."), true);
    assert.equal(endsWithSentence("Who is there?"), true);
    assert.equal(endsWithSentence("Look!"), true);
    assert.equal(endsWithSentence("And then…"), true);
    assert.equal(endsWithSentence("And then..."), true);
    assert.equal(endsWithSentence('He said, "Enough."'), true);
    assert.equal(endsWithSentence("It was 2025."), true);
  });

  it("rejects unfinished lines and abbreviations", () => {
    assert.equal(endsWithSentence("years to come ("), false);
    assert.equal(endsWithSentence("Mr."), false);
    assert.equal(endsWithSentence("U.S."), false);
    assert.equal(endsWithSentence("See Dr."), false);
    assert.equal(endsWithSentence("A."), false);
    assert.equal(endsWithSentence(""), false);
  });
});

describe("trimToLastSentence", () => {
  it("drops a trailing fragment after the last real sentence end", () => {
    const text =
      "much less how other days felt in theirs. However 21 st century feels better. As we watch the first men put feet on Mars in  years to come (";
    assert.equal(trimToLastSentence(text), "much less how other days felt in theirs. However 21 st century feels better.");
  });
});

describe("clipStepsToSentence", () => {
  it("keeps tokens through the last sentence-ending piece", () => {
    const steps = ["Hello", " there", ".", " And", " then", " ("].map((token) => ({ token }));
    const clipped = clipStepsToSentence(steps);
    assert.equal(clipped.text, "Hello there.");
    assert.equal(clipped.steps.length, 3);
  });
});
