import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { rankLogits } from "./sampling.js";

describe("rankLogits", () => {
  it("softmaxes the top-k logits and reports the chosen token", () => {
    const logits = new Float32Array(20);
    logits[3] = 5;
    logits[8] = 3;
    logits[1] = 1;
    const ranked = rankLogits(logits, 3, 5, 3);
    assert.ok(ranked.chosenProbability > 0.5);
    const first = ranked.alternatives[0];
    assert.ok(first);
    assert.equal(first.id, 3);
    assert.equal(ranked.alternatives.length, 3);
    const sum = ranked.alternatives.reduce((total, row) => total + row.probability, 0);
    assert.ok(sum > 0.9 && sum <= 1.0001);
  });

  it("still includes the chosen token if it falls outside the displayed set", () => {
    const logits = new Float32Array(10);
    for (let i = 0; i < 10; i++) {
      logits[i] = 10 - i;
    }
    const ranked = rankLogits(logits, 9, 10, 3);
    assert.equal(ranked.alternatives.length, 3);
    assert.ok(ranked.alternatives.some((row) => row.id === 9));
  });
});
