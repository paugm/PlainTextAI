import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { branchSteps, continuePrefix, continuationsFrom } from "./continuations.js";

describe("continuationsFrom", () => {
  it("returns next-piece options from the following step", () => {
    const steps = [
      { token: " night", tokenId: 1, probability: 0.4, alternatives: [] },
      {
        token: " had",
        tokenId: 2,
        probability: 0.5,
        alternatives: [
          { token: " had", tokenId: 2, probability: 0.5 },
          { token: " was", tokenId: 3, probability: 0.2 },
          { token: " air", tokenId: 4, probability: 0.1 },
        ],
      },
    ];
    const result = continuationsFrom(steps, 0);
    assert.ok(result.from);
    assert.equal(result.from.token, " night");
    assert.equal(result.options.length, 3);
    assert.equal(result.options[0]?.token, " had");
    assert.equal(result.options[0]?.chosen, true);
    assert.equal(result.options[1]?.chosen, false);
  });

  it("has no options when the model stopped after that piece", () => {
    const steps = [{ token: " end", tokenId: 9, probability: 0.2, alternatives: [] }];
    const result = continuationsFrom(steps, 0);
    assert.ok(result.from);
    assert.equal(result.from.token, " end");
    assert.equal(result.options.length, 0);
  });

  it("marks the newly chosen piece after a branch", () => {
    const steps = [
      { token: " night", tokenId: 1, probability: 0.4, alternatives: [] },
      {
        token: " had",
        tokenId: 2,
        probability: 0.5,
        alternatives: [
          { token: " had", tokenId: 2, probability: 0.5 },
          { token: " was", tokenId: 3, probability: 0.2 },
        ],
      },
      { token: " only", tokenId: 4, probability: 0.3, alternatives: [] },
    ];
    const branched = branchSteps(steps, 0, {
      token: " was",
      tokenId: 3,
      probability: 0.2,
    });
    assert.equal(branched.length, 2);
    assert.equal(branched[1]?.token, " was");
    const result = continuationsFrom(branched, 0);
    assert.equal(result.options.find((row) => row.tokenId === 3)?.chosen, true);
    assert.equal(result.options.find((row) => row.tokenId === 2)?.chosen, false);
  });
});

describe("continuePrefix", () => {
  it("keeps the opening and tokens up to the newly chosen piece", () => {
    const opening = "The night had only just begun,";
    const steps = [
      { token: " and" },
      { token: " the" },
      { token: " was" },
    ];
    assert.equal(continuePrefix(opening, steps), "The night had only just begun,  and the was");
    assert.equal(continuePrefix(opening, []), "The night had only just begun,");
  });
});
