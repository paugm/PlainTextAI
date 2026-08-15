import assert from "node:assert/strict";
import { describe, it } from "node:test";
import {
  assertNever,
  CancelledError,
  workerAbortIn,
  workerCompleteOut,
  workerErrorOut,
  workerGenerateIn,
  workerLoadIn,
  workerProgressOut,
  workerReadyOut,
  workerTokenOut,
  WORKER_IN,
  WORKER_OUT,
} from "./worker-protocol.js";

describe("worker protocol", () => {
  it("uses shared type strings instead of ad-hoc literals", () => {
    assert.equal(workerLoadIn().type, WORKER_IN.LOAD);
    assert.equal(workerAbortIn().type, WORKER_IN.ABORT);
    assert.equal(
      workerGenerateIn({
        id: 1,
        messages: [],
        prefix: "The night",
        temperature: 1,
        maxNewTokens: 8,
      }).type,
      WORKER_IN.GENERATE
    );
    assert.equal(workerProgressOut().type, WORKER_OUT.PROGRESS);
    assert.equal(workerReadyOut({ device: "wasm", dtype: "q4" }).type, WORKER_OUT.READY);
    assert.equal(
      workerTokenOut({
        id: 1,
        step: { token: " night", tokenId: 1, probability: 0.4, alternatives: [] },
      }).type,
      WORKER_OUT.TOKEN
    );
    assert.equal(
      workerCompleteOut({ id: 1, text: " night", steps: [] }).type,
      WORKER_OUT.COMPLETE
    );
    assert.equal(workerErrorOut("nope").type, WORKER_OUT.ERROR);
  });

  it("marks user aborts as CancelledError", () => {
    const error = new CancelledError();
    assert.equal(error.cancelled, true);
    assert.ok(error instanceof Error);
  });

  it("assertNever throws on a value that escaped the union", () => {
    assert.throws(() => assertNever(/** @type {never} */ ("mystery")));
  });
});
