import { LogitsProcessor } from "@huggingface/transformers";

/** Copies the last-token logits after Transformers.js has applied its processors. Worker-only. */
export class CaptureLogitsProcessor extends LogitsProcessor {
  /** @type {Float32Array | null} */
  pending = null;

  constructor() {
    super();
  }

  /**
   * Transformers.js calls this hook; the name must stay `_call`.
   *
   * @param {bigint[][]} _inputIds
   * @param {import("@huggingface/transformers").Tensor} logits
   */
  _call(_inputIds, logits) {
    const data = logits.data;
    const vocab = logits.dims.at(-1);
    if (
      typeof vocab !== "number" ||
      !data ||
      typeof data !== "object" ||
      !("subarray" in data) ||
      typeof data.subarray !== "function"
    ) {
      return logits;
    }
    const typed = /** @type {ArrayLike<number> & { subarray: (start: number) => ArrayLike<number> }} */ (
      data
    );
    this.pending = Float32Array.from(typed.subarray(typed.length - vocab));
    return logits;
  }
}
