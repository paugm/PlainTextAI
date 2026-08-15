/**
 * Optional check that the cached model can continue an opening in the Romeo style.
 * Uses the same excerpt, opening-prefix, and token-step path as the web app.
 *
 *   npm run smoke
 */
import {
  AutoModelForCausalLM,
  AutoTokenizer,
  LogitsProcessorList,
  TextStreamer,
} from "@huggingface/transformers";
import { readFileSync } from "node:fs";
import { CaptureLogitsProcessor } from "../src/lib/capture-logits.js";
import { DEFAULT_OPENING, MODEL_ID, TOP_K } from "../src/lib/config.js";
import { prepareVoice } from "../src/lib/excerpts.js";
import { buildMessages, withOpeningPrefix } from "../src/lib/prompt.js";
import { tokenStep } from "../src/lib/token-step.js";

const text = readFileSync(new URL("../public/samples/romeo-and-juliet.txt", import.meta.url), "utf8");
const voice = prepareVoice(text, { title: "Romeo and Juliet" });
const messages = buildMessages(voice.excerpts);

console.log(`Text: ${voice.excerptCount} passages, ${voice.excerptChars} characters`);
console.log(`Opening: ${DEFAULT_OPENING}`);
console.log(`Loading ${MODEL_ID} (CPU q4)…`);

const tokenizer = await AutoTokenizer.from_pretrained(MODEL_ID);
const model = await AutoModelForCausalLM.from_pretrained(MODEL_ID, {
  device: "cpu",
  dtype: "q4",
});

const capture = new CaptureLogitsProcessor();
const processors = new LogitsProcessorList();
processors.push(capture);
const chatPrompt = tokenizer.apply_chat_template(messages, {
  add_generation_prompt: true,
  tokenize: false,
});
const inputs = tokenizer(withOpeningPrefix(chatPrompt, DEFAULT_OPENING), {
  add_special_tokens: false,
});

const specialIds = new Set((tokenizer.all_special_ids || []).map(Number));
const steps = [];
let output = "";

const streamer = new TextStreamer(tokenizer, {
  skip_prompt: true,
  skip_special_tokens: true,
  callback_function: () => {},
  token_callback_function: (tokens) => {
    const tokenId = Number(tokens[0]);
    if (specialIds.has(tokenId)) {
      capture.pending = null;
      return;
    }
    const step = tokenStep(tokenizer, capture.pending, tokenId);
    capture.pending = null;
    if (!step) {
      return;
    }
    steps.push(step);
    output += step.token;
  },
});

await model.generate({
  ...inputs,
  max_new_tokens: 20,
  do_sample: true,
  temperature: 1,
  top_k: TOP_K,
  logits_processor: processors,
  streamer,
});

if (!output.trim() || steps.length === 0) {
  throw new Error("Model returned no tokens.");
}

const first = steps[0];
console.log("Continuation:", output);
console.log(
  `First token "${first.token}" ${(first.probability * 100).toFixed(1)}% vs`,
  first.alternatives
    .slice(0, 3)
    .map((row) => `"${row.token}" ${(row.probability * 100).toFixed(1)}%`)
    .join(", ")
);
