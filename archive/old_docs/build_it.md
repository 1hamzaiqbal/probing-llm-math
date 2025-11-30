Short version: with 32GB you’re squarely in the “run the good 7B math models + a judge/eval stack” regime. Here’s a concrete setup that hits all your constraints:


I WANT TO RUN THIS ON GOOGLE COLAB VIA A GOOGLE TPU OR A NVIDIA T4 GPU.
---

## 1. Models to actually answer math questions (fit easily in 32GB, HF-friendly, good on MATH/GSM8K)

All of these are standard decoder-only transformers with open weights and HuggingFace support, so you can extract hidden states cleanly.

### 1) Qwen2.5-Math-7B(-Instruct)

* IDs:

  * `Qwen/Qwen2.5-Math-7B` (base)
  * `Qwen/Qwen2.5-Math-7B-Instruct` (chat/instruct) ([Hugging Face][1])
* Why it’s good:

  * Math-specialized continuation of Qwen2.5, with strong GSM8K / Hendrycks MATH performance vs other open models in the same size range. ([Emergent Mind][2])
  * Trained specifically for math (CoT and tool-integrated reasoning), but you can clamp it to short answers with prompting/sampling.
* VRAM:

  * 7B bf16 weights ≈ 14GB; with KV cache and overhead you’re still very comfy in 32GB even for 8k ctx.
* Use-case match:

  * If you want one main math model that “just works” and is fairly SOTA and open, this is probably the default pick.

**Not-overthinking:**
Use the Instruct model but give a strict system prompt, `temperature=0`, and small `max_new_tokens`, e.g.:

> “Solve the problem and respond with **only** the final answer as a single number or algebraic expression. No explanation.”

That stops it from going full essay-CoT even though it *can*.

---

### 2) DeepSeek-Math-7B (base / instruct / RL)

* IDs:

  * `deepseek-ai/deepseek-math-7b-base`
  * `deepseek-ai/deepseek-math-7b-instruct`
  * `deepseek-ai/deepseek-math-7b-rl` ([Hugging Face][3])
* Why it’s good:

  * Math-specialized continuation from DeepSeek-Coder, with ~51.7% on Hendrycks MATH without voting/tools, competitive with much larger closed models per FLOP. ([GitHub][4])
  * Lots of community use and quantizations; easy to run with HF or vLLM.
* VRAM:

  * Similar story: 7B bf16 is trivial on 32GB; you can even run two instances (answerer + small judge) on one GPU if you quantize.
* Which variant:

  * For probing/hidden states, use **`deepseek-math-7b-base`** (no instruction adapters).
  * For just solving problems, `deepseek-math-7b-instruct` with the same “final answer only” style prompt as above.

Importantly: this is *not* DeepSeek-R1; you avoid the heavy auto-COT “overthinking” behavior while still getting a math-tuned backbone. ([Hugging Face][5])

---

### 3) Llemma-7B (for more formal/math-code flavor)

* ID: `EleutherAI/llemma_7b` ([arXiv][6])
* Why:

  * Pretrained heavily on Proof-Pile-2 (papers, math web, code); good at more “theorem/proof-like” math.
  * Shown to outperform earlier open math models and Minerva at similar parameter counts. ([arXiv][6])
* Caveats:

  * Less instruction-tuned; you may need a small prompt template and a bit of babysitting to get nice, short “answer-only” outputs.
* Good if you care more about internal representations & proofs than about raw benchmark SOTA.

---

### 4) Maybe: Qwen2.5-Math-1.5B as a cheap judge or side model

* ID: `Qwen/Qwen2.5-Math-1.5B-Instruct` ([Azure AI][7])
* It actually does surprisingly well on GSM8K/MATH for its size; some recent work reports respectable scores with 1.5B variants. ([ResearchGate][8])
* Fits in a few GB of VRAM; great candidate if you want a tiny “helper” model for judging or secondary tasks.

---

## 2. Hidden state extraction with an API

You have two main options:

### A) Vanilla `transformers` + your own HTTP API

All of the above models work with HuggingFace Transformers and can return per-layer hidden states:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2.5-Math-7B"
tok = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="bfloat16",
    device_map="auto",
)

prompt = "Solve: 2x + 3 = 11. What is x? Answer only."
inputs = tok(prompt, return_tensors="pt").to(model.device)

out = model(**inputs, output_hidden_states=True)
hidden_states = out.hidden_states   # tuple: (layer0, layer1, ..., layerN)
logits = out.logits
```

Wrap this in FastAPI/Flask and expose e.g.:

* `/generate` → returns text, logits, maybe last-layer states
* `/hidden_states` → returns serialized hidden states for a given prompt (or token index)

That gives you full control and no ambiguity about what “hidden states” means.

### B) vLLM or similar OpenAI-compatible servers

* vLLM now has plumbing for returning hidden states / pooling models and discussions/PRs around returning per-token hidden states. ([VLLM Docs][9])
* Reality: support is evolving; for serious representation work, I’d still recommend **doing a second HF forward pass** just for hidden states (cheap compared to the original decoding, especially if you only run it on the prompt or on a small window around the final token).

Pattern that works well:

1. Use vLLM (or TGI) as your “fast OpenAI-like generation API”.
2. After you have a completion, call a local HF model with `output_hidden_states=True` on:

   * the prompt, or
   * prompt + completion truncated to `N` tokens
3. Save those hidden states for probing.

That keeps infra simple while giving you everything you need for interpretability.

---

## 3. “Not overly thinking”: controlling reasoning length

Regardless of which math model you pick, you can keep it from being a maximalist CoT beast by:

* **System / task prompt**:

  > “You are a math solver. For each problem, compute the answer internally and output only the final numeric or algebraic answer on a single line, with no explanation.”

* **Decoding settings**:

  * `temperature = 0` (or ≤0.2)
  * `top_p = 1.0`
  * `max_new_tokens` something like 64–128
  * No explicit “let’s think step by step” or “show all work” in the prompt.

Even math-specialized models like Qwen2.5-Math/DeepSeek-Math can behave like simple answerers under those constraints. ([Qwen][10])

---

## 4. Cheap correctness checking / evaluation scaffold

You **do not** need a big second LLM to judge correctness if you have gold answers. The standard trick (and what most math papers do) is:

### A) Use a math eval harness (existing scaffold)

Two popular options that already implement “feed questions + check correctness”:

1. **LLM Math Evaluation Harness** (ZubinGou / math-evaluation-harness)

   * GitHub: `ZubinGou/math-evaluation-harness` ([GitHub][11])
   * Designed specifically to evaluate LLMs on GSM8K, MATH, Olympiad, AIME, etc.
   * Handles:

     * Running your model via HF / vLLM backends.
     * Extracting the final numeric answer using regex heuristics.
     * Comparing vs ground truth and computing accuracy / pass@k.

2. **HuggingFace Lighteval**

   * `huggingface/lighteval`, with built-in math/numeracy tasks and sample-by-sample reports. ([GitHub][12])

Both give you exactly what you described: “existing scaffold where I feed in questions and get correctness metrics” with minimal glue.

### B) Simple custom checker (if you have your own dataset)

If you have (question, ground_truth_answer) pairs:

1. Prompt the model to output **just** an answer (as above).
2. Parse its answer:

   * Last number or the content after a marker like `####` or `Answer:`
   * Normalize formatting (strip spaces, standardize fractions).
3. Use **rule-based or SymPy-based** equality:

   * For numeric answers: cast to `Fraction` or decimal and compare.
   * For expressions: use `sympy.simplify(model_ans - gt_ans) == 0` when possible.

That’s practically free and more reliable than a tiny LLM judge for standard contest-style problems.

### C) If you *really* want a model-based judge

Given 32GB VRAM and desire for cheapness:

* Use a **small** math or general model as judge, e.g.:

  * `Qwen/Qwen2.5-Math-1.5B-Instruct`
  * Or another 1–3B instruct model. ([Azure AI][7])

Prompt pattern:

> “Problem: …
> Gold answer: …
> Model answer: …
> Is the model answer mathematically equivalent to the gold answer? Reply with exactly `CORRECT` or `INCORRECT`.”

But in most math-benchmark-style setups, a rule-based / SymPy / harness approach is simpler and more robust.

---

## 5. Concrete “minimal viable stack” recommendation

Given your constraints and 32GB:

1. **Backbone**:

   * Start with **`Qwen/Qwen2.5-Math-7B-Instruct`** as the main solver.
   * Optionally keep **`Qwen/Qwen2.5-Math-7B` (base)** around for clean hidden-state analysis.

2. **Runtime**:

   * Serve it with vLLM or TGI for fast decoding (OpenAI-compatible endpoint). ([Azure AI][13])
   * For hidden states, run a second HF forward with `output_hidden_states=True` on the same prompt (or prompt + short completion).

3. **Evaluation scaffold**:

   * Plug the model into **`math-evaluation-harness`** or **Lighteval** for GSM8K / MATH, or adapt their task definitions to your own dataset. ([GitHub][11])
   * Use the harness’s built-in answer extraction / matching instead of a judge LLM unless you truly need one.

4. **Non-overthinking behavior**:

   * System prompt: “final answer only, no steps.”
   * `temperature=0`, `max_new_tokens` small.

Once that’s wired, you can start logging hidden states per problem and correlating layer activations with correctness, difficulty, or specific math skills.

[1]: https://huggingface.co/Qwen/Qwen2.5-Math-7B?utm_source=chatgpt.com "Qwen/Qwen2.5-Math-7B"
[2]: https://www.emergentmind.com/topics/qwen2-5-math-7b?utm_source=chatgpt.com "Qwen2.5-Math-7B: Math-Specialized 7B LLM"
[3]: https://huggingface.co/deepseek-ai/deepseek-math-7b-instruct?utm_source=chatgpt.com "deepseek-ai/deepseek-math-7b-instruct"
[4]: https://github.com/deepseek-ai/DeepSeek-Math?utm_source=chatgpt.com "DeepSeekMath: Pushing the Limits of Mathematical ..."
[5]: https://huggingface.co/deepseek-ai/DeepSeek-R1?utm_source=chatgpt.com "deepseek-ai/DeepSeek-R1"
[6]: https://arxiv.org/abs/2310.10631?utm_source=chatgpt.com "Llemma: An Open Language Model For Mathematics"
[7]: https://ai.azure.com/catalog/models/qwen-qwen2.5-math-1.5b-instruct?utm_source=chatgpt.com "qwen-qwen2.5-math-1.5b-instruct"
[8]: https://www.researchgate.net/publication/395213427_VerlTool_Towards_Holistic_Agentic_Reinforcement_Learning_with_Tool_Use?utm_source=chatgpt.com "Towards Holistic Agentic Reinforcement Learning with Tool ..."
[9]: https://docs.vllm.ai/en/v0.6.6/models/pooling_models.html?utm_source=chatgpt.com "Pooling Models — vLLM"
[10]: https://qwenlm.github.io/blog/qwen2.5-math/?utm_source=chatgpt.com "Qwen2.5-Math: The world's leading open-sourced ..."
[11]: https://github.com/ZubinGou/math-evaluation-harness?utm_source=chatgpt.com "ZubinGou/math-evaluation-harness"
[12]: https://github.com/huggingface/lighteval?utm_source=chatgpt.com "Lighteval is your all-in-one toolkit for evaluating LLMs ..."
[13]: https://ai.azure.com/catalog/models/deepseek-ai-deepseek-math-7b-base?utm_source=chatgpt.com "AI Model Catalog | Azure AI Foundry Models"
