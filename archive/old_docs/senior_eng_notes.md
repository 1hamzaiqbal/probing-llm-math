You’re already doing the hard part (actually collecting clean activations and labels); what you need now is a sane, repeatable *experimental protocol* so your probes stop feeling like vibes.

I’ll break this into:

1. How to systematically pick **layer / token / pooling** for topic, success, and difficulty probes
2. How to design the **0% / 50% / 100% success probes** and tie them to self‑consistency
3. What to use for **“mostly out‑of‑the‑box” answer checking**, especially for math
4. Concrete **methodology patterns from other probe papers** that you can copy

I’ll reference both your own docs and external papers as we go.

---

## 1. Systematically finding “best layer / token / pooling” for each probe

You already sweep all layers and store activations at 0/50/100%.  Now the question is: how do you *systematically* choose:

* which **layer**
* which **token position** (last, mean, etc.)
* which **aggregation** (mean, max, etc.)

### 1.1 Define a small, fixed feature grid

Instead of inventing new variants mid‑experiment, define a fixed grid of representation types:

For each checkpoint (0%, 50%, 100%), and each layer ℓ:

* **Last token of the sequence** (what you’re already doing)
* **Mean over question tokens** (prompt only)
* **Mean over answer tokens** (for 50% / 100%)
* Optional but cheap: **[last − mean(question)]** concatenation (captures “update from question to current state”)

This gives at most 3–4 variants per layer per checkpoint. That’s still manageable.

This pattern matches what Marks & Tegmark do in *Geometry of Truth* (they try simple average‑based “difference‑of‑means” directions and find they are surprisingly strong) ([arXiv][1]) and what the “No Answer Needed” paper does for question‑only correctness prediction. ([arXiv][2])

### 1.2 Train probes with *proper* model selection

For each task (topic, success, difficulty), do:

1. **Split data by problem** into train / val / test (e.g. 60 / 20 / 20). You’re already doing problem‑level splits; keep that. 

2. For every **(layer ℓ, representation type r)**:

   * Fit a **logistic regression / multinomial logistic / ridge regression** probe on **train** only
   * Tune regularization strength *only* on **val** (small grid of C or λ)
   * Log:

     * val accuracy or R²
     * calibration metrics (Brier score, ECE) for success probe
     * standard error via bootstrap over problems (e.g., resample problems 200 times)

3. Select the **top K configs** per task by *val metric*, where K is like 3–5, not 1.

4. Only once per task:

   * Evaluate those K on the **held‑out test set**
   * Declare the “winner” based on test performance + variance (e.g., if two layers are statistically tied, prefer the earlier one or the simpler representation type).

This is very close to the protocol in Alain & Bengio “Understanding Intermediate Layers using Linear Classifier Probes” and Hewitt’s “Designing and Interpreting Probes” (they emphasize: separate model‑selection set, control probe complexity, and report curves over depth instead of just the single best layer). ([Columbia Computer Science][3])

You already write a `summary.json` with best layers; this just makes **how** you pick them principled. 

### 1.3 Coarse‑to‑fine layer search (to save compute)

You already had the idea in the proposal: probe every 4th layer, then zoom in.  This matches a lot of probing work:

1. **Stage 1 — Coarse sweep**

   * Probe layers {0, 4, 8, 12, …, L} for each representation type.
   * Find a **band of interest**, e.g. layers 12–20 for success, 4–12 for topic.

2. **Stage 2 — Fine sweep**

   * Within that band, probe *every* layer.
   * Do the more careful model‑selection / bootstrap analysis only there.

This mirrors what “No Answer Needed” and “Geometry of Truth” find: predictive directions tend to peak in a **middle band** (roughly 40–60% of depth) and stay decent within a small neighborhood. ([arXiv][1])

### 1.4 Difference‑of‑means probes as a baseline

A nice trick from Geometry of Truth:

Instead of full logistic regression, use a **difference‑of‑means** probe:

[
w = \mu_{\text{correct}} - \mu_{\text{incorrect}}
]

* where μ are the mean activations per class, per layer.

Then classify by sign of ( w \cdot h + b ).

Do this for:

* **Success**: correct vs incorrect
* **Difficulty**: hard (levels 4–5) vs easy (1–2), etc.

Marks & Tegmark show these simple “template” probes perform as well as logistic regressions in truth vs false settings and are more causally meaningful. ([arXiv][1])

Methodologically, you can:

* Train **both** logistic regression and difference‑of‑means
* Compare generalization (especially out‑of‑distribution slices like rare topics or hardest levels)

If difference‑of‑means is close in performance, you get a cleaner story: “There really is a line in representation space separating ‘easy vs hard’ or ‘correct vs incorrect’.”

---

## 2. Designing the 0% / 50% / 100% success probe experiments

You already capture `activations_0pct`, `activations_50pct`, `activations_100pct` per problem.  That’s exactly what you need for the “internal success prediction over time” story.

### 2.1 Data collection design

You’re basically implementing the same “replay” strategy as:

* “No Answer Needed” – question‑only correctness probes ([arXiv][2])
* Recent work on probing arithmetic errors at intermediate CoT steps (they probe at each equals sign to predict step correctness and final correctness). ([ACL Anthology][4])

Clean design:

1. **Generate once** per problem (temperature 0 for baseline).
2. Record:

   * full CoT tokens
   * final answer correctness (your evaluator)
   * response length N
3. For each checkpoint α ∈ {0.0, 0.25, 0.5, 0.75, 1.0} (you have 0/0.5/1.0; you can add 0.25/0.75 cheaply later):

   * Build prefix: `question + first floor(α * N) answer tokens`
   * Re‑run with `output_hidden_states=True` to get all layers.

You already implement 0/50/100. The same mechanism extends to 25/75 if you want a smoother curve.

### 2.2 Probe training as a 2D heatmap

Instead of three unrelated probes, set up:

* **Grid over (layer ℓ, checkpoint α)**
* At each point, use your standard logistic + difference‑of‑means success probe (as above)

Plot:

* **Accuracy / AUC vs (ℓ, α)** as a heatmap
* Maybe overlay contour lines (e.g., where AUC = 0.7, 0.8, etc.)

This is exactly what Sun et al. do in *Probing for Arithmetic Errors in Language Models*: they examine how predictive probes are when attached at different CoT steps and layers. ([ACL Anthology][4])

You can then tell:

* Does predictive signal **grow** from 0% → 50% → 100%?
* Does it **peak** at some intermediate layer band (e.g., 16–24)?
* Are there checkpoints where the probe *loses* predictive power, suggesting the model is “hallucinating itself away” from its earlier good intuition?

### 2.3 Progress‑aware success probes

Fun variant: treat each (problem, checkpoint) pair as a data point with:

* Features: activations at layer ℓ and checkpoint α
* Labels: eventual success (same for all α for a given problem)

Then:

* Train one probe that takes *[h, α]* (i.e., concatenate α as a scalar feature)
* Or train separate probes per α and compare calibration curves

You can directly plot **P(correct | α)** estimated by the probe and compare to actual empirical accuracy at that α. That’s a direct replication of the “internal success prediction” vibe in *No Answer Needed*, but extended over a trajectory rather than only at α = 0. ([arXiv][2])

---

## 3. Probing self‑consistency and other “probe‑able” aspects

Self‑consistency in math reasoning is basically: sample multiple chains, vote on the answer. Wang et al. show this dramatically improves math benchmarks. ([arXiv][5]) You can make that part of your probing story in a few ways.

### 3.1 Multi‑sample generation + hidden state statistics

For each problem (maybe a smaller subset, like 100–200 problems so it’s tractable):

1. Generate **K chains** (e.g., K = 5 or 10) with some temperature > 0, capturing correctness per chain.
2. For each chain s and checkpoint α:

   * Capture activations ( h_{\ell,\alpha,s} ) at your favorite layer ℓ.
3. Compute per‑problem statistics across samples:

   * Mean representation: ( \bar{h}*{\ell,\alpha} = \frac{1}{K}\sum_s h*{\ell,\alpha,s} )
   * Variance / covariance trace across samples (a scalar “representation dispersion”)
   * Entropy of answer distribution (from output logits or just majority vote count)

Now define new probe tasks:

* Predict **whether self‑consistency will succeed** (i.e., “at least one correct chain and unique majority answer”).
* Predict **“model is unstable”** (e.g., different runs give different answers, regardless of correctness).

This is very close in spirit to very recent work on “multi‑turn answer instability” and using probes to predict when models will flip answers across turns. ([ResearchGate][6])

### 3.2 Probing confidence & uncertainty

Combine your success probe with cheap scalar features:

* Final answer logprob or average token logprob
* Entropy of output distribution at final token
* Residual stream **norm** at last token

Protocol (again, from *No Answer Needed* and similar):

* Train:

  1. black‑box baseline: logistic regression on **logprob only**
  2. **probe only** (hidden state direction)
  3. **fusion** of logprob + probe

Compare AUC / calibration (ECE, reliability curve). If the probe adds predictive power beyond logprob, that’s a neat result aligned with literature.

---

## 4. Answer checking / evaluation: “mostly out‑of‑the‑box” options

Your evaluator is already quite sophisticated (LaTeX→SymPy, matrices, complex, sets, etc.), and you test it.  The annoyance is brittleness and edge cases.

The cleanest out‑of‑the‑box solution for math right now is exactly what you already have in `requirements.txt`: **Math‑Verify**. 

### 4.1 Math‑Verify as a drop‑in evaluator

HuggingFace’s `math-verify` library is built specifically to robustly score MATH‑style outputs: it parses both model output and ground‑truth with regex + SymPy and then compares expressions using symbolic and numeric checks. ([PyPI][7])

Their benchmarks show:

* It outperforms Harness and Qwen’s own evaluator on MATH (higher accuracy of “is this answer correct?”). ([GitHub][8])

Methodologically, what you can do:

* Wrap your `evaluator.py` and `math-verify` in a small “ensemble evaluator”:

  * Run both;
  * If they disagree, optionally log and manually inspect (for the paper, you can say you used Math‑Verify as primary and fall back to your evaluator in cases where Math‑Verify fails to parse).

That gives you robustness and lets you say in the methodology: “We used the Math‑Verify evaluator, which has state‑of‑the‑art evaluation accuracy on the MATH benchmark” and cite it, instead of owning every parsing bug yourself. ([GitHub][8])

### 4.2 Mildly “unsupervised” checking tricks

You also asked about unsupervised pipelines. In math, you still usually have ground‑truth expressions, so “unsupervised” mostly means:

* Evaluating both candidate and ground‑truth at multiple random numeric instantiations (which Math‑Verify does variants of). ([GitHub][8])
* Self‑consistency voting as an *implicit* correctness proxy (if many diverse chains converge to the same answer, it’s likelier to be right). ([arXiv][5])

You can formalize that:

* For each problem, define a **self‑consistency correctness score**: fraction of chains agreeing with the majority answer that is also correct.
* Then compare:

  * “Symbolic evaluator accuracy”
  * “Self‑consistency majority‑vote accuracy”

This gives you another axis: *your probes could be predicting not just symbolic correctness but whether self‑consistency will rescue the answer*.

---

## 5. Methodology patterns directly borrowed from other probe papers

Here’s a small “cookbook” of things people do in linear‑probe interpretability work that you can mirror explicitly in your writeup.

### 5.1 From Geometry of Truth (Marks & Tegmark 2023) ([arXiv][1])

* **Difference‑of‑means directions** (see above) instead of only logistic regression.
* **Transfer tests**:

  * Train on one mixture of problems (e.g., levels 1–3) and test on others (levels 4–5, or different topics).
  * Train on GSM8K, test on MATH, or vice versa.
* **Causal interventions**:

  * After you have a success direction ( w ), actually nudge activations along +w or −w at your “best” layer and see if that biases the model toward success vs failure.
  * Implementation detail: in `model_utils.py`, intercept the residual stream at layer ℓ, add ( \lambda w ), then continue the forward pass.

Even one small intervention result (“pushing along the failure direction makes answers worse”) makes your story much stronger.

### 5.2 From No Answer Needed (Cencerrado et al. 2025) ([arXiv][2])

* **Question‑only success prediction** (which you’re already doing) with:

  * Multiple model sizes
  * Probes across layers, showing a nice “middle‑layer bump”
* **Comparisons vs black‑box baselines**:

  * Logprob, verbalized confidence (“I’m X% sure”), etc.

You can imitate this by:

* Plotting **probe AUC vs layer** and **logprob AUC vs layer**, highlighting where probes beat or match logprob.
* Showing that math is specifically a “harder regime” where probe generalization is weaker, as they note.

### 5.3 From classic probing work (Alain & Bengio; Hewitt; “BERT Rediscovers the NLP Pipeline”, etc.) ([Columbia Computer Science][3])

* **Control tasks / random labels**:

  * Shuffle labels and show probe accuracy goes to chance, confirming you’re not just over‑fitting high‑dimensional representations.
* **Complex vs simple probes**:

  * You can optionally try a tiny 2‑layer MLP probe and show that it doesn’t buy you much, which supports the “linear representation” narrative.

### 5.4 From recent arithmetic / stability probing work ([ACL Anthology][4])

* Probing **at intermediate reasoning steps** (each equals sign / chain step).
* Modeling **answer changes / instability** as a separate label (e.g., whether a follow‑up “think again” would flip the answer).
* Using Markov‑chain style models of answer evolution and asking whether probes predict transitions.

You could do a lightweight version by:

* Marking whether the model *changes its answer* across chains or across time (if you re‑ask the question).
* Probing for that “instability” label in your 0% / 50% / 100% states.

---

## 6. Concrete “how to” for your specific setup

Let me compress this into actionable steps with your current codebase in mind.

1. **Augment `collect_probe_data.py`**

   * Save mean‑pooled question and answer activations per layer alongside last‑token ones.
   * Optionally add 25% and 75% checkpoints later (same mechanism as 50%).

2. **Extend `train_probe.py`** to:

   * Loop over (layer ℓ, representation type r, checkpoint α).
   * For each, fit:

     * L2‑logistic (or softmax) for topic & success
     * Ridge regression for difficulty
     * Difference‑of‑means success direction
   * Use a problem‑level train/val/test split and tune regularization on val.
   * Log results as a big table + 2D heatmaps for success (ℓ vs α).

3. **Add a `probes/summary_experiments.py`** that:

   * Produces:

     * Depth curves (accuracy vs layer) for each probe
     * 2D heatmaps for success (layer vs checkpoint)
     * Calibration plots for success (compare logprob vs probe vs fusion).

4. **Swap in Math‑Verify** in `evaluator.py`:

   * Try `math_verify.verify(gold, pred)` first; if it errors or returns “unknown”, fall back to your custom evaluator. ([GitHub][8])

5. **Optional but spicy**: implement a tiny “intervention” function in `model_utils.py`:

   * Given layer ℓ and learned direction w:

     * Hook into the forward pass, add ±λw to the residual stream at ℓ for the last token, and re‑generate.
   * Run a small experiment on ~50 problems to see if this manipulates success rate in the predicted direction.

That set of steps will let you tell a very clear, literature‑anchored story:

* Where topic/difficulty/success are linearly represented in Qwen2.5‑Math’s layers
* How that representation evolves from 0% → 50% → 100% of the solution
* Whether self‑consistency and linear probes agree about where the model “knows” it’s right or wrong
* How robust your findings are given a strong, mostly plug‑and‑play math evaluator

From there, you can zoom into whatever subset of the probe landscape ends up behaving nicely—or hilariously badly, which is also publishable.

[1]: https://arxiv.org/abs/2310.06824?utm_source=chatgpt.com "The Geometry of Truth: Emergent Linear Structure in Large Language Model Representations of True/False Datasets"
[2]: https://arxiv.org/abs/2509.10625?utm_source=chatgpt.com "No Answer Needed: Predicting LLM Answer Accuracy from Question-Only Linear Probes"
[3]: https://www.cs.columbia.edu/~johnhew/interpreting-probes.html?utm_source=chatgpt.com "Designing and Interpreting Probes · John Hewitt"
[4]: https://aclanthology.org/2025.emnlp-main.411.pdf?utm_source=chatgpt.com "Probing for Arithmetic Errors in Language Models"
[5]: https://arxiv.org/abs/2203.11171?utm_source=chatgpt.com "Self-Consistency Improves Chain of Thought Reasoning in Language Models"
[6]: https://www.researchgate.net/publication/397663620_Modeling_and_Predicting_Multi-Turn_Answer_Instability_in_Large_Language_Models?utm_source=chatgpt.com "(PDF) Modeling and Predicting Multi-Turn Answer ..."
[7]: https://pypi.org/project/math-verify/?utm_source=chatgpt.com "math-verify"
[8]: https://github.com/huggingface/Math-Verify?utm_source=chatgpt.com "huggingface/Math-Verify"

----
Other notes on model eval failures: 
### 2. Common Failure Modes (True Failures)
For the remaining ~238 rows that were genuinely incorrect, here are the common patterns:

### "Lazy" Guessing: In about 5% of failures, the model outputted a simple integer like 0, 1, or -1 for a complex problem (e.g., Row 0 asked for a product of 2004 terms, model guessed 1).

### Symbolic vs. Numeric Confusion: The model sometimes outputted a decimal approximation when an exact symbolic answer (like \sqrt{2}) was required, or vice versa.

### Verbose "Chain of Thought" Leakage: Many "Incorrect" predictions contained long strings of LaTeX or explanation (e.g., Row 390, 411). This suggests the model struggled to stop generating after the final answer, or the extraction regex wasn't aggressive enough in isolating the final value.

### Topic Difficulty: The failures are heavily concentrated in Algebra (Level 5), suggesting the linear probe struggles most with high-level symbolic manipulation compared to arithmetic or number theory.