# Presentation Guide: Probing Internal Signals in Math LLMs

**Time Budget**: 5 minutes presentation + 2 minutes Q&A  
**Slides**: ~6-7 slides (45-50 seconds per slide)

---

## Slide 1: Title & Motivation (45 sec)

### Title
**"Probing Internal Signals of Topic, Difficulty, and Imminent Failure in Math LLMs"**

### Key Question (Hook)
> "Can we decode from a model's hidden states whether it will answer correctly—**before it even starts generating**?"

### Talking Points
- LLMs are increasingly used for math reasoning (education, verification, tutoring)
- Problem: Models confidently generate wrong answers
- Our approach: Use **linear probes** on hidden states to decode internal signals
- Practical goal: Early failure detection → smarter compute allocation, retry strategies

### Visual
- [PLACEHOLDER: Single compelling figure - maybe the 0% checkpoint success probe result]

---

## Slide 2: Background & Related Work (45 sec)

### Key Concepts
1. **Linear Probing**: Train simple classifier on frozen activations to "read" what model knows
2. **Activation Checkpoints**: Capture hidden states at different generation stages:
   - 0% = question-only (before any answer)
   - 50% = mid-generation
   - 100% = full response

### Related Work
- **Representations in LLMs**: Linear representation hypothesis (Mikolov et al., 2013; Nanda et al., 2023)
- **Probing for Knowledge**: Probing classifiers reveal linguistic structure (Belinkov & Glass, 2019)
- **Math Reasoning**: Chain-of-thought prompting (Wei et al., 2022), MATH benchmark (Hendrycks et al., 2021)
- **Uncertainty Estimation**: Kadavath et al. (2022) - "Language Models (Mostly) Know What They Know"

### Talking Points
- Prior work shows LLMs encode semantic features linearly
- We extend this to **metacognition**: does the model know when it will fail?
- Novel: Multi-checkpoint analysis during generation

---

## Slide 3: Method & Experimental Setup (50 sec)

### Models
- **Qwen2.5-Math-1.5B-Instruct** (main model, 500 samples)
- **Qwen2.5-Math-7B-Instruct** (comparison, 200 samples)
- Both: 4-bit quantized, 29 layers

### Dataset
- **MATH** benchmark (Hendrycks et al., 2021)
- 4 topics: Algebra, Precalculus, Number Theory, Counting & Probability
- 5 difficulty levels
- Balanced correct/incorrect samples

### Probe Types
| Probe | Task | Features |
|-------|------|----------|
| **Success** | Binary classification | Will answer be correct? |
| **Topic** | 4-class classification | What math domain? |
| **Difficulty** | 5-class / Regression | Human-rated difficulty |

### Architecture
- **Logistic Regression** (primary)
- **Difference-of-Means** (interpretable baseline)
- **MLP** (to test linearity hypothesis)

### Visual
- [PLACEHOLDER: Pipeline diagram - Question → Model → Hidden States → Probes → Predictions]

---

## Slide 4: Main Results - Success Prediction (60 sec) ⭐

### Headline Finding
> **Models "know" they will fail—before generating any answer!**

### Results Table
| Model | 0% Checkpoint | 100% Checkpoint |
|-------|---------------|-----------------|
| 1.5B  | 73%           | 81%             |
| 7B    | **80%**       | **90%**         |

### Key Insights
1. **Metacognition exists**: 73-80% accuracy at 0% = signal present before answering
2. **Signal strengthens**: Accuracy increases as model generates (81% → 90%)
3. **Scales with size**: 7B model has +7-9% better "self-knowledge"
4. **Layer localization**: Best performance at layers 16-22 (middle-to-late)

### Visuals
- [PLACEHOLDER: Success probe accuracy plot (layer × checkpoint heatmap)]
- [PLACEHOLDER: 1.5B vs 7B comparison bar chart]

### Talking Points
- This is the **main contribution** - evidence of metacognitive signals
- Practical implication: Could use probes for early stopping, retry decisions
- Control task passed (shuffled labels → 50% accuracy) - not memorization

---

## Slide 5: Additional Findings (50 sec)

### Finding 1: Linear Probes Sufficient
| Checkpoint | Linear | MLP | Δ |
|------------|--------|-----|---|
| 0%         | 73%    | 72% | -1% |
| 100%       | 81%    | 80% | -1% |

> "Success signal is **linearly encoded** in hidden states"

### Finding 2: Token Length Predicts Failure
- Correlation: **r = -0.49** (longer = more likely wrong)
- Correct: 458 tokens avg
- Incorrect: 731 tokens avg
- Long responses (>1024 tokens): only **2.4% accuracy**

> "Models ramble when uncertain"

### Finding 3: Topic vs Difficulty
- Topic probe: **86%** accuracy (questions have distinctive vocabulary)
- Difficulty probe: **46%** (model doesn't perceive human difficulty)

### Visual
- [PLACEHOLDER: Token length distribution (correct vs incorrect)]
- [PLACEHOLDER: MLP vs Linear comparison plot]

---

## Slide 6: Analysis & Discussion (45 sec)

### Why Does This Work?
1. **Causal reasoning encoded early**: Model recognizes problem structure
2. **Uncertainty manifests as spreading activation**: Wrong paths → longer, meandering outputs
3. **Linear representation hypothesis**: Abstract concepts (correctness) encoded as directions

### Limitations
- Only tested on Qwen2.5-Math family
- Balanced sampling may not reflect real-world distribution
- 50% checkpoint is noisy (mid-generation signal unclear)

### Confidence Calibration
- ECE improves during generation: 0.228 (0%) → 0.174 (100%)
- 93% of probe predictions are **stable** across checkpoints
- Model rarely "changes its mind"

### Visual
- [PLACEHOLDER: Confidence trajectory plot]
- [PLACEHOLDER: PCA visualization colored by correctness]

---

## Slide 7: Conclusions & Future Work (30 sec)

### Summary
1. ✅ **Metacognition exists**: Models encode success/failure signals internally
2. ✅ **Signal is linear**: Simple probes sufficient
3. ✅ **Scales with model size**: Larger models "know themselves" better
4. ✅ **Practical signals**: Token length as simple failure heuristic

### Future Directions
- Test on other model families (Llama, GPT)
- Real-time intervention during generation
- Per-topic success probes (algebra vs precalculus)
- Cross-dataset transfer (MATH → GSM8K)

### Final Message
> "LLMs may not know the right answer, but they often know when they don't know."

---

## Q&A Preparation (2 min)

### Likely Questions & Answers

**Q: How is this different from just looking at output confidence?**
> A: We probe hidden states, not output logits. The signal exists at 0% before any tokens are generated. Output confidence comes from final layer only.

**Q: Why not use the model's own "I don't know" statements?**
> A: Models rarely say "I don't know" - they hallucinate confidently. Probes access the latent uncertainty the model doesn't verbalize.

**Q: Could this be used in practice?**
> A: Yes! Ideas: (1) Early stopping when probe predicts failure, (2) Trigger re-prompting or tool use, (3) Flag answers for human review.

**Q: Why does 50% checkpoint perform worse?**
> A: Mid-generation is noisy. The model may be exploring multiple solution paths. Signal crystallizes at endpoints.

**Q: What if the probe is just detecting question difficulty?**
> A: Difficulty probes only achieve 46%, while success probes achieve 81%. They're detecting something beyond difficulty.

---

## Speaker Notes / Narrative Flow

### Opening (Slide 1)
Start with the provocative question. Establish stakes: LLMs in education, automated tutoring, math verification. The problem is real - models confidently give wrong answers.

### Background (Slide 2)
Quickly establish what linear probing is. Emphasize novelty: we look at *when* during generation the signal appears.

### Method (Slide 3)
Keep technical but brief. Emphasize the three checkpoints (0%, 50%, 100%) as the key experimental design.

### Results (Slide 4) - SPEND TIME HERE
This is your main contribution. Hit the numbers hard:
- "73% accuracy BEFORE the model types anything"
- "7B model achieves 90%"
Show the heatmap, let it sink in.

### Additional Findings (Slide 5)
Support your main claim with convergent evidence. Linear = simple story. Token length = practical heuristic.

### Discussion (Slide 6)
Show you've thought critically. Acknowledge limitations upfront.

### Conclusion (Slide 7)
End strong with the take-home message. Future work shows the project has legs.

---

## Figures Checklist

| Figure | Source | Slide |
|--------|--------|-------|
| Success probe heatmap (Layer × Checkpoint) | `success_probe_heatmap_2d.png` | 4 |
| Success probe accuracy by layer | `success_probe_accuracy.png` | 4 |
| MLP vs Linear comparison | `mlp_vs_linear.png` | 5 |
| Token distribution by correctness | `token_distribution.png` | 5 |
| PCA colored by correctness | `pca_layer20_correctness.png` | 6 |
| Confidence trajectories | `confidence_trajectories.png` | 6 |
| Topic probe accuracy | `topic_probe_accuracy.png` | 5 (optional) |
| Control task (shuffled) | `control_task_shuffled.png` | 4 (backup) |

---

## Time Breakdown

| Slide | Content | Time |
|-------|---------|------|
| 1 | Title & Motivation | 0:45 |
| 2 | Background & Related Work | 0:45 |
| 3 | Method & Setup | 0:50 |
| 4 | **Main Results** | 1:00 |
| 5 | Additional Findings | 0:50 |
| 6 | Analysis & Discussion | 0:45 |
| 7 | Conclusions | 0:30 |
| **Total** | | **5:25** |

*Aim for 5:00-5:15 to leave buffer*

---

## Key References

1. Hendrycks, D., et al. (2021). "Measuring Mathematical Problem Solving With the MATH Dataset." *NeurIPS*.
2. Wei, J., et al. (2022). "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models." *NeurIPS*.
3. Kadavath, S., et al. (2022). "Language Models (Mostly) Know What They Know." *arXiv*.
4. Nanda, N., et al. (2023). "Progress Measures for Grokking via Mechanistic Interpretability." *ICLR*.
5. Belinkov, Y., & Glass, J. (2019). "Analysis Methods in Neural Language Processing: A Survey." *TACL*.
6. Mikolov, T., et al. (2013). "Linguistic Regularities in Continuous Space Word Representations." *NAACL*.

---

## Lessons Learned (For Discussion)

1. **Evaluation is hard**: Spent significant time on robust answer checking (LaTeX parsing, symbolic equivalence)
2. **Smaller models = faster iteration**: 1.5B allowed rapid experimentation before 7B confirmation
3. **Balance matters**: Balanced correct/incorrect sampling critical for interpretable probes
4. **Simple baselines work**: Linear probes competitive with MLPs - Occam's razor applies

---

*Good luck with your presentation! 🎉*

