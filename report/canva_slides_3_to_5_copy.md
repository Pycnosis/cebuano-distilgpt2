# Canva Slide Deck: Slides 3, 4, and 5 Copy & Defense Speaker Notes

**Project:** Teaching DistilGPT2 Cebuano: Fine-Tuning, Decoding & Alignment  
**Authors (Slides 3–5):** Joshua (`@JoshyEyeBa11` — Slides 3 & 5), Lej (`@lejxz` — Slide 4 & Cross-Slide Cebuano QA)  
**Team Members:** Nico (`@Pycnosis`), Lej (`@lejxz`), Joshua (`@JoshyEyeBa11`)  
**Repository:** [https://github.com/Pycnosis/cebuano-distilgpt2](https://github.com/Pycnosis/cebuano-distilgpt2)

---

## Slide 3 — Step 2: Decoding Strategies, KV-Cache Latency & MoE Inference Trade-offs

### 1. Slide Headline & Subheading
- **Headline:** Step 2: Decoding Benchmarks & Autoregressive Inference Mechanics
- **Subheading:** Mitigating repetition traps, measuring KV-cache speedup, and analyzing MoE memory constraints

### 2. Visual Layout & Assets to Embed
- **Primary Asset:** Side-by-side table of 4 decoding strategies on Prompt `P2` / `P4` (Greedy, Top-$k$, Top-$p$, Beam Search).
- **Secondary Asset:** `results/step2_kvcache_chart.png` (Latency vs. sequence length with and without KV-cache).
- **Architecture Callout Box:** Mixture-of-Experts (MoE) inference trade-off diagram (Active FLOPs vs. Total VRAM footprint).

### 3. Key Content & Metric Cards
- **Decoding Strategies Evaluated on Cebuano:**
  1. *Greedy (`do_sample=False`):* Deterministic, prone to catastrophic n-gram loops (`"sa kapulisan sa kapulisan..."`) due to probability traps.
  2. *Top-$k$ Sampling (`k=50`, `temp=0.8`):* Truncates low-probability tail, preserving fluency while introducing lexical variety.
  3. *Nucleus / Top-$p$ Sampling (`p=0.9`, `temp=0.8`):* Dynamically scales candidate pool based on cumulative probability mass; eliminates unnatural repetitions while retaining natural Bisaya sentence transitions.
  4. *Beam Search (`num_beams=4`, `no_repeat_ngram_size=2`):* Maximizes sequence likelihood while structurally banning 2-gram duplication, eliminating looping at the cost of higher latency.
- **KV-Cache Complexity:**
  - *Without Cache ($O(T^2)$):* At step $t$, recomputes Query, Key, and Value projections across all $t$ prior tokens, leading to quadratic scaling in attention computation.
  - *With Cache ($O(T)$):* Stores historical Key and Value tensors in memory, requiring only a single forward projection for the newest query token per generation step.
- **Mixture-of-Experts (MoE) Inference Note:**
  - *Compute Advantage:* Activates only top-$k$ experts per token (e.g., 2 of 8), reducing active parameter FLOPs and latency to that of a small model.
  - *Memory Bottleneck:* **All** expert weights must permanently reside in high-bandwidth memory (VRAM); offloading unselected experts to CPU RAM causes severe memory-transfer latency that destroys generation throughput.

### 4. Oral Defense Speaker Notes
> *"In Step 2, we examined how decoding algorithms interact with our LoRA-adapted Cebuano model. Under pure greedy decoding, the model frequently falls into self-reinforcing repetition loops—such as repeating 'sa kapulisan'—because high-frequency n-grams dominate the probability distribution. Nucleus sampling and beam search with n-gram blocking effectively break these attractor basins, producing diverse and grammatically fluent continuations.*  
>  
> *On the systems side, autoregressive generation requires computing attention across all previously generated tokens. Without a KV-cache, inference scales quadratically at $O(T^2)$ FLOPs as each step re-attends to the entire history from scratch. Enabling the KV-cache stores prior Keys and Values, reducing per-token generation to linear $O(T)$ complexity.*  
>  
> *Finally, regarding Mixture-of-Experts models: while sparse routing reduces per-token compute by only activating top-$k$ experts, it does not alleviate memory capacity constraints. All expert weights must remain in VRAM simultaneously, highlighting the classic systems trade-off between active FLOPs and total memory footprint."*

---

## Slide 4 — Step 3: Direct Preference Optimization (DPO) Alignment

### 1. Slide Headline & Subheading
- **Headline:** Step 3: Direct Preference Optimization (DPO) Alignment
- **Subheading:** Eliminating English code-switching and repetition loops with zero reward-model overhead

### 2. Visual Layout & Assets to Embed
- **Primary Asset:** `results/step3_dpo_margins_chart.png` (Distribution of log-probability margins $\Delta = \log \pi(y_w) - \log \pi(y_l)$ before vs. after DPO).
- **Secondary Asset:** Qualitative triplet comparison box (`dpo_37` held-out test triplet:
  - *Prompt:* `"Gikuptan sa inahan ang kamot sa iyang anak "`
  - *Chosen (Idiomatic Cebuano):* `"samtang naglakaw sila padulong sa gamay nga simbahan sa baryo."`
  - *Rejected (English Drift):* `"as they walked quietly along the narrow path near the river."`
  - *Log-Prob Margin ($\Delta$) Shift:* **`-7.0109` $\to$ `+5.0159` (`+12.0268` gain)**).
- **Mathematical Callout Box:** DPO closed-form objective vs. 4-model PPO RLHF architecture (using frozen merged Step 1 Cebuano LoRA as $\pi_{\text{ref}}$).

### 3. Key Content & Metric Cards
- **Cebuano Preference Dataset (`cebuano_dpo_preferences.jsonl`):**
  - **`38` Original Triplets:** `32` training pairs, `6` held-out evaluation pairs, spanning Conversational (`14`), News/Gov (`12`), and Literary/School (`12`) registers.
  - **Target Behavior:** Clean, idiomatic Cebuano continuation; suppressing English code-switching (`15` pairs), repetitive n-gram loops (`13` pairs), and subword gibberish (`10` pairs).
  - **100% Native-Speaker Verified:** Curated and QA'd by `@lejxz` to ensure authentic Cebuano syntax and case/focus morphology (`ang`, `mga`, `sa`, `nga`, `ug`, `og`, `aron`) without machine-translation artifacts.
- **DPO Empirical Gains:**
  - **Held-Out Test Preference Accuracy:** Shifted from **`0.0%` to `33.3%`** (**`+33.3 pp`**; `dpo_26` `-3.46 -> +2.67` and `dpo_37` `-7.01 -> +5.02` flipped positive), with **`100%`** implicit DPO reward accuracy (`eval_loss = 0.3502`, `eval_rewards/margins = +0.9105`).
  - **Test Mean Margin Shift:** Improved by **`+8.3521`** (`-27.5982` $\to$ `-19.2461`), with **all 6 held-out test pairs improving strictly** (`+2.88` to `+13.20`).
  - **Repetition & Diversity Shift:** Greedy 3-gram repetition dropped **`68.18%` $\to$ `66.76%`** (`-1.42 pp`) and sampled `distinct-2` lexical diversity improved from **`0.8897` to `0.9568`** (`+0.0671`).
  - **Perplexity Stability:** Validation perplexity remained stable at **`35.33`** (vs. `34.88` post-LoRA), demonstrating zero alignment tax or catastrophic forgetting.
- **DPO vs. Full RLHF (PPO):**
  $$\mathcal{L}_{\text{DPO}} = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right) \right]$$
  - *No Separate Reward Model:* Bypasses training and storing an auxiliary Bradley-Terry reward model.
  - *Lightweight Memory Footprint:* Requires only 2 models ($\pi_\theta$ and frozen $\pi_{\text{ref}}$) rather than 4 models in PPO (Policy, Value/Critic, Reference, and Reward Model).
  - *Consumer Hardware Feasibility:* 4 epochs completed in **126.8 seconds on CPU** (`beta=0.1`, `lr=5e-5`).

### 4. Oral Defense Speaker Notes
> *"In Step 3, we aligned the LoRA-adapted model using Direct Preference Optimization to address the failure modes identified in Steps 0 and 1: English code-switching, repetitive loops, and non-word subword blends. Our teammate Lej curated 38 original Cebuano preference triplets across conversational, news, and narrative registers, verified by native-speaker QA.*  
>  
> *Rather than executing complex online PPO reinforcement learning—which requires four models loaded in memory simultaneously—DPO analytically reparameterizes the reward function. This optimizes the policy directly via a binary cross-entropy loss over the reference policy ratio.*  
>  
> *In just 126 seconds of CPU training, test log-probability margins improved across 100% of held-out pairs, lifting test preference accuracy by 33.3 percentage points and increasing 2-gram lexical diversity from 0.89 to 0.96, all while keeping validation perplexity completely stable at 35.3."*

---

## Slide 5 — Step 4: Architectural Trade-Offs & Paradigm Comparison

### 1. Slide Headline & Subheading
- **Headline:** Step 4: Positional Embeddings & Model Architecture Trade-Offs
- **Subheading:** Analyzing DistilGPT2's context limits, modern relative encodings, and encoder vs. decoder paradigms

### 2. Visual Layout & Assets to Embed
- **Two-Column Comparative Diagram:**
  - *Left Column:* Positional Encoding Mechanisms (`wpe` vs. RoPE vs. ALiBi).
  - *Right Column:* Causal Attention Masking (Decoder) vs. Bidirectional Attention (Encoder).
- **Dataset Grounding Callout:** `CebuaNER` IOB tagging vs. open-ended Cebuano generative continuation.

### 3. Key Content & Metric Cards
- **Positional Encoding Comparison:**
  - *DistilGPT2 Learned Absolute (`wpe`):* Uses a static lookup matrix of **`786,432` parameters** ($1024 \times 768$). Hard-caps context at $1024$ tokens; cannot extrapolate. Due to Cebuano's high subword fertility (**$4.25$ tokens/word**), the effective context is restricted to **$\approx 241$ words**.
  - *Rotary Position Embedding (RoPE; Su et al., 2024):* Rotates Query and Key vectors in 2D chunks proportional to position $m$. **$0$ learned parameters**, relative-distance invariant, and naturally extensible to long sequences.
  - *Attention with Linear Biases (ALiBi; Press et al., 2022):* Injects a static linear distance penalty $-m(i - j)$ to pre-softmax logits. **$0$ parameters**, with exceptional zero-shot length extrapolation.
- **Decoder-Only vs. Encoder-Only Paradigms:**
  - *Decoder-Only (`distilgpt2`):* Employs lower-triangular causal masking ($j \le i$) for autoregressive next-token prediction. Essential for open-ended text continuation and interactive generation.
  - *Encoder-Only (`mBERT` / `XLM-R`):* Employs unconstrained bidirectional self-attention. Superior for sequence labeling and classification because predicting entity tags on **`CebuaNER`** (e.g., identifying `"Gwen"` as `B-PER` followed by `"Garcia"`) requires observing subsequent right-hand context.
- **Summary Architecture Matrix:**

| Feature | `distilgpt2` (`wpe`) | Modern Decoder (RoPE / ALiBi) | Encoder-Only (`mBERT` / `XLM-R`) |
| :--- | :--- | :--- | :--- |
| **Attention Mask** | Causal (Unidirectional) | Causal (Unidirectional) | Bidirectional (Full) |
| **Position Parameters** | $786{,}432$ ($1024 \times 768$) | **$0$ parameters** | $393{,}216$ ($512 \times 768$) |
| **Max Context Ceiling** | Strict $1024$ tokens (~$241$ words) | Extensible (32k+ tokens) | Strict $512$ tokens |
| **Best Suited Task** | Autoregressive generation | Long-context generation & RAG | Token classification / NER |

### 4. Oral Defense Speaker Notes
> *"To conclude in Step 4, we evaluated the structural foundations of DistilGPT2 against modern transformer architectures. DistilGPT2 uses learned absolute positional embeddings, dedicating over 786,000 parameters to a fixed 1,024-token lookup table. Because Cebuano words fragment into 4.25 tokens per word on average, this creates a severe bottleneck, capping our effective context window to roughly 241 words.*  
>  
> *Modern architectures resolve this using parameter-free relative mechanisms: RoPE rotates query and key vectors so attention scores depend purely on token distance, while ALiBi applies a static linear bias penalty enabling seamless length extrapolation beyond the training window.*  
>  
> *Finally, regarding model family: our project used a causal decoder-only architecture because our primary objective is open-ended text generation. However, for structured token-level tasks like CebuaNER entity tagging, encoder-only models like mBERT or XLM-RoBERTa remain superior, as bidirectional self-attention allows tokens to attend to subsequent right-side context necessary for resolving entity boundaries. Parameter-efficient fine-tuning via LoRA and alignment via DPO successfully bridge the low-resource gap on consumer hardware."*
