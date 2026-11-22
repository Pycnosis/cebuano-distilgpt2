# Step 4 Written Analysis: Positional Embeddings and Architectural Paradigms in Low-Resource Language Modeling

**Project:** Teaching DistilGPT2 Cebuano: Fine-Tuning, Decoding & Alignment  
**Target Model:** `distilgpt2` (82M parameters, 6 layers, 12 attention heads, $d_{\text{model}} = 768$, max context $T = 1024$)  
**Datasets:** `josephimperial/CebuaNER` (CC BY-NC 4.0), `filbench/cebuano-readability` (CC BY 4.0)

---

## 1. Introduction & Motivation

Modern Transformer language models rely fundamentally on two design choices that dictate their representational capacity, sequence length scaling, and task suitability:
1. **Positional Encoding Mechanism:** How the model injects token order into permutation-invariant self-attention operations.
2. **Attention Masking & Directionality:** Whether self-attention is causal (unidirectional autoregressive) or bidirectional (masked language modeling).

In this analysis, we evaluate the architectural foundations of `distilgpt2`—a direct 6-layer distillation of Radford et al.'s (2019) GPT-2 architecture—against modern alternatives across two concrete axes:
- **Axis 1:** Learned Absolute Positional Embeddings (`wpe`) versus modern relative schemes, specifically Rotary Position Embeddings (RoPE; Su et al., 2024) and Attention with Linear Biases (ALiBi; Press et al., 2022).
- **Axis 2:** Decoder-Only Causal Architectures versus Encoder-Only Bidirectional Architectures (e.g., mBERT, XLM-RoBERTa), analyzed through the specific lens of our two Cebuano corpus sources: open-ended generation vs. token classification on `CebuaNER` and passage classification on `cebuano-readability`.

---

## 2. Axis 1: Positional Embeddings (`wpe` vs. RoPE vs. ALiBi)

### 2.1 Theoretical Framework

Because the standard dot-product self-attention operation,
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^\top}{\sqrt{d_k}}\right)V$$
is permutation-equivariant with respect to the input sequence, position information must be explicitly injected. The mechanism chosen fundamentally governs whether the model understands absolute position, relative token distance, and whether it can extrapolate to sequence lengths unseen during training.

```
1. Learned Absolute (wpe):
   Token x_i  ──► Embedding e_i + Positional Vector p_i ──► Standard Q, K, V

2. Rotary Position Embedding (RoPE):
   Token x_m  ──► Q_m = R_{\Theta, m}^d (W_q x_m),   K_n = R_{\Theta, n}^d (W_k x_n)
   Attention Score: (Q_m)^\top K_n = x_m^\top W_q^\top R_{\Theta, m-n}^d W_k x_n   [Depends purely on (m - n)]

3. Attention with Linear Biases (ALiBi):
   Attention Score: q_i k_j^\top - m \cdot |i - j|   [Static linear penalty on relative distance]
```

---

### 2.2 DistilGPT2 / GPT-2 Learned Absolute Positional Embeddings (`wpe`)

GPT-2 and `distilgpt2` inherit the original architecture of Radford et al. (2019), implementing a learned positional embedding matrix $W_{\text{pos}} \in \mathbb{R}^{T_{\text{max}} \times d_{\text{model}}}$, where $T_{\text{max}} = 1024$ and $d_{\text{model}} = 768$. For an input token sequence $(x_0, x_1, \dots, x_{T-1})$, the input representation to the first transformer block is:

$$h_i^{(0)} = W_{\text{token}}[x_i] + W_{\text{pos}}[i], \quad i \in \{0, 1, \dots, T-1\}$$

#### Architectural Characteristics & Limitations:
1. **Dedicated Parameter Budget:** The `wpe` matrix requires $1024 \times 768 = 786{,}432$ parameters. In `distilgpt2` (total parameters: $83{,}534{,}592$), `wpe` accounts for approximately **$0.94\%$** of the entire parameter budget.
2. **Strict Hard Context Horizon:** The lookup table is fixed at dimension $1024$. The model structurally **cannot process or generate a sequence of length $T > 1024$ tokens** without re-initializing or interpolating weights. An input at index $1025$ triggers an index out-of-bounds error.
3. **Absolute vs. Relative Representation:** The inner product between query at position $m$ and key at position $n$ expands into four additive components:
   $$q_m^\top k_n = (W_q (e_m + p_m))^\top (W_k (e_n + p_n)) = e_m^\top W_q^\top W_k e_n + e_m^\top W_q^\top W_k p_n + p_m^\top W_q^\top W_k e_n + p_m^\top W_q^\top W_k p_n$$
   Notice that the interaction depends on the absolute indices $m$ and $n$ directly, rather than their relative distance $(m - n)$. The model must independently learn that token pairs separated by distance $k = m - n$ share syntactic relations across different absolute starting positions $m$.
4. **Impact on Low-Resource Cebuano:** In Step 0, our diagnostic revealed that GPT-2's byte-level BPE tokenizer fragments Cebuano words into an average of **$4.25$ tokens per word** (compared to $2.08$ tokens/word for length-matched English words). Because of this severe subword fragmentation, `distilgpt2`'s $1024$-token window can accommodate at most:
   $$\frac{1024 \text{ tokens}}{4.25 \text{ tokens/word}} \approx 241 \text{ Cebuano words}$$
   In contrast, an equivalent English text yields $\frac{1024}{2.08} \approx 492$ words. The learned `wpe` bottleneck severely constrains the effective narrative context for Cebuano texts to roughly 4–5 short paragraphs before truncation occurs.

---

### 2.3 Rotary Position Embedding (RoPE; Su et al., 2024)

Adopted by virtually all modern open-weight architectures (LLaMA 1–3, Mistral, Qwen 2.5, Gemma 2), Rotary Position Embedding encodes relative position by rotating the query and key representations in complex vector space rather than adding vectors to token embeddings.

Given a 2D slice of a query vector $q_m = [q_m^{(1)}, q_m^{(2)}]^\top$ at position $m$, RoPE applies an orthogonal rotation matrix $R_{\theta, m}$:

$$R_{\theta, m} = \begin{pmatrix} \cos(m\theta) & -\sin(m\theta) \\ \sin(m\theta) & \cos(m\theta) \end{pmatrix}$$

For a $d$-dimensional head vector, RoPE applies a block-diagonal matrix $R_{\Theta, m}^d$ with frequencies $\theta_i = 10000^{-2(i-1)/d}$:

$$\tilde{q}_m = R_{\Theta, m}^d (W_q x_m), \quad \tilde{k}_n = R_{\Theta, n}^d (W_k x_n)$$

The inner product between query $\tilde{q}_m$ and key $\tilde{k}_n$ satisfies:

$$\langle \tilde{q}_m, \tilde{k}_n \rangle = (R_{\Theta, m}^d W_q x_m)^\top (R_{\Theta, n}^d W_k x_n) = x_m^\top W_q^\top \left( R_{\Theta, m}^d \right)^\top R_{\Theta, n}^d W_k x_n = x_m^\top W_q^\top R_{\Theta, n-m}^d W_k x_n$$

#### Architectural Advantages:
1. **Zero Learned Parameters:** RoPE introduces **0 additional parameters** into the model graph. Query and key vectors are rotated deterministically using closed-form trigonometric operations.
2. **Pure Relative Distance Invariance:** The attention score depends strictly on the relative offset $(m - n)$ through $R_{\Theta, n-m}^d$, directly capturing linguistic syntactic head-dependent distances regardless of where the clause appears in a 32k-token document.
3. **Decay with Distance:** As $|m - n|$ increases, the inner product naturally decays, encoding the inductive bias that adjacent words are more syntactically and semantically related than distant words.
4. **Context Length Extensibility:** While vanilla RoPE degrades when evaluating beyond its pretraining context, modern frequency-scaling techniques (NTK-aware RoPE scaling, YaRN) allow models pretrained at 4k context to extrapolate cleanly to 32k–128k tokens without retraining from scratch.

---

### 2.4 Attention with Linear Biases (ALiBi; Press et al., 2022)

ALiBi (employed in MPT-7B and BLOOM) takes an even more minimalist approach by completely removing position embeddings from the input and query/key projections. Instead, it injects a static, non-learned linear penalty directly into the pre-softmax attention logits based on key-query distance $|i - j|$:

$$A_{i, j} = q_i k_j^\top - m \cdot (i - j), \quad \text{for } j \le i$$

where $m$ is a head-specific geometric slope fixed during model definition:
$$m \in \left\{ 2^{-8/H}, 2^{-16/H}, \dots, 2^{-8} \right\}$$
for $H$ attention heads.

#### Architectural Advantages:
1. **Zero Learned Parameters:** Exactly **0 parameters** required; slopes $m$ are static geometric constants.
2. **Strong Zero-Shot Length Extrapolation:** Unlike `wpe` (which crashes at $T > 1024$) and vanilla RoPE (which requires interpolation), ALiBi allows models trained on sequences of length $T = 512$ or $1024$ to extrapolate directly to $T = 2048$ or $4096$ at test time with stable perplexity and zero additional fine-tuning.
3. **Computational Simplicity:** Injects a simple subtraction during scaled dot-product attention before softmax, avoiding trigonometric rotations.

---

### 2.5 Cross-Mechanism Comparison Table

| Feature / Dimension | GPT-2 / DistilGPT2 (`wpe`) | RoPE (Su et al., 2024) | ALiBi (Press et al., 2022) |
| :--- | :--- | :--- | :--- |
| **Position Encoding Type** | Absolute (Learned Lookup Table) | Relative (Orthogonal Rotation Matrix) | Relative (Linear Logit Bias Penalty) |
| **Where Injected** | Input layer: $h^{(0)} = e_{\text{token}} + e_{\text{pos}}$ | Attention heads: rotated $\tilde{q}_m, \tilde{k}_n$ | Attention matrix: $q_i k_j^\top - m(i - j)$ |
| **Parameter Overhead** | **$786{,}432$ params** ($1024 \times 768$) | **$0$ params** | **$0$ params** |
| **Relative Distance Invariance** | ✗ No (dependent on absolute $m, n$) | ✓ Yes (inner product depends on $m - n$) | ✓ Yes (penalty is linear in $i - j$) |
| **Maximum Context (Base)** | Strict hard ceiling: **$1024$ tokens** | Defined by training window (e.g., $4096$) | Defined by training window |
| **Zero-Shot Length Extrapolation**| ✗ Impossible (index out-of-bounds) | Moderate (requires NTK / YaRN scaling) | **Excellent** (evaluates seamlessly at $2\times$–$4\times$) |
| **Cebuano Effective Context** | **$\approx 241$ words** (due to $4.25$ tok/word) | Extensible to thousands of words | Extensible to thousands of words |
| **Representative Models** | GPT-2, DistilGPT2, OPT | LLaMA 1–3, Mistral, Qwen 2.5, Gemma 2 | MPT-7B/30B, BLOOM, Falcon |

---

## 3. Axis 2: Decoder-Only Causal LM vs. Encoder-Only Masked LM

### 3.1 Attention Masking & Directionality

The fundamental distinction between decoder-only architectures (e.g., `distilgpt2`) and encoder-only architectures (e.g., `bert-base-multilingual`, `xlm-roberta-base`) lies in the structure of the self-attention connectivity matrix $M \in \mathbb{R}^{T \times T}$:

```
Decoder-Only (Causal / Unidirectional):          Encoder-Only (Bidirectional / Full):
M_{i,j} = { 0   if j <= i                       M_{i,j} = 0   for all (i, j)
          { -inf if j > i

[  0 -inf -inf -inf ]                            [  0   0   0   0 ]
[  0   0  -inf -inf ]                            [  0   0   0   0 ]
[  0   0   0  -inf ]                            [  0   0   0   0 ]
[  0   0   0   0  ]                            [  0   0   0   0 ]
Token i attends ONLY to tokens <= i              Token i attends to ALL tokens 0 .. T-1
Objective: Next-Token Prediction (p(w_t | w_<t))  Objective: Masked Language Modeling (p(w_mask | w_\mask))
```

---

### 3.2 Grounded Comparison on Our Project Datasets

To evaluate why `distilgpt2` was chosen for this project—and where an encoder-only model would be superior—we examine our two corpus sources:

#### 1. Corpus Source A: `josephimperial/CebuaNER` (News Register; 4,258 documents)
- **Original Task:** Named Entity Recognition (NER), sequence labeling in IOB format (`B-PER`, `I-PER`, `B-ORG`, `I-ORG`, `B-LOC`, `I-LOC`).
- **Why Encoder-Only Excels for NER:**
  - In sequence labeling, predicting the entity type of a token $w_t$ requires looking both backward *and* forward. Consider the Cebuano phrase:
    $$\text{"Matud pa ni } \underbrace{\text{Gwen}}_{\text{B-PER}} \underbrace{\text{Garcia}}_{\text{I-PER}} \text{ sa Kapitolyo..."}$$
    To classify `"Gwen"` as `B-PER`, an encoder model benefits from seeing the subsequent surname `"Garcia"`. Conversely, to classify an ambiguous capitalized word, subsequent context (e.g., verbs or titles occurring *to the right*) resolves whether it is a person, municipality, or organization.
  - A decoder-only model with causal masking cannot look ahead to $w_{t+1}$ when computing the hidden representation $h_t$. Predicting token-level tags autoregressively forces the model to classify entities with strictly unilateral left context, yielding suboptimal boundary detection.
  - Thus, for the original `CebuaNER` task, **mBERT or XLM-RoBERTa is theoretically and empirically superior**.
- **Why Decoder-Only Was Used in Our Project:**
  - Our project goal was **not** token classification; we deliberately discarded the NER tags in `scripts/prepare_corpus.py` to reconstruct raw news sentences for **generative autoregressive modeling**. For generating open-ended news text (e.g., continuing Prompt `P2` `"Gipahibalo sa lokal nga kagamhanan sa Sugbo nga..."`), a causal decoder is required because tokens are generated sequentially from left to right.

#### 2. Corpus Source B: `filbench/cebuano-readability` (Literary / Pedagogical Prose; 350 documents)
- **Original Task:** Readability assessment (classifying a reading passage into target grade levels or readability scores).
- **Encoder-Only Suitability:** Passage-level classification maps an entire document to a categorical label. An encoder model utilizes bidirectional self-attention to pool information across all sentences into the `[CLS]` token representation, capturing global vocabulary difficulty and syntactic complexity holistically.
- **Decoder-Only Role in Our Project:** In our pipeline, `cebuano-readability` provided high-quality literary narrative register text to balance the news-heavy register of `CebuaNER`. Fine-tuning `distilgpt2` on these passages enabled the model to maintain narrative continuity without collapsing into repetitive news attribution formulas (as demonstrated in Prompt `P4` `"Sa usa ka gamay nga baryo duol sa dagat, adunay..."`).

---

### 3.3 Structural & Task Suitability Comparison Table

| Feature / Dimension | Decoder-Only (`distilgpt2`) | Encoder-Only (`mBERT` / `XLM-R` / `Tagalog-RoBERTa`) |
| :--- | :--- | :--- |
| **Self-Attention Mask** | Lower-triangular causal mask ($j \le i$) | Unconstrained bidirectional mask (all $j$) |
| **Primary Pretraining Task** | Causal Language Modeling (CLM): Next-token prediction | Masked Language Modeling (MLM): Cloze completion |
| **Generation Capability** | **Native Autoregressive Generation:** Generates token-by-token via KV-cache | **Non-Generative:** Generating text requires inefficient iterative Gibbs-style sampling |
| **Token Representation Context** | Past context only ($w_1, \dots, w_t$) | Full bidirectional context ($w_1, \dots, w_T$) |
| **Token Classification (`CebuaNER`)** | Suboptimal (cannot inspect right context for entity resolution) | **Optimal:** Full syntactic field resolves entity boundaries |
| **Sequence Classification (`readability`)** | Read from final token representation $h_T$ | Read from pooled `[CLS]` representation with bidirectional attention |
| **Autoregressive Text Continuation** | **Optimal:** Steps 1–3 LoRA & DPO generative pipeline | Inapplicable for standard open-ended text generation |
| **KV-Cache Feasibility** | Enables $O(T)$ autoregressive decoding via cached Key/Value tensors | N/A (processes entire sequence in a single forward pass) |
| **Parameter Efficiency in Our Project** | Adapted $1.62\text{M}$ LoRA parameters ($1.94\%$) on lightweight $82\text{M}$ base | Requires full fine-tuning or cross-lingual classification heads |

---

## 4. Synthesis & Architectural Takeaways for Cebuano NLP

1. **The Subword Multiplier Amplifies Context Bottlenecks:**
   The combination of a high subword fragmentation rate (**$4.25$ tokens/word** in Cebuano vs. $2.08$ in English) with `distilgpt2`'s static learned absolute positional embedding (`wpe`, $T_{\text{max}} = 1024$) produces an effective ceiling of roughly **$241$ Cebuano words**. Future Cebuano language modeling efforts should adopt **RoPE** or **ALiBi** backbones (such as LLaMA-3 or Qwen-2.5) to enable long-context document understanding without parameter-wasteful lookup tables.

2. **Task-Architecture Alignment:**
   Decoder-only models are non-negotiable for generative tasks (free-form continuation, dialogue, creative writing), making `distilgpt2` the correct architectural vehicle for evaluating LoRA parameter adaptation (Step 1), decoding strategies (Step 2), and preference optimization (Step 3). Conversely, for benchmark evaluation on structured Cebuano NLP datasets like `CebuaNER`, encoder-only models remain the gold standard due to bidirectional context availability.

3. **Portability of Alignment Pipelines:**
   The parameter-efficient LoRA fine-tuning and Direct Preference Optimization (DPO) methodologies verified in this project operate entirely on attention projections ($\Delta W = BA$) and log-probability likelihood ratios. Consequently, the techniques demonstrated here transfer directly to modern RoPE-based decoder architectures without modification.

---

## References

- Aghajanyan, A., Zettlemoyer, L., & Gupta, S. (2020). *Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning*. arXiv preprint arXiv:2012.13255.
- Conneau, A., Khandelwal, K., Goyal, N., Chaudhary, V., Wenzek, G., Guzmán, F., Grave, E., Ott, M., Zettlemoyer, L., & Stoyanov, V. (2020). *Unsupervised Cross-lingual Representation Learning at Scale* (XLM-RoBERTa). Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics (ACL 2020), 8440–8451.
- Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. Proceedings of NAACL-HLT 2019, 4171–4186.
- Hoffmann, J., Borgeaud, S., Mensch, A., Buchatskaya, E., Cai, T., Rutherford, E., de Las Casas, D., Hendricks, L. A., Welbl, J., Clark, A., Hennigan, T., Noland, E., Millican, K., van den Driessche, G., Damoc, B., Guy, A., Osindero, S., Simonyan, K., Elsen, E., Rae, J. W., Vinyals, O., & Sifre, L. (2022). *Training Compute-Optimal Large Language Models* (Chinchilla). arXiv preprint arXiv:2203.15556.
- Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2022). *LoRA: Low-Rank Adaptation of Large Language Models*. International Conference on Learning Representations (ICLR 2022). arXiv:2106.09685.
- Imperial, J. M., & Ong, E. (2020). *Baseline Readability Assessment Model for Cebuano*. Proceedings of the 1st Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics (AACL-IJCNLP 2020). Dataset: `filbench/cebuano-readability` (Licensed under CC BY 4.0).
- Kaplan, J., McCandlish, S., Henighan, T., Brown, T. B., Chess, B., Child, R., Gray, S., Radford, A., Wu, J., & Amodei, D. (2020). *Scaling Laws for Neural Language Models*. arXiv preprint arXiv:2001.08361.
- Pilar, M. B. E. N., & Imperial, J. M. (2023). *CebuaNER: A New Baseline Cebuano Named Entity Recognition Model*. Proceedings of the 37th Pacific Asia Conference on Language, Information and Computation (PACLIC 37). Dataset: `josephimperial/CebuaNER` (Licensed under CC BY-NC 4.0).
- Press, O., Smith, N. A., & Lewis, M. (2022). *Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation* (ALiBi). International Conference on Learning Representations (ICLR 2022). arXiv:2108.12409.
- Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I. (2019). *Language Models are Unsupervised Multitask Learners* (GPT-2). OpenAI Technical Report.
- Rafailov, R., Sharma, A., Mitchell, E., Manning, C. D., Ermon, S., & Finn, C. (2023). *Direct Preference Optimization: Your Language Model is Secretly a Reward Model*. Advances in Neural Information Processing Systems (NeurIPS 2023). arXiv:2305.18290.
- Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). *DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter*. arXiv preprint arXiv:1910.01108.
- Su, J., Ahmed, M., Lu, Y., Pan, S., Bo, W., & Liu, Y. (2024). *RoFormer: Enhanced Transformer with Rotary Position Embedding* (RoPE). *Neurocomputing*, 568, 127063. arXiv:2104.09864.
- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). *Attention Is All You Need*. Advances in Neural Information Processing Systems (NeurIPS 2017), 5998–6008.
