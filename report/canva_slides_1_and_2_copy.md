# Canva Slide Deck: Slides 1 and 2 Copy & Defense Speaker Notes

**Project:** Teaching DistilGPT2 Cebuano: Fine-Tuning, Decoding & Alignment  
**Author (Slides 1 & 2):** Nico (`@Pycnosis`)  
**Team Members:** Nico (`@Pycnosis`), Lej (`@lejxz`), Joshua (`@JoshyEyeBa11`)  
**Repository:** [https://github.com/Pycnosis/cebuano-distilgpt2](https://github.com/Pycnosis/cebuano-distilgpt2)

---

## Slide 1 — Step 0: Baseline Diagnostic ("Prove DistilGPT2 Doesn't Know Cebuano")

### 1. Slide Headline & Subheading
- **Headline:** Step 0: Baseline Diagnostic — Untouched `distilgpt2` on Cebuano
- **Subheading:** Quantifying BPE tokenizer fragmentation, subword breakdown, and held-out Cebuano perplexity before fine-tuning

### 2. Visual Layout & Assets to Embed
- **Primary Asset:** `results/step0_fragmentation_chart.png` (English vs. Cebuano Token Fertility and Characters/Token bar charts).
- **Secondary Asset:** Compact subword fragmentation callout (`kagamhanan` → `k | ag | am | han | an` [5 tokens] vs. `government` → `government` [1 token]).
- **Qualitative Callout Box:** Prompt `P4_literary` untouched continuation showing immediate gibberish + English code-switching (`"I am going to be writing about the issue of the state-owned water project..."`).

### 3. Key Content & Metric Cards
- **Base Model:** Stock `distilgpt2` (`81,912,576` parameters, 6 transformer decoder layers, `vocab_size=50,257`, max context `1,024` tokens).
- **Cebuano Corpus Prepared (`4,608` docs / `236,487` words / `535,319` BPE tokens):**
  - `josephimperial/CebuaNER` (`CC BY-NC 4.0`): `4,258` news blocks (`169,727` words; `371,628` BPE tokens), reconstructed by discarding CoNLL IOB NER tags.
  - `filbench/cebuano-readability` (`CC BY 4.0`): `350` literary & pedagogical passages (`66,760` words; `163,691` BPE tokens).
  - Stratified 90/10 split (`seed=42`): `480,130` train BPE tokens / `55,189` validation BPE tokens.
- **BPE Tokenizer Fragmentation Penalty:**
  - **12 Length-Matched Words:** Cebuano **`4.25` tokens/word** (`2.196` chars/token; `51` tokens) vs. English **`2.083` tokens/word** (`4.48` chars/token; `25` tokens) — **2.04× worse fragmentation**.
  - **5 Parallel Sentences:** Cebuano **`2.544` tokens/word** (`2.372` chars/token; `145` tokens) vs. English **`1.161` tokens/word** (`5.123` chars/token; `65` tokens) — **2.19× higher token fertility**.
- **Baseline Quantitative & Qualitative Failure:**
  - **Held-Out Cebuano Perplexity:** **`211.56`** (Val Loss `5.3545` across `219` blocks × `256` tokens = `56,064` tokens).
  - **Generation Failure Modes:** Echo loops (`nga nga nga...`, `kagamhanan sa Sugbo nga...`), syllable gibberish (`kapat`, `kabayon`, `kapunna lang`), and abrupt English drift on `P4`.

### 4. Oral Defense Speaker Notes
> *"Good day! Our midterm project adapts pretrained `distilgpt2`—an 82-million-parameter English language model—to Cebuano without training from scratch or requiring datacenter GPUs. In Step 0, our first objective was to establish a rigorous baseline proving the untouched model does not know Cebuano.*
>
> *We combined two properly licensed open Cebuano datasets—`CebuaNER` news articles and `cebuano-readability` literary/school passages—yielding 4,608 documents and 535,319 GPT-2 BPE tokens. Because GPT-2's 50,257-token BPE vocabulary was trained almost exclusively on English web text, Cebuano words are shattered into short character fragments: on 12 length-matched word pairs, Cebuano requires 4.25 tokens per word versus 2.08 for English—for example, `kagamhanan` splits into five pieces (`k|ag|am|han|an`).*
>
> *On our held-out Cebuano validation split, untouched `distilgpt2` records a high perplexity of 211.56, and its generations collapse into prompt-echo loops, nonsense syllables, and sudden switches into English prose. This establishes the exact baseline that Step 1 LoRA fine-tuning must overcome."*

---

## Slide 2 — Step 1: LoRA Fine-Tuning & Pretraining Scaling-Laws Analysis

### 1. Slide Headline & Subheading
- **Headline:** Step 1: LoRA Adaptation & Sidestepping Pretraining Scaling Laws
- **Subheading:** Cutting Cebuano validation perplexity by 83.5% while training only 1.94% of parameters in 4 minutes

### 2. Visual Layout & Assets to Embed
- **Primary Asset:** `results/step1_loss_curve.png` (Training loss `5.5938 -> 3.5291` & Validation loss convergence across 6 epochs / 714 steps).
- **Secondary Asset:** Before-vs-After KPI Strip (`PPL: 211.56 → 34.88`, `Trainable Params: 1.94%`, `Runtime: 253.4s`).
- **Before/After Text Comparison Box:** `P4_literary` (Step 0 English drift vs. Step 1 coastal Cebuano continuation `gamay nga sakyanan... sa dagat`) and `P5_pedagogical` (`Matod ni Enero, nahimutang ang mga buhatan sa eskwelahan alang sa lugar`).

### 3. Key Content & Metric Cards
- **Why From-Scratch Pretraining Fails (Scaling Laws):**
  - Chinchilla compute-optimal pretraining (Hoffmann et al., 2022) prescribes $D \approx 20N$: for `distilgpt2` ($N = 81.9\text{M}$), that requires **$\approx 1.64 \times 10^9$ tokens**.
  - Our Cebuano training split (`480,130` BPE tokens) is **$>3{,}000\times$ smaller** than Chinchilla-optimal scale—from-scratch pretraining would severely underfit syntax and overfit surface n-grams.
- **LoRA Configuration (`checkpoints/distilgpt2-cebuano-lora`):**
  - Low-Rank Adaptation (Hu et al., 2022) freezes all `81,912,576` base weights and updates $h = W_0 x + \frac{\alpha}{r} B A x$ on attention projections (`c_attn`, `c_proj`), `r=32`, `lora_alpha=64`, `lora_dropout=0.05`.
  - **Trainable Budget:** **`1,622,016` / `83,534,592` (`1.94%`)**; trained `6` epochs (`1,891` blocks × `256` tokens) in **`253.4 seconds`** on a single RTX 4050 laptop GPU.
- **Observed Step 1 Gains & Frozen-Tokenizer Insight:**
  - **Held-Out Perplexity:** **`211.56` $\to$ `34.88`** (**`-176.68 PPL` / `-83.5%`**; Val Loss `5.3545` $\to$ `3.5518`).
  - **Frozen Tokenizer Check:** Subword counts remain identical (`kagamhanan`: `5 → 5`, `eskwelahan`: `4 → 4`), proving LoRA teaches attention layers to *compose* fragmented subwords into valid Cebuano syntax (`ang`, `mga`, `sa`, `nga`, `gipahibalo`, `gipahigayon`).
  - **Native-Speaker Diagnostic (Lej):** Authentic Cebuano grammar & news/narrative registers emerge, while greedy decoding still hits n-gram attractor loops (`sa kapulisan...`)—directly motivating Step 2 decoding and Step 3 DPO alignment.

### 4. Oral Defense Speaker Notes
> *"In Step 1, we asked why we can teach `distilgpt2` Cebuano on just 480,000 training tokens when Chinchilla scaling laws say an 82-million-parameter model needs roughly 1.64 billion tokens to pretrain from scratch—over 3,000 times more data than we have.*
>
> *Low-Rank Adaptation solves this by exploiting low intrinsic dimensionality: we freeze all 81.9 million pretrained weights—retaining universal transformer primitives like causal routing and byte-level subword embeddings—and train low-rank rank-32 matrices solely on the attention projections (`c_attn` and `c_proj`). That accounts for just 1.94% of total parameters, cutting optimizer memory by roughly 98% and finishing 6 epochs in 253 seconds on a laptop GPU.*
>
> *The result is an 83.5% drop in held-out Cebuano perplexity—from 211.56 down to 34.88. Crucially, our tokenizer check confirms that subword splits do not change (`kagamhanan` stays 5 tokens); instead, LoRA re-routes attention heads to assemble those pieces into real Cebuano morphosyntax. Native-speaker review confirms authentic Cebuano phrasing, while also revealing greedy repetition traps that set up Step 2."*
