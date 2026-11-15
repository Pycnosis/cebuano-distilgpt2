# Teaching DistilGPT2 Cebuano: Fine-Tuning, Decoding & Alignment

Midterm Project — Adapting pretrained `distilgpt2` (82M parameters) to Cebuano (Bisaya) using LoRA fine-tuning, decoding strategy benchmarking + KV-cache latency measurement, and lightweight DPO alignment.

## Team Members

- **Nico** ([@Pycnosis](https://github.com/Pycnosis)) — Modeling Pipeline & Decoding Benchmarks
- **Lej** — Native-Speaker Linguistic Sanity Checks & DPO Preference Alignment
- **Joshua** ([@JoshyEyeBa11](https://github.com/JoshyEyeBa11)) — Architecture Analysis, Technical Reporting & Presentation Lead

## Project Deliverables & Links

- **Repository:** [https://github.com/Pycnosis/cebuano-distilgpt2](https://github.com/Pycnosis/cebuano-distilgpt2)
- **Canva Presentation Deck (5 Slides):** [Canva Presentation Link (Placeholder / View Deck)](https://www.canva.com/design/)
- **Course Instructor Invitation:** `evm@usjr.edu.ph`

## Project Structure

```text
cebuano-distilgpt2/
├── data/
│   ├── cebuano_corpus.txt             # Combined CebuaNER + cebuano-readability corpus
│   ├── cebuano_corpus_train.txt       # 90% train split
│   ├── cebuano_corpus_val.txt         # 10% held-out validation split
│   └── corpus_stats.json              # Dataset statistics & BPE token counts
├── scripts/
│   ├── prepare_corpus.py              # Step 1A: Load HF datasets, detokenize & merge corpus
│   ├── step0_baseline.py              # Step 0: Tokenizer fragmentation, generation & baseline PPL
│   └── step1_train_lora.py            # Step 1B: LoRA fine-tuning, loss curve & before/after eval
├── notebooks/
│   └── week1_step0_and_step1.ipynb    # Combined Week 1 interactive notebook
├── checkpoints/
│   └── distilgpt2-cebuano-lora/       # Saved PEFT LoRA adapter & tokenizer
├── results/
│   ├── step0_tokenizer_comparison.csv # English vs. Cebuano BPE fertility table
│   ├── step0_fragmentation_chart.png  # Tokenizer fragmentation visual comparison
│   ├── step0_baseline_generations.md  # Untouched DistilGPT2 outputs
│   ├── step1_loss_curve.png           # Training & validation loss curve
│   └── step1_before_after_eval.md     # Before vs. after LoRA comparison
└── report/
    └── step1_scaling_laws_writeup.md  # Step 1 scaling-laws & LoRA writeup
```

## Setup & Reproduction

```bash
pip install -r requirements.txt
```

### Week 1 Pipeline
1. **Prepare Cebuano Corpus:**
   ```bash
   python scripts/prepare_corpus.py
   ```
2. **Run Step 0 Baseline Diagnostic:**
   ```bash
   python scripts/step0_baseline.py
   ```
3. **Run Step 1 LoRA Fine-Tuning & Evaluation:**
   ```bash
   python scripts/step1_train_lora.py
   ```

## Week 1 Results Summary (Step 0 & Step 1)

| Metric | Value | Notes |
| :--- | :--- | :--- |
| **Combined Corpus** | `4,608` docs / `236,487` words / `535,319` BPE tokens | `4,258` CebuaNER blocks + `350` readability passages |
| **Train / Val Split (90/10)** | `480,130` / `55,189` BPE tokens | Stratified across registers (`seed=42`) |
| **Word Fertility (12 Pairs)** | English `2.08` vs. Cebuano `4.25` tokens/word | Unchanged by LoRA (frozen BPE table) |
| **Sentence Fertility (5 Pairs)** | English `1.16` vs. Cebuano `2.54` tokens/word | `2.37` vs. `5.12` chars/token |
| **LoRA Trainable Budget** | `1,622,016 / 83,534,592` (`1.94%`) | `r=32`, `alpha=64`, `c_attn` + `c_proj`, 6 epochs |
| **Step 0 Baseline Perplexity** | `211.56` (Val Loss `5.3545`) | Stock `distilgpt2` on `cebuano_corpus_val.txt` |
| **Step 1 Post-LoRA Perplexity** | **`34.88`** (Val Loss `3.5518`) | **-176.68 PPL (-83.5%)** |

## Corpus Sources & Licenses

- **CebuaNER** (`josephimperial/CebuaNER`): 4,258 Cebuano news articles (Yes the Best Dumaguete, Filipinas Bisaya, Sunstar Cebu) originally IOB-tagged for NER. Plain sentences reconstructed by discarding NER tags. License: **CC BY-NC 4.0** (academic/non-commercial use).
- **Cebuano Readability** (`filbench/cebuano-readability`): 350 literary and pedagogical passages from Bloom Library, Let's Read Asia, and DepEd materials. License: **CC BY 4.0**.
