# Step 1 Evaluation: Before vs. After Cebuano LoRA Fine-Tuning

## 1. LoRA Parameter Budget & Training Summary
- **Base Model:** `distilgpt2` (81,912,576 frozen parameters)
- **LoRA Config:** `r=32`, `lora_alpha=64`, `lora_dropout=0.05`, `target_modules=['c_attn', 'c_proj']`, `fan_in_fan_out=True`
- **Trainable LoRA Parameters:** 1,622,016 / 83,534,592 (**1.94%**)
- **Initial Logged Train Loss:** `5.5938` → **Final Train Loss:** `3.5291`

## 2. Quantitative Comparison: Held-Out Validation Loss & Perplexity
| Model Stage | Validation Loss | Validation Perplexity | Change |
| :--- | ---: | ---: | :--- |
| **Step 0: Untouched `distilgpt2`** | `5.3545` | `211.56` | Baseline |
| **Step 1: LoRA Fine-Tuned `distilgpt2`** | `3.5518` | `34.88` | **-176.68 PPL (-83.5%)** |

## 3. Tokenizer Fragmentation Check (Before vs. After LoRA)
> **Key Architectural Insight:** LoRA adapts attention projection matrices (`W + BA`), leaving the BPE merge rules and 50,257-token embedding vocabulary untouched. Therefore, token counts per word remain identical after Step 1, while the transformer's attention layers learn to compose those fragmented subwords into coherent Cebuano sequences.

| Cebuano Word | Step 0 Token Count | Step 1 Token Count | Subword Pieces (Unchanged) |
| :--- | ---: | ---: | :--- |
| `kagamhanan` | 5 | 5 | `k | ag | am | han | an` |
| `eskwelahan` | 4 | 4 | `es | kw | el | ahan` |
| `kapulisan` | 4 | 4 | `k | ap | ul | isan` |
| `tinun-an` | 4 | 4 | `tin | un | - | an` |
| `malipayon` | 4 | 4 | `mal | ip | ay | on` |
| `panaghiusa` | 4 | 4 | `pan | agh | ius | a` |

## 4. Qualitative Comparison: Before vs. After on 5 Cebuano Prompts

### P1_conversational (Conversational)
- **Prompt:** `Kumusta ka? Manghinaut ko nga`
- **Step 0 Greedy (Before):** nga nga nga nga nga nga nga nga nga nga nga nga nga nga nga nga nga nga nga nga nga nga nga nga nga nga nga nga nga nga
- **Step 1 Greedy (After LoRA):** nagkalit ang mga kalibutan?
Nagkalibutan ang mga kalibutan?
Nagkalibutan ang mga kalibutan?
Nagkalibutan ang mga kalibutan?
N
- **Step 0 Sampled (Before):** ji ko ka?


Manga na nga, na nga ko, na kapunna lang na kapunna lang na nga, na kapunna lang na kapunna lang nga, na kapunna lang nga, na
- **Step 1 Sampled (After LoRA):** si Della.
Nakahubag na si Della ang iyang nakahayag si Della.
Nahunahuna niya si Della.
Nagkaon niya si Della.
Nakahuna niya si Della.

### P2_news_gov (News (Local Government))
- **Prompt:** `Gipahibalo sa lokal nga kagamhanan sa Sugbo nga`
- **Step 0 Greedy (Before):** kagamhanan sa Sugbo nga kagamhanan sa Sugbo nga kagamhanan sa Sugbo nga kagamhanan sa Sugbo nga kagamhanan sa Sugbo nga kagamhanan sa Sugbo nga
- **Step 1 Greedy (After LoRA):** gipahibalo sa kapulisan sa kapulisan sa kapulisan sa kapulisan sa kapulisan sa kapulisan sa kapulisan sa kapulisan sa kapulisan sa kapulisan sa kapulisan
- **Step 0 Sampled (Before):** kagamhanan sa Sugbo nga kagamhanan sa Sugbo nga kagamhanan sa Sugbo nga kagamhanan sa Sugbo nga kagamhanan sa Sugbo nga kagamhanan sa Sugbo nga
- **Step 1 Sampled (After LoRA):** gipahimutang sa mga pagsalit sa mga hulagway nga nagpangutog sa ilang kapulisan sa kapulisan sa mga gipahibalo nga gipahimutang sa mga p

### P3_news_police (News (Police Report))
- **Prompt:** `Matud pa sa kapulisan sa dakbayan,`
- **Step 0 Greedy (Before):** sa kapulisan sa dakbayan, sa kapulisan sa dakbayan, sa kapulisan sa dakbayan, sa kapulisan sa dakbayan, sa kapulisan sa dakbayan, sa kapulisan
- **Step 1 Greedy (After LoRA):** nga nagpuyo sa mga kapulisan sa kapulisan sa kapulisan sa kapulisan sa kapulisan sa kapulisan sa kapulisan sa kapulisan sa kapulisan sa kapulisan sa
- **Step 0 Sampled (Before):** pa sa kapulisan sa dakbayan, pa sa kapulisan sa dakbayan, pa sa kapulisan sa dakbayan, pa sa kapulisan sa dakbayan, pa sa kapulisan sa dakbayan,
- **Step 1 Sampled (After LoRA):** nasud nga ilabi sa kapulisan nga sa ilabi. Gipag-abot na sab niini ang mga kompuyente ug buntag sa kaso. Gipatak na sa kapulisan ang kapulisan sa

### P4_literary (Literary / Narrative)
- **Prompt:** `Sa usa ka gamay nga baryo duol sa dagat, adunay`
- **Step 0 Greedy (Before):** na kapat na kapat na kapat na kapat na kapat na kapat na kapat na kapat na kapat na kapat na kapat na kapat na kapat na kapat na kapat
- **Step 1 Greedy (After LoRA):** mga kalibutan sa mga kalibutan sa mga kalibutan. Sumala pa sa mga kalibutan, gipahigayon niini ang mga kalibutan sa mga kalibutan sa mga kalib
- **Step 0 Sampled (Before):** sa ao na sabay, katay ditakay ka dar, nakay ka sabay.

I am going to be writing about the issue of the state-owned water project, and I am not going to be giving a talk on that. It will be
- **Step 1 Sampled (After LoRA):** gamay nga sakyanan, apan nakita ka dili nag-ampay sa dagat, apan nakita ka mga buhaton, apan nakita ka paggamit, apan nga sakyanan, apan

### P5_pedagogical (Pedagogical / School)
- **Prompt:** `Ang mga bata sa eskwelahan malipayon nga`
- **Step 0 Greedy (Before):** bata sa kapat kapat kapat kapat kapat kapat kapat kapat kapat kapat kapat kapat kapat kapat kapat kapat kapat kapat kapat
- **Step 1 Greedy (After LoRA):** gipahigayon sa kapulisan sa kapulisan sa kapulisan sa kapulisan sa kapulisan sa kapulisan sa kapulisan sa kapulisan sa kapulisan sa kapulisan sa kapul
- **Step 0 Sampled (Before):** kabayon nga kabayon mga bata sa eskwelahan malipayon nga kabayon mga kabayon mga kabayon mga kabayon mga kabayon mga kabayon m
- **Step 1 Sampled (After LoRA):** kabii ug nagpasabot ug nakita-bata sa kabii. Matod ni Enero, nahimutang ang mga buhatan sa eskwelahan alang sa lugar. Ang dakbayan nga k

## Native-Speaker Linguistic Review (Lej)

### 1. Per-Prompt Cebuano Evaluation (`P1`–`P5`)

- **`P1_conversational` (`"Kumusta ka? Manghinaut ko nga"`):**
  - **Grammatical & Lexical Gains:** Step 0 collapsed into `nga nga nga...` (greedy) and syllable noise (`ji ko ka?`, `kapunna lang`, sampled). After LoRA, the model outputs real Cebuano words and grammatical markers (`ang`, `mga`, `si`, `iyang`, `niya`, and verb prefixes `Naka-`, `Nag-`, `Nahu-`).
  - **Remaining Issues:** Greedy gets trapped repeating `nagkalit/Nagkalibutan ang mga kalibutan?`, while sampled switches into third-person narrative (`si Della`) with some unnatural verb-argument agreement (`Nagkaon niya si Della`) and non-standard forms (`Nakahubag`, `Nakahuna`). Instead of answering `"Kumusta ka?"`, it continues as prose.

- **`P2_news_gov` (`"Gipahibalo sa lokal nga kagamhanan sa Sugbo nga"`):**
  - **Grammatical & Lexical Gains:** Step 0 merely echoed the prompt tail (`kagamhanan sa Sugbo nga...`). Step 1 produces authentic Cebuano news-register collocations and passive/locative verb forms (`gipahibalo`, `gipahimutang`), plural/case markers (`sa mga`, `nga`, `ilang`), and domain nouns (`kapulisan`, `hulagway`).
  - **Remaining Issues:** Greedy locks into `sa kapulisan sa kapulisan...`; sampled stays topically grounded in local news but still invents occasional subword blends (`pagsalit`, `nagpangutog`).

- **`P3_news_police` (`"Matud pa sa kapulisan sa dakbayan,"`):**
  - **Grammatical & Lexical Gains:** Overcomes Step 0's prompt-copying loop by introducing valid relative clauses (`nga nagpuyo sa mga...`), discourse particles (`na sab`, `niini`), conjunctions (`ug`), and police-report vocabulary (`sa kaso`, `ang kapulisan`).
  - **Remaining Issues:** Greedy loops on `sa kapulisan`; sampled captures newsroom cadence well despite minor lexical/syntactic slips (`kompuyente`, `Gipatak`, `nga sa ilabi`).

- **`P4_literary` (`"Sa usa ka gamay nga baryo duol sa dagat, adunay"`):**
  - **Grammatical & Lexical Gains:** Completely eliminates Step 0's severe English code-switching (`"I am going to be writing about the issue of the state-owned water project..."`). Following the existential `adunay`, sampled supplies a contextually fitting coastal noun phrase (`gamay nga sakyanan`) paired with `sa dagat`, `paggamit`, and the adversative linker `apan`. Greedy introduces valid news-style attribution (`Sumala pa sa...`, `gipahigayon niini ang...`).
  - **Remaining Issues:** Greedy over-repeats `mga kalibutan`, and sampled recycles the clause frame `apan nakita ka...` with awkward `ka` pronoun placement.

- **`P5_pedagogical` (`"Ang mga bata sa eskwelahan malipayon nga"`):**
  - **Grammatical & Lexical Gains:** Replaces Step 0's gibberish loops (`kapat`, `kabayon`) with coherent Cebuano function words (`ug`, `alang sa`, `Matod ni`) and school/community vocabulary (`mga buhatan sa eskwelahan`, `lugar`, `dakbayan`). Notably, `"Matod ni Enero, nahimutang ang mga buhatan sa eskwelahan alang sa lugar"` is a grammatically well-formed Cebuano sentence.
  - **Remaining Issues:** Greedy abruptly shifts domain to `gipahigayon sa kapulisan`, reflecting the corpus imbalance (**4,258** CebuaNER news blocks vs. **350** Cebuano Readability passages), while sampled retains minor subword artifacts (`kabii`, `nakita-bata`).

### 2. Cross-Prompt Diagnostic Takeaways

1. **Why Greedy Decoding Still Loops:** Under `do_sample=False`, `distilgpt2` deterministically picks the highest-probability next token at every step. Because news-corpus collocations like `sa kapulisan` and `sa mga kalibutan` have very high conditional probability, once entered they form self-reinforcing n-gram basins. This directly motivates **Step 2**, where `top-k`, `top-p`, and beam search with `no_repeat_ngram_size` are evaluated to break repetitive loops.
2. **Text Continuation vs. Question-Answering:** The fine-tuning corpus contains raw news articles (`CebuaNER`) and short narratives/DepEd passages (`cebuano-readability`), with no conversational or instruction-tuning pairs. Consequently, the adapted model behaves as a **Cebuano text-continuation engine**—continuing prompts in news or story registers rather than answering questions directly.

