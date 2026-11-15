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
