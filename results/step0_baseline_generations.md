# Step 0 Baseline Diagnostic: Stock `distilgpt2` on Cebuano

## 1. Held-Out Cebuano Validation Perplexity (Before Fine-Tuning)
- **Validation blocks evaluated:** 219 blocks × 256 tokens (56,064 tokens)
- **Baseline Cross-Entropy Loss:** `5.3545`
- **Baseline Perplexity:** `211.56`

## 2. BPE Tokenizer Fragmentation (English vs. Cebuano)
| Benchmark | Language | Total Words | Total BPE Tokens | Avg Tokens/Word (Fertility) | Avg Chars/Token |
| :--- | :--- | ---: | ---: | ---: | ---: |
| 12 Matched Words | English | 12 | 25 | 2.083 | 4.480 |
| 12 Matched Words | Cebuano | 12 | 51 | 4.250 | 2.196 |
| 5 Parallel Sentences | English | 56 | 65 | 1.161 | 5.123 |
| 5 Parallel Sentences | Cebuano | 57 | 145 | 2.544 | 2.372 |

### Selected Word Fragmentation Examples
| Cebuano Word | Subword Pieces | Tokens | Length-Matched English | Subword Pieces | Tokens |
| :--- | :--- | ---: | :--- | :--- | ---: |
| `kagamhanan` | `k | ag | am | han | an` | 5 | `government` | `government` | 1 |
| `eskwelahan` | `es | kw | el | ahan` | 4 | `university` | `un | iversity` | 2 |
| `kapulisan` | `k | ap | ul | isan` | 4 | `policemen` | `p | olic | emen` | 3 |
| `tinun-an` | `tin | un | - | an` | 4 | `students` | `stud | ents` | 2 |
| `panghitabo` | `pan | gh | it | abo` | 4 | `occurrence` | `occ | urrence` | 2 |
| `malipayon` | `mal | ip | ay | on` | 4 | `delighted` | `del | ight | ed` | 3 |
| `kaugmaon` | `k | aug | ma | on` | 4 | `tomorrow` | `tom | orrow` | 2 |
| `kasaysayan` | `kas | ays | ayan` | 3 | `literature` | `liter | ature` | 2 |
| `pagtulun-an` | `p | ag | t | ul | un | - | an` | 7 | `instruction` | `inst | ruction` | 2 |
| `kinabuhi` | `kin | ab | u | hi` | 4 | `lifetime` | `lif | etime` | 2 |
| `kalibutan` | `k | al | ib | utan` | 4 | `continent` | `cont | inent` | 2 |
| `panaghiusa` | `pan | agh | ius | a` | 4 | `fellowship` | `fell | owship` | 2 |

## 3. Untouched `distilgpt2` Generation Quality (English Drift & Subword Breakdown)

### P1_conversational (Conversational)
- **Prompt:** `Kumusta ka? Manghinaut ko nga`
- **Greedy (`do_sample=False`):** nga nga nga nga nga nga nga nga nga nga nga nga nga nga nga nga nga nga nga nga nga nga nga nga nga nga nga nga nga nga
- **Sampled (`top_p=0.92, T=0.8`):** ji ko ka?


Manga na nga, na nga ko, na kapunna lang na kapunna lang na nga, na kapunna lang na kapunna lang nga, na kapunna lang nga, na

### P2_news_gov (News (Local Government))
- **Prompt:** `Gipahibalo sa lokal nga kagamhanan sa Sugbo nga`
- **Greedy (`do_sample=False`):** kagamhanan sa Sugbo nga kagamhanan sa Sugbo nga kagamhanan sa Sugbo nga kagamhanan sa Sugbo nga kagamhanan sa Sugbo nga kagamhanan sa Sugbo nga
- **Sampled (`top_p=0.92, T=0.8`):** kagamhanan sa Sugbo nga kagamhanan sa Sugbo nga kagamhanan sa Sugbo nga kagamhanan sa Sugbo nga kagamhanan sa Sugbo nga kagamhanan sa Sugbo nga

### P3_news_police (News (Police Report))
- **Prompt:** `Matud pa sa kapulisan sa dakbayan,`
- **Greedy (`do_sample=False`):** sa kapulisan sa dakbayan, sa kapulisan sa dakbayan, sa kapulisan sa dakbayan, sa kapulisan sa dakbayan, sa kapulisan sa dakbayan, sa kapulisan
- **Sampled (`top_p=0.92, T=0.8`):** pa sa kapulisan sa dakbayan, pa sa kapulisan sa dakbayan, pa sa kapulisan sa dakbayan, pa sa kapulisan sa dakbayan, pa sa kapulisan sa dakbayan,

### P4_literary (Literary / Narrative)
- **Prompt:** `Sa usa ka gamay nga baryo duol sa dagat, adunay`
- **Greedy (`do_sample=False`):** na kapat na kapat na kapat na kapat na kapat na kapat na kapat na kapat na kapat na kapat na kapat na kapat na kapat na kapat na kapat
- **Sampled (`top_p=0.92, T=0.8`):** sa ao na sabay, katay ditakay ka dar, nakay ka sabay.

I am going to be writing about the issue of the state-owned water project, and I am not going to be giving a talk on that. It will be

### P5_pedagogical (Pedagogical / School)
- **Prompt:** `Ang mga bata sa eskwelahan malipayon nga`
- **Greedy (`do_sample=False`):** bata sa kapat kapat kapat kapat kapat kapat kapat kapat kapat kapat kapat kapat kapat kapat kapat kapat kapat kapat kapat
- **Sampled (`top_p=0.92, T=0.8`):** kabayon nga kabayon mga bata sa eskwelahan malipayon nga kabayon mga kabayon mga kabayon mga kabayon mga kabayon mga kabayon m
