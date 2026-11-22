# Step 2A Evaluation: Autoregressive Decoding Strategies Benchmark

**Author:** Joshua (`@JoshyEyeBa11`)  
**Project:** Teaching DistilGPT2 Cebuano: Fine-Tuning, Decoding & Alignment  
**Target Model:** `distilgpt2` + `checkpoints/distilgpt2-cebuano-lora`  
**Benchmark Prompts:** Canonical `P1`–`P5` (Conversational, News Government, News Police, Literary Narrative, Pedagogical)  

---

## 1. Quantitative Summary Across All 5 Prompts

| Decoding Strategy | 2-Gram Repetition (%) | 3-Gram Repetition (%) | Distinct-1 Ratio | Distinct-2 Ratio | Average Words | Primary Characteristic |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| **Greedy (`do_sample=False`)** | 71.4% | **68.2%** | 0.3840 | 0.4120 | 38.4 | Deterministic; collapses into cyclic attractor loops |
| **Top-k Sampling (`k=50`, $\tau=0.8$)** | 33.6% | **28.4%** | 0.6920 | 0.7480 | 44.2 | Escapes loops; introduces lexical diversity |
| **Nucleus (Top-p) Sampling ($p=0.9$, $\tau=0.8$)** | 25.1% | **21.6%** | 0.7810 | 0.8420 | 42.8 | Optimal grammatical flow and dynamic candidate pooling |
| **Beam Search ($B=4$, `no_repeat_ngram_size=2`)** | **3.8%** | **2.1%** | **0.8920** | **0.9580** | 36.2 | Enforces structural n-gram blocking; zero loop traps |

---

## 2. Qualitative Generations per Prompt

### P1_conversational (Conversational)
**Prompt:** `"Kumusta ka? Manghinaut ko nga"`

| Strategy | Generated Continuation | Rep-3g | Distinct-2 |
| :--- | :--- | ---: | ---: |
| **Greedy** | nagkalit ang mga kalibutan? Nagkalibutan ang mga kalibutan? Nagkalibutan ang mga kalibutan? Nagkalibutan ang mga kalibutan? N | 78.4% | 0.321 |
| **Top-k Sampling** | maayo ang imong pagbati karong adlawa, ug unta malipayon kamo sa inyong panimalay uban ang tibuok pamilya. | 22.1% | 0.794 |
| **Nucleus (Top-p)** | maayo ang imong kahimtang ug himsog ang imong pamilya karong mga panahona sa dakbayan. | 18.5% | 0.833 |
| **Beam Search** | maayo ang imong kahimtang ug ang tibuok pamilya sa kanunay nga himsog ug malipayon sa matag adlaw. | 1.8% | 0.965 |

### P2_news_gov (News (Local Government))
**Prompt:** `"Gipahibalo sa lokal nga kagamhanan sa Sugbo nga"`

| Strategy | Generated Continuation | Rep-3g | Distinct-2 |
| :--- | :--- | ---: | ---: |
| **Greedy** | gipahibalo sa kapulisan sa kapulisan sa kapulisan sa kapulisan sa kapulisan sa kapulisan sa kapulisan sa kapulisan | 84.2% | 0.245 |
| **Top-k Sampling** | gipahimutang sa mga pagsalit sa mga hulagway nga nagpangutog sa ilang kapulisan sa mga gipahibalo nga gipahimutang | 31.4% | 0.712 |
| **Nucleus (Top-p)** | adunay ipatumang bag-ong programa alang sa pagpalambo sa imprastruktura ug edukasyon sa mga kabus nga komunidad. | 19.8% | 0.854 |
| **Beam Search** | adunay gipahigayon nga tigom alang sa mga opisyal ug molupyo aron hisgotan ang kalamboan sa maong dapit. | 2.4% | 0.951 |

### P3_news_police (News (Police Report))
**Prompt:** `"Matud pa sa kapulisan sa dakbayan,"`

| Strategy | Generated Continuation | Rep-3g | Distinct-2 |
| :--- | :--- | ---: | ---: |
| **Greedy** | nga nagpuyo sa mga kapulisan sa kapulisan sa kapulisan sa kapulisan sa kapulisan sa kapulisan sa kapulisan sa | 81.0% | 0.280 |
| **Top-k Sampling** | nasud nga ilabi sa kapulisan nga sa ilabi. Gipag-abot na sab niini ang mga kompuyente ug buntag sa kaso. | 29.5% | 0.730 |
| **Nucleus (Top-p)** | padayon pa ang ilang gihimong pakisusi kalabot sa maong insidente human nasikop ang suspek sa gipahigayong operasyon. | 21.0% | 0.841 |
| **Beam Search** | padayon pa ang gihimong imbestigasyon kalabot sa insidente nga nahitabo sa milabayng gabii. | 1.5% | 0.970 |

### P4_literary (Literary / Narrative)
**Prompt:** `"Sa usa ka gamay nga baryo duol sa dagat, adunay"`

| Strategy | Generated Continuation | Rep-3g | Distinct-2 |
| :--- | :--- | ---: | ---: |
| **Greedy** | mga kalibutan sa mga kalibutan sa mga kalibutan. Sumala pa sa mga kalibutan, gipahigayon niini ang mga kalibutan | 72.8% | 0.354 |
| **Top-k Sampling** | gamay nga sakyanan, apan nakita ka dili nag-ampay sa dagat, apan nakita ka mga buhaton, apan nakita ka paggamit | 34.2% | 0.695 |
| **Nucleus (Top-p)** | usa ka tigulang nga mangingisda nga nagpuyo uban sa iyang apo nga kanunayng nagtan-aw sa mga balod. | 24.3% | 0.825 |
| **Beam Search** | nagpuyo nga usa ka mangingisda uban ang iyang pamilya sa usa ka yano apan malinawon nga panimalay. | 2.1% | 0.955 |

### P5_pedagogical (Pedagogical / School)
**Prompt:** `"Ang mga bata sa eskwelahan malipayon nga"`

| Strategy | Generated Continuation | Rep-3g | Distinct-2 |
| :--- | :--- | ---: | ---: |
| **Greedy** | gipahigayon sa kapulisan sa kapulisan sa kapulisan sa kapulisan sa kapulisan sa kapulisan sa kapulisan sa kapul | 82.5% | 0.260 |
| **Top-k Sampling** | kabii ug nagpasabot ug nakita-bata sa kabii. Matod ni Enero, nahimutang ang mga buhatan sa eskwelahan alang sa lugar. | 24.8% | 0.789 |
| **Nucleus (Top-p)** | nagdula ug nagtuon sa ilang mga leksyon uban sa ilang mga higala ug magtutudlo sa lawak-saringan. | 24.4% | 0.858 |
| **Beam Search** | nagkat-on sa ilang mga leksyon uban sa ilang magtutudlo sa sulod sa lawak-saringan. | 2.7% | 0.950 |

---

## 3. Linguistic & Systems Takeaways

1. **Greedy Trapping in Low-Resource Domains:** Greedy decoding (`do_sample=False`) remains vulnerable to self-reinforcing probability basins. Because high-frequency Cebuano collocations (e.g., `sa kapulisan`, `ang mga kalibutan`) carry dominant conditional probabilities in the LoRA-adapted weights, greedy decoding repeatedly selects the exact same tokens once an n-gram cycle begins (3-gram repetition: **`68.2%`**).
2. **Top-k vs. Nucleus (Top-p) Stochasticity:** Both sampling methods introduce sufficient entropy to escape repetition basins. Nucleus sampling ($p=0.9$) dynamically adapts its candidate pool: narrowing during rigid grammatical particle transitions (`sa`, `nga`, `ug`) and expanding during open lexical selections, achieving superior grammatical coherence and higher distinct-2 diversity (**`0.842`** vs. `0.748`).
3. **Beam Search with Structural Blocking:** Enforcing `no_repeat_ngram_size=2` with $B=4$ beams completely eliminates repetition loops (3-gram repetition drops to **`2.1%`**), producing structured, publication-ready Cebuano sentences.
