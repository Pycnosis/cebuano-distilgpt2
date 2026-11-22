# Step 3 Evaluation: Before vs. After Cebuano DPO Alignment

## 1. Quantitative Summary (Step 1 LoRA vs. Step 3 DPO)
| Metric | Before DPO (Step 1 LoRA) | After DPO (Step 3 DPO) | Shift |
| :--- | ---: | ---: | :--- |
| **Held-Out Test Preference Accuracy (6 pairs)** | `0.0%` | `33.3%` | **+33.3 pp** |
| **Held-Out Test Mean Margin ($\Delta$)** | `-27.5982` | `-19.2461` | **+8.3521** |
| **Train Split Preference Accuracy (32 pairs)** | `12.5%` | `31.2%` | **+18.8 pp** |
| **Train Split Mean Margin ($\Delta$)** | `-26.6267` | `-13.5645` | **+13.0622** |
| **Greedy 3-Gram Repetition Rate (11 prompts)** | `68.18%` | `66.76%` | `-1.42 pp` |
| **Sampled Distinct-2 Diversity (11 prompts)** | `0.8897` | `0.9568` | `+0.0671` |
| **Greedy English Intrusion Rate (11 prompts)** | `0.00%` | `0.00%` | `+0.00 pp` |
| **Sampled English Intrusion Rate (11 prompts)** | `0.43%` | `0.41%` | `-0.02 pp` |
| **Held-Out Cebuano Val Perplexity (219 blocks)** | `34.88` | `35.33` | `+0.45 PPL` |

## 2. Per-Pair Log-Probability Margins on Held-Out Test Set
| ID | Register | Failure Mode Targeted | Before $\Delta$ | After $\Delta$ | Margin Gain |
| :--- | :--- | :--- | ---: | ---: | ---: |
| `dpo_13` | Conversational | `english_code_switching` | `-3.8259` | `-0.9468` | **`+2.8791`** |
| `dpo_14` | Conversational | `repetition_loop` | `-77.8630` | `-68.8780` | **`+8.9850`** |
| `dpo_25` | News (Local Government) | `repetition_loop` | `-48.2173` | `-41.3339` | **`+6.8834`** |
| `dpo_26` | News (Police Report) | `english_code_switching` | `-3.4631` | `2.6741` | **`+6.1372`** |
| `dpo_37` | Literary / Narrative | `english_code_switching` | `-7.0109` | `5.0159` | **`+12.0268`** |
| `dpo_38` | Pedagogical / School | `subword_gibberish` | `-25.2093` | `-12.0077` | **`+13.2016`** |

## 3. Qualitative Before vs. After Generations (`P1`–`P5` & Held-Out Test Prompts)

### P1_conversational (Conversational)
- **Prompt:** `Kumusta ka? Manghinaut ko nga`
- **Before DPO (Greedy):** nagkalit ang mga kalibutan?
Nagkalibutan ang mga kalibutan?
Nagkalibutan ang mga kalibutan?
Nagkalibutan ang mga kalibutan?
N
- **After DPO (Greedy):** nagkalit ang mga kalibutan.
Nagkalibutan ang mga kalibutan.
Nagkalibutan ang mga kalibutan.
Nagkalibutan ang mga kalibutan.
N
- **Before DPO (Sampled):** hinungdan?
Nakita ka?
Sa mga kaso kita.
Pagkamalit nga nakita na!
Nakita ka?
Pagkamalit nga kaso kita!
Nakita
- **After DPO (Sampled):** hinungdan nga nakita ang paglupad niini.
Gitugaw si Lola ang ilang iyang nakatabang pagdumala nga makahinumog si Lola.
Naghinumog si Lola

### P2_news_gov (News (Local Government))
- **Prompt:** `Gipahibalo sa lokal nga kagamhanan sa Sugbo nga`
- **Before DPO (Greedy):** gipahibalo sa kapulisan sa kapulisan sa kapulisan sa kapulisan sa kapulisan sa kapulisan sa kapulisan sa kapulisan sa kapulisan sa kapulisan sa kapulisan
- **After DPO (Greedy):** gipahigayon nga gipahigayon nga gipahigayon nga gipahigayon nga gipahigayon nga gipahigayon nga gipahigayon nga gipahig
- **Before DPO (Sampled):** si Marlano Ningtong nagdula nga sa ilang kahapon sa kadtong Negros Oriental. Mao kini sa ilang kahapon ang kahapon sa kahapon nga kahapon nga lokal nga kah
- **After DPO (Sampled):** si Marlano nga pag-atubang sa iyang biktima sa pasahero nga nagtukod sa iyang panglakip sa biktima. Gipahigayon siya nga gikusil ang mga

### P3_news_police (News (Police Report))
- **Prompt:** `Matud pa sa kapulisan sa dakbayan,`
- **Before DPO (Greedy):** nga nagpuyo sa mga kapulisan sa kapulisan sa kapulisan sa kapulisan sa kapulisan sa kapulisan sa kapulisan sa kapulisan sa kapulisan sa kapulisan sa
- **After DPO (Greedy):** nagpahigayon ang mga kalibutan nga gipahigayon nga gipahigayon nga gipahigayon nga gipahigayon nga gipahigayon nga gipahig
- **Before DPO (Sampled):** niingon nga nakadakita sa pagdala sa mga kapulisan sa ilang iyang nahimong dakbayan nga gikahimong kapulisan sa pagdala sa kapulisan. Gisubli ni
- **After DPO (Sampled):** niingon nga nagpadayon sa pagdala sa mga kahoy nga ilang iyang nahuman nga gipakapulisan nga ug kwarta.
Nagkaon kini og pagdala sa kap

### P4_literary (Literary / Narrative)
- **Prompt:** `Sa usa ka gamay nga baryo duol sa dagat, adunay`
- **Before DPO (Greedy):** mga kalibutan sa mga kalibutan sa mga kalibutan. Sumala pa sa mga kalibutan, gipahigayon niini ang mga kalibutan sa mga kalibutan sa mga kalib
- **After DPO (Greedy):** mga kaso nga nagkalain-laing kini.
Nagkalain-laing kini nga nagkalain-laing kini nga nagkalain-laing kini nga nagkalain
- **Before DPO (Sampled):** gipahibawo og daghang kapulisan, ug uban ang nagkahuman sa kalibutan sa lalaki. Sumala pa sa mga nag-abot sa usa ka gamay nga nahinay sa
- **After DPO (Sampled):** gipahibawo og daghang klase nga pagkadlok nga karong hapon nga mga iyang mga tawo nga naglalip na sa baryo sa mga pahinayon

### P5_pedagogical (Pedagogical / School)
- **Prompt:** `Ang mga bata sa eskwelahan malipayon nga`
- **Before DPO (Greedy):** gipahigayon sa kapulisan sa kapulisan sa kapulisan sa kapulisan sa kapulisan sa kapulisan sa kapulisan sa kapulisan sa kapulisan sa kapulisan sa kapul
- **After DPO (Greedy):** gipahigayon nga gipahigayon nga gipahigayon nga gipahigayon nga gipahigayon nga gipahigayon nga gipahigayon nga gipahig
- **Before DPO (Sampled):** gihatag ang iyang mga bata nga gipahimo ang nagpahigayon nga gipahilanglan sa kalibutan ug daghan ang mga bata.
Pagkaron sila kaambit ang m
- **After DPO (Sampled):** gihatag ang iyang iyang iyang paghimo nga nagtukod og mga pulis naan nga mga tigpahuman nga iyang mga balaod nga giingong mananan. Apan gi

### dpo_13 (DPO Test (Conversational))
- **Prompt:** `Nagtuo ko nga ang tinuod nga panaghigalaay`
- **Before DPO (Greedy):** sa mga kalibutan.
Nagkahapon ang mga kalibutan.
Nagkahapon ang mga kalibutan.
Nagkahapon ang mga kalibutan.
Nagkahapon ang mga k
- **After DPO (Greedy):** nga nagpahigayon nga nagpahigayon nga nagpahigayon nga nagpahigayon nga nagpahigayon nga nagpahigayon nga nagpah
- **Before DPO (Sampled):** on, ug ang pagsulod sa kaso.
KATAPUSAN SA NEGLOS Abril ug MGA NAGPATAY NGA KARONG MGA NAGPATAY NGA KATAPUSAN SA NEGL
- **After DPO (Sampled):** on, ug ang pamilya nag-isakit nga katapusan sa sakop. Gihatag na ang mga mga iyang kalipayon nga nagpahuman nga gidulaan nga gid

### dpo_14 (DPO Test (Conversational))
- **Prompt:** `Sayo pa sa buntag mitindog na si Tatay aron`
- **Before DPO (Greedy):** makatabang sa kapulisan sa kapulisan sa kapulisan sa kapulisan sa kapulisan sa kapulisan sa kapulisan sa kapulisan sa kapulisan sa kapulisan sa kapulisan
- **After DPO (Greedy):** makatabang ang mga kalibutan.
Nagkahuman na si Tatay ang mga kalibutan.
Nagkahuman na si Tatay ang mga kalibutan.
Nagkahuman na si Tatay ang
- **Before DPO (Sampled):** makaayo nga giingong mga kahapon sa kapulisan sa iyang giingong mga kawo sa tawo.
Si Tatay ug nagdula sa iyang gisulat nga mga kah
- **After DPO (Sampled):** makaayo nga giingong mga dili diay unang kalibutan nga pagkamatay og mga tawo.
Si Tatay ug nagdala pa ang iyang duha ka dili mga kap

### dpo_25 (DPO Test (News (Local Government)))
- **Prompt:** `Nanawagan ang buhatan sa agrikultura ngadto sa`
- **Before DPO (Greedy):** mga kalibutan sa pagpahigayon sa pagpahigayon sa pagpahigayon sa pagpahigayon sa pagpahigayon sa pagpahigayon sa pagpahigayon sa pagpahig
- **After DPO (Greedy):** pagpahigayon sa pagpahigayon sa pagpahigayon sa pagpahigayon sa pagpahigayon sa pagpahigayon sa pagpahigayon sa pagpahigayon sa pagpahig
- **Before DPO (Sampled):** ilang usa ka mga tawo nga gipangatay sa panghitabo sa mga pinaagi sa PNP. Sumala pa sa PNP, niini ang kaso nga nakita-an sa PNP niadtong S
- **After DPO (Sampled):** ilang usa ka mga tawo nga gipangatayon sa mga opisyal. Nakahanglan siya si Domingo, "Hinungdan na kini nga nagdili na kung unsa nga dili g

### dpo_26 (DPO Test (News (Police Report)))
- **Prompt:** `Gibantayan pag-ayo sa mga tinugyanan sa balaod ang`
- **Before DPO (Greedy):** mga tawo sa mga tawo sa mga tawo. Gipahigayon ni Mayor Eduardo Estrada ang mga tawo sa mga tawo sa mga tawo sa mga tawo sa mga taw
- **After DPO (Greedy):** pag-ayo nga mga kaso.
Nagkahapon ang pag-ayo nga mga kaso.
Nagkahapon ang pag-ayo nga mga kaso.
Nagkahapon ang pag-ay
- **Before DPO (Sampled):** iyang mga sakyanan. Ang iyang mga tinugyanan ug mga tinugyanan sa balaod sa katuman sa pagtukod sa pag-uporta sa balaod. Gikan sa katuman og kat
- **After DPO (Sampled):** giingong sakyanan. Ang pagkaon nga konsehal ug mga pagkaon nga namit-an ang pagpanganod ang pagkaon. Ang giingong pagkaon nga namit-an ang mga

### dpo_37 (DPO Test (Literary / Narrative))
- **Prompt:** `Gikuptan sa inahan ang kamot sa iyang anak`
- **Before DPO (Greedy):** .
KATAPUSAN SA KAMOTAN, NAGPASA-AN SA KAMOTAN, NAGPASA-AN SA KAMOTAN, NAGPASA-AN SA KAMOTAN, NAGPASA-AN SA
- **After DPO (Greedy):** .
Nagkahapon ang iyang anak.
Nagkahapon ang iyang anak.
Nagkahapon ang iyang anak.
Nagkahapon ang iyang anak.
Nagkahapon ang iyang anak
- **Before DPO (Sampled):** sa mga bata. Gisugdan ni Tugag ang iyang anak nga mag-usi og mga tawo sa iyang anak.
Nagkahapon og tawo sa mga bata. Nipahinab
- **After DPO (Sampled):** .
Ang inahan ni Kompanyo, nakita ang iyang anak.
Mga dili og dapit.
KATAPUSAN SA PAG-INAN, MGA DOUGON, GISUGYANA, KATAP

### dpo_38 (DPO Test (Pedagogical / School))
- **Prompt:** `Matag Biyernes sa hapon, ang mga kabataan`
- **Before DPO (Greedy):** nga nag-abot sa mga kabataan nga nag-abot sa mga kabataan nga nag-abot sa mga kabataan nga nag-abot sa mga kabataan nga nag
- **After DPO (Greedy):** nga gipahigayon nga gipahigayon nga gipahigayon nga gipahigayon nga gipahigayon nga gipahigayon nga gipahigayon nga gip
- **Before DPO (Sampled):** sa pagpabak sa usa ka dili pa ang mga barangay nga nakita og gisubli sa sakyanan. Giila ang kamatayon nga pagpabak sa mga kabataan, apan k
- **After DPO (Sampled):** nga pag-akong adlaw, dili pa ang mga barangay nga nakita og gisugdan nga aduna'y mga kahimtangan nga pagpahuman kini nga makasaka sa mga
