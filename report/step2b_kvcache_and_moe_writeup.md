# Step 2B Written Analysis: KV-Cache Inference Complexity & Mixture-of-Experts (MoE) Trade-Offs

## 1. Theoretical Derivation: Why KV-Caching Reduces Autoregressive Decoding from $O(T^2)$ to $O(T)$

In a causal decoder-only transformer such as `distilgpt2` ($L = 6$ layers, $H = 12$ attention heads, $d_{\text{model}} = 768$, $d_{\text{head}} = 64$), generating $T$ new tokens after a prompt of length $P$ proceeds one token at a time for steps $t = 1, \dots, T$, with active prefix length $S_t = P + t - 1$. At each step $t$, every attention layer projects hidden states into queries ($Q$), keys ($K$), and values ($V$) and computes causal scaled dot-product attention:

$$\text{Attention}(Q_t, K_{1:S_t}, V_{1:S_t}) = \text{softmax}\!\left(\frac{Q_t K_{1:S_t}^\top}{\sqrt{d_{\text{head}}}} + M\right) V_{1:S_t}$$

Because causal masking prevents past tokens $i < S_t$ from attending to future tokens, the key and value vectors $(k_i, v_i)$ for every previous position $i \in \{1, \dots, S_t - 1\}$ are **strictly invariant** across subsequent generation steps.

- **Without KV-Cache (`use_cache=False`):** At every decoding step $t$, the model re-runs the entire prefix $x_{1:S_t}$ through all $L$ transformer layers—recomputing $Q, K, V$ projections, full $S_t \times S_t$ attention matrices, and MLP feed-forward blocks for every historical token.
  - Per-step projection + MLP compute scales as $O(S_t \, d_{\text{model}}^2)$, and self-attention scales as $O(S_t^2 \, d_{\text{model}})$.
  - Summing across $t = 1 \dots T$ generated tokens yields **quadratic projection/MLP work** $\sum_{t=1}^{T} (P + t) = O(T^2)$ and **cubic attention work** $\sum_{t=1}^{T} (P + t)^2 = O(T^3)$—meaning per-token latency grows linearly with sequence length while total latency scales **at least quadratically ($O(T^2)$)**.
- **With KV-Cache (`use_cache=True`):** After the single prompt prefill pass ($O(P)$), each step $t$ feeds **only the 1 newest token** $x_{S_t}$ through the $Q, K, V$ projections and MLP block ($O(d_{\text{model}}^2)$ constant work per step), appends the new $(k_{S_t}, v_{S_t})$ to the cached history, and attends $q_{S_t}$ against the cached $K_{1:S_t}, V_{1:S_t}$ ($O(S_t \, d_{\text{model}})$).
  - Total projection and MLP compute across $T$ tokens drops from **$O(T^2)$ to $O(T)$** (constant work per generated token), while total attention dot-product compute drops from $O(T^3)$ to $O(T^2)$ with a tiny constant factor relative to the linear projections at $S_t \le 277$.

### KV-Cache Memory Footprint for `distilgpt2`

Caching keys and values trades a modest amount of memory for this massive compute reduction. For batch size $B = 1$ and sequence length $S$, the cache stores $2$ tensors ($K$ and $V$) of size $S \times d_{\text{model}}$ across $L$ layers at $b$ bytes per element:

$$M_{\text{KV}}(S) = 2 \cdot L \cdot S \cdot d_{\text{model}} \cdot b$$

For `distilgpt2` ($L = 6$, $d_{\text{model}} = 768$), this equals **$36{,}864$ bytes (`36.0 KB`) per token in `fp32`** (`18.0 KB/token` in `fp16`). Even at `max_new_tokens = 256` ($S = 277$), the entire KV-cache occupies only **`9,972.0 KB` (`~9.74 MB`)**, and at `distilgpt2`'s full `1,024`-token context ceiling it uses just **`36.0 MB` in `fp32`** (`18.0 MB` in `fp16`)—negligible compared to the model weights. (At 7B–70B+ parameters with long contexts and high batch sizes, however, KV-cache VRAM growth becomes a major serving bottleneck addressed by Grouped-Query Attention and PagedAttention.)

---

## 2. Empirical Benchmark Results (CPU vs. RTX 4050 GPU)

In `scripts/step2b_kvcache.py`, we benchmarked our Step 1 LoRA-adapted model (`distilgpt2` + `checkpoints/distilgpt2-cebuano-lora`, `fp32`) on canonical prompt `P2_news_gov` (`"Gipahibalo sa lokal nga kagamhanan sa Sugbo nga"`, `21` BPE tokens) with fixed-length greedy decoding (`do_sample=False`) across `max_new_tokens` $\in \{32, 64, 128, 256\}$. Across every length and both devices, **`outputs_identical = True`**, confirming that KV-caching is mathematically equivalent to full recomputation.

### A. CPU Benchmark — Compute-Bound Regime (`AMD Ryzen 7 7735HS`, 3 Timed Runs/Setting)

| `max_new_tokens` | Total Seq ($S$) | `use_cache=True` Mean (s) | `use_cache=True` (ms/tok) | `use_cache=False` Mean (s) | `use_cache=False` (ms/tok) | Speedup (`no_cache / cache`) | Latency Saved | KV-Cache (`fp32`) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **32** | 53 | `1.0806` ± `0.0215` | `33.77` | `1.7762` ± `0.0715` | `55.51` | **`1.64×`** | `39.2%` | `1,908 KB` |
| **64** | 85 | `2.1972` ± `0.0415` | `34.33` | `3.8363` ± `0.0403` | `59.94` | **`1.75×`** | `42.7%` | `3,060 KB` |
| **128** | 149 | `4.3394` ± `0.0220` | `33.90` | `9.0170` ± `0.1324` | `70.45` | **`2.08×`** | `51.9%` | `5,364 KB` |
| **256** | 277 | `8.7728` ± `0.1452` | `34.27` | `24.9816` ± `0.5498` | `97.58` | **`2.85×`** | `64.9%` | `9,972 KB` |

- **Empirical Confirmation of $O(T)$ vs. $O(T^2)$:** On CPU (our project's primary *"No GPU Required"* hardware target), `use_cache=True` maintains a nearly flat **`33.77–34.33 ms/token`** (`~29.2–29.6 tok/s`) as sequence length increases $8\times$ from 32 to 256 tokens—confirming **linear $O(T)$ total scaling** (`1.08s -> 2.20s -> 4.34s -> 8.77s`). By contrast, `use_cache=False` degrades steadily from **`55.51 ms/tok` to `97.58 ms/tok`**, pushing latency from `1.78s` to **`24.98s`** (`14.1×` slower for $8\times$ the tokens) and widening the KV-cache speedup monotonically from **`1.64×` at 32 tokens to `2.85×` at 256 tokens** (`64.9%` wall-clock reduction).

### B. CUDA GPU Benchmark — Kernel-Launch & Dispatch Regime (`NVIDIA GeForce RTX 4050 Laptop GPU`, 5 Timed Runs/Setting)

| `max_new_tokens` | `use_cache=True` Mean (s) | `use_cache=True` (ms/tok) | `use_cache=False` Mean (s) | `use_cache=False` (ms/tok) | Speedup (`no_cache / cache`) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| **32** | `0.3405` ± `0.0354` | `10.64` | `0.3171` ± `0.0289` | `9.91` | `0.93×` |
| **64** | `0.6989` ± `0.0672` | `10.92` | `0.6472` ± `0.0335` | `10.11` | `0.93×` |
| **128** | `1.4276` ± `0.1336` | `11.15` | `1.5402` ± `0.1570` | `12.03` | **`1.08×`** |
| **256** | `2.6986` ± `0.2057` | `10.54` | `3.0144` ± `0.2711` | `11.78` | **`1.12×`** |

- **Systems Insight (Why GPU Crossover Occurs Around 128 Tokens):** For an 82M-parameter 6-layer model at `batch_size=1` wrapped in PEFT LoRA layers, an RTX 4050 GPU has vast parallel compute slack: at `32`–`64` tokens, wall-clock latency (`~10.5 ms/tok`) is dominated by Python/PyTorch/PEFT kernel-launch overhead and cache tensor concatenation (`0.93×` parity). Once the sequence reaches **`128`–`256` tokens**, recomputing the full prefix starts saturating GPU compute and memory bandwidth, flipping the crossover to **`1.08×`** and **`1.12×`** (`10.5%` faster) while `use_cache=True` remains flat at `10.54 ms/tok`. Note also how Cebuano's high BPE fertility (`4.25` tokens/word on Step 0) makes KV-caching doubly important in practice: a 60-word Cebuano paragraph already consumes ~150–255 BPE tokens, pushing CPU inference squarely into the `2.08×–2.85×` speedup regime.

*Visual reference:* See `results/step2_kvcache_chart.png` and `results/step2_kvcache_benchmark.csv`.

---

## 3. Mixture-of-Experts (MoE) Inference Trade-Off Note: Per-Token FLOPs vs. Full-Model Memory

Modern sparse **Mixture-of-Experts (MoE)** transformers (Shazeer et al., 2017; Fedus et al., 2022; Jiang et al., 2024) replace each dense feed-forward network (FFN) block with $E$ parallel expert FFNs plus a lightweight learned router $G(x)$ that activates only the top-$k$ experts per token ($k \ll E$, e.g., $k=2$ of $E=8$ experts in Mixtral 8×7B):

$$y = \sum_{i \in \text{Top-}k(G(x))} G_i(x)\, E_i(x)$$

This decouples **model capacity** (total parameters $N_{\text{total}}$) from **per-token compute** (active parameters $N_{\text{active}}$), creating a sharp inference-time systems trade-off:

| Inference Dimension | Dense Transformer (e.g., `distilgpt2` or Dense 47B) | Sparse MoE Transformer (e.g., Mixtral 8×7B: 46.7B total / 12.9B active) |
| :--- | :--- | :--- |
| **Per-Token FLOPs ($C_{\text{tok}}$)** | Scales with **all** parameters ($\approx 2 N_{\text{total}}$ FLOPs/token) | Scales only with **active** parameters ($\approx 2 N_{\text{active}}$ FLOPs/token) — e.g., ~3.6× fewer FLOPs than a dense 47B model |
| **VRAM / RAM Weight Footprint** | Must hold $N_{\text{total}}$ parameters in memory | Still must hold **all $E$ experts** ($N_{\text{total}}$ parameters) in VRAM/RAM simultaneously |
| **Memory-Bandwidth Bottleneck (`batch=1`)** | Reads $N_{\text{total}}$ weights per token | Reads $N_{\text{active}}$ weights per token, *provided* all experts stay resident in fast HBM/VRAM |
| **Offloading Penalty (Consumer Hardware)** | Fits on single laptop if $N_{\text{total}}$ is small (`distilgpt2`: ~319 MB `fp32`) | Offloading inactive experts to CPU RAM / NVMe incurs severe PCIe transfer stalls whenever the router switches experts between tokens, collapsing throughput |

**Key Takeaway for Low-Resource Consumer Deployment:** Sparse MoE models achieve the **compute latency** of a much smaller dense model, making them attractive on multi-GPU servers with abundant aggregate HBM VRAM (e.g., >90 GB in `fp16` for 8×7B). However, because routing is dynamic and token-dependent, the serving engine cannot predict ahead of time which experts will be needed at step $t+1$—so **all $E$ experts must remain resident in memory**. On consumer laptops with 6–8 GB VRAM (like our RTX 4050) or CPU-only setups, a compact dense model adapted with LoRA (`distilgpt2` + `1.62M` adapter weights) paired with KV-caching remains dramatically more practical than an MoE architecture whose full parameter footprint exceeds local memory capacity.

---

## References

- Fedus, W., Zoph, B., & Shazeer, N. (2022). *Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity*. Journal of Machine Learning Research (JMLR), 23(120), 1–39. arXiv:2101.03961.
- Jiang, A. Q., Sablayrolles, A., Roux, A., Mensch, A., Savary, B., Bamford, C., Chaplot, D. S., de las Casas, D., et al. (2024). *Mixtral of Experts*. arXiv preprint arXiv:2401.04088.
- Kwon, W., Li, Z., Zhuang, S., Sheng, Y., Zheng, L., Yu, C. H., Gonzalez, J., Zhang, H., & Stoica, I. (2023). *Efficient Memory Management for Large Language Model Serving with PagedAttention*. SOSP 2023. arXiv:2309.06180.
- Pope, R., Douglas, S., Chowdhery, A., Devlin, J., Bradbury, J., Heek, J., Xiao, K., Agrawal, S., & Dean, J. (2023). *Efficiently Scaling Transformer Inference*. MLSys 2023. arXiv:2211.05102.
- Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I. (2019). *Language Models are Unsupervised Multitask Learners* (GPT-2). OpenAI Technical Report.
- Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). *DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter*. arXiv preprint arXiv:1910.01108.
- Shazeer, N., Mirhoseini, A., Maziarz, K., Davis, A., Le, Q., Hinton, G., & Dean, J. (2017). *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer*. ICLR 2017. arXiv:1701.06538.
