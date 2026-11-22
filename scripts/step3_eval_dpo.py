"""Step 3B (Part 2): Evaluate Before vs. After Step 3 DPO Alignment.

Compares Step 1 LoRA (checkpoints/distilgpt2-cebuano-lora) against
Step 3 DPO (checkpoints/distilgpt2-cebuano-dpo) on:
1. Per-pair log-probability margins (logp_chosen - logp_rejected) & preference accuracy
   across the 6 held-out test pairs (plus 32 train pairs summary).
2. Greedy and sampled text generations on canonical P1-P5 and the 6 held-out test prompts,
   measuring 3-gram repetition rate, distinct-1/distinct-2, and English intrusion rate.
3. Held-out validation perplexity on data/cebuano_corpus_val.txt (219 blocks).

Outputs:
- results/step3_dpo_metrics.json
- results/step3_dpo_before_after.md
- results/step3_dpo_margins_chart.png
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from step0_baseline import PROMPTS, evaluate_validation_perplexity  # noqa: E402

# Unambiguous English function/content words (excluding any Cebuano homographs)
ENGLISH_STOPWORDS = {
    "the", "and", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "will", "would", "could", "should", "can", "may",
    "with", "from", "this", "that", "these", "those", "they", "their", "them",
    "there", "here", "what", "which", "when", "where", "who", "whom", "why", "how",
    "because", "about", "into", "through", "during", "before", "after",
    "you", "your", "we", "our", "he", "his", "she", "her", "it", "its",
    "not", "but", "for", "of", "to", "in", "on", "at", "by", "as", "or", "if",
    "very", "today", "early", "going", "writing", "issue", "project", "water",
}


def compute_completion_logprob(
    model: Any,
    tokenizer: Any,
    prompt: str,
    completion: str,
    device: torch.device,
) -> float:
    """Compute sum of log-probabilities of completion tokens conditioned on prompt."""
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    full_ids = tokenizer.encode(prompt + completion, add_special_tokens=False)
    if len(full_ids) <= 1:
        return 0.0

    # Find exact common prefix length in case GPT-2 BPE merges trailing space
    prefix_len = 0
    for p_id, f_id in zip(prompt_ids, full_ids):
        if p_id == f_id:
            prefix_len += 1
        else:
            break
    prefix_len = max(1, min(prefix_len, len(full_ids) - 1))

    input_tensor = torch.tensor([full_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(input_ids=input_tensor).logits

    log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
    target_ids = input_tensor[:, 1:]
    token_logps = log_probs.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)

    # Sum log-probs for tokens predicting indices >= prefix_len (shifted by -1 in target_ids)
    comp_logps = token_logps[0, prefix_len - 1 :]
    return float(comp_logps.sum().item())


def evaluate_pairs(
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    device: torch.device,
) -> dict[str, Any]:
    """Compute per-pair chosen/rejected log-probs, margins, and preference accuracy."""
    pair_results = []
    for r in rows:
        lp_chosen = compute_completion_logprob(model, tokenizer, r["prompt"], r["chosen"], device)
        lp_rejected = compute_completion_logprob(model, tokenizer, r["prompt"], r["rejected"], device)
        margin = lp_chosen - lp_rejected
        pair_results.append(
            {
                "id": r["id"],
                "register": r["register"],
                "failure_mode": r["failure_mode"],
                "split": r["split"],
                "logp_chosen": round(lp_chosen, 4),
                "logp_rejected": round(lp_rejected, 4),
                "margin": round(margin, 4),
                "preferred_chosen": margin > 0.0,
            }
        )

    n = len(pair_results)
    mean_margin = sum(p["margin"] for p in pair_results) / max(1, n)
    acc = sum(1 for p in pair_results if p["preferred_chosen"]) / max(1, n)
    return {
        "count": n,
        "mean_margin": round(mean_margin, 4),
        "preference_accuracy": round(acc, 4),
        "pairs": pair_results,
    }


def compute_text_metrics(text: str) -> dict[str, float]:
    """Compute distinct-1, distinct-2, 3-gram repetition rate, and English intrusion rate."""
    words = re.findall(r"[A-Za-z0-9'-]+", text.lower())
    n = len(words)
    if n == 0:
        return {"word_count": 0, "distinct_1": 0.0, "distinct_2": 0.0, "rep_3_rate": 0.0, "eng_intrusion_rate": 0.0, "eng_count": 0}

    distinct_1 = len(set(words)) / n
    bigrams = list(zip(words, words[1:]))
    distinct_2 = len(set(bigrams)) / len(bigrams) if bigrams else 1.0

    trigrams = list(zip(words, words[1:], words[2:]))
    rep_3_rate = 1.0 - (len(set(trigrams)) / len(trigrams)) if trigrams else 0.0

    eng_count = sum(1 for w in words if w in ENGLISH_STOPWORDS)
    eng_rate = eng_count / n

    return {
        "word_count": n,
        "distinct_1": round(distinct_1, 4),
        "distinct_2": round(distinct_2, 4),
        "rep_3_rate": round(rep_3_rate, 4),
        "eng_intrusion_rate": round(eng_rate, 4),
        "eng_count": eng_count,
    }


@torch.no_grad()
def generate_eval_suite(
    model: Any,
    tokenizer: Any,
    prompts: list[dict[str, str]],
    device: torch.device,
    max_new_tokens: int = 60,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Generate greedy and nucleus-sampled continuations and score quality metrics."""
    model.eval()
    outputs = []
    for idx, item in enumerate(prompts):
        prompt = item["prompt"]
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        prompt_len = inputs["input_ids"].shape[1]

        greedy_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        greedy_text = tokenizer.decode(greedy_ids[0][prompt_len:], skip_special_tokens=True).strip()

        set_seed(seed + idx)
        sampled_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=0.92,
            temperature=0.8,
            pad_token_id=tokenizer.eos_token_id,
        )
        sampled_text = tokenizer.decode(sampled_ids[0][prompt_len:], skip_special_tokens=True).strip()

        outputs.append(
            {
                "id": item["id"],
                "register": item["register"],
                "prompt": prompt,
                "greedy_continuation": greedy_text,
                "greedy_metrics": compute_text_metrics(greedy_text),
                "sampled_continuation": sampled_text,
                "sampled_metrics": compute_text_metrics(sampled_text),
            }
        )
    return outputs


def aggregate_gen_metrics(gens: list[dict[str, Any]], mode_key: str) -> dict[str, float]:
    """Average text quality metrics across a list of generated prompts."""
    n = len(gens)
    return {
        "avg_distinct_1": round(sum(g[mode_key]["distinct_1"] for g in gens) / max(1, n), 4),
        "avg_distinct_2": round(sum(g[mode_key]["distinct_2"] for g in gens) / max(1, n), 4),
        "avg_rep_3_rate": round(sum(g[mode_key]["rep_3_rate"] for g in gens) / max(1, n), 4),
        "avg_eng_intrusion_rate": round(sum(g[mode_key]["eng_intrusion_rate"] for g in gens) / max(1, n), 4),
        "total_eng_tokens": sum(g[mode_key]["eng_count"] for g in gens),
    }


def plot_dpo_charts(
    before_test: dict[str, Any],
    after_test: dict[str, Any],
    before_greedy_agg: dict[str, float],
    after_greedy_agg: dict[str, float],
    out_path: Path,
) -> None:
    """Create 2-panel comparison figure: Test-Pair Margins & Behavioral Shift Rates."""
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), dpi=160)

    # Panel A: Per-Test-Pair Log-Probability Margins
    pair_ids = [p["id"] for p in before_test["pairs"]]
    b_margins = [p["margin"] for p in before_test["pairs"]]
    a_margins = [p["margin"] for p in after_test["pairs"]]

    x = list(range(len(pair_ids)))
    w = 0.36
    bars1 = axes[0].bar([i - w / 2 for i in x], b_margins, w, label="Before DPO (Step 1 LoRA)", color="#2b6cb0")
    bars2 = axes[0].bar([i + w / 2 for i in x], a_margins, w, label="After DPO (Step 3 DPO)", color="#dd6b20")
    axes[0].axhline(0.0, color="#4a5568", linestyle="--", linewidth=1.0)
    axes[0].set_title("Held-Out Test Pair Log-Prob Margins\n$\\Delta = \\log p(y_{chosen}|x) - \\log p(y_{rejected}|x)$", fontsize=10.5, fontweight="bold")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(pair_ids, rotation=20, fontsize=9)
    axes[0].set_ylabel("Log-Probability Margin ($\\Delta$)")
    axes[0].grid(axis="y", linestyle="--", alpha=0.4)
    axes[0].legend(fontsize=8.5)
    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        offset = 0.35 if h >= 0 else -1.1
        axes[0].text(bar.get_x() + bar.get_width() / 2, h + offset, f"{h:.1f}", ha="center", va="bottom", fontsize=7.5)

    # Panel B: Preference Accuracy & Greedy 3-Gram Repetition Rate (%)
    cats = ["Test Pref.\nAccuracy (%)", "Train Pref.\nAccuracy (%)", "Greedy 3-Gram\nRepetition (%)", "Greedy Eng.\nIntrusion (%)"]
    b_vals = [
        before_test["preference_accuracy"] * 100.0,
        before_test["train_acc_ref"] * 100.0,
        before_greedy_agg["avg_rep_3_rate"] * 100.0,
        before_greedy_agg["avg_eng_intrusion_rate"] * 100.0,
    ]
    a_vals = [
        after_test["preference_accuracy"] * 100.0,
        after_test["train_acc_ref"] * 100.0,
        after_greedy_agg["avg_rep_3_rate"] * 100.0,
        after_greedy_agg["avg_eng_intrusion_rate"] * 100.0,
    ]
    x2 = list(range(len(cats)))
    bars3 = axes[1].bar([i - w / 2 for i in x2], b_vals, w, label="Before DPO (Step 1 LoRA)", color="#2b6cb0")
    bars4 = axes[1].bar([i + w / 2 for i in x2], a_vals, w, label="After DPO (Step 3 DPO)", color="#dd6b20")
    axes[1].set_title("Preference Accuracy & Generation Behavior\n(Higher Acc = Better | Lower Rep/Eng = Better)", fontsize=10.5, fontweight="bold")
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(cats, fontsize=8.5)
    axes[1].set_ylabel("Percentage (%)")
    axes[1].set_ylim(0, 118)
    axes[1].grid(axis="y", linestyle="--", alpha=0.4)
    axes[1].legend(fontsize=8.5)
    for bar in list(bars3) + list(bars4):
        h = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width() / 2, h + 1.5, f"{h:.1f}%", ha="center", va="bottom", fontsize=7.8)

    fig.suptitle("Step 3: Direct Preference Optimization (DPO) Before vs. After Alignment", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def write_markdown_report(
    metrics: dict[str, Any],
    before_gens: list[dict[str, Any]],
    after_gens: list[dict[str, Any]],
    out_path: Path,
) -> None:
    """Save formatted Step 3 before/after evaluation markdown report."""
    bt = metrics["test_pairs"]["before_dpo"]
    at = metrics["test_pairs"]["after_dpo"]
    btr = metrics["train_pairs"]["before_dpo"]
    atr = metrics["train_pairs"]["after_dpo"]
    bp = metrics["validation_perplexity"]["before_dpo"]
    ap = metrics["validation_perplexity"]["after_dpo"]
    bg = metrics["generation_summary"]["before_dpo"]
    ag = metrics["generation_summary"]["after_dpo"]

    lines = [
        "# Step 3 Evaluation: Before vs. After Cebuano DPO Alignment",
        "",
        "## 1. Quantitative Summary (Step 1 LoRA vs. Step 3 DPO)",
        "| Metric | Before DPO (Step 1 LoRA) | After DPO (Step 3 DPO) | Shift |",
        "| :--- | ---: | ---: | :--- |",
        f"| **Held-Out Test Preference Accuracy (6 pairs)** | `{bt['preference_accuracy']*100:.1f}%` | `{at['preference_accuracy']*100:.1f}%` | **+{(at['preference_accuracy']-bt['preference_accuracy'])*100:.1f} pp** |",
        f"| **Held-Out Test Mean Margin ($\\Delta$)** | `{bt['mean_margin']:.4f}` | `{at['mean_margin']:.4f}` | **+{at['mean_margin']-bt['mean_margin']:.4f}** |",
        f"| **Train Split Preference Accuracy (32 pairs)** | `{btr['preference_accuracy']*100:.1f}%` | `{atr['preference_accuracy']*100:.1f}%` | **+{(atr['preference_accuracy']-btr['preference_accuracy'])*100:.1f} pp** |",
        f"| **Train Split Mean Margin ($\\Delta$)** | `{btr['mean_margin']:.4f}` | `{atr['mean_margin']:.4f}` | **+{atr['mean_margin']-btr['mean_margin']:.4f}** |",
        f"| **Greedy 3-Gram Repetition Rate (11 prompts)** | `{bg['greedy']['avg_rep_3_rate']*100:.2f}%` | `{ag['greedy']['avg_rep_3_rate']*100:.2f}%` | `{(ag['greedy']['avg_rep_3_rate']-bg['greedy']['avg_rep_3_rate'])*100:+.2f} pp` |",
        f"| **Sampled Distinct-2 Diversity (11 prompts)** | `{bg['sampled']['avg_distinct_2']:.4f}` | `{ag['sampled']['avg_distinct_2']:.4f}` | `{ag['sampled']['avg_distinct_2']-bg['sampled']['avg_distinct_2']:+.4f}` |",
        f"| **Greedy English Intrusion Rate (11 prompts)** | `{bg['greedy']['avg_eng_intrusion_rate']*100:.2f}%` | `{ag['greedy']['avg_eng_intrusion_rate']*100:.2f}%` | `{(ag['greedy']['avg_eng_intrusion_rate']-bg['greedy']['avg_eng_intrusion_rate'])*100:+.2f} pp` |",
        f"| **Sampled English Intrusion Rate (11 prompts)** | `{bg['sampled']['avg_eng_intrusion_rate']*100:.2f}%` | `{ag['sampled']['avg_eng_intrusion_rate']*100:.2f}%` | `{(ag['sampled']['avg_eng_intrusion_rate']-bg['sampled']['avg_eng_intrusion_rate'])*100:+.2f} pp` |",
        f"| **Held-Out Cebuano Val Perplexity (219 blocks)** | `{bp['val_perplexity']:.2f}` | `{ap['val_perplexity']:.2f}` | `{ap['val_perplexity']-bp['val_perplexity']:+.2f} PPL` |",
        "",
        "## 2. Per-Pair Log-Probability Margins on Held-Out Test Set",
        "| ID | Register | Failure Mode Targeted | Before $\\Delta$ | After $\\Delta$ | Margin Gain |",
        "| :--- | :--- | :--- | ---: | ---: | ---: |",
    ]
    for bp_row, ap_row in zip(bt["pairs"], at["pairs"]):
        gain = ap_row["margin"] - bp_row["margin"]
        lines.append(
            f"| `{bp_row['id']}` | {bp_row['register']} | `{bp_row['failure_mode']}` | "
            f"`{bp_row['margin']:.4f}` | `{ap_row['margin']:.4f}` | **`{gain:+.4f}`** |"
        )

    lines.extend(["", "## 3. Qualitative Before vs. After Generations (`P1`–`P5` & Held-Out Test Prompts)", ""])
    for b_gen, a_gen in zip(before_gens, after_gens):
        lines.extend(
            [
                f"### {b_gen['id']} ({b_gen['register']})",
                f"- **Prompt:** `{b_gen['prompt']}`",
                f"- **Before DPO (Greedy):** {b_gen['greedy_continuation']}",
                f"- **After DPO (Greedy):** {a_gen['greedy_continuation']}",
                f"- **Before DPO (Sampled):** {b_gen['sampled_continuation']}",
                f"- **After DPO (Sampled):** {a_gen['sampled_continuation']}",
                "",
            ]
        )

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / "data" / "cebuano_dpo_preferences.jsonl"
    val_path = project_root / "data" / "cebuano_corpus_val.txt"
    lora_ckpt_dir = project_root / "checkpoints" / "distilgpt2-cebuano-lora"
    dpo_ckpt_dir = project_root / "checkpoints" / "distilgpt2-cebuano-dpo"
    results_dir = project_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    if not dpo_ckpt_dir.exists():
        raise FileNotFoundError(f"Missing DPO checkpoint {dpo_ckpt_dir}. Run scripts/step3_train_dpo.py first.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Step 3 before/after evaluation on device: {device}")

    rows = [json.loads(l) for l in data_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    train_rows = [r for r in rows if r.get("split", "train") == "train"]
    test_rows = [r for r in rows if r.get("split") == "test"]

    eval_prompts = list(PROMPTS) + [
        {"id": r["id"], "register": f"DPO Test ({r['register']})", "prompt": r["prompt"].rstrip()}
        for r in test_rows
    ]

    tokenizer = AutoTokenizer.from_pretrained(lora_ckpt_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 1. Evaluate Before DPO (Step 1 LoRA)
    print("1/2 Evaluating Step 1 LoRA (Before DPO)...")
    base_before = AutoModelForCausalLM.from_pretrained("distilgpt2")
    base_before.config.pad_token_id = tokenizer.eos_token_id
    model_before = PeftModel.from_pretrained(base_before, lora_ckpt_dir, is_trainable=False).to(device)
    model_before.eval()

    before_train = evaluate_pairs(model_before, tokenizer, train_rows, device)
    before_test = evaluate_pairs(model_before, tokenizer, test_rows, device)
    before_gens = generate_eval_suite(model_before, tokenizer, eval_prompts, device, max_new_tokens=60, seed=42)
    before_ppl = evaluate_validation_perplexity(model_before, tokenizer, val_path, device=device, block_size=256)

    # 2. Evaluate After DPO (Step 3 DPO)
    print("2/2 Evaluating Step 3 DPO (After DPO)...")
    base_after = AutoModelForCausalLM.from_pretrained("distilgpt2")
    base_after.config.pad_token_id = tokenizer.eos_token_id
    model_after = PeftModel.from_pretrained(base_after, dpo_ckpt_dir, is_trainable=False).to(device)
    model_after.eval()

    after_train = evaluate_pairs(model_after, tokenizer, train_rows, device)
    after_test = evaluate_pairs(model_after, tokenizer, test_rows, device)
    after_gens = generate_eval_suite(model_after, tokenizer, eval_prompts, device, max_new_tokens=60, seed=42)
    after_ppl = evaluate_validation_perplexity(model_after, tokenizer, val_path, device=device, block_size=256)

    before_greedy_agg = aggregate_gen_metrics(before_gens, "greedy_metrics")
    before_sampled_agg = aggregate_gen_metrics(before_gens, "sampled_metrics")
    after_greedy_agg = aggregate_gen_metrics(after_gens, "greedy_metrics")
    after_sampled_agg = aggregate_gen_metrics(after_gens, "sampled_metrics")

    # Attach train accuracy reference for plotting Panel B cleanly
    before_test_plot = dict(before_test, train_acc_ref=before_train["preference_accuracy"])
    after_test_plot = dict(after_test, train_acc_ref=after_train["preference_accuracy"])
    plot_dpo_charts(
        before_test_plot,
        after_test_plot,
        before_greedy_agg,
        after_greedy_agg,
        results_dir / "step3_dpo_margins_chart.png",
    )

    metrics_payload = {
        "device": str(device),
        "dataset_summary": {
            "total_pairs": len(rows),
            "train_pairs": len(train_rows),
            "test_pairs": len(test_rows),
        },
        "test_pairs": {"before_dpo": before_test, "after_dpo": after_test},
        "train_pairs": {
            "before_dpo": {
                "count": before_train["count"],
                "mean_margin": before_train["mean_margin"],
                "preference_accuracy": before_train["preference_accuracy"],
            },
            "after_dpo": {
                "count": after_train["count"],
                "mean_margin": after_train["mean_margin"],
                "preference_accuracy": after_train["preference_accuracy"],
            },
        },
        "generation_summary": {
            "prompt_count": len(eval_prompts),
            "before_dpo": {"greedy": before_greedy_agg, "sampled": before_sampled_agg},
            "after_dpo": {"greedy": after_greedy_agg, "sampled": after_sampled_agg},
        },
        "validation_perplexity": {"before_dpo": before_ppl, "after_dpo": after_ppl},
        "generations": {"before_dpo": before_gens, "after_dpo": after_gens},
    }

    (results_dir / "step3_dpo_metrics.json").write_text(
        json.dumps(metrics_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_markdown_report(
        metrics_payload,
        before_gens,
        after_gens,
        results_dir / "step3_dpo_before_after.md",
    )

    print("\n=== Step 3 DPO Evaluation Summary ===")
    print(f"Test Pref Accuracy: {before_test['preference_accuracy']*100:.1f}% -> {after_test['preference_accuracy']*100:.1f}%")
    print(f"Test Mean Margin:   {before_test['mean_margin']:.4f} -> {after_test['mean_margin']:.4f}")
    print(f"Greedy 3-Gram Rep:  {before_greedy_agg['avg_rep_3_rate']*100:.2f}% -> {after_greedy_agg['avg_rep_3_rate']*100:.2f}%")
    print(f"Val Perplexity:     {before_ppl['val_perplexity']:.2f} -> {after_ppl['val_perplexity']:.2f}")


if __name__ == "__main__":
    main()
