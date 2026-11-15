"""Step 1B: LoRA Fine-Tuning of DistilGPT2 on Cebuano & Before/After Diagnostic.

Loads the prepared Cebuano corpus (data/cebuano_corpus_train.txt & val.txt),
attaches LoRA adapters (r=16, alpha=32) to DistilGPT2's attention projection
layers (c_attn, c_proj), trains with a causal LM objective, plots the training
and validation loss curve, saves the LoRA adapter checkpoint, and compares
before-vs-after perplexity, tokenizer behavior, and text generations.

Outputs:
- checkpoints/distilgpt2-cebuano-lora/
- results/step1_loss_curve.png
- results/step1_training_history.json
- results/step1_before_after_eval.md
- results/step1_summary.json
"""

from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from step0_baseline import evaluate_validation_perplexity, format_bpe_tokens, generate_samples  # noqa: E402


def build_block_dataset(file_path: Path, tokenizer: Any, block_size: int = 256) -> Dataset:
    """Tokenize documents separated by EOS and pack into fixed-length LM blocks."""
    raw_text = file_path.read_text(encoding="utf-8")
    docs = [d.strip() for d in raw_text.split("\n\n") if d.strip()]

    stream: list[int] = []
    eos_id = tokenizer.eos_token_id
    for i in range(0, len(docs), 128):
        batch = docs[i : i + 128]
        encoded = tokenizer(batch, add_special_tokens=False)["input_ids"]
        for ids in encoded:
            stream.extend(ids)
            stream.append(eos_id)

    total_len = (len(stream) // block_size) * block_size
    blocks = [stream[i : i + block_size] for i in range(0, total_len, block_size)]
    return Dataset.from_dict({"input_ids": blocks, "attention_mask": [[1] * block_size] * len(blocks)})


def count_parameters(model: Any) -> dict[str, Any]:
    """Return total, trainable, and frozen parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_params": total,
        "trainable_params": trainable,
        "frozen_params": total - trainable,
        "trainable_percent": round(100.0 * trainable / total, 4),
    }


def make_training_arguments(runs_dir: Path, use_fp16: bool) -> TrainingArguments:
    """Construct TrainingArguments portably across transformers v4.x and v5.x."""
    sig_params = inspect.signature(TrainingArguments.__init__).parameters
    kwargs: dict[str, Any] = {
        "output_dir": str(runs_dir),
        "num_train_epochs": 6,
        "per_device_train_batch_size": 16,
        "per_device_eval_batch_size": 16,
        "learning_rate": 1e-3,
        "lr_scheduler_type": "cosine",
        "weight_decay": 0.01,
        "logging_steps": 20,
        "eval_steps": 50,
        "save_strategy": "no",
        "fp16": use_fp16,
        "report_to": "none",
        "seed": 42,
    }
    if "eval_strategy" in sig_params:
        kwargs["eval_strategy"] = "steps"
    else:
        kwargs["evaluation_strategy"] = "steps"

    if "warmup_ratio" in sig_params:
        kwargs["warmup_ratio"] = 0.05
    elif "warmup_steps" in sig_params:
        kwargs["warmup_steps"] = 0.05

    return TrainingArguments(**kwargs)


def make_trainer(
    model: Any,
    args: TrainingArguments,
    train_ds: Dataset,
    eval_ds: Dataset,
    collator: Any,
    tokenizer: Any,
) -> Trainer:
    """Construct Trainer portably across transformers v4.x and v5.x."""
    sig_params = inspect.signature(Trainer.__init__).parameters
    kwargs: dict[str, Any] = {
        "model": model,
        "args": args,
        "train_dataset": train_ds,
        "eval_dataset": eval_ds,
        "data_collator": collator,
    }
    if "processing_class" in sig_params:
        kwargs["processing_class"] = tokenizer
    elif "tokenizer" in sig_params:
        kwargs["tokenizer"] = tokenizer
    return Trainer(**kwargs)


def plot_loss_curve(log_history: list[dict[str, Any]], out_path: Path) -> dict[str, Any]:
    """Plot training and validation loss curves and return parsed history arrays."""
    train_steps, train_losses = [], []
    eval_steps, eval_losses = [], []

    for entry in log_history:
        if "loss" in entry and "step" in entry:
            train_steps.append(entry["step"])
            train_losses.append(entry["loss"])
        if "eval_loss" in entry and "step" in entry:
            eval_steps.append(entry["step"])
            eval_losses.append(entry["eval_loss"])

    fig, ax = plt.subplots(figsize=(9.5, 5.0), dpi=160)
    if train_steps:
        ax.plot(train_steps, train_losses, label="Training Loss", color="#2b6cb0", linewidth=2.0, marker="o", markersize=3.5)
    if eval_steps:
        ax.plot(eval_steps, eval_losses, label="Validation Loss", color="#dd6b20", linewidth=2.2, marker="s", markersize=5.0)

    ax.set_title("Step 1: DistilGPT2 Cebuano LoRA Fine-Tuning Loss Curve", fontsize=12.5, fontweight="bold")
    ax.set_xlabel("Optimization Step", fontsize=11)
    ax.set_ylabel("Cross-Entropy Loss", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.45)
    ax.legend(frameon=True, fontsize=10)

    if train_losses and eval_losses:
        annot_text = (
            f"Initial Train Loss: {train_losses[0]:.3f}\n"
            f"Final Train Loss:   {train_losses[-1]:.3f}\n"
            f"Final Eval Loss:    {eval_losses[-1]:.3f}"
        )
        ax.text(
            0.98,
            0.78,
            annot_text,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9.5,
            bbox={"boxstyle": "round,pad=0.4", "facecolor": "#f7fafc", "edgecolor": "#cbd5e0"},
        )

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    return {
        "train_steps": train_steps,
        "train_losses": train_losses,
        "eval_steps": eval_steps,
        "eval_losses": eval_losses,
    }


def write_comparison_report(
    param_stats: dict[str, Any],
    loss_summary: dict[str, Any],
    step0_summary: dict[str, Any],
    post_ppl: dict[str, float],
    post_gens: list[dict[str, str]],
    tok_check: list[dict[str, Any]],
    out_path: Path,
) -> None:
    """Write side-by-side Step 0 (Before) vs. Step 1 (After LoRA) report."""
    pre_ppl = step0_summary["validation_perplexity"]
    ppl_drop = pre_ppl["val_perplexity"] - post_ppl["val_perplexity"]
    ppl_pct = 100.0 * ppl_drop / pre_ppl["val_perplexity"]

    lines = [
        "# Step 1 Evaluation: Before vs. After Cebuano LoRA Fine-Tuning",
        "",
        "## 1. LoRA Parameter Budget & Training Summary",
        f"- **Base Model:** `distilgpt2` ({param_stats['frozen_params']:,} frozen parameters)",
        f"- **LoRA Config:** `r=32`, `lora_alpha=64`, `lora_dropout=0.05`, `target_modules=['c_attn', 'c_proj']`, `fan_in_fan_out=True`",
        f"- **Trainable LoRA Parameters:** {param_stats['trainable_params']:,} / {param_stats['total_params']:,} (**{param_stats['trainable_percent']:.2f}%**)",
        f"- **Initial Logged Train Loss:** `{loss_summary['train_losses'][0]:.4f}` → **Final Train Loss:** `{loss_summary['train_losses'][-1]:.4f}`",
        "",
        "## 2. Quantitative Comparison: Held-Out Validation Loss & Perplexity",
        "| Model Stage | Validation Loss | Validation Perplexity | Change |",
        "| :--- | ---: | ---: | :--- |",
        f"| **Step 0: Untouched `distilgpt2`** | `{pre_ppl['val_loss']:.4f}` | `{pre_ppl['val_perplexity']:.2f}` | Baseline |",
        f"| **Step 1: LoRA Fine-Tuned `distilgpt2`** | `{post_ppl['val_loss']:.4f}` | `{post_ppl['val_perplexity']:.2f}` | **-{ppl_drop:.2f} PPL (-{ppl_pct:.1f}%)** |",
        "",
        "## 3. Tokenizer Fragmentation Check (Before vs. After LoRA)",
        "> **Key Architectural Insight:** LoRA adapts attention projection matrices (`W + BA`), leaving the BPE merge rules and 50,257-token embedding vocabulary untouched. Therefore, token counts per word remain identical after Step 1, while the transformer's attention layers learn to compose those fragmented subwords into coherent Cebuano sequences.",
        "",
        "| Cebuano Word | Step 0 Token Count | Step 1 Token Count | Subword Pieces (Unchanged) |",
        "| :--- | ---: | ---: | :--- |",
    ]
    for row in tok_check:
        lines.append(f"| `{row['word']}` | {row['step0_tokens']} | {row['step1_tokens']} | `{row['subwords']}` |")

    lines.extend(["", "## 4. Qualitative Comparison: Before vs. After on 5 Cebuano Prompts", ""])
    pre_by_id = {g["id"]: g for g in step0_summary["generations"]}
    for post_g in post_gens:
        pre_g = pre_by_id[post_g["id"]]
        lines.extend(
            [
                f"### {post_g['id']} ({post_g['register']})",
                f"- **Prompt:** `{post_g['prompt']}`",
                f"- **Step 0 Greedy (Before):** {pre_g['greedy_continuation']}",
                f"- **Step 1 Greedy (After LoRA):** {post_g['greedy_continuation']}",
                f"- **Step 0 Sampled (Before):** {pre_g['sampled_continuation']}",
                f"- **Step 1 Sampled (After LoRA):** {post_g['sampled_continuation']}",
                "",
            ]
        )

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    set_seed(42)
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data"
    results_dir = project_root / "results"
    ckpt_dir = project_root / "checkpoints" / "distilgpt2-cebuano-lora"
    runs_dir = project_root / "checkpoints" / "runs"

    train_path = data_dir / "cebuano_corpus_train.txt"
    val_path = data_dir / "cebuano_corpus_val.txt"
    step0_json_path = results_dir / "step0_summary.json"
    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError("Missing train/val splits. Run scripts/prepare_corpus.py first.")
    if not step0_json_path.exists():
        raise FileNotFoundError("Missing results/step0_summary.json. Run scripts/step0_baseline.py first.")

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_fp16 = torch.cuda.is_available()
    print(f"Running Step 1 LoRA fine-tuning on device: {device} (fp16={use_fp16})")

    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    base_model.config.pad_token_id = tokenizer.eos_token_id

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        target_modules=["c_attn", "c_proj"],
        fan_in_fan_out=True,
    )
    model = get_peft_model(base_model, lora_config)
    param_stats = count_parameters(model)
    print(
        f"Trainable parameters: {param_stats['trainable_params']:,} / {param_stats['total_params']:,} "
        f"({param_stats['trainable_percent']:.2f}%)"
    )

    print("Building 256-token causal LM dataset blocks...")
    train_ds = build_block_dataset(train_path, tokenizer, block_size=256)
    val_ds = build_block_dataset(val_path, tokenizer, block_size=256)
    print(f"  Train blocks: {len(train_ds):,} | Val blocks: {len(val_ds):,}")

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    training_args = make_training_arguments(runs_dir, use_fp16=use_fp16)
    trainer = make_trainer(model, training_args, train_ds, val_ds, collator, tokenizer)

    print("Starting LoRA training...")
    train_result = trainer.train()
    print(f"Training completed! Final trainer train_loss: {train_result.training_loss:.4f}")

    print(f"Saving LoRA adapter & tokenizer to {ckpt_dir}...")
    model.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)

    log_history = trainer.state.log_history
    (results_dir / "step1_training_history.json").write_text(
        json.dumps(log_history, indent=2) + "\n",
        encoding="utf-8",
    )
    loss_summary = plot_loss_curve(log_history, results_dir / "step1_loss_curve.png")

    print("Running post-LoRA validation perplexity and text generation...")
    model.to(device)
    post_ppl = evaluate_validation_perplexity(model, tokenizer, val_path, device=device, block_size=256)
    post_gens = generate_samples(model, tokenizer, device=device, max_new_tokens=60, seed=42)

    step0_summary = json.loads(step0_json_path.read_text(encoding="utf-8"))
    sample_words = ["kagamhanan", "eskwelahan", "kapulisan", "tinun-an", "malipayon", "panaghiusa"]
    step0_tok = AutoTokenizer.from_pretrained("distilgpt2")
    tok_check = []
    for word in sample_words:
        t0 = step0_tok.tokenize(word)
        t1 = tokenizer.tokenize(word)
        tok_check.append(
            {
                "word": word,
                "step0_tokens": len(t0),
                "step1_tokens": len(t1),
                "subwords": format_bpe_tokens(t1),
            }
        )

    write_comparison_report(
        param_stats=param_stats,
        loss_summary=loss_summary,
        step0_summary=step0_summary,
        post_ppl=post_ppl,
        post_gens=post_gens,
        tok_check=tok_check,
        out_path=results_dir / "step1_before_after_eval.md",
    )

    step1_summary = {
        "device": str(device),
        "lora_config": {
            "r": 32,
            "lora_alpha": 64,
            "lora_dropout": 0.05,
            "target_modules": ["c_attn", "c_proj"],
            "fan_in_fan_out": True,
        },
        "parameters": param_stats,
        "dataset_blocks": {"train_blocks": len(train_ds), "val_blocks": len(val_ds), "block_size": 256},
        "training_metrics": {
            "initial_train_loss": loss_summary["train_losses"][0] if loss_summary["train_losses"] else None,
            "final_train_loss": loss_summary["train_losses"][-1] if loss_summary["train_losses"] else None,
            "final_eval_loss": loss_summary["eval_losses"][-1] if loss_summary["eval_losses"] else None,
            "trainer_avg_train_loss": round(train_result.training_loss, 4),
        },
        "before_perplexity": step0_summary["validation_perplexity"],
        "after_perplexity": post_ppl,
        "tokenizer_check": tok_check,
        "generations": post_gens,
    }
    (results_dir / "step1_summary.json").write_text(
        json.dumps(step1_summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("\n=== Step 1 Post-LoRA Summary ===")
    print(f"Step 0 Val Loss: {step0_summary['validation_perplexity']['val_loss']:.4f} | PPL: {step0_summary['validation_perplexity']['val_perplexity']:.2f}")
    print(f"Step 1 Val Loss: {post_ppl['val_loss']:.4f} | PPL: {post_ppl['val_perplexity']:.2f}")


if __name__ == "__main__":
    main()
