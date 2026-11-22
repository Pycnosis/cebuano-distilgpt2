"""Step 3B (Part 1): Direct Preference Optimization (DPO) Alignment on Cebuano.

Loads the Step 1 LoRA-adapted DistilGPT2 checkpoint (checkpoints/distilgpt2-cebuano-lora)
as both the trainable policy and a frozen reference policy, then aligns it using
trl.DPOTrainer on data/cebuano_dpo_preferences.jsonl to prefer clean, idiomatic Cebuano
continuations over English code-switching, repetitive n-gram loops, and subword gibberish.

Outputs:
- checkpoints/distilgpt2-cebuano-dpo/
- results/step3_dpo_train_history.json
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import DPOConfig, DPOTrainer


def load_dpo_splits(jsonl_path: Path) -> tuple[Dataset, Dataset]:
    """Load train and test splits from cebuano_dpo_preferences.jsonl."""
    rows = [
        json.loads(line)
        for line in jsonl_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    train_rows = [
        {"prompt": r["prompt"], "chosen": r["chosen"], "rejected": r["rejected"]}
        for r in rows
        if r.get("split", "train") == "train"
    ]
    test_rows = [
        {"prompt": r["prompt"], "chosen": r["chosen"], "rejected": r["rejected"]}
        for r in rows
        if r.get("split") == "test"
    ]
    if not train_rows:
        raise ValueError(f"No training pairs found in {jsonl_path}")
    return Dataset.from_list(train_rows), Dataset.from_list(test_rows)


def make_dpo_config(runs_dir: Path, has_eval: bool) -> DPOConfig:
    """Construct DPOConfig portably across trl versions."""
    sig_params = inspect.signature(DPOConfig.__init__).parameters
    use_cuda = torch.cuda.is_available()

    candidates: dict[str, Any] = {
        "output_dir": str(runs_dir),
        "beta": 0.1,
        "learning_rate": 5e-5,
        "num_train_epochs": 4,
        "per_device_train_batch_size": 2,
        "per_device_eval_batch_size": 2,
        "gradient_accumulation_steps": 2,
        "lr_scheduler_type": "cosine",
        "warmup_steps": 3,
        "logging_steps": 2,
        "save_strategy": "no",
        "max_length": 256,
        "max_prompt_length": 96,
        "fp16": use_cuda,
        "bf16": False,
        "use_cpu": not use_cuda,
        "remove_unused_columns": False,
        "report_to": "none",
        "seed": 42,
    }
    eval_value = "epoch" if has_eval else "no"
    if "eval_strategy" in sig_params:
        candidates["eval_strategy"] = eval_value
    elif "evaluation_strategy" in sig_params:
        candidates["evaluation_strategy"] = eval_value

    filtered = {k: v for k, v in candidates.items() if k in sig_params}
    return DPOConfig(**filtered)


def make_dpo_trainer(
    policy_model: Any,
    ref_model: Any,
    dpo_args: DPOConfig,
    train_ds: Dataset,
    eval_ds: Dataset,
    tokenizer: Any,
) -> DPOTrainer:
    """Construct DPOTrainer portably across trl versions."""
    sig_params = inspect.signature(DPOTrainer.__init__).parameters
    kwargs: dict[str, Any] = {
        "model": policy_model,
        "ref_model": ref_model,
        "args": dpo_args,
        "train_dataset": train_ds,
    }
    if len(eval_ds) > 0 and "eval_dataset" in sig_params:
        kwargs["eval_dataset"] = eval_ds

    if "processing_class" in sig_params:
        kwargs["processing_class"] = tokenizer
    elif "tokenizer" in sig_params:
        kwargs["tokenizer"] = tokenizer

    return DPOTrainer(**kwargs)


def main() -> None:
    set_seed(42)
    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / "data" / "cebuano_dpo_preferences.jsonl"
    lora_ckpt_dir = project_root / "checkpoints" / "distilgpt2-cebuano-lora"
    dpo_ckpt_dir = project_root / "checkpoints" / "distilgpt2-cebuano-dpo"
    runs_dir = project_root / "checkpoints" / "runs_dpo"
    results_dir = project_root / "results"

    if not data_path.exists():
        raise FileNotFoundError(f"Missing preference dataset: {data_path}")
    if not lora_ckpt_dir.exists():
        raise FileNotFoundError(f"Missing Step 1 LoRA checkpoint: {lora_ckpt_dir}")

    dpo_ckpt_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running Step 3 DPO alignment on device: {device} (fp16={torch.cuda.is_available()})")

    tokenizer = AutoTokenizer.from_pretrained(lora_ckpt_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load trainable policy starting from Step 1 Cebuano LoRA adapter
    base_policy = AutoModelForCausalLM.from_pretrained("distilgpt2")
    base_policy.config.pad_token_id = tokenizer.eos_token_id
    policy_model = PeftModel.from_pretrained(base_policy, lora_ckpt_dir, is_trainable=True)

    # Load frozen Step 1 LoRA adapter merged into a plain GPT2LMHeadModel as ref_model
    # so DPOTrainer accepts it alongside a PeftModel policy and regularizes against
    # the Step 1 Cebuano-adapted model rather than untouched English distilgpt2.
    base_ref = AutoModelForCausalLM.from_pretrained("distilgpt2")
    base_ref.config.pad_token_id = tokenizer.eos_token_id
    ref_model = PeftModel.from_pretrained(base_ref, lora_ckpt_dir, is_trainable=False).merge_and_unload()
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad_(False)

    trainable_params = sum(p.numel() for p in policy_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in policy_model.parameters())
    print(f"Trainable policy params: {trainable_params:,} / {total_params:,} ({100.0 * trainable_params / total_params:.2f}%)")

    train_ds, eval_ds = load_dpo_splits(data_path)
    print(f"Loaded DPO dataset: {len(train_ds)} train pairs | {len(eval_ds)} held-out test pairs")

    dpo_args = make_dpo_config(runs_dir, has_eval=len(eval_ds) > 0)
    trainer = make_dpo_trainer(policy_model, ref_model, dpo_args, train_ds, eval_ds, tokenizer)

    print("Starting DPO training...")
    train_result = trainer.train()
    print(f"DPO training completed! Final train_loss: {train_result.training_loss:.4f}")

    print(f"Saving aligned DPO LoRA adapter & tokenizer to {dpo_ckpt_dir}...")
    policy_model.save_pretrained(dpo_ckpt_dir)
    tokenizer.save_pretrained(dpo_ckpt_dir)

    history_payload = {
        "device": device,
        "hyperparameters": {
            "beta": 0.1,
            "learning_rate": 5e-5,
            "num_train_epochs": 4,
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 2,
            "lr_scheduler_type": "cosine",
            "max_length": 256,
            "seed": 42,
        },
        "dataset_counts": {"train": len(train_ds), "test": len(eval_ds)},
        "final_training_loss": round(float(train_result.training_loss), 4),
        "log_history": trainer.state.log_history,
    }
    history_path = results_dir / "step3_dpo_train_history.json"
    history_path.write_text(json.dumps(history_payload, indent=2) + "\n", encoding="utf-8")
    print(f"Saved training history to {history_path}")


if __name__ == "__main__":
    main()
