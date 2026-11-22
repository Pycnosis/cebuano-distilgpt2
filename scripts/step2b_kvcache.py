"""Step 2B: KV-Cache Inference Latency Benchmark on LoRA-Adapted DistilGPT2.

Loads base distilgpt2 + the Step 1 Cebuano LoRA adapter
(checkpoints/distilgpt2-cebuano-lora) and benchmarks autoregressive generation
latency with use_cache=True vs. use_cache=False across max_new_tokens in
[32, 64, 128, 256] on the canonical Cebuano prompt P2_news_gov.

When CUDA is available and --device=auto (default), benchmarks both CPU
(compute-bound regime exposing textbook O(T^2) vs. O(T) scaling) and CUDA GPU
(highlighting kernel-launch/dispatch overhead for an 82M model at batch_size=1).
On CPU-only machines, runs cleanly on CPU alone.

Outputs:
- results/step2_kvcache_benchmark.csv
- results/step2_kvcache_chart.png
- results/step2_kvcache_summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from step0_baseline import PROMPTS  # noqa: E402

DEFAULT_LENGTHS: list[int] = [32, 64, 128, 256]
TARGET_PROMPT_ID = "P2_news_gov"


def get_target_prompt(prompt_id: str = TARGET_PROMPT_ID) -> dict[str, str]:
    """Return the canonical Cebuano prompt entry from Step 0."""
    for item in PROMPTS:
        if item["id"] == prompt_id:
            return item
    raise KeyError(f"Prompt id {prompt_id!r} not found in step0_baseline.PROMPTS.")


def get_cpu_name() -> str:
    """Return a clean human-readable CPU identifier."""
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.lower().startswith("model name"):
                _, val = line.split(":", 1)
                return val.strip()
    proc = platform.processor().strip()
    return proc if proc else "CPU"


def sync_device(device: torch.device) -> None:
    """Synchronize CUDA stream before reading wall-clock timer."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def benchmark_single_setting(
    model: Any,
    inputs: dict[str, torch.Tensor],
    pad_token_id: int,
    max_new_tokens: int,
    use_cache: bool,
    runs: int,
    device: torch.device,
    warmup: bool = True,
) -> dict[str, Any]:
    """Run optional warm-up plus timed greedy generations for one (length, use_cache) pair."""
    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "min_new_tokens": max_new_tokens,
        "do_sample": False,
        "use_cache": use_cache,
        "pad_token_id": pad_token_id,
    }

    if warmup:
        sync_device(device)
        _ = model.generate(**inputs, **gen_kwargs)
        sync_device(device)

    latencies: list[float] = []
    last_output_ids: torch.Tensor | None = None
    for _ in range(runs):
        sync_device(device)
        t0 = time.perf_counter()
        out_ids = model.generate(**inputs, **gen_kwargs)
        sync_device(device)
        t1 = time.perf_counter()
        latencies.append(t1 - t0)
        last_output_ids = out_ids

    assert last_output_ids is not None
    prompt_len = int(inputs["input_ids"].shape[1])
    gen_tokens = int(last_output_ids.shape[1] - prompt_len)

    mean_s = statistics.mean(latencies)
    std_s = statistics.stdev(latencies) if len(latencies) > 1 else 0.0
    ms_per_token = (mean_s * 1000.0) / max(1, gen_tokens)
    tokens_per_sec = gen_tokens / mean_s if mean_s > 0 else 0.0

    return {
        "max_new_tokens": max_new_tokens,
        "prompt_tokens": prompt_len,
        "generated_tokens": gen_tokens,
        "total_sequence_tokens": prompt_len + gen_tokens,
        "use_cache": use_cache,
        "runs": runs,
        "trial_latencies_s": [round(x, 6) for x in latencies],
        "mean_latency_s": round(mean_s, 6),
        "std_latency_s": round(std_s, 6),
        "min_latency_s": round(min(latencies), 6),
        "max_latency_s": round(max(latencies), 6),
        "ms_per_token": round(ms_per_token, 4),
        "tokens_per_sec": round(tokens_per_sec, 2),
        "output_ids": last_output_ids[0].detach().cpu().tolist(),
    }


def compute_kv_cache_memory_kb(
    n_layer: int,
    d_model: int,
    seq_len: int,
    bytes_per_element: int = 4,
) -> float:
    """Compute theoretical KV-cache memory (KB) for batch_size=1: 2 * L * S * d * b."""
    total_bytes = 2 * n_layer * seq_len * d_model * bytes_per_element
    return round(total_bytes / 1024.0, 2)


def benchmark_on_device(
    lora_dir: Path,
    tokenizer: Any,
    prompt_entry: dict[str, str],
    lengths: list[int],
    runs: int,
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run the full KV-cache length sweep on a single target device."""
    if device.type == "cuda":
        device_name = torch.cuda.get_device_name(device)
        device_label = f"CUDA ({device_name})"
    else:
        device_name = get_cpu_name()
        device_label = f"CPU ({device_name})"

    print(f"\nBenchmarking on {device_label} ({runs} timed runs/setting)...")
    base_model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    model = PeftModel.from_pretrained(base_model, lora_dir).to(device)
    model.eval()

    config = model.base_model.model.config
    n_layer = int(config.n_layer)
    d_model = int(config.n_embd)
    dtype_bytes = next(model.parameters()).element_size()

    inputs = tokenizer(prompt_entry["prompt"], return_tensors="pt").to(device)
    prompt_tokens = int(inputs["input_ids"].shape[1])

    # Device-level warm-up pass (both cache paths) before timing
    for uc in (True, False):
        sync_device(device)
        _ = model.generate(
            **inputs,
            max_new_tokens=16,
            min_new_tokens=16,
            do_sample=False,
            use_cache=uc,
            pad_token_id=tokenizer.eos_token_id,
        )
        sync_device(device)

    paired_results: list[dict[str, Any]] = []
    csv_rows: list[dict[str, Any]] = []
    per_setting_warmup = device.type == "cuda"

    for max_new in lengths:
        print(f"  [{device.type}] max_new_tokens={max_new}...")
        res_cache = benchmark_single_setting(
            model=model,
            inputs=inputs,
            pad_token_id=tokenizer.eos_token_id,
            max_new_tokens=max_new,
            use_cache=True,
            runs=runs,
            device=device,
            warmup=per_setting_warmup,
        )
        res_nocache = benchmark_single_setting(
            model=model,
            inputs=inputs,
            pad_token_id=tokenizer.eos_token_id,
            max_new_tokens=max_new,
            use_cache=False,
            runs=runs,
            device=device,
            warmup=per_setting_warmup,
        )

        outputs_identical = res_cache["output_ids"] == res_nocache["output_ids"]
        speedup = round(res_nocache["mean_latency_s"] / res_cache["mean_latency_s"], 4)
        latency_saved_pct = round(
            100.0 * (1.0 - res_cache["mean_latency_s"] / res_nocache["mean_latency_s"]),
            2,
        )
        total_seq = prompt_tokens + max_new
        kv_mem_kb = compute_kv_cache_memory_kb(n_layer, d_model, total_seq, bytes_per_element=dtype_bytes)
        sample_decoded = tokenizer.decode(
            res_cache["output_ids"][prompt_tokens:],
            skip_special_tokens=True,
        ).strip()

        res_cache_clean = {k: v for k, v in res_cache.items() if k != "output_ids"}
        res_nocache_clean = {k: v for k, v in res_nocache.items() if k != "output_ids"}

        paired_results.append(
            {
                "max_new_tokens": max_new,
                "prompt_tokens": prompt_tokens,
                "total_sequence_tokens": total_seq,
                "with_cache": res_cache_clean,
                "without_cache": res_nocache_clean,
                "speedup_factor": speedup,
                "latency_saved_pct": latency_saved_pct,
                "outputs_identical": outputs_identical,
                "kv_cache_memory_kb": kv_mem_kb,
                "generated_preview": sample_decoded[:160],
            }
        )

        for entry, spd in [(res_cache_clean, speedup), (res_nocache_clean, 1.0)]:
            csv_rows.append(
                {
                    "device": device.type,
                    "device_name": device_name,
                    "prompt_id": prompt_entry["id"],
                    "prompt_tokens": prompt_tokens,
                    "max_new_tokens": max_new,
                    "generated_tokens": entry["generated_tokens"],
                    "use_cache": entry["use_cache"],
                    "runs": runs,
                    "mean_latency_s": entry["mean_latency_s"],
                    "std_latency_s": entry["std_latency_s"],
                    "min_latency_s": entry["min_latency_s"],
                    "max_latency_s": entry["max_latency_s"],
                    "ms_per_token": entry["ms_per_token"],
                    "tokens_per_sec": entry["tokens_per_sec"],
                    "speedup_factor": spd,
                    "kv_cache_memory_kb": kv_mem_kb if entry["use_cache"] else 0.0,
                    "outputs_identical": outputs_identical,
                }
            )

        print(
            f"    use_cache=True:  {res_cache_clean['mean_latency_s']:.4f}s ({res_cache_clean['ms_per_token']:.2f} ms/tok) | "
            f"use_cache=False: {res_nocache_clean['mean_latency_s']:.4f}s ({res_nocache_clean['ms_per_token']:.2f} ms/tok) | "
            f"Speedup: {speedup:.2f}x | Identical: {outputs_identical}"
        )

    device_summary = {
        "device": device.type,
        "device_name": device_name,
        "dtype": str(next(model.parameters()).dtype),
        "dtype_bytes": dtype_bytes,
        "timed_runs_per_setting": runs,
        "results": paired_results,
    }

    del model
    del base_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return device_summary, csv_rows


def plot_kvcache_chart(
    benchmarks: dict[str, dict[str, Any]],
    out_path: Path,
) -> None:
    """Render a 2-panel latency curve + KV-cache speedup bar chart."""
    has_cpu = "cpu" in benchmarks
    has_cuda = "cuda" in benchmarks
    primary_key = "cpu" if has_cpu else "cuda"
    primary_rows = benchmarks[primary_key]["results"]
    lengths = [r["max_new_tokens"] for r in primary_rows]

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.9), dpi=160)

    # Panel 1: Generation Latency vs. max_new_tokens
    max_y = 0.0
    if has_cpu:
        cpu_rows = benchmarks["cpu"]["results"]
        cpu_nc = [r["without_cache"]["mean_latency_s"] for r in cpu_rows]
        cpu_nc_std = [r["without_cache"]["std_latency_s"] for r in cpu_rows]
        cpu_c = [r["with_cache"]["mean_latency_s"] for r in cpu_rows]
        cpu_c_std = [r["with_cache"]["std_latency_s"] for r in cpu_rows]
        max_y = max(max_y, max(cpu_nc))

        axes[0].errorbar(
            lengths,
            cpu_nc,
            yerr=cpu_nc_std,
            marker="o",
            linewidth=2.3,
            capsize=4,
            color="#dd6b20",
            label="CPU: use_cache=False — O(T²)",
        )
        axes[0].errorbar(
            lengths,
            cpu_c,
            yerr=cpu_c_std,
            marker="s",
            linewidth=2.3,
            capsize=4,
            color="#2b6cb0",
            label="CPU: use_cache=True — O(T)",
        )
        for x_val, y_nc, y_c in zip(lengths, cpu_nc, cpu_c):
            axes[0].annotate(
                f"{y_nc:.2f}s",
                (x_val, y_nc),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=8.5,
                fontweight="bold",
                color="#9c4221",
            )
            c_offset = (18, -2) if x_val <= 64 else (0, -15)
            axes[0].annotate(
                f"{y_c:.2f}s",
                (x_val, y_c),
                textcoords="offset points",
                xytext=c_offset,
                ha="center",
                fontsize=8.5,
                fontweight="bold",
                color="#1a365d",
            )

    if has_cuda:
        gpu_rows = benchmarks["cuda"]["results"]
        gpu_nc = [r["without_cache"]["mean_latency_s"] for r in gpu_rows]
        gpu_c = [r["with_cache"]["mean_latency_s"] for r in gpu_rows]
        max_y = max(max_y, max(gpu_nc))
        ls_nc = "--" if has_cpu else "-"
        ls_c = ":" if has_cpu else "-"

        axes[0].plot(
            lengths,
            gpu_nc,
            marker="^",
            linestyle=ls_nc,
            linewidth=1.8,
            color="#c53030",
            label="RTX 4050 GPU: use_cache=False",
        )
        axes[0].plot(
            lengths,
            gpu_c,
            marker="D",
            linestyle=ls_c,
            linewidth=1.8,
            color="#2f855a",
            label="RTX 4050 GPU: use_cache=True",
        )
        if not has_cpu:
            for x_val, y_nc, y_c in zip(lengths, gpu_nc, gpu_c):
                axes[0].annotate(
                    f"{y_nc:.2f}s",
                    (x_val, y_nc),
                    textcoords="offset points",
                    xytext=(0, 8),
                    ha="center",
                    fontsize=8.5,
                    color="#9b2c2c",
                )
                axes[0].annotate(
                    f"{y_c:.2f}s",
                    (x_val, y_c),
                    textcoords="offset points",
                    xytext=(0, -14),
                    ha="center",
                    fontsize=8.5,
                    color="#1c4532",
                )

    axes[0].set_title("Autoregressive Generation Latency\nvs. Output Length (P2_news_gov)", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Generated Tokens (max_new_tokens)")
    axes[0].set_ylabel("Mean Wall-Clock Latency (seconds)")
    axes[0].set_xticks(lengths)
    axes[0].set_ylim(0, max_y * 1.22)
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[0].legend(loc="upper left", fontsize=8.3)

    # Panel 2: Speedup Factor (time_no_cache / time_with_cache)
    x_pos = list(range(len(lengths)))
    max_spd = 1.0
    if has_cpu and has_cuda:
        width = 0.35
        cpu_spds = [r["speedup_factor"] for r in benchmarks["cpu"]["results"]]
        gpu_spds = [r["speedup_factor"] for r in benchmarks["cuda"]["results"]]
        max_spd = max(max(cpu_spds), max(gpu_spds))

        b_cpu = axes[1].bar(
            [i - width / 2 for i in x_pos],
            cpu_spds,
            width,
            label="CPU (Compute-Bound)",
            color="#2b6cb0",
            edgecolor="#1a365d",
            linewidth=0.8,
        )
        b_gpu = axes[1].bar(
            [i + width / 2 for i in x_pos],
            gpu_spds,
            width,
            label="RTX 4050 GPU (Launch-Bound)",
            color="#2f855a",
            edgecolor="#1c4532",
            linewidth=0.8,
        )
        for bar in b_cpu:
            h = bar.get_height()
            axes[1].text(
                bar.get_x() + bar.get_width() / 2.0,
                h + max_spd * 0.025,
                f"{h:.2f}×",
                ha="center",
                va="bottom",
                fontsize=8.8,
                fontweight="bold",
                color="#1a365d",
            )
        for bar in b_gpu:
            h = bar.get_height()
            axes[1].text(
                bar.get_x() + bar.get_width() / 2.0,
                h + max_spd * 0.025,
                f"{h:.2f}×",
                ha="center",
                va="bottom",
                fontsize=8.8,
                fontweight="bold",
                color="#1c4532",
            )
    else:
        spds = [r["speedup_factor"] for r in primary_rows]
        max_spd = max(spds)
        label = "CPU" if has_cpu else "CUDA GPU"
        bars = axes[1].bar(x_pos, spds, width=0.52, label=f"{label} Speedup", color="#2b6cb0", edgecolor="#1a365d")
        for bar, spd in zip(bars, spds):
            axes[1].text(
                bar.get_x() + bar.get_width() / 2.0,
                spd + max_spd * 0.03,
                f"{spd:.2f}×",
                ha="center",
                va="bottom",
                fontsize=9.2,
                fontweight="bold",
                color="#1a365d",
            )

    axes[1].axhline(1.0, color="#718096", linestyle=":", linewidth=1.2, label="1.0× Parity Baseline")
    axes[1].set_title("KV-Cache Speedup Factor\n(time_no_cache / time_with_cache)", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Generated Tokens (max_new_tokens)")
    axes[1].set_ylabel("Speedup Factor (×)")
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels([str(n) for n in lengths])
    axes[1].set_ylim(0, max_spd * 1.26)
    axes[1].grid(axis="y", linestyle="--", alpha=0.4)
    axes[1].legend(loc="upper left", fontsize=8.5)

    subtitle = "CPU (Compute-Bound) vs. RTX 4050 GPU" if (has_cpu and has_cuda) else benchmarks[primary_key]["device_name"]
    fig.suptitle(
        f"Step 2B: DistilGPT2-Cebuano LoRA KV-Cache Benchmark — {subtitle}",
        fontsize=12.3,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def run_benchmark(
    lengths: list[int] | None = None,
    cpu_runs: int = 3,
    gpu_runs: int = 5,
    device_preference: str = "auto",
) -> dict[str, Any]:
    """Execute the Step 2B KV-cache benchmark and save CSV, chart, and JSON."""
    project_root = Path(__file__).resolve().parent.parent
    lora_dir = project_root / "checkpoints" / "distilgpt2-cebuano-lora"
    results_dir = project_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    if not lora_dir.exists():
        raise FileNotFoundError(
            f"Missing LoRA checkpoint at {lora_dir}. Run scripts/step1_train_lora.py first."
        )

    target_lengths = lengths if lengths is not None else list(DEFAULT_LENGTHS)
    prompt_entry = get_target_prompt(TARGET_PROMPT_ID)

    tokenizer = AutoTokenizer.from_pretrained(lora_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    prompt_tokens = len(tokenizer.encode(prompt_entry["prompt"], add_special_tokens=False))

    # Determine devices to benchmark
    devices_to_run: list[tuple[torch.device, int]] = []
    if device_preference == "auto":
        devices_to_run.append((torch.device("cpu"), cpu_runs))
        if torch.cuda.is_available():
            devices_to_run.append((torch.device("cuda"), gpu_runs))
    elif device_preference == "cpu":
        devices_to_run.append((torch.device("cpu"), cpu_runs))
    elif device_preference == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested via --device cuda, but no GPU is available.")
        devices_to_run.append((torch.device("cuda"), gpu_runs))
    elif device_preference == "both":
        devices_to_run.append((torch.device("cpu"), cpu_runs))
        if torch.cuda.is_available():
            devices_to_run.append((torch.device("cuda"), gpu_runs))

    benchmarks: dict[str, dict[str, Any]] = {}
    all_csv_rows: list[dict[str, Any]] = []
    for dev, n_runs in devices_to_run:
        dev_summary, dev_csv = benchmark_on_device(
            lora_dir=lora_dir,
            tokenizer=tokenizer,
            prompt_entry=prompt_entry,
            lengths=target_lengths,
            runs=n_runs,
            device=dev,
        )
        benchmarks[dev.type] = dev_summary
        all_csv_rows.extend(dev_csv)

    # Read architecture metadata from distilgpt2 config
    cfg = AutoModelForCausalLM.from_pretrained("distilgpt2").config
    n_layer = int(cfg.n_layer)
    n_head = int(cfg.n_head)
    d_model = int(cfg.n_embd)
    d_head = d_model // n_head

    csv_path = results_dir / "step2_kvcache_benchmark.csv"
    fieldnames = list(all_csv_rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_csv_rows)

    chart_path = results_dir / "step2_kvcache_chart.png"
    plot_kvcache_chart(benchmarks, out_path=chart_path)

    primary_device = "cpu" if "cpu" in benchmarks else "cuda"
    summary = {
        "model": "distilgpt2 + checkpoints/distilgpt2-cebuano-lora",
        "primary_device": primary_device,
        "devices_benchmarked": list(benchmarks.keys()),
        "architecture": {
            "n_layer": n_layer,
            "n_head": n_head,
            "d_model": d_model,
            "d_head": d_head,
            "max_positions": int(cfg.n_positions),
            "kv_bytes_per_token_fp32": 2 * n_layer * d_model * 4,
            "kv_bytes_per_token_fp16": 2 * n_layer * d_model * 2,
            "kv_cache_kb_at_1024_fp32": compute_kv_cache_memory_kb(n_layer, d_model, int(cfg.n_positions), 4),
            "kv_cache_kb_at_1024_fp16": compute_kv_cache_memory_kb(n_layer, d_model, int(cfg.n_positions), 2),
        },
        "prompt": {
            "id": prompt_entry["id"],
            "register": prompt_entry["register"],
            "text": prompt_entry["prompt"],
            "prompt_tokens": prompt_tokens,
        },
        "benchmark_config": {
            "decoding": "greedy (do_sample=False)",
            "fixed_length_enforced": True,
            "lengths": target_lengths,
            "cpu_timed_runs": cpu_runs if "cpu" in benchmarks else None,
            "gpu_timed_runs": gpu_runs if "cuda" in benchmarks else None,
        },
        "benchmarks": benchmarks,
    }

    summary_path = results_dir / "step2_kvcache_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("\n=== Step 2B KV-Cache Benchmark Summary ===")
    for dev_key, dev_data in benchmarks.items():
        print(f"\n[{dev_key.upper()}] {dev_data['device_name']} ({dev_data['timed_runs_per_setting']} runs/setting):")
        for item in dev_data["results"]:
            print(
                f"  max_new_tokens={item['max_new_tokens']:>3d}: "
                f"cache={item['with_cache']['mean_latency_s']:.4f}s ({item['with_cache']['ms_per_token']:.2f} ms/tok) vs. "
                f"no_cache={item['without_cache']['mean_latency_s']:.4f}s ({item['without_cache']['ms_per_token']:.2f} ms/tok) "
                f"-> {item['speedup_factor']:.2f}x speedup ({item['latency_saved_pct']:.1f}% faster)"
            )
    print(f"\nSaved: {csv_path.name}, {chart_path.name}, {summary_path.name}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark KV-cache speedup on DistilGPT2-Cebuano LoRA.")
    parser.add_argument(
        "--lengths",
        type=int,
        nargs="+",
        default=DEFAULT_LENGTHS,
        help="List of max_new_tokens values to benchmark (default: 32 64 128 256).",
    )
    parser.add_argument(
        "--cpu-runs",
        type=int,
        default=3,
        help="Number of timed runs per setting on CPU (default: 3).",
    )
    parser.add_argument(
        "--gpu-runs",
        type=int,
        default=5,
        help="Number of timed runs per setting on CUDA GPU (default: 5).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "both", "cpu", "cuda"],
        help="Target device(s): auto runs CPU + CUDA if available (default: auto).",
    )
    args = parser.parse_args()
    run_benchmark(
        lengths=args.lengths,
        cpu_runs=args.cpu_runs,
        gpu_runs=args.gpu_runs,
        device_preference=args.device,
    )


if __name__ == "__main__":
    main()
