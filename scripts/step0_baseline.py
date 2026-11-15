"""Step 0: Baseline Diagnostic — Prove untouched distilgpt2 doesn't know Cebuano.

Evaluates:
1. BPE tokenizer fragmentation on 12 length-matched Cebuano vs. English words
   and 5 parallel sentences (fertility = tokens/word, chars/token).
2. Untouched distilgpt2 text generation (greedy & sampling) across 5 Cebuano prompts.
3. Held-out Cebuano validation loss and perplexity on data/cebuano_corpus_val.txt.

Outputs:
- results/step0_tokenizer_comparison.csv
- results/step0_fragmentation_chart.png
- results/step0_baseline_generations.md
- results/step0_summary.json
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

WORD_PAIRS: list[tuple[str, str]] = [
    ("kagamhanan", "government"),
    ("eskwelahan", "university"),
    ("kapulisan", "policemen"),
    ("tinun-an", "students"),
    ("panghitabo", "occurrence"),
    ("malipayon", "delighted"),
    ("kaugmaon", "tomorrow"),
    ("kasaysayan", "literature"),
    ("pagtulun-an", "instruction"),
    ("kinabuhi", "lifetime"),
    ("kalibutan", "continent"),
    ("panaghiusa", "fellowship"),
]

SENTENCE_PAIRS: list[tuple[str, str]] = [
    (
        "Gipahibalo sa lokal nga kagamhanan sa Sugbo nga adunay bag-ong proyekto.",
        "The local government of Cebu announced that there is a new project.",
    ),
    (
        "Ang mga bata sa eskwelahan malipayon nga nagbasa og mga libro.",
        "The children in school are happily reading several books together.",
    ),
    (
        "Sumala pa sa imbestigasyon sa kapulisan, nahitabo ang aksidente sa gabii.",
        "According to the police investigation, the accident happened at night.",
    ),
    (
        "Sa usa ka gamay nga baryo duol sa dagat, nagpuyo ang usa ka pamilya.",
        "In a small village near the sea, a family lived peacefully together.",
    ),
    (
        "Kinahanglan nato ang panaghiusa ug kalinaw alang sa maayong kaugmaon.",
        "We really need unity and peace for a much better future ahead.",
    ),
]

PROMPTS: list[dict[str, str]] = [
    {
        "id": "P1_conversational",
        "register": "Conversational",
        "prompt": "Kumusta ka? Manghinaut ko nga",
    },
    {
        "id": "P2_news_gov",
        "register": "News (Local Government)",
        "prompt": "Gipahibalo sa lokal nga kagamhanan sa Sugbo nga",
    },
    {
        "id": "P3_news_police",
        "register": "News (Police Report)",
        "prompt": "Matud pa sa kapulisan sa dakbayan,",
    },
    {
        "id": "P4_literary",
        "register": "Literary / Narrative",
        "prompt": "Sa usa ka gamay nga baryo duol sa dagat, adunay",
    },
    {
        "id": "P5_pedagogical",
        "register": "Pedagogical / School",
        "prompt": "Ang mga bata sa eskwelahan malipayon nga",
    },
]


def format_bpe_tokens(tokens: list[str]) -> str:
    """Render GPT-2 BPE subword pieces legibly (replacing byte-space Ġ with _)."""
    return " | ".join(tok.replace("Ġ", "_") for tok in tokens)


def analyze_tokenizer(tokenizer: Any, results_dir: Path) -> dict[str, Any]:
    """Compare BPE tokenization fertility and chars/token on Cebuano vs. English."""
    rows: list[dict[str, Any]] = []

    def measure(category: str, pair_idx: int, lang: str, text: str) -> dict[str, Any]:
        tokens = tokenizer.tokenize(text)
        n_chars = len(text)
        n_words = len(text.split())
        n_tokens = len(tokens)
        return {
            "category": category,
            "pair_id": pair_idx,
            "language": lang,
            "text": text,
            "char_count": n_chars,
            "word_count": n_words,
            "bpe_token_count": n_tokens,
            "tokens_per_word": round(n_tokens / max(1, n_words), 3),
            "chars_per_token": round(n_chars / max(1, n_tokens), 3),
            "subwords": format_bpe_tokens(tokens),
        }

    for idx, (ceb, eng) in enumerate(WORD_PAIRS, start=1):
        rows.append(measure("word", idx, "Cebuano", ceb))
        rows.append(measure("word", idx, "English", eng))

    for idx, (ceb, eng) in enumerate(SENTENCE_PAIRS, start=1):
        rows.append(measure("sentence", idx, "Cebuano", ceb))
        rows.append(measure("sentence", idx, "English", eng))

    csv_path = results_dir / "step0_tokenizer_comparison.csv"
    fieldnames = [
        "category",
        "pair_id",
        "language",
        "text",
        "char_count",
        "word_count",
        "bpe_token_count",
        "tokens_per_word",
        "chars_per_token",
        "subwords",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    def agg(category: str, lang: str) -> dict[str, float]:
        subset = [r for r in rows if r["category"] == category and r["language"] == lang]
        tot_words = sum(r["word_count"] for r in subset)
        tot_tokens = sum(r["bpe_token_count"] for r in subset)
        tot_chars = sum(r["char_count"] for r in subset)
        return {
            "total_words": tot_words,
            "total_tokens": tot_tokens,
            "total_chars": tot_chars,
            "avg_tokens_per_word": round(tot_tokens / tot_words, 3),
            "avg_chars_per_token": round(tot_chars / tot_tokens, 3),
        }

    stats = {
        "words": {"Cebuano": agg("word", "Cebuano"), "English": agg("word", "English")},
        "sentences": {"Cebuano": agg("sentence", "Cebuano"), "English": agg("sentence", "English")},
    }

    # Plot comparison chart
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), dpi=160)
    categories = ["12 Matched Words", "5 Parallel Sentences"]
    ceb_fert = [stats["words"]["Cebuano"]["avg_tokens_per_word"], stats["sentences"]["Cebuano"]["avg_tokens_per_word"]]
    eng_fert = [stats["words"]["English"]["avg_tokens_per_word"], stats["sentences"]["English"]["avg_tokens_per_word"]]
    ceb_cpt = [stats["words"]["Cebuano"]["avg_chars_per_token"], stats["sentences"]["Cebuano"]["avg_chars_per_token"]]
    eng_cpt = [stats["words"]["English"]["avg_chars_per_token"], stats["sentences"]["English"]["avg_chars_per_token"]]

    x = [0, 1]
    width = 0.34

    # Subplot 1: Token Fertility (lower is better)
    b1 = axes[0].bar([i - width / 2 for i in x], eng_fert, width, label="English", color="#2b6cb0")
    b2 = axes[0].bar([i + width / 2 for i in x], ceb_fert, width, label="Cebuano", color="#dd6b20")
    axes[0].set_title("Token Fertility (Tokens / Word)\nHigher = Worse Fragmentation", fontsize=11, fontweight="bold")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(categories)
    axes[0].set_ylabel("BPE Tokens per Word")
    axes[0].set_ylim(0, max(ceb_fert) * 1.25)
    axes[0].grid(axis="y", linestyle="--", alpha=0.4)
    axes[0].legend()
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width() / 2, h + 0.06, f"{h:.2f}", ha="center", va="bottom", fontsize=9)

    # Subplot 2: Characters per Token (higher is more efficient)
    b3 = axes[1].bar([i - width / 2 for i in x], eng_cpt, width, label="English", color="#2b6cb0")
    b4 = axes[1].bar([i + width / 2 for i in x], ceb_cpt, width, label="Cebuano", color="#dd6b20")
    axes[1].set_title("Compression Efficiency (Chars / Token)\nLower = Shorter Subword Pieces", fontsize=11, fontweight="bold")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(categories)
    axes[1].set_ylabel("Characters per BPE Token")
    axes[1].set_ylim(0, max(eng_cpt) * 1.25)
    axes[1].grid(axis="y", linestyle="--", alpha=0.4)
    axes[1].legend()
    for bar in list(b3) + list(b4):
        h = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width() / 2, h + 0.12, f"{h:.2f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle("Step 0 Baseline: DistilGPT2 BPE Tokenizer Fragmentation (English vs. Cebuano)", fontsize=12.5, fontweight="bold")
    fig.tight_layout()
    fig.savefig(results_dir / "step0_fragmentation_chart.png", bbox_inches="tight")
    plt.close(fig)

    return {"rows": rows, "summary": stats}


@torch.no_grad()
def evaluate_validation_perplexity(
    model: Any,
    tokenizer: Any,
    val_path: Path,
    device: torch.device,
    block_size: int = 256,
) -> dict[str, float]:
    """Compute cross-entropy loss and perplexity on non-overlapping validation blocks."""
    model.eval()
    text = val_path.read_text(encoding="utf-8")
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    total_blocks = len(token_ids) // block_size
    if total_blocks == 0:
        raise ValueError(f"Validation text in {val_path} has fewer than {block_size} tokens.")

    trimmed = token_ids[: total_blocks * block_size]
    blocks = torch.tensor(trimmed, dtype=torch.long).view(total_blocks, block_size)

    total_loss = 0.0
    batch_size = 16
    for start in range(0, total_blocks, batch_size):
        batch = blocks[start : start + batch_size].to(device)
        outputs = model(input_ids=batch, labels=batch)
        total_loss += outputs.loss.item() * batch.size(0)

    avg_loss = total_loss / total_blocks
    ppl = math.exp(avg_loss)
    return {
        "val_blocks": total_blocks,
        "block_size": block_size,
        "val_tokens_evaluated": total_blocks * block_size,
        "val_loss": round(avg_loss, 4),
        "val_perplexity": round(ppl, 2),
    }


@torch.no_grad()
def generate_samples(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    max_new_tokens: int = 60,
    seed: int = 42,
) -> list[dict[str, str]]:
    """Generate greedy and nucleus-sampled continuations on standardized prompts."""
    model.eval()
    outputs_list: list[dict[str, str]] = []

    for idx, item in enumerate(PROMPTS):
        prompt = item["prompt"]
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        prompt_len = inputs["input_ids"].shape[1]

        greedy_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        greedy_cont = tokenizer.decode(greedy_ids[0][prompt_len:], skip_special_tokens=True)

        set_seed(seed + idx)
        sample_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=0.92,
            temperature=0.8,
            pad_token_id=tokenizer.eos_token_id,
        )
        sample_cont = tokenizer.decode(sample_ids[0][prompt_len:], skip_special_tokens=True)

        outputs_list.append(
            {
                "id": item["id"],
                "register": item["register"],
                "prompt": prompt,
                "greedy_continuation": greedy_cont.strip(),
                "sampled_continuation": sample_cont.strip(),
            }
        )

    return outputs_list


def write_markdown_report(
    tok_results: dict[str, Any],
    gen_results: list[dict[str, str]],
    ppl_stats: dict[str, float],
    out_path: Path,
) -> None:
    """Save a formatted Step 0 diagnostic report."""
    s = tok_results["summary"]
    lines = [
        "# Step 0 Baseline Diagnostic: Stock `distilgpt2` on Cebuano",
        "",
        "## 1. Held-Out Cebuano Validation Perplexity (Before Fine-Tuning)",
        f"- **Validation blocks evaluated:** {ppl_stats['val_blocks']} blocks × {ppl_stats['block_size']} tokens ({ppl_stats['val_tokens_evaluated']:,} tokens)",
        f"- **Baseline Cross-Entropy Loss:** `{ppl_stats['val_loss']:.4f}`",
        f"- **Baseline Perplexity:** `{ppl_stats['val_perplexity']:.2f}`",
        "",
        "## 2. BPE Tokenizer Fragmentation (English vs. Cebuano)",
        "| Benchmark | Language | Total Words | Total BPE Tokens | Avg Tokens/Word (Fertility) | Avg Chars/Token |",
        "| :--- | :--- | ---: | ---: | ---: | ---: |",
        f"| 12 Matched Words | English | {s['words']['English']['total_words']} | {s['words']['English']['total_tokens']} | {s['words']['English']['avg_tokens_per_word']:.3f} | {s['words']['English']['avg_chars_per_token']:.3f} |",
        f"| 12 Matched Words | Cebuano | {s['words']['Cebuano']['total_words']} | {s['words']['Cebuano']['total_tokens']} | {s['words']['Cebuano']['avg_tokens_per_word']:.3f} | {s['words']['Cebuano']['avg_chars_per_token']:.3f} |",
        f"| 5 Parallel Sentences | English | {s['sentences']['English']['total_words']} | {s['sentences']['English']['total_tokens']} | {s['sentences']['English']['avg_tokens_per_word']:.3f} | {s['sentences']['English']['avg_chars_per_token']:.3f} |",
        f"| 5 Parallel Sentences | Cebuano | {s['sentences']['Cebuano']['total_words']} | {s['sentences']['Cebuano']['total_tokens']} | {s['sentences']['Cebuano']['avg_tokens_per_word']:.3f} | {s['sentences']['Cebuano']['avg_chars_per_token']:.3f} |",
        "",
        "### Selected Word Fragmentation Examples",
        "| Cebuano Word | Subword Pieces | Tokens | Length-Matched English | Subword Pieces | Tokens |",
        "| :--- | :--- | ---: | :--- | :--- | ---: |",
    ]

    word_rows = [r for r in tok_results["rows"] if r["category"] == "word"]
    for i in range(0, len(word_rows), 2):
        ceb_r, eng_r = word_rows[i], word_rows[i + 1]
        lines.append(
            f"| `{ceb_r['text']}` | `{ceb_r['subwords']}` | {ceb_r['bpe_token_count']} | "
            f"`{eng_r['text']}` | `{eng_r['subwords']}` | {eng_r['bpe_token_count']} |"
        )

    lines.extend(["", "## 3. Untouched `distilgpt2` Generation Quality (English Drift & Subword Breakdown)", ""])
    for item in gen_results:
        lines.extend(
            [
                f"### {item['id']} ({item['register']})",
                f"- **Prompt:** `{item['prompt']}`",
                f"- **Greedy (`do_sample=False`):** {item['greedy_continuation']}",
                f"- **Sampled (`top_p=0.92, T=0.8`):** {item['sampled_continuation']}",
                "",
            ]
        )

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data"
    results_dir = project_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    val_path = data_dir / "cebuano_corpus_val.txt"
    if not val_path.exists():
        raise FileNotFoundError(f"Missing {val_path}. Run scripts/prepare_corpus.py first.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Step 0 baseline diagnostic on device: {device}")

    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained("distilgpt2").to(device)

    print("1/3 Analyzing GPT-2 BPE tokenizer fragmentation...")
    tok_results = analyze_tokenizer(tokenizer, results_dir)

    print("2/3 Generating untouched distilgpt2 continuations on 5 Cebuano prompts...")
    gen_results = generate_samples(model, tokenizer, device=device, max_new_tokens=60, seed=42)

    print("3/3 Computing baseline perplexity on held-out Cebuano validation split...")
    ppl_stats = evaluate_validation_perplexity(model, tokenizer, val_path, device=device, block_size=256)

    write_markdown_report(tok_results, gen_results, ppl_stats, results_dir / "step0_baseline_generations.md")

    summary = {
        "model": "distilgpt2",
        "device": str(device),
        "tokenizer_summary": tok_results["summary"],
        "validation_perplexity": ppl_stats,
        "generations": gen_results,
    }
    (results_dir / "step0_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("\n=== Step 0 Baseline Summary ===")
    print(f"Word Fertility  — English: {tok_results['summary']['words']['English']['avg_tokens_per_word']:.2f} vs. Cebuano: {tok_results['summary']['words']['Cebuano']['avg_tokens_per_word']:.2f} tokens/word")
    print(f"Sent Fertility  — English: {tok_results['summary']['sentences']['English']['avg_tokens_per_word']:.2f} vs. Cebuano: {tok_results['summary']['sentences']['Cebuano']['avg_tokens_per_word']:.2f} tokens/word")
    print(f"Baseline Val Loss: {ppl_stats['val_loss']:.4f} | Baseline Perplexity: {ppl_stats['val_perplexity']:.2f}")


if __name__ == "__main__":
    main()
