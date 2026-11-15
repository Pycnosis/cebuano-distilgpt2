"""Step 1A: Load, clean, reconstruct, and combine Cebuano corpora.

Loads two Hugging Face datasets:
1. josephimperial/CebuaNER (news-register tokens with CoNLL NER tags)
2. filbench/cebuano-readability (literary & pedagogical passages)

Outputs:
- data/cebuano_corpus.txt       (full combined corpus)
- data/cebuano_corpus_train.txt (90% stratified train split, seed=42)
- data/cebuano_corpus_val.txt   (10% stratified validation split, seed=42)
- data/corpus_stats.json        (document, sentence, word, char, and BPE token counts)
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any

from datasets import load_dataset
from transformers import AutoTokenizer

ATTACH_LEFT = {".", ",", "!", "?", ":", ";", ")", "]", "}", "%", "...", "”", "’", "»"}
ATTACH_RIGHT = {"(", "[", "{", "“", "‘", "«", "$", "₱"}
QUOTE_CHARS = {'"', "'"}


def detokenize_tokens(tokens: list[str]) -> str:
    """Reconstruct a clean plain-text sentence from CoNLL tokens."""
    if not tokens:
        return ""

    parts: list[str] = []
    in_double_quote = False
    in_single_quote = False

    for i, tok in enumerate(tokens):
        if tok == '"':
            if not in_double_quote:
                if parts:
                    parts.append(" ")
                parts.append(tok)
                in_double_quote = True
            else:
                parts.append(tok)
                in_double_quote = False
            continue

        if tok == "'":
            if not in_single_quote:
                if parts:
                    parts.append(" ")
                parts.append(tok)
                in_single_quote = True
            else:
                parts.append(tok)
                in_single_quote = False
            continue

        if not parts:
            parts.append(tok)
            continue

        prev_char = parts[-1][-1]
        if tok in ATTACH_LEFT or prev_char in ATTACH_RIGHT:
            parts.append(tok)
        elif prev_char == '"' and in_double_quote:
            parts.append(tok)
        elif prev_char == "'" and in_single_quote:
            parts.append(tok)
        elif tok in {"''", "``"}:
            if tok == "``":
                parts.append(' "')
                in_double_quote = True
            else:
                parts.append('"')
                in_double_quote = False
        else:
            parts.append(" " + tok)

    text = "".join(parts)
    # Collapse stray spaces before standard punctuation if any token merged weirdly
    text = re.sub(r"\s+([.,!?;:)\]}%])", r"\1", text)
    text = re.sub(r"([(\[{])\s+", r"\1", text)
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text


def reconstruct_cebuaner_docs(dataset_dict: Any) -> tuple[list[str], int]:
    """Parse CoNLL-style CebuaNER blocks into plain-text news articles/passages."""
    docs: list[str] = []
    total_tokens = 0
    current_tokens: list[str] = []

    def flush_block() -> None:
        if current_tokens:
            text = detokenize_tokens(current_tokens)
            if text:
                docs.append(text)
            current_tokens.clear()

    for split_name in dataset_dict.keys():
        for row in dataset_dict[split_name]:
            raw_line = (row.get("text") or "").strip()
            if not raw_line or raw_line.startswith("-DOCSTART-"):
                flush_block()
                continue

            if " -X-" in raw_line:
                token = raw_line.rsplit(" -X-", 1)[0].strip()
            else:
                token = raw_line.split()[0]

            if token:
                current_tokens.append(token)
                total_tokens += 1

    flush_block()
    return docs, total_tokens


def clean_readability_passage(raw_text: str) -> str:
    """Normalize PDF/story line wraps and excessive blank lines in readability prose."""
    normalized = raw_text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        return ""

    # Paragraph/page breaks in Bloom/DepEd texts use >=3 newlines, while line wraps use 1-2 newlines.
    raw_blocks = re.split(r"\n{3,}", normalized)
    cleaned_blocks: list[str] = []
    for block in raw_blocks:
        collapsed = re.sub(r"\s*\n+\s*", " ", block)
        collapsed = re.sub(r"\s{2,}", " ", collapsed).strip()
        if collapsed:
            cleaned_blocks.append(collapsed)

    return "\n".join(cleaned_blocks)


def load_readability_docs(dataset_dict: Any) -> list[str]:
    """Extract cleaned plain-prose passages from filbench/cebuano-readability."""
    docs: list[str] = []
    for split_name in dataset_dict.keys():
        for row in dataset_dict[split_name]:
            cleaned = clean_readability_passage(row.get("text") or "")
            if cleaned:
                docs.append(cleaned)
    return docs


def compute_text_stats(docs: list[str], tokenizer: Any) -> dict[str, int]:
    """Compute document, line, word, character, and distilgpt2 BPE token counts."""
    joined = "\n\n".join(docs)
    lines = [ln for doc in docs for ln in doc.splitlines() if ln.strip()]
    words = joined.split()
    # Tokenize per document in batches to avoid single-sequence length warnings
    total_bpe_tokens = 0
    for i in range(0, len(docs), 128):
        batch = docs[i : i + 128]
        encoded = tokenizer(batch, add_special_tokens=False)
        total_bpe_tokens += sum(len(ids) for ids in encoded["input_ids"])

    return {
        "documents": len(docs),
        "lines_or_sentences": len(lines),
        "words": len(words),
        "characters": len(joined),
        "distilgpt2_bpe_tokens": total_bpe_tokens,
    }


def stratified_split(
    ner_docs: list[str],
    read_docs: list[str],
    val_ratio: float = 0.10,
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    """Create a deterministic 90/10 train/validation split across both registers."""
    rng = random.Random(seed)

    ner_shuffled = list(ner_docs)
    read_shuffled = list(read_docs)
    rng.shuffle(ner_shuffled)
    rng.shuffle(read_shuffled)

    ner_val_n = max(1, int(round(len(ner_shuffled) * val_ratio)))
    read_val_n = max(1, int(round(len(read_shuffled) * val_ratio)))

    val_docs = ner_shuffled[:ner_val_n] + read_shuffled[:read_val_n]
    train_docs = ner_shuffled[ner_val_n:] + read_shuffled[read_val_n:]

    rng.shuffle(train_docs)
    rng.shuffle(val_docs)
    return train_docs, val_docs


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    print("Loading josephimperial/CebuaNER from Hugging Face...")
    cebuaner_ds = load_dataset("josephimperial/CebuaNER")
    cebuaner_docs, cebuaner_raw_tokens = reconstruct_cebuaner_docs(cebuaner_ds)
    print(f"  Reconstructed {len(cebuaner_docs)} articles/blocks ({cebuaner_raw_tokens:,} raw CoNLL tokens).")

    print("Loading filbench/cebuano-readability from Hugging Face...")
    readability_ds = load_dataset("filbench/cebuano-readability")
    readability_docs = load_readability_docs(readability_ds)
    print(f"  Loaded {len(readability_docs)} literary/pedagogical passages.")

    combined_docs = cebuaner_docs + readability_docs
    train_docs, val_docs = stratified_split(cebuaner_docs, readability_docs, val_ratio=0.10, seed=42)

    corpus_path = data_dir / "cebuano_corpus.txt"
    train_path = data_dir / "cebuano_corpus_train.txt"
    val_path = data_dir / "cebuano_corpus_val.txt"

    corpus_path.write_text("\n\n".join(combined_docs) + "\n", encoding="utf-8")
    train_path.write_text("\n\n".join(train_docs) + "\n", encoding="utf-8")
    val_path.write_text("\n\n".join(val_docs) + "\n", encoding="utf-8")

    print("Computing corpus & BPE tokenization statistics...")
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    tokenizer.model_max_length = 1_000_000  # For full-document token counting without length warnings

    stats = {
        "sources": {
            "cebuaner": {
                "dataset": "josephimperial/CebuaNER",
                "license": "CC BY-NC 4.0",
                "register": "News articles",
                "raw_conll_tokens": cebuaner_raw_tokens,
                **compute_text_stats(cebuaner_docs, tokenizer),
            },
            "cebuano_readability": {
                "dataset": "filbench/cebuano-readability",
                "license": "CC BY 4.0",
                "register": "Literary / pedagogical prose",
                **compute_text_stats(readability_docs, tokenizer),
            },
        },
        "combined": compute_text_stats(combined_docs, tokenizer),
        "splits": {
            "seed": 42,
            "val_ratio": 0.10,
            "train": compute_text_stats(train_docs, tokenizer),
            "validation": compute_text_stats(val_docs, tokenizer),
        },
    }

    stats_path = data_dir / "corpus_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("\n=== Corpus Preparation Summary ===")
    print(f"Combined corpus saved to: {corpus_path}")
    print(f"Total documents: {stats['combined']['documents']:,}")
    print(f"Total words:     {stats['combined']['words']:,}")
    print(f"Total BPE tokens:{stats['combined']['distilgpt2_bpe_tokens']:,}")
    print(f"Train words:     {stats['splits']['train']['words']:,} | Val words: {stats['splits']['validation']['words']:,}")


if __name__ == "__main__":
    main()
