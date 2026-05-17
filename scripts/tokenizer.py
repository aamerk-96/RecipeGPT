from __future__ import annotations

import ast
from pathlib import Path
from typing import Iterable

import pandas as pd
import sentencepiece as spm


SPECIAL_TOKENS = [
    "<|startofrecipe|>",
    "<|endofrecipe|>",
    "<|title|>",
    "<|ingredients|>",
    "<|directions|>",
]


def process_row(row: pd.Series) -> str:
    title = row["title"]
    ingredients = ast.literal_eval(row["ingredients"])
    directions = ast.literal_eval(row["directions"])

    ingredients_str = " | ".join(ingredients)
    directions_str = " | ".join(directions)

    return (
        f"<|startofrecipe|> <|title|> {title} "
        f"<|ingredients|> {ingredients_str} "
        f"<|directions|> {directions_str} <|endofrecipe|>"
    )


def build_processed_corpus(input_csv: Path, output_txt: Path) -> None:
    df = pd.read_csv(input_csv)
    df["processed"] = df.apply(process_row, axis=1)

    output_txt.parent.mkdir(parents=True, exist_ok=True)
    with output_txt.open("w", encoding="utf-8") as f:
        for recipe in df["processed"]:
            f.write(recipe + "\n")


def preview_processed_corpus(processed_txt: Path, n: int = 5) -> None:
    with processed_txt.open("r", encoding="utf-8") as f:
        for _ in range(n):
            line = f.readline()
            if not line:
                break
            print(line.rstrip("\n"))


def train_sentencepiece(processed_txt: Path, model_prefix: Path, vocab_size: int = 32000) -> None:
    spm.SentencePieceTrainer.train(
        input=str(processed_txt),
        model_prefix=str(model_prefix),
        vocab_size=vocab_size,
        model_type="bpe",
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        user_defined_symbols=SPECIAL_TOKENS,
        character_coverage=1.0,
        input_sentence_size=1_000_000,
        shuffle_input_sentence=True,
    )


def validate_tokenizer(model_path: Path) -> None:
    sp = spm.SentencePieceProcessor()
    sp.load(str(model_path))

    print(f"Vocab size: {sp.get_piece_size()}")

    for token in SPECIAL_TOKENS:
        token_id = sp.piece_to_id(token)
        print(f"{token} -> ID {token_id}")

    test = "<|title|> Garlic Butter Pasta <|ingredients|> 2 cloves garlic | 3 Tbsp. butter"
    tokens = sp.encode(test, out_type=str)
    ids = sp.encode(test)
    print(f"\nTokens: {tokens}")
    print(f"IDs: {ids}")
    print(f"Decoded: {sp.decode(ids)}")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data"

    input_csv = data_dir / "full_dataset.csv"
    processed_txt = data_dir / "processed_recipes.txt"
    model_prefix = data_dir / "recipe_tokenizer"
    model_path = data_dir / "recipe_tokenizer.model"

    build_processed_corpus(input_csv, processed_txt)
    preview_processed_corpus(processed_txt, n=5)
    train_sentencepiece(processed_txt, model_prefix, vocab_size=32000)
    validate_tokenizer(model_path)


if __name__ == "__main__":
    main()