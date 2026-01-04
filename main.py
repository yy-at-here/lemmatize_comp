#!/usr/bin/env python3
"""
Lemmatize Comparison Tool

A tool to compare two text files and extract words that exist in the target file
but not in the reference file.
"""

import argparse
import csv
from collections import Counter
from pathlib import Path
import re
import unicodedata

import spacy
from spacy.language import Language

def normalize_text(text: str) -> str:
    """Simple normalization before feeding to spaCy.

    - Convert fullwidth commas/periods to halfwidth
    - Normalize compatible characters like fullwidth alphanumerics, circled numbers, Roman numerals (NFKC)
    - Replace certain symbols (bullets, etc.) with spaces for easier tokenization
    """
    normalized = unicodedata.normalize("NFKC", text)

    # Japanese punctuation is not ASCII-fied by NFKC, so explicitly convert
    # Expected behavior: fullwidth comma to halfwidth comma, fullwidth period to ". "
    normalized = normalized.replace("、", ", ")
    normalized = normalized.replace("，", ", ")
    normalized = normalized.replace("。", ". ")
    normalized = normalized.replace("．", ". ")

    # Replace bullets, arrows, etc. that tend to stick to words with spaces
    normalized = normalized.translate(
        str.maketrans(
            {
                "・": " ",
                "→": " ",
                "⇒": " ",
                "★": " ",
                "☆": " ",
                "※": " ",
            }
        )
    )

    # Convert wave dashes (e.g., in ①～④) to ASCII equivalents
    normalized = normalized.replace("～", "~")
    normalized = normalized.replace("〜", "~")

    # Add spaces around brackets to split cases where they stick to words (e.g., "feet),the")
    normalized = re.sub(r"([\[\]\(\){}])", r" \1 ", normalized)

    # Separate numbered list notation from educational materials (e.g., "3.many" -> "3. many", ".There" -> ". There")
    # This allows numbers to be dropped later while preserving the English words.
    normalized = re.sub(r"(\d)\.(?=[A-Za-z])", r"\1. ", normalized)
    normalized = re.sub(r"(^|\s)\.(?=[A-Za-z])", r"\1. ", normalized)

    # Light whitespace normalization (preserve line breaks)
    normalized = re.sub(r"[ \t]+", " ", normalized)
    normalized = re.sub(r" *\n *", "\n", normalized)

    return normalized


_CJK_RE = re.compile(
    r"[\u3040-\u309F\u30A0-\u30FF\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF]"
)


def _contains_cjk(text: str) -> bool:
    return bool(_CJK_RE.search(text))


_CURRENCY_AMOUNT_RE = re.compile(r"^(?:[$£€])\d")
# _SHORT_PREFIX_HYPHEN_NUM_RE = re.compile(r"^[a-z]{1,2}-\d+$")


def _skip_reason_for_lemma(lemma: str) -> str | None:
    """Return the reason for exclusion (None if not excluded)."""
    if not lemma:
        return "empty"

    if _contains_cjk(lemma):
        return "contains_cjk"

    # Exclude abbreviations/references containing "." (e.g., U.S., e.g., l.10, p.m.)
    if "." in lemma:
        return "contains_dot"

    # Exclude time notation (e.g., 9:30)
    if ":" in lemma:
        return "contains_colon"

    # Exclude fragments with remaining brackets (e.g., "(2", "[your")
    if any(ch in lemma for ch in "()[]{}"): 
        return "contains_brackets"

    # Exclude currency+number fragments (e.g., $21, £15)
    if _CURRENCY_AMOUNT_RE.match(lemma):
        return "currency_amount"

    # Exclude single letters as they tend to be noise (a, b, etc.)
    # However, keep the article "a"
    if len(lemma) == 1 and lemma != "a":
        return "single_letter"

    # Exclude if no alphabetic or numeric characters at all
    if not any(c.isalpha() or c.isdigit() for c in lemma):
        return "no_alnum"

    return None


def save_skipped_to_csv(skipped: Counter, filepath: Path) -> None:
    """Save skipped tokens to CSV (sorted by frequency descending).

    Keys in skipped are tuples of (token_text, lemma, reason).
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["token", "lemma", "reason", "frequency"])
        for (token_text, lemma, reason), freq in skipped.most_common():
            writer.writerow([token_text, lemma, reason, freq])


def load_text(filepath: str) -> str:
    """Load text file with UTF-8 encoding"""
    with open(filepath, "r", encoding="utf-8") as f:
        return normalize_text(f.read())


def extract_lemmas(text: str, nlp: Language) -> tuple[Counter, Counter]:
    """
    Analyze text with spaCy and count lemmatized word frequencies

    - Exclude punctuation and symbols
    - Exclude numeric-only tokens
    - Convert to lowercase
    - Include proper nouns
    """
    doc = nlp(text)
    words = []
    skipped: Counter = Counter()

    for token in doc:
        # Exclude punctuation, symbols, and whitespace
        if token.is_punct or token.is_space:
            continue

        # Exclude numeric-only tokens
        if token.like_num or token.text.isdigit():
            continue

        # Lemmatize and convert to lowercase
        lemma = token.lemma_.lower()

        # Exclude tokens containing Japanese (drop whether before or after lemmatization)
        if _contains_cjk(token.text) or _contains_cjk(lemma):
            skipped[(token.text, lemma, "contains_cjk")] += 1
            continue

        reason = _skip_reason_for_lemma(lemma)
        if reason is not None:
            skipped[(token.text, lemma, reason)] += 1
            continue

        # Skip if empty or symbols only
        if not lemma or not any(c.isalnum() for c in lemma):
            continue

        words.append(lemma)

    return Counter(words), skipped


def save_to_csv(counter: Counter, filepath: Path) -> None:
    """Save Counter to CSV file (sorted by frequency descending)"""
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Sort by frequency descending
    sorted_items = counter.most_common()

    with open(filepath, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["word", "frequency"])
        for word, freq in sorted_items:
            writer.writerow([word, freq])


def find_difference(target_counter: Counter, ref_counter: Counter) -> Counter:
    """Extract words that do not exist in reference"""
    diff = Counter()
    for word, freq in target_counter.items():
        if word not in ref_counter:
            diff[word] = freq
    return diff


def main():
    parser = argparse.ArgumentParser(
        description="Compare two text files and extract words not present in reference"
    )
    parser.add_argument(
        "--target",
        required=True,
        help="Text file to compare",
    )
    parser.add_argument(
        "--ref",
        required=True,
        help="Reference text file",
    )
    parser.add_argument(
        "--output",
        default="./output",
        help="Output directory (default: ./output)",
    )

    args = parser.parse_args()

    # Output directory path
    output_dir = Path(args.output)

    print("Loading spaCy model...")
    nlp = spacy.load("en_core_web_sm")

    # Process target file
    print(f"Processing: {args.target}")
    target_text = load_text(args.target)
    target_counter, target_skipped = extract_lemmas(target_text, nlp)

    # Process reference file
    print(f"Processing: {args.ref}")
    ref_text = load_text(args.ref)
    ref_counter, ref_skipped = extract_lemmas(ref_text, nlp)

    # CSV output
    target_csv = output_dir / "target_lemmatized.csv"
    ref_csv = output_dir / "reference_lemmatized.csv"
    diff_csv = output_dir / "not_in_reference.csv"
    target_skipped_csv = output_dir / "target_skipped.csv"
    ref_skipped_csv = output_dir / "reference_skipped.csv"

    print(f"Writing: {target_csv}")
    save_to_csv(target_counter, target_csv)

    print(f"Writing: {ref_csv}")
    save_to_csv(ref_counter, ref_csv)

    print(f"Writing: {target_skipped_csv}")
    save_skipped_to_csv(target_skipped, target_skipped_csv)

    print(f"Writing: {ref_skipped_csv}")
    save_skipped_to_csv(ref_skipped, ref_skipped_csv)

    # Calculate difference
    diff_counter = find_difference(target_counter, ref_counter)

    print(f"Writing: {diff_csv}")
    save_to_csv(diff_counter, diff_csv)

    # Result summary
    print("\n--- Result Summary ---")
    print(f"Target unique words: {len(target_counter)}")
    print(f"Target skipped tokens: {len(target_skipped)}")
    print(f"Reference unique words: {len(ref_counter)}")
    print(f"Reference skipped tokens: {len(ref_skipped)}")
    print(f"Words not in reference: {len(diff_counter)}")



if __name__ == "__main__":
    main()
