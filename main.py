#!/usr/bin/env python3
"""
Lemmatize Comparison Tool

2つのテキストファイルを比較し、reference に存在しない単語を抽出するツール。
"""

import argparse
import csv
from collections import Counter
from pathlib import Path

import spacy


def load_text(filepath: str) -> str:
    """テキストファイルを UTF-8 で読み込む"""
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def extract_lemmas(text: str, nlp) -> Counter:
    """
    spaCy でテキストを解析し、lemmatize した単語の頻度をカウントする

    - 句読点・記号を除外
    - 数字のみのトークンを除外
    - 小文字に変換
    - 固有名詞は含める
    """
    doc = nlp(text)
    words = []

    for token in doc:
        # 句読点・記号・空白を除外
        if token.is_punct or token.is_space:
            continue

        # 数字のみのトークンを除外
        if token.like_num or token.text.isdigit():
            continue

        # lemmatize して小文字に変換
        lemma = token.lemma_.lower()

        # 空文字や記号のみの場合はスキップ
        if not lemma or not any(c.isalnum() for c in lemma):
            continue

        words.append(lemma)

    return Counter(words)


def save_to_csv(counter: Counter, filepath: Path) -> None:
    """Counter を CSV ファイルに保存（頻度の降順）"""
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # 頻度の降順でソート
    sorted_items = counter.most_common()

    with open(filepath, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["word", "frequency"])
        for word, freq in sorted_items:
            writer.writerow([word, freq])


def find_difference(object_counter: Counter, ref_counter: Counter) -> Counter:
    """reference に存在しない単語を抽出"""
    diff = Counter()
    for word, freq in object_counter.items():
        if word not in ref_counter:
            diff[word] = freq
    return diff


def main():
    parser = argparse.ArgumentParser(
        description="2つのテキストファイルを比較し、reference に存在しない単語を抽出する"
    )
    parser.add_argument(
        "--object",
        required=True,
        help="比較対象のテキストファイル",
    )
    parser.add_argument(
        "--ref",
        required=True,
        help="基準となるテキストファイル",
    )
    parser.add_argument(
        "--output",
        default="./output",
        help="出力ディレクトリ（デフォルト: ./output）",
    )

    args = parser.parse_args()

    # 出力ディレクトリのパス
    output_dir = Path(args.output)

    print("spaCy モデルを読み込み中...")
    nlp = spacy.load("en_core_web_sm")

    # object ファイルの処理
    print(f"処理中: {args.object}")
    object_text = load_text(args.object)
    object_counter = extract_lemmas(object_text, nlp)

    # reference ファイルの処理
    print(f"処理中: {args.ref}")
    ref_text = load_text(args.ref)
    ref_counter = extract_lemmas(ref_text, nlp)

    # CSV 出力
    object_csv = output_dir / "object_lemmatized.csv"
    ref_csv = output_dir / "reference_lemmatized.csv"
    diff_csv = output_dir / "not_in_reference.csv"

    print(f"出力中: {object_csv}")
    save_to_csv(object_counter, object_csv)

    print(f"出力中: {ref_csv}")
    save_to_csv(ref_counter, ref_csv)

    # 差分を計算
    diff_counter = find_difference(object_counter, ref_counter)

    print(f"出力中: {diff_csv}")
    save_to_csv(diff_counter, diff_csv)

    # 結果サマリー
    print("\n--- 結果サマリー ---")
    print(f"object の単語数（ユニーク）: {len(object_counter)}")
    print(f"reference の単語数（ユニーク）: {len(ref_counter)}")
    print(f"reference にない単語数: {len(diff_counter)}")


if __name__ == "__main__":
    main()
