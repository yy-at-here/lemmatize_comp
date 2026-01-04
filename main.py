#!/usr/bin/env python3
"""
Lemmatize Comparison Tool

2つのテキストファイルを比較し、reference に存在しない単語を抽出するツール。
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
    """spaCy投入前の簡易正規化。

    - 全角カンマ/句点を半角に寄せる
    - 全角英数字/丸数字/ローマ数字などの互換文字を正規化（NFKC）
    - 一部の記号（箇条書きなど）を空白に寄せて分割されやすくする
    """
    normalized = unicodedata.normalize("NFKC", text)

    # 日本語系の句読点はNFKCでもASCII化されないため、明示的に寄せる
    # 期待挙動: 全角カンマは半角カンマに、全角ピリオドは ". " にする
    normalized = normalized.replace("、", ", ")
    normalized = normalized.replace("，", ", ")
    normalized = normalized.replace("。", ". ")
    normalized = normalized.replace("．", ". ")

    # 箇条書き/矢印など、単語の前にくっつきやすい記号は空白に寄せる
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

    # 波ダッシュ系（①～④ など）をASCII側に寄せる
    normalized = normalized.replace("～", "~")
    normalized = normalized.replace("〜", "~")

    # 括弧系が単語にくっつくケース（例: "feet),the"）を分割しやすくする
    normalized = re.sub(r"([\[\]\(\){}])", r" \1 ", normalized)

    # 教材由来の連番表記などを分離（例: "3.many" -> "3. many", ".There" -> ". There"）
    # これにより、数字側は後段で落ち、英単語側は拾えるようになる。
    normalized = re.sub(r"(\d)\.(?=[A-Za-z])", r"\1. ", normalized)
    normalized = re.sub(r"(^|\s)\.(?=[A-Za-z])", r"\1. ", normalized)

    # 空白を軽く正規化（改行は維持）
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


def _should_skip_lemma(lemma: str) -> bool:
    """語彙として数えないトークン（教材のメタ情報・記号混入など）を除外する。"""
    if not lemma:
        return True

    # 日本語（ひらがな/カタカナ/漢字）を除外
    if _contains_cjk(lemma):
        return True

    # U.S / e.g. / l.10 / p.m. など「. を含む」略語・参照記法は除外
    if "." in lemma:
        return True

    # 時刻など（9:30 など）は除外
    if ":" in lemma:
        return True

    # 括弧・角括弧が残る断片は除外（例: "(2", "[your"）
    if any(ch in lemma for ch in "()[]{}"):
        return True

    # 通貨+数字の断片は除外（$21, £15 など）
    if _CURRENCY_AMOUNT_RE.match(lemma):
        return True

    # アルファベット1文字だけはノイズになりやすいので除外（a, b 等）
    # ただし冠詞の a は残す
    if len(lemma) == 1 and lemma != "a":
        return True

    # 英字/数字がまったく含まれないものは除外
    if not any(c.isalpha() or c.isdigit() for c in lemma):
        return True

    return False


def _skip_reason_for_lemma(lemma: str) -> str | None:
    """除外理由を返す（除外しない場合は None）。"""
    if not lemma:
        return "empty"

    if _contains_cjk(lemma):
        return "contains_cjk"

    # U.S / e.g. / l.10 / p.m. など「. を含む」略語・参照記法は除外
    if "." in lemma:
        return "contains_dot"

    # 時刻など（9:30 など）は除外
    if ":" in lemma:
        return "contains_colon"

    # 括弧・角括弧が残る断片は除外（例: "(2", "[your"）
    if any(ch in lemma for ch in "()[]{}"): 
        return "contains_brackets"

    # 通貨+数字の断片は除外（$21, £15 など）
    if _CURRENCY_AMOUNT_RE.match(lemma):
        return "currency_amount"

    # アルファベット1文字だけはノイズになりやすいので除外（a, b 等）
    # ただし冠詞の a は残す
    if len(lemma) == 1 and lemma != "a":
        return "single_letter"

    # 英字/数字がまったく含まれないものは除外
    if not any(c.isalpha() or c.isdigit() for c in lemma):
        return "no_alnum"

    return None


def save_skipped_to_csv(skipped: Counter, filepath: Path) -> None:
    """除外されたトークンを CSV に保存（頻度の降順）。

    skipped のキーは (token_text, lemma, reason) のタプル。
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["token", "lemma", "reason", "frequency"])
        for (token_text, lemma, reason), freq in skipped.most_common():
            writer.writerow([token_text, lemma, reason, freq])


def load_text(filepath: str) -> str:
    """テキストファイルを UTF-8 で読み込む"""
    with open(filepath, "r", encoding="utf-8") as f:
        return normalize_text(f.read())


def extract_lemmas(text: str, nlp: Language) -> tuple[Counter, Counter]:
    """
    spaCy でテキストを解析し、lemmatize した単語の頻度をカウントする

    - 句読点・記号を除外
    - 数字のみのトークンを除外
    - 小文字に変換
    - 固有名詞は含める
    """
    doc = nlp(text)
    words = []
    skipped: Counter = Counter()

    for token in doc:
        # 句読点・記号・空白を除外
        if token.is_punct or token.is_space:
            continue

        # 数字のみのトークンを除外
        if token.like_num or token.text.isdigit():
            continue

        # lemmatize して小文字に変換
        lemma = token.lemma_.lower()

        # 日本語を含むトークンは除外（lemmatize 前後どちらでも落とす）
        if _contains_cjk(token.text) or _contains_cjk(lemma):
            skipped[(token.text, lemma, "contains_cjk")] += 1
            continue

        reason = _skip_reason_for_lemma(lemma)
        if reason is not None:
            skipped[(token.text, lemma, reason)] += 1
            continue

        # 空文字や記号のみの場合はスキップ
        if not lemma or not any(c.isalnum() for c in lemma):
            continue

        words.append(lemma)

    return Counter(words), skipped


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
    object_counter, object_skipped = extract_lemmas(object_text, nlp)

    # reference ファイルの処理
    print(f"処理中: {args.ref}")
    ref_text = load_text(args.ref)
    ref_counter, ref_skipped = extract_lemmas(ref_text, nlp)

    # CSV 出力
    object_csv = output_dir / "object_lemmatized.csv"
    ref_csv = output_dir / "reference_lemmatized.csv"
    diff_csv = output_dir / "not_in_reference.csv"
    object_skipped_csv = output_dir / "object_skipped.csv"
    ref_skipped_csv = output_dir / "reference_skipped.csv"

    print(f"出力中: {object_csv}")
    save_to_csv(object_counter, object_csv)

    print(f"出力中: {ref_csv}")
    save_to_csv(ref_counter, ref_csv)

    print(f"出力中: {object_skipped_csv}")
    save_skipped_to_csv(object_skipped, object_skipped_csv)

    print(f"出力中: {ref_skipped_csv}")
    save_skipped_to_csv(ref_skipped, ref_skipped_csv)

    # 差分を計算
    diff_counter = find_difference(object_counter, ref_counter)

    print(f"出力中: {diff_csv}")
    save_to_csv(diff_counter, diff_csv)

    # 結果サマリー
    print("\n--- 結果サマリー ---")
    print(f"object の単語数（ユニーク）: {len(object_counter)}")
    print(f"object でスキップされた単語数: {len(object_skipped)}")
    print(f"reference の単語数（ユニーク）: {len(ref_counter)}")
    print(f"reference でスキップされた単語数: {len(ref_skipped)}")
    print(f"reference にない単語数: {len(diff_counter)}")



if __name__ == "__main__":
    main()
