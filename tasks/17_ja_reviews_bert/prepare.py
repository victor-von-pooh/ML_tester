import argparse
from pathlib import Path
import sys

import numpy as np

from ml_eda import export_dataset, load_data, load_dataset, profile

# このタスクのディレクトリと, 元のデータと書き出す先
TASK_DIR = Path(__file__).resolve().parent
SOURCE = TASK_DIR.parents[1] / "data" / "reviews_ja.csv"
UNLABELED = TASK_DIR.parents[1] / "data" / "reviews_ja_unlabeled.csv"


def main() -> None:
    """
    レビューを文のデータとして分析し, ML-Fathom で作ったレシピで書き出し,
    ラベルの無いレビューを同じ語彙で番号の並びにする関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # レシピと書き出す先 (区切り方を変えたレシピと比べるため)
    parser = argparse.ArgumentParser(description="レビューの分析と書き出し")
    parser.add_argument(
        "--recipe", default="recipe.yaml", help="レシピ (タスクのディレクトリから)"
    )
    parser.add_argument(
        "--out", default="outputs/dataset", help="データセットの書き出し先"
    )
    parser.add_argument(
        "--corpus", default="outputs/corpus.npy",
        help="ラベルの無いレビューの番号の並びの書き出し先"
    )
    args = parser.parse_args()
    recipe = TASK_DIR / args.recipe
    dataset_dir = TASK_DIR / args.out
    corpus_path = TASK_DIR / args.corpus

    # 元のデータが無ければ作り方の案内と終了
    if not SOURCE.is_file() or not UNLABELED.is_file():
        print("data/reviews_ja*.csv がありません. 先に make_data.py を"
              "実行してください")
        sys.exit(1)

    # 表として読んだ場合と, 文の列を文のデータとして読んだ場合の気付き
    for options in ({"kind": "table"}, {"text_column": "review"}):
        data = load_data(SOURCE, **options)
        result = profile(data, target="label")
        print(f"読み方 {options}: 種類 {data.kind}, 行 {len(data.frame)}")
        for item in result["findings"]:
            print(f"  [{item['severity']}] {item['code']}: "
                  f"{item['message'][:70]}")

    # 文の列として読んだデータの, レシピでの書き出し (語彙は学習用の文から)
    data = load_data(SOURCE, text_column="review")
    info = export_dataset(
        data, recipe, dataset_dir, name="reviews_ja",
        overwrite=True
    )
    shapes = {name: item["shape"] for name, item in info["splits"].items()}
    print(f"書き出し: {dataset_dir}, 形 {shapes}, クラス {info['classes']}")

    # ラベルの無いレビューへの, 書き出したときと同じ中身の手順
    dataset = load_dataset(dataset_dir)
    unlabeled = load_data(UNLABELED, text_column="review")
    corpus = np.stack([
        dataset.transform_item(unlabeled.load_item(i))
        for i in range(len(unlabeled.frame))
    ])
    np.save(corpus_path, corpus)
    print(f"事前学習用: {corpus_path}, 形 {corpus.shape}")


if __name__ == "__main__":
    main()
