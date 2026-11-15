from pathlib import Path
import sys

import numpy as np

from ml_eda import export_dataset, load_data, load_dataset, profile

# このタスクのディレクトリと, 元のデータと書き出す先
TASK_DIR = Path(__file__).resolve().parent
SOURCE = TASK_DIR.parents[1] / "data" / "reviews_ja.csv"
UNLABELED = TASK_DIR.parents[1] / "data" / "reviews_ja_unlabeled.csv"
DATASET = TASK_DIR / "outputs" / "dataset"
CORPUS = TASK_DIR / "outputs" / "corpus.npy"


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

    # レシピで書き出す (語彙は学習用の文だけで作る)
    info = export_dataset(
        data, TASK_DIR / "recipe.yaml", DATASET, name="reviews_ja",
        overwrite=True
    )
    shapes = {name: item["shape"] for name, item in info["splits"].items()}
    print(f"書き出し: {DATASET}, 形 {shapes}, クラス {info['classes']}")

    # ラベルの無いレビューに, 書き出したときと同じ中身の手順をかける
    dataset = load_dataset(DATASET)
    unlabeled = load_data(UNLABELED, text_column="review")
    corpus = np.stack([
        dataset.transform_item(unlabeled.load_item(i))
        for i in range(len(unlabeled.frame))
    ])
    np.save(CORPUS, corpus)
    print(f"事前学習用: {CORPUS}, 形 {corpus.shape}")


if __name__ == "__main__":
    main()
