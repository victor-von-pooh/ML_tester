import argparse
from pathlib import Path
import sys

from ml_eda import export_dataset, load_data, profile

# このタスクのディレクトリと, 元のデータと書き出す先
TASK_DIR = Path(__file__).resolve().parent
SOURCE = TASK_DIR.parents[1] / "data" / "churn.csv"


def parse_args() -> argparse.Namespace:
    """
    コマンドライン引数を読む関数

    Parameters
    ----------
    None

    Returns
    ----------
    args: argparse.Namespace
        recipe (レシピ) と out (書き出す先, タスクのディレクトリから)
    """
    # 既定は recipe.yaml と outputs/dataset
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe", default="recipe.yaml")
    parser.add_argument("--out", default="outputs/dataset")
    args = parser.parse_args()

    return args


def main() -> None:
    """
    元のデータを分析して気付きを表示し, レシピで学習用に書き出す関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # レシピと書き出す先
    args = parse_args()
    recipe = TASK_DIR / args.recipe
    dataset = TASK_DIR / args.out

    # 元のデータが無ければ作り方を案内して終える
    if not SOURCE.is_file():
        print("data/churn.csv がありません. 先に make_data.py を実行してください")
        sys.exit(1)

    # 読み込みと分析 (目的の列 churn との関係を含む)
    data = load_data(SOURCE)
    result = profile(data, target="churn")
    print(f"種類: {data.kind}, 行: {len(data.frame)}")
    for item in result["findings"]:
        print(f"[{item['severity']}] {item['code']}: {item['message']}")

    # レシピで前処理し, 分割して書き出す (前のものは置き換える)
    info = export_dataset(
        data, recipe, dataset, name=dataset.name, overwrite=True
    )
    rows = {name: item["rows"] for name, item in info["splits"].items()}
    print(f"書き出し: {dataset}")
    print(f"分割: {rows}, 学習に使う列: {len(info['features'])}")
    print(f"クラス: {info['classes']}, 注意: {info['notes'] or 'なし'}")


if __name__ == "__main__":
    main()
