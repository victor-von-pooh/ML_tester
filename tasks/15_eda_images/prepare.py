import argparse
from pathlib import Path
import sys

from ml_eda import export_dataset, load_data, prepare, profile

# このタスクのディレクトリと, 元のデータ
TASK_DIR = Path(__file__).resolve().parent
SOURCE = TASK_DIR.parents[1] / "data" / "shapes"


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
    parser.add_argument(
        "--recipe", default="recipe.yaml",
        help="レシピ (タスクのディレクトリから)"
    )
    parser.add_argument(
        "--out", default="outputs/dataset",
        help="書き出す先 (タスクのディレクトリから)"
    )
    args = parser.parse_args()

    return args


def main() -> None:
    """
    画像のフォルダを分析して気付きを表示し, 中身を読まずにレシピを
    確かめてから, 学習用の配列に書き出す関数

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

    # 元のデータが無ければ作り方の案内と終了
    if not SOURCE.is_dir():
        print("data/shapes がありません. 先に make_data.py を実行してください")
        sys.exit(1)

    # 読み込みと分析 (クラスのフォルダの名前が label になる)
    data = load_data(SOURCE)
    result = profile(data, target="label")
    print(f"種類: {data.kind}, 件数: {len(data.frame)}")
    for item in result["findings"]:
        print(f"[{item['severity']}] {item['code']}: {item['message']}")

    # 中身を読まないレシピの確認 (ML-Fathom の前と後の様子と同じ) と,
    # 画像を読んで配列にした書き出し (前のものは置き換える)
    try:
        checked = prepare(data, recipe, read_items=False)
        rows = {n: len(frame) for n, frame in checked.frames.items()}
        print(f"分割: {rows}, 注意: {checked.notes or 'なし'}")
        info = export_dataset(
            data, recipe, dataset, name=dataset.name, overwrite=True
        )
    except (ValueError, FileExistsError) as error:
        print(f"書き出せません: {error}")
        sys.exit(1)
    shapes = {name: item["shape"] for name, item in info["splits"].items()}
    print(f"書き出し: {dataset}")
    print(f"配列の形: {shapes}, クラス: {info['classes']}")


if __name__ == "__main__":
    main()
