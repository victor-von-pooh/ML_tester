import argparse
from pathlib import Path
import sys

from ml_eda import export_dataset, load_data, profile

# このタスクのディレクトリと, 元のデータ
TASK_DIR = Path(__file__).resolve().parent
SOURCE = TASK_DIR.parents[1] / "data" / "store_sales.csv"


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
    店ごとの売上を分析して気付きを表示し, ML-Fathom で作ったレシピで
    書き出す関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 元のデータが無ければ作り方の案内と終了
    args = parse_args()
    if not SOURCE.is_file():
        print("data/store_sales.csv がありません. 先に make_data.py を"
              "実行してください")
        sys.exit(1)

    # 読み込みと分析 (目的の列 sales との関係, 店ごとの系列)
    data = load_data(SOURCE)
    result = profile(data, target="sales")
    series = result["timeseries"] or {}
    print(f"種類: {data.kind}, 行: {len(data.frame)}, 系列を分ける列: "
          f"{series.get('series_by')} ({series.get('series_count')} 本)")
    for item in result["findings"]:
        print(f"[{item['severity']}] {item['code']}: {item['message'][:80]}")

    # レシピで前処理し, 時刻の順に分けて書き出す (前のものは置き換える)
    out = TASK_DIR / args.out
    try:
        info = export_dataset(
            data, TASK_DIR / args.recipe, out, name=out.name, overwrite=True
        )
    except (ValueError, FileExistsError) as error:
        print(f"書き出せません: {error}")
        sys.exit(1)
    rows = {name: item["rows"] for name, item in info["splits"].items()}
    print(f"書き出し: {out}, 分割 {rows}, 学習に使う列 "
          f"{len(info['features'])}, 注意: {info['notes'] or 'なし'}")


if __name__ == "__main__":
    main()
