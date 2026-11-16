import argparse
from pathlib import Path
import sys
import unicodedata

import pandas as pd
import yaml

from ml_eda import load_data, load_dataset
from ml_inference import backtest, decide, make_bundle, predict

# このタスクのディレクトリと, 学習の結果を使うタスクのディレクトリ
TASK_DIR = Path(__file__).resolve().parent
TASKS = TASK_DIR.parent
DATA_DIR = TASKS.parent / "data"

# 18 で比べる, 足りない損と余る損の組 (売上 1 円あたり)
SALES_COSTS = [(9, 1), (3, 1), (1, 1), (1, 3)]


def latest_run(experiment: Path) -> Path:
    """
    実験の名前のディレクトリから, 最も新しい実験のディレクトリを返す関数

    Parameters
    ----------
    experiment: Path
        outputs/<experiment.name>

    Returns
    ----------
    run: Path
        <日付>/<時刻> の最も新しいもの

    Raises
    ----------
    SystemExit
        学習を終えた実験が無い場合
    """
    # final_model.pt のある実験だけ (日付と時刻の名前の順)
    runs = sorted(p.parent for p in experiment.glob("*/*/final_model.pt"))
    if not runs:
        print(f"{experiment} に学習を終えた実験がありません. 先に 17 と 18 の"
              "README の手順で学習してください")
        sys.exit(1)
    run = runs[-1]

    return run


def run_settings(task: Path, run: Path) -> tuple[Path, Path]:
    """
    実験の config から, 学習に使ったデータセットとモデルの定義を返す関数

    Parameters
    ----------
    task: Path
        タスクのディレクトリ
    run: Path
        実験のディレクトリ

    Returns
    ----------
    paths: tuple[Path, Path]
        データセットのディレクトリと, モデルの定義のファイル
    """
    # 上書きも含めた, 学習のときの config
    config = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8"))
    paths = (task / config["data"]["dataset"], task / config["model"]["spec"])

    return paths


def sales_bundle(out: Path) -> tuple[Path, pd.Timestamp]:
    """
    18 の実験から, 外れ方を測った一式を作る関数

    Parameters
    ----------
    out: Path
        一式を書く場所

    Returns
    ----------
    result: tuple[Path, pd.Timestamp]
        一式のディレクトリと, テスト用の期間の始め
    """
    # 学習のスクリプトと同じ, 学習用の平均と標準偏差での売上の標準化
    task = TASKS / "18_store_sales"
    run = latest_run(task / "outputs" / "store_sales")
    dataset_dir, spec = run_settings(task, run)
    bundle = out / "bundle_sales"
    info = make_bundle(bundle, spec, dataset_dir, run, target_scale="train",
                       name="store_sales", overwrite=True)
    uncertainty = info["uncertainty"]
    print(f"18 の一式 (実験 {run.parent.name}/{run.name}): 外れ方の形 "
          f"{uncertainty['kind']}, 検証用で求めた 80 % と 90 % の幅がテスト"
          f"用で覆った割合 {uncertainty['coverage']['80']:.3f}, "
          f"{uncertainty['coverage']['90']:.3f}")
    start = load_dataset(dataset_dir).frames["test"]["date"].min()
    result = (bundle, start)

    return result


def last_week(frame: pd.DataFrame) -> pd.Series:
    """
    店ごとに 7 日前の売上 (今のやり方: 前の週の同じ曜日と同じだけ置く)
    を返す関数

    Parameters
    ----------
    frame: pd.DataFrame
        元のデータ (日付と店と売上)

    Returns
    ----------
    values: pd.Series
        行ごとの 7 日前の同じ店の売上 (無ければ欠損)
    """
    # 店の表記をそろえ, 重複を除いた日付と店の売上から引く
    store = frame["store"].map(
        lambda name: unicodedata.normalize("NFKC", str(name)).strip()
    )
    table = pd.DataFrame({"date": frame["date"], "store": store,
                          "sales": frame["sales"]}).drop_duplicates(
        ["date", "store"]
    )
    earlier = table.assign(date=table["date"] + pd.Timedelta(days=7))
    merged = pd.DataFrame({"date": frame["date"], "store": store}).merge(
        earlier, on=["date", "store"], how="left"
    )
    values = merged["sales"].set_axis(frame.index)

    return values


def sales_decisions(out: Path) -> None:
    """
    18 の売上で, 損の比ごとに置く量の方針をテスト用の期間でたどり直す
    関数

    Parameters
    ----------
    out: Path
        一式を書く場所

    Returns
    ----------
    None
    """
    # テスト用の期間と, 過去の値の列のための前の 30 日
    bundle, start = sales_bundle(out)
    frame = load_data(DATA_DIR / "store_sales.csv").frame
    window = frame[frame["date"] >= start - pd.Timedelta(days=30)]
    window = window.reset_index(drop=True)
    window["last_week"] = last_week(window)
    count = (window["date"] >= start).to_numpy() & window[
        "last_week"
    ].notna().to_numpy()

    # 損の比ごとの実際の損 (今のやり方: 前の週の同じ曜日の売上を置く)
    print("18 のテスト用の期間で, 置く量の方針の実際の損 (百万円):")
    for under, over in SALES_COSTS:
        decision = {"type": "quantity", "under_cost": under,
                    "over_cost": over, "current": "last_week"}
        result = backtest(bundle, window, decision, count=count)
        values = {name: -value / 1e6
                  for name, value in result["policies"].items()}
        print(f"  足りない {under} : 余る {over} ({result['rows']:,} 行): "
              + ", ".join(f"{name} {value:,.1f}"
                          for name, value in values.items()))
        for sentence in result["sentences"][1:]:
            print(f"    {sentence}")

    # 3 : 1 でテスト用の期間に置く量を決めたときの文と前提 (前の 30 日は
    # 過去の値の列のためだけに使う)
    table = predict(bundle, window)
    table = table[(window["date"] >= start).to_numpy()]
    decided = decide(bundle, table.reset_index(drop=True),
                     {"type": "quantity", "under_cost": 3, "over_cost": 1})
    for sentence in decided["sentences"] + decided["notes"]:
        print(f"  {sentence}")


def reviews_decisions(out: Path) -> None:
    """
    17 のレビューで, 人が確かめる悪いレビューを決める方針を, テスト用の
    分割でたどり直す関数

    Parameters
    ----------
    out: Path
        一式を書く場所

    Returns
    ----------
    None
    """
    # 17 の実験の一式 (学習のスクリプトが [CLS] を先頭に足した)
    task = TASKS / "17_ja_reviews_bert"
    run = latest_run(task / "outputs" / "natural_subword_scratch_48k")
    dataset_dir, spec = run_settings(task, run)
    dataset = load_dataset(dataset_dir)
    vocabulary = next(
        s["state"]["vocabulary"] for s in dataset.recipe["item_steps"]
        if s["op"] == "vocabulary"
    )
    bundle = out / "bundle_reviews"
    info = make_bundle(
        bundle, spec, dataset_dir, run,
        inputs={"ids": {"source": "items", "prepend": [len(vocabulary) + 1]}},
        output="logits", name="ja_reviews", overwrite=True
    )
    uncertainty = info["uncertainty"]
    print(f"17 の一式: 悪いレビューの確率と実際の割合のずれ (ECE) 検証用 "
          f"{uncertainty['calibration']['ece']:.3f}, テスト用 "
          f"{uncertainty['test_ece']:.3f}")

    # テスト用の分割 (ラベルはクラスの名前に) で, 見逃しの損 10, 空振りの
    # 損 1, 人手の上限の有る無し
    test = dataset.frames["test"]
    classes = dataset.info["classes"]
    past = pd.DataFrame({
        "review": test["review"],
        "label": [classes[i] for i in test["label"]],
    })
    for capacity in (None, 500):
        decision = {"type": "triage", "positive": "悪い", "miss_cost": 10,
                    "false_alarm_cost": 1}
        if capacity is not None:
            decision["capacity"] = capacity
        result = backtest(bundle, past, decision)
        label = "上限なし" if capacity is None else f"上限 {capacity} 件"
        print(f"17 のテスト用の分割で, 人が確かめるレビュー ({label}, "
              f"{result['rows']:,} 件): " + ", ".join(
                  f"{name} 損 {-value:,.0f}"
                  for name, value in result["policies"].items()
              ))
        for sentence in result["sentences"][1:]:
            print(f"    {sentence}")


def main() -> None:
    """
    17 と 18 の学習の結果で, 意思決定の方針を過去のデータでたどり直す
    関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 一式の書き出し先
    parser = argparse.ArgumentParser(description="意思決定のたどり直し")
    parser.add_argument(
        "--out", default="outputs",
        help="書き出し先 (タスクのディレクトリからの相対パスか絶対パス)"
    )
    args = parser.parse_args()
    out = TASK_DIR / args.out
    out.mkdir(parents=True, exist_ok=True)

    # 売上の量と, レビューの振り分け
    sales_decisions(out)
    reviews_decisions(out)


if __name__ == "__main__":
    main()
