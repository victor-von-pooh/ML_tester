import argparse
from pathlib import Path
import shutil
import sys
import unicodedata

import pandas as pd
import yaml

from ml_eda import load_data, load_dataset
from ml_inference import (
    backtest, decide, load_bundle, make_bundle, predict, remeasure
)

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
    # 最新の実験の一式 (売上の標準化は学習のスクリプトと同じ学習用の平均と
    # 標準偏差) と, テスト用の期間の始め
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


def store_names(frame: pd.DataFrame) -> pd.Series:
    """
    店の表記 (全角と半角, 前後の空白) をそろえる関数

    Parameters
    ----------
    frame: pd.DataFrame
        元のデータ

    Returns
    ----------
    store: pd.Series
        そろえた店の名前
    """
    # NFKC と前後の空白の除去
    store = frame["store"].map(
        lambda name: unicodedata.normalize("NFKC", str(name)).strip()
    )

    return store


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
    # 表記をそろえた店と, 重複を除いた日付と店の売上からの引き当て
    store = store_names(frame)
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


def sales_window(start: pd.Timestamp) -> pd.DataFrame:
    """
    テスト用の期間と, 過去の値の列のための前の 30 日の売上を返す関数

    Parameters
    ----------
    start: pd.Timestamp
        テスト用の期間の始め

    Returns
    ----------
    window: pd.DataFrame
        元のデータの行 (同じ日と店の重複は 1 行) に, 前の週の同じ曜日の
        売上の列 last_week を足した表
    """
    # 同じ日と店の重複は 1 行に (前処理と同じく, 損を 2 回数えないため)
    frame = load_data(DATA_DIR / "store_sales.csv").frame
    window = frame[frame["date"] >= start - pd.Timedelta(days=30)]
    window = window.assign(_store=store_names(window)).drop_duplicates(
        ["date", "_store"]
    ).drop(columns="_store").reset_index(drop=True)
    window["last_week"] = last_week(window)

    return window


def sales_decisions(out: Path) -> tuple[Path, pd.DataFrame, pd.Timestamp]:
    """
    18 の売上で, 損の比ごとに置く量の方針をテスト用の期間でたどり直す
    関数

    Parameters
    ----------
    out: Path
        一式を書く場所

    Returns
    ----------
    result: tuple[Path, pd.DataFrame, pd.Timestamp]
        一式のディレクトリ, sales_window の表, テスト用の期間の始め
    """
    # 18 の一式と, テスト用の期間と前の 30 日の売上
    bundle, start = sales_bundle(out)
    window = sales_window(start)
    count = (window["date"] >= start).to_numpy() & window[
        "last_week"
    ].notna().to_numpy()

    # 損の比ごとの実際の損 (今のやり方: 前の週の同じ曜日の売上を置く. 差の
    # 幅は日ごとの選び直し. 同じ日の店どうしは似ているため. 18 は行の順で
    # ずらす一式なので, 自動では行ごと)
    print("18 のテスト用の期間で, 置く量の方針の実際の損 (百万円):")
    for under, over in SALES_COSTS:
        decision = {"type": "quantity", "under_cost": under,
                    "over_cost": over, "current": "last_week"}
        result = backtest(bundle, window, decision, count=count,
                          group="date")
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
    result = (bundle, window, start)

    return result


def remeasured_levels(
    bundle: Path, window: pd.DataFrame, rows: pd.Series, decision: dict
) -> pd.Series:
    """
    一式で, 選んだ行に置く量を決める関数

    Parameters
    ----------
    bundle: Path
        一式のディレクトリ
    window: pd.DataFrame
        sales_window の表 (過去の値の列のために全ての行で予測する)
    rows: pd.Series
        量を決める行の印
    decision: dict
        量の決め方

    Returns
    ----------
    levels: pd.Series
        行ごとの置く量 (選ばなかった行は欠損)
    """
    # 全ての行での予測と, 選んだ行だけでの決定
    table = predict(bundle, window)
    part = table[rows.to_numpy()].reset_index(drop=True)
    action = decide(bundle, part, decision)["frame"]["action"]
    levels = pd.Series(float("nan"), index=window.index)
    levels[rows.to_numpy()] = pd.to_numeric(action, errors="coerce").to_numpy()

    return levels


def remeasure_decisions(
    out: Path, bundle: Path, window: pd.DataFrame, start: pd.Timestamp
) -> None:
    """
    テスト用の期間の前半の実績で外れ方を測り直し, 後半で前の外れ方の
    方針と比べる関数

    Parameters
    ----------
    out: Path
        一式を書く場所
    bundle: Path
        18 の一式のディレクトリ (書き換えない)
    window: pd.DataFrame
        sales_window の表
    start: pd.Timestamp
        テスト用の期間の始め

    Returns
    ----------
    None
    """
    # 期間の前半での測り直し (一式は写してから書き換え)
    days = sorted(window.loc[window["date"] >= start, "date"].unique())
    middle = days[len(days) // 2]
    recent = out / "bundle_sales_recent"
    if recent.exists():
        shutil.rmtree(recent)
    shutil.copytree(bundle, recent)
    first = (window["date"] >= start) & (window["date"] < middle)
    result = remeasure(recent, window, count=first.to_numpy())
    print(f"18 のテスト用の期間の前半 ({len(days) // 2} 日) で外れ方を"
          "測り直した:")
    for sentence in result["sentences"]:
        print(f"  {sentence}")

    # 後半での, 前の外れ方の方針と測り直した外れ方の方針の比べ (測り直した
    # 方の量を今のやり方の列として渡し, 差の 90 % の幅も求める)
    second = (window["date"] >= middle) & window["last_week"].notna()
    for name, path in (("前の記録の", bundle), ("測り直した", recent)):
        check = remeasure(path, window, count=second.to_numpy(), write=False)
        coverage = check["before"]["coverage"]
        print(f"  {name}外れ方の 80 % と 90 % の幅が後半の実績を覆った割合: "
              f"{coverage['80']:.3f}, {coverage['90']:.3f}")
    old = load_bundle(bundle).info["uncertainty"]
    split = {"valid": "検証用", "train": "学習用"}.get(
        old["split"], str(old["split"])
    )
    print(f"後半で, 前の記録 ({split}の分割の {old['rows']:,} 行) "
          "と測り直した外れ方の量の実際の損 (百万円):")
    for under, over in SALES_COSTS:
        decision = {"type": "quantity", "under_cost": under,
                    "over_cost": over}
        frame = window.assign(remeasured=remeasured_levels(
            recent, window, second, decision
        ))
        replay = backtest(bundle, frame, {**decision,
                                          "current": "remeasured"},
                          count=second.to_numpy(), group="date")
        gap = replay["gaps"]["今のやり方"]
        low, high = gap["range"]
        print(f"  足りない {under} : 余る {over} ({replay['rows']:,} 行): "
              f"前の記録 {-replay['policies']['この方針'] / 1e6:,.1f}, "
              f"測り直し {-replay['policies']['今のやり方'] / 1e6:,.1f}, "
              f"予測の値どおり "
              f"{-replay['policies']['予測の値どおりに置く'] / 1e6:,.1f} "
              f"(測り直しの改善 {-gap['gap'] / 1e6:,.2f}, 90 % の幅 "
              f"{-high / 1e6:,.2f} から {-low / 1e6:,.2f})")


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

    # 売上の量と外れ方の測り直しと, レビューの振り分け
    bundle, window, start = sales_decisions(out)
    remeasure_decisions(out, bundle, window, start)
    reviews_decisions(out)


if __name__ == "__main__":
    main()
