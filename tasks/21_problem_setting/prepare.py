from pathlib import Path
import sys

import ml_quest
import yaml
from ml_eda import export_dataset, load_data, load_dataset, profile

# このタスクのディレクトリ, 元のデータ, 書き出す ML-Found のプロジェクト
TASK_DIR = Path(__file__).resolve().parent
SOURCE = TASK_DIR.parents[1] / "data" / "sozai.csv"
PROJECT_DIR = TASK_DIR / "project"


def recipe_of(stages: dict, columns: list[str]) -> dict:
    """
    問題の設定の受け渡しの値から, ML-EDA のレシピを作る関数

    Parameters
    ----------
    stages: dict
        ML-Quest の受け渡し (handoff の結果)
    columns: list[str]
        元のデータの列

    Returns
    ----------
    recipe: dict
        学習に使ってよい列だけを残し, 系列ごとの過去の値 (決める時点で
        分かる期だけ), 曜日, ワンホット, 標準化をする時刻の順の分割の
        レシピ
    """
    # 受け渡しの目的の列, タスク, 分割, 時刻と系列の列, 使ってよい列.
    # 過去の値は決める時点で分かる期 (min_lag 期前) と 1 週間前だけ (移動
    # 平均は決める時点で分からない期を含むので使わない)
    data = stages["data"]
    series = data["series_by"]
    keep = {data["target"], data["time_column"], *series, *data["known"]}
    drop = [name for name in columns if name not in keep]
    lags = sorted({data["min_lag"], max(data["min_lag"], 7)})
    recipe = {
        "format_version": 1, "target": data["target"], "task": data["task"],
        "steps": [
            {"op": "drop_columns", "columns": drop},
            {"op": "lag", "columns": [data["target"]], "lags": lags,
             "by": series, "time": data["time_column"]},
            {"op": "drop_missing"},
            {"op": "datetime_parts", "columns": [data["time_column"]],
             "parts": ["weekday"], "drop": False},
            {"op": "one_hot", "columns": series},
            {"op": "standardize", "columns": "auto"},
        ],
        "split": {"method": data["split"], "valid": 0.15, "test": 0.15,
                  "seed": 0},
    }

    return recipe


def model_spec(width: int) -> dict:
    """
    入力の列の数に合わせた, 2 層の全結合のモデルの定義を作る関数

    Parameters
    ----------
    width: int
        学習に使う列の数

    Returns
    ----------
    spec: dict
        ML-ModelControl の形のモデルの定義 (出力 1)
    """
    # width → 64 → 64 → 1
    nodes = {
        "fc1": {"unit": "nn.Linear", "inputs": {"input": "x"},
                "params": {"in_features": width, "out_features": 64}},
        "act1": {"unit": "nn.ReLU", "params": {}, "inputs": {"input": "fc1"}},
        "fc2": {"unit": "nn.Linear", "inputs": {"input": "act1"},
                "params": {"in_features": 64, "out_features": 64}},
        "act2": {"unit": "nn.ReLU", "params": {}, "inputs": {"input": "fc2"}},
        "out": {"unit": "nn.Linear", "inputs": {"input": "act2"},
                "params": {"in_features": 64, "out_features": 1}},
    }
    spec = {"format_version": 1, "name": "sozai_mlp",
            "inputs": {"x": {"shape": [64, width]}}, "nodes": nodes,
            "outputs": {"y": "out"}}

    return spec


def main() -> None:
    """
    問題の設定の受け渡しの値で, データセットとモデルの定義をプロジェクトに
    置く関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 元のデータと問題の設定 (無ければ作り方の案内)
    if not SOURCE.is_file():
        print("data/sozai.csv がありません. 先に make_data.py を実行して"
              "ください")
        sys.exit(1)
    if not (TASK_DIR / "problem.yaml").is_file() or not (
        PROJECT_DIR / "train.py"
    ).is_file():
        print("problem.yaml か骨組みがありません. 先に setup_problem.py を"
              "実行してください")
        sys.exit(1)
    problem = ml_quest.load_problem(TASK_DIR / "problem.yaml")
    stages = ml_quest.handoff(problem)

    # 時刻の列は読み込みで指定 (系列を分ける列が 2 つあると推定できない)
    data = load_data(SOURCE, time_column=stages["data"]["time_column"])
    result = profile(data, target=stages["data"]["target"])
    series = result["timeseries"] or {}
    print(f"種類: {data.kind}, 行: {len(data.frame)}, 系列を分ける列: "
          f"{series.get('series_by')}")
    print(f"受け渡しの値: {stages['data']}, 監視する指標 "
          f"{stages['training']['monitor']}")
    path = PROJECT_DIR / "outputs" / "dataset"
    export_dataset(data, recipe_of(stages, list(data.frame.columns)), path,
                   name="sozai", overwrite=True)
    dataset = load_dataset(path)
    print(f"学習に使う列 {len(dataset.features)}: {dataset.features}")

    # 入力の列の数に合わせたモデルの定義
    spec_path = PROJECT_DIR / "models" / "model.yaml"
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    with open(spec_path, "w", encoding="utf-8") as file:
        yaml.safe_dump(model_spec(len(dataset.features)), file,
                       allow_unicode=True, sort_keys=False)
    print(f"モデルの定義: {spec_path.relative_to(TASK_DIR)}")


if __name__ == "__main__":
    main()
