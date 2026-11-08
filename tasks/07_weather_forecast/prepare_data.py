import argparse
import csv
from pathlib import Path
from typing import Optional

# このタスクのディレクトリ (データの保存先の相対パスの基準)
TASK_DIR = Path(__file__).resolve().parent

# 気象庁の要素名 (と 2 段目の見出し) から, 整形後の列名への対応
COLUMNS = {
    ("平均気温(℃)", ""): "temp_mean",
    ("最高気温(℃)", ""): "temp_max",
    ("最低気温(℃)", ""): "temp_min",
    ("降水量の合計(mm)", ""): "precip",
    ("日照時間(時間)", ""): "sunshine",
    ("平均風速(m/s)", ""): "wind_mean",
    ("最大風速(m/s)", ""): "wind_max",
    ("最大風速(m/s)", "風向"): "wind_max_dir",
    ("最大瞬間風速(m/s)", ""): "gust_max",
    ("最大瞬間風速(m/s)", "風向"): "gust_dir",
    ("最多風向(16方位)", ""): "wind_mode_dir",
    ("平均蒸気圧(hPa)", ""): "vapor_pressure",
    ("平均湿度(％)", ""): "humidity_mean",
    ("最小相対湿度(％)", ""): "humidity_min",
    ("平均現地気圧(hPa)", ""): "pressure_local",
    ("平均海面気圧(hPa)", ""): "pressure_sea",
    ("最低海面気圧(hPa)", ""): "pressure_sea_min",
    ("平均雲量(10分比)", ""): "cloud_mean",
    ("天気概況(昼：06時～18時)", ""): "weather_day",
    ("天気概況(夜：18時～翌日06時)", ""): "weather_night"
}

# 角度に変換する風向の列
DIRECTION_COLUMNS = ("wind_max_dir", "gust_dir", "wind_mode_dir")

# 16 方位を北から時計回りに並べたもの (22.5 度ずつ)
DIRECTIONS = [
    "北", "北北東", "北東", "東北東", "東", "東南東", "南東", "南南東",
    "南", "南南西", "南西", "西南西", "西", "西北西", "北西", "北北西"
]

# 見出しの行の位置 (ダウンロードした時刻, 空行, 地点名の後)
ELEMENT_ROW = 3
SUB_ROW = 4
INFO_ROW = 5


def parse_args() -> argparse.Namespace:
    """
    コマンドライン引数を読み込む関数

    Parameters
    ----------
    None

    Returns
    ----------
    args: argparse.Namespace
        コマンドライン引数, 入力と出力のパスを持つ
    """
    # 入力と出力のパスを受け取り, 相対パスはタスクのディレクトリから辿る
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", default="../../data/jma/raw/weather.csv",
        help="気象庁からダウンロードした CSV (Shift_JIS)"
    )
    parser.add_argument(
        "--output", default="../../data/jma/tokyo_daily.csv",
        help="整形した CSV (UTF-8) の保存先"
    )
    args = parser.parse_args()

    return args


def resolve(path: str) -> Path:
    """
    パスを解決する関数

    Parameters
    ----------
    path: str
        コマンドライン引数で指定されたパス

    Returns
    ----------
    resolved: Path
        絶対パス, 相対パスの場合はタスクのディレクトリから辿る
    """
    # 相対パスはタスクのディレクトリを基準に解決
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = (TASK_DIR / resolved).resolve()

    return resolved


def direction_to_degree(value: str) -> Optional[float]:
    """
    16 方位の風向を北から時計回りの角度に変換する関数

    Parameters
    ----------
    value: str
        北, 北北東 などの風向, 静穏や空欄の場合もある

    Returns
    ----------
    degree: Optional[float]
        0 以上 360 未満の角度, 方位でない場合は None
    """
    # 16 方位のどれかなら 22.5 度刻みの角度に変換
    if value not in DIRECTIONS:
        degree = None
        return degree
    degree = DIRECTIONS.index(value) * 22.5

    return degree


def find_value_columns(rows: list[list[str]]) -> dict[str, int]:
    """
    値の列の位置を見出しから探す関数

    Parameters
    ----------
    rows: list[list[str]]
        CSV の全ての行

    Returns
    ----------
    positions: dict[str, int]
        整形後の列名から, 元の CSV の列の位置への対応
    """
    # 3 段目の見出しが空の列が値の列 (品質情報などは名前が入る)
    element, sub, info = rows[ELEMENT_ROW], rows[SUB_ROW], rows[INFO_ROW]
    positions = {}
    for j in range(1, len(element)):
        if info[j] != "":
            continue
        key = (element[j], sub[j])
        if key not in COLUMNS:
            raise ValueError(f"知らない要素の列があります: {key}")
        positions[COLUMNS[key]] = j

    return positions


def convert(rows: list[list[str]]) -> tuple[list[dict], dict[str, int]]:
    """
    データの行を整形後の列の辞書に変換する関数

    Parameters
    ----------
    rows: list[list[str]]
        CSV の全ての行

    Returns
    ----------
    records: list[dict]
        日ごとの値, 欠測は空文字
    counts: dict[str, int]
        品質情報ごとの値の数
    """
    # 値の列の位置を確認
    positions = find_value_columns(rows)
    info = rows[INFO_ROW]
    records = []
    counts: dict[str, int] = {}
    for row in rows[INFO_ROW + 1:]:
        if len(row) == 0:
            continue

        # 日付を ISO 形式にし, 値の列を順に取り出し
        year, month, day = (int(v) for v in row[0].split("/"))
        record = {"date": f"{year:04d}-{month:02d}-{day:02d}"}
        for name, j in positions.items():
            value = row[j]
            if name in DIRECTION_COLUMNS:
                degree = direction_to_degree(value)
                value = "" if degree is None else f"{degree:g}"
            record[name] = value

            # 値の後ろの品質情報の集計 (現象なし情報を挟む列にも対応)
            for k in range(j + 1, min(j + 4, len(info))):
                if info[k] == "品質情報":
                    counts[row[k]] = counts.get(row[k], 0) + 1
                    break
        records.append(record)

    return records, counts


def main() -> None:
    """
    気象庁の CSV を学習に使える形に整形して保存する関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # Shift_JIS (cp932) の CSV を読み込み
    args = parse_args()
    input_path = resolve(args.input)
    with open(input_path, encoding="cp932", newline="") as f:
        rows = list(csv.reader(f))

    # 値の列だけを英語の列名で取り出し, 風向は角度に変換
    records, counts = convert(rows)

    # UTF-8 の 1 行見出しの CSV として保存
    output_path = resolve(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    # 件数, 期間, 欠測と品質情報を表示
    n_missing = sum(
        1 for r in records for v in r.values() if v == ""
    )
    print(f"保存先: {output_path}")
    first, last = records[0]["date"], records[-1]["date"]
    print(f"日数: {len(records)} ({first} から {last})")
    print(f"列: {list(records[0].keys())}")
    print(f"空の値の数: {n_missing}")
    print(f"品質情報ごとの値の数: {counts}")


if __name__ == "__main__":
    main()
