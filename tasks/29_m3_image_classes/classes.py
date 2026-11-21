"""
画像と文の分類 (行ごとの受け持ちで, 中身がファイル) を, 25 の apps.py の
run_journey で通すための部品 (29 と 30 で使う). 25 の run_journey に足りない
所 (同梱の例からのモデルの作り方) だけを持つ
"""
import sys
from pathlib import Path
from typing import Any, Callable, Union

import yaml

# 部品は 25 のディレクトリの apps.py
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "25_m1_churn"))
from apps import Client

# 例の定義の手直しの関数 (定義, データセットから手で直した所の説明)
Fix = Callable[[dict, Path], list[str]]


def unit_params(ca: Client) -> dict[str, set[str]]:
    """
    ML-CreateArchitecture の部品の一覧から, 部品ごとの引数の名前を返す関数
    (画面の hasParam が見る登録簿と同じもの)

    Parameters
    ----------
    ca: Client
        ML-CreateArchitecture の客

    Returns
    ----------
    params: dict[str, set[str]]
        部品の名前から引数の名前の集まり
    """
    # GET /api/units は ML-ModelControl の describe_units
    params = {
        str(unit["name"]): {str(p["name"]) for p in unit.get("params") or []}
        for unit in ca.ok("GET", "/api/units")
    }

    return params


def sources_of(value: Any) -> list[str]:
    """
    ノードの inputs から, つなぎ元のノードか入力の名前を並べる関数

    Parameters
    ----------
    value: Any
        ノードの inputs (文字, 並び, 名前から文字の辞書)

    Returns
    ----------
    names: list[str]
        つなぎ元の名前 (「名前:口」の口は除く)
    """
    # 画面の parseRef と同じく, 「:」より前と「.」より前の名前
    if isinstance(value, str):
        refs = [value]
    elif isinstance(value, dict):
        refs = [v for v in value.values() if isinstance(v, str)]
    elif isinstance(value, list):
        refs = [v for v in value if isinstance(v, str)]
    else:
        refs = []
    names = [ref.split(":")[0].split(".")[0] for ref in refs]

    return names


def apply_dataset_inputs(
    spec: dict, info: dict, units: dict[str, set[str]]
) -> tuple[dict, list[str]]:
    """
    ML-CreateArchitecture の「データセットから入力と出力を作る」と同じく,
    返った入力の名前, 形, 型と出す値の数を, 入力が 1 つの例に当てはめる
    関数

    画面の applyDatasetInputs (frontend/src/lib/dataset.ts) のうち, 入力が
    1 つで部品を持つ例に要る所 (名前の付け替え, 形と型, 入力に直につながる
    in_features と in_channels, 出力のつなぎ元の out_features と
    out_channels) を写す. 当てられない所は当てずに飛ばす (画面も同じ)

    Parameters
    ----------
    spec: dict
        例の定義
    info: dict
        POST /api/dataset/inputs の応答 (ML-EDA の model_inputs)
    units: dict[str, set[str]]
        部品ごとの引数の名前

    Returns
    ----------
    result: tuple[dict, list[str]]
        当てはめた定義 (同じ辞書) と, 当てた所の説明
    """
    notes: list[str] = []
    inputs = info.get("inputs") or []
    nodes = spec.get("nodes") or {}
    if len(inputs) != 1 or len(spec.get("inputs") or {}) != 1:
        result = (spec, notes)
        return result
    item = inputs[0]

    # ノードと同じ名前の入力は作らない (画面も何も変えずに知らせる)
    if item.get("name") in nodes:
        notes.append(f"ノード {item['name']} がデータセットの入力と同じ名前"
                     "なので, 何も変えていません")
        result = (spec, notes)
        return result

    # 1 つずつで名前が違えば, 名前を変えてつなぎ先も書き換える
    name = next(iter(spec["inputs"]))
    if item.get("name") and item["name"] != name:
        spec["inputs"] = {item["name"]: spec["inputs"][name]}
        for node in nodes.values():
            node["inputs"] = renamed(node.get("inputs"), name, item["name"])
        notes.append(f"入力 {name} の名前を {item['name']} にしました")
        name = item["name"]

    # 形と型 (値の作り方 fill と位置 ui は残す). 行ごとの受け持ちの
    # データセットは決まらない次元が無い (dataset_shape を返さない)
    # (バッチの次元が決まらない値 None なら, 例の大きさ 4)
    entry = spec["inputs"][name]
    shape = [4 if size is None else int(size)
             for size in item.get("shape") or []]
    entry["shape"] = shape
    if item.get("dtype"):
        entry["dtype"] = item["dtype"]

    # 入力に直につながる in_features を最後の次元に, in_channels を 3 次元
    # 以上の入力の 2 番目の次元に
    for node_name, node in nodes.items():
        if name not in sources_of(node.get("inputs")) or len(shape) < 2:
            continue
        fits = [("in_features", shape[-1])]
        if len(shape) >= 3:
            fits.append(("in_channels", shape[1]))
        params = node.setdefault("params", {})
        for param, size in fits:
            if (param in units.get(node.get("unit"), set())
                    and params.get(param) != size):
                params[param] = size
                notes.append(f"{node_name} の {param} を {size} にしました")

    # 出す値の数. 出力が 1 つで, つなぎ元が out_features か out_channels
    outputs = info.get("outputs")
    refs = list((spec.get("outputs") or {}).values())
    if isinstance(outputs, int) and len(refs) == 1:
        source = sources_of(refs[0])[0] if sources_of(refs[0]) else None
        head = nodes.get(source) if source else None
        has = units.get((head or {}).get("unit"), set())
        param = next(
            (p for p in ("out_features", "out_channels") if p in has), None
        )
        if head is not None and param:
            head.setdefault("params", {})[param] = outputs
            notes.append(f"{source} の {param} を {outputs} にしました")
    result = (spec, notes)

    return result


def renamed(value: Any, old: str, new: str) -> Any:
    """
    ノードの inputs の中の入力の名前を書き換える関数

    Parameters
    ----------
    value: Any
        ノードの inputs
    old: str
        元の名前
    new: str
        新しい名前

    Returns
    ----------
    result: Any
        書き換えた inputs
    """
    # 「名前:口」の口は残す
    def one(ref: Any) -> Any:
        if not isinstance(ref, str):
            return ref
        head, _, port = ref.partition(":")
        value = new + (":" + port if port else "") if head == old else ref
        return value

    if isinstance(value, dict):
        result: Any = {key: one(ref) for key, ref in value.items()}
    elif isinstance(value, list):
        result = [one(ref) for ref in value]
    else:
        result = one(value)

    return result


def check_note(ca: Client, spec: dict) -> str:
    """
    ML-CreateArchitecture の確かめ (形の推定) を 1 行にする関数

    Parameters
    ----------
    ca: Client
        ML-CreateArchitecture の客
    spec: dict
        モデルの定義

    Returns
    ----------
    note: str
        確かめの結果 (通ったか, 誤り)
    """
    # 画面の「確かめる」と同じ API. 誤りは最初の 1 つだけ
    checked = ca.ok("POST", "/api/check", {"spec": spec, "infer": True})
    errors = [i["message"] for i in checked.get("issues") or []
              if i.get("severity") == "error"]
    note = ("形の確かめが通る" if checked.get("ok") and not errors
            else f"形の確かめの誤り {str(errors[:1])[:160]}")

    return note


def example_model(
    example: str, fix: Fix
) -> Callable[[Client, dict, Path], tuple[dict, str, str]]:
    """
    同梱の例から定義を作る関数 (25 の run_journey の make_model) を返す
    関数

    Parameters
    ----------
    example: str
        ML-CreateArchitecture の同梱の例の名前
    fix: Fix
        画面が当てた後に, 例を手で直す関数 (直した所の説明を返す)

    Returns
    ----------
    make_model: Callable[[Client, dict, Path], tuple[dict, str, str]]
        客, 入力と出力, データセットから, 定義, 手で直した所, 画面が
        当てた所を返す関数
    """
    def make_model(
        ca: Client, info: dict, dataset: Path
    ) -> tuple[dict, str, str]:
        # 例を開き, 画面と同じく当ててから確かめ, 残る手直しの後も確かめる
        spec = ca.ok("GET", f"/api/examples/{example}")
        spec = spec.get("spec", spec)
        spec, notes = apply_dataset_inputs(spec, info, unit_params(ca))
        applied = (f"例 {example}: {'; '.join(notes) or '無し'} "
                   f"({check_note(ca, spec)})")
        changes = fix(spec, dataset)
        manual = (f"{'; '.join(changes)} ({check_note(ca, spec)})"
                  if changes else "")
        result = (spec, manual, applied)

        return result

    return make_model


def no_fix(spec: dict, dataset: Path) -> list[str]:
    """
    例を手で直さない関数 (画面が当てるだけで済む例)

    Parameters
    ----------
    spec: dict
        定義
    dataset: Path
        データセット

    Returns
    ----------
    changes: list[str]
        直した所 (無し)
    """
    changes: list[str] = []

    return changes


def table_columns(data: Union[str, Path]) -> list[str]:
    """
    ML-EDA の load_data の表の列 (問題の設定の「データの列」に答える値) を
    返す関数

    Parameters
    ----------
    data: Union[str, Path]
        元のデータのディレクトリ

    Returns
    ----------
    columns: list[str]
        表の列の名前
    """
    # 画面の問題の設定で, 読み込んだデータの列から選ぶのと同じ値
    from ml_eda import load_data

    columns = [str(c) for c in load_data(data).frame.columns]

    return columns


def recipe_of(dataset: Path) -> dict[str, Any]:
    """
    書き出したデータセットのレシピ (recipe.yaml) を読む関数

    Parameters
    ----------
    dataset: Path
        データセットのディレクトリ

    Returns
    ----------
    recipe: dict[str, Any]
        レシピ (無ければ空)
    """
    # 当てはめた状態 (語彙など) を持つ書き出しのレシピ
    path = dataset / "recipe.yaml"
    recipe = (yaml.safe_load(path.read_text(encoding="utf-8")) or {}
              if path.is_file() else {})

    return recipe
