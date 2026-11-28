from pathlib import Path
import sys
from typing import Any

import ml_quest
import yaml
from ml_eda import export_dataset, load_data

# このタスクのディレクトリ, データの置き場, 画像の生成の骨組み
TASK_DIR = Path(__file__).resolve().parent
DATA_DIR = TASK_DIR.parents[1] / "data"
IMAGE_DIR = DATA_DIR / "mnist_png"
PROJECT_DIR = TASK_DIR / "generate" / "project"

# PNG にする MNIST の学習用の画像の枚数 (先頭から)
COUNT = 10000

# 画像の 1 辺の画素の数
SIZE = 28


def write_images() -> Path:
    """
    MNIST の学習用の画像の先頭の COUNT 枚を, クラスの無い PNG の
    ディレクトリに書く関数 (あれば書かない)

    Parameters
    ----------
    None

    Returns
    ----------
    path: Path
        <番号>.png のディレクトリ
    """
    # 11 と同じ torchvision の MNIST (data/torchvision に取得. 08 と共有)
    if len(list(IMAGE_DIR.glob("*.png"))) == COUNT:
        return IMAGE_DIR
    from torchvision.datasets import MNIST
    mnist = MNIST(DATA_DIR / "torchvision", train=True, download=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    for index in range(COUNT):
        image, _ = mnist[index]
        image.save(IMAGE_DIR / f"{index:05d}.png")

    return IMAGE_DIR


def conv(name: str, source: str, channels: tuple[int, int],
         **params: Any) -> dict:
    """
    3 × 3 の畳み込みのノードを作る関数

    Parameters
    ----------
    name: str
        ノードの名前
    source: str
        入力のノードの名前
    channels: tuple[int, int]
        入力と出力のチャンネルの数
    **params: Any
        足す引数 (stride など)

    Returns
    ----------
    node: dict
        ノードの名前から定義
    """
    # 大きさを変えない余白 1
    node = {name: {"unit": "nn.Conv2d", "inputs": source,
                   "params": {"in_channels": channels[0],
                              "out_channels": channels[1],
                              "kernel_size": 3, "padding": 1, **params}}}

    return node


def silu(name: str, source: str) -> dict:
    """
    SiLU のノードを作る関数

    Parameters
    ----------
    name: str
        ノードの名前
    source: str
        入力のノードの名前

    Returns
    ----------
    node: dict
        ノードの名前から定義
    """
    # 引数の無い活性化
    node = {name: {"unit": "nn.SiLU", "inputs": source}}

    return node


def step_nodes(name: str, channels: int) -> dict:
    """
    時刻の埋め込みを, 特徴に足せる (チャンネル, 1, 1) の形にするノードを
    作る関数

    Parameters
    ----------
    name: str
        ノードの名前の頭
    channels: int
        足す先のチャンネルの数

    Returns
    ----------
    nodes: dict
        全結合と形の変更の 2 つのノード
    """
    # 64 次元の埋め込みからチャンネルの数へ
    nodes = {
        f"{name}_fc": {"unit": "nn.Linear", "inputs": "t_act",
                       "params": {"in_features": 64,
                                  "out_features": channels}},
        f"{name}_shape": {"unit": "op.Reshape", "inputs": f"{name}_fc",
                          "params": {"shape": [channels, 1, 1]}},
    }

    return nodes


def denoiser_spec() -> dict:
    """
    雑音を加えた画像と時刻から加えた雑音を当てる, ML-ModelControl の小さな
    U-Net の形の定義を作る関数

    Parameters
    ----------
    None

    Returns
    ----------
    spec: dict
        28 × 28 で 2 段の畳み込みと, 14 × 14 で 1 段の畳み込みに, 時刻の
        埋め込みを足し, 28 × 28 の特徴を飛ばしてつなぐ定義
    """
    # 時刻の埋め込みを各段のチャンネルの数の (チャンネル, 1, 1) にした足し算
    nodes = {
        "t_emb": {"unit": "op.SinusoidalEmbedding", "inputs": "step",
                  "params": {"embedding_dim": 64}},
        "t_fc": {"unit": "nn.Linear", "inputs": "t_emb",
                 "params": {"in_features": 64, "out_features": 64}},
        **silu("t_act", "t_fc"),
        **step_nodes("t1", 32), **step_nodes("t2", 64),
        **conv("c1", "noisy", (1, 32)), **silu("a1", "c1"),
        "add1": {"unit": "op.Add", "inputs": {"xs": ["a1", "t1_shape"]}},
        **conv("c2", "add1", (32, 32)), **silu("a2", "c2"),
        **conv("down", "a2", (32, 64), stride=2), **silu("a3", "down"),
        "add2": {"unit": "op.Add", "inputs": {"xs": ["a3", "t2_shape"]}},
        **conv("c3", "add2", (64, 64)), **silu("a4", "c3"),
        "up": {"unit": "nn.ConvTranspose2d", "inputs": "a4",
               "params": {"in_channels": 64, "out_channels": 32,
                          "kernel_size": 4, "stride": 2, "padding": 1}},
        **silu("a5", "up"),
        "skip": {"unit": "op.Concat", "inputs": {"xs": ["a2", "a5"]},
                 "params": {"dim": 1}},
        **conv("c4", "skip", (64, 32)), **silu("a6", "c4"),
        **conv("head", "a6", (32, 1)),
    }
    spec = {
        "format_version": 1, "name": "mnist_denoiser",
        "inputs": {"noisy": {"shape": [2, 1, SIZE, SIZE]},
                   "step": {"shape": [2], "dtype": "int64",
                            "fill": "arange"}},
        "nodes": nodes, "outputs": {"noise": "head"},
    }

    return spec


def main() -> None:
    """
    問題の設定のレシピの雛形で画像のデータセットを書き出し, 雑音を当てる
    モデルの定義を骨組みの場所に置く関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 問題の設定と骨組み (無ければ作り方の案内)
    path = TASK_DIR / "generate" / "problem.yaml"
    if not path.is_file() or not (PROJECT_DIR / "train.py").is_file():
        print("generate/problem.yaml か骨組みがありません. 先に "
              "setup_problem.py を実行してください")
        sys.exit(1)

    # 画像のディレクトリを入れた問題の設定 (手元の絶対パスを含むので
    # problem.yaml には書かない) と, レシピの雛形
    problem = ml_quest.attach_data(ml_quest.load_problem(path),
                                   str(write_images()))
    start = ml_quest.recipe_from(problem)
    load = start["load"]
    print(f"読み方: {load['loader']}, レシピ: {start['recipe']}")
    for note in start["notes"]:
        print(f"案内: {note}")

    # 骨組みの config の場所 (train.py が読む場所) に書き出し
    data = load_data(load["path"], loader=load["loader"], **load["options"])
    recorded = ml_quest.record_project(problem, PROJECT_DIR)["project"]
    splits = export_dataset(data, start["recipe"], recorded["dataset"],
                            name="mnist_png", overwrite=True)["splits"]
    print("データセット: " + ", ".join(
        f"{name} {split['shape']}" for name, split in splits.items()
    ))

    # 雑音を当てるモデルの定義
    spec_path = Path(recorded["model_spec"])
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    with open(spec_path, "w", encoding="utf-8") as file:
        yaml.safe_dump(denoiser_spec(), file, allow_unicode=True,
                       sort_keys=False)
    print(f"モデルの定義: {spec_path.relative_to(TASK_DIR)}")


if __name__ == "__main__":
    main()
