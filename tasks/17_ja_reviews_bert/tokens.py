from pathlib import Path
from typing import Optional

import torch
import yaml
from torch import nn

from ml_eda import Dataset
from ml_modelcontrol import build_model


def special_ids(dataset: Dataset) -> tuple[int, int, int]:
    """
    モデルの語の数と, 足す 2 つの特別な番号を返す関数

    ML-EDA の語彙は 0 が詰め物 (<pad>), 1 が語彙に無い文字 (<unk>). その
    後ろに, 隠した文字の印 ([MASK]) と文の先頭の印 ([CLS]) を足す

    Parameters
    ----------
    dataset: Dataset
        ML-EDA のデータセット (中身の手順に vocabulary を持つ)

    Returns
    ----------
    ids: tuple[int, int, int]
        モデルの語の数 (語彙 + 2), [MASK] の番号, [CLS] の番号
    """
    # 当てはめた語彙の長さ
    vocabulary = next(
        step["state"]["vocabulary"] for step in dataset.recipe["item_steps"]
        if step["op"] == "vocabulary"
    )
    size = len(vocabulary)
    ids = (size + 2, size, size + 1)

    return ids


def add_cls(ids: torch.Tensor, cls_id: int) -> torch.Tensor:
    """
    番号の並びの先頭に [CLS] を足す関数 (モデルは先頭の表現で分類する)

    Parameters
    ----------
    ids: torch.Tensor
        (件数, 長さ) の番号の並び
    cls_id: int
        [CLS] の番号

    Returns
    ----------
    added: torch.Tensor
        (件数, 長さ + 1) の番号の並び
    """
    # 先頭の列の追加
    head = torch.full((len(ids), 1), cls_id, dtype=ids.dtype)
    added = torch.cat([head, ids], dim=1)

    return added


def load_model(
    task_dir: Path, spec_path: str, init: Optional[str], seed: int,
    vocab: int, length: int
) -> nn.Module:
    """
    ML-CreateArchitecture で組んだ定義からモデルを作り, 重みを読む関数

    Parameters
    ----------
    task_dir: Path
        タスクのディレクトリ
    spec_path: str
        定義のパス (タスクのディレクトリから)
    init: Optional[str]
        読み込む重みのパス (None なら読まない)
    seed: int
        重みの初期値の乱数のシード
    vocab: int
        データセットから求めた語の数 (定義の埋め込みの大きさと比べる)
    length: int
        [CLS] を足した後の並びの長さ (位置の埋め込みに収まるかを試す)

    Returns
    ----------
    model: nn.Module
        組み立てたモデル

    Raises
    ----------
    ValueError
        定義の語の数や出力の幅がデータセットと合わない場合, 重みが無い場合
    """
    # 入力 ids を受ける埋め込みの大きさと, 文字を当てる出力 (mlm) の幅の確認
    spec = yaml.safe_load((task_dir / spec_path).read_text(encoding="utf-8"))
    for name, node in spec["nodes"].items():
        refs = node.get("inputs", {})
        refs = refs.values() if isinstance(refs, dict) else [refs]
        size = node.get("params", {}).get("num_embeddings")
        if "ids" in refs and size is not None and size != vocab:
            raise ValueError(
                f"{name} の埋め込みの大きさ {size} がデータセットの {vocab} "
                "(語彙 + [MASK] + [CLS]) と合いません. "
                "ML-CreateArchitecture で num_embeddings を直してください"
            )
    model = build_model(task_dir / spec_path, seed=seed)
    with torch.no_grad():
        outputs = model(torch.ones(1, length, dtype=torch.long))
    width = outputs["mlm"].shape[-1] if isinstance(outputs, dict) and (
        "mlm" in outputs
    ) else vocab
    if width != vocab:
        raise ValueError(f"出力 mlm の幅 {width} が語の数 {vocab} と合いません")
    if init:
        path = task_dir / init
        if not path.is_file():
            raise ValueError(f"{path} がありません. 先に pretrain.py を実行"
                             "してください (使わない場合は model.init=null)")
        model.load_state_dict(torch.load(path, map_location="cpu"))

    return model
