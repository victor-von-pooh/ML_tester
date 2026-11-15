from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ml_eda import load_dataset
from ml_found import Trainer, load_project_config, start_experiment
from tokens import add_cls, load_model, special_ids

# このタスクのディレクトリ (ML-Found のプロジェクトのディレクトリ)
TASK_DIR = Path(__file__).resolve().parent

# 事前学習の後の重みの置き場 (train.py の model.init の既定)
PRETRAINED = TASK_DIR / "outputs" / "pretrained.pt"


def make_step(mask_id: int, cls_id: int, vocab: int, ratio: float):
    """
    隠した文字を当てる (MLM) 1 バッチ分の損失を求める関数を作る関数

    Parameters
    ----------
    mask_id: int
        [MASK] の番号
    cls_id: int
        [CLS] の番号
    vocab: int
        モデルの語の数
    ratio: float
        隠す文字の割合

    Returns
    ----------
    step: Callable
        ML-Found の step (model, batch, phase) -> 損失と正解率
    """
    def step(model: nn.Module, batch: tuple, phase: str) -> dict:
        """
        詰め物と [CLS] 以外の文字の一部を [MASK] にし, 元の文字を当てる
        関数

        Parameters
        ----------
        model: nn.Module
            学習するモデル
        batch: tuple
            [CLS] を足した番号の並び
        phase: str
            train または valid

        Returns
        ----------
        outputs: dict
            隠した文字の交差エントロピー損失と正解率
        """
        # 隠す位置を決め, 隠した位置だけを当てる
        (ids,) = batch
        chosen = (torch.rand(ids.shape, device=ids.device) < ratio) & (
            ids > 0
        ) & (ids != cls_id)
        inputs = ids.masked_fill(chosen, mask_id)
        labels = ids.masked_fill(~chosen, -100)
        logits = model(inputs)["mlm"]
        ce = nn.CrossEntropyLoss(ignore_index=-100)
        loss = ce(logits.reshape(-1, vocab), labels.reshape(-1))
        hit = (logits.argmax(dim=-1) == ids) & chosen
        acc = hit.sum() / chosen.sum().clamp(min=1)
        outputs = {"loss": loss, "acc": acc}

        return outputs

    return step


def main() -> None:
    """
    ラベルの無いレビューで, ML-CreateArchitecture で組んだ BERT を事前学習
    する関数 (train.py の微調整の前に実行する)

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # config (事前学習の差分) とデータ
    cfg, diff = load_project_config(str(TASK_DIR))
    corpus_path = TASK_DIR / cfg.data.corpus
    if not corpus_path.is_file():
        print(f"{corpus_path} がありません. 先に prepare.py を実行してください")
        sys.exit(1)
    dataset = load_dataset(TASK_DIR / cfg.data.dataset)
    vocab, mask_id, cls_id = special_ids(dataset)
    ids = add_cls(torch.from_numpy(np.load(corpus_path)).long(), cls_id)

    # 5 % を検証用にする (事前学習の進み具合を見るため)
    generator = torch.Generator().manual_seed(0)
    order = torch.randperm(len(ids), generator=generator)
    cut = len(ids) // 20
    train_loader = DataLoader(
        TensorDataset(ids[order[cut:]]), batch_size=cfg.data.batch_size,
        shuffle=True
    )
    valid_loader = DataLoader(
        TensorDataset(ids[order[:cut]]), batch_size=cfg.data.batch_size
    )

    # 定義からモデルを作り, 隠した文字を当てる学習
    model = load_model(
        TASK_DIR, cfg.model.spec, cfg.model.init, cfg.experiment.seed, vocab
    )
    experiment = start_experiment(cfg, diff, str(TASK_DIR))
    experiment.logger.info(
        f"事前学習: {len(ids)} 件, 語の数 {vocab} ([MASK] {mask_id}, "
        f"[CLS] {cls_id})"
    )
    step = make_step(mask_id, cls_id, vocab, cfg.pretrain.mask_ratio)
    trainer = Trainer(model, experiment, step)
    history = trainer.fit(train_loader, valid_loader)

    # 微調整で読み込む重み
    torch.save(trainer.model.state_dict(), PRETRAINED)
    print(f"保存先: {experiment.run_dir}")
    print(f"検証用の隠した文字の正解率: {history.get('valid/acc')[-1]:.4f}")
    print(f"微調整で読み込む重み: {PRETRAINED}")


if __name__ == "__main__":
    main()
