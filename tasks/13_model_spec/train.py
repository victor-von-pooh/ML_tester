import importlib.util
from pathlib import Path
import shutil
import sys

from omegaconf import DictConfig
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ml_found import Trainer, load_project_config, start_experiment
from ml_modelcontrol import SpecError, build_model, export_python, infer

# このタスクのディレクトリ (ML-Found のプロジェクトのディレクトリ)
TASK_DIR = Path(__file__).resolve().parent

# 手書き数字のクラスの数
NUM_CLASSES = 10


def make_loaders(
    cfg: DictConfig
) -> tuple[DataLoader, DataLoader, torch.Tensor]:
    """
    手書き数字のデータから学習用と検証用の DataLoader を作る関数

    Parameters
    ----------
    cfg: DictConfig
        学習に使う config

    Returns
    ----------
    train_loader: DataLoader
        学習用の DataLoader
    valid_loader: DataLoader
        検証用の DataLoader
    x_valid: torch.Tensor
        検証用の画像 (組み立て直したモデルとの比較に使う)
    """
    # 0 から 16 の画素を 0 から 1 にし, (枚数, 1, 8, 8) の画像に変換
    x, y = load_digits(return_X_y=True)
    x = x.reshape(-1, 1, 8, 8) / 16.0
    x_train, x_valid, y_train, y_valid = train_test_split(
        x, y, test_size=cfg.data.valid_size, stratify=y,
        random_state=cfg.experiment.seed
    )

    # テンソルにして DataLoader に集約
    x_train = torch.from_numpy(x_train).float()
    x_valid = torch.from_numpy(x_valid).float()
    train_loader = DataLoader(
        TensorDataset(x_train, torch.from_numpy(y_train).long()),
        batch_size=cfg.data.batch_size, shuffle=True
    )
    valid_loader = DataLoader(
        TensorDataset(x_valid, torch.from_numpy(y_valid).long()),
        batch_size=cfg.data.batch_size, shuffle=False
    )

    return train_loader, valid_loader, x_valid


def step(model: nn.Module, batch: tuple, phase: str) -> dict:
    """
    1 バッチ分の損失と正解率を求める関数

    Parameters
    ----------
    model: nn.Module
        学習するモデル (定義から組み立てたもの. 出力が 1 つなのでテンソル)
    batch: tuple
        画像と正解ラベル
    phase: str
        train または valid

    Returns
    ----------
    outputs: dict
        交差エントロピー損失と正解率
    """
    # 予測して損失と正解率の計算
    x, y = batch
    logits = model(x)
    ce = nn.CrossEntropyLoss()
    loss = ce(logits, y)
    acc = (logits.argmax(dim=1) == y).float().mean()
    outputs = {"loss": loss, "acc": acc}

    return outputs


def build_from_spec(spec_path: Path, seed: int) -> nn.Module:
    """
    モデルの定義を確かめて組み立てる関数

    誤りがあれば, 実験の出力先を作る前に理由を表示して終了する

    Parameters
    ----------
    spec_path: Path
        モデルの定義のファイル
    seed: int
        重みの初期値の乱数のシード

    Returns
    ----------
    model: nn.Module
        組み立てたモデル
    """
    # 定義の誤り (引数の不足など) は組み立てで SpecError になる
    try:
        model = build_model(spec_path, seed=seed)
    except SpecError as error:
        print(f"モデルの定義 {spec_path.name} に誤りがあります")
        for issue in error.issues:
            print(f"- {issue['node'] or '定義'}: {issue['message']}")
        sys.exit(1)

    # 出力の形がクラスの数と合うかを形の推定で確認
    result = infer(spec_path)
    shape = result["outputs"]["logits"]["shape"]
    if shape[-1] != NUM_CLASSES:
        print(f"出力の形 {shape} の最後がクラスの数 {NUM_CLASSES} と違います")
        sys.exit(1)
    print(f"定義 {spec_path.name}: パラメータ数 {result['params']}")

    return model


def check_rebuild(
    run_dir: Path, model: nn.Module, x_valid: torch.Tensor
) -> None:
    """
    実験の出力先の定義と重みから, 同じモデルを作り直せるかを確かめる関数

    組み立て直したモデルと, 書き出した Python のコードのモデルの両方で,
    学習の後のモデルと同じ出力になるかを表示する

    Parameters
    ----------
    run_dir: Path
        実験の出力先
    model: nn.Module
        学習の後のモデル (final_model.pt と同じ重み)
    x_valid: torch.Tensor
        検証用の画像

    Returns
    ----------
    None
    """
    # 学習の後のモデルの出力 (推論モード)
    model.eval()
    with torch.no_grad():
        expected = model(x_valid)

    # 写した定義から組み立て直し, 保存した重みを読み込み
    state = torch.load(run_dir / "final_model.pt")
    rebuilt = build_model(run_dir / "model_spec.yaml")
    rebuilt.load_state_dict(state)
    rebuilt.eval()

    # 定義から書き出したコードのクラスを読み込み, 同じ重みを読み込み
    code_path = run_dir / "model_export.py"
    export_python(run_dir / "model_spec.yaml", path=code_path)
    module_spec = importlib.util.spec_from_file_location(
        "model_export", code_path
    )
    module = importlib.util.module_from_spec(module_spec)
    # 実験の出力先に __pycache__ を作らないように, 読み込みの間だけ止める
    sys.dont_write_bytecode, before = True, sys.dont_write_bytecode
    try:
        module_spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = before
    exported = module.Model()
    exported.load_state_dict(state)
    exported.eval()

    # 出力の差の最大
    with torch.no_grad():
        rebuilt_diff = (rebuilt(x_valid) - expected).abs().max().item()
        exported_diff = (exported(x_valid) - expected).abs().max().item()
    print(f"組み立て直したモデルとの出力の差: {rebuilt_diff:.3g}")
    print(f"書き出したコードのモデルとの出力の差: {exported_diff:.3g}")


def main() -> None:
    """
    モデルの定義から組み立てたモデルで手書き数字の分類を学習する関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # config を読み込み, 実験を始める前に定義から組み立て
    cfg, diff = load_project_config(str(TASK_DIR))
    spec_path = TASK_DIR / cfg.model.spec
    model = build_from_spec(spec_path, cfg.experiment.seed)

    # 実験を開始し, 使った定義を実験の出力先に残す
    experiment = start_experiment(cfg, diff, str(TASK_DIR))
    shutil.copy(spec_path, experiment.run_dir / "model_spec.yaml")

    # 学習
    train_loader, valid_loader, x_valid = make_loaders(cfg)
    trainer = Trainer(model, experiment, step)
    history = trainer.fit(train_loader, valid_loader)

    # ベストモデルのエポックの結果を表示
    best_epoch = trainer.checkpoint.best_epoch
    best_acc = history.get("valid/acc")[best_epoch - 1]
    print(f"保存先: {experiment.run_dir}")
    print(f"学習したエポック数: {len(history.records)}")
    print(f"ベストモデルのエポック: {best_epoch}")
    print(f"ベストモデルの検証用データの正解率: {best_acc:.4f}")

    # 出力先の定義と重みから作り直せるかの確認
    check_rebuild(experiment.run_dir, trainer.model, x_valid)


if __name__ == "__main__":
    main()
