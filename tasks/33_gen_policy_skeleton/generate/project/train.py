"""
問題の設定 (problem.yaml) から ML-Quest が書き出した学習のスクリプト

ML-EDA の画像のデータセットで, 条件の無い拡散のモデル (雑音を加えた画像
から加えた雑音を当てる) を学ぶ骨組みなので, 印 (TODO) の所を書き足す
"""
import math
from pathlib import Path
import shutil
import sys
from typing import Any, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import yaml

from ml_eda import Dataset, load_dataset
from ml_found import (
    Experiment, FrechetInceptionDistance, Trainer, load_project_config,
    save_evaluation, save_sample, start_experiment, write_task
)

# このプロジェクトのディレクトリ (ML-Found のプロジェクト)
PROJECT_DIR = Path(__file__).resolve().parent

# 問題の名前と, 作るもの
PROBLEM_NAME = "手書きの数字の画像を作る"
PREDICT = "手書きの数字の画像"

# task.yaml の受け持ち, 出力の形と作り方 (問題の設定の答え)
ROLE = "generate"
OUTPUT = "image"
METHOD = "scratch"

# テスト用の評価の名前, 説明, 指標の向き (evaluations/<名前>.json)
EVALUATION = "test_loss"
EVALUATION_TEXT = ("テスト用の分割の雑音の予測の損失 ({weights}. 時刻と雑音"
                   "は決まった乱数で選ぶ. 加えた雑音との平均二乗誤差)")
EVALUATION_BETTER = {"loss": "lower"}

# 学習の後に作る見本の件数と, 見比べる検証用の本物の画像の件数
SAMPLES = 8
REAL_SAMPLES = 4

# 検証で FID を数えるか (最良を選ぶ指標が FID のときだけ) と, FID の特徴
# の種類 (config の fid.features)
COUNT_FID = False
FID_FEATURES = ("inception", "pixels")


class PlaceholderModel(nn.Module):
    def __init__(self, channels: int, width: int):
        # 親クラスのコンストラクタを呼び出す
        super(PlaceholderModel, self).__init__()

        # 画像のチャンネル数と, 中の幅を保存
        self.channels = channels
        self.width = width

        # TODO: 課題に合うモデル (時刻の埋め込みを持つ U-Net など) に置き
        # 換える. 今は動くことを確かめるための, 時刻の埋め込みを足す 3x3 の
        # 畳み込み 3 段
        self.time_fc = nn.Linear(self.width, self.width)
        self.conv1 = nn.Conv2d(self.channels, self.width, 3, padding=1)
        self.conv2 = nn.Conv2d(self.width, self.width, 3, padding=1)
        self.conv3 = nn.Conv2d(self.width, self.channels, 3, padding=1)

        # 活性化関数の定義
        self.silu = nn.SiLU()

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        順伝播を行う関数

        Parameters
        ----------
        x: torch.Tensor
            (バッチ, チャンネル, 高さ, 幅) の雑音を加えた画像
        t: torch.Tensor
            (バッチ,) の時刻 (0 から学習の時刻の数 - 1 の整数)

        Returns
        ----------
        noise: torch.Tensor
            (バッチ, チャンネル, 高さ, 幅) の加えた雑音の予測
        """
        # 時刻の正弦波の埋め込みを, 1 段目の畳み込みの後の各チャンネルに足す
        emb = self.time_fc(time_embedding(t, self.width))
        h = self.silu(self.conv1(x) + emb[:, :, None, None])
        h = self.silu(self.conv2(h))
        noise = self.conv3(h)

        return noise


class PixelFeatures(nn.Module):
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        画素をそのまま並べて特徴にする関数 (FID が動くことを重みを取らずに
        確かめるためのもの)

        Parameters
        ----------
        images: torch.Tensor
            (N, 3, H, W) の [0, 1] の画像

        Returns
        ----------
        features: torch.Tensor
            (N, 3 * H * W) の特徴
        """
        # 平らにした画素
        features = images.flatten(1)

        return features


class DiffusionTrainer(Trainer):
    def __init__(
        self, model: nn.Module, experiment: Experiment, inputs: tuple,
        schedule: torch.Tensor, real: torch.Tensor, cfg: Any
    ):
        # 親クラスのコンストラクタを呼び出す (step を上書きするので
        # step_fn は None)
        super(DiffusionTrainer, self).__init__(model, experiment, None)

        # モデルの入力の名前 (雑音を加えた画像, 時刻), 時刻ごとの累積の
        # alpha, 損失関数, 検証の乱数の種と, 検証で毎回同じ時刻と雑音を引く
        # 乱数の生成器を保存
        self.inputs = inputs
        self.schedule = schedule
        self.mse = nn.MSELoss()
        self.seed = cfg.experiment.seed
        self.generator = torch.Generator()

        # FID の本物の画像 (検証用の分割), 作る繰り返しの回数と種, バッチの
        # 大きさと, FID を数えるもの (inception は初めて使うときに重みを
        # 取得. 数えないなら None)
        self.real = real
        self.sample_steps = cfg.diffusion.sample_steps
        self.sample_seed = cfg.diffusion.sample_seed
        self.batch_size = cfg.data.batch_size
        self.fid = None
        if COUNT_FID:
            features = (PixelFeatures() if cfg.fid.features == "pixels"
                        else None)
            self.fid = FrechetInceptionDistance(
                feature_model=features, dims=cfg.fid.dims,
                device=self.device
            )

    def on_epoch_start(self, phase: str) -> None:
        """
        検証の前に, 検証の乱数を同じ状態に戻す関数

        Parameters
        ----------
        phase: str
            train, valid, train_eval のどれか

        Returns
        ----------
        None
        """
        # 検証の損失をエポックの間で同じ時刻と雑音で比べるため
        if phase == "valid":
            self.generator.manual_seed(self.seed)

    def step(self, batch: tuple, phase: str) -> dict:
        """
        1 バッチ分の雑音の予測の損失を求める関数

        Parameters
        ----------
        batch: tuple
            (画像,) (ML-EDA の tensors. 目的の値は無い)
        phase: str
            train, valid, train_eval のどれか

        Returns
        ----------
        outputs: dict
            加えた雑音との平均二乗誤差 (loss)
        """
        # 学習は全体の乱数, 検証は決まった乱数で時刻と雑音を選ぶ. 学習は
        # 学習しているモデル, 検証は評価のモデル (EMA があれば EMA)
        x0 = batch[0]
        generator = self.generator if phase == "valid" else None
        model = self.model if phase == "train" else self.eval_model()
        loss = noise_loss(model, self.inputs, self.schedule, x0, generator)
        outputs = {"loss": loss}

        return outputs

    def on_epoch_end(self, phase: str) -> dict:
        """
        検証のときに, 作った画像と検証用の本物の画像の FID を数える関数

        Parameters
        ----------
        phase: str
            train, valid, train_eval のどれか

        Returns
        ----------
        metrics: dict
            FID を数えるときの検証なら fid (valid/fid として記録. 小さいほど
            良い). 他は空
        """
        # FID を数えるときの, 検証するエポック (trainer.valid_every_n_epochs
        # ごとと最後) だけ
        if self.fid is None or phase != "valid":
            return {}

        # 本物の特徴は 1 回だけ足して使い回す
        if not self.fid.has_real:
            for (images,) in DataLoader(TensorDataset(self.real),
                                        batch_size=self.batch_size):
                self.fid.update_real(images)

        # 作った画像は毎回入れ替える. 見本と同じ種と回数で本物と同じ件数を
        # 作り (始まりの雑音が同じなのでエポックの間で比べられる), FID が
        # 受け取る 0 から 1 に収める
        self.fid.reset_fake()
        noise = start_noise(list(self.real.shape[1:]), len(self.real),
                            self.sample_seed)
        for chunk in noise.split(self.batch_size):
            images = sample(self.eval_model(), self.inputs, self.schedule,
                            chunk, self.sample_steps)
            self.fid.update_fake(images.clamp(0, 1))
        metrics = {"fid": self.fid.compute()}

        return metrics


def time_embedding(t: torch.Tensor, width: int) -> torch.Tensor:
    """
    時刻を正弦波の埋め込みにする関数

    Parameters
    ----------
    t: torch.Tensor
        (バッチ,) の時刻
    width: int
        埋め込みの幅 (偶数)

    Returns
    ----------
    emb: torch.Tensor
        (バッチ, 幅) の sin と cos の並び
    """
    # 周波数の違う正弦波
    half = width // 2
    freqs = torch.exp(
        -math.log(10000.0)
        * torch.arange(half, device=t.device, dtype=torch.float32) / half
    )
    args = t.float()[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)

    return emb


def make_schedule(cfg: Any) -> torch.Tensor:
    """
    線形の雑音の予定から, 時刻ごとの累積の alpha を求める関数

    Parameters
    ----------
    cfg: Any
        config (diffusion の train_steps, beta_start, beta_end)

    Returns
    ----------
    schedule: torch.Tensor
        (学習の時刻の数,) の累積の alpha (ML-Inference の ddim と同じ予定)
    """
    # 1 - beta の累積の積 (推論の一式の ddim の設定と同じ値を使う)
    betas = torch.linspace(
        cfg.diffusion.beta_start, cfg.diffusion.beta_end,
        cfg.diffusion.train_steps, dtype=torch.float64
    )
    schedule = torch.cumprod(1 - betas, dim=0).float()

    return schedule


def call_model(
    model: nn.Module, inputs: tuple, x: torch.Tensor, t: torch.Tensor
) -> torch.Tensor:
    """
    雑音を加えた画像と時刻を, 入力の名前でモデルに渡す関数

    Parameters
    ----------
    model: nn.Module
        雑音を当てるモデル
    inputs: tuple
        (雑音を加えた画像の入力の名前, 時刻の入力の名前)
    x: torch.Tensor
        雑音を加えた画像
    t: torch.Tensor
        (バッチ,) の時刻

    Returns
    ----------
    noise: torch.Tensor
        加えた雑音の予測 (形は x と同じ)
    """
    # ML-ModelControl の定義の入力の名前か, 仮のモデルの引数の名前
    noise = model(**{inputs[0]: x, inputs[1]: t})

    return noise


def noise_loss(
    model: nn.Module, inputs: tuple, schedule: torch.Tensor,
    x0: torch.Tensor, generator: Optional[torch.Generator]
) -> torch.Tensor:
    """
    時刻と雑音を選んで画像に雑音を加え, 加えた雑音を当てた誤差を求める関数

    Parameters
    ----------
    model: nn.Module
        雑音を当てるモデル
    inputs: tuple
        モデルの入力の名前 (雑音を加えた画像, 時刻)
    schedule: torch.Tensor
        時刻ごとの累積の alpha
    x0: torch.Tensor
        (バッチ, チャンネル, 高さ, 幅) の元の画像 (データセットの値のまま)
    generator: Optional[torch.Generator]
        時刻と雑音を選ぶ乱数 (None なら全体の乱数)

    Returns
    ----------
    loss: torch.Tensor
        予測した雑音と加えた雑音の平均二乗誤差
    """
    # 時刻ごとの混ぜ方で, 元の画像と雑音を一度に混ぜる
    count = len(x0)
    t = torch.randint(0, len(schedule), (count,), generator=generator)
    noise = torch.randn(x0.shape, generator=generator)
    t, noise = t.to(x0.device), noise.to(x0.device)
    alpha = schedule.to(x0.device)[t][:, None, None, None]
    x_t = alpha.sqrt() * x0 + (1 - alpha).sqrt() * noise
    loss = nn.MSELoss()(call_model(model, inputs, x_t, t), noise)

    return loss


def start_noise(shape: list, count: int, seed: int) -> torch.Tensor:
    """
    作る画像の始まりの雑音を引く関数

    Parameters
    ----------
    shape: list
        1 件の形 (チャンネル, 高さ, 幅)
    count: int
        作る件数
    seed: int
        始まりの雑音の乱数の種

    Returns
    ----------
    noise: torch.Tensor
        (件数, チャンネル, 高さ, 幅) の cpu の雑音
    """
    # 推論の一式と同じく, 種の生成器から 1 件ずつ引く (先頭の件は件数に
    # よらず同じ)
    generator = torch.Generator().manual_seed(seed)
    noise = torch.cat([torch.randn((1, *shape), generator=generator)
                       for _ in range(count)])

    return noise


@torch.no_grad()
def sample(
    model: nn.Module, inputs: tuple, schedule: torch.Tensor,
    noise: torch.Tensor, steps: int
) -> torch.Tensor:
    """
    雑音から DDIM のサンプリング (決まった道) で画像を作る関数

    Parameters
    ----------
    model: nn.Module
        雑音を当てるモデル
    inputs: tuple
        モデルの入力の名前
    schedule: torch.Tensor
        時刻ごとの累積の alpha
    noise: torch.Tensor
        (件数, チャンネル, 高さ, 幅) の始まりの雑音 (start_noise)
    steps: int
        繰り返しの回数 (学習の時刻の数以下)

    Returns
    ----------
    x: torch.Tensor
        (件数, チャンネル, 高さ, 幅) の作った画像 (モデルのデバイス.
        データセットの値の尺度)
    """
    # 雑音をモデルのデバイスへ. 間引いた時刻 (大きい順) で, 見積もった元の
    # 画像から 1 つ前の時刻へ (推論の一式の ddim の eta 0 と同じ道)
    model.eval()
    device = next(model.parameters(), torch.empty(0)).device
    x = noise.to(device)
    times = torch.linspace(len(schedule) - 1, 0, steps).round().long()
    times = torch.unique_consecutive(times).tolist()
    for index, t in enumerate(times):
        now = float(schedule[t])
        before = (float(schedule[times[index + 1]])
                  if index + 1 < len(times) else 1.0)
        eps = call_model(model, inputs, x,
                         torch.full((len(x),), t, dtype=torch.long,
                                    device=device))
        x0 = (x - math.sqrt(1 - now) * eps) / math.sqrt(now)
        x = math.sqrt(before) * x0 + math.sqrt(1 - before) * eps

    return x


def noise_inputs(spec: Optional[Path]) -> tuple:
    """
    モデルの入力の名前 (雑音を加えた画像, 時刻) を決める関数

    Parameters
    ----------
    spec: Optional[Path]
        ML-ModelControl の定義 (無ければ None で, 仮のモデルの x と t)

    Returns
    ----------
    names: tuple
        (雑音を加えた画像の入力, 時刻の入力). 定義では dtype が int64 の
        入力が時刻, もう 1 つが画像

    Raises
    ----------
    ValueError
        定義の入力が 2 つでないか, int64 の入力が 1 つでない場合
    """
    # 仮のモデルは引数の名前
    if spec is None:
        return ("x", "t")
    loaded = yaml.safe_load(spec.read_text(encoding="utf-8")) or {}
    entries = loaded.get("inputs") or {}
    times = [name for name, entry in entries.items()
             if isinstance(entry, dict) and entry.get("dtype") == "int64"]
    if len(entries) != 2 or len(times) != 1:
        raise ValueError(
            "定義の入力は, 雑音を加えた画像と時刻 (dtype: int64) の 2 つに"
            f"してください (今は {list(entries)})"
        )
    names = (next(name for name in entries if name != times[0]), times[0])

    return names


def project_spec(cfg: Any) -> Optional[Path]:
    """
    config の model.spec の, プロジェクトの ML-ModelControl の定義を返す関数

    Parameters
    ----------
    cfg: Any
        config (model.spec に定義の場所. プロジェクトのディレクトリから)

    Returns
    ----------
    spec: Optional[Path]
        定義のファイル (場所を書いていないか, ファイルがまだ無ければ None)
    """
    # ML-CreateArchitecture で組んで保存するまでは無い
    spec = PROJECT_DIR / cfg.model.spec if cfg.model.spec else None
    if spec is not None and not spec.is_file():
        spec = None

    return spec


def spec_errors(spec: Optional[Path]) -> list[str]:
    """
    ML-ModelControl の定義の誤りを, 実験の出力先を作る前に確かめる関数

    Parameters
    ----------
    spec: Optional[Path]
        プロジェクトの定義 (無ければ None)

    Returns
    ----------
    errors: list[str]
        誤りの文 (定義が無いか誤りが無ければ空)
    """
    # 定義の無いうちは ML-ModelControl を読まない. 読めない定義も誤りと
    # して返る (例外にならない). 入力の名前も確かめる
    if spec is None:
        return []
    from ml_modelcontrol import validate
    errors = [issue["message"] for issue in validate(spec)
              if issue["severity"] == "error"]
    if not errors:
        try:
            noise_inputs(spec)
        except ValueError as error:
            errors.append(str(error))

    return errors


def spec_file(
    spec: Optional[Path], experiment: Experiment
) -> tuple[Optional[Path], Optional[Path]]:
    """
    モデルを組み立てる ML-ModelControl の定義のファイルと, 定義の中の相対
    の参照 (外部の重みなど) の起点を返す関数

    Parameters
    ----------
    spec: Optional[Path]
        プロジェクトの定義 (無ければ None)
    experiment: Experiment
        実験 (ML-Found が実験に写した定義の場所 model_spec_path と, 写した
        元の定義 model_spec_source)

    Returns
    ----------
    path: Optional[Path]
        ML-Found が実験に写した定義. 写せなければプロジェクトの定義. 定義
        が無ければ None
    base_dir: Optional[Path]
        相対の参照の起点 (写しなら元の定義のディレクトリ. 他は None)
    """
    # 定義が無いうちは仮のモデル
    if spec is None:
        return None, None

    # 写しの相対の参照は元の定義の場所から (写しの隣には重みが無い)
    if experiment.model_spec_path is not None:
        path = Path(experiment.model_spec_path)
        base_dir = Path(experiment.model_spec_source).parent
        return path, base_dir

    # 写しが無ければ理由を知らせて, プロジェクトの定義
    experiment.logger.warning(
        "モデルの定義を実験に写せなかったので, 学習のプロジェクトの定義から"
        "組み立てます (理由は source.yaml の model_spec の status)"
    )
    path = spec
    base_dir = None

    return path, base_dir


def make_model(
    shape: list, cfg: Any, spec: Optional[Path], base_dir: Optional[Path]
) -> nn.Module:
    """
    モデルを作る関数

    Parameters
    ----------
    shape: list
        1 件の画像の形 (チャンネル, 高さ, 幅)
    cfg: Any
        config (仮のモデルの幅と乱数の種)
    spec: Optional[Path]
        ML-ModelControl の定義のファイル (無ければ None)
    base_dir: Optional[Path]
        定義の中の相対の参照の起点 (None なら定義のディレクトリ)

    Returns
    ----------
    model: nn.Module
        定義から組み立てたモデルか, 仮のモデル
    """
    # 定義があれば組み立て (定義の無いうちは ML-ModelControl を読まない)
    if spec is not None:
        from ml_modelcontrol import build_model
        model = build_model(spec, seed=cfg.experiment.seed,
                            base_dir=base_dir)
        return model
    model = PlaceholderModel(shape[0], cfg.model.width)

    return model


def evaluate_test(
    trainer: DiffusionTrainer, x_test: torch.Tensor, cfg: Any,
    experiment: Experiment
) -> list[dict]:
    """
    テスト用の分割の雑音の予測の損失を記録する関数

    Parameters
    ----------
    trainer: DiffusionTrainer
        学習した後の Trainer
    x_test: torch.Tensor
        テスト用の画像
    cfg: Any
        config (バッチの大きさと乱数の種)
    experiment: Experiment
        実験 (train.log に記録する)

    Returns
    ----------
    rows: list[dict]
        loss (件数で重みを付けた平均) と件数 (rows) の 1 行
    """
    # 学習と同じ数え方を, 検証と別の決まった乱数で
    model = trainer.model.cpu().eval()
    generator = torch.Generator().manual_seed(cfg.experiment.seed + 1)
    total = 0.0
    with torch.no_grad():
        for (x0,) in DataLoader(TensorDataset(x_test),
                                batch_size=cfg.data.batch_size):
            loss = noise_loss(model, trainer.inputs, trainer.schedule, x0,
                              generator)
            total += loss.item() * len(x0)
    rows = [{"loss": total / len(x_test), "rows": len(x_test)}]
    experiment.logger.info(f"テスト用の損失: {rows[0]['loss']:.4f}")

    return rows


def save_samples(
    trainer: DiffusionTrainer, x_valid: torch.Tensor, shape: list, cfg: Any,
    experiment: Experiment
) -> None:
    """
    学習した後の重みで作った画像と, 見比べる検証用の本物の画像を見本に
    書く関数

    Parameters
    ----------
    trainer: DiffusionTrainer
        学習した後の Trainer
    x_valid: torch.Tensor
        検証用の画像
    shape: list
        1 件の形
    cfg: Any
        config (作る繰り返しの回数と乱数の種)
    experiment: Experiment
        実験 (samples/ に書く)

    Returns
    ----------
    None
    """
    # 推論の一式と同じ種と回数で作り, 0 から 1 に収めて画像に
    model = trainer.model.cpu().eval()
    noise = start_noise(shape, SAMPLES, cfg.diffusion.sample_seed)
    images = sample(model, trainer.inputs, trainer.schedule, noise,
                    cfg.diffusion.sample_steps)
    for index, image in enumerate(images.clamp(0, 1)):
        save_sample(experiment, f"generated_{index}", image.numpy(),
                    kind="image", caption=f"作った画像 {index} (種 "
                    f"{cfg.diffusion.sample_seed} の {index} 件目)")
    for index in range(min(REAL_SAMPLES, len(x_valid))):
        save_sample(experiment, f"valid_{index}_real",
                    x_valid[index].numpy(), kind="image", split="valid",
                    caption=f"見比べる検証用の本物の画像 {index}")


def make_inference(
    cfg: Any, spec: Optional[Path], shape: list, run_dir: Path,
    experiment: Experiment
) -> None:
    """
    学習した重みから, 実験の中に ML-Inference の雑音だけの一式を作る関数

    Parameters
    ----------
    cfg: Any
        config (一式の名前と ddim の設定)
    spec: Optional[Path]
        ML-ModelControl の定義のファイル (無ければ None)
    shape: list
        1 件の形 (チャンネル, 高さ, 幅)
    run_dir: Path
        実験のディレクトリ
    experiment: Experiment
        実験 (train.log に記録する)

    Returns
    ----------
    None
    """
    # 仮のモデルは ML-ModelControl の定義が無いので一式にできない
    if spec is None:
        experiment.logger.info(
            "仮のモデル (PlaceholderModel) は ML-ModelControl の定義が無い"
            "ので, 推論の一式を作りません. config の model.spec の場所に"
            "定義を置くと, 学習の後に作ります"
        )
        return

    # 学習と同じ雑音の予定の ddim で, データセットを持たない一式 (件数と
    # 種で作る)
    from ml_inference import make_bundle
    state, time = noise_inputs(spec)
    procedure = {
        "name": "ddim", "steps": cfg.diffusion.sample_steps,
        "seed": cfg.diffusion.sample_seed,
        "options": {"train_steps": cfg.diffusion.train_steps,
                    "beta_start": cfg.diffusion.beta_start,
                    "beta_end": cfg.diffusion.beta_end, "eta": 0.0},
    }
    info = make_bundle(
        run_dir / "inference", spec, None, run_dir, role=ROLE, form=OUTPUT,
        inputs={state: {"source": "state"}, time: {"source": "time"}},
        output_options={"shape": list(shape)}, procedure=procedure,
        name=cfg.experiment.name
    )
    experiment.logger.info(
        "推論の一式 (雑音だけ): inference/ (件数と種で画像を作る. 版 "
        f"{info['format_version']})"
    )
    for note in info["notes"]:
        experiment.logger.warning(f"推論の一式の注意: {note}")


def main() -> None:
    """
    ML-EDA の画像のデータセットで拡散のモデルを学習し, テスト用の評価, 見本
    と推論の一式を残す関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # config とデータセット (無ければ作り方の案内)
    cfg, diff = load_project_config(str(PROJECT_DIR))
    path = PROJECT_DIR / cfg.data.dataset
    if not (path / "dataset.yaml").is_file():
        print(f"{path} がありません. README の「次にすること」の 1 で, "
              "データセットを書き出してください")
        sys.exit(1)
    dataset: Dataset = load_dataset(path)
    if dataset.info.get("kind") != "image":
        print(f"{path} は画像のデータセットではありません (種類 "
              f"{dataset.info.get('kind')})")
        sys.exit(1)
    if "valid" not in dataset.frames:
        print("データセットに検証用の分割がありません. 最も良いモデルを選ぶ"
              "ために, 検証用の割合を 0 より大きくして書き出してください")
        sys.exit(1)

    # 画像 (データセットの 0 から 1 の値のまま学ぶ. 作る画像も同じ尺度).
    # FID は共分散を求めるので, 数えるなら検証用の画像が 2 件以上要る
    x_train = dataset.tensors("train")[0]
    x_valid = dataset.tensors("valid")[0]
    shape = list(x_train.shape[1:])
    if COUNT_FID and len(x_valid) < 2:
        print("検証用の画像が 2 件より少ないので FID を数えられません. 検証用"
              "の割合を大きくして書き出してください")
        sys.exit(1)

    # 繰り返しの回数と, モデルの定義の確かめ (誤りなら出力先を作る前に
    # 止まる)
    if not 1 <= cfg.diffusion.sample_steps <= cfg.diffusion.train_steps:
        print("config の diffusion.sample_steps は 1 から "
              f"diffusion.train_steps ({cfg.diffusion.train_steps}) に"
              f"してください (今は {cfg.diffusion.sample_steps})")
        sys.exit(1)
    if COUNT_FID and cfg.fid.features not in FID_FEATURES:
        print(f"config の fid.features は {' か '.join(FID_FEATURES)} に"
              f"してください (今は {cfg.fid.features})")
        sys.exit(1)
    spec = project_spec(cfg)
    errors = spec_errors(spec)
    if errors:
        print(f"{spec} の定義に誤りがあります. README の「次にすること」の "
              "2 で直してください")
        print("\n".join(f"- {message}" for message in errors))
        sys.exit(1)

    # 何の実験か (受け持ち, 出力の形, 作るもの, 作り方は問題の設定から).
    # TODO: 既製のモデルを元にするなら, 元にした実験か重みを task["base"]
    experiment = start_experiment(cfg, diff, str(PROJECT_DIR))
    write_task(experiment, {"role": ROLE, "output": OUTPUT,
                            "what": PREDICT, "method": METHOD})

    # 画像の件数と学習と検証の loader
    experiment.logger.info(
        f"問題: {PROBLEM_NAME}. 作るもの: {PREDICT}. 1 件の画像の形 (チャンネル, "
        f"高さ, 幅): {tuple(shape)}. 学習用 {len(x_train)} 件, 検証用 "
        f"{len(x_valid)} 件"
    )
    train_loader = DataLoader(TensorDataset(x_train),
                              batch_size=cfg.data.batch_size, shuffle=True)
    valid_loader = DataLoader(TensorDataset(x_valid),
                              batch_size=cfg.data.batch_size)

    # モデルの組み立てと学習 (定義がまだ無ければ仮のモデル. 重みの指数
    # 移動平均は ML-Found の trainer.ema)
    spec, base_dir = spec_file(spec, experiment)
    if spec is None:
        experiment.logger.info(
            f"モデルの定義 ({cfg.model.spec or '場所なし'}) が無いので, "
            "仮のモデル (PlaceholderModel) で学習します. モデルの作成・選定の段階で組んだ定義を"
            "その場所に置くと, 定義のモデルで学習します"
        )
    model = make_model(shape, cfg, spec, base_dir)
    trainer = DiffusionTrainer(model, experiment, noise_inputs(spec),
                               make_schedule(cfg), x_valid, cfg)
    trainer.fit(train_loader, valid_loader)

    # 実験だけで見直せるように, 問題の設定の写し (元の問題の場所
    # origin も残る. モデルの定義は ML-Found が start_experiment で写す)
    run_dir = Path(experiment.run_dir)
    shutil.copy(PROJECT_DIR / "problem.yaml", run_dir / "problem.yaml")

    # 結果の表示. 学習した後の重みは, 最良のモデルに戻した場合だけ最良の
    # エポック (select_best: false などは最後)
    best = trainer.final_weights == "best"
    epoch = trainer.checkpoint.best_epoch if best else trainer.epoch
    weights = "最良のモデル" if best else "最後のエポックの重み"
    print(f"保存先: {run_dir}")
    print(f"評価する重み: {weights} (エポック {epoch})")
    if best:
        print(f"ベストの {cfg.trainer.monitor}: "
              f"{trainer.checkpoint.best_value}")

    # テスト用の分割があれば, 学習した後の重みの損失を evaluations/ へ保存
    if "test" in dataset.frames:
        rows = evaluate_test(trainer, dataset.tensors("test")[0], cfg,
                             experiment)
        save_evaluation(experiment, EVALUATION, rows, "test",
                        description=EVALUATION_TEXT.format(weights=weights),
                        epoch=epoch, better=EVALUATION_BETTER)
    else:
        experiment.logger.info("テスト用の分割が無いので, 評価しません")

    # 作った画像と本物の画像の見本 (ML-Dashboard の見本のタブ). 書けなく
    # ても学習の結果は残すので, 警告だけ
    try:
        save_samples(trainer, x_valid, shape, cfg, experiment)
    except ValueError as error:
        experiment.logger.warning(f"見本を書けません: {error}")

    # 実験の中に雑音だけの推論の一式 (ML-ModelControl の定義がある場合
    # だけ)
    make_inference(cfg, spec, shape, run_dir, experiment)


if __name__ == "__main__":
    main()
