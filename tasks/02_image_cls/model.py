import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        # 親クラスのコンストラクタを呼び出す
        super(ConvBlock, self).__init__()

        # 入力と出力のチャネル数を保存
        self.in_channels = in_channels
        self.out_channels = out_channels

        # 畳み込み層, 正規化層, プーリング層の定義
        self.conv = nn.Conv2d(
            self.in_channels, self.out_channels, kernel_size=3, padding=1
        )
        self.bn = nn.BatchNorm2d(self.out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2)

        # 活性化関数の定義
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        順伝播を行う関数

        Parameters
        ----------
        x: torch.Tensor
            入力の特徴マップ, 形状は (バッチ, チャネル, 高さ, 幅)

        Returns
        ----------
        out: torch.Tensor
            高さと幅を半分にした特徴マップ
        """
        # 畳み込みと正規化の後に活性化関数を適用し, 縦横を半分に縮小
        h = self.relu(self.bn(self.conv(x)))
        out = self.pool(h)

        return out


class FashionCNN(nn.Module):
    def __init__(
        self, in_channels: int, image_size: int, channels: list[int],
        hidden_dim: int, num_classes: int
    ):
        # 親クラスのコンストラクタを呼び出す
        super(FashionCNN, self).__init__()

        # 入力のチャネル数と画像の大きさを保存
        self.in_channels = in_channels
        self.image_size = image_size

        # 各ブロックのチャネル数, 全結合層の次元数, クラス数を保存
        self.channels = channels
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        # 畳み込みブロックを順に定義
        self.blocks = nn.ModuleList()
        in_ch = self.in_channels
        for out_ch in self.channels:
            self.blocks.append(ConvBlock(in_ch, out_ch))
            in_ch = out_ch

        # ブロックごとに縦横が半分になるので, 平らにした後の次元数を計算
        feature_size = self.image_size // 2 ** len(self.channels)
        flat_dim = self.channels[-1] * feature_size ** 2

        # 全結合層の定義
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(flat_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.num_classes)

        # 活性化関数の定義
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        順伝播を行う関数

        Parameters
        ----------
        x: torch.Tensor
            標準化した画像, 形状は (バッチ, 1, 28, 28)

        Returns
        ----------
        logits: torch.Tensor
            クラスごとのスコア
        """
        # 畳み込みブロックで特徴を抽出
        h = x
        for block in self.blocks:
            h = block(h)

        # 平らにして全結合層でクラスごとのスコアを計算
        h = self.relu(self.fc1(self.flatten(h)))
        logits = self.fc2(h)

        return logits
