import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        # 親クラスのコンストラクタを呼び出す
        super(ConvBlock, self).__init__()

        # 入力と出力のチャンネル数を保存
        self.in_channels = in_channels
        self.out_channels = out_channels

        # 3x3 の畳み込みを 2 段
        self.conv1 = nn.Conv2d(
            self.in_channels, self.out_channels, 3, padding=1
        )
        self.conv2 = nn.Conv2d(
            self.out_channels, self.out_channels, 3, padding=1
        )
        self.norm1 = nn.BatchNorm2d(self.out_channels)
        self.norm2 = nn.BatchNorm2d(self.out_channels)

        # 活性化関数の定義
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        順伝播を行う関数

        Parameters
        ----------
        x: torch.Tensor
            (バッチ, 入力のチャンネル数, 高さ, 幅) の特徴

        Returns
        ----------
        h: torch.Tensor
            (バッチ, 出力のチャンネル数, 高さ, 幅) の特徴
        """
        # 畳み込み, 正規化, 活性化を 2 回
        h = self.relu(self.norm1(self.conv1(x)))
        h = self.relu(self.norm2(self.conv2(h)))

        return h


class TinyUNet(nn.Module):
    def __init__(self, in_channels: int, channels: int, num_classes: int):
        # 親クラスのコンストラクタを呼び出す
        super(TinyUNet, self).__init__()

        # 入力のチャンネル数, 幅, クラス数を保存
        self.in_channels = in_channels
        self.channels = channels
        self.num_classes = num_classes

        # 1 段だけ縮めて戻す U-Net (縮める前の特徴を戻した後につなぐ)
        self.down = ConvBlock(self.in_channels, self.channels)
        self.pool = nn.MaxPool2d(2)
        self.middle = ConvBlock(self.channels, self.channels * 2)
        self.up = nn.ConvTranspose2d(
            self.channels * 2, self.channels, 2, stride=2
        )
        self.merge = ConvBlock(self.channels * 2, self.channels)
        self.head = nn.Conv2d(self.channels, self.num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        順伝播を行う関数

        Parameters
        ----------
        x: torch.Tensor
            (バッチ, チャンネル, 高さ, 幅) の 0 から 1 の画像. 高さと幅は
            偶数

        Returns
        ----------
        logits: torch.Tensor
            (バッチ, クラス数, 高さ, 幅) の画素ごとのスコア
        """
        # 縮めた特徴を戻し, 縮める前の特徴とつないで画素ごとに分類
        skip = self.down(x)
        h = self.up(self.middle(self.pool(skip)))
        h = self.merge(torch.cat([skip, h], dim=1))
        logits = self.head(h)

        return logits
