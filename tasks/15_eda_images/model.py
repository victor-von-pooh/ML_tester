import torch
from torch import nn


class ShapeCNN(nn.Module):
    def __init__(self, in_channels: int, channels: int, num_classes: int):
        # 親クラスのコンストラクタを呼び出す
        super(ShapeCNN, self).__init__()

        # 入力のチャンネル数, 畳み込みのチャンネル数, クラス数を保存
        self.in_channels = in_channels
        self.channels = channels
        self.num_classes = num_classes

        # 畳み込みとプーリングを 2 段 (32x32 → 8x8)
        self.conv1 = nn.Conv2d(self.in_channels, self.channels, 3, padding=1)
        self.conv2 = nn.Conv2d(self.channels, self.channels * 2, 3, padding=1)
        self.pool = nn.MaxPool2d(2)

        # 全体の平均と出力層
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(self.channels * 2, self.num_classes)

        # 活性化関数の定義
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        順伝播を行う関数

        Parameters
        ----------
        x: torch.Tensor
            (バッチ, チャンネル, 高さ, 幅) の正規化した画像

        Returns
        ----------
        logits: torch.Tensor
            クラスごとのスコア
        """
        # 畳み込みで特徴を取り出し, 平均して 1 つのベクトルに
        h = self.pool(self.relu(self.conv1(x)))
        h = self.pool(self.relu(self.conv2(h)))
        h = self.gap(h).flatten(1)

        # 出力層でクラスごとのスコアを計算
        logits = self.fc(h)

        return logits
