import torch
from torch import nn


class PointEncoder(nn.Module):
    def __init__(self, channels: int):
        # 親クラスのコンストラクタを呼び出す
        super(PointEncoder, self).__init__()

        # チャンネル数を保存
        self.channels = channels

        # 全ての点に同じ重みをかける層 (1x1 の畳み込み) を 3 段
        self.conv1 = nn.Conv1d(3, self.channels, 1)
        self.conv2 = nn.Conv1d(self.channels, self.channels * 2, 1)
        self.conv3 = nn.Conv1d(self.channels * 2, self.channels * 4, 1)
        self.norm1 = nn.BatchNorm1d(self.channels)
        self.norm2 = nn.BatchNorm1d(self.channels * 2)
        self.norm3 = nn.BatchNorm1d(self.channels * 4)

        # 活性化関数の定義
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        順伝播を行う関数

        Parameters
        ----------
        x: torch.Tensor
            (バッチ, 点の数, 3) の座標

        Returns
        ----------
        feature: torch.Tensor
            (バッチ, チャンネル数 * 4) の点群全体の特徴
        """
        # 点ごとに特徴を作り, 点の並びに依らないように最大値でまとめる
        h = x.transpose(1, 2)
        h = self.relu(self.norm1(self.conv1(h)))
        h = self.relu(self.norm2(self.conv2(h)))
        h = self.relu(self.norm3(self.conv3(h)))
        feature = h.max(dim=2).values

        return feature


class PointClassifier(nn.Module):
    def __init__(self, channels: int, num_classes: int):
        # 親クラスのコンストラクタを呼び出す
        super(PointClassifier, self).__init__()

        # チャンネル数とクラス数を保存
        self.channels = channels
        self.num_classes = num_classes

        # 点群の特徴と, 出力層までの全結合の層
        self.encoder = PointEncoder(self.channels)
        self.fc1 = nn.Linear(self.channels * 4, self.channels * 2)
        self.fc2 = nn.Linear(self.channels * 2, self.num_classes)

        # 活性化関数の定義
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        順伝播を行う関数

        Parameters
        ----------
        x: torch.Tensor
            (バッチ, 点の数, 3) の正規化した座標

        Returns
        ----------
        logits: torch.Tensor
            クラスごとのスコア
        """
        # 点群全体の特徴からクラスごとのスコアを計算
        h = self.relu(self.fc1(self.encoder(x)))
        logits = self.fc2(h)

        return logits
