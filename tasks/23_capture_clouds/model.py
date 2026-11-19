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
        # 点ごとの特徴と, 点の並びに依らない最大値でのまとめ
        h = x.transpose(1, 2)
        h = self.relu(self.conv1(h))
        h = self.relu(self.conv2(h))
        h = self.relu(self.conv3(h))
        feature = h.max(dim=2).values

        return feature


class PointDecoder(nn.Module):
    def __init__(self, in_features: int, hidden: int, num_points: int):
        # 親クラスのコンストラクタを呼び出す
        super(PointDecoder, self).__init__()

        # 入力の特徴の数, 中間の幅, 出す点の数を保存
        self.in_features = in_features
        self.hidden = hidden
        self.num_points = num_points

        # 決まった数の点の座標を出す全結合の層
        self.fc1 = nn.Linear(self.in_features, self.hidden)
        self.fc2 = nn.Linear(self.hidden, self.hidden)
        self.fc3 = nn.Linear(self.hidden, self.num_points * 3)

        # 活性化関数の定義
        self.relu = nn.ReLU()

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        """
        順伝播を行う関数

        Parameters
        ----------
        feature: torch.Tensor
            (バッチ, 入力の特徴の数) の点群全体の特徴

        Returns
        ----------
        points: torch.Tensor
            (バッチ, 出す点の数, 3) の座標
        """
        # 特徴から座標を並べた値と, 点ごとの形への戻し
        h = self.relu(self.fc1(feature))
        h = self.relu(self.fc2(h))
        points = self.fc3(h).view(-1, self.num_points, 3)

        return points


class CloudPredictor(nn.Module):
    def __init__(self, channels: int, hidden: int, num_points: int):
        # 親クラスのコンストラクタを呼び出す
        super(CloudPredictor, self).__init__()

        # チャンネル数, 中間の幅, 出す点の数を保存
        self.channels = channels
        self.hidden = hidden
        self.num_points = num_points

        # 深度から戻した点の特徴と, 決まった数の点を出す部分
        self.encoder = PointEncoder(self.channels)
        self.decoder = PointDecoder(
            self.channels * 4, self.hidden, self.num_points
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        順伝播を行う関数

        Parameters
        ----------
        x: torch.Tensor
            (バッチ, 入力の点の数, 3) の, 深度から世界に戻した点
            (正規化した座標. メートル)

        Returns
        ----------
        points: torch.Tensor
            (バッチ, 出す点の数, 3) の予測した点群
        """
        # 入力の点の特徴からの点群
        points = self.decoder(self.encoder(x))

        return points
