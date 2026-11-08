import torch
from torch import nn


class Generator(nn.Module):
    def __init__(self, noise_dim: int, channels: int):
        # 親クラスのコンストラクタを呼び出す
        super(Generator, self).__init__()

        # ノイズの次元数とチャネル数の基準を保存
        self.noise_dim = noise_dim
        self.channels = channels

        # ノイズから 7×7 の特徴マップへの線形層の定義
        self.fc = nn.Linear(self.noise_dim, 2 * self.channels * 7 * 7)
        self.bn0 = nn.BatchNorm1d(2 * self.channels * 7 * 7)
        self.unflatten = nn.Unflatten(1, (2 * self.channels, 7, 7))

        # 縦横を 2 倍にする転置畳み込み層の定義
        self.deconv1 = nn.ConvTranspose2d(
            2 * self.channels, self.channels, kernel_size=4, stride=2,
            padding=1
        )
        self.bn1 = nn.BatchNorm2d(self.channels)
        self.deconv2 = nn.ConvTranspose2d(
            self.channels, 1, kernel_size=4, stride=2, padding=1
        )

        # 活性化関数の定義
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        順伝播を行う関数

        Parameters
        ----------
        z: torch.Tensor
            ノイズ, 形状は (バッチ, noise_dim)

        Returns
        ----------
        images: torch.Tensor
            生成した画像, 画素値は -1 から 1, 形状は (バッチ, 1, 28, 28)
        """
        # ノイズを 7×7 の特徴マップに展開
        h = self.unflatten(self.relu(self.bn0(self.fc(z))))

        # 7 → 14 → 28 に拡大し, 画素値を -1 から 1 に変換
        h = self.relu(self.bn1(self.deconv1(h)))
        images = self.tanh(self.deconv2(h))

        return images


class Discriminator(nn.Module):
    def __init__(self, channels: int):
        # 親クラスのコンストラクタを呼び出す
        super(Discriminator, self).__init__()

        # チャネル数の基準を保存
        self.channels = channels

        # 縦横を半分にする畳み込み層の定義
        self.conv1 = nn.Conv2d(
            1, self.channels, kernel_size=4, stride=2, padding=1
        )
        self.conv2 = nn.Conv2d(
            self.channels, 2 * self.channels, kernel_size=4, stride=2,
            padding=1
        )
        self.bn2 = nn.BatchNorm2d(2 * self.channels)

        # 本物らしさのロジットを出力する線形層の定義
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(2 * self.channels * 7 * 7, 1)

        # 活性化関数の定義
        self.lrelu = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        順伝播を行う関数

        Parameters
        ----------
        x: torch.Tensor
            画像, 画素値は -1 から 1

        Returns
        ----------
        logits: torch.Tensor
            本物らしさのロジット, 形状は (バッチ,)
        """
        # 28 → 14 → 7 に縮小
        h = self.lrelu(self.conv1(x))
        h = self.lrelu(self.bn2(self.conv2(h)))

        # 1 つのロジットに集約
        logits = self.fc(self.flatten(h)).squeeze(-1)

        return logits


class GAN(nn.Module):
    def __init__(self, noise_dim: int, g_channels: int, d_channels: int):
        # 親クラスのコンストラクタを呼び出す
        super(GAN, self).__init__()

        # ノイズの次元数を保存
        self.noise_dim = noise_dim

        # 生成器と識別器の定義
        self.generator = Generator(self.noise_dim, g_channels)
        self.discriminator = Discriminator(d_channels)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        順伝播を行う関数

        Parameters
        ----------
        z: torch.Tensor
            ノイズ

        Returns
        ----------
        images: torch.Tensor
            生成した画像
        """
        # 生成器で画像を生成
        images = self.generator(z)

        return images
