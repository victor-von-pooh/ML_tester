import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self, latent_dim: int):
        # 親クラスのコンストラクタを呼び出す
        super(Encoder, self).__init__()

        # 潜在変数の次元数を保存
        self.latent_dim = latent_dim

        # 縦横を半分にする畳み込み層を 2 回と, 潜在変数への線形層の定義
        self.conv1 = nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(64 * 7 * 7, self.latent_dim)

        # 活性化関数の定義
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        順伝播を行う関数

        Parameters
        ----------
        x: torch.Tensor
            画像, 形状は (バッチ, 1, 28, 28)

        Returns
        ----------
        z: torch.Tensor
            潜在変数, 形状は (バッチ, latent_dim)
        """
        # 畳み込みで 28 → 14 → 7 に縮小
        h = self.relu(self.conv1(x))
        h = self.relu(self.conv2(h))

        # 平らにして潜在変数に圧縮
        z = self.fc(self.flatten(h))

        return z


class Decoder(nn.Module):
    def __init__(self, latent_dim: int):
        # 親クラスのコンストラクタを呼び出す
        super(Decoder, self).__init__()

        # 潜在変数の次元数を保存
        self.latent_dim = latent_dim

        # 潜在変数から特徴マップへの線形層と, 縦横を 2 倍にする転置畳み込み層の定義
        self.fc = nn.Linear(self.latent_dim, 64 * 7 * 7)
        self.unflatten = nn.Unflatten(1, (64, 7, 7))
        self.deconv1 = nn.ConvTranspose2d(
            64, 32, kernel_size=4, stride=2, padding=1
        )
        self.deconv2 = nn.ConvTranspose2d(
            32, 1, kernel_size=4, stride=2, padding=1
        )

        # 活性化関数の定義
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        順伝播を行う関数

        Parameters
        ----------
        z: torch.Tensor
            潜在変数, 形状は (バッチ, latent_dim)

        Returns
        ----------
        outputs: torch.Tensor
            再構成した画像, 画素値は 0 から 1, 形状は (バッチ, 1, 28, 28)
        """
        # 潜在変数を 7×7 の特徴マップに展開
        h = self.unflatten(self.relu(self.fc(z)))

        # 転置畳み込みで 7 → 14 → 28 に拡大し, 画素値を 0 から 1 に変換
        h = self.relu(self.deconv1(h))
        outputs = self.sigmoid(self.deconv2(h))

        return outputs


class AutoEncoder(nn.Module):
    def __init__(self, latent_dim: int):
        # 親クラスのコンストラクタを呼び出す
        super(AutoEncoder, self).__init__()

        # 潜在変数の次元数を保存
        self.latent_dim = latent_dim

        # エンコーダとデコーダの定義
        self.encoder = Encoder(self.latent_dim)
        self.decoder = Decoder(self.latent_dim)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        順伝播を行う関数

        Parameters
        ----------
        x: torch.Tensor
            画像, 形状は (バッチ, 1, 28, 28)

        Returns
        ----------
        outputs: torch.Tensor
            再構成した画像
        z: torch.Tensor
            潜在変数
        """
        # エンコーダとデコーダを順に適用
        z = self.encoder(x)
        outputs = self.decoder(z)

        return outputs, z

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        エンコードのみを行う関数

        Parameters
        ----------
        x: torch.Tensor
            画像

        Returns
        ----------
        z: torch.Tensor
            潜在変数
        """
        # エンコーダでの計算
        z = self.encoder(x)

        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        デコードのみを行う関数

        Parameters
        ----------
        z: torch.Tensor
            潜在変数

        Returns
        ----------
        outputs: torch.Tensor
            再構成した画像
        """
        # デコーダでの計算
        outputs = self.decoder(z)

        return outputs
