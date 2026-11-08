import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self, model_type: str, hidden_dim: int, latent_dim: int):
        # 親クラスのコンストラクタを呼び出す
        super(Encoder, self).__init__()

        # 構造の種類, 隠れ層の次元数, 潜在変数の次元数を保存
        self.model_type = model_type
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # 画像から隠れ層への部分の定義, mlp は OddNN の VAE と同じ全結合
        self.flatten = nn.Flatten()
        if self.model_type == "mlp":
            self.conv1 = None
            self.conv2 = None
            self.linear = nn.Linear(28 * 28, self.hidden_dim)
        elif self.model_type == "cnn":
            self.conv1 = nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1)
            self.conv2 = nn.Conv2d(
                32, 64, kernel_size=4, stride=2, padding=1
            )
            self.linear = nn.Linear(64 * 7 * 7, self.hidden_dim)
        else:
            raise ValueError(
                f"model.type は mlp か cnn にしてください: {self.model_type}"
            )

        # 事後分布の平均と対数分散を出力する層の定義
        self.linear_mu = nn.Linear(self.hidden_dim, self.latent_dim)
        self.linear_var = nn.Linear(self.hidden_dim, self.latent_dim)

        # 活性化関数の定義
        self.relu = nn.ReLU()

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        順伝播を行う関数

        Parameters
        ----------
        x: torch.Tensor
            画像, 形状は (バッチ, 1, 28, 28)

        Returns
        ----------
        mu: torch.Tensor
            事後分布の平均 μ
        log_var: torch.Tensor
            事後分布の対数分散 log σ^2
        z: torch.Tensor
            潜在変数
        """
        # cnn の場合は畳み込みで 28 → 14 → 7 に縮小
        h = x
        if self.model_type == "cnn":
            h = self.relu(self.conv1(h))
            h = self.relu(self.conv2(h))

        # 事後分布の平均・分散を計算
        h = self.relu(self.linear(self.flatten(h)))
        mu = self.linear_mu(h)
        log_var = self.linear_var(h)

        # 潜在変数を reparameterization trick でサンプリング
        eps = torch.randn_like(mu)
        z = mu + torch.exp(log_var / 2) * eps

        return mu, log_var, z


class Decoder(nn.Module):
    def __init__(self, model_type: str, hidden_dim: int, latent_dim: int):
        # 親クラスのコンストラクタを呼び出す
        super(Decoder, self).__init__()

        # 構造の種類, 隠れ層の次元数, 潜在変数の次元数を保存
        self.model_type = model_type
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # 潜在変数から隠れ層への線形層の定義
        self.linear = nn.Linear(self.latent_dim, self.hidden_dim)

        # 隠れ層から画像への部分の定義
        if self.model_type == "mlp":
            self.linear_out = nn.Linear(self.hidden_dim, 28 * 28)
            self.deconv1 = None
            self.deconv2 = None
        elif self.model_type != "cnn":
            raise ValueError(
                f"model.type は mlp か cnn にしてください: {self.model_type}"
            )
        else:
            self.linear_out = nn.Linear(self.hidden_dim, 64 * 7 * 7)
            self.deconv1 = nn.ConvTranspose2d(
                64, 32, kernel_size=4, stride=2, padding=1
            )
            self.deconv2 = nn.ConvTranspose2d(
                32, 1, kernel_size=4, stride=2, padding=1
            )

        # 活性化関数の定義
        # OddNN の Decoder は self.relu を定義せずに使っていたので追加
        self.relu = nn.ReLU()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        順伝播を行う関数

        Parameters
        ----------
        z: torch.Tensor
            潜在変数

        Returns
        ----------
        logits: torch.Tensor
            画素ごとのロジット, 形状は (バッチ, 1, 28, 28)
        """
        # 潜在変数を隠れ層に展開
        h = self.relu(self.linear(z))

        # mlp は全結合で 784 次元に, cnn は転置畳み込みで 7 → 14 → 28 に拡大
        # 損失を安定して計算できるように Sigmoid は掛けずにロジットのまま出力
        if self.model_type == "mlp":
            logits = self.linear_out(h).view(-1, 1, 28, 28)
        else:
            h = self.relu(self.linear_out(h)).view(-1, 64, 7, 7)
            h = self.relu(self.deconv1(h))
            logits = self.deconv2(h)

        return logits


class VAE(nn.Module):
    def __init__(self, model_type: str, hidden_dim: int, latent_dim: int):
        # 親クラスのコンストラクタを呼び出す
        super(VAE, self).__init__()

        # 構造の種類, 隠れ層の次元数, 潜在変数の次元数を保存
        self.model_type = model_type
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # エンコーダとデコーダの定義
        self.encoder = Encoder(
            self.model_type, self.hidden_dim, self.latent_dim
        )
        self.decoder = Decoder(
            self.model_type, self.hidden_dim, self.latent_dim
        )

        # 生成時にロジットを画素値に変換する活性化関数の定義
        self.sigmoid = nn.Sigmoid()

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        順伝播を行う関数

        Parameters
        ----------
        x: torch.Tensor
            画像

        Returns
        ----------
        outputs: torch.Tensor
            再構成した画像のロジット
        mu: torch.Tensor
            事後分布の平均 μ
        log_var: torch.Tensor
            事後分布の対数分散 log σ^2
        z: torch.Tensor
            潜在変数
        """
        # エンコーダでの計算
        mu, log_var, z = self.encoder(x)

        # デコーダでの計算
        outputs = self.decoder(z)

        return outputs, mu, log_var, z

    def encode(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        エンコードのみを行う関数

        Parameters
        ----------
        x: torch.Tensor
            画像

        Returns
        ----------
        mu: torch.Tensor
            事後分布の平均 μ
        log_var: torch.Tensor
            事後分布の対数分散 log σ^2
        z: torch.Tensor
            潜在変数
        """
        # エンコーダでの計算
        mu, log_var, z = self.encoder(x)

        return mu, log_var, z

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
            画素値を 0 から 1 にした画像
        """
        # デコーダのロジットを画素値に変換
        outputs = self.sigmoid(self.decoder(z))

        return outputs
