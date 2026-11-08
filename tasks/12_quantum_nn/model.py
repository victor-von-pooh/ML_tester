from typing import Literal, Optional

from qiskit import QuantumCircuit
from qiskit.circuit.library import real_amplitudes, zz_feature_map
from qiskit.primitives import StatevectorEstimator as Estimator
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.neural_networks import EstimatorQNN, SamplerQNN
from qiskit_machine_learning.utils import algorithm_globals
import torch
from torch import nn


def parity(bits: int) -> int:
    """
    測定結果のビット列の 1 の数の偶奇を求める関数

    Parameters
    ----------
    bits: int
        測定結果のビット列を表す整数

    Returns
    ----------
    label: int
        1 の数が偶数なら 0, 奇数なら 1
    """
    # 1 の数を数えて 2 で割った余りを計算
    label = bin(bits).count("1") % 2

    return label


class QEstimator(nn.Module):
    def __init__(
        self, qcs: list[QuantumCircuit], seed: Optional[int],
        observables: list[SparsePauliOp], input_gradients: bool = False
    ):
        # 親クラスのコンストラクタを呼び出す
        super(QEstimator, self).__init__()

        # 量子回路のリストを取得
        self.qc = qcs[0]
        self.feature_map = qcs[1]
        self.ansatz = qcs[2]

        # 量子層の定義
        self.qnn_est = self.set_estimator(
            seed, observables, input_gradients
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        順伝播を行う関数

        Parameters
        ----------
        x: torch.Tensor
            入力データ

        Returns
        ----------
        out: torch.Tensor
            観測量の期待値, 形状は (バッチ, 観測量の数)
        """
        # 量子層の順伝播計算
        out = self.qnn_est(x)

        return out

    def set_estimator(
        self, seed: Optional[int], observables: list[SparsePauliOp],
        input_gradients: bool = False
    ) -> TorchConnector:
        """
        Estimator の QNN を用意する関数

        Parameters
        ----------
        seed: Optional[int]
            乱数シード値, EstimatorQNN が期待値に加える雑音
            (既定の精度 0.015625) に使用
        observables: list[SparsePauliOp]
            observable のリスト
        input_gradients: bool = False
            入力に対する勾配も計算するかどうか, 前に層を挟む場合に必要

        Returns
        ----------
        qnn: TorchConnector
            Estimator の QNN 層
        """
        # Estimator のインスタンスを用意
        estimator = Estimator(seed=seed)

        # QNN をセットアップ
        qnn_base = EstimatorQNN(
            circuit=self.qc,
            estimator=estimator,
            observables=observables,
            input_params=self.feature_map.parameters,
            weight_params=self.ansatz.parameters,
            input_gradients=input_gradients
        )
        initial_weights = 0.01 * (
            2 * algorithm_globals.random.random(qnn_base.num_weights) - 1
        )
        qnn = TorchConnector(qnn_base, initial_weights=initial_weights)

        return qnn


class QSampler(nn.Module):
    def __init__(
        self, qcs: list[QuantumCircuit], seed: Optional[int],
        output_dim: int, input_gradients: bool = False
    ):
        # 親クラスのコンストラクタを呼び出す
        super(QSampler, self).__init__()

        # 量子回路のリストを取得
        self.qc = qcs[0]
        self.feature_map = qcs[1]
        self.ansatz = qcs[2]

        # 量子層の出力次元を設定
        self.output_dim = output_dim

        # 量子層の定義
        self.qnn_sam = self.set_sampler(seed, input_gradients)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        順伝播を行う関数

        Parameters
        ----------
        x: torch.Tensor
            入力データ

        Returns
        ----------
        out: torch.Tensor
            クラスごとの確率, 形状は (バッチ, output_dim)
        """
        # 量子層の順伝播計算
        out = self.qnn_sam(x)

        return out

    def set_sampler(
        self, seed: Optional[int], input_gradients: bool = False
    ) -> TorchConnector:
        """
        Sampler の QNN を用意する関数

        Parameters
        ----------
        seed: Optional[int]
            乱数シード値, 測定のサンプリングに使用
        input_gradients: bool = False
            入力に対する勾配も計算するかどうか, 前に層を挟む場合に必要

        Returns
        ----------
        qnn: TorchConnector
            Sampler の QNN 層
        """
        # Sampler のインスタンスを用意
        sampler = Sampler(seed=seed)

        # QNN をセットアップ
        # OddNN の版は interpret を渡さず, 出力が全ての測定結果
        # (2 の量子ビット数乗) の確率になるので,
        # 1 の数の偶奇で 2 クラスに集約
        qnn_base = SamplerQNN(
            circuit=self.qc,
            sampler=sampler,
            input_params=self.feature_map.parameters,
            weight_params=self.ansatz.parameters,
            interpret=parity,
            output_shape=self.output_dim,
            input_gradients=input_gradients
        )
        initial_weights = 0.01 * (
            2 * algorithm_globals.random.random(qnn_base.num_weights) - 1
        )
        qnn = TorchConnector(qnn_base, initial_weights=initial_weights)

        return qnn


class QuantumNeuralNetwork(nn.Module):
    def __init__(
        self, input_dim: int, num_reuploads: int,
        net_type: Literal["Estimator", "Sampler"], seed: Optional[int],
        feature_map_reps: int = 1, ansatz_reps: int = 1,
        observables: Optional[list[SparsePauliOp]] = None,
        output_dim: int = 2, input_gradients: bool = False
    ):
        # 親クラスのコンストラクタを呼び出す
        super(QuantumNeuralNetwork, self).__init__()

        # 入出力層の次元を設定
        self.input_dim = input_dim
        self.net_type = net_type
        self.output_dim = output_dim

        # 量子回路を設計
        qcs = self.build_reuploading_circuit(
            self.input_dim, num_reuploads, feature_map_reps, ansatz_reps
        )
        self.qc = qcs[0]
        self.feature_map = qcs[1]
        self.ansatz = qcs[2]
        self.qcs = [self.qc, self.feature_map, self.ansatz]

        # observable を定義
        # OddNN の版は省略すると self.observables が未定義になるので,
        # 全ビットの Z を既定に設定
        if observables is None:
            observables = [SparsePauliOp("Z" * self.input_dim)]
        self.observables = observables

        # QNN を構築
        if self.net_type == "Estimator":
            self.qnn = QEstimator(
                self.qcs, seed, self.observables, input_gradients
            )
        elif self.net_type == "Sampler":
            self.qnn = QSampler(
                self.qcs, seed, self.output_dim, input_gradients
            )
        else:
            raise ValueError(
                f"net_type は Estimator か Sampler にしてください: {net_type}"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        順伝播を行う関数

        Parameters
        ----------
        x: torch.Tensor
            入力データ

        Returns
        ----------
        out: torch.Tensor
            Estimator は期待値, Sampler はクラスごとの確率
        """
        # 量子層の順伝播計算
        out = self.qnn(x)

        return out

    def build_reuploading_circuit(
        self, num_qubits: int, num_reuploads: int,
        feature_map_reps: int = 1, ansatz_reps: int = 1
    ) -> list[QuantumCircuit]:
        """
        QNN 用の量子回路を設計する関数

        Parameters
        ----------
        num_qubits: int
            量子ビット数
        num_reuploads: int
            Re-Upload の回数
        feature_map_reps: int = 1
            特徴マップのリピート数
        ansatz_reps: int = 1
            ansatz のリピート数

        Returns
        ----------
        out: list[QuantumCircuit]
            量子回路のリスト
        """
        # 量子回路を初期化
        qc = QuantumCircuit(num_qubits)

        # 特徴マップと ansatz の定義
        feature_map = zz_feature_map(num_qubits, reps=feature_map_reps)
        ansatz = real_amplitudes(
            num_qubits, entanglement="linear", reps=ansatz_reps
        )

        # Re-Upload
        for _ in range(num_reuploads):
            qc.compose(feature_map, inplace=True)
            qc.compose(ansatz, inplace=True)

        # 量子回路をまとめてリスト化
        out = [qc, feature_map, ansatz]

        return out


class HybridQNN(nn.Module):
    def __init__(self, input_dim: int, qnn: QuantumNeuralNetwork):
        # 親クラスのコンストラクタを呼び出す
        super(HybridQNN, self).__init__()

        # 入力次元数を保存
        self.input_dim = input_dim

        # 古典の前処理層, 量子層, 古典の出力層の定義
        self.fc_in = nn.Linear(self.input_dim, qnn.input_dim)
        self.qnn = qnn
        # 量子層の出力 (Estimator は観測量ごとの期待値, Sampler はクラスの確率) を
        # 2 クラス目のロジット 1 つに変換
        if qnn.net_type == "Estimator":
            out_dim = len(qnn.observables)
        else:
            out_dim = qnn.output_dim
        self.fc_out = nn.Linear(out_dim, 1)

        # 量子回路の回転角に収まるように前処理の出力を変換する活性化関数の定義
        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        順伝播を行う関数

        Parameters
        ----------
        x: torch.Tensor
            入力データ

        Returns
        ----------
        out: torch.Tensor
            2 クラス目のロジット, 形状は (バッチ, 1)
        """
        # 古典の層で -π から π に変換してから量子層に入力
        h = torch.pi * self.tanh(self.fc_in(x))
        out = self.fc_out(self.qnn(h))

        return out


class ClassicalMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        # 親クラスのコンストラクタを呼び出す
        super(ClassicalMLP, self).__init__()

        # 入力次元数, 隠れ層の次元数, 出力次元数を保存
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # 隠れ層と出力層の定義
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.output_dim)

        # 活性化関数の定義
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        順伝播を行う関数

        Parameters
        ----------
        x: torch.Tensor
            入力データ

        Returns
        ----------
        out: torch.Tensor
            Estimator の場合と同じ形の 1 つのロジット
        """
        # 隠れ層を通して出力
        out = self.fc2(self.relu(self.fc1(x)))

        return out
