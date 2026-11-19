from pathlib import Path
from typing import Callable

import numpy as np
from omegaconf import OmegaConf
import pytest

from ml_found.utils import requirements
from model import PolicyNet
import train

# このタスクの基準の config
DEFAULT_CONFIG = (
    Path(__file__).resolve().parent / "configs" / "default" / "train.yaml"
)


class HiddenModules:
    def __init__(self, original: Callable[[str], bool], hidden: set[str]):
        # 元の確かめ方と, 入っていないように見せる名前を保存
        self.original = original
        self.hidden = hidden

    def __call__(self, name: str) -> bool:
        """
        隠す名前なら見つからないとし, それ以外は元の関数で確かめる関数

        Parameters
        ----------
        name: str
            モジュールの名前

        Returns
        ----------
        found: bool
            見つかった場合は True
        """
        # 隠す名前の判定
        if name in self.hidden:
            found = False
            return found
        found = self.original(name)

        return found


def exported_policy(tmp_path: Path, setting_name: str) -> object:
    """
    config の守りで学習前の方策を書き出し, 読み直す関数

    Parameters
    ----------
    tmp_path: Path
        書き出す先
    setting_name: str
        guards (本番の候補) か drills の名前

    Returns
    ----------
    policy: object
        load_policy の方策
    """
    # 守りは train.py と同じ関数で作る (重みは学習前のままでよい)
    from ml_found.rl import export_policy, load_policy, make_env

    cfg = OmegaConf.load(DEFAULT_CONFIG)
    setting = cfg.guards
    for drill in cfg.drills:
        if drill.name == setting_name:
            setting = drill
    env = make_env(cfg.env.id)
    model = PolicyNet(4, cfg.model.hidden_dim, 2)
    guards = train.build_guards(env, setting)
    place = export_policy(
        tmp_path / setting_name, model, env.observation_space,
        env.action_space, guards
    )
    policy = load_policy(place, PolicyNet(4, cfg.model.hidden_dim, 2))

    return policy


@pytest.mark.parametrize("observation", [
    [2.5, 0.0, 0.0, 0.0],
    [-2.01, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.19, 0.0],
    [0.0, 0.0, -0.5, 0.0],
    [float("nan"), 0.0, 0.0, 0.0],
])
def test_out_of_bounds_stops(tmp_path: Path, observation: list) -> None:
    """
    止める条件の外の観測で, 行動を返さずに止まることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        書き出す先
    observation: list
        止める条件の外の観測

    Returns
    ----------
    None
    """
    # 範囲の外の観測では行動を返さずに止める
    pytest.importorskip("gymnasium")
    policy = exported_policy(tmp_path, "guards")
    result = policy.act(np.array(observation, dtype=np.float32))
    assert result["status"] == "stop"
    assert result["action"] is None
    assert result["reasons"]


def test_inside_bounds_acts(tmp_path: Path) -> None:
    """
    止める条件の中の観測では, 選んでよい行動を返すことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        書き出す先

    Returns
    ----------
    None
    """
    # 範囲の中 (速さと角速度は制限なし) では選んでよい行動
    pytest.importorskip("gymnasium")
    policy = exported_policy(tmp_path, "guards")
    result = policy.act(np.array([1.9, 5.0, -0.17, -5.0], dtype=np.float32))
    assert result["status"] == "ok"
    assert result["action"] in (0, 1)


def test_drill_fallback_action(tmp_path: Path) -> None:
    """
    外れたときの動きが action の練習では, 決めた行動で続くことを確かめる
    関数

    Parameters
    ----------
    tmp_path: Path
        書き出す先

    Returns
    ----------
    None
    """
    # 練習の狭い範囲の外では決めた行動 0
    pytest.importorskip("gymnasium")
    policy = exported_policy(tmp_path, "narrow_fallback")
    result = policy.act(np.array([0.5, 0.0, 0.0, 0.0], dtype=np.float32))
    assert result["status"] == "fallback"
    assert result["action"] == 0


def test_missing_gymnasium_shows_install(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """
    Gymnasium が無い場合, 学習の前に入れ方を示して止まることを確かめる
    関数

    Parameters
    ----------
    monkeypatch: pytest.MonkeyPatch
        確かめ方の差し替えに使う
    capsys: pytest.CaptureFixture
        表示の読み取りに使う

    Returns
    ----------
    None
    """
    # Gymnasium だけを隠し, 学習の前の確認で終了すること
    monkeypatch.setattr(
        requirements, "module_found",
        HiddenModules(requirements.module_found, {"gymnasium"})
    )
    with pytest.raises(SystemExit) as stopped:
        train.main()
    text = capsys.readouterr().out
    assert stopped.value.code == 1
    assert "gymnasium" in text
    assert 'pip install -e "./ML-Found[rl]"' in text
