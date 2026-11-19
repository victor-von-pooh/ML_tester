from collections import Counter
from pathlib import Path
import sys
from typing import Any

from ml_found import (
    Experiment, Trainer, load_project_config, require, save_evaluation,
    start_experiment, write_task
)
from ml_found.utils import MissingRequirementError  # ml_found の直下に無い
from model import PolicyNet

# このタスクのディレクトリ (ML-Found のプロジェクトのディレクトリ)
TASK_DIR = Path(__file__).resolve().parent

# 守りの結果の種類 (ML-Found の apply_guards の status)
STATUSES = ("ok", "clipped", "fallback", "stop", "handover")


def build_guards(env: Any, setting: Any) -> dict:
    """
    config の守りの設定から, 方策の一式に書く守りを作る関数

    Parameters
    ----------
    env: Any
        Gymnasium の環境 (観測と行動の空間)
    setting: Any
        allowed_actions (無い場合は全て), position_limit, angle_limit,
        fallback, fallback_action

    Returns
    ----------
    guards: dict
        make_guards の守り. 台車の位置 (要素 0) と棒の角度 (要素 2) に
        止める条件, 速さ (要素 1 と 3) は制限なし
    """
    # 追加の依存 rl が要る関数なので, 確認の後に関数の中で読み込み
    from ml_found.rl import make_guards

    limits = [setting.position_limit, None, setting.angle_limit, None]
    allowed = setting.get("allowed_actions")
    guards = make_guards(
        env.observation_space, env.action_space,
        allowed_actions=None if allowed is None else list(allowed),
        observation_low=[
            None if value is None else -value for value in limits
        ],
        observation_high=limits, fallback=setting.fallback,
        fallback_action=setting.get("fallback_action")
    )

    return guards


def run_guarded(policy: Any, env: Any, episodes: int, seed: int) -> dict:
    """
    守りを当てた方策でエピソードを動かし, 守りの結果を数える関数

    stop と handover では行動を返さないので, そこでエピソードを終える
    (元の制御や人に戻した扱い). fallback は決めた行動で続ける

    Parameters
    ----------
    policy: Any
        load_policy の方策 (act で行動と status を返す)
    env: Any
        評価に使う環境 (シミュレータ)
    episodes: int
        エピソード数. 乱数の種は seed, seed + 1, ...
    seed: int
        最初のエピソードの乱数の種

    Returns
    ----------
    summary: dict
        episodes, reward (報酬の和の平均), steps (歩数の合計),
        stopped (stop か handover で終えたエピソード数), status ごとの数
    """
    # 1 歩ごとに守りを当てた行動を決め, 返った status を数える
    counts = Counter()
    rewards = []
    stopped = 0
    for index in range(episodes):
        observation, _ = env.reset(seed=seed + index)
        total = 0.0
        done = False
        while not done:
            result = policy.act(observation)
            counts[result["status"]] += 1
            if result["action"] is None:
                stopped += 1
                break
            observation, reward, terminated, truncated, _ = env.step(
                result["action"]
            )
            total += float(reward)
            done = terminated or truncated
        rewards.append(total)
    summary = {
        "episodes": episodes, "reward": sum(rewards) / len(rewards),
        "steps": sum(counts.values()), "stopped": stopped
    }
    for status in STATUSES:
        summary[status] = counts[status]

    return summary


def describe_guards(guards: dict) -> str:
    """
    守りを 1 行の文にする関数 (評価の表と train.log のため)

    Parameters
    ----------
    guards: dict
        make_guards の守り

    Returns
    ----------
    text: str
        選んでよい行動, 観測の上限, 外れたときの動き
    """
    # 上限は下限と対称なので上限だけ
    fallback = guards["fallback"]
    text = (
        f"allowed_actions {guards['allowed_actions']}, "
        f"observation_high {guards['observation_high']}, "
        f"fallback {fallback['kind']}"
    )
    if fallback["kind"] == "action":
        text += f" ({fallback['action']})"

    return text


def evaluate_guards(
    model: Any, env: Any, cfg: Any, experiment: Experiment, best_epoch: int
) -> list[dict]:
    """
    方策を守りと一緒に書き出し, 読み直して守り付きのエピソードを動かす
    関数

    Parameters
    ----------
    model: Any
        学習した後の方策 (ベストモデルの重み)
    env: Any
        評価に使う環境 (観測と行動の空間もここから)
    cfg: Any
        config (guards, drills, env, model, evaluation)
    experiment: Experiment
        実験 (policy/ と drills/ に書き出し, train.log に記録)
    best_epoch: int
        書き出す重みのエポック (metadata に残す)

    Returns
    ----------
    rows: list[dict]
        守りごとの結果の行 (name, guards と run_guarded の値)
    """
    from ml_found.rl import export_policy, load_policy

    # 本番の候補 (policy/) と, 止める条件を狭めた練習 (drills/<名前>/)
    sets = [("policy", cfg.guards, Path(experiment.run_dir) / "policy")]
    for drill in cfg.drills:
        place = Path(experiment.run_dir) / "drills" / drill.name
        sets.append((drill.name, drill, place))
    obs_dim = env.observation_space.shape[0]
    n_actions = int(env.action_space.n)
    rows = []
    for name, setting, place in sets:
        guards = build_guards(env, setting)
        export_policy(
            place, model, env.observation_space, env.action_space, guards,
            metadata={
                "env_id": cfg.env.id, "algorithm": "REINFORCE",
                "best_epoch": best_epoch,
                "model": {"hidden_dim": cfg.model.hidden_dim},
                "drill": name != "policy"
            }
        )

        # 書き出した一式を新しいモデルに読み直して動かす (重みは一式から)
        policy = load_policy(
            place, PolicyNet(obs_dim, cfg.model.hidden_dim, n_actions)
        )
        summary = run_guarded(
            policy, env, cfg.evaluation.guarded_episodes,
            cfg.evaluation.guarded_seed
        )
        row = {"name": name, "guards": describe_guards(guards), **summary}
        rows.append(row)
        text = (
            f"守り付きの評価 {name}: 報酬の和の平均 {row['reward']:.1f}, "
            f"歩数 {row['steps']}, 止めたエピソード {row['stopped']}, "
            + ", ".join(f"{status} {row[status]}" for status in STATUSES)
        )
        experiment.logger.info(text)
        print(text)

    return rows


def main() -> None:
    """
    CartPole の方策を REINFORCE で学習し, 守りと一緒に書き出して確かめる
    関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 追加の依存 rl (Gymnasium) が無ければ入れ方を示して止まる.
    # 出力先を作る前の確認
    try:
        require("rl")
    except MissingRequirementError as error:
        print(error)
        sys.exit(1)
    from ml_found.rl import (
        EvaluationLoader, Exploration, ReinforceStep, RolloutLoader,
        make_env
    )

    # config と実験. 方策は受け持ち (role) の選択肢に無いので null
    cfg, diff = load_project_config(str(TASK_DIR))
    experiment = start_experiment(cfg, diff, str(TASK_DIR))
    write_task(experiment, {
        "what": "CartPole の観測 (台車の位置と速さ, 棒の角度と角速度) から"
        "押す向きを選ぶ方策",
        "target": {
            "origin": {
                "type": "unknown",
                "name": f"{cfg.env.id} の報酬 (シミュレータ)"
            }
        },
        "method": "scratch"
    })

    # 学習用, 検証用, 守り付きの評価用の環境は別
    train_env = make_env(cfg.env.id)
    valid_env = make_env(cfg.env.id)
    guard_env = make_env(cfg.env.id)
    obs_dim = train_env.observation_space.shape[0]
    n_actions = int(train_env.action_space.n)
    model = PolicyNet(obs_dim, cfg.model.hidden_dim, n_actions)

    # 1 エポックは rollout.episodes のエピソード, 検証は別の種の
    # エピソードを最もスコアの高い行動で
    train_loader = RolloutLoader(
        train_env, model, episodes=cfg.rollout.episodes,
        batch_episodes=cfg.rollout.batch_episodes, gamma=cfg.rollout.gamma,
        exploration=Exploration(
            cfg.rollout.temperature, cfg.rollout.epsilon
        ),
        seed=cfg.experiment.seed
    )
    valid_loader = EvaluationLoader(
        valid_env, model, episodes=cfg.evaluation.episodes,
        seed=cfg.env.valid_seed
    )
    trainer = Trainer(
        model, experiment, ReinforceStep(entropy_coef=cfg.loss.entropy_coef)
    )
    history = trainer.fit(train_loader, valid_loader)

    # 結果の表示
    best_epoch = trainer.checkpoint.best_epoch
    rewards = history.get("valid/reward")
    print(f"保存先: {experiment.run_dir}")
    print(f"学習したエポック数: {len(history.records)}")
    print(f"ベストモデルのエポック: {best_epoch}")
    print(f"ベストの検証の報酬の和の平均: {rewards[best_epoch - 1]:.1f}")

    # 守りと一緒に書き出し, 読み直して守り付きで動かした結果を守る指標の
    # 表に
    rows = evaluate_guards(
        trainer.model, guard_env, cfg, experiment, best_epoch
    )
    save_evaluation(
        experiment, "guarded_episodes", rows, "simulator",
        description="書き出した方策を読み直し, 守りを当てて動かした "
        "エピソード (シミュレータ). 歩ごとの守りの結果の数. policy が "
        "本番の候補, 他は止める条件を狭めた練習",
        guard=True, epoch=best_epoch
    )
    print(f"方策の一式: {experiment.run_dir}/policy")


if __name__ == "__main__":
    main()
