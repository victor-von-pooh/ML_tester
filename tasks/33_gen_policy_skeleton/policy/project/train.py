"""
問題の設定 (problem.yaml) から ML-Quest が書き出した学習のスクリプト

Gymnasium の形の環境で, 観測から行動を続けて選ぶ方策を学び, 問題の設定の
守りと一緒に書き出して, 守りを当てたまま環境で確かめる骨組みなので, 印
(TODO) の所を書き足す
"""
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Optional

from ml_found import (
    Experiment, MissingRequirementError, Trainer, load_project_config,
    require, save_evaluation, start_experiment, write_task
)
from model import PolicyNet
import yaml

# このプロジェクトのディレクトリ (ML-Found のプロジェクト)
PROJECT_DIR = Path(__file__).resolve().parent

# 問題の名前と, 方策が出すもの
PROBLEM_NAME = "台車の棒を立て続ける"
PREDICT = "押す向きを選ぶ方策"

# task.yaml の作り方 (問題の設定の答え)
METHOD = "scratch"

# 方策を使ってよいかと守りの範囲を決める人 (問題の設定の答え)
APPROVER = "設備の責任者"

# 問題の守りの文 (行動の上限と止める条件. config の guards に数で
# 書く元)
ACTION_LIMITS = "左右に押すだけ"
STOP = "台車が中心から 2.4 を出たら"

# 守りの結果の種類 (ML-Found の evaluate_policy の列) と, 行動を返さずに
# エピソードを終える結果
STATUSES = ("ok", "clipped", "fallback", "stop", "handover")
STOP_STATUSES = ("stop", "handover")

# 守る指標の表の列ごとの向き
GUARDED_BETTER = {"reward": "higher", "fallback": "lower", "stop": "lower",
                  "handover": "lower"}

# 学習の方法の名前
ALGORITHMS = ("auto", "reinforce", "actor_critic")

# 方策の一式に写すモデルの定義のファイルの名前 (metadata.model.spec)
POLICY_SPEC = "model.yaml"

# 守りの数が無いことの知らせ
NO_LIMITS = ("config の guards に行動の範囲も止める条件も数で書いていない"
             "ので, 外れたときの動きは観測やモデルの出力が読めないときだけ"
             "です. 問題の守り (ACTION_LIMITS と STOP) を観測と行動"
             "の空間に合わせて数にしてください")


def make_environment(cfg: Any) -> Any:
    """
    config の環境を作る関数

    Parameters
    ----------
    cfg: Any
        config (env.id に Gymnasium の環境の登録名. null なら env.py の
        ProjectEnv)

    Returns
    ----------
    env: Any
        Gymnasium の形の環境
    """
    # 追加の依存 rl が要る関数なので, 確認の後に関数の中で読み込み
    from ml_found.rl import make_env

    # 登録名があれば Gymnasium から, 無ければ自作の環境
    if cfg.env.id:
        env = make_env(cfg.env.id)
        return env
    from env import ProjectEnv
    env = ProjectEnv()

    return env


def environment_name(cfg: Any) -> str:
    """
    記録に残す環境の名前を返す関数

    Parameters
    ----------
    cfg: Any
        config (env.id)

    Returns
    ----------
    name: str
        Gymnasium の環境の登録名か, 自作の環境の env:ProjectEnv
    """
    # 自作の環境は Gymnasium の entry_point の書き方 (登録名ではない)
    name = cfg.env.id or "env:ProjectEnv"

    return name


def space_kind(env: Any) -> str:
    """
    行動の空間の種類を返す関数

    Parameters
    ----------
    env: Any
        環境

    Returns
    ----------
    kind: str
        Discrete (番号で選ぶ行動) か Box (連続の行動)

    Raises
    ----------
    ValueError
        他の空間の場合 (ML-Found の強化学習の action_kind と同じく, 形が ()
        で 0 から始まる Discrete と, PolicyNet が扱える 1 次元の Box だけ)
    """
    # 空間のクラスの名前, 形, 番号の始まり
    space = env.action_space
    kind = type(space).__name__
    shape = tuple(getattr(space, "shape", None) or ())
    if kind == "Discrete" and shape == () and int(
        getattr(space, "start", 0)
    ) == 0:
        return kind
    if kind == "Box" and len(shape) == 1:
        return kind
    raise ValueError(
        f"行動の空間 {space!r} は扱えません (0 から始まる Discrete か, 1 次元"
        " の Box にしてください. 形が () の Box は (1,) に)"
    )


def model_args(env: Any, cfg: Any) -> dict:
    """
    観測と行動の空間から, 方策のモデルの引数を決める関数

    Parameters
    ----------
    env: Any
        環境
    cfg: Any
        config (model.hidden_dim)

    Returns
    ----------
    args: dict
        PolicyNet の引数 (方策の頭を除く. 既定の定義の大きさにも使う)

    Raises
    ----------
    ValueError
        行動の空間が Discrete か Box でない場合と, 観測の空間が 1 次元の
        Box でない場合 (PolicyNet は観測を要素の並びとして受け取る)
    """
    # 観測は 1 次元の Box だけ (Discrete などは要素の数が無い)
    observation = env.observation_space
    kind = type(observation).__name__
    if kind != "Box" or len(getattr(observation, "shape", None) or ()) != 1:
        raise ValueError(
            f"観測の空間 {observation!r} は扱えません (1 次元の Box にして"
            "ください. Gymnasium の FlattenObservation などで平らにできます)"
        )

    # 番号で選ぶ行動は行動の数, 連続の行動は要素の数
    continuous = space_kind(env) == "Box"
    action_dim = (int(env.action_space.shape[0]) if continuous
                  else int(env.action_space.n))
    args = {"obs_dim": int(env.observation_space.shape[0]),
            "hidden_dim": int(cfg.model.hidden_dim),
            "action_dim": action_dim, "continuous": continuous}

    return args

def default_definition(args: dict) -> dict:
    """
    観測と行動の空間から, 方策の頭の既定の定義を作る関数

    Parameters
    ----------
    args: dict
        PolicyNet の引数 (obs_dim, hidden_dim, action_dim)

    Returns
    ----------
    definition: dict
        ML-ModelControl の形のモデルの定義 (観測の 1 つの入力から, 全結合
        と活性化を 2 段と, 行動ごとのスコアか平均の 1 つの出力)
    """
    # 大きさは観測と行動の空間と config の model.hidden_dim から
    obs_dim = args["obs_dim"]
    hidden = args["hidden_dim"]
    action_dim = args["action_dim"]
    definition = {
        "format_version": 1,
        "name": "policy",
        "description": "観測から行動ごとのスコアか連続の行動の平均",
        "inputs": {"observation": {"shape": [1, obs_dim]}},
        "nodes": {
            "fc1": {"unit": "nn.Linear",
                    "params": {"in_features": obs_dim,
                               "out_features": hidden},
                    "inputs": "observation"},
            "act1": {"unit": "nn.Tanh", "inputs": "fc1"},
            "fc2": {"unit": "nn.Linear",
                    "params": {"in_features": hidden,
                               "out_features": hidden},
                    "inputs": "act1"},
            "act2": {"unit": "nn.Tanh", "inputs": "fc2"},
            "head": {"unit": "nn.Linear",
                     "params": {"in_features": hidden,
                                "out_features": action_dim},
                     "inputs": "act2"},
        },
        "outputs": {"action": "head"},
    }

    return definition


def definition_errors(spec: Path, args: dict) -> list[str]:
    """
    方策の頭の定義の誤りを, 実験の出力先を作る前に確かめる関数

    Parameters
    ----------
    spec: Path
        config の model.spec の, プロジェクトの定義
    args: dict
        PolicyNet の引数 (観測の次元数と行動の数)

    Returns
    ----------
    errors: list[str]
        誤りの文 (無ければ空)
    """
    # 追加の依存 modelcontrol が要るので, 関数の中で読み込み
    import torch
    from ml_modelcontrol import build_model, load_spec, validate

    # 読めない定義も誤りとして返る (例外にならない)
    errors = [issue["message"] for issue in validate(spec)
              if issue["severity"] == "error"]
    if errors:
        return errors

    # 方策の一式の読む側の決まり (観測の入力と行動の出力を 1 つずつ)
    definition = load_spec(spec)
    if len(definition["inputs"]) != 1 or len(definition["outputs"]) != 1:
        errors.append("入力 (観測) と出力 (行動) を 1 つずつにしてください")
        return errors

    # 観測の 1 件の組から, 行動の数の出力が出るか
    model = build_model(spec, seed=0, load_weights=False).eval()
    try:
        with torch.no_grad():
            output = model(torch.zeros(1, args["obs_dim"]))
    except Exception as error:
        errors.append(f"観測 (要素 {args['obs_dim']}) を入れられません "
                      f"({type(error).__name__}: {error}). 定義の入力の形"
                      f"を (1, {args['obs_dim']}) にし, 最初の層が要素 "
                      f"{args['obs_dim']} つを受け取るようにしてください")
        return errors
    if tuple(output.shape) != (1, args["action_dim"]):
        errors.append(f"出力の形 {tuple(output.shape)} が (バッチ, 行動の"
                      f"数 {args['action_dim']}) と違います")

    return errors


def policy_definition(cfg: Any, args: dict) -> Optional[Path]:
    """
    方策の頭の定義を確かめ, 無ければ既定の定義を書く関数

    Parameters
    ----------
    cfg: Any
        config (model.spec に定義の場所. プロジェクトのディレクトリから)
    args: dict
        PolicyNet の引数

    Returns
    ----------
    spec: Optional[Path]
        プロジェクトの定義 (誤りがあれば知らせて None)
    """
    # 場所が空なら案内. 無ければ観測と行動の空間から既定の定義を書く
    # (ML-CreateArchitecture で開いて直せる)
    if not cfg.model.spec:
        print("config の model.spec に定義の場所を書いてください (既定は "
              "models/policy.yaml)")
        return None
    spec = PROJECT_DIR / cfg.model.spec
    if not spec.exists():
        spec.parent.mkdir(parents=True, exist_ok=True)
        text = yaml.safe_dump(default_definition(args), allow_unicode=True,
                              sort_keys=False)
        spec.write_text(text, encoding="utf-8")
        print(f"{spec} に観測と行動の空間から既定の定義を書きました")
    errors = definition_errors(spec, args)
    if errors:
        print(f"{spec} の定義に誤りがあります. README の「次にすること」"
              "の 3 のとおり, モデルの作成・選定の段階で開いて直してください "
              "(ファイルを消すと次の学習で既定の定義を書き直します)")
        for message in errors:
            print(f"- {message}")
        return None

    return spec


def model_file(
    spec: Path, experiment: Experiment
) -> tuple[Path, Optional[Path]]:
    """
    方策の頭を組み立てる定義のファイルと, 定義の中の相対の参照 (外部の
    重みなど) の起点を返す関数

    Parameters
    ----------
    spec: Path
        config の model.spec の, プロジェクトの定義
    experiment: Experiment
        実験 (ML-Found が実験に写した定義の場所 model_spec_path と, 写した
        元の定義 model_spec_source)

    Returns
    ----------
    path: Path
        ML-Found が実験に写した定義 (学習を始めたときの定義). 写せなければ
        プロジェクトの定義
    base_dir: Optional[Path]
        相対の参照の起点 (写しなら元の定義のディレクトリ. プロジェクトの
        定義なら None で, 定義のディレクトリ)
    """
    # 写しの相対の参照は元の定義の場所から (写しの隣には重みが無い)
    if experiment.model_spec_path is not None:
        path = Path(experiment.model_spec_path)
        base_dir = Path(experiment.model_spec_source).parent
        return path, base_dir

    # 写しが無ければ理由を知らせて, プロジェクトの定義
    experiment.logger.warning(
        "モデルの定義を実験に写せなかったので, 学習のプロジェクトの定義から"
        "組み立てます (理由は source.yaml の model_spec の status)"
    )
    path = spec
    base_dir = None

    return path, base_dir


def algorithm_of(env: Any, cfg: Any) -> str:
    """
    学習の方法を決める関数

    Parameters
    ----------
    env: Any
        環境
    cfg: Any
        config (algorithm. auto なら空間から)

    Returns
    ----------
    algorithm: str
        reinforce か actor_critic (auto は番号で選ぶ行動なら reinforce,
        連続の行動なら actor_critic)
    """
    # 連続の行動は価値の関数で利得を見積もる方が安定しやすい
    algorithm = str(cfg.algorithm)
    if algorithm == "auto":
        algorithm = ("actor_critic" if space_kind(env) == "Box"
                     else "reinforce")
    if algorithm not in ("reinforce", "actor_critic"):
        raise ValueError(
            f"algorithm の {algorithm} は知らない方法です (auto, reinforce, "
            "actor_critic)"
        )

    return algorithm


def make_step(algorithm: str, cfg: Any) -> Any:
    """
    学習の方法の step を作る関数

    Parameters
    ----------
    algorithm: str
        reinforce か actor_critic
    cfg: Any
        config (loss の係数)

    Returns
    ----------
    step: Any
        ML-Found の ReinforceStep か ActorCriticStep
    """
    # 追加の依存 rl が要るので, 関数の中で読み込み
    from ml_found.rl import ActorCriticStep, ReinforceStep

    if algorithm == "actor_critic":
        step = ActorCriticStep(
            value_coef=cfg.loss.value_coef,
            entropy_coef=cfg.loss.entropy_coef,
            gae_lambda=cfg.loss.gae_lambda
        )
        return step
    step = ReinforceStep(entropy_coef=cfg.loss.entropy_coef)

    return step


def optional_list(value: Any) -> Optional[list]:
    """
    config の並びを, 守りに渡す並びにする関数

    Parameters
    ----------
    value: Any
        config の値 (null か並び)

    Returns
    ----------
    items: Optional[list]
        並び (null なら None. 要素の null は制限なし)
    """
    # OmegaConf の並びを Python の並びに
    items = None if value is None else list(value)

    return items


def build_guards(env: Any, cfg: Any) -> dict:
    """
    config の守りから, 方策の一式に書く守りを作る関数

    Parameters
    ----------
    env: Any
        環境 (観測と行動の空間)
    cfg: Any
        config (guards)

    Returns
    ----------
    guards: dict
        ML-Found の make_guards の守り
    """
    # 追加の依存 rl が要るので, 関数の中で読み込み
    from ml_found.rl import make_guards

    setting = cfg.guards
    action = setting.fallback_action
    guards = make_guards(
        env.observation_space, env.action_space,
        action_low=optional_list(setting.action_low),
        action_high=optional_list(setting.action_high),
        allowed_actions=optional_list(setting.allowed_actions),
        observation_low=optional_list(setting.observation_low),
        observation_high=optional_list(setting.observation_high),
        fallback=setting.fallback,
        fallback_action=(list(action) if hasattr(action, "__len__")
                         else action)
    )

    return guards


def guards_numbered(guards: dict) -> bool:
    """
    守りに数の範囲が 1 つでもあるかを返す関数

    Parameters
    ----------
    guards: dict
        make_guards の守り

    Returns
    ----------
    numbered: bool
        行動の範囲か止める条件の要素に, null でない値があるか
    """
    # 範囲の項目の要素のどれかが数
    keys = ("action_low", "action_high", "allowed_actions",
            "observation_low", "observation_high")
    numbered = any(value is not None for key in keys
                   for value in (guards.get(key) or []))

    return numbered


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
        守りの項目と値 (JSON の書き方. 制限なしは null)
    """
    # 外れたときの動きと, 他の項目
    fallback = guards["fallback"]
    items = [f"{key} {json.dumps(value)}" for key, value in guards.items()
             if key != "fallback"]
    items.append(f"fallback {fallback['kind']}"
                 + (f" ({fallback['action']})"
                    if fallback["kind"] == "action" else ""))
    text = ", ".join(items)

    return text


def summarize(rows: list[dict]) -> dict:
    """
    evaluate_policy のエピソードごとの行を, 1 つの要約にする関数

    Parameters
    ----------
    rows: list[dict]
        evaluate_policy の行

    Returns
    ----------
    summary: dict
        reward (報酬の和の平均), steps (歩数の合計), stopped (stop か
        handover で終えたエピソードの数), 守りの結果ごとの数の合計
    """
    # エピソードの平均と合計
    summary = {
        "reward": sum(row["reward"] for row in rows) / len(rows),
        "steps": sum(row["length"] for row in rows),
        "stopped": sum(row["end"] in STOP_STATUSES for row in rows),
    }
    for status in STATUSES:
        summary[status] = sum(row[status] for row in rows)

    return summary


def evaluate_guards(
    model: Any, env: Any, cfg: Any, experiment: Experiment, spec: Path,
    guards: dict, metadata: dict
) -> list[dict]:
    """
    方策を守りと一緒に書き出し, 読み直して守りを当てたまま環境で動かす
    関数

    Parameters
    ----------
    model: Any
        学習した後の方策
    env: Any
        守り付きの評価の環境 (観測と行動の空間もここから)
    cfg: Any
        config (evaluation)
    experiment: Experiment
        実験 (policy/ に書き出し, train.log に記録)
    spec: Path
        方策の頭を組み立てた定義 (一式に写し, 読み直しに使う)
    guards: dict
        守り
    metadata: dict
        方策の一式の metadata

    Returns
    ----------
    rows: list[dict]
        エピソードごとの evaluate_policy の行 (guards の文を足す)
    """
    # 追加の依存 rl が要るので, 関数の中で読み込み
    from ml_found.rl import evaluate_policy, export_policy, load_policy
    from ml_modelcontrol import build_model

    # 実験の中の policy/ に方策の頭 (重みの名前が定義から組み立てたモデル
    # と同じ) を書き出し, 定義を写す. 写した定義から組み立てた新しいモデル
    # に読み直して (重みは一式から) 守りを当てたエピソードを動かす
    place = Path(experiment.run_dir) / "policy"
    export_policy(place, model.actor, env.observation_space,
                  env.action_space, guards, metadata=metadata)
    shutil.copy(spec, place / POLICY_SPEC)
    policy = load_policy(
        place, build_model(place / POLICY_SPEC, seed=0, load_weights=False)
    )
    episodes = evaluate_policy(
        policy, env, episodes=cfg.evaluation.guarded_episodes,
        seed=cfg.evaluation.guarded_seed,
        max_steps=cfg.env.max_episode_steps
    )
    text = describe_guards(guards)
    rows = [{"guards": text, **row} for row in episodes]
    summary = summarize(episodes)
    message = (
        f"守り付きの評価: 報酬の和の平均 {summary['reward']:.2f}, 歩数 "
        f"{summary['steps']}, 止めたエピソード {summary['stopped']}, "
        + ", ".join(f"{status} {summary[status]}" for status in STATUSES)
    )
    experiment.logger.info(message)
    print(message)

    return rows


def main() -> None:
    """
    環境で方策を学び, 守りと一緒に書き出して守り付きで確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # config. 学習の方法の名前の確かめ
    cfg, diff = load_project_config(str(PROJECT_DIR))
    if str(cfg.algorithm) not in ALGORITHMS:
        print(f"config の algorithm の {cfg.algorithm} は知らない方法です "
              f"({', '.join(ALGORITHMS)} のどれかにしてください)")
        sys.exit(1)

    # 学習用, 検証用, 守り付きの評価用の環境は別に作る. 守りも学習の前に
    # 作る. Gymnasium が足りないか守りが誤りなら, 出力先を作る前に止める
    try:
        require("rl")
        train_env = make_environment(cfg)
        valid_env = make_environment(cfg)
        guard_env = make_environment(cfg)
    except MissingRequirementError as error:
        print(error)
        sys.exit(1)
    # 空間を先に確かめる (扱えない空間を守りの誤りと取り違えないように)
    try:
        args = model_args(train_env, cfg)
    except ValueError as error:
        print(f"環境の空間を扱えません ({error})")
        sys.exit(1)
    # 方策の頭の定義 (無ければ空間から既定の定義を書く)
    try:
        require("modelcontrol")
    except MissingRequirementError as error:
        print(error)
        sys.exit(1)
    spec = policy_definition(cfg, args)
    if spec is None:
        sys.exit(1)
    try:
        guards = build_guards(guard_env, cfg)
    except ValueError as error:
        print(f"config の guards を直してください ({error}). 外れたときの"
              "動きが action なら, 番号で選ぶ行動では guards.fallback_action"
              " に決めた行動の番号を書きます")
        sys.exit(1)
    from ml_found.rl import EvaluationLoader, Exploration, RolloutLoader
    algorithm = algorithm_of(train_env, cfg)
    name = environment_name(cfg)
    numbered = guards_numbered(guards)

    # 実験. config の requires が足りなければ, start_experiment が出力先を
    # 作る前に止まるので, 入れ方を示して終了
    try:
        experiment = start_experiment(cfg, diff, str(PROJECT_DIR))
    except MissingRequirementError as error:
        print(error)
        sys.exit(1)

    # 何の実験か. 方策は受け持ちでは効果を見積もる (effect), 出力の形は
    # 行動 (action), 正解の出どころは環境の報酬 (environment)
    write_task(experiment, {
        "role": "effect", "output": "action", "what": PREDICT,
        "target": {"kind": "reward",
                   "origin": {"type": "environment", "name": name}},
        "method": METHOD,
    })
    experiment.logger.info(
        f"問題: {PROBLEM_NAME}. 方策が出すもの: {PREDICT}. 環境: {name}, 行動の空間: "
        f"{space_kind(train_env)}, 学習の方法: {algorithm}, 守り: "
        f"{describe_guards(guards)}"
    )
    if not numbered:
        experiment.logger.warning(NO_LIMITS)

    # 1 エポックは rollout.episodes のエピソード (batch_episodes ごとに
    # 更新), 検証は別の種のエピソードを最もスコアの高い行動 (連続の行動は
    # 平均) で
    from ml_modelcontrol import build_model
    path, base_dir = model_file(spec, experiment)
    actor = build_model(path, seed=cfg.experiment.seed, base_dir=base_dir)
    model = PolicyNet(actor, **args)
    train_loader = RolloutLoader(
        train_env, model, episodes=cfg.rollout.episodes,
        batch_episodes=cfg.rollout.batch_episodes, gamma=cfg.rollout.gamma,
        exploration=Exploration(cfg.rollout.temperature,
                                cfg.rollout.epsilon),
        seed=cfg.experiment.seed,
        max_episode_steps=cfg.env.max_episode_steps,
        safety_key=cfg.safety_key
    )
    valid_loader = EvaluationLoader(
        valid_env, model, episodes=cfg.evaluation.episodes,
        seed=cfg.env.valid_seed,
        max_episode_steps=cfg.env.max_episode_steps,
        safety_key=cfg.safety_key
    )
    trainer = Trainer(model, experiment, make_step(algorithm, cfg))
    trainer.fit(train_loader, valid_loader)

    # 実験だけで見直せるように, 問題の設定の写し (元の問題の場所
    # origin も残る)
    run_dir = Path(experiment.run_dir)
    shutil.copy(PROJECT_DIR / "problem.yaml", run_dir / "problem.yaml")

    # 結果の表示. 書き出す重みは, 最良のモデルに戻した場合だけ最良の
    # エポック (select_best: false などは最後)
    best = trainer.final_weights == "best"
    epoch = trainer.checkpoint.best_epoch if best else trainer.epoch
    weights = "最良のモデル" if best else "最後のエポックの重み"
    print(f"保存先: {run_dir}")
    print(f"書き出す重み: {weights} (エポック {epoch})")

    # 守りと一緒に方策の一式 (policy/) に書き出し, 読み直して守り付きで
    # 動かした結果を守る指標の表に
    # metadata には, 守りを数にしたか (guards_set) と問題の守りの
    # 文も残す (数が無いまま書き出した一式と分かるように)
    metadata = {
        "env_id": cfg.env.id or None, "algorithm": algorithm,
        "epoch": epoch,
        "model": {"spec": POLICY_SPEC},
        "approver": APPROVER, "guards_set": numbered,
        "guards_text": {"action_limits": ACTION_LIMITS, "stop": STOP},
    }
    if not cfg.env.id:
        metadata["entry_point"] = name
    rows = evaluate_guards(trainer.model, guard_env, cfg, experiment, path,
                           guards, metadata)
    save_evaluation(
        experiment, "guarded_episodes", rows, "environment",
        description="書き出した方策を読み直し, 守りを当てて環境で動かした"
        f"エピソード ({weights}). 1 行が 1 エピソードで, 歩ごとの守りの"
        "結果の数. 実機での結果は別に確かめる"
        + ("" if numbered else f". {NO_LIMITS}"),
        guard=True, epoch=epoch, better=GUARDED_BETTER
    )
    for env in (train_env, valid_env, guard_env):
        env.close()
    print(f"方策の一式: {run_dir / 'policy'} (使ってよいかと守りの範囲は "
          f"{APPROVER or '承認する人'} が決める)")


if __name__ == "__main__":
    main()
