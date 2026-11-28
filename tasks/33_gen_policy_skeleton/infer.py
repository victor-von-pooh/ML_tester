from collections import Counter
from pathlib import Path
import sys

import gymnasium as gym
import numpy as np
import yaml
from PIL import Image
from ml_inference import load_bundle, load_policy, predict

# このタスクのディレクトリと, 推論の結果を書く場所
TASK_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = TASK_DIR / "outputs"

# 雑音だけの一式で作る画像の件数 (4 × 4 に並べるので平方数)
COUNT = 16

# 方策を守り付きで動かすエピソード数と, 最初のエピソードの乱数の種
# (学習の検証と守り付きの評価の種と重ならない値)
EPISODES = 10
EPISODE_SEED = 30000


def latest_run(project: Path, name: str) -> Path:
    """
    骨組みの実験の中で最も新しい, 推論の一式を持つ実験の一式の場所を返す
    関数

    Parameters
    ----------
    project: Path
        骨組みのディレクトリ
    name: str
        実験の中の一式のディレクトリの名前 (inference か policy)

    Returns
    ----------
    path: Path
        outputs/<実験>/<日付>/<時刻>/<name>
    """
    # 日付と時刻の順で最後のもの (無ければ学習の案内)
    found = sorted(project.glob(f"outputs/*/*/*/{name}"),
                   key=lambda path: path.parent.parts[-2:])
    if not found:
        print(f"{project.relative_to(TASK_DIR)} に学習の結果がありません. "
              f"先に {project.relative_to(TASK_DIR)}/train.py を実行して"
              "ください")
        sys.exit(1)

    return found[-1]


def generate_images() -> None:
    """
    画像の生成の雑音だけの一式で画像を作り, 骨組みの見本と同じかを確かめ,
    並べた画像を書く関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 一式の読み込みと, 件数と種での推論 (種は骨組みの見本と同じ, 一式の
    # 手順の種)
    bundle_dir = latest_run(TASK_DIR / "generate" / "project", "inference")
    with open(bundle_dir / "inference.yaml", encoding="utf-8") as file:
        seed = yaml.safe_load(file)["procedure"]["seed"]
    frame = predict(load_bundle(bundle_dir), {"count": COUNT, "seed": seed})
    status = Counter(frame["status"])
    print(f"== 画像の生成 ({bundle_dir.relative_to(TASK_DIR)})")
    if set(status) != {"ok"}:
        print(f"結果: {dict(status)}, 理由: {set(frame['reason'])}")
        sys.exit(1)
    images = np.stack([np.asarray(value)[0] for value in frame["output"]])
    print(f"結果: {dict(status)}, 種: {seed}, 形: {list(images.shape)}, "
          f"値の範囲: {images.min():.3f} から {images.max():.3f}")

    # 骨組みの見本 (samples/generated_<番号>/final.png. 同じ種の先頭の
    # 数件を 0 から 1 に収めて PNG にしたもの) との最も大きい差
    samples = bundle_dir.parent / "samples"
    diffs = []
    for number in range(COUNT):
        png = samples / f"generated_{number}" / "final.png"
        if not png.is_file():
            break
        saved = np.asarray(Image.open(png), dtype=float) / 255
        saved = saved if saved.ndim == 2 else saved[..., 0]
        diffs.append(np.abs(np.clip(images[number], 0, 1) - saved).max())
    if diffs:
        print(f"骨組みの見本 {len(diffs)} 件との最も大きい差: "
              f"{max(diffs) * 255:.2f} / 255")

    # 縦と横に COUNT の平方根の数ずつ並べた画像
    side = int(np.sqrt(COUNT))
    clipped = (np.clip(images, 0, 1) * 255).astype(np.uint8)
    grid = clipped.reshape(side, side, *clipped.shape[1:])
    grid = grid.transpose(0, 2, 1, 3).reshape(
        side * clipped.shape[1], side * clipped.shape[2]
    )
    OUTPUT_DIR.mkdir(exist_ok=True)
    Image.fromarray(grid).resize(
        (grid.shape[1] * 4, grid.shape[0] * 4), Image.NEAREST
    ).save(OUTPUT_DIR / "generated.png")
    print(f"並べた画像: {(OUTPUT_DIR / 'generated.png').relative_to(TASK_DIR)}")


def run_policy() -> None:
    """
    方策の一式を ML-Inference の load_policy で読み, CartPole で守りを
    当てたまま動かす関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # モデルを渡さず, 一式の定義の写しから組み立てた方策
    policy_dir = latest_run(TASK_DIR / "policy" / "project", "policy")
    policy = load_policy(policy_dir)
    spec = policy.spec
    print(f"== 方策 ({policy_dir.relative_to(TASK_DIR)})")
    print(f"環境: {spec['metadata']['env_id']}, 守りを数にしたか: "
          f"{spec['metadata']['guards_set']}, 余った重み: "
          f"{policy.unused_keys}")

    # 1 歩ごとの act の行動での進めと, stop と handover でのエピソードの終わり
    env = gym.make(spec["metadata"]["env_id"])
    rewards, counts = [], Counter()
    for episode in range(EPISODES):
        observation, _ = env.reset(seed=EPISODE_SEED + episode)
        total, done = 0.0, False
        while not done:
            result = policy.act(observation)
            counts[result["status"]] += 1
            if result["action"] is None:
                break
            observation, reward, terminated, truncated, _ = env.step(
                result["action"]
            )
            total += float(reward)
            done = terminated or truncated
        rewards.append(total)
    env.close()
    print(f"報酬の和: 平均 {np.mean(rewards):.1f}, 最小 {min(rewards):.1f}, "
          f"最大 {max(rewards):.1f} ({EPISODES} エピソード)")
    print(f"守りの結果: {dict(counts)}")


def main() -> None:
    """
    書き換えない骨組みが作った 2 つの一式を, 推論の段階 (ML-Inference) で
    読んで動かす関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 画像の生成と方策
    generate_images()
    run_policy()


if __name__ == "__main__":
    main()
