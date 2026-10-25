from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from src.environment import make_env
from src.metrics import build_episode_row, save_results
from src.policies import PPOPolicyWrapper
from src.utils import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained PPO policy")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--model-path", type=str, default="models/ppo_ev_charging.zip")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--deterministic", action="store_true", default=False)
    return parser.parse_args()


def run_ppo_eval(
    env_cfg: Dict[str, Any],
    policy: PPOPolicyWrapper,
    episodes: int,
    base_seed: int,
    deterministic: bool,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for episode in range(episodes):
        seed = base_seed + episode
        env = make_env(env_cfg)
        obs, _ = env.reset(seed=seed)
        done = False
        truncated = False

        while not (done or truncated):
            action = policy.action(obs=obs, deterministic=deterministic)
            obs, _, done, truncated, info = env.step(action)

        summary = info.get("episode_summary")
        if summary is None:
            raise RuntimeError("Episode ended without summary in info")
        rows.append(build_episode_row(summary=summary, episode_id=episode, policy_name="ppo", seed=seed))
        env.close()

    return rows


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    env_cfg = cfg["env"]
    project_seed = int(cfg.get("project", {}).get("seed", 42))
    eval_cfg = cfg.get("evaluation", {})

    episodes = int(args.episodes if args.episodes is not None else eval_cfg.get("episodes", 50))
    base_seed = int(args.seed if args.seed is not None else project_seed + 1_000)
    deterministic = bool(args.deterministic or eval_cfg.get("deterministic_policy", True))

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}. Train first with train.py")

    ppo = PPOPolicyWrapper.from_path(str(model_path))
    rows = run_ppo_eval(
        env_cfg=env_cfg,
        policy=ppo,
        episodes=episodes,
        base_seed=base_seed,
        deterministic=deterministic,
    )

    outputs = save_results(episode_rows=rows, results_dir=Path("results"), stem="ppo_evaluation")
    print("[evaluate] Saved files:")
    for key, path in outputs.items():
        print(f"  - {key}: {path}")


if __name__ == "__main__":
    main()
