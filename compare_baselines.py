from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from src.environment import EVChargingEnv, make_env
from src.metrics import build_episode_row, save_results
from src.policies import PPOPolicyWrapper, fcfs_action, random_action
from src.utils import load_config


PolicyFn = Callable[[EVChargingEnv, np.ndarray], int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare PPO, FCFS, and Random baselines")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--model-path", type=str, default="models/ppo_ev_charging.zip")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--include-ppo", action="store_true", default=True)
    parser.add_argument("--no-ppo", action="store_true", default=False)
    return parser.parse_args()


def rollout(
    env_cfg: Dict[str, Any],
    policy_name: str,
    policy_fn: PolicyFn,
    episodes: int,
    base_seed: int,
    save_representative: bool,
) -> tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    representative: Optional[Dict[str, Any]] = None

    for episode in range(episodes):
        seed = base_seed + episode
        env = make_env(env_cfg)
        obs, _ = env.reset(seed=seed)
        done = False
        truncated = False

        while not (done or truncated):
            action = policy_fn(env, obs)
            obs, _, done, truncated, info = env.step(action)

        summary = info["episode_summary"]
        rows.append(
            build_episode_row(summary=summary, episode_id=episode, policy_name=policy_name, seed=seed)
        )

        if save_representative and representative is None:
            representative = {
                "policy": policy_name,
                "seed": seed,
                "history": summary["history"],
                "summary": {k: v for k, v in summary.items() if k != "history"},
            }

        env.close()

    return rows, representative


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    env_cfg = cfg["env"]
    project_seed = int(cfg.get("project", {}).get("seed", 42))
    baseline_cfg = cfg.get("baselines", {})

    episodes = int(args.episodes if args.episodes is not None else baseline_cfg.get("episodes", 100))
    base_seed = int(args.seed if args.seed is not None else project_seed + 2_000)

    results_rows: List[Dict[str, Any]] = []
    representative_store: Dict[str, Dict[str, Any]] = {}

    def fcfs_policy(env: EVChargingEnv, obs: np.ndarray) -> int:
        del obs
        return fcfs_action(env)

    def rand_policy(env: EVChargingEnv, obs: np.ndarray) -> int:
        del obs
        return random_action(env, env.np_random)

    fcfs_rows, fcfs_rep = rollout(
        env_cfg=env_cfg,
        policy_name="fcfs",
        policy_fn=fcfs_policy,
        episodes=episodes,
        base_seed=base_seed,
        save_representative=True,
    )
    results_rows.extend(fcfs_rows)
    if fcfs_rep is not None:
        representative_store["fcfs"] = fcfs_rep

    random_rows, random_rep = rollout(
        env_cfg=env_cfg,
        policy_name="random",
        policy_fn=rand_policy,
        episodes=episodes,
        base_seed=base_seed,
        save_representative=True,
    )
    results_rows.extend(random_rows)
    if random_rep is not None:
        representative_store["random"] = random_rep

    model_path = Path(args.model_path)
    include_ppo = args.include_ppo and (not args.no_ppo)

    if include_ppo and model_path.exists():
        ppo = PPOPolicyWrapper.from_path(str(model_path))

        def ppo_policy(env: EVChargingEnv, obs: np.ndarray) -> int:
            del env
            return ppo.action(obs, deterministic=True)

        ppo_rows, ppo_rep = rollout(
            env_cfg=env_cfg,
            policy_name="ppo",
            policy_fn=ppo_policy,
            episodes=episodes,
            base_seed=base_seed,
            save_representative=True,
        )
        results_rows.extend(ppo_rows)
        if ppo_rep is not None:
            representative_store["ppo"] = ppo_rep
    elif include_ppo:
        print(f"[compare] PPO model not found at {model_path}; skipping PPO policy")

    output_paths = save_results(
        episode_rows=results_rows,
        results_dir=Path("results"),
        stem="policy_comparison",
    )

    rep_path = Path("results") / "representative_episodes.json"
    rep_path.write_text(json.dumps(representative_store, indent=2), encoding="utf-8")

    print("[compare] Saved files:")
    for key, value in output_paths.items():
        print(f"  - {key}: {value}")
    print(f"  - representative: {rep_path}")


if __name__ == "__main__":
    main()
