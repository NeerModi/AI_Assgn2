from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from src.environment import make_env
from src.utils import ensure_dir, load_config, set_global_seeds


class TrainingProgressCallback(BaseCallback):
    """Captures episodic training rewards from monitor info."""

    def __init__(self) -> None:
        super().__init__()
        self.records: List[Dict[str, float]] = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            episode = info.get("episode")
            if episode:
                self.records.append(
                    {
                        "timesteps": float(self.num_timesteps),
                        "episode_reward": float(episode["r"]),
                        "episode_length": float(episode["l"]),
                    }
                )
        return True


def build_train_env(env_cfg: Dict[str, Any], base_seed: int, n_envs: int) -> VecMonitor:
    def _factory(rank: int):
        def _init():
            env = make_env(env_cfg)
            env = Monitor(env)
            env.reset(seed=base_seed + rank)
            return env

        return _init

    vec_env = DummyVecEnv([_factory(i) for i in range(n_envs)])
    return VecMonitor(vec_env)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO for EV coordinated charging")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML config")
    parser.add_argument("--timesteps", type=int, default=None, help="Override total timesteps")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    env_cfg = cfg["env"]
    train_cfg = cfg["training"]
    project_cfg = cfg.get("project", {})

    seed = int(args.seed if args.seed is not None else project_cfg.get("seed", 42))
    total_timesteps = int(
        args.timesteps if args.timesteps is not None else train_cfg.get("total_timesteps", 50_000)
    )

    set_global_seeds(seed)

    models_dir = ensure_dir("models")
    logs_dir = ensure_dir("logs")
    results_dir = ensure_dir("results")

    n_envs = int(train_cfg.get("n_envs", 4))
    train_env = build_train_env(env_cfg=env_cfg, base_seed=seed, n_envs=n_envs)

    eval_env = DummyVecEnv(
        [
            lambda: Monitor(make_env(env_cfg))
        ]
    )
    eval_env.reset()

    progress_callback = TrainingProgressCallback()
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(models_dir / "best_model"),
        log_path=str(logs_dir / "eval"),
        eval_freq=max(1, int(train_cfg.get("eval_freq", 5000)) // max(1, n_envs)),
        n_eval_episodes=int(train_cfg.get("n_eval_episodes", 8)),
        deterministic=bool(train_cfg.get("deterministic_eval", True)),
        render=False,
    )

    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=float(train_cfg.get("learning_rate", 3e-4)),
        n_steps=int(train_cfg.get("n_steps", 256)),
        batch_size=int(train_cfg.get("batch_size", 256)),
        gamma=float(train_cfg.get("gamma", 0.99)),
        gae_lambda=float(train_cfg.get("gae_lambda", 0.95)),
        clip_range=float(train_cfg.get("clip_range", 0.2)),
        ent_coef=float(train_cfg.get("ent_coef", 0.01)),
        vf_coef=float(train_cfg.get("vf_coef", 0.5)),
        max_grad_norm=float(train_cfg.get("max_grad_norm", 0.5)),
        tensorboard_log=str(logs_dir / "tb"),
        seed=seed,
        verbose=1,
    )

    print(f"[train] Starting PPO training for {total_timesteps} timesteps with seed={seed}")
    model.learn(total_timesteps=total_timesteps, callback=[progress_callback, eval_callback])

    model_path = models_dir / "ppo_ev_charging"
    model.save(str(model_path))
    print(f"[train] Saved trained model to {model_path}.zip")

    progress_path = results_dir / "training_progress.csv"
    if progress_callback.records:
        import pandas as pd

        pd.DataFrame(progress_callback.records).to_csv(progress_path, index=False)
        print(f"[train] Saved training progress to {progress_path}")

    metadata = {
        "seed": seed,
        "total_timesteps": total_timesteps,
        "config_path": str(Path(args.config).resolve()),
        "env_config": env_cfg,
        "training_config": train_cfg,
        "model_path": str((model_path.with_suffix(".zip")).resolve()),
    }
    (results_dir / "training_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
