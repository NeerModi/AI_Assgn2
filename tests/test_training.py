from __future__ import annotations

from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from src.environment import make_env


def _mini_env_config() -> dict:
    return {
        "num_vehicles": 4,
        "num_ports": 2,
        "station_max_power_kw": 12.0,
        "time_step_minutes": 30,
        "episode_horizon_hours": 8,
        "charging_efficiency": 0.95,
        "vehicle": {
            "arrival_window_hours": [0.0, 2.0],
            "departure_after_arrival_hours": [2.0, 3.5],
            "battery_capacity_kwh": [40.0, 40.0],
            "initial_soc": [0.15, 0.25],
            "target_soc": [0.8, 0.9],
            "max_charge_power_kw": [6.0, 7.0],
        },
        "price": {
            "base": 0.2,
            "daily_amplitude": 0.05,
            "noise_std": 0.0,
            "min": 0.1,
            "max": 0.5,
        },
        "reward": {
            "alpha": 1.0,
            "beta": 0.3,
            "gamma": 2.0,
            "delta": 1.0,
            "eta": 0.2,
        },
    }


def test_ppo_smoke_train_save_load(tmp_path: Path) -> None:
    env_cfg = _mini_env_config()
    vec_env = DummyVecEnv([lambda: make_env(env_cfg)])

    model = PPO(
        "MlpPolicy",
        vec_env,
        n_steps=32,
        batch_size=32,
        learning_rate=3e-4,
        gamma=0.99,
        verbose=0,
        seed=123,
    )
    model.learn(total_timesteps=256)

    model_path = tmp_path / "smoke_model"
    model.save(str(model_path))

    loaded = PPO.load(str(model_path))
    obs = vec_env.reset()
    action, _ = loaded.predict(obs, deterministic=True)
    assert action is not None

    vec_env.close()
