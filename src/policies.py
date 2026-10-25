from __future__ import annotations

from typing import List, Optional

import numpy as np
from stable_baselines3 import PPO

from .environment import EVChargingEnv


def eligible_vehicle_indices(env: EVChargingEnv) -> List[int]:
    """Returns indices of vehicles that can be started now."""
    if env._free_ports() <= 0:  # noqa: SLF001 - intentional for simple baseline access
        return []
    eligible: List[int] = []
    for i, v in enumerate(env.vehicles):
        if v.is_arrived and (not v.is_departed) and (not v.is_completed) and (not v.is_charging):
            eligible.append(i)
    return eligible


def fcfs_action(env: EVChargingEnv) -> int:
    """Earliest-arrived-first among currently eligible vehicles. Returns action id."""
    candidates = eligible_vehicle_indices(env)
    if not candidates:
        return 0
    chosen = min(candidates, key=lambda idx: env.vehicles[idx].arrival_time_h)
    return chosen + 1


def random_action(env: EVChargingEnv, rng: np.random.Generator) -> int:
    """Randomly select one eligible vehicle or no-op."""
    candidates = eligible_vehicle_indices(env)
    if not candidates:
        return 0
    chosen = int(rng.choice(candidates))
    return chosen + 1


class PPOPolicyWrapper:
    """Thin adapter to make SB3 PPO policy callable like baseline policies."""

    def __init__(self, model: PPO):
        self.model = model

    @classmethod
    def from_path(cls, model_path: str) -> "PPOPolicyWrapper":
        model = PPO.load(model_path)
        return cls(model=model)

    def action(self, obs: np.ndarray, deterministic: bool = True) -> int:
        action, _ = self.model.predict(obs, deterministic=deterministic)
        if isinstance(action, np.ndarray):
            return int(action.item())
        return int(action)


def maybe_load_ppo(model_path: Optional[str]) -> Optional[PPOPolicyWrapper]:
    if not model_path:
        return None
    return PPOPolicyWrapper.from_path(model_path)
