from __future__ import annotations

import numpy as np

from src.environment import EVChargingEnv


def _test_config() -> dict:
    return {
        "num_vehicles": 4,
        "num_ports": 2,
        "station_max_power_kw": 12.0,
        "time_step_minutes": 30,
        "episode_horizon_hours": 8,
        "charging_efficiency": 0.95,
        "vehicle": {
            "arrival_window_hours": [0.0, 2.0],
            "departure_after_arrival_hours": [2.0, 3.0],
            "battery_capacity_kwh": [40.0, 40.0],
            "initial_soc": [0.2, 0.2],
            "target_soc": [0.8, 0.8],
            "max_charge_power_kw": [6.0, 6.0],
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
            "beta": 0.2,
            "gamma": 2.0,
            "delta": 1.0,
            "eta": 0.3,
        },
    }


def test_reset_observation_shape_and_finite_values() -> None:
    env = EVChargingEnv(_test_config())
    obs, info = env.reset(seed=123)
    assert obs.shape == env.observation_space.shape
    assert np.all(np.isfinite(obs))
    assert "price" in info


def test_observation_bounds() -> None:
    env = EVChargingEnv(_test_config())
    obs, _ = env.reset(seed=123)
    assert np.all(obs <= 1.0 + 1e-6)
    assert np.all(obs >= -1.0 - 1e-6)


def test_valid_and_invalid_action_handling() -> None:
    env = EVChargingEnv(_test_config())
    obs, _ = env.reset(seed=3)

    # No-op should always be valid.
    obs, reward, terminated, truncated, info = env.step(0)
    assert np.isfinite(reward)
    assert not (terminated and truncated)
    assert info["invalid_action"] == 0

    # Selecting a not-yet-eligible or unavailable vehicle is handled safely.
    _, _, _, _, info2 = env.step(env.max_vehicles)
    assert info2["invalid_action"] in (0, 1)


def test_arrival_departure_and_completion_transitions() -> None:
    env = EVChargingEnv(_test_config())
    obs, _ = env.reset(seed=7)

    seen_arrival = False
    seen_departure = False
    done = False
    trunc = False

    while not (done or trunc):
        eligible = [
            i + 1
            for i, v in enumerate(env.vehicles)
            if v.is_arrived and (not v.is_departed) and (not v.is_completed) and (not v.is_charging)
        ]
        action = eligible[0] if eligible else 0
        obs, _, done, trunc, info = env.step(action)

        if any(v.is_arrived for v in env.vehicles):
            seen_arrival = True
        if any(v.is_departed for v in env.vehicles):
            seen_departure = True

    assert seen_arrival
    assert seen_departure
    assert "episode_summary" in info


def test_power_and_soc_constraints() -> None:
    env = EVChargingEnv(_test_config())
    obs, _ = env.reset(seed=11)

    done = False
    trunc = False
    while not (done or trunc):
        eligible = [
            i + 1
            for i, v in enumerate(env.vehicles)
            if v.is_arrived and (not v.is_departed) and (not v.is_completed) and (not v.is_charging)
        ]
        action = eligible[0] if eligible else 0
        obs, _, done, trunc, info = env.step(action)

        power_kw = info["step_grid_energy_kwh"] / env.step_h
        assert power_kw <= env.station_max_power_kw + 1e-6

        for v in env.vehicles:
            assert 0.0 <= v.current_soc <= 1.0 + 1e-9
            assert v.remaining_energy_kwh >= -1e-6


def test_reward_component_accounting_and_seed_reproducibility() -> None:
    env1 = EVChargingEnv(_test_config())
    env2 = EVChargingEnv(_test_config())

    obs1, _ = env1.reset(seed=99)
    obs2, _ = env2.reset(seed=99)
    assert np.allclose(obs1, obs2)

    actions = [0, 1, 2, 0, 3, 0]
    for a in actions:
        _, r1, d1, t1, i1 = env1.step(min(a, env1.action_space.n - 1))
        _, r2, d2, t2, i2 = env2.step(min(a, env2.action_space.n - 1))
        assert np.isclose(r1, r2)
        assert i1["reward_components"].keys() == i2["reward_components"].keys()
        if d1 or t1:
            break
