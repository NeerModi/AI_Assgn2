from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .vehicle import Vehicle


@dataclass
class RewardWeights:
    alpha: float = 1.0
    beta: float = 0.6
    gamma: float = 6.0
    delta: float = 3.5
    eta: float = 0.25


class EVChargingEnv(gym.Env[np.ndarray, int]):
    """
    Custom EV charging environment.

    Action space:
    - 0: no-op
    - 1..max_vehicles: attempt to start charging vehicle (action - 1)

    Observation ordering:
    - Per vehicle (repeated max_vehicles times, 9 features each):
      [arrived, departed, waiting, charging, completed,
       soc, remaining_energy_norm, time_to_departure_norm, max_power_norm]
    - Global (8 features):
      [current_price_norm, hour_sin, hour_cos, free_ports_norm,
       active_ports_norm, queue_len_norm, station_power_used_norm,
       station_power_left_norm]
    """

    metadata = {"render_modes": []}

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config

        self.max_vehicles = int(config.get("num_vehicles", 8))
        self.num_ports = int(config.get("num_ports", 3))
        self.station_max_power_kw = float(config.get("station_max_power_kw", 30.0))
        self.time_step_minutes = float(config.get("time_step_minutes", 15.0))
        self.step_h = self.time_step_minutes / 60.0
        self.episode_horizon_h = float(config.get("episode_horizon_hours", 24.0))
        self.horizon_steps = int(round(self.episode_horizon_h / self.step_h))
        self.charging_efficiency = float(config.get("charging_efficiency", 0.93))

        vehicle_cfg = config.get("vehicle", {})
        self.arrival_window_h = tuple(vehicle_cfg.get("arrival_window_hours", [0.0, 18.0]))
        self.departure_after_arrival_h = tuple(
            vehicle_cfg.get("departure_after_arrival_hours", [2.0, 10.0])
        )
        self.capacity_kwh_range = tuple(vehicle_cfg.get("battery_capacity_kwh", [35.0, 75.0]))
        self.initial_soc_range = tuple(vehicle_cfg.get("initial_soc", [0.1, 0.6]))
        self.target_soc_range = tuple(vehicle_cfg.get("target_soc", [0.75, 0.95]))
        self.max_vehicle_power_range = tuple(vehicle_cfg.get("max_charge_power_kw", [7.0, 11.0]))

        price_cfg = config.get("price", {})
        self.price_base = float(price_cfg.get("base", 0.20))
        self.price_amp = float(price_cfg.get("daily_amplitude", 0.08))
        self.price_noise_std = float(price_cfg.get("noise_std", 0.02))
        self.price_min = float(price_cfg.get("min", 0.05))
        self.price_max = float(price_cfg.get("max", 0.55))

        reward_cfg = config.get("reward", {})
        self.reward_weights = RewardWeights(
            alpha=float(reward_cfg.get("alpha", 1.0)),
            beta=float(reward_cfg.get("beta", 0.6)),
            gamma=float(reward_cfg.get("gamma", 6.0)),
            delta=float(reward_cfg.get("delta", 3.5)),
            eta=float(reward_cfg.get("eta", 0.25)),
        )

        self.per_vehicle_obs_dim = 9
        self.global_obs_dim = 8
        obs_dim = self.max_vehicles * self.per_vehicle_obs_dim + self.global_obs_dim

        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(self.max_vehicles + 1)

        self.np_random = np.random.default_rng(0)
        self.vehicles: List[Vehicle] = []
        self.current_step = 0
        self.current_time_h = 0.0

        self.price_now = self.price_base

        self.total_cost = 0.0
        self.total_grid_energy_kwh = 0.0
        self.total_battery_energy_kwh = 0.0
        self.total_waiting_h = 0.0
        self.total_invalid_actions = 0
        self.total_missed = 0
        self.total_completed = 0
        self.arrived_count = 0
        self.episode_return = 0.0

        self.reward_component_totals = {
            "cost": 0.0,
            "waiting_h": 0.0,
            "missed": 0.0,
            "completed": 0.0,
            "invalid": 0.0,
        }

        self.history: Dict[str, List[float]] = {}

    def _reset_accumulators(self) -> None:
        self.total_cost = 0.0
        self.total_grid_energy_kwh = 0.0
        self.total_battery_energy_kwh = 0.0
        self.total_waiting_h = 0.0
        self.total_invalid_actions = 0
        self.total_missed = 0
        self.total_completed = 0
        self.arrived_count = 0
        self.episode_return = 0.0
        self.reward_component_totals = {
            "cost": 0.0,
            "waiting_h": 0.0,
            "missed": 0.0,
            "completed": 0.0,
            "invalid": 0.0,
        }
        self.history = {
            "time_h": [],
            "price": [],
            "active_ports": [],
            "queue_len": [],
            "power_kw": [],
            "cost": [],
        }

    def _sample_vehicles(self) -> List[Vehicle]:
        vehicles: List[Vehicle] = []
        for i in range(self.max_vehicles):
            arrival_h = float(
                self.np_random.uniform(self.arrival_window_h[0], self.arrival_window_h[1])
            )
            dep_delta_h = float(
                self.np_random.uniform(
                    self.departure_after_arrival_h[0],
                    self.departure_after_arrival_h[1],
                )
            )
            departure_h = min(self.episode_horizon_h, arrival_h + dep_delta_h)

            capacity = float(
                self.np_random.uniform(self.capacity_kwh_range[0], self.capacity_kwh_range[1])
            )
            init_soc = float(
                self.np_random.uniform(self.initial_soc_range[0], self.initial_soc_range[1])
            )
            target_soc = float(
                self.np_random.uniform(self.target_soc_range[0], self.target_soc_range[1])
            )
            if target_soc < init_soc:
                target_soc = min(1.0, init_soc + 0.15)

            max_power = float(
                self.np_random.uniform(
                    self.max_vehicle_power_range[0],
                    self.max_vehicle_power_range[1],
                )
            )

            vehicles.append(
                Vehicle(
                    vehicle_id=i,
                    arrival_time_h=arrival_h,
                    departure_time_h=departure_h,
                    battery_capacity_kwh=capacity,
                    initial_soc=init_soc,
                    target_soc=target_soc,
                    max_charge_power_kw=max_power,
                )
            )

        vehicles.sort(key=lambda v: v.arrival_time_h)
        for idx, v in enumerate(vehicles):
            v.vehicle_id = idx
        return vehicles

    def _price_at(self, time_h: float) -> float:
        daily_angle = (2.0 * np.pi * (time_h % 24.0)) / 24.0
        deterministic = self.price_base + self.price_amp * (np.sin(daily_angle - np.pi / 2.0) + 1.0) / 2.0
        noise = float(self.np_random.normal(0.0, self.price_noise_std))
        return float(np.clip(deterministic + noise, self.price_min, self.price_max))

    def _update_arrivals(self) -> None:
        for v in self.vehicles:
            if not v.is_arrived and self.current_time_h >= v.arrival_time_h:
                v.is_arrived = True

    def _active_charging(self) -> List[Vehicle]:
        return [
            v
            for v in self.vehicles
            if v.is_arrived and not v.is_departed and not v.is_completed and v.is_charging
        ]

    def _waiting_vehicles(self) -> List[Vehicle]:
        return [v for v in self.vehicles if v.waiting]

    def _free_ports(self) -> int:
        return self.num_ports - len(self._active_charging())

    def _allocate_fair_power(self, charging_vehicles: List[Vehicle]) -> Dict[int, float]:
        if not charging_vehicles:
            return {}

        remaining = list(charging_vehicles)
        remaining_power = self.station_max_power_kw
        allocation: Dict[int, float] = {v.vehicle_id: 0.0 for v in charging_vehicles}

        # Progressive filling for fair allocation with per-vehicle caps.
        while remaining and remaining_power > 1e-9:
            share = remaining_power / float(len(remaining))
            capped = [v for v in remaining if v.max_charge_power_kw <= share + 1e-9]
            if not capped:
                for v in remaining:
                    allocation[v.vehicle_id] += share
                remaining_power = 0.0
                break

            for v in capped:
                allocation[v.vehicle_id] += v.max_charge_power_kw
                remaining_power -= v.max_charge_power_kw
                remaining.remove(v)

        return allocation

    def _build_observation(self) -> np.ndarray:
        obs: List[float] = []
        for v in self.vehicles:
            time_to_departure = max(0.0, v.departure_time_h - self.current_time_h)
            rem_energy_norm = (
                v.remaining_energy_kwh / max(v.battery_capacity_kwh, 1e-6)
                if v.battery_capacity_kwh > 0
                else 0.0
            )
            obs.extend(
                [
                    float(v.is_arrived),
                    float(v.is_departed),
                    float(v.waiting),
                    float(v.is_charging),
                    float(v.is_completed),
                    float(v.current_soc),
                    float(np.clip(rem_energy_norm, 0.0, 1.0)),
                    float(np.clip(time_to_departure / self.episode_horizon_h, 0.0, 1.0)),
                    float(np.clip(v.max_charge_power_kw / self.station_max_power_kw, 0.0, 1.0)),
                ]
            )

        active = self._active_charging()
        queue = self._waiting_vehicles()
        free_ports = self._free_ports()

        power_used_kw = sum(v.max_charge_power_kw for v in active)
        power_used_kw = min(power_used_kw, self.station_max_power_kw)

        hour_angle = (2.0 * np.pi * (self.current_time_h % 24.0)) / 24.0
        global_features = [
            float(np.clip((self.price_now - self.price_min) / (self.price_max - self.price_min + 1e-9), 0.0, 1.0)),
            float(np.sin(hour_angle)),
            float(np.cos(hour_angle)),
            float(np.clip(free_ports / max(1, self.num_ports), 0.0, 1.0)),
            float(np.clip(len(active) / max(1, self.num_ports), 0.0, 1.0)),
            float(np.clip(len(queue) / max(1, self.max_vehicles), 0.0, 1.0)),
            float(np.clip(power_used_kw / self.station_max_power_kw, 0.0, 1.0)),
            float(np.clip((self.station_max_power_kw - power_used_kw) / self.station_max_power_kw, 0.0, 1.0)),
        ]
        obs.extend(global_features)

        arr = np.asarray(obs, dtype=np.float32)
        arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=-1.0)
        return arr

    def _episode_done(self) -> bool:
        all_departed_or_completed = all(v.is_departed or v.is_completed for v in self.vehicles)
        return all_departed_or_completed

    def _update_departures(self) -> int:
        missed_this_step = 0
        for v in self.vehicles:
            if v.is_arrived and not v.is_departed and self.current_time_h >= v.departure_time_h:
                v.is_departed = True
                v.is_charging = False
                if (not v.is_completed) and (not v.missed_deadline_penalized):
                    v.missed_deadline_penalized = True
                    missed_this_step += 1
        return missed_this_step

    def _collect_episode_summary(self) -> Dict[str, Any]:
        arrived = [v for v in self.vehicles if v.is_arrived]
        completed = [v for v in arrived if v.is_completed]
        started = [v for v in arrived if v.charging_start_time_h is not None]
        missed = [v for v in arrived if (v.is_departed and not v.is_completed)]

        mean_wait_h = float(np.mean([v.waiting_time_h for v in arrived])) if arrived else 0.0
        mean_wait_started_h = float(np.mean([v.waiting_time_h for v in started])) if started else 0.0

        port_utilization = 0.0
        power_utilization = 0.0
        if self.history["active_ports"]:
            port_utilization = float(
                np.mean(np.asarray(self.history["active_ports"], dtype=float) / max(1, self.num_ports))
            )
        if self.history["power_kw"]:
            power_utilization = float(
                np.mean(np.asarray(self.history["power_kw"], dtype=float) / self.station_max_power_kw)
            )

        return {
            "arrived_vehicles": len(arrived),
            "completed_vehicles": len(completed),
            "missed_vehicles": len(missed),
            "completion_rate": (len(completed) / len(arrived)) if arrived else 0.0,
            "missed_rate": (len(missed) / len(arrived)) if arrived else 0.0,
            "total_electricity_cost": self.total_cost,
            "avg_cost_per_completed_vehicle": self.total_cost / max(1, len(completed)),
            "mean_waiting_time_minutes": mean_wait_h * 60.0,
            "mean_waiting_time_started_minutes": mean_wait_started_h * 60.0,
            "total_energy_delivered_kwh": self.total_battery_energy_kwh,
            "total_grid_energy_kwh": self.total_grid_energy_kwh,
            "port_utilization": port_utilization,
            "power_utilization": power_utilization,
            "episode_return": self.episode_return,
            "invalid_actions": self.total_invalid_actions,
            "reward_components_total": dict(self.reward_component_totals),
            "history": self.history,
        }

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Dict[str, Any] | None = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        self.current_step = 0
        self.current_time_h = 0.0
        self._reset_accumulators()
        self.vehicles = self._sample_vehicles()
        self.price_now = self._price_at(self.current_time_h)
        self._update_arrivals()

        obs = self._build_observation()
        info = {
            "time_h": self.current_time_h,
            "price": self.price_now,
        }
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        if not self.action_space.contains(action):
            raise ValueError(f"Action {action} is out of bounds for {self.action_space}")

        self._update_arrivals()

        invalid_action = 0
        completions_this_step = 0

        if action > 0:
            idx = action - 1
            chosen = self.vehicles[idx]
            can_start = (
                chosen.is_arrived
                and not chosen.is_departed
                and not chosen.is_completed
                and not chosen.is_charging
                and self._free_ports() > 0
            )
            if can_start:
                chosen.is_charging = True
                if chosen.charging_start_time_h is None:
                    chosen.charging_start_time_h = self.current_time_h
            else:
                invalid_action = 1
                self.total_invalid_actions += 1

        # Waiting time accumulates for arrived/present/waiting vehicles.
        waiting_added_h = 0.0
        for v in self._waiting_vehicles():
            v.waiting_time_h += self.step_h
            waiting_added_h += self.step_h

        charging_vehicles = self._active_charging()
        fair_alloc_kw = self._allocate_fair_power(charging_vehicles)

        step_grid_energy_kwh = 0.0
        step_battery_energy_kwh = 0.0
        step_cost = 0.0
        total_power_kw = 0.0

        for v in charging_vehicles:
            proposed_power_kw = max(0.0, fair_alloc_kw.get(v.vehicle_id, 0.0))
            max_power_from_need_kw = (
                (v.remaining_energy_kwh / max(self.charging_efficiency, 1e-9)) / self.step_h
            )
            power_kw = max(0.0, min(proposed_power_kw, max_power_from_need_kw, v.max_charge_power_kw))

            grid_energy_kwh = power_kw * self.step_h
            battery_energy_kwh = grid_energy_kwh * self.charging_efficiency

            accepted_battery_energy = v.add_energy(
                battery_energy_kwh=battery_energy_kwh,
                event_time_h=self.current_time_h + self.step_h,
            )
            if accepted_battery_energy + 1e-9 < battery_energy_kwh:
                battery_energy_kwh = accepted_battery_energy
                grid_energy_kwh = accepted_battery_energy / max(self.charging_efficiency, 1e-9)
                power_kw = grid_energy_kwh / self.step_h

            total_power_kw += power_kw
            step_grid_energy_kwh += grid_energy_kwh
            step_battery_energy_kwh += battery_energy_kwh

            if v.is_completed:
                completions_this_step += 1

        step_cost = step_grid_energy_kwh * self.price_now

        self.total_grid_energy_kwh += step_grid_energy_kwh
        self.total_battery_energy_kwh += step_battery_energy_kwh
        self.total_cost += step_cost

        self.current_step += 1
        self.current_time_h = min(self.episode_horizon_h, self.current_time_h + self.step_h)

        missed_this_step = self._update_departures()
        self.total_missed += missed_this_step
        self.total_completed += completions_this_step

        reward = (
            -self.reward_weights.alpha * step_cost
            -self.reward_weights.beta * waiting_added_h
            -self.reward_weights.gamma * float(missed_this_step)
            +self.reward_weights.delta * float(completions_this_step)
            -self.reward_weights.eta * float(invalid_action)
        )
        self.episode_return += reward

        self.reward_component_totals["cost"] += step_cost
        self.reward_component_totals["waiting_h"] += waiting_added_h
        self.reward_component_totals["missed"] += float(missed_this_step)
        self.reward_component_totals["completed"] += float(completions_this_step)
        self.reward_component_totals["invalid"] += float(invalid_action)

        active_count = len(self._active_charging())
        queue_count = len(self._waiting_vehicles())
        self.history["time_h"].append(self.current_time_h)
        self.history["price"].append(self.price_now)
        self.history["active_ports"].append(float(active_count))
        self.history["queue_len"].append(float(queue_count))
        self.history["power_kw"].append(float(total_power_kw))
        self.history["cost"].append(float(step_cost))

        self.price_now = self._price_at(self.current_time_h)

        terminated = self._episode_done()
        truncated = self.current_step >= self.horizon_steps

        self.arrived_count = sum(1 for v in self.vehicles if v.is_arrived)

        info: Dict[str, Any] = {
            "time_h": self.current_time_h,
            "price": self.price_now,
            "step_cost": step_cost,
            "step_grid_energy_kwh": step_grid_energy_kwh,
            "step_battery_energy_kwh": step_battery_energy_kwh,
            "waiting_added_h": waiting_added_h,
            "missed_this_step": missed_this_step,
            "completed_this_step": completions_this_step,
            "invalid_action": invalid_action,
            "reward_components": {
                "cost_term": -self.reward_weights.alpha * step_cost,
                "waiting_term": -self.reward_weights.beta * waiting_added_h,
                "missed_term": -self.reward_weights.gamma * float(missed_this_step),
                "completion_term": self.reward_weights.delta * float(completions_this_step),
                "invalid_term": -self.reward_weights.eta * float(invalid_action),
            },
        }

        if terminated or truncated:
            info["episode_summary"] = self._collect_episode_summary()

        obs = self._build_observation()
        return obs, float(reward), terminated, truncated, info

    def render(self) -> None:
        return None


def make_env(config: Dict[str, Any]) -> EVChargingEnv:
    """Factory helper for scripts and tests."""
    return EVChargingEnv(config=config)
