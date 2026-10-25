from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Vehicle:
    """Represents a single EV in the simulation."""

    vehicle_id: int
    arrival_time_h: float
    departure_time_h: float
    battery_capacity_kwh: float
    initial_soc: float
    target_soc: float
    max_charge_power_kw: float

    current_soc: float = field(init=False)
    current_energy_kwh: float = field(init=False)
    energy_required_kwh: float = field(init=False)
    is_arrived: bool = field(default=False)
    is_departed: bool = field(default=False)
    is_charging: bool = field(default=False)
    is_completed: bool = field(default=False)
    charging_start_time_h: Optional[float] = field(default=None)
    waiting_time_h: float = field(default=0.0)
    completion_time_h: Optional[float] = field(default=None)
    missed_deadline_penalized: bool = field(default=False)

    def __post_init__(self) -> None:
        self.current_soc = float(min(max(self.initial_soc, 0.0), 1.0))
        self.current_energy_kwh = self.current_soc * self.battery_capacity_kwh
        target_energy_kwh = min(max(self.target_soc, 0.0), 1.0) * self.battery_capacity_kwh
        self.energy_required_kwh = max(0.0, target_energy_kwh - self.current_energy_kwh)
        if self.energy_required_kwh <= 1e-9:
            self.is_completed = True

    @property
    def has_arrived_and_present(self) -> bool:
        return self.is_arrived and not self.is_departed

    @property
    def remaining_energy_kwh(self) -> float:
        return max(0.0, self.energy_required_kwh)

    @property
    def waiting(self) -> bool:
        return (
            self.has_arrived_and_present
            and not self.is_completed
            and not self.is_charging
        )

    def add_energy(self, battery_energy_kwh: float, event_time_h: float) -> float:
        """Adds battery energy and returns actual accepted battery energy."""
        if self.is_completed or self.is_departed or battery_energy_kwh <= 0:
            return 0.0

        accepted = min(self.energy_required_kwh, battery_energy_kwh)
        self.energy_required_kwh = max(0.0, self.energy_required_kwh - accepted)
        self.current_energy_kwh = min(
            self.battery_capacity_kwh,
            self.current_energy_kwh + accepted,
        )
        self.current_soc = min(1.0, self.current_energy_kwh / self.battery_capacity_kwh)

        if self.energy_required_kwh <= 1e-6:
            self.is_completed = True
            self.is_charging = False
            if self.completion_time_h is None:
                self.completion_time_h = event_time_h

        return accepted
