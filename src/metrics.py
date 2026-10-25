from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def build_episode_row(summary: Dict, episode_id: int, policy_name: str, seed: int) -> Dict:
    """Flattens episode summary info into one CSV row."""
    return {
        "episode": episode_id,
        "policy": policy_name,
        "seed": seed,
        "arrived_vehicles": summary["arrived_vehicles"],
        "completed_vehicles": summary["completed_vehicles"],
        "missed_vehicles": summary["missed_vehicles"],
        "completion_rate": summary["completion_rate"],
        "missed_rate": summary["missed_rate"],
        "total_electricity_cost": summary["total_electricity_cost"],
        "avg_cost_per_completed_vehicle": summary["avg_cost_per_completed_vehicle"],
        "mean_waiting_time_minutes": summary["mean_waiting_time_minutes"],
        "mean_waiting_time_started_minutes": summary["mean_waiting_time_started_minutes"],
        "total_energy_delivered_kwh": summary["total_energy_delivered_kwh"],
        "total_grid_energy_kwh": summary["total_grid_energy_kwh"],
        "port_utilization": summary["port_utilization"],
        "power_utilization": summary["power_utilization"],
        "episode_return": summary["episode_return"],
        "invalid_actions": summary["invalid_actions"],
    }


def summarize_results(df: pd.DataFrame) -> pd.DataFrame:
    """Computes mean/std/min/max across numeric metrics by policy."""
    numeric_cols = [
        c for c in df.columns if c not in {"episode", "policy", "seed"} and np.issubdtype(df[c].dtype, np.number)
    ]
    summary = (
        df.groupby("policy", as_index=False)[numeric_cols]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )
    summary.columns = ["_".join(col).strip("_") for col in summary.columns.values]
    return summary


def save_results(
    episode_rows: List[Dict],
    results_dir: Path,
    stem: str,
) -> Dict[str, Path]:
    """Saves episode-level and aggregated metrics as CSV/JSON."""
    results_dir.mkdir(parents=True, exist_ok=True)
    episodes_path = results_dir / f"{stem}_episodes.csv"
    summary_path = results_dir / f"{stem}_summary.csv"
    summary_json_path = results_dir / f"{stem}_summary.json"

    df = pd.DataFrame(episode_rows)
    df.to_csv(episodes_path, index=False)

    summary_df = summarize_results(df)
    summary_df.to_csv(summary_path, index=False)

    json_payload = summary_df.to_dict(orient="records")
    summary_json_path.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")

    return {
        "episodes_csv": episodes_path,
        "summary_csv": summary_path,
        "summary_json": summary_json_path,
    }
