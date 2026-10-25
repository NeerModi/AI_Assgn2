from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.utils import rolling_mean

sns.set_theme(style="whitegrid", context="talk")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate report figures from real output files")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--figures-dir", type=str, default="figures")
    return parser.parse_args()


def _require_file(path: Path, message: str) -> None:
    if not path.exists():
        raise FileNotFoundError(message)


def plot_training_curve(results_dir: Path, figures_dir: Path) -> None:
    progress_path = results_dir / "training_progress.csv"
    _require_file(
        progress_path,
        "Missing results/training_progress.csv. Run train.py to generate real training logs.",
    )

    df = pd.read_csv(progress_path)
    if df.empty:
        raise ValueError("training_progress.csv exists but is empty")

    x = df["timesteps"].to_numpy(dtype=float)
    y = df["episode_reward"].to_numpy(dtype=float)
    smooth = rolling_mean(y, window=min(20, max(2, len(y) // 10)))

    plt.figure(figsize=(10, 5))
    plt.plot(x, y, alpha=0.35, label="Episode reward")
    plt.plot(x, smooth, linewidth=2.5, label="Smoothed reward")
    plt.xlabel("Training timesteps")
    plt.ylabel("Episode return")
    plt.title("PPO Training Reward vs Timesteps")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "training_reward_curve.png", dpi=220)
    plt.close()


def plot_policy_bars(results_dir: Path, figures_dir: Path) -> None:
    episodes_path = results_dir / "policy_comparison_episodes.csv"
    _require_file(
        episodes_path,
        "Missing results/policy_comparison_episodes.csv. Run compare_baselines.py first.",
    )

    df = pd.read_csv(episodes_path)
    if df.empty:
        raise ValueError("policy_comparison_episodes.csv exists but is empty")

    metrics = [
        ("total_electricity_cost", "Electricity Cost by Policy", "Cost (currency unit)"),
        ("mean_waiting_time_minutes", "Average Waiting Time by Policy", "Minutes"),
        ("completion_rate", "On-Time Completion Rate by Policy", "Fraction"),
        ("total_energy_delivered_kwh", "Energy Delivered by Policy", "kWh"),
    ]

    for metric, title, ylabel in metrics:
        plt.figure(figsize=(8, 5))
        sns.barplot(data=df, x="policy", y=metric, errorbar="sd", palette="Set2")
        plt.title(title)
        plt.xlabel("Policy")
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.savefig(figures_dir / f"{metric}_by_policy.png", dpi=220)
        plt.close()


def plot_representative_episode(results_dir: Path, figures_dir: Path) -> None:
    rep_path = results_dir / "representative_episodes.json"
    _require_file(
        rep_path,
        "Missing results/representative_episodes.json. Run compare_baselines.py first.",
    )

    data = json.loads(rep_path.read_text(encoding="utf-8"))
    if not data:
        raise ValueError("representative_episodes.json exists but has no policy entries")

    preferred_order = ["ppo", "fcfs", "random"]
    chosen_policy = next((p for p in preferred_order if p in data), next(iter(data.keys())))
    history = data[chosen_policy]["history"]

    t = np.asarray(history["time_h"], dtype=float)
    p = np.asarray(history["price"], dtype=float)
    active_ports = np.asarray(history["active_ports"], dtype=float)
    queue = np.asarray(history["queue_len"], dtype=float)
    power = np.asarray(history["power_kw"], dtype=float)

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(t, p, color="#1f77b4", label="Electricity price")
    ax1.set_xlabel("Time (hours)")
    ax1.set_ylabel("Price (currency per kWh)", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")

    ax2 = ax1.twinx()
    ax2.plot(t, power, color="#d62728", label="Station power (kW)")
    ax2.plot(t, active_ports, color="#2ca02c", linestyle="--", label="Active ports")
    ax2.plot(t, queue, color="#9467bd", linestyle=":", label="Waiting vehicles")
    ax2.set_ylabel("Power / Count")

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc="upper right")

    plt.title(f"Representative Episode Dynamics ({chosen_policy.upper()})")
    plt.tight_layout()
    plt.savefig(figures_dir / "representative_episode_dynamics.png", dpi=220)
    plt.close()


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    plot_training_curve(results_dir=results_dir, figures_dir=figures_dir)
    plot_policy_bars(results_dir=results_dir, figures_dir=figures_dir)
    plot_representative_episode(results_dir=results_dir, figures_dir=figures_dir)

    print("[figures] Generated:")
    for png in sorted(figures_dir.glob("*.png")):
        print(f"  - {png}")


if __name__ == "__main__":
    main()
