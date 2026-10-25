# Reinforcement Learning for Coordinated EV Charging

## Project Title
Reinforcement Learning for Coordinated Electric Vehicle Charging Under Limited Station Capacity and Dynamic Electricity Prices

## Objective
This mini-project implements a **custom Gymnasium environment** and trains a **PPO agent (Stable-Baselines3)** to coordinate EV charging decisions under:

- limited charging ports,
- station-wide power constraints,
- dynamic electricity prices,
- uncertain arrivals and departure deadlines.

The policy goal is to reduce electricity cost and waiting time while increasing the percentage of vehicles that reach target SOC before departure.

## Why This Is a Reinforcement Learning Project
This project uses online interaction between an agent and a stochastic environment:

- the agent observes state $s_t$,
- chooses action $a_t$,
- receives reward $r_t$,
- and learns a policy $\pi(a_t\mid s_t)$ that maximizes discounted return.

No supervised labels or hand-crafted scheduler are used to train PPO.

## Project Structure

```text
ev-rl-charging/
    README.md
    requirements.txt
    .gitignore
    config.yaml
    train.py
    evaluate.py
    compare_baselines.py
    generate_report_assets.py
    src/
        __init__.py
        environment.py
        vehicle.py
        policies.py
        metrics.py
        utils.py
    tests/
        test_environment.py
        test_training.py
    results/
        .gitkeep
    models/
        .gitkeep
    logs/
        .gitkeep
    figures/
        .gitkeep
    reports/
        .gitkeep
```

## MDP Formulation

- **State $S$**: fixed-length observation vector with per-vehicle and global features:
  - Per-vehicle: arrival status, departure status, waiting, charging, completion flag, SOC, normalized remaining energy, normalized time to departure, normalized max charge power.
  - Global: normalized current price, time-of-day sine/cosine, free-port fraction, active-port fraction, queue fraction, station power used fraction, station power left fraction.
- **Action $A$**: discrete actions $\{0,1,\dots,N\}$ where $N$ is number of vehicle slots.
  - $0$: no-op.
  - $i>0$: attempt to start charging vehicle index $i-1$.
- **Transition $P$**: stochastic, from random arrivals/departures/prices and deterministic charging dynamics under constraints.
- **Reward $R$**: weighted step objective combining cost, waiting, missed deadlines, completions, and invalid actions.
- **Discount factor**: controlled by PPO hyperparameter $\gamma$ (default 0.99).

## Environment Design

- Discrete-time simulation with default 15-minute step.
- Default episode horizon: 24h.
- Default vehicles per episode: 8.
- Charging ports: 3.
- Station max power: 30 kW.
- Vehicle attributes sampled each episode: arrival, departure, battery size, initial SOC, target SOC, max charging power.
- SOC is always clipped to $[0,1]$.
- Station power is fairly allocated across active charging EVs with per-vehicle power caps.

### Action Validity Rules
An action to start a vehicle is valid only if the selected vehicle:

- has already arrived,
- has not departed,
- is not completed,
- is not already charging,
- and at least one charging port is available.

Invalid/unproductive actions are handled safely and get a modest penalty.

### Reward Function
At each step:

$$
r_t = -\alpha C_t - \beta W_t - \gamma M_t + \delta D_t - \eta I_t
$$

where:

- $C_t$: electricity expenditure in this step (currency units),
- $W_t$: waiting time added in this step (hours),
- $M_t$: number of newly missed deadlines in this step,
- $D_t$: number of newly completed vehicles in this step,
- $I_t$: invalid-action indicator (0 or 1).

Important implementation details:

- Missed-deadline penalty is applied once per vehicle (not repeated every future step).
- Completion reward is applied once per completion event.
- Cost accounting is tracked separately from reward.
- Reward components are logged in `info["reward_components"]`.

## PPO Training
Main algorithm: **Stable-Baselines3 PPO with `MlpPolicy`**.

Defaults are in `config.yaml` and include:

- total timesteps,
- number of vectorized environments,
- learning rate,
- rollout length (`n_steps`),
- batch size,
- entropy coefficient,
- clipping range,
- evaluation frequency.

Model is saved to `models/ppo_ev_charging.zip` after actual training.

## Baselines
Implemented comparison policies:

1. **FCFS**: earliest-arrived eligible waiting vehicle.
2. **Random**: random eligible waiting vehicle.
3. **PPO**: trained policy loaded from saved model.

Baselines are evaluation-only and do not train PPO.

## Evaluation Metrics
Per-episode and aggregate statistics include:

- total electricity cost,
- average cost per completed vehicle,
- mean waiting time per arrived vehicle (minutes),
- mean waiting time for vehicles that started charging (minutes),
- completion percentage before departure,
- missed deadline percentage,
- total energy delivered (kWh),
- completed vehicle count,
- port utilization,
- station power utilization,
- episode return.

Outputs are saved as CSV and JSON in `results/`.

## Installation

### Linux / macOS

```bash
cd /home/manav/Desktop/Neer
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Windows PowerShell

```powershell
cd C:\path\to\Neer
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run Commands

### 1) Train PPO

```bash
python train.py --config config.yaml
```

Optional short smoke training:

```bash
python train.py --config config.yaml --timesteps 5000 --seed 123
```

### 2) Evaluate PPO only

```bash
python evaluate.py --config config.yaml --model-path models/ppo_ev_charging.zip --episodes 50 --deterministic
```

### 3) Compare PPO vs FCFS vs Random

```bash
python compare_baselines.py --config config.yaml --model-path models/ppo_ev_charging.zip --episodes 100
```

### 4) Generate figures for report

```bash
python generate_report_assets.py --results-dir results --figures-dir figures
```

### 5) Run tests

```bash
pytest -q
```

## Reproducibility

- Global seeds are controlled from `config.yaml` and script flags.
- Evaluation episodes use deterministic seed sequences (`base_seed + episode_id`).
- Keep config and seed metadata from `results/training_metadata.json` with outputs.

## Expected Output Files

- `models/ppo_ev_charging.zip`
- `results/training_progress.csv`
- `results/training_metadata.json`
- `results/ppo_evaluation_episodes.csv`
- `results/ppo_evaluation_summary.csv`
- `results/policy_comparison_episodes.csv`
- `results/policy_comparison_summary.csv`
- `results/representative_episodes.json`
- PNG plots in `figures/`

## Environment Assumptions and Limitations

This simulator is intentionally simplified for a mini-project:

- Synthetic arrivals, deadlines, and prices.
- No battery degradation model.
- No network/grid constraints beyond station max power.
- Fair-share power allocation is simplified and does not solve a continuous optimization each step.
- No explicit user behavior model (e.g., changing target SOC after arrival).

Therefore, this project supports **method demonstration and educational comparison**, not deployment claims.

## Future Improvements

- Multi-objective tuning and sensitivity analysis.
- Observation engineering with demand forecasts.
- Richer battery/charger physics.
- Robustness testing under domain shift.
- Multi-agent or hierarchical control.

## Important Disclaimer
This simulator is synthetic. Results in this repository do **not** establish real-world charging-station deployment performance or operational guarantees.
