# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

EV2Gym is a realistic EV-V2G (Electric Vehicle Vehicle-to-Grid) Gymnasium-compatible simulator for developing and evaluating smart charging algorithms. It supports multiple algorithm types: heuristics, Model Predictive Control (MPC), Mathematical Programming (Gurobi), and Reinforcement Learning.

The simulator is based on real-world data sources (ElaadNL, Pecan Street, Renewables Ninja, ENTSO-e) and implements realistic battery models including CCCV (Constant Current/Constant Voltage) charging curves and electrochemical degradation.

**Key Paper**: [arXiv](https://arxiv.org/abs/2404.01849) | [IEEE](https://dl.acm.org/doi/abs/10.1109/TITS.2024.3510945)

## Development Commands

### Installation
```bash
# Install from source (development mode)
pip install -e .

# Install from PyPI
pip install ev2gym

# Install dependencies
pip install -r requirements.txt
```

### Running Simulations

```bash
# Run example simulation
python example.py

# Run evaluation script with replay files
python evaluator.py

# Train RL agent with Stable Baselines3
python train_stable_baselines.py --algorithm ppo --config_file ev2gym/example_config_files/V2GProfitPlusLoads.yaml --train_steps 20000

# Available algorithms: ppo, a2c, ddpg, sac, td3, tqc, trpo, ars
```

### Building and Publishing

```bash
# Build package
rm -rf build dist *.egg-info
python -m build

# Upload to PyPI
python -m twine upload --repository pypi dist/*
# username: __token__
# use pypi-api-token
```

### Documentation

```bash
# Build documentation with Sphinx
cd docs
make html
# Output in docs/_build/html/
```

## Architecture

### Component Hierarchy

The simulator follows a hierarchical composition pattern:

```
EV2Gym (Gymnasium environment)
├── ChargingStations[] (EV_Charger objects)
│   └── Ports[] → EVs[]
├── Transformers[] (power aggregation/constraints)
├── Grid (optional PowerGrid for voltage simulation)
├── State/Reward/Cost Functions (pluggable)
└── Replay System (deterministic reproducibility)
```

### Execution Flow

**Initialization**: Config YAML → Loaders → Component instantiation → Action/Observation spaces

**Step Flow**:
1. Agent provides actions (per port, normalized [-1,1])
2. Actions distributed to charging stations
3. Each CS steps its connected EVs (charging/discharging)
4. Power aggregated at transformer level
5. Grid simulation updates voltages (if `simulate_grid=True`)
6. Reward calculated from configured reward function
7. State observation generated for next step

### Key Model Classes

**EV** (`ev2gym/models/ev.py`):
- Implements two-stage CCCV battery charging (not simple linear)
- Supports heterogeneous specs from JSON (battery capacity, charge rates, etc.)
- Tracks calendar and cyclic degradation using electrochemical models
- Efficiency can vary with current level (dict-based efficiency curves)

**EV_Charger** (`ev2gym/models/ev_charger.py`):
- Manages multiple ports (configurable, default 1)
- Normalizes actions: if sum>1, proportionally scales down to prevent over-subscription
- Converts normalized actions to amperage based on charger voltage/phase specs
- Penalizes invalid actions on empty ports

**Transformer** (`ev2gym/models/transformer.py`):
- Supports demand response events with configurable advance notification
- Generates forecasts for inflexible loads and PV with uncertainty
- Dynamic power limits (not static) that change during simulation
- Aggregates: inflexible loads + solar PV + EV charging power

**Grid** (`ev2gym/models/grid.py`):
- Optional power flow simulation (Laurent's method or PandaPower)
- Simulates voltage magnitudes across distribution network
- Uses GridTensor for efficient calculations
- Requires 15-minute timescale (hard requirement)
- Synthetic load/PV data via augmentor.pkl

### RL Agent System

The RL system uses **pluggable functions** for flexibility:

**State Functions** (`ev2gym/rl_agent/state.py`):
- Available: `PublicPST`, `V2G_profit_max`, `V2G_profit_max_loads`, `V2G_grid_state`
- Functions accept `env` and return flattened numpy arrays
- Include temporal features (timestep, day-of-week, sin/cos of hour)
- Per-EV features: SoC, time-to-departure, energy exchanged, max power
- Future information: price forecasts, power setpoints, load/PV forecasts

**Reward Functions** (`ev2gym/rl_agent/reward.py`):
- Multi-objective: tracking error, transformer violations, user satisfaction, profits
- `SquaredTrackingErrorReward`: For power setpoint tracking (PST) scenarios
- `profit_maximization`: For V2G profit maximization
- `ProfitMax_TrPenalty_UserIncentives`: Combined profit + transformer penalty + user satisfaction
- Grid-aware rewards include voltage violation penalties
- Some use exponential penalties (e.g., `exp(-10*score)`)

**Action Wrappers** (`ev2gym/rl_agent/action_wrappers.py`):
- `BinaryAction`: Simplifies to charge/idle
- `ThreeStep_Action`: Discrete actions (charge/idle/discharge)
- `Rescale_RepairLayer`: Proportionally adjusts all EV actions to meet power setpoints (most sophisticated)

### Baseline Algorithms

**Heuristics** (`ev2gym/baselines/heuristics.py`):
- `ChargeAsFastAsPossible`: Greedy charging
- `ChargeAsLateAsPossible`: Delays charging until necessary
- `ChargeAsFastAsPossibleToDesiredCapacity`: Charges to target then stops
- `RoundRobin`: Fair rotation among EVs
- `RandomAgent`: Random actions

**MPC** (`ev2gym/baselines/mpc/`):
- Abstract base class with shared matrix construction
- Builds state-space models (Amono, Bmono matrices) for battery dynamics
- Receding horizon control with configurable prediction horizon
- Variants: `eMPC`, `V2GProfitMaxOracle`, `OCMF_MPC`

**Gurobi Models** (`ev2gym/baselines/gurobi_models/`):
- Offline optimization (optimal solutions with perfect foresight)
- Used for benchmarking and creating optimal replay baselines
- Variants: `profit_max`, `tracking_error`, `v2g_grid`
- Requires Gurobi license (free academic licenses available)

### Configuration System

Configuration files are YAML-based with hierarchical structure in `ev2gym/example_config_files/`:

**Key sections**:
- `simulation_length`, `timescale`: Simulation parameters (in steps and minutes)
- `year`, `month`, `day`, `random_day`: Date/time settings
- `scenario`: `public`, `private`, or `workplace` (affects EV arrival patterns)
- `v2g_enabled`: Enable Vehicle-to-Grid (bidirectional charging)
- `number_of_charging_stations`, `number_of_transformers`: Network size
- `simulate_grid`: Enable grid voltage simulation
- `power_setpoint_enabled`: Power setpoint tracking objective
- `inflexible_loads`, `solar_power`, `demand_response`: Additional features
- `heterogeneous_ev_specs`: Use realistic EV models from JSON

**Important interactions**:
- `simulate_grid=True` forces 15-minute timescale
- Grid simulation overrides transformer topology from config
- `charging_network_topology: None` creates randomized network with specified parameters
- Separate RNG seeds: `seed` for EVs, `tr_seed` for transformers

### Replay System

**EvCityReplay** (`ev2gym/models/replay.py`):
- Enables deterministic reproducibility for fair algorithm comparisons
- Captures: EV spawn schedule, transformer loads/PV, grid matrices, prices, setpoints
- Can store multiple solutions (unstirred, optimal)

**Usage pattern**:
1. Run simulation with `save_replay=True`
2. Load replay with `load_from_replay_path="path/to/replay.pkl"`
3. Different algorithms see identical stochastic scenario

### Data Sources

**EV Arrival Patterns** (`ev2gym/data/`):
- CSV files with hourly distributions conditioned on time/day/month
- Based on ElaadNL Dutch public charging data
- Scenario-specific patterns (workplace has no weekends)

**Electricity Prices**:
- `Netherlands_day-ahead-2015-2024.csv` (9 years, hourly)
- Maps simulation datetime to historical ENTSO-e day-ahead prices

**Inflexible Loads**:
- `residential_loads.csv` (Pecan Street data, 15-min resolution)
- Samples random load profiles per transformer

**PV Generation**:
- `pv_netherlands.csv` (Renewables Ninja, hourly)
- Smoothed and scaled per transformer

**Grid Data**:
- Network topologies: `node_34/`, `node_69/`, `node_123/`, `node_25/`
- Contains node and line CSVs for power flow

**EV Specifications**:
- JSON files with real vehicle models (Tesla Model 3, Nissan Leaf, etc.)
- Probabilistic sampling based on registration numbers
- Charge efficiency curves indexed by current level

## Important Implementation Details

### Action Normalization
Actions are normalized **twice**:
1. Environment level: If sum>1 across ports, proportionally scale down
2. Charger level: Convert normalized [-1,1] to amperage based on specs

### EV Departure Timing
`time_of_departure` is a **minimum**, not exact. EVs may depart later depending on energy requirements.

### Transformer Constraints
Transformer power limits are **time-varying** due to:
- Demand response events (with advance notification)
- Dynamic inflexible loads and PV generation

### Grid Simulation Requirements
- Grid simulation requires 15-minute timescale (hard requirement)
- Cannot run arbitrary timescales without retraining augmentor.pkl
- Grid matrices (K, L) are pre-computed for efficiency

### Replay Mode Behavior
Many loaders check `env.load_from_replay_path` first and load deterministic data from replay file instead of sampling.

### Battery Degradation
Degradation models are research-grade (not simplified), based on electrochemical papers. Include both calendar and cyclic aging.

### Performance Optimization
`lightweight_plots` flag auto-enables for >100 charging stations to avoid memory issues.

## Example Workflows

### Basic Simulation
```python
from ev2gym.models.ev2gym_env import EV2Gym
from ev2gym.baselines.heuristics import ChargeAsFastAsPossible

config_file = "ev2gym/example_config_files/V2GProfitPlusLoads.yaml"
env = EV2Gym(config_file=config_file, save_replay=True, save_plots=True)
state, _ = env.reset()

agent = ChargeAsFastAsPossible()
for t in range(env.simulation_length):
    actions = agent.get_action(env)
    new_state, reward, done, truncated, stats = env.step(actions)
```

### Custom RL Training
```python
import gymnasium as gym
from stable_baselines3 import DDPG
from ev2gym.rl_agent.reward import ProfitMax_TrPenalty_UserIncentives
from ev2gym.rl_agent.state import V2G_profit_max_loads

env = gym.make('EV2Gym-v1',
               config_file=config_file,
               reward_function=ProfitMax_TrPenalty_UserIncentives,
               state_function=V2G_profit_max_loads)

model = DDPG("MlpPolicy", env)
model.learn(total_timesteps=1_000_000)
```

### Creating Custom Reward/State Functions
```python
# Custom state function
def my_state_function(env):
    # Return flattened numpy array
    state = []
    # Add temporal features
    state.append(env.current_step / env.simulation_length)
    # Add EV features
    for ev in env.active_evs:
        state.extend([ev.soc, ev.time_to_departure, ev.max_charge_power])
    return np.array(state)

# Custom reward function
def my_reward_function(env):
    # Calculate profit
    profit = sum(cs.total_profit for cs in env.charging_stations)
    # Penalize transformer violations
    penalty = sum(max(0, tr.total_power - tr.max_power)**2 for tr in env.transformers)
    return profit - 0.1 * penalty

# Use in environment
env = gym.make('EV2Gym-v1',
               config_file=config_file,
               reward_function=my_reward_function,
               state_function=my_state_function)
```

## File Organization

- `ev2gym/models/`: Core simulation components (EV, Charger, Transformer, Grid, Environment)
- `ev2gym/baselines/`: Reference algorithms (heuristics, MPC, Gurobi)
- `ev2gym/rl_agent/`: RL-specific code (rewards, states, action wrappers)
- `ev2gym/data/`: Real-world data files (EV patterns, prices, loads, PV, networks)
- `ev2gym/utilities/`: Helper functions (loaders, statistics, arg parsing)
- `ev2gym/visuals/`: Plotting and rendering
- `ev2gym/example_config_files/`: Pre-configured scenarios
- `ev2gym/scripts/`: Utility scripts for replay generation
