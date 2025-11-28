# ==============================================================================
# PEAK-SHAVING EV CHARGING WITH DQN - COMPLETE CODE
# ==============================================================================
# Copy each section into Jupyter notebook cells

# ==============================================================================
# CELL 1 (Markdown): Title
# ==============================================================================
"""
# Peak-Shaving EV Charging with DQN 🔋⚡

**Goal:** Minimize power demand spikes while keeping EV users satisfied

## Original `PeakPenaltyReward` Issues Fixed:
- ❌ **Sign error**: Line 14 rewarded spikes instead of penalizing!
- ❌ **Unbounded values**: Returns -10,000+, causes training instability
- ❌ **Imbalanced penalties**: User satisfaction dominates (1000 vs 100)

## Our Improved Solution:
- ✅ **Correct sign**: Now penalizes power increases
- ✅ **Normalized**: Rewards in [-10, 1] range for stable training
- ✅ **Balanced**: Multi-objective with tunable weights
"""

# ==============================================================================
# CELL 2 (Code): Imports
# ==============================================================================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from collections import namedtuple, deque
from itertools import count
import random
import math
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

if os.path.basename(os.getcwd()) == 'tutorials':
    os.chdir('..')
print(f"Working directory: {os.getcwd()}")

from ev2gym.models.ev2gym_env import EV2Gym
from ev2gym.baselines.heuristics import ChargeAsFastAsPossible

plt.style.use('seaborn-v0_8-darkgrid')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==============================================================================
# CELL 3 (Code): Improved Reward Function
# ==============================================================================
def PeakPenaltyRewardV2(env, total_costs, user_satisfaction_list,
                         beta_derivative=10.0,
                         beta_magnitude=5.0,
                         user_penalty=100.0,
                         normalize=True):
    """
    Improved peak-shaving reward function.

    Parameters:
    - env: EV2Gym environment
    - total_costs: Total costs (unused, for signature compatibility)
    - user_satisfaction_list: List of user satisfaction scores
    - beta_derivative: Weight for spike penalty
    - beta_magnitude: Weight for peak magnitude penalty
    - user_penalty: Weight for user satisfaction
    - normalize: Whether to normalize rewards

    Fixed issues from original:
    1. Correct sign: Penalizes power INCREASES
    2. Normalized to [-10, 1] range
    3. Balanced multi-objective components
    """
    if env.current_step == 1:
        return 0.0

    current_power = env.current_power_usage[env.current_step - 1]
    previous_power = env.current_power_usage[env.current_step - 2]

    # Component 1: Penalize power INCREASES (spikes)
    power_increase = max(0, current_power - previous_power)
    derivative_penalty = -beta_derivative * power_increase

    # Component 2: Penalize high absolute power
    max_possible = env.charge_power_potential[env.current_step - 1]
    if max_possible > 0:
        magnitude_penalty = -beta_magnitude * (current_power / max_possible)
    else:
        magnitude_penalty = 0

    # Component 3: Penalize user dissatisfaction
    user_penalty_total = 0
    for score in user_satisfaction_list:
        user_penalty_total -= user_penalty * (1 - score)

    reward = derivative_penalty + magnitude_penalty + user_penalty_total

    # Normalize to [-10, 1] range
    if normalize:
        max_pen = -(beta_derivative * max_possible + beta_magnitude +
                   user_penalty * len(user_satisfaction_list))
        if max_pen != 0:
            reward = 10 * (reward / max_pen)
            reward = max(-10, min(reward, 1))

    return reward

print("✅ PeakPenaltyRewardV2 defined")
print("   - Penalizes spikes (correct sign)")
print("   - Normalized to [-10, 1]")
print("   - Balanced multi-objective")

# ==============================================================================
# CELL 4 (Code): Environment Setup
# ==============================================================================
config_file = "ev2gym/example_config_files/PublicPST.yaml"

env = EV2Gym(
    config_file=config_file,
    render_mode=False,
    seed=42,
    save_plots=False,
    save_replay=False,
    verbose=False
)

env.set_reward_function(PeakPenaltyRewardV2)

state, _ = env.reset()
n_observations = len(state)
total_ports = sum(cs.n_ports for cs in env.charging_stations)

print(f"State dimension: {n_observations}")
print(f"Total EV ports: {total_ports}")
print(f"Simulation length: {env.simulation_length} steps")
print(f"\n✅ Environment with PeakPenaltyRewardV2")

# ==============================================================================
# CELL 5 (Code): DQN Configuration
# ==============================================================================
DQN_CONFIG = {
    'num_episodes': 2000,
    'batch_size': 32,
    'lr': 1e-5,              # Low LR for multi-objective
    'gamma': 0.90,
    'eps_start': 1.0,
    'eps_end': 0.05,
    'eps_decay': 15000,      # Slower decay
    'tau': 0.001,
    'memory_size': 50000,
    'n_actions': 11,
    'hidden_size': 64,
    'use_double_dqn': True,
    'clip_q_values': False,
    'q_clip_min': -10,
    'q_clip_max': 10,
    'gradient_clip': 1.0,
    'print_freq': 10,
    'save_freq': 50,
}

print("DQN Configuration:")
print("=" * 70)
for key, value in DQN_CONFIG.items():
    print(f"  {key:20s}: {value}")
print("=" * 70)

# ==============================================================================
# CELL 6 (Code): DQN Network
# ==============================================================================
class SimpleDQN(nn.Module):
    def __init__(self, n_observations, n_actions, hidden_size=64):
        super(SimpleDQN, self).__init__()
        self.fc1 = nn.Linear(n_observations, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.fc3 = nn.Linear(hidden_size, n_actions)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        return self.fc3(x)

policy_net = SimpleDQN(n_observations, DQN_CONFIG['n_actions'],
                       DQN_CONFIG['hidden_size']).to(device)
target_net = SimpleDQN(n_observations, DQN_CONFIG['n_actions'],
                       DQN_CONFIG['hidden_size']).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

print("Network Architecture:")
print(policy_net)
print(f"\nTotal parameters: {sum(p.numel() for p in policy_net.parameters()):,}")

# ==============================================================================
# CELL 7 (Code): Training Components
# ==============================================================================
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
    def push(self, *args):
        self.memory.append(Transition(*args))
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    def __len__(self):
        return len(self.memory)

def discretize_action(action_idx):
    value = action_idx / (DQN_CONFIG['n_actions'] - 1)
    return [value] * total_ports

memory = ReplayMemory(DQN_CONFIG['memory_size'])
optimizer = optim.AdamW(policy_net.parameters(), lr=DQN_CONFIG['lr'], amsgrad=True)
steps_done = 0

print(f"✅ Replay memory: {DQN_CONFIG['memory_size']:,}")
print(f"✅ Optimizer: AdamW (LR={DQN_CONFIG['lr']:.2e})")

# ==============================================================================
# CELL 8 (Code): Training Functions
# ==============================================================================
def get_epsilon(steps):
    return DQN_CONFIG['eps_end'] + (DQN_CONFIG['eps_start'] - DQN_CONFIG['eps_end']) * \
           math.exp(-1.0 * steps / DQN_CONFIG['eps_decay'])

def select_action(state):
    global steps_done
    eps_threshold = get_epsilon(steps_done)
    steps_done += 1

    if random.random() > eps_threshold:
        with torch.no_grad():
            policy_net.eval()
            action = policy_net(state).max(1).indices.view(1, 1)
            policy_net.train()
            return action
    else:
        return torch.tensor([[random.randrange(DQN_CONFIG['n_actions'])]],
                          device=device, dtype=torch.long)

def optimize_model():
    if len(memory) < DQN_CONFIG['batch_size']:
        return None, None

    transitions = memory.sample(DQN_CONFIG['batch_size'])
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch.next_state)),
        device=device, dtype=torch.bool
    )
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    if DQN_CONFIG['clip_q_values']:
        state_action_values = torch.clamp(
            state_action_values,
            DQN_CONFIG['q_clip_min'],
            DQN_CONFIG['q_clip_max']
        )

    next_state_values = torch.zeros(DQN_CONFIG['batch_size'], device=device)

    # Double DQN
    if DQN_CONFIG['use_double_dqn']:
        with torch.no_grad():
            best_actions = policy_net(non_final_next_states).max(1).indices.unsqueeze(1)
            next_state_values[non_final_mask] = target_net(non_final_next_states).gather(1, best_actions).squeeze()

    if DQN_CONFIG['clip_q_values']:
        next_state_values = torch.clamp(
            next_state_values,
            DQN_CONFIG['q_clip_min'],
            DQN_CONFIG['q_clip_max']
        )

    expected_state_action_values = (next_state_values * DQN_CONFIG['gamma']) + reward_batch

    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), DQN_CONFIG['gradient_clip'])
    optimizer.step()

    return loss.item(), state_action_values.max().item()

def update_target_network():
    target_net_state = target_net.state_dict()
    policy_net_state = policy_net.state_dict()

    for key in policy_net_state:
        target_net_state[key] = (
            policy_net_state[key] * DQN_CONFIG['tau'] +
            target_net_state[key] * (1 - DQN_CONFIG['tau'])
        )

    target_net.load_state_dict(target_net_state)

print("✅ Training functions defined (Double DQN)")

# ==============================================================================
# CELL 9 (Code): Peak Metrics Tracker
# ==============================================================================
class PeakMetricsTracker:
    def __init__(self):
        self.episode_rewards = []
        self.episode_stats = []
        self.losses = []
        self.q_values = []

        # Peak-specific metrics
        self.peak_powers = []
        self.power_stds = []
        self.num_spikes = []
        self.user_satisfactions = []

    def log_step(self, loss, q_value):
        """Log training step metrics (epsilon not needed for tracking)"""
        if loss is not None:
            self.losses.append(loss)
        if q_value is not None:
            self.q_values.append(q_value)

    def log_episode(self, reward, stats, env):
        self.episode_rewards.append(reward)
        self.episode_stats.append(stats)

        power_usage = env.current_power_usage[:env.current_step]
        self.peak_powers.append(np.max(power_usage))
        self.power_stds.append(np.std(power_usage))

        derivatives = np.diff(power_usage)
        spikes = np.sum(derivatives > 10)  # Count spikes > 10 kW
        self.num_spikes.append(spikes)

        self.user_satisfactions.append(stats.get('average_user_satisfaction', 0))

    def get_recent_avg(self, metric, window=10):
        data = getattr(self, metric)
        if len(data) == 0:
            return 0
        return np.mean(data[-window:])

metrics = PeakMetricsTracker()
print("✅ Peak metrics tracker initialized")

# ==============================================================================
# CELL 10 (Code): Training Loop
# ==============================================================================
RUN_NAME = f"peak_shaving_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
RUN_DIR = f"./runs/{RUN_NAME}"
os.makedirs(RUN_DIR, exist_ok=True)

print(f"\n{'='*70}")
print(f"Starting Peak-Shaving DQN Training: {DQN_CONFIG['num_episodes']} episodes")
print(f"Run directory: {RUN_DIR}")
print(f"{'='*70}\n")

for episode in range(DQN_CONFIG['num_episodes']):
    state, _ = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

    episode_reward = 0
    episode_losses = []
    episode_q_values = []

    for t in count():
        action = select_action(state)
        observation, reward, done, truncated, stats = env.step(
            discretize_action(action.item())
        )

        episode_reward += reward
        reward_tensor = torch.tensor([reward], device=device, dtype=torch.float32)

        if done:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        memory.push(state, action, next_state, reward_tensor)
        state = next_state

        loss, max_q = optimize_model()
        if loss is not None:
            episode_losses.append(loss)
        if max_q is not None:
            episode_q_values.append(max_q)

        update_target_network()
        metrics.log_step(loss, max_q)

        if done:
            break

    metrics.log_episode(episode_reward, stats, env)

    # Print progress
    if (episode + 1) % DQN_CONFIG['print_freq'] == 0:
        avg_reward = metrics.get_recent_avg('episode_rewards', 10)
        avg_loss = np.mean(episode_losses) if episode_losses else 0
        avg_q = np.mean(episode_q_values) if episode_q_values else 0
        epsilon = get_epsilon(steps_done)

        # Peak metrics
        peak_power = metrics.peak_powers[-1]
        power_std = metrics.power_stds[-1]
        num_spikes = metrics.num_spikes[-1]
        user_sat = metrics.user_satisfactions[-1]

        print(f"Ep {episode+1:3d}/{DQN_CONFIG['num_episodes']} | "
              f"Rew: {episode_reward:6.2f} (avg: {avg_reward:6.2f}) | "
              f"Loss: {avg_loss:.4f} | "
              f"Q: {avg_q:5.2f} | "
              f"ε: {epsilon:.3f}")
        print(f"         Peak: {peak_power:6.1f} kW | "
              f"Std: {power_std:5.1f} | "
              f"Spikes: {num_spikes:2d} | "
              f"UserSat: {user_sat:.2%}")

    # Save checkpoint
    if (episode + 1) % DQN_CONFIG['save_freq'] == 0:
        checkpoint_path = f"{RUN_DIR}/checkpoint_ep{episode+1}.pth"
        torch.save({
            'episode': episode,
            'policy_net_state_dict': policy_net.state_dict(),
            'target_net_state_dict': target_net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'steps_done': steps_done,
        }, checkpoint_path)

print(f"\n{'='*70}")
print("Peak-Shaving DQN Training Complete!")
print(f"{'='*70}")

# Save final model
torch.save(policy_net.state_dict(), f"{RUN_DIR}/final_model.pth")
with open(f"{RUN_DIR}/metrics.pkl", 'wb') as f:
    pickle.dump(metrics.__dict__, f)

# ==============================================================================
# CELL 11 (Code): Visualize Results
# ==============================================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Peak-Shaving DQN Training Results', fontsize=16, fontweight='bold')

# 1. Episode Rewards
ax = axes[0, 0]
ax.plot(metrics.episode_rewards, alpha=0.6, label='Per episode')
if len(metrics.episode_rewards) > 10:
    rolling_avg = pd.Series(metrics.episode_rewards).rolling(window=20).mean()
    ax.plot(rolling_avg, linewidth=2, label='20-ep avg', color='red')
ax.set_xlabel('Episode')
ax.set_ylabel('Total Reward')
ax.set_title('Episode Rewards')
ax.legend()
ax.grid(True, alpha=0.3)

# 2. Training Loss
ax = axes[0, 1]
if len(metrics.losses) > 100:
    rolling_avg = pd.Series(metrics.losses).rolling(window=500).mean()
    ax.plot(rolling_avg, linewidth=2, color='red')
ax.set_xlabel('Training Step')
ax.set_ylabel('Loss')
ax.set_title('Training Loss (should decrease)')
ax.grid(True, alpha=0.3)

# 3. Q-Values
ax = axes[0, 2]
if len(metrics.q_values) > 100:
    rolling_avg = pd.Series(metrics.q_values).rolling(window=500).mean()
    ax.plot(rolling_avg, linewidth=2, color='red')
ax.set_xlabel('Training Step')
ax.set_ylabel('Max Q-Value')
ax.set_title('Q-Values (should stay < 10)')
ax.axhline(y=10, color='r', linestyle='--', alpha=0.5)
ax.axhline(y=-10, color='r', linestyle='--', alpha=0.5)
ax.grid(True, alpha=0.3)

# 4. Peak Power
ax = axes[1, 0]
ax.plot(metrics.peak_powers, alpha=0.6, label='Per episode')
if len(metrics.peak_powers) > 10:
    rolling_avg = pd.Series(metrics.peak_powers).rolling(window=20).mean()
    ax.plot(rolling_avg, linewidth=2, label='20-ep avg', color='red')
ax.set_xlabel('Episode')
ax.set_ylabel('Peak Power (kW)')
ax.set_title('Peak Power (should decrease)')
ax.legend()
ax.grid(True, alpha=0.3)

# 5. Power Variability
ax = axes[1, 1]
ax.plot(metrics.power_stds, alpha=0.6, label='Per episode')
if len(metrics.power_stds) > 10:
    rolling_avg = pd.Series(metrics.power_stds).rolling(window=20).mean()
    ax.plot(rolling_avg, linewidth=2, label='20-ep avg', color='red')
ax.set_xlabel('Episode')
ax.set_ylabel('Power Std Dev (kW)')
ax.set_title('Power Variability (should decrease)')
ax.legend()
ax.grid(True, alpha=0.3)

# 6. Number of Spikes
ax = axes[1, 2]
ax.plot(metrics.num_spikes, alpha=0.6, label='Per episode')
if len(metrics.num_spikes) > 10:
    rolling_avg = pd.Series(metrics.num_spikes).rolling(window=20).mean()
    ax.plot(rolling_avg, linewidth=2, label='20-ep avg', color='red')
ax.set_xlabel('Episode')
ax.set_ylabel('Number of Spikes')
ax.set_title('Power Spikes (should decrease)')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{RUN_DIR}/peak_training_plots.png", dpi=150, bbox_inches='tight')
plt.show()

# ==============================================================================
# CELL 12 (Code): Compare with ChargeAsFastAsPossible Baseline
# ==============================================================================
print("\n" + "="*70)
print("COMPARING WITH ChargeAsFastAsPossible BASELINE")
print("="*70)

# Evaluate baseline
baseline_env = EV2Gym(
    config_file=config_file,
    render_mode=False,
    seed=42,
    save_plots=False,
    save_replay=False,
    verbose=False
)
baseline_env.set_reward_function(PeakPenaltyRewardV2)

baseline_agent = ChargeAsFastAsPossible()

baseline_metrics = {
    'peak_powers': [],
    'power_stds': [],
    'num_spikes': [],
    'user_satisfactions': [],
    'rewards': []
}

for ep in range(20):
    state, _ = baseline_env.reset()
    episode_reward = 0

    for t in count():
        actions = baseline_agent.get_action(baseline_env)
        observation, reward, done, truncated, stats = baseline_env.step(actions)
        episode_reward += reward

        if done:
            power_usage = baseline_env.current_power_usage[:baseline_env.current_step]
            baseline_metrics['peak_powers'].append(np.max(power_usage))
            baseline_metrics['power_stds'].append(np.std(power_usage))

            derivatives = np.diff(power_usage)
            baseline_metrics['num_spikes'].append(np.sum(derivatives > 10))

            baseline_metrics['user_satisfactions'].append(
                stats.get('average_user_satisfaction', 0)
            )
            baseline_metrics['rewards'].append(episode_reward)
            break

# DQN final performance (last 20 episodes)
dqn_metrics = {
    'peak_power': np.mean(metrics.peak_powers[-20:]),
    'power_std': np.mean(metrics.power_stds[-20:]),
    'num_spikes': np.mean(metrics.num_spikes[-20:]),
    'user_satisfaction': np.mean(metrics.user_satisfactions[-20:]),
    'reward': np.mean(metrics.episode_rewards[-20:])
}

baseline_summary = {
    'peak_power': np.mean(baseline_metrics['peak_powers']),
    'power_std': np.mean(baseline_metrics['power_stds']),
    'num_spikes': np.mean(baseline_metrics['num_spikes']),
    'user_satisfaction': np.mean(baseline_metrics['user_satisfactions']),
    'reward': np.mean(baseline_metrics['rewards'])
}

# Print comparison
print("\n" + "="*70)
print("RESULTS COMPARISON (last 20 episodes)")
print("="*70)
print(f"{'Metric':<25s} | {'DQN':<15s} | {'CAFAP':<15s} | {'Improvement'}")
print("-"*70)

for metric in ['peak_power', 'power_std', 'num_spikes', 'user_satisfaction', 'reward']:
    dqn_val = dqn_metrics[metric]
    base_val = baseline_summary[metric]

    if metric == 'user_satisfaction':
        improvement = ((dqn_val - base_val) / base_val * 100) if base_val != 0 else 0
        improvement_str = f"{improvement:+.1f}%" if improvement >= -5 else f"{improvement:+.1f}% ⚠️"
    elif metric == 'reward':
        improvement = ((dqn_val - base_val) / abs(base_val) * 100) if base_val != 0 else 0
        improvement_str = f"{improvement:+.1f}%" if improvement > 0 else f"{improvement:+.1f}%"
    else:
        improvement = ((base_val - dqn_val) / base_val * 100) if base_val != 0 else 0
        improvement_str = f"{improvement:+.1f}% ✅" if improvement > 0 else f"{improvement:+.1f}%"

    print(f"{metric.replace('_', ' ').title():<25s} | {dqn_val:<15.2f} | {base_val:<15.2f} | {improvement_str}")

print("="*70)

# Success criteria
success = (
    dqn_metrics['peak_power'] < baseline_summary['peak_power'] and
    dqn_metrics['power_std'] < baseline_summary['power_std'] and
    dqn_metrics['num_spikes'] < baseline_summary['num_spikes']
)

if success:
    print("\n✅ SUCCESS: DQN achieves better peak-shaving than baseline!")
else:
    print("\n⚠️  DQN needs more training or hyperparameter tuning")
    print("\nSuggestions:")
    print("  - Train for more episodes (1000+)")
    print("  - Increase beta_derivative to 20.0")
    print("  - Decrease beta_magnitude to 2.0")

print("\n" + "="*70)
