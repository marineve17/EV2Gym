# Analysis of Custom DQN RL Algorithm for EV2Gym

## Overview

This document analyzes the modifications made to a Deep Q-Network (DQN) reinforcement learning algorithm for the EV2Gym power setpoint tracking problem. The analysis compares the original tutorial implementation with Helena's modified version (v2), identifies convergence issues, and explains the solutions applied.

---

## Problem Statement

**Goal**: Train a DQN agent to control 20 EV charging stations to minimize the difference between a power setpoint and actual power consumption.

**Environment**:
- EV2Gym simulator with `PublicPST.yaml` configuration
- 20 charging stations, each with 1 port
- Power Setpoint Tracking (PST) objective

**Action Space**:
- Discrete actions controlling charging intensity for all ports
- Original discretization: 6 actions {0, 0.2, 0.4, 0.6, 0.8, 1.0}
- Each action applies the same charging percentage to all 20 ports

**State Space**:
- 63-dimensional continuous vector containing:
  - Temporal features (timestep, day-of-week, time-of-day)
  - Power setpoint and charging potential
  - Per-EV information (SoC, time-to-departure, energy exchanged, etc.)

---

## Original Implementation (CreateYourCustomRLAlgorithm.ipynb)

### Network Architecture
```python
class Qnetwork(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(Qnetwork, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)  # 128 neurons
        self.layer2 = nn.Linear(128, 128)              # 128 neurons
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
```

### Hyperparameters
- **Learning Rate (LR)**: `1e-4`
- **Discount Factor (GAMMA)**: `0.99`
- **Batch Size**: `64`
- **Epsilon Decay**: `200`
- **TAU (target network update)**: `0.005`
- **Replay Memory**: `10,000`

### Reward Function
```python
def reward_function(env, *args):
    reward = -(min(env.power_setpoints[env.current_step-1],
                   env.charge_power_potential[env.current_step-1]) -
               env.current_power_usage[env.current_step-1])**2
    return reward
```

**Characteristics**:
- Simple squared tracking error (negative)
- **No normalization** - rewards can be extremely large in magnitude
- Range: `(-∞, 0]`
- Large errors result in very large negative rewards (e.g., `-15000`)

---

## Modified Implementation (Helena_CreateYourCustomRLAlgorithm_v2.ipynb)

### Key Changes Documented in Notebook

Helena explicitly documented the following changes:

```
Changes in V2:

- Learning Rate Too High/Low
  If the learning rate is too high, the NN may diverge; too low, it may not learn.
  Try lowering LR. Changed from 1e-4 to 1e-5

- Network Architecture
  Too few or too many layers/neurons can affect learning.
  Changed 128 neurons to 64 neurons in each hidden layer.

- Reward Clipping/Normalization
  If rewards vary too much, consider clipping or normalizing them.
  Reward was normalized

- Discount Factor (GAMMA)
  If GAMMA is too close to 1, the agent may focus too much on future rewards.
  Changed GAMMA from 0.99 to 0.95
```

### Modified Network Architecture
```python
class Qnetwork(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(Qnetwork, self).__init__()
        self.layer1 = nn.Linear(n_observations, 64)   # Reduced to 64
        self.layer2 = nn.Linear(64, 64)                # Reduced to 64
        self.layer3 = nn.Linear(64, n_actions)
```

### Modified Hyperparameters
- **Learning Rate (LR)**: `1e-5` (**10x smaller**)
- **Discount Factor (GAMMA)**: `0.95` (**reduced from 0.99**)
- Batch Size: `64` (unchanged)
- Epsilon Decay: `200` (unchanged)
- TAU: `0.005` (unchanged)

### Modified Reward Function
```python
def reward_function(env, *args):
    # Calculate tracking error
    error = min(env.power_setpoints[env.current_step-1],
                env.charge_power_potential[env.current_step-1]) - \
            env.current_power_usage[env.current_step-1]
    squared_error = error ** 2

    # Normalization factor: maximum possible error
    max_possible = max(abs(env.power_setpoints[env.current_step-1]),
                       abs(env.charge_power_potential[env.current_step-1]))
    max_squared_error = max_possible ** 2 if max_possible > 0 else 1

    # Normalized reward (between -1 and 0)
    reward = -squared_error / max_squared_error

    # Small bonus for near-perfect tracking
    if abs(error) < 0.05 * max_possible:
        reward += 0.1

    # Clip reward to [-1, 0.1]
    reward = max(-1, min(reward, 0.1))

    return reward
```

**Key Improvements**:
1. **Normalization**: Rewards scaled by maximum possible error
2. **Bounded range**: `[-1, 0.1]` instead of `(-∞, 0]`
3. **Performance bonus**: `+0.1` for tracking within 5% of target
4. **Explicit clipping**: Prevents extreme values

---

## Issues with Original Implementation

### 1. **Unstable Training / Divergence**

**Problem**: The learning rate of `1e-4` combined with unnormalized rewards caused unstable gradient updates.

**Why it happens**:
- Rewards ranging from `0` to `-15,000+` create massive gradient magnitudes
- Loss function: `L = (Q(s,a) - (r + γ·max Q(s',a')))`²
- Large rewards → large TD targets → large gradients → weight updates that overshoot
- Network oscillates or diverges instead of converging

**Evidence**:
```
Episode 0 reward: -3277.88
Episode 10 reward: -18541.30  (worse!)
Episode 42 reward: -4658.26   (better)
Episode 46 reward: -11762.75  (worse again)
```
High variance and no clear improvement trend indicates instability.

### 2. **Reward Scale Problem**

**Problem**: Squared error rewards grow quadratically with tracking error magnitude.

**Why it's problematic**:
- Small error (10 kW): reward = `-100`
- Medium error (50 kW): reward = `-2,500`
- Large error (100 kW): reward = `-10,000`

The reward scale differences make it hard for the network to learn consistent Q-values across different states.

### 3. **Network Capacity Mismatch**

**Problem**: 128-neuron layers might be too large for this problem.

**Why it matters**:
- State space: 63 dimensions
- Action space: 6 discrete actions
- Network parameters: ~33,000 with 128 neurons vs ~8,000 with 64 neurons
- Larger networks can overfit on limited replay memory (10,000 transitions)
- With episodic RL, early episodes have poor data quality → overfitting risk

### 4. **Excessive Future Discounting**

**Problem**: `GAMMA = 0.99` makes the agent heavily weight future rewards.

**Why it's an issue**:
- Episode length: 96 steps (simulation_length in config)
- Effective horizon: `1/(1-γ) = 100` steps → considers entire episode
- In PST, immediate tracking is more important than long-term optimization
- High gamma can make learning slower and less stable

### 5. **No Positive Reinforcement**

**Problem**: All rewards are negative → agent only learns to "avoid bad" not "do good"

**Psychology of learning**:
- Negative-only rewards: agent explores randomly until finding "least bad" actions
- Mixed rewards: agent actively seeks positive rewards → faster convergence
- Bonus for good performance creates clearer gradient signal

---

## Solutions Implemented in V2

### Solution 1: Learning Rate Reduction (`1e-4` → `1e-5`)

**How it helps**:
- Smaller weight updates per gradient step
- More stable convergence, less likely to overshoot
- Allows network to fine-tune Q-values gradually

**Trade-off**:
- Slower learning → may need more episodes
- Helena trained for 200 episodes (vs 10 in original demo)

**Best practice**:
```python
# Start with lower LR and use learning rate scheduling
LR = 1e-5
# Optional: add scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
```

### Solution 2: Reward Normalization and Clipping

**How it helps**:
```python
# Before: reward in (-∞, 0]
reward_unnormalized = -squared_error  # e.g., -10000

# After: reward in [-1, 0.1]
reward_normalized = -squared_error / max_squared_error  # e.g., -0.8
```

**Benefits**:
- **Consistent scale**: Network sees rewards in predictable range
- **Better gradient stability**: Loss function operates on bounded values
- **Faster convergence**: Q-values stabilize in a known range

**Implementation details**:
- Division by `max_possible²` normalizes by theoretical maximum error
- Fallback to `1` prevents division by zero
- Explicit `max(-1, min(reward, 0.1))` ensures bounds even with numerical errors

### Solution 3: Network Size Reduction (128 → 64 neurons)

**How it helps**:
- **Fewer parameters**: Reduces overfitting risk
- **Faster training**: Less computation per forward/backward pass
- **Better generalization**: Simpler model for relatively simple task

**Parameter count comparison**:
```
Original: 63×128 + 128×128 + 128×6 = 8,064 + 16,384 + 768 = 25,216 params
Modified: 63×64 + 64×64 + 64×6 = 4,032 + 4,096 + 384 = 8,512 params
```
**67% reduction** in parameters!

**When to use smaller networks**:
- Limited training data (10K replay buffer, ~100 episodes)
- Simple state-action relationships
- Stability issues with larger networks

### Solution 4: Discount Factor Reduction (`0.99` → `0.95`)

**How it helps**:
- Effective horizon: `1/(1-0.95) = 20` steps (vs 100 before)
- Agent focuses on immediate tracking performance
- More appropriate for PST problem where each timestep matters

**Mathematical impact**:
```
Timestep:        0     1     2     3     ...   10
γ=0.99 weight:  1.00  0.99  0.98  0.97  ...   0.90  (still 90%)
γ=0.95 weight:  1.00  0.95  0.90  0.86  ...   0.60  (only 60%)
```

Lower gamma reduces influence of distant rewards → faster credit assignment.

### Solution 5: Performance Bonus

**How it helps**:
```python
if abs(error) < 0.05 * max_possible:  # Within 5% of target
    reward += 0.1
```

**Benefits**:
- **Positive reinforcement**: Creates target behavior to aim for
- **Clearer gradient**: Sharp boundary at 5% threshold
- **Exploration incentive**: Agent actively seeks "bonus zone"

**Threshold choice**:
- 5% is reasonable for power setpoint tracking
- Could be tuned: 1% (strict), 10% (lenient)

---

## Performance Analysis

### Original Implementation Results
From the notebook output (50 episodes shown):
```
Mean Tracking Error: ~9,848 kW²
Min Tracking Error:  4,658 kW² (episode 42)
Max Tracking Error:  18,541 kW² (episode 10)
Standard Deviation: ~3,700 kW²
```

**Observations**:
- High variance (no convergence)
- No clear learning trend
- Some episodes worse than random

### Expected V2 Results
Based on the modifications:
- **More stable training**: Smaller variance in episode rewards
- **Gradual improvement**: Clearer downward trend in tracking error
- **Better final performance**: Lower minimum tracking error
- **Consistent behavior**: Less random fluctuation

**Note**: The notebook shows V2 was run for 200 episodes but outputs aren't fully shown. The reward plot would reveal convergence.

---

## Additional Issues & Advanced Solutions

### Issue 1: Action Space Design

**Current problem**: Single action controls all 20 ports identically

```python
def discretize_action(action):
    # All ports get the same value!
    return [value] * total_ports
```

**Why it's limiting**:
- Cannot charge different EVs at different rates
- Ignores per-EV state information (SoC, time-to-departure)
- Severely limits optimization potential

**Better approach**:
```python
# Option A: Independent discrete actions per port (combinatorial explosion)
n_actions = 6^20  # Infeasible!

# Option B: Continuous action space with DDPG/TD3/SAC
action_space = Box(low=-1, high=1, shape=(20,))

# Option C: Discretize into "charge profile" patterns
def discretize_action(action):
    if action == 0:  # Charge all
        return [1.0] * total_ports
    elif action == 1:  # Charge half
        return [1.0] * 10 + [0.0] * 10
    elif action == 2:  # Rotate
        offset = env.current_step % 20
        return [1.0 if i == offset else 0.0 for i in range(20)]
    # ... more intelligent patterns
```

**Recommendation**: Use **continuous action space** with DDPG or TD3 algorithm:
```python
from stable_baselines3 import DDPG

env = gym.make('EV2Gym-v1',
               config_file=config_file,
               reward_function=reward_function,
               state_function=state_function)

model = DDPG("MlpPolicy", env, learning_rate=1e-4)
model.learn(total_timesteps=100_000)
```

### Issue 2: Exploration Strategy

**Current**: ε-greedy with exponential decay

```python
eps_threshold = EPS_END + (EPS_START - EPS_END) * exp(-steps / EPS_DECAY)
# 0.9 → 0.05 over ~600 steps
```

**Problems**:
- Decay based on total steps, not episodes
- With 96-step episodes, epsilon decays in ~6 episodes
- Later episodes have minimal exploration (ε=0.05)
- May get stuck in local optima

**Better approach**:
```python
# Episode-based decay
def get_epsilon(episode, total_episodes):
    decay_fraction = episode / total_episodes
    return EPS_END + (EPS_START - EPS_END) * (1 - decay_fraction)

# Or: Step-based but slower
EPS_DECAY = 2000  # Decay over ~20 episodes instead of 6
```

### Issue 3: Replay Memory Sampling

**Current**: Uniform random sampling from replay buffer

**Problem**:
- Early bad experiences stay in memory
- Equal weight to poor early episodes and good later episodes
- No priority given to important transitions

**Better approach - Prioritized Experience Replay**:
```python
class PrioritizedReplayMemory:
    def __init__(self, capacity, alpha=0.6):
        self.memory = []
        self.priorities = []
        self.alpha = alpha  # Priority exponent

    def push(self, transition, td_error):
        priority = (abs(td_error) + 1e-5) ** self.alpha
        self.priorities.append(priority)
        self.memory.append(transition)

    def sample(self, batch_size, beta=0.4):
        # Sample based on priority
        probs = np.array(self.priorities) / sum(self.priorities)
        indices = np.random.choice(len(self.memory), batch_size, p=probs)

        # Importance sampling weights
        weights = (len(self.memory) * probs[indices]) ** (-beta)
        weights /= weights.max()

        return [self.memory[i] for i in indices], weights, indices
```

**Benefits**:
- Prioritizes transitions with high TD error (more learning potential)
- Corrects sampling bias with importance weights
- Faster convergence in practice

### Issue 4: Network Architecture Alternatives

**Current**: Simple 2-layer MLP

**Alternative architectures**:

**A) Deeper network with batch normalization**:
```python
class QNetwork(nn.Module):
    def __init__(self, n_observations, n_actions):
        super().__init__()
        self.layer1 = nn.Linear(n_observations, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.layer2 = nn.Linear(64, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.layer3 = nn.Linear(64, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.output = nn.Linear(64, n_actions)

    def forward(self, x):
        x = F.relu(self.bn1(self.layer1(x)))
        x = F.relu(self.bn2(self.layer2(x)))
        x = F.relu(self.bn3(self.layer3(x)))
        return self.output(x)
```

**B) Dueling DQN architecture**:
```python
class DuelingQNetwork(nn.Module):
    def __init__(self, n_observations, n_actions):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(n_observations, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        # Value stream
        self.value = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        # Advantage stream
        self.advantage = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, n_actions)
        )

    def forward(self, x):
        features = self.feature(x)
        value = self.value(features)
        advantage = self.advantage(features)
        # Combine: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        return value + (advantage - advantage.mean(dim=1, keepdim=True))
```

**Benefits**: Better separates state value from action advantages → faster learning

### Issue 5: Reward Engineering

**Current reward focuses only on tracking error**

**Enhanced multi-objective reward**:
```python
def enhanced_reward_function(env, *args):
    # Primary objective: tracking error (normalized)
    error = min(env.power_setpoints[env.current_step-1],
                env.charge_power_potential[env.current_step-1]) - \
            env.current_power_usage[env.current_step-1]
    max_possible = max(abs(env.power_setpoints[env.current_step-1]),
                       abs(env.charge_power_potential[env.current_step-1]))

    tracking_reward = -abs(error) / (max_possible + 1e-6)

    # Secondary objective: user satisfaction
    # Penalize if EVs won't reach desired charge
    satisfaction_penalty = 0
    for ev in env.active_evs:
        if ev.time_to_departure < 5:  # Last 5 steps
            charge_deficit = ev.desired_capacity - ev.soc
            if charge_deficit > 0.1:  # More than 10% short
                satisfaction_penalty -= 0.1

    # Tertiary objective: efficiency
    # Reward for using available capacity (avoid waste)
    if env.charge_power_potential[env.current_step-1] > 0:
        utilization = env.current_power_usage[env.current_step-1] / \
                      env.charge_power_potential[env.current_step-1]
        efficiency_bonus = 0.05 * utilization
    else:
        efficiency_bonus = 0

    # Weighted combination
    total_reward = tracking_reward + 0.3 * satisfaction_penalty + efficiency_bonus

    return np.clip(total_reward, -1, 0.5)
```

---

## Recommended Hyperparameter Tuning

### Learning Rate Schedule
```python
# Start higher, decay over time
initial_lr = 1e-4
optimizer = optim.AdamW(policy_net.parameters(), lr=initial_lr)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.995)

# Update after each episode
for episode in range(num_episodes):
    # ... training loop ...
    scheduler.step()
```

### Adaptive Epsilon
```python
# Keep higher exploration for longer
EPS_START = 1.0      # Start with full random
EPS_END = 0.1        # End with 10% random (not 5%)
EPS_DECAY = 5000     # Decay over ~50 episodes
```

### Batch Size Scaling
```python
# Start small, increase as memory fills
def get_batch_size(memory_size):
    if memory_size < 1000:
        return 32
    elif memory_size < 5000:
        return 64
    else:
        return 128
```

### Target Network Update
```python
# Current: soft update every step with TAU=0.005
# Alternative: hard update every N episodes
if episode % 10 == 0:
    target_net.load_state_dict(policy_net.state_dict())
```

---

## Training Diagnostics

### Metrics to Track

**1. Episode Rewards (already tracked)**
```python
plt.plot(episode_rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
```

**2. Tracking Error Trend**
```python
tracking_errors = [stats['tracking_error'] for stats in episode_stats]
plt.plot(tracking_errors)
plt.xlabel('Episode')
plt.ylabel('Tracking Error (kW²)')
```

**3. Q-Value Statistics**
```python
def log_q_values(state_batch):
    with torch.no_grad():
        q_values = policy_net(state_batch)
        return {
            'mean': q_values.mean().item(),
            'std': q_values.std().item(),
            'max': q_values.max().item(),
            'min': q_values.min().item()
        }
```

**4. Loss Over Time**
```python
# In optimize_model():
loss_history.append(loss.item())

# Plot
plt.plot(loss_history)
plt.xlabel('Optimization Step')
plt.ylabel('Huber Loss')
```

**5. Exploration Rate**
```python
epsilon_history.append(eps_threshold)
plt.plot(epsilon_history)
```

### Convergence Indicators

**Good signs**:
- Episode rewards increasing (less negative)
- Tracking error decreasing
- Q-values stabilizing (not exploding)
- Loss decreasing over time
- Epsilon decaying smoothly

**Bad signs**:
- Rewards diverging or oscillating wildly
- Q-values exploding (>1000 in magnitude)
- Loss increasing or NaN
- No improvement after 50+ episodes

---

## Comparison with Baseline Algorithms

### Test Against Heuristics

```python
from ev2gym.baselines.heuristics import ChargeAsFastAsPossible, ChargeAsLateAsPossible

# Test DQN
env = EV2Gym(config_file, seed=42)
dqn_rewards = test_agent(env, dqn_agent, n_episodes=10)

# Test heuristics
env = EV2Gym(config_file, seed=42)
cafap_rewards = test_agent(env, ChargeAsFastAsPossible(), n_episodes=10)

env = EV2Gym(config_file, seed=42)
calap_rewards = test_agent(env, ChargeAsLateAsPossible(), n_episodes=10)

# Compare
print(f"DQN mean reward: {np.mean(dqn_rewards):.2f}")
print(f"CAFAP mean reward: {np.mean(cafap_rewards):.2f}")
print(f"CALAP mean reward: {np.mean(calap_rewards):.2f}")
```

### Test Against MPC
```python
from ev2gym.baselines.mpc.eMPC import eMPC_G2V

env = EV2Gym(config_file, seed=42)
mpc_agent = eMPC_G2V(env, control_horizon=15)
mpc_rewards = test_agent(env, mpc_agent, n_episodes=10)
```

**Realistic expectations**:
- DQN should beat simple heuristics (CAFAP)
- DQN may not beat MPC without extensive tuning
- DQN advantage: learns from data, no model needed

---

## Final Recommendations

### For Better Convergence

1. **Use V2 hyperparameters as baseline**:
   - LR = `1e-5`
   - GAMMA = `0.95`
   - 64-neuron layers
   - Normalized rewards

2. **Increase training duration**:
   - Run 500-1000 episodes (not just 200)
   - Monitor convergence plots

3. **Upgrade to continuous actions**:
   - Use DDPG or TD3 instead of DQN
   - Allows per-port control

4. **Add curriculum learning**:
   ```python
   # Start with easier scenarios
   if episode < 100:
       config['number_of_charging_stations'] = 5  # Fewer stations
   else:
       config['number_of_charging_stations'] = 20
   ```

5. **Use Stable Baselines3**:
   ```python
   from stable_baselines3 import DQN
   from ev2gym.rl_agent.reward import SquaredTrackingErrorReward
   from ev2gym.rl_agent.state import PublicPST

   env = gym.make('EV2Gym-v1',
                  config_file=config_file,
                  reward_function=SquaredTrackingErrorReward,
                  state_function=PublicPST)

   model = DQN("MlpPolicy", env,
               learning_rate=1e-5,
               gamma=0.95,
               buffer_size=50000,
               learning_starts=1000,
               batch_size=64,
               tau=0.005,
               verbose=1)

   model.learn(total_timesteps=200_000)
   ```

### For Production Use

1. **Save trained models**:
   ```python
   torch.save(policy_net.state_dict(), 'dqn_pst_model.pth')
   ```

2. **Evaluation mode**:
   ```python
   policy_net.eval()
   with torch.no_grad():
       action = policy_net(state).max(1).indices
   ```

3. **Benchmark against optimal**:
   ```python
   from ev2gym.baselines.gurobi_models.tracking_error import PowerTrackingErrorMin

   # Get optimal solution
   optimal_agent = PowerTrackingErrorMin(replay_path)
   optimal_reward = test_agent(env, optimal_agent)

   # Calculate optimality gap
   gap = (optimal_reward - dqn_reward) / abs(optimal_reward) * 100
   print(f"DQN is {gap:.1f}% from optimal")
   ```

---

## Conclusion

Helena's modifications (v2) addressed the core convergence issues through:
1. ✅ **Learning rate reduction** (stability)
2. ✅ **Reward normalization** (consistent gradients)
3. ✅ **Network size reduction** (prevent overfitting)
4. ✅ **Discount factor adjustment** (appropriate horizon)
5. ✅ **Performance bonus** (positive reinforcement)

These changes should significantly improve training stability and convergence. However, the fundamental limitation remains the **action space design** (all ports controlled identically), which limits the maximum achievable performance.

**Next steps**:
- Verify convergence with reward plots from 200-episode training
- Test against baseline algorithms (heuristics, MPC)
- Consider upgrading to continuous action space (DDPG/TD3)
- Implement advanced techniques (prioritized replay, dueling architecture)
- Benchmark against optimal solution (Gurobi)

The modifications demonstrate good understanding of RL training challenges and appropriate solutions for stabilizing DQN learning.
