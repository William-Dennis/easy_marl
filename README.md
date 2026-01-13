# Multi-Agent Environment Abstract Base Class (ABC) Guide

## Overview

This package provides an abstract base class (ABC) template for creating Gymnasium-compatible multi-agent reinforcement learning (MARL) environments. The template generalizes common patterns from market-based environments and provides a structured approach to implementing custom MARL scenarios.

## Architecture

The package consists of three main components:

1. **`BaseMultiAgentEnv`**: Core ABC for general multi-agent environments
2. **`BaseMarketEnv`**: Extended ABC specifically for market-based environments
3. **Example implementations**: Concrete classes showing how to use the ABCs

### Inheritance Hierarchy

```
gym.Env
    └── BaseMultiAgentEnv (ABC)
            └── BaseMarketEnv (ABC)
                    └── YourCustomEnv (Concrete)
```

## Key Features

### 1. Agent Management
- Support for multiple agents with heterogeneous or homogeneous policies
- Fixed policies for opponent agents during single-agent training
- Flexible agent indexing for training specific agents

### 2. Observation & Action Spaces
- Customizable observation functions per agent
- Flexible action space definitions
- Support for continuous and discrete actions

### 3. Stochasticity & Reproducibility
- Built-in random seed management
- Stochastic perturbation framework
- Deterministic evaluation mode

### 4. Market-Specific Features (BaseMarketEnv)
- Abstract market clearing mechanism
- Reward computation based on market outcomes
- Bid/offer processing

## Core Abstract Methods

### Required Methods in BaseMultiAgentEnv

| Method | Purpose | Returns |
|--------|---------|---------|
| `_initialize_parameters()` | Set up environment parameters | None |
| `_setup_observation_space()` | Define observation space | None |
| `_setup_action_space()` | Define action space | None |
| `_initialize_state()` | Reset environment state | None |
| `_apply_stochasticity()` | Add random perturbations | None |
| `_get_obs()` | Generate agent observations | `np.ndarray` |
| `_update_agent_action()` | Process agent actions | None |
| `_compute_step()` | Execute environment dynamics | `Tuple[rewards, info]` |
| `_is_terminal()` | Check episode termination | `bool` |
| `_render_implementation()` | Visualize environment | None |
| `get_metadata()` | Return environment config | `Dict` |

### Additional Methods in BaseMarketEnv

| Method | Purpose | Returns |
|--------|---------|---------|
| `_initialize_market_parameters()` | Set market-specific params | None |
| `_market_clearing()` | Execute clearing mechanism | `Tuple[prices, allocations]` |
| `_compute_rewards()` | Calculate agent rewards | `np.ndarray` |

## Implementation Guide

### Step 1: Choose Your Base Class

**Use `BaseMultiAgentEnv` when:**
- Building general multi-agent environments
- Not modeling economic/market interactions
- Need maximum flexibility

**Use `BaseMarketEnv` when:**
- Modeling markets, auctions, or trading
- Agents submit bids/offers
- Have a centralized clearing mechanism

### Step 2: Implement Required Methods

Here's a minimal implementation example:

```python
from base_marl_env import BaseMultiAgentEnv
import numpy as np
from gymnasium import spaces

class MyCustomEnv(BaseMultiAgentEnv):
    
    def _initialize_parameters(self, params):
        self.N = params["num_agents"]
        self.T = params["num_timesteps"]
        # Add your domain-specific parameters
        
    def _setup_observation_space(self, observer_name):
        self.obs_dim = 10  # Example dimension
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(self.obs_dim,), dtype=np.float32
        )
        
    def _setup_action_space(self):
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, 
            shape=(2,), dtype=np.float32
        )
        
    def _initialize_state(self):
        self.t = 0
        self.state = np.zeros(self.N)
        
    def _apply_stochasticity(self):
        # Add randomness if needed
        pass
        
    def _get_obs(self, agent_index=None):
        if agent_index is None:
            agent_index = self.agent_index
        # Generate observation for agent_index
        return np.zeros(self.obs_dim, dtype=np.float32)
        
    def _update_agent_action(self, action, agent_idx):
        # Store agent's action
        self.actions[agent_idx] = action
        
    def _compute_step(self):
        # Update state based on all actions
        # Compute rewards for all agents
        rewards = np.zeros(self.N)
        return rewards, {}
        
    def _is_terminal(self):
        return self.t >= self.T
        
    def _render_implementation(self):
        print(f"Step {self.t}: state={self.state}")
        
    def get_metadata(self):
        return {
            "num_agents": self.N,
            "num_timesteps": self.T
        }
```

### Step 3: Parameter Configuration

Create a parameter dictionary when instantiating your environment:

```python
params = {
    "num_agents": 5,
    "num_timesteps": 100,
    # Add domain-specific parameters
}

env = MyCustomEnv(
    agents=agent_list,
    params=params,
    seed=42,
    agent_index=0,  # Which agent are we training?
    observer_name="default"
)
```

### Step 4: Training Loop

```python
# Reset environment
obs, info = env.reset(seed=42)

# Training loop
for episode in range(num_episodes):
    done = False
    while not done:
        # Agent selects action
        action = policy(obs)
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Update policy
        # ...
```

## Design Patterns

### Pattern 1: Observation Functions

Create modular observation functions for different information structures:

```python
def obs_full_info(state, agent_idx):
    """Full state visibility"""
    return state

def obs_partial_info(state, agent_idx):
    """Only see own data + aggregate info"""
    own_data = state[agent_idx]
    aggregate = np.mean(state)
    return np.concatenate([own_data, [aggregate]])
```

### Pattern 2: Fixed Policies

When training one agent against fixed opponents:

```python
# Opponents use fixed policies
for agent in agents[1:]:
    agent.fixed_act_function = lambda obs: some_policy(obs)

# Train agent 0
env = MyEnv(agents=agents, agent_index=0, ...)
```

### Pattern 3: Evaluation Mode

Evaluate all agents with their learned policies:

```python
# fixed_evaluation=True uses all agents' fixed policies
obs, reward, done, truncated, info = env.step(
    action=None,  # Action is ignored
    fixed_evaluation=True
)
```

## Advanced Topics

### Scaling and Normalization

Always normalize observations and rewards for stable learning:

```python
def _initialize_parameters(self, params):
    # Store raw values
    self.raw_values = np.array(params["values"])
    
    # Compute scales
    self.value_scale = np.max(self.raw_values)
    
    # Normalize
    self.normalized_values = self.raw_values / self.value_scale

def _compute_step(self):
    # Compute raw rewards
    raw_rewards = ...
    
    # Normalize
    normalized_rewards = raw_rewards / self.reward_scale
    
    return normalized_rewards, {}
```

### Multi-Objective Rewards

Structure rewards with multiple components:

```python
def _compute_rewards(self, outcomes):
    # Base reward
    base_reward = self._compute_base_reward(outcomes)
    
    # Penalties
    constraint_penalty = self._compute_constraints(outcomes)
    exploration_bonus = self._compute_exploration(outcomes)
    
    # Combine
    total_reward = base_reward - constraint_penalty + exploration_bonus
    
    return total_reward
```

### Vectorized Operations

Use NumPy broadcasting for efficiency:

```python
def _market_clearing(self):
    # Sort all agents at once
    sorted_indices = np.argsort(self.bids)
    
    # Vectorized allocation
    cumsum_capacity = np.cumsum(self.capacities[sorted_indices])
    dispatch_mask = cumsum_capacity <= self.demand
    
    allocations = np.where(
        dispatch_mask,
        self.capacities[sorted_indices],
        0
    )
    
    return allocations
```

## Common Pitfalls

### 1. Forgetting to Update Time

```python
# ❌ Wrong
def _compute_step(self):
    # ... compute rewards
    return rewards, {}

# ✅ Correct
def _compute_step(self):
    # ... compute rewards
    self.t += 1  # Advance time!
    return rewards, {}
```

### 2. Not Handling Edge Cases

```python
# ❌ Wrong
def _get_obs(self, agent_index=None):
    return self.state[self.t]  # Crashes if self.t >= len(state)

# ✅ Correct
def _get_obs(self, agent_index=None):
    t_idx = min(self.t, len(self.state) - 1)
    return self.state[t_idx]
```

### 3. Mutable Default Arguments

```python
# ❌ Wrong
def __init__(self, params={}):
    self.params = params

# ✅ Correct
def __init__(self, params=None):
    self.params = params or {}
```

### 4. Not Copying Arrays

```python
# ❌ Wrong
def _compute_step(self):
    self.output["actions"][self.t] = self.actions  # Reference!

# ✅ Correct
def _compute_step(self):
    self.output["actions"][self.t] = self.actions.copy()
```

## Testing Your Environment

```python
import pytest
from gymnasium.utils.env_checker import check_env

def test_environment():
    """Test environment satisfies Gymnasium API"""
    env = MyCustomEnv(agents=agents, params=params)
    check_env(env, skip_render_check=True)

def test_reset():
    """Test reset functionality"""
    env = MyCustomEnv(agents=agents, params=params)
    
    obs1, _ = env.reset(seed=42)
    obs2, _ = env.reset(seed=42)
    
    assert np.allclose(obs1, obs2)  # Same seed = same obs

def test_step_shapes():
    """Test output shapes"""
    env = MyCustomEnv(agents=agents, params=params)
    obs, _ = env.reset()
    
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    
    assert obs.shape == env.observation_space.shape
    assert isinstance(reward, (int, float))
    assert isinstance(done, bool)
```

## Migration Guide: From Specific to ABC

If you have an existing environment and want to refactor it to use this ABC:

1. **Identify parameters**: Extract all hardcoded values into `params` dict
2. **Separate concerns**: Split monolithic methods into abstract components
3. **Inherit and implement**: Choose appropriate base class and implement required methods
4. **Test incrementally**: Verify each method works before moving to the next
5. **Add flexibility**: Make observer functions and action spaces configurable

## Example Use Cases

### 1. Traffic Control
- Agents: Traffic signals
- Actions: Light timings
- Observations: Vehicle counts
- Rewards: Minimize wait time

### 2. Supply Chain
- Agents: Warehouses
- Actions: Inventory orders
- Observations: Stock levels, demand forecasts
- Rewards: Minimize costs + stockouts

### 3. Spectrum Allocation
- Agents: Wireless networks
- Actions: Power levels, frequency bands
- Observations: Interference, throughput
- Rewards: Maximize capacity

### 4. Peer-to-Peer Trading
- Agents: Prosumers
- Actions: Buy/sell bids
- Observations: Prices, inventory
- Rewards: Profit from trades

## Contributing

When extending the ABC:

1. **Keep it abstract**: Don't add domain-specific logic to base classes
2. **Document thoroughly**: Every abstract method needs clear docstrings
3. **Provide examples**: Show how to use new features
4. **Test extensively**: Edge cases matter
5. **Maintain backwards compatibility**: Don't break existing implementations

## License

[Your license here]

## References

- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [Multi-Agent RL Survey](https://arxiv.org/abs/1911.10635)
- Your original implementation: `MARLElectricityMarketEnv`
