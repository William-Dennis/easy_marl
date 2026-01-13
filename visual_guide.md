# Multi-Agent Environment ABC: Visual Guide

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      gymnasium.Env                          │
│                  (Gymnasium Base Class)                     │
└────────────────────────┬────────────────────────────────────┘
                         │
                         │ inherits
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                  BaseMultiAgentEnv (ABC)                    │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Core Multi-Agent Functionality:                    │   │
│  │  • Agent management                                 │   │
│  │  • Observation/action spaces                        │   │
│  │  • Fixed policy support                             │   │
│  │  • Stochasticity handling                           │   │
│  └─────────────────────────────────────────────────────┘   │
└────────────────────────┬────────────────────────────────────┘
                         │
                         │ extends
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   BaseMarketEnv (ABC)                       │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Market-Specific Functionality:                     │   │
│  │  • Bid/offer processing                             │   │
│  │  • Market clearing mechanism                        │   │
│  │  • Price/allocation computation                     │   │
│  └─────────────────────────────────────────────────────┘   │
└────────────────────────┬────────────────────────────────────┘
                         │
                         │ implements
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              YourCustomEnvironment (Concrete)               │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Domain-Specific Implementation:                    │   │
│  │  • Electricity market rules                         │   │
│  │  • Traffic dynamics                                 │   │
│  │  • Supply chain logic                               │   │
│  │  • etc.                                             │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Episode Flow Diagram

```
START
  │
  ▼
┌─────────────────┐
│  __init__()     │  ◄── Create environment with agents & params
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  reset()        │  ◄── Initialize episode
│  ├─ seed RNG    │
│  ├─ _apply_stochasticity()
│  ├─ _initialize_state()
│  └─ return obs  │
└────────┬────────┘
         │
         │ ┌────────────────────────────────┐
         └►│ EPISODE LOOP                   │
           │                                │
           │  ┌──────────────────────────┐  │
           │  │ step(action)             │  │
           │  │  │                       │  │
           │  │  ├─ _update_agent_action()  │
           │  │  │   (for learning agent)   │
           │  │  │                       │  │
           │  │  ├─ _update_all_agent_actions()
           │  │  │   (for fixed agents)  │  │
           │  │  │                       │  │
           │  │  ├─ _compute_step()     │  │
           │  │  │   └─ Update state    │  │
           │  │  │   └─ Compute rewards │  │
           │  │  │                       │  │
           │  │  ├─ _is_terminal()      │  │
           │  │  │                       │  │
           │  │  └─ _get_obs()          │  │
           │  │      (if not done)      │  │
           │  │                          │  │
           │  │  return (obs, reward,   │  │
           │  │          done, info)    │  │
           │  └──────────────────────────┘  │
           │            │                   │
           │            │ not done          │
           │            └───────────────────┘
           │                │
           │                │ done
           │                ▼
           └───────► END EPISODE
```

## Market Environment Flow (BaseMarketEnv)

```
step(action)
     │
     ├─► Collect all agent actions
     │   ├─ Fixed agents use their policies
     │   └─ Learning agent uses provided action
     │
     ├─► _market_clearing()
     │   │
     │   ├─► Sort bids/offers
     │   ├─► Match supply & demand
     │   └─► Determine prices & allocations
     │       └─ return (prices, allocations)
     │
     ├─► _compute_rewards(prices, allocations)
     │   │
     │   ├─► Base rewards (revenue - cost)
     │   ├─► Apply penalties (constraints, losses)
     │   └─► Normalize & scale
     │       └─ return rewards[N]
     │
     └─► return (obs, reward, done, info)
```

## Method Call Hierarchy

```
User calls reset()
    └─► reset()
        ├─► _apply_stochasticity()      [Abstract - You implement]
        ├─► _initialize_state()         [Abstract - You implement]
        └─► _get_obs()                  [Abstract - You implement]

User calls step(action)
    └─► step()
        ├─► _update_agent_action(action, agent_idx)  [Abstract]
        ├─► _update_all_agent_actions()
        │   └─► For each fixed agent:
        │       ├─► _get_obs(agent_j)   [Abstract]
        │       └─► _update_agent_action(action_j, j)
        │
        ├─► _compute_step()             [Abstract - You implement]
        │   ├─► [Your simulation logic]
        │   └─► return (rewards, info)
        │
        ├─► _is_terminal()              [Abstract - You implement]
        └─► _get_obs()                  [Abstract - You implement]
```

## Data Flow Example: 3-Agent Electricity Market

```
Time t=0:
    State: {D=100, K=[50,75,100], c=[20,30,40]}
          │
    ┌─────┴─────┬─────────┬─────────┐
    │           │         │         │
Agent 0     Agent 1   Agent 2   Demand
 (Train)   (Fixed)   (Fixed)   Profile
    │           │         │         │
    ├─ obs[10]  ├─ obs[10] ├─ obs[10]│
    │           │         │         │
    ▼           ▼         ▼         ▼
[a₀=0.3]   [a₁=0.5]  [a₂=0.4]   [D=100]
    │           │         │         │
    └───────────┴─────────┴─────────┘
                │
                ▼
         Market Clearing
         ┌─────────────┐
         │ Merit Order │
         │ Dispatch    │
         └─────────────┘
                │
        ┌───────┴───────┐
        │               │
    Prices          Allocations
    P=[35]          q=[50,50,0]
        │               │
        └───────┬───────┘
                │
                ▼
         Compute Rewards
         r=[+15.2, +2.5, -1.0]
                │
                ▼
         Return r[0]=+15.2
         (for training agent)
```

## Observation Space Design Patterns

### Pattern 1: Full Observability
```
Observation = [All agent states + Environment state]
            = [pos₀, vel₀, ..., posₙ, velₙ, env_vars]

Pros: Agents have complete information
Cons: High dimensional, may be unrealistic
```

### Pattern 2: Partial Observability (Local)
```
Observation = [Own state + Nearby agents + Local env]
            = [pos_i, vel_i, nearby_positions, local_resources]

Pros: More realistic, lower dimension
Cons: Agents have limited information
```

### Pattern 3: Partial Observability (Aggregate)
```
Observation = [Own state + Aggregate statistics]
            = [pos_i, vel_i, mean_pos, total_resources]

Pros: Compact, captures global trends
Cons: Loss of detailed information
```

### Pattern 4: Communication-Based
```
Observation = [Own state + Received messages]
            = [state_i, msg_from_j, msg_from_k]

Pros: Flexible information sharing
Cons: Requires message protocol
```

## Action Space Design Patterns

### Continuous Actions
```python
# Movement in 2D space
action_space = Box(low=[-1,-1], high=[1,1], shape=(2,))

# Bid: [quantity_fraction, price_markup]
action_space = Box(low=[0,-10], high=[1,10], shape=(2,))
```

### Discrete Actions
```python
# Cardinal directions
action_space = Discrete(5)  # [N, S, E, W, Stay]

# Bid levels
action_space = Discrete(10)  # [Low price ... High price]
```

### Mixed Actions
```python
# Discrete choice + continuous parameter
action_space = Dict({
    'action_type': Discrete(3),  # [Buy, Sell, Hold]
    'amount': Box(low=0, high=1, shape=(1,))  # How much
})
```

## Common Implementation Patterns

### Scaling/Normalization
```python
# In _initialize_parameters():
self.position_scale = max_position
self.velocity_scale = max_velocity

# In _get_obs():
obs = np.concatenate([
    positions / self.position_scale,     # Normalize to [0,1]
    velocities / self.velocity_scale,
])

# In _compute_step():
raw_rewards = ...
normalized_rewards = raw_rewards / self.reward_scale
```

### Timestep Management
```python
# Option 1: Counter in _initialize_state(), increment in _compute_step()
def _initialize_state(self):
    self.t = 0

def _compute_step(self):
    # ... do computation
    self.t += 1
    return rewards, {}

# Option 2: Counter in _initialize_state(), increment in step()
# (Less common, but valid)
```

### Fixed Policy Storage
```python
# Store during initialization
self.fixed_policies = [
    agent.fixed_act_function(deterministic=True) 
    for agent in self.agents
]

# Use during step
for j in range(self.N):
    if j != self.agent_index:  # Skip learning agent
        obs_j = self._get_obs(agent_index=j)
        action_j = self.fixed_policies[j](obs_j)
        self._update_agent_action(action_j, j)
```

## Debugging Checklist

```
□ Does reset() return correct observation shape?
□ Does step() return correct tuple (obs, reward, terminated, truncated, info)?
□ Are rewards float/int (not arrays)?
□ Does _is_terminal() return bool?
□ Are all array operations vectorized (no Python loops)?
□ Are observations normalized/scaled consistently?
□ Does the environment handle seed properly?
□ Are arrays copied (not referenced) when stored?
□ Do agent indices stay in range [0, N-1]?
□ Is timestep counter properly managed?
```

## Quick Reference: Abstract Methods

| Must Implement | Purpose | Common Pattern |
|----------------|---------|----------------|
| `_initialize_parameters` | Extract params | `self.N = params["N"]` |
| `_setup_observation_space` | Define obs space | `spaces.Box(...)` |
| `_setup_action_space` | Define action space | `spaces.Box(...)` |
| `_initialize_state` | Reset variables | `self.t = 0; self.state = ...` |
| `_apply_stochasticity` | Add randomness | `self.rng.normal(...)` |
| `_get_obs` | Generate observation | `return normalized_state` |
| `_update_agent_action` | Process action | `self.actions[idx] = action` |
| `_compute_step` | Run simulation | `update state; return rewards` |
| `_is_terminal` | Check if done | `return self.t >= self.T` |
| `_render_implementation` | Visualize | `print(state)` or `plt.plot(...)` |
| `get_metadata` | Return config | `return {"N": self.N, ...}` |

## Performance Tips

1. **Use NumPy broadcasting** instead of loops
2. **Pre-allocate arrays** in `_initialize_state()`
3. **Avoid repeated normalization** - normalize once and store
4. **Use views instead of copies** when safe
5. **Profile your `_compute_step()`** - it's called most often

---

This visual guide complements the code documentation. Refer to README.md for detailed explanations and examples.