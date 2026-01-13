"""Quick-start template for creating a custom multi-agent environment.

Copy this file and fill in the TODOs with your domain-specific logic.
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional
from gymnasium import spaces

from base_marl_env import BaseMultiAgentEnv  # or BaseMarketEnv


class MyCustomEnv(BaseMultiAgentEnv):
    """
    TODO: Add a description of your environment.

    What problem does it model?
    What are the agents?
    What are the actions?
    What are the observations?
    """

    def _initialize_parameters(self, params: Dict[str, Any]) -> None:
        """Initialize environment parameters from the params dictionary."""

        # TODO: Extract basic parameters
        self.N = params["num_agents"]  # Number of agents
        self.T = params["num_timesteps"]  # Episode length

        # TODO: Add your domain-specific parameters
        # Examples:
        # self.grid_size = params["grid_size"]
        # self.max_speed = params["max_speed"]
        # self.resource_capacity = params["resource_capacity"]

        pass

    def _setup_observation_space(self, observer_name: str) -> None:
        """Configure the observation space."""

        # TODO: Define observation dimension
        self.obs_dim = None  # e.g., 10

        # TODO: Create observation space
        self.observation_space = spaces.Box(
            low=-np.inf,  # or specific lower bounds
            high=np.inf,  # or specific upper bounds
            shape=(self.obs_dim,),
            dtype=np.float32,
        )

        # Optional: Initialize observation buffers
        self.obs_buf = np.empty(self.obs_dim, dtype=np.float32)

    def _setup_action_space(self) -> None:
        """Configure the action space."""

        # TODO: Define action space based on your problem

        # Example 1: Continuous action space
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),  # e.g., [velocity_x, velocity_y]
            dtype=np.float32,
        )

        # Example 2: Discrete action space
        # self.action_space = spaces.Discrete(5)  # e.g., [up, down, left, right, stay]

        # Example 3: Multi-discrete action space
        # self.action_space = spaces.MultiDiscrete([3, 4, 2])  # multiple discrete choices

    def _initialize_state(self) -> None:
        """Initialize the environment state."""

        # TODO: Reset timestep counter
        self.t = 0

        # TODO: Initialize state variables
        # Examples:
        # self.positions = np.zeros((self.N, 2))
        # self.velocities = np.zeros((self.N, 2))
        # self.resources = np.zeros(self.N)

        # TODO: Initialize storage for trajectory data (optional but recommended)
        self.output = {
            # "positions": np.zeros((self.T, self.N, 2)),
            # "rewards": np.zeros((self.T, self.N)),
            # Add whatever you want to track
        }

        # TODO: Initialize current action storage
        self.current_actions = np.zeros((self.N, self.action_space.shape[0]))

    def _apply_stochasticity(self) -> None:
        """Apply random perturbations to the environment."""

        # TODO: Add stochastic elements if needed
        # Examples:
        # - Random initial positions
        # - Random resource distributions
        # - Demand/supply shocks

        # Example: Random initial positions
        # self.initial_positions = self.rng.uniform(
        #     low=0, high=self.grid_size, size=(self.N, 2)
        # )

        pass  # Remove if you add stochasticity

    def _get_obs(self, agent_index: Optional[int] = None) -> np.ndarray:
        """Generate observation for a specific agent."""

        if agent_index is None:
            agent_index = self.agent_index

        # TODO: Construct observation vector for the agent
        # Consider:
        # - What information does this agent have access to?
        # - Should it see other agents' states? (Full vs partial observability)
        # - How should you normalize/scale the values?

        # Example: Simple observation
        # obs = np.concatenate([
        #     self.positions[agent_index],  # Own position
        #     self.velocities[agent_index],  # Own velocity
        #     np.mean(self.positions, axis=0),  # Average position of all agents
        # ])

        obs = np.zeros(self.obs_dim, dtype=np.float32)  # Placeholder
        return obs

    def _update_agent_action(self, action: np.ndarray, agent_idx: int) -> None:
        """Process and store an agent's action."""

        # TODO: Validate and transform the action if needed
        # Examples:
        # - Clip to valid range
        # - Transform continuous to discrete
        # - Apply action constraints

        # TODO: Store the action for use in _compute_step
        self.current_actions[agent_idx] = action

    def _compute_step(self) -> Tuple[np.ndarray, Any]:
        """Execute one timestep of environment dynamics."""

        # TODO: Update environment state based on all agents' actions
        # This is the core simulation logic

        # Example physics update:
        # for i in range(self.N):
        #     self.velocities[i] += self.current_actions[i] * dt
        #     self.positions[i] += self.velocities[i] * dt

        # TODO: Compute rewards for all agents
        rewards = np.zeros(self.N, dtype=np.float32)

        # Example reward computation:
        # for i in range(self.N):
        #     rewards[i] = -np.linalg.norm(self.positions[i] - self.target)

        # TODO: Store trajectory data if needed
        # self.output["positions"][self.t] = self.positions.copy()
        # self.output["rewards"][self.t] = rewards.copy()

        # TODO: Advance time
        self.t += 1

        # Return rewards and auxiliary info (can be None)
        return rewards, {}

    def _is_terminal(self) -> bool:
        """Check if the episode has ended."""

        # TODO: Define termination conditions
        # Examples:
        # - Timestep limit reached: self.t >= self.T
        # - Goal achieved: np.all(distances < threshold)
        # - Failure state: collision detected

        return self.t >= self.T

    def _render_implementation(self) -> None:
        """Render/visualize the environment state."""

        # TODO: Implement visualization
        # Options:
        # - Print text summary
        # - Plot with matplotlib
        # - Use pygame for interactive visualization

        # Simple text output:
        print(f"Step {self.t}/{self.T}")
        # print(f"  Positions: {self.positions}")
        # print(f"  Rewards: {self.output['rewards'][self.t-1]}")

    def get_metadata(self) -> Dict[str, Any]:
        """Return environment configuration as a dictionary."""

        # TODO: Return all relevant parameters
        return {
            "num_agents": self.N,
            "num_timesteps": self.T,
            # Add all your custom parameters
            # "grid_size": self.grid_size,
            # "max_speed": self.max_speed,
        }


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # TODO: Define your agent objects
    class DummyAgent:
        """Placeholder agent with a fixed policy."""

        def fixed_act_function(self, deterministic=True):
            return lambda obs: np.zeros(2)  # TODO: Replace with real policy

    # TODO: Create agents
    num_agents = 3
    agents = [DummyAgent() for _ in range(num_agents)]

    # TODO: Define parameters
    params = {
        "num_agents": num_agents,
        "num_timesteps": 100,
        # Add your domain-specific parameters
    }

    # TODO: Create environment
    env = MyCustomEnv(
        agents=agents,
        params=params,
        seed=42,
        agent_index=0,  # Which agent to train
        observer_name="default",
    )

    # Test the environment
    print("Testing environment...")
    obs, info = env.reset()
    print(f"✓ Reset successful. Observation shape: {obs.shape}")

    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"✓ Step {i + 1}: reward={reward:.4f}, done={terminated}")

        if terminated:
            break

    print("\n✓ Environment test complete!")
    print("Metadata:", env.get_metadata())

    # TODO: Replace dummy agents with real RL agents and train!
