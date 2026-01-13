"""Example: Implementing the Electricity Market Environment using the ABC."""

import numpy as np
from typing import Dict, Any, Tuple, Optional
from gymnasium import spaces

from easy_marl.core.base_envs import BaseMarketEnv


class ElectricityMarketEnv(BaseMarketEnv):
    """
    Concrete implementation of a multi-agent electricity market environment.

    This example shows how to use the BaseMarketEnv ABC to create a specific
    market environment with bidding, clearing, and reward computation.
    """

    def _initialize_parameters(self, params: Dict[str, Any]) -> None:
        """Initialize electricity market parameters."""
        # Extract basic parameters
        self.N = params["N_generators"]
        self.T = params["T"]
        self.max_bid_delta = params.get("max_bid_delta", 50.0)
        self.lambda_bid_penalty = params.get("lambda_bid_penalty", 0.01)

        # Initialize market-specific parameters
        self._initialize_market_parameters(params)

    def _initialize_market_parameters(self, params: Dict[str, Any]) -> None:
        """Initialize generator capacities, costs, and demand profile."""
        # Demand profile over time
        self.D_profile = np.array(params["demand_profile"], dtype=np.float32)
        self.base_D_profile = self.D_profile.copy()

        # Generator parameters
        self.K = np.array(params["capacities"], dtype=np.float32)  # Capacities
        self.c = np.array(params["costs"], dtype=np.float32)  # Marginal costs

        # Validation
        assert len(self.D_profile) == self.T, "Demand profile length must match T"
        assert np.all(self.base_D_profile > 0), "Demand must be positive"

        # Compute scaling factors for normalization
        self.demand_scale = (
            np.max(self.D_profile) if np.max(self.D_profile) > 0 else 1.0
        )
        self.cost_scale = np.max(self.c) if np.max(self.c) > 0 else 1.0
        self.capacity_scale = np.max(self.K) if np.max(self.K) > 0 else 1.0
        self.system_capacity = np.sum(self.K)

        # Stochasticity bounds
        self.lower_stochastic_bound = -0.5 * np.min(self.D_profile)
        self.upper_stochastic_bound = 0.5 * (
            self.system_capacity - np.max(self.D_profile)
        )

        # Loss of load penalty
        self.unit_lol_penalty = params.get("unit_lol_penalty", 1.0)

    def _setup_observation_space(self, observer_name: str) -> None:
        """Setup observation space using a registered observer function."""
        # In practice, you'd look up the observer from your OBSERVERS dict
        # obs_dim_fn, obs_fn = OBSERVERS[observer_name]
        # self.obs_dim = obs_dim_fn(self.N)
        # self._obs_fn = obs_fn

        # For this example, we'll use a simple observation
        self.obs_dim = 2 * self.N + 1  # Example: [capacities, costs, current_demand]
        self.other_mask = np.arange(self.N) != self.agent_index
        self.obs_buf = np.empty(self.obs_dim, dtype=np.float32)

        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )

    def _setup_action_space(self) -> None:
        """
        Setup action space for quantity and price bidding.

        Action format: [quantity_fraction, price_delta]
        - quantity_fraction: [0, 1] representing fraction of capacity to offer
        - price_delta: [-bound, bound] representing markup/markdown from cost
        """
        reasonable_bound = 10
        self.action_space = spaces.Box(
            low=np.array([0.0, -reasonable_bound], dtype=np.float32),
            high=np.array([1.0, reasonable_bound], dtype=np.float32),
        )

    def _initialize_state(self) -> None:
        """Initialize timestep counter and output storage."""
        self.t = 0

        # Storage for trajectory data
        self.output = {
            "bids": np.zeros((self.T, self.N), dtype=np.float32),
            "q_offered": np.zeros((self.T, self.N), dtype=np.float32),
            "q_cleared": np.zeros((self.T, self.N), dtype=np.float32),
            "market_prices": np.zeros(self.T, dtype=np.float32),
            "rewards": np.zeros((self.T, self.N), dtype=np.float32),
        }

        # Current timestep actions
        self.b_all = np.zeros(self.N, dtype=np.float32)  # Bids (prices)
        self.q_all = np.zeros(self.N, dtype=np.float32)  # Quantities offered

    def _apply_stochasticity(self) -> None:
        """Apply stochastic perturbations to demand profile."""
        # Generate random demand shock
        demand_perturbation = self.rng.normal(
            loc=0.0, scale=0.05 * self.demand_scale, size=self.T
        ).astype(np.float32)

        # Clip to avoid negative or infeasible demand
        demand_perturbation = np.clip(
            demand_perturbation,
            a_min=self.lower_stochastic_bound,
            a_max=self.upper_stochastic_bound,
        )

        # Update demand profile
        self.D_profile = self.base_D_profile + demand_perturbation

    def _get_obs(self, agent_index: Optional[int] = None) -> np.ndarray:
        """Generate observation for a specific agent."""
        if agent_index is None:
            agent_index = self.agent_index

        # Simple observation: normalized capacities, costs, and current demand
        # In practice, use your custom observer function
        obs = np.concatenate(
            [
                self.K / self.capacity_scale,
                self.c / self.cost_scale,
                [self.D_profile[min(self.t, self.T - 1)] / self.demand_scale],
            ]
        )

        return obs.astype(np.float32)

    def _update_agent_action(self, action: np.ndarray, agent_idx: int) -> None:
        """
        Convert agent's action into quantity offered and price bid.

        Args:
            action: [quantity_fraction, price_delta] from agent
            agent_idx: Index of the agent
        """
        # Quantity: fraction of capacity to offer
        q_t = float(1 - action[0]) * self.K[agent_idx]

        # Price: marginal cost + tanh-transformed delta
        b_t = float(self.c[agent_idx]) + float(np.tanh(action[1])) * self.max_bid_delta

        self.q_all[agent_idx] = q_t
        self.b_all[agent_idx] = b_t

    def _market_clearing(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Execute merit-order market clearing.

        Returns:
            Tuple of (clearing_prices, quantities_cleared)
        """
        # Call your domain-specific clearing function
        # from easy_marl.examples.bidding.market import market_clearing
        # P_t, q_cleared = market_clearing(self.b_all, self.q_all, self.D_profile[self.t])

        # Placeholder implementation (replace with your actual clearing logic)
        demand = self.D_profile[self.t]

        # Sort generators by bid price (merit order)
        sorted_indices = np.argsort(self.b_all)
        q_cleared = np.zeros(self.N, dtype=np.float32)

        remaining_demand = demand
        clearing_price = 0.0

        for idx in sorted_indices:
            if remaining_demand <= 0:
                break

            # Dispatch up to capacity or remaining demand
            dispatched = min(self.q_all[idx], remaining_demand)
            q_cleared[idx] = dispatched
            remaining_demand -= dispatched
            clearing_price = self.b_all[idx]  # Marginal price

        # Uniform pricing: all generators receive the clearing price
        prices = np.full(self.N, clearing_price, dtype=np.float32)

        return prices, q_cleared

    def _compute_rewards(
        self, prices: np.ndarray, allocations: np.ndarray
    ) -> np.ndarray:
        """
        Calculate rewards based on market outcomes.

        Reward components:
        1. Revenue minus cost: (price - cost) * quantity
        2. Bid penalty: discourage extreme bids
        3. Loss of load penalty: shared penalty for unmet demand
        """
        # Base revenue
        base_rewards = (prices - self.c) * allocations

        # Bid regularization penalty
        bid_penalties = self.lambda_bid_penalty * (self.b_all - self.c) ** 2

        # System-wide loss of load penalty
        demand = self.D_profile[self.t]
        total_cleared = np.sum(allocations)
        loss_of_load = max(0, demand - total_cleared)
        lol_penalty = self.unit_lol_penalty * loss_of_load

        # Combine rewards
        rewards = base_rewards - bid_penalties - lol_penalty

        # Normalize rewards
        rewards /= self.demand_scale * self.cost_scale * max(1, self.T)
        rewards *= 20  # Scale to reasonable range for RL

        return rewards

    def _compute_step(self) -> Tuple[np.ndarray, float]:
        """
        Execute one market clearing step.

        Returns:
            Tuple of (rewards_array, clearing_price)
        """
        # Run market clearing
        prices, q_cleared = self._market_clearing()

        # Compute rewards
        rewards = self._compute_rewards(prices, q_cleared)

        # Store outputs for analysis
        t_idx = self.t
        self.output["bids"][t_idx] = self.b_all.copy()
        self.output["q_offered"][t_idx] = self.q_all.copy()
        self.output["q_cleared"][t_idx] = q_cleared
        self.output["market_prices"][t_idx] = prices[0]
        self.output["rewards"][t_idx] = rewards

        # Advance time
        self.t += 1

        return rewards, float(prices[0])

    def _is_terminal(self) -> bool:
        """Check if episode has ended (all timesteps completed)."""
        return self.t >= self.T

    def _render_implementation(self) -> None:
        """Print current market state."""
        t_idx = min(self.t - 1, self.T - 1)
        agent_idx = self.agent_index

        print(f"Timestep: {self.t}/{self.T}")
        print(f"  Demand: {self.D_profile[t_idx]:.2f}")
        print(f"  Agent {agent_idx} bid: {self.output['bids'][t_idx, agent_idx]:.2f}")
        print(
            f"  Agent {agent_idx} quantity: {self.output['q_cleared'][t_idx, agent_idx]:.2f}"
        )
        print(f"  Market price: {self.output['market_prices'][t_idx]:.2f}")
        print(
            f"  Agent {agent_idx} reward: {self.output['rewards'][t_idx, agent_idx]:.4f}"
        )

    def get_metadata(self) -> Dict[str, Any]:
        """Return environment configuration."""
        return {
            "N_generators": self.N,
            "T": self.T,
            "capacities": self.K.tolist(),
            "costs": self.c.tolist(),
            "demand_profile": self.D_profile.tolist(),
            "max_bid_delta": self.max_bid_delta,
            "lambda_bid_penalty": self.lambda_bid_penalty,
            "unit_lol_penalty": self.unit_lol_penalty,
        }


# Example usage
if __name__ == "__main__":
    # Define mock agents
    class MockAgent:
        def fixed_act_function(self, deterministic=True):
            # Return a simple fixed policy
            return lambda obs: np.array([0.5, 0.0])  # Offer 50% capacity at cost

    # Create environment
    agents = [MockAgent() for _ in range(3)]

    params = {
        "N_generators": 3,
        "T": 24,
        "demand_profile": [100 + 50 * np.sin(i * np.pi / 12) for i in range(24)],
        "capacities": [50, 75, 100],
        "costs": [20, 30, 40],
        "max_bid_delta": 50.0,
        "lambda_bid_penalty": 0.01,
    }

    env = ElectricityMarketEnv(
        agents=agents,
        params=params,
        seed=42,
        agent_index=0,
    )

    # Run a simple episode
    obs, info = env.reset()
    print(f"Initial observation shape: {obs.shape}")

    for t in range(5):
        action = env.action_space.sample()  # Random action
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()

        if terminated:
            break

    print("\nMetadata:", env.get_metadata())
