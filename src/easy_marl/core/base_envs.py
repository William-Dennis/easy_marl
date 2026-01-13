"""Abstract base class for Gymnasium-compatible multi-agent environments."""

from abc import ABC, abstractmethod
import gymnasium as gym
import numpy as np
from typing import Dict, Optional, Tuple, Any, List


class BaseMultiAgentEnv(ABC, gym.Env):
    """
    Abstract base class for multi-agent reinforcement learning environments.

    This template provides a structure for environments where:
    - Multiple agents interact in a shared environment
    - Each agent has its own observation and action space
    - Agents can have fixed policies or be trainable
    - The environment progresses through discrete timesteps
    - Observations can be customized per agent
    """

    metadata = {"render_modes": ["human"], "render_fps": 1}

    def __init__(
        self,
        agents: List[Any],
        params: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
        agent_index: int = 0,
        observer_name: str = "default",
    ) -> None:
        """
        Initialize the multi-agent environment.

        Args:
            agents: List of agent objects
            params: Dictionary of environment-specific parameters
            seed: Random seed for reproducibility
            agent_index: Index of the agent being trained (0-indexed)
            observer_name: Name of the observation function to use
        """
        super().__init__()
        self.seed(seed)
        self.rng = np.random.default_rng(seed)
        self.agents = agents
        self.agent_index = agent_index

        # Extract fixed policies from agents
        self.fixed_policies = [
            agent.fixed_act_function(deterministic=True) for agent in self.agents
        ]

        # Initialize parameters
        params = params or {}
        self._initialize_parameters(params)

        # Setup observation and action spaces
        self._setup_observation_space(observer_name)
        self._setup_action_space()

        # Initialize internal state
        self._initialize_state()

    @abstractmethod
    def _initialize_parameters(self, params: Dict[str, Any]) -> None:
        """
        Initialize environment-specific parameters from the params dictionary.

        This should set all necessary instance variables such as:
        - Number of agents (self.N)
        - Number of timesteps (self.T)
        - Any domain-specific parameters

        Args:
            params: Dictionary containing environment parameters
        """
        pass

    @abstractmethod
    def _setup_observation_space(self, observer_name: str) -> None:
        """
        Configure the observation space for the environment.

        This should:
        - Define self.obs_dim (dimension of observation vector)
        - Set self.observation_space (gymnasium.spaces.Space object)
        - Initialize any observation-related buffers or functions

        Args:
            observer_name: Name/identifier for the observation function
        """
        pass

    @abstractmethod
    def _setup_action_space(self) -> None:
        """
        Configure the action space for the environment.

        This should set self.action_space to a gymnasium.spaces.Space object
        that defines valid actions for each agent.
        """
        pass

    @abstractmethod
    def _initialize_state(self) -> None:
        """
        Initialize or reset the internal state of the environment.

        This should set up:
        - Timestep counter (e.g., self.t = 0)
        - Output storage structures
        - Any other stateful variables
        """
        pass

    @abstractmethod
    def _apply_stochasticity(self) -> None:
        """
        Apply stochastic perturbations to the environment state.

        This is called during reset to add randomness/variation.
        Can be a no-op if the environment is deterministic.
        """
        pass

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to an initial state.

        Args:
            seed: Optional random seed
            options: Optional reset options

        Returns:
            Tuple of (initial_observation, info_dict)
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self._apply_stochasticity()

        self._initialize_state()

        return self._get_obs(), {}

    @abstractmethod
    def _get_obs(self, agent_index: Optional[int] = None) -> np.ndarray:
        """
        Generate the observation for a specific agent.

        Args:
            agent_index: Index of agent to generate observation for.
                        If None, use self.agent_index

        Returns:
            Observation vector as numpy array
        """
        pass

    @abstractmethod
    def _update_agent_action(self, action: np.ndarray, agent_idx: int) -> None:
        """
        Process and store an agent's action.

        This should:
        - Validate the action
        - Transform it into environment-specific representation
        - Store it for use in the step function

        Args:
            action: Action array from the agent
            agent_idx: Index of the acting agent
        """
        pass

    def _update_all_agent_actions(self, exclude_agent_index: bool = True) -> None:
        """
        Populate actions for all agents using their fixed policies.

        Args:
            exclude_agent_index: If True, skip the agent at self.agent_index
        """
        for j in range(self.N):
            if exclude_agent_index and j == self.agent_index:
                continue

            fn = self.fixed_policies[j]
            if fn is not None:
                obs_j = self._get_obs(agent_index=j)
                act_j = fn(obs_j)
                act_j = np.asarray(act_j, dtype=np.float32)
                self._update_agent_action(act_j, j)

    @abstractmethod
    def _compute_step(self) -> Tuple[np.ndarray, float]:
        """
        Execute one timestep of environment dynamics.

        This is the core simulation logic that should:
        - Process all agent actions
        - Update environment state
        - Compute outcomes for all agents

        Returns:
            Tuple of (rewards_array, auxiliary_info)
            where rewards_array contains rewards for all agents
        """
        pass

    @abstractmethod
    def _is_terminal(self) -> bool:
        """
        Check if the episode has ended.

        Returns:
            True if episode is complete, False otherwise
        """
        pass

    def step(
        self, action: np.ndarray, fixed_evaluation: bool = False
    ) -> Tuple[Optional[np.ndarray], float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.

        Args:
            action: Action from the learning agent
            fixed_evaluation: If True, use fixed policies for all agents

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Collect actions from all agents
        if fixed_evaluation:
            # Use fixed policies for all agents (evaluation mode)
            self._update_all_agent_actions(exclude_agent_index=False)
        else:
            # Use fixed policies for other agents, provided action for learning agent
            self._update_all_agent_actions(exclude_agent_index=True)
            self._update_agent_action(action, self.agent_index)

        # Execute environment dynamics
        rewards, _ = self._compute_step()

        # Check termination
        terminated = self._is_terminal()
        truncated = False

        # Generate next observation
        obs = self._get_obs() if not terminated else None

        return obs, rewards[self.agent_index], terminated, truncated, {}

    def render(self) -> None:
        """Render the environment state (for debugging/visualization)."""
        self._render_implementation()

    @abstractmethod
    def _render_implementation(self) -> None:
        """
        Implement environment-specific rendering logic.

        This could print text, display graphics, etc.
        """
        pass

    def seed(self, seed: Optional[int] = None) -> List[int]:
        """
        Seed the environment's random number generator.

        Args:
            seed: Random seed value

        Returns:
            List containing the seed used
        """
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """
        Return environment configuration and parameters.

        Returns:
            Dictionary containing JSON-serializable environment metadata
        """
        pass


class BaseMarketEnv(BaseMultiAgentEnv):
    """
    Extended base class for market-based multi-agent environments.

    Adds common functionality for markets where:
    - Agents submit bids/offers
    - A clearing mechanism determines allocations and prices
    - Rewards are based on market outcomes
    """

    @abstractmethod
    def _initialize_market_parameters(self, params: Dict[str, Any]) -> None:
        """
        Initialize market-specific parameters.

        Should set:
        - Agent capacities/endowments
        - Cost/valuation parameters
        - Market rules and constraints
        """
        pass

    @abstractmethod
    def _market_clearing(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Execute the market clearing mechanism.

        Returns:
            Tuple of (prices, allocations)
            where both are arrays indexed by agent
        """
        pass

    @abstractmethod
    def _compute_rewards(
        self, prices: np.ndarray, allocations: np.ndarray
    ) -> np.ndarray:
        """
        Calculate rewards for all agents based on market outcomes.

        Args:
            prices: Market clearing prices
            allocations: Quantities allocated to each agent

        Returns:
            Array of rewards for each agent
        """
        pass

    def _compute_step(self) -> Tuple[np.ndarray, float]:
        """
        Execute market clearing and compute rewards.

        Returns:
            Tuple of (rewards_array, market_price)
        """
        prices, allocations = self._market_clearing()
        rewards = self._compute_rewards(prices, allocations)
        return rewards, prices[0] if len(prices) > 0 else 0.0
