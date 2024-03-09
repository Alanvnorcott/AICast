# TribeEnvironment.py

import numpy as np
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from tribe import Tribe

class TribeEnvironment(py_environment.PyEnvironment):
    def __init__(self, tribes, num_actions, num_features):
        # Initialize the environment
        self._tribes = tribes  # Use the correct attribute name (_tribes instead of _tribe)
        self._num_actions = num_actions
        self._num_features = num_features

        # Define action and observation specs
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=num_actions - 1, name='action')

        self._observation_spec = array_spec.ArraySpec(
            shape=(num_features,), dtype=np.float32, name='observation')

        # Define initial time step
        self._episode_ended = False
        self._current_time_step = None
        self._reset()

    def perform_ai_action(self, action):
        # Perform actions based on the AI agent's decision
        # Example: Assuming action is an integer representing the chosen action
        ai_decision = {"attack": 0.3, "collect": 0.5, "pass": 0.2}  # Replace with actual decision probabilities
        self._tribe.perform_actions(other_tribe=None, ai_decision=ai_decision)

    def _reset(self):
        # Reset the environment to its initial state
        # Initialize any variables or state needed for a new episode
        # Return the initial time step
        self._episode_ended = False
        # Your reset logic here

        # Choose a random tribe for the current episode
        self._current_tribe = np.random.choice(self._tribes)

        # Example: Assuming your initial observation is the tribe's population, resources, and happiness
        initial_observation = np.array(
            [self._current_tribe.population, self._current_tribe.resources, self._current_tribe.happiness])
        return ts.restart(initial_observation)

    def _step(self, action):
        # Take a step in the environment based on the given action
        # Update the state, calculate the reward, and check if the episode is done
        # Return the new time step

        if self._episode_ended:
            # The last episode ended, reset the environment
            return self.reset()

        # Choose a random tribe for the current episode
        self._current_tribe = np.random.choice(self._tribes)

        # Your step logic here
        # Update the environment state based on the action
        self.perform_ai_action(action)

        # Calculate the reward based on the new state and action
        reward = self._calculate_reward()

        # Check if the episode is done
        episode_is_done = self._check_episode_completion()

        if episode_is_done:
            self._episode_ended = True
            # Example: Assuming your final observation is the tribe's population, resources, and happiness
            final_observation = np.array(
                [self._current_tribe.population, self._current_tribe.resources, self._current_tribe.happiness])
            return ts.termination(final_observation, reward)
        else:
            # Example: Assuming your new observation is the updated tribe's population, resources, and happiness
            new_observation = np.array(
                [self._current_tribe.population, self._current_tribe.resources, self._current_tribe.happiness])
            return ts.transition(new_observation, reward)

    def action_spec(self):
        # Return the action spec for the environment
        return self._action_spec

    def observation_spec(self):
        # Return the observation spec for the environment
        return self._observation_spec

    def _is_terminal(self):
        # Return whether the current episode is terminated
        return self._episode_ended

    def _calculate_reward(self):
        # Implement your reward calculation logic based on the tribe's state and action
        # Example: Reward based on the tribe's happiness
        reward = self._tribe.happiness / 100.0  # Normalize to a range of [0, 1]
        return reward

    def _check_episode_completion(self):
        # Implement your logic to check if the episode should end
        # Example: End the episode if the tribe's population drops below a threshold
        return self._tribe.population <= 0