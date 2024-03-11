#Env

import numpy as np
from tf_agents.environments import py_environment
from tf_agents.trajectories import time_step as ts
import tensorflow as tf
from tf_agents.specs import array_spec
class TribeEnvironment(py_environment.PyEnvironment):
    def __init__(self, tribes, num_actions, num_features):
        self._tribes = tribes
        self._num_actions = num_actions
        self._num_features = num_features
        self._current_tribe = None

        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=num_actions - 1, name='action')

        self._observation_spec = array_spec.ArraySpec(
            shape=(num_features,), dtype=np.float32, name='observation')

        self._episode_ended = False

    def _reset(self):
        self._episode_ended = False
        np.random.shuffle(self._tribes)
        self._current_tribe_index = 0
        self._current_tribe = self._tribes[self._current_tribe_index]

        initial_observation = np.array(
            [self._current_tribe.population, self._current_tribe.resources, self._current_tribe.happiness],
            dtype=np.float32)

        initial_time_step = ts.TimeStep(
            step_type=tf.constant(ts.StepType.FIRST, dtype=tf.int32),
            reward=tf.constant(0.0, dtype=tf.float32),
            discount=tf.constant(1.0, dtype=tf.float32),
            observation=tf.constant(initial_observation, dtype=tf.float32)
        )

        return initial_time_step

    def _step(self, action):
        if self._episode_ended:
            return self.reset()

        self._current_tribe = self._tribes[0]

        self.perform_ai_action(action)

        reward = self._calculate_reward()
        episode_is_done = self._check_episode_completion()

        if episode_is_done:
            self._episode_ended = True
            final_observation = np.array(
                [self._current_tribe.population, self._current_tribe.resources, self._current_tribe.happiness],
                dtype=np.float32)

            final_time_step = ts.termination(
                observation=tf.constant(final_observation, dtype=tf.float32),
                reward=tf.constant(reward, dtype=tf.float32),
            )
            return final_time_step
        else:
            new_observation = np.array(
                [self._current_tribe.population, self._current_tribe.resources, self._current_tribe.happiness],
                dtype=np.float32)

            new_time_step = ts.transition(
                observation=tf.constant(new_observation, dtype=tf.float32),
                reward=tf.constant(reward, dtype=tf.float32),
                discount=tf.constant(1.0, dtype=tf.float32),
                step_type=tf.constant(ts.StepType.MID, dtype=tf.int32)  # Add this line
            )
            return new_time_step

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _is_terminal(self):
        return self._episode_ended

    def _calculate_reward(self):
        current_tribe = self._current_tribe
        reward = current_tribe.happiness / 100.0
        reward = np.float32(reward)
        return reward

    def _check_episode_completion(self):
        total_population = sum(tribe.population for tribe in self._tribes)
        episode_is_done = total_population <= 0
        return episode_is_done
