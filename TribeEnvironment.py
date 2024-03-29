# TribeEnvironment.py

import numpy as np
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from tribe import Tribe, TraitsHandler
import random
from utils import convert_state_to_tensor

class TribeEnvironment(py_environment.PyEnvironment):
    FIXED_TIME_STEP_INTERVAL = 1
    SEASONS = {"Spring": 1.2, "Summer": 1.5, "Fall": 0.8, "Winter": 0.6}

    @staticmethod
    def apply_trait_multipliers(tribe):
        trait_multipliers = {"Health Buff": 0.2, "Damage Buff": 0.2, "Resourceful": 0.2, "Aggressive": 0.2,
                             "Nomadic": 0.1, "Cooperative": 0.1, "Cautious": 0.1}
        for trait in tribe.traits:
            if trait in trait_multipliers:
                setattr(tribe, f"{trait.lower().replace(' ', '_')}_multiplier", trait_multipliers[trait])

    def __init__(self, num_tribes, num_actions, num_features):
        self._tribes = Tribe.create_and_initialize_tribes(num_tribes)
        for tribe in self._tribes:
            self.apply_trait_multipliers(tribe)
        self._num_actions = num_actions
        self._num_features = num_features
        self._action_spec = array_spec.BoundedArraySpec(shape=(), dtype=np.int32, minimum=0, maximum=num_actions - 1, name='action')
        self._observation_spec = array_spec.ArraySpec(shape=(num_features,), dtype=np.float32, name='observation')
        self._episode_ended = False
        self._current_time_step = 0
        self._reset()
        self._event_timer = 0
        self._event_interval = 10
        self.step_counter = 0
        self.max_steps = 200
        self._current_season = random.choice(list(self.SEASONS.keys()))
        print("Starting season: " + self._current_season)

    def _perform_random_events(self):
        self._event_timer += 1
        if self._event_timer >= self._event_interval:
            self._event_timer = 0
            if random.random() < 0.1:
                print("Random event: Collecting is impossible for 10 turns.")
                self._disable_collecting()
                return True
            if random.random() < 0.2:
                print("Random event: Happiness buff added to attacking.")
                self._add_happiness_buff_to_attacking()
                return True
            if random.random() < 0.15:
                for tribe in self._tribes:
                    if tribe.population > 0:
                        random_population_loss = random.randint(1, min(10, tribe.population))
                        tribe.population -= random_population_loss
                        print(f"Random event: {tribe.name} loses {random_population_loss} population.")
                return True
        return False

    def _disable_collecting(self):
        self._collecting_disabled = True
        self._collecting_disabled_duration = 10

    def _add_happiness_buff_to_attacking(self):
        self._happiness_buff_to_attacking = True
        self._happiness_buff_duration = 5

    def _kill_random_population(self):
        for tribe in self._tribes:
            if tribe.population > 0:
                random_population_loss = random.randint(1, min(10, tribe.population))
                tribe.population -= random_population_loss
                print(f"Random event: {tribe.name} loses {random_population_loss} population.")

    def get_season_multiplier(self):
        season_multipliers = {"Spring": 1.2, "Summer": 1.5, "Fall": 0.8, "Winter": 0.6}
        return season_multipliers[self._current_season]

    def perform_ai_action(self, action):
        chosen_action_key = int(np.squeeze(action))
        actions_mapping = {0: "attack", 1: "collect", 2: "trade", 3: "conflict", 4: "form_alliance", 5: "pass"}
        chosen_action_key = actions_mapping.get(chosen_action_key, "pass")
        if self._perform_random_events():
            return
        for tribe in self._tribes:
            ai_decision = {chosen_action_key: 1.0}
            other_tribes = [t for t in self._tribes if t != tribe]
            tribe.perform_actions(other_tribes, ai_decision)
            self.apply_trait_multipliers(tribe)

    def perform_actions(self, other_tribes, ai_decision):
        for tribe in other_tribes:
            if tribe is not None:
                tribe_action = None
                for action, probability in ai_decision.items():
                    if random.random() < probability:
                        tribe_action = action
                        break
                if tribe_action == "collect":
                    tribe.collect_resources()
                elif tribe_action == "trade":
                    self.trade_resources(tribe)
                elif tribe_action == "conflict":
                    self.conflict(tribe)
                elif tribe_action == "form_alliance":
                    self.form_alliance(tribe)

    def _reset(self):
        self._episode_ended = False
        self._current_tribe = np.random.choice(self._tribes)
        initial_observation = np.array([self._current_tribe.population, self._current_tribe.resources, self._current_tribe.happiness], dtype=np.float32)
        return ts.restart(initial_observation)

    def _step(self, action):
        if self._episode_ended:
            return self.reset()

        self._current_time_step = (self._current_time_step[0] + 1,)

        if self._current_time_step[0] % self.FIXED_TIME_STEP_INTERVAL == 0:
            self.reward_happiness_based_on_population()

            for tribe in self._tribes:
                self._current_tribe = tribe

                if self._current_tribe.population > 0:
                    self.perform_ai_action(action)
                    self._adjust_happiness_based_on_resources()
                    self.reproduce()
                    self.update_relationship_score()

                    reward = self._calculate_reward()
                    episode_is_done = self._check_episode_completion()

                    if episode_is_done:
                        self._episode_ended = True
                        final_observation = np.array(
                            [self._current_tribe.population, self._current_tribe.resources,
                             self._current_tribe.happiness],
                            dtype=np.float32)
                        return ts.termination(final_observation, reward)
                    else:
                        new_observation = np.array(
                            [self._current_tribe.population, self._current_tribe.resources,
                             self._current_tribe.happiness],
                            dtype=np.float32)
                        return ts.transition(new_observation, reward)

        if self._current_time_step[0] >= self.max_steps or self._check_episode_completion():
            final_observation = np.array([0, 0, 0], dtype=np.float32)
            return ts.termination(final_observation, 0.0)

        observation = np.array(
            [self._current_tribe.population, self._current_tribe.resources, self._current_tribe.happiness],
            dtype=np.float32)
        return ts.transition(observation, 0.0)

    def _update_season(self):
        seasons = list(self.SEASONS.keys())
        current_index = seasons.index(self._current_season)
        next_index = (current_index + 1) % len(seasons)
        new_season = seasons[next_index]

        if new_season != self._current_season:
            print(f"Season changed: {self._current_season} -> {new_season}")

            season_multiplier = self.get_season_multiplier()
            for tribe in self._tribes:
                tribe.resources *= int(season_multiplier)

            implications = {
                "Spring": "Abundant resources.",
                "Summer": "Possible drought.",
                "Fall": "Bountiful harvest, winter coming.",
                "Winter": "Resources may dwindle."
            }
            print(f"Implication: {implications[new_season]}")

            self._current_season = new_season
            print(f"Resources adjusted for {new_season}.")

    def _distribute_resources(self, eliminated_tribe):
        remaining_tribes = [tribe for tribe in self._tribes if tribe != eliminated_tribe]
        total_remaining_population = sum(tribe.population for tribe in remaining_tribes)
        if total_remaining_population > 0:
            resource_per_individual = eliminated_tribe.resources / total_remaining_population
            for tribe in remaining_tribes:
                tribe.resources += tribe.population * resource_per_individual
                print(
                    f"{tribe.name} receives {tribe.population * resource_per_individual} resources from {eliminated_tribe.name}.")
            eliminated_tribe.resources = 0
            eliminated_tribe.happiness = 0

    def _adjust_happiness_based_on_resources(self):
        if self._current_tribe.resources < int(self._current_tribe.population / 2):
            scarcity_factor = 1.0 - (self._current_tribe.resources / (self._current_tribe.population * 3))
            happiness_loss = int(
                self._current_tribe.happiness * scarcity_factor * 0.4)
            original_happiness = self._current_tribe.happiness
            new_happiness = self._current_tribe.happiness - happiness_loss
            self._current_tribe.happiness = max(0, new_happiness)
            print(f"{self._current_tribe.name} experiences happiness loss due to resource scarcity: {happiness_loss}")
            print(f"Original happiness: {original_happiness}, New happiness: {self._current_tribe.happiness}")

        elif self._current_tribe.resources > int(self._current_tribe.population * 2):
            abundance_factor = (self._current_tribe.resources / (self._current_tribe.population * 3)) - 1.0
            happiness_gain = int(
                self._current_tribe.happiness * abundance_factor * 0.4)
            original_happiness = self._current_tribe.happiness
            new_happiness = self._current_tribe.happiness + happiness_gain
            self._current_tribe.happiness = min(100, new_happiness)
            print(f"{self._current_tribe.name} experiences happiness gain due to resource abundance: {happiness_gain}")
            print(f"Original happiness: {original_happiness}, New happiness: {self._current_tribe.happiness}")

    def action_spec(self):
        return self._action_spec

    def interact(self, other_tribe, ai_decision):
        other_tribe.interact(self, ai_decision)

    def reproduce(self):
        for tribe in self._tribes:
            if tribe.population > 2 and tribe.resources >= tribe.population + 50:
                added_individuals = random.randint(1, 2) * int((tribe.population / 2))
                tribe.population += added_individuals
                print(
                    f"{tribe.name} reproduced. Added {added_individuals} individuals. New population: {tribe.population}, Resources: {tribe.resources}")

    def reward_happiness_based_on_population(self):
        for tribe in self._tribes:
            if tribe.population > 500:
                happiness_reward = (
                                           tribe.population // 1000) * 5
                tribe.happiness += happiness_reward

    def update_relationship_score(self):
        for tribe in self._tribes:
            tribe.update_relationship_score()

    def attack(self, other_tribe):
        for tribe in self._tribes:
            tribe.attack(other_tribe)

    def trade_resources(self, other_tribe):
        for tribe in self._tribes:
            tribe.trade_resources(other_tribe)

    def conflict(self, other_tribe):
        for tribe in self._tribes:
            tribe.conflict(other_tribe)

    def form_alliance(self, other_tribe):
        for tribe in self._tribes:
            tribe.form_alliance(other_tribe)

    def observation_spec(self):
        return self._observation_spec

    def _is_terminal(self):
        return self._episode_ended

    def _calculate_reward(self):
        if self._tribes:
            current_tribe = np.random.choice(self._tribes)

            base_reward = current_tribe.happiness / 100.0

            if current_tribe.happiness < 50:
                penalty_factor = 2.0
                reward = np.float32(base_reward / penalty_factor)
            else:
                reward = np.float32(base_reward)

            return reward
        else:
            return np.float32(0.0)

    def _check_episode_completion(self):
        num_remaining_tribes = sum(1 for tribe in self._tribes if tribe.population > 0)
        episode_is_done = num_remaining_tribes <= 1
        if episode_is_done:
            self._episode_ended = True  # Set episode ended flag
            print("Episode finished. No tribes remain.")
        return episode_is_done
