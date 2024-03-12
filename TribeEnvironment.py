# TribeEnvironment.py

import numpy as np
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from tribe import Tribe, TraitsHandler
import random
from utils import convert_state_to_tensor
def apply_trait_multipliers(tribe):
    trait_multipliers = {
        "Health Buff": 0.2,
        "Damage Buff": 0.2,
        "Resourceful": 0.2,
        "Aggressive": 0.2,
        "Nomadic": 0.1,
        "Cooperative": 0.1,
        "Cautious": 0.1
    }

    for trait in tribe.traits:
        if trait in trait_multipliers:
            setattr(tribe, f"{trait.lower().replace(' ', '_')}_multiplier", trait_multipliers[trait])

class TribeEnvironment(py_environment.PyEnvironment):

    def __init__(self, num_tribes, num_actions, num_features):
        self._tribes = Tribe.create_and_initialize_tribes(num_tribes)
        self._num_actions = num_actions
        self._num_features = num_features

        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=num_actions - 1, name='action')

        self._observation_spec = array_spec.ArraySpec(
            shape=(num_features,), dtype=np.float32, name='observation')

        self._episode_ended = False
        self._current_time_step = None
        self._reset()
        self._event_timer = 0
        self._event_interval = 60
        self.step_counter = 0
        self.max_steps = 200  # Adjust this value according to your needs

    def _perform_random_events(self):
        # Increment the timer in each step
        self._event_timer += 1

        # Check if it's time to trigger events
        if self._event_timer >= self._event_interval:
            self._event_timer = 0  # Reset the timer

            # Random event: Make collecting impossible for a bit
            if random.random() < 0.1:  # Adjust probability as needed
                print("Random event: Collecting is impossible for 10 turns.")
                self._disable_collecting()
                return True  # Event occurred

            # Random event: Add a happiness buff to attacking
            if random.random() < 0.2:  # Adjust probability as needed
                print("Random event: Happiness buff added to attacking.")
                self._add_happiness_buff_to_attacking()
                return True  # Event occurred

            # Random event: Kill off a random number of population from each tribe
            if random.random() < 0.15:  # Adjust probability as needed
                for tribe in self._tribes:
                    if tribe.population > 0:
                        random_population_loss = random.randint(1, min(10, tribe.population))
                        tribe.population -= random_population_loss
                        print(f"Random event: {tribe.name} loses {random_population_loss} population.")
                return True  # Event occurred

        return False  # No event occurred

    def _disable_collecting(self):
        self._collecting_disabled = True
        self._collecting_disabled_duration = 10
    pass

    def _add_happiness_buff_to_attacking(self):
        self._happiness_buff_to_attacking = True
        self._happiness_buff_duration = 5  # Adjust the duration as needed
        pass

    def _kill_random_population(self):
        # Implement logic to kill off a random number of population from each tribe
        for tribe in self._tribes:
            if tribe.population > 0:
                random_population_loss = random.randint(1, min(10, tribe.population))
                tribe.population -= random_population_loss
                print(f"Random event: {tribe.name} loses {random_population_loss} population.")

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
            apply_trait_multipliers(tribe)  # Apply trait multipliers

    def perform_actions(self, other_tribes, ai_decision):
        for tribe in other_tribes:
            if tribe is not None:
                tribe.perform_ai_action(ai_decision)
                if self._perform_random_events():
                    return
                # Check AI decision and call corresponding methods
                for action, probability in ai_decision.items():
                    if random.random() < probability:
                        if action == "trade":
                            self.trade_resources(tribe)
                        elif action == "conflict":
                            self.conflict(tribe)
                        elif action == "form_alliance":
                            self.form_alliance(tribe)
                        # Add more conditions for other actions as needed
    def _reset(self):
        self._episode_ended = False
        self._current_tribe = np.random.choice(self._tribes)

        initial_observation = np.array(
            [self._current_tribe.population, self._current_tribe.resources, self._current_tribe.happiness],
            dtype=np.float32)
        return ts.restart(initial_observation)

    def _step(self, action):
        if self._episode_ended:
            return self.reset()

        # Increment the timer in each step
        self._event_timer += 1

        # Check if it's time to trigger events
        if self._event_timer >= self._event_interval:
            self._event_timer = 0  # Reset the timer

            # Random event: Make collecting impossible for a bit
            if random.random() < 0.1:  # Adjust probability as needed
                print("Random event: Collecting is impossible for 10 turns.")
                self._disable_collecting()

            # Random event: Add a happiness buff to attacking
            if random.random() < 0.2:  # Adjust probability as needed
                print("Random event: Happiness buff added to attacking.")
                self._add_happiness_buff_to_attacking()

            # Random event: Kill off a random number of population from each tribe
            if random.random() < 0.15:  # Adjust probability as needed
                for tribe in self._tribes:
                    if tribe.population > 0:
                        random_population_loss = random.randint(1, min(10, tribe.population))
                        tribe.population -= random_population_loss
                        print(f"Random event: {tribe.name} loses {random_population_loss} population.")

        # Select a tribe randomly to perform actions
        self._current_tribe = np.random.choice(self._tribes)

        # Check if the tribe is eliminated before performing actions
        if self._current_tribe.population > 0:
            self.perform_ai_action(action)
            self._adjust_happiness_based_on_resources()  # Adjust happiness based on resources
            self.reproduce()
            self.update_relationship_score()  # Update relationship scores

            reward = self._calculate_reward()
            episode_is_done = self._check_episode_completion()

            if episode_is_done:
                self._episode_ended = True
                final_observation = np.array(
                    [self._current_tribe.population, self._current_tribe.resources, self._current_tribe.happiness],
                    dtype=np.float32)
                return ts.termination(final_observation, reward)
            else:
                new_observation = np.array(
                    [self._current_tribe.population, self._current_tribe.resources, self._current_tribe.happiness],
                    dtype=np.float32)
                return ts.transition(new_observation, reward)

        # If the current tribe is eliminated, proceed to the next step
        return self._step(action)

    def _adjust_happiness_based_on_resources(self):
        # Check if resources are below the amount needed to feed everyone
        if self._current_tribe.resources < self._current_tribe.population * 3:
            scarcity_factor = 1.0 - (self._current_tribe.resources / (self._current_tribe.population * 3))
            happiness_loss = int(self._current_tribe.happiness * scarcity_factor)
            original_happiness = self._current_tribe.happiness  # Store original happiness for logging if needed
            self._current_tribe.happiness -= happiness_loss
            print(f"{self._current_tribe.name} experiences happiness loss due to resource scarcity: {happiness_loss}")
            # Adjust happiness exponentially based on scarcity_factor or any other desired formula
            # Example: self._current_tribe.happiness *= scarcity_factor
            print(f"Original happiness: {original_happiness}, New happiness: {self._current_tribe.happiness}")

    def action_spec(self):
        return self._action_spec

    def interact(self, other_tribe, ai_decision):
        other_tribe.interact(self, ai_decision)

    def reproduce(self):
        for tribe in self._tribes:
            if tribe.population > 2 and tribe.resources > tribe.population * 3:
                added_individuals = random.randint(1, 2) * int((tribe.population / 2))
                tribe.population += added_individuals
                print(
                    f"{tribe.name} reproduced. Added {added_individuals} individuals. New population: {tribe.population}, Resources: {tribe.resources}")

    def update_relationship_score(self):
        for tribe in self._tribes:
            tribe.update_relationship_score()

    def attack(self, other_tribe):
        for tribe in self._tribes:
            tribe.attack(other_tribe)

            # Apply happiness bonus to attacking if the buff is active
            if tribe._happiness_buff_to_attacking:
                base_happiness_gain = int(
                    0.2 * tribe.final_happiness_loss)  # Example: 20% of happiness loss as base gain
                happiness_buff_bonus = int(
                    0.1 * tribe.final_happiness_loss)  # Additional 10% bonus if the buff is active
                self_happiness_gain = min(base_happiness_gain + happiness_buff_bonus, 100)  # Cap at 100

                # Update attacker's happiness
                tribe.happiness = max(0, min(100, tribe.happiness + self_happiness_gain))

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

            # Calculate base reward based on happiness
            base_reward = current_tribe.happiness / 100.0

            # Penalize more if happiness is below 50
            if current_tribe.happiness < 50:
                penalty_factor = 2.0  # You can adjust this factor based on how much more penalty you want
                reward = np.float32(base_reward / penalty_factor)
            else:
                reward = np.float32(base_reward)

            return reward
        else:
            return np.float32(0.0)

    def _check_episode_completion(self):
        if self._tribes:
            total_population = sum(tribe.population for tribe in self._tribes)
            episode_is_done = total_population <= 0
            return episode_is_done
        else:
            return False
