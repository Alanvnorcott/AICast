# tribe.py

import random
from traits import TraitsHandler

class Tribe:
    def __init__(self, traits, name):
        self.traits = traits
        self.name = name
        self.population = 10  # Initial population size
        self.resources = 5000  # Initial resource amount
        self.turns_without_enough_resources = 0  # Tracks consecutive turns without enough resources
        self.happiness = 100  # Initial happiness value (between 0 and 100)

        self.health_multiplier = 1.0
        self.damage_multiplier = 1.0
        self.resource_gain_multiplier = 1.0
        self.attack_multiplier = 1.0
        self.alliance_bonus = 0.0
        self.happiness_multiplier = 1.0

        self.relationship_score = 5  # Initial relationship score, between 1 and 10
        TraitsHandler.apply_trait_multipliers(self)

    def perform_actions(self, other_tribe):
        # Perform actions based on traits, e.g., affect health, damage, resource gathering, etc.
        action = random.choices(["Attack", "Collect", "Pass"], weights=[0.3, 0.4, 0.3])[0]

        if other_tribe is not None:  # Add a check for None
            if action == "Attack":
                self.attack(other_tribe)
            elif action == "Collect":
                self.collect_resources()
            elif action == "Pass":
                self.pass_action()

        # Update happiness based on various factors
        if self.population > 0:
            consumed_resources = self.population * 9
            self.resources -= consumed_resources
            self.happiness = max(0, min(100, int(self.resources) / (int(self.population) * 3) * 100))

        # Check resource consumption and population control
        if self.resources < self.population * 3:
            self.turns_without_enough_resources += 1
        else:
            self.turns_without_enough_resources = 0

        if self.turns_without_enough_resources >= 2:
            self.population = max(0, self.population - 10)  # Perish 10 individuals if not enough resources for 2 turns


    def attack(self, other_tribe):
        # Implement attack action
        attack_strength = 10  # Adjust based on traits
        loser_population_loss = min(other_tribe.population, int(attack_strength * (other_tribe.population / (self.population + other_tribe.population))))
        loser_population_loss = max(1, loser_population_loss)  # Ensure at least one individual is lost

        other_tribe.population -= loser_population_loss
        print(f"{self.name} attacks {other_tribe.name}. {other_tribe.name} loses {loser_population_loss} population.")

        # Ensure the winner has at least two survivors
        remaining_population = max(2, self.population - loser_population_loss)
        self.population = remaining_population

    def collect_resources(self):
        # Implement collect action
        resource_gain = 20  # Adjust based on traits
        self.resources += resource_gain

    def pass_action(self):
        # Implement pass action
        pass

    def reproduce(self):
        if self.population > 2 and self.resources > self.population * 3:
            added_individuals = random.randint(1, 3)
            self.population += added_individuals
        pass

    def break_off(self):
        # Implement mechanism for tribes to break off into sub-tribes based on happiness
        if self.happiness < random.uniform(0, 50):
            print(f"{self.name} breaks off into a sub-tribe.")

    def interact(self, other_tribe):
        # Update relationship score based on previous interactions
        self.update_relationship_score()
        other_tribe.update_relationship_score()

        # Adjust weights for interaction based on relationship score
        weights = [0.4, 0.3, 0.3]  # Default weights
        if self.relationship_score > 5:
            weights[0] += 0.1  # Increase weight for trade with positive relationship
        elif self.relationship_score < 5:
            weights[1] += 0.1  # Increase weight for conflict with negative relationship
        else:
            weights[2] += 0.1  # Increase weight for alliance with neutral relationship

        interaction_type = random.choices(["Trade", "Conflict", "Alliance"], weights=weights)[0]

        if interaction_type == "Trade":
            self.trade_resources(other_tribe)
        elif interaction_type == "Conflict":
            self.conflict(other_tribe)
        elif interaction_type == "Alliance":
            self.form_alliance(other_tribe)

    def update_relationship_score(self):
        # Update relationship score based on past interactions
        if self.population > 0:
            self.relationship_score += random.randint(-1, 1)  # Randomly adjust score by -1, 0, or 1
            self.relationship_score = max(1, min(10, self.relationship_score))  # Ensure score stays within bounds

    def trade_resources(self, other_tribe):
        # Implement trade action
        trade_amount = 50  # Adjust based on traits
        if self.resources >= trade_amount and other_tribe.resources >= trade_amount:
            self.resources -= trade_amount
            other_tribe.resources += trade_amount
            print(f"{self.name} and {other_tribe.name} engage in trade.")

    def conflict(self, other_tribe):
        # Implement conflict action
        conflict_strength_self = 20  # Adjust based on traits
        conflict_strength_other = 20  # Adjust based on traits

        total_population = self.population + other_tribe.population

        if total_population > 0:
            self_population_loss = min(self.population, int(conflict_strength_other * (self.population / total_population)))
            other_population_loss = min(other_tribe.population, int(conflict_strength_self * (other_tribe.population / total_population)))

            self.population -= self_population_loss
            other_tribe.population -= other_population_loss

            print(f"{self.name} and {other_tribe.name} engage in conflict.")
            print(f"{self.name} loses {self_population_loss} population.")
            print(f"{other_tribe.name} loses {other_population_loss} population.")
        else:
            print(f"{self.name} and {other_tribe.name} can't engage in conflict due to zero total population.")

    def form_alliance(self, other_tribe):
        # Implement alliance action
        alliance_bonus = 0.2  # Adjust based on traits
        self.resources += int(other_tribe.resources * alliance_bonus)
        other_tribe.resources += int(self.resources * alliance_bonus)

        print(f"{self.name} and {other_tribe.name} form an alliance.")


def create_tribes(num_tribes):
    traits_list = ["Health Buff", "Damage Buff", "Resourceful", "Aggressive", "Nomadic", "Cooperative", "Cautious"]
    tribes = []
    for i in range(num_tribes):
        traits = random.sample(traits_list, 2)
        name = f"Tribe {chr(ord('A') + i)}"
        tribes.append(Tribe(traits, name))
    return tribes

def initialize_tribes(tribes):
    for tribe in tribes:
        print(f"Initialized {tribe.name} with traits {tribe.traits}")
