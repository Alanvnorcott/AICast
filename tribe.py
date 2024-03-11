# tribe.py

import random
from traits import TraitsHandler

class Tribe:
    def __init__(self, traits, name):
        self.traits = traits
        self.name = name
        self.population = 1000  # Initial population size
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

    @staticmethod
    def create_and_initialize_tribes(num_tribes):
        traits_list = ["Health Buff", "Damage Buff", "Resourceful", "Aggressive", "Nomadic", "Cooperative", "Cautious"]
        tribes = []
        for i in range(num_tribes):
            traits = random.sample(traits_list, 2)
            name = f"Tribe {chr(ord('A') + i)}"
            tribe = Tribe(traits, name)
            tribes.append(tribe)
        Tribe.initialize_tribes(tribes)  # Call the initialize_tribes function within the Tribe class
        return tribes

    @staticmethod
    def initialize_tribes(tribes):
        for tribe in tribes:
            print(f"Initialized {tribe.name} with traits {tribe.traits}")
    @staticmethod
    def apply_trait_multipliers(tribe):
        for trait in tribe.traits:
            if trait == "Health Buff":
                tribe.health_multiplier += 0.2  # Example: 20% increase in health
            elif trait == "Damage Buff":
                tribe.damage_multiplier += 0.2  # Example: 20% increase in damage
            elif trait == "Resourceful":
                tribe.resource_gain_multiplier += 0.2  # Example: 20% increase in resource gain
            elif trait == "Aggressive":
                tribe.attack_multiplier += 0.2  # Example: 20% increase in attack strength
            elif trait == "Nomadic":
                tribe.resource_gain_multiplier += 0.1  # Example: 10% increase in resource gain
            elif trait == "Cooperative":
                tribe.alliance_bonus += 0.1  # Example: 10% increase in alliance bonus
            elif trait == "Cautious":
                tribe.happiness_multiplier += 0.1  # Example: 10% increase in happiness

    def create_sub_tribe(self):
        if self.name.startswith("Sub Tribe") or self.population < 1500:
            # Sub-tribes are not possible if the parent tribe is already a sub-tribe or if the population is less than 200
            return None

        random_letter = random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        sub_tribe_name = f"Sub Tribe {random_letter} ({self.name})"
        sub_tribe_traits = random.sample(self.traits,
                                         min(len(self.traits), 2))  # Inherit up to 2 traits from the parent tribe
        sub_tribe = Tribe(sub_tribe_traits, sub_tribe_name)

        # Adjust sub-tribe population based on parent tribe's population
        sub_tribe.population = int(self.population / 2)

        # Apply trait multipliers to the sub-tribe
        TraitsHandler.apply_trait_multipliers(sub_tribe)

        return sub_tribe

    def perform_actions(self, other_tribe, ai_decision):
        # Check if the "attack" key is present in ai_decision
        attack_weight = ai_decision.get("attack", 0.0)
        collect_weight = ai_decision.get("collect", 0.0)
        pass_weight = ai_decision.get("pass", 0.0)

        action = random.choices(["Attack", "Collect", "Pass"], weights=[attack_weight, collect_weight, pass_weight])[0]

        if other_tribe is not None:  # Add a check for None
            if action == "Attack":
                self.attack(other_tribe)
            elif action == "Collect":
                self.collect_resources()
            elif action == "Pass":
                self.pass_action()

        # Update happiness based on various factors
        if self.population > 0 and self.resources > 0:
            consumed_resources = self.population * 9
            self.resources -= consumed_resources
            self.happiness = max(0, min(100, int(self.resources) / (int(self.population) * 3) * 100))

        # Check resource consumption and population control
        if self.resources < 0:
            self.resources = 0  # Ensure resources don't go below 0
            self.population = max(0, int(self.population * 0.5))  # Exponential population decrease

        # Additional check for resource consumption and population control
        if self.resources < self.population * 3:
            self.turns_without_enough_resources += 1
        else:
            self.turns_without_enough_resources = 0

        if self.turns_without_enough_resources >= 2:
            self.population = max(0, self.population - 10)  # Perish 10 individuals if not enough resources for 2 turns

    def attack(self, other_tribe):
        # Implement attack action
        attack_strength = 10  # Adjust based on traits

        # Calculate initial loser's population loss
        if other_tribe.population > 0:
            initial_loser_population_loss = min(other_tribe.population, int(attack_strength * (
                    other_tribe.population / (self.population + other_tribe.population))))
            initial_loser_population_loss = max(1,
                                                initial_loser_population_loss)  # Ensure at least one individual is lost

            initial_happiness_loss = int((initial_loser_population_loss / other_tribe.population) * 100)
        else:
            initial_loser_population_loss = 0
            initial_happiness_loss = 0

        # Update loser's happiness
        other_tribe.happiness = max(0, other_tribe.happiness - initial_happiness_loss)

        # Apply Aggressive trait multiplier
        if "Aggressive" in self.traits:
            attack_strength += 0.2  # Example: 20% increase in attack strength

        # Apply Damage Buff trait multiplier
        if "Damage Buff" in self.traits:
            damage_multiplier = 0.2  # Example: 20% increase in damage
            attack_strength *= (1 + damage_multiplier)

            # Chance to instantly kill a set number of the other tribe's population
            instant_kill_chance = 0.1  # Example: 10% chance
            instant_kill_amount = min(other_tribe.population, int(instant_kill_chance * attack_strength))

            # Apply Health Buff trait to mitigate population loss
            if "Health Buff" in self.traits:
                mitigation_amount = min(instant_kill_amount, int(0.2 * instant_kill_amount))  # Example: 20% mitigation
                instant_kill_amount -= mitigation_amount
                print(
                    f"{self.name} with Health Buff trait mitigates {mitigation_amount} of the population loss.")

            other_tribe.population -= instant_kill_amount
            print(
                f"{self.name} with Damage Buff trait has a chance to instantly kill {instant_kill_amount} of {other_tribe.name}'s population.")

        # Recalculate loser's population loss after accounting for instant kills and mitigation
        if other_tribe.population > 0:  # Add this check
            final_loser_population_loss = min(other_tribe.population, int(attack_strength * (
                    other_tribe.population / (self.population + other_tribe.population))))
            final_loser_population_loss = max(1, final_loser_population_loss)  # Ensure at least one individual is lost
        else:
            final_loser_population_loss = 0

        # Mitigate final population loss with Health Buff trait
        if "Health Buff" in self.traits:
            mitigation_amount = min(final_loser_population_loss,
                                    int(0.2 * final_loser_population_loss))  # Example: 20% mitigation
            final_loser_population_loss -= mitigation_amount
            print(
                f"{self.name} with Health Buff trait mitigates {mitigation_amount} of the population loss.")

        # Chance for the attacker to take resources from the pool of the attacked tribe
        resource_steal_chance = 0.3  # Example: 30% chance
        if random.random() < resource_steal_chance:
            stolen_resources = min(other_tribe.resources, 100)  # Example: Steal up to 100 resources
            other_tribe.resources -= stolen_resources
            self.resources += stolen_resources
            print(f"{self.name} steals {stolen_resources} resources from {other_tribe.name}.")

        # Update loser's population and happiness
        other_tribe.population -= final_loser_population_loss
        final_happiness_loss = int(
            (final_loser_population_loss / other_tribe.population) * 100) if other_tribe.population > 0 else 0
        other_tribe.happiness = max(0, other_tribe.happiness - final_happiness_loss)

        # Calculate happiness gain for the attacker
        self_happiness_gain = int(0.2 * final_happiness_loss)  # Example: 20% of happiness loss as gain

        # Update attacker's happiness
        self.happiness = max(0, min(100, self.happiness + self_happiness_gain))

        print(
            f"{self.name} attacks {other_tribe.name}. {other_tribe.name} loses {final_loser_population_loss} population and {final_happiness_loss} happiness.")
        print(f"{self.name} gains {self_happiness_gain} happiness as the attacker.")

        # Ensure the winner has at least two survivors
        remaining_population = max(2, self.population - final_loser_population_loss)
        self.population = remaining_population

    def collect_resources(self):
        # Implement collect action
        base_resource_gain = 10  # Base resource gain, adjust as needed
        resource_gain = base_resource_gain * max(1, int((self.population / 3)))  # Ensure a minimum of 1 resource gain

        # Apply resourceful and nomadic trait multipliers
        if "Resourceful" in self.traits:
            resource_gain *= int(1.2)  # Example: 20% increase in resource gain
        if "Nomadic" in self.traits:
            if random.randint(1, 4) == 2:
                resource_gain *= int(1.1)  # Example: 10% increase in resource gain
                print(f"{self.name} is full of Nomads! {resource_gain} resources collected!")

        self.resources += resource_gain

        # Calculate happiness gained from collecting resources
        happiness_gain = int(5 * (resource_gain / base_resource_gain))  # Example: 5% of resource gain as happiness

        # Update happiness

        self.happiness = max(0, min(100, self.happiness + happiness_gain))

        print(f"{self.name} collects {resource_gain} resources and gains {happiness_gain} happiness.")

    def pass_action(self):
        # Implement pass action
        pass

    def reproduce(self):
        if self.population > 2 and self.resources > self.population * 3:
            added_individuals = random.randint(1, 2) * int((self.population / 2))
            self.population += added_individuals
        pass

    def break_off(self):
        # Implement mechanism for tribes to break off into sub-tribes based on happiness
        if self.happiness < random.uniform(0, 50):
            sub_tribe = self.create_sub_tribe()
            if sub_tribe:
                print(f"{self.name} breaks off into a sub-tribe: {sub_tribe.name}.")
                return sub_tribe
            else:
                print(f"{self.name} cannot create a sub-tribe due to low population or existing sub-tribe status.")
        return None

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

        # Apply Aggressive trait multiplier
        if "Aggressive" in self.traits:
            conflict_strength_self += 0.2  # Example: 20% increase in conflict strength

        # Apply Damage Buff trait multiplier
        if "Damage Buff" in self.traits:
            damage_multiplier = 0.2  # Example: 20% increase in damage
            conflict_strength_self *= (1 + damage_multiplier)

            # Chance to instantly kill a set number of the other tribe's population
            instant_kill_chance = 0.1  # Example: 10% chance
            instant_kill_amount = min(other_tribe.population, int(instant_kill_chance * conflict_strength_self))

            # Apply Health Buff trait to mitigate population loss
            if "Health Buff" in self.traits:
                mitigation_amount = min(instant_kill_amount, int(0.2 * instant_kill_amount))  # Example: 20% mitigation
                instant_kill_amount -= mitigation_amount
                print(f"{self.name} with Health Buff trait mitigates {mitigation_amount} of the population loss.")

            total_population = self.population + other_tribe.population

            if total_population > 0:
                self_population_loss = min(self.population,
                                           int(conflict_strength_other * (self.population / total_population)))
                other_population_loss = min(other_tribe.population,
                                            int(conflict_strength_self * (other_tribe.population / total_population)))

                # Apply Health Buff trait to mitigate population loss
                if "Health Buff" in self.traits:
                    self_mitigation_amount = min(self_population_loss, int(0.2 * self_population_loss))
                    other_mitigation_amount = min(other_population_loss, int(0.2 * other_population_loss))

                    self_population_loss -= self_mitigation_amount
                    other_population_loss -= other_mitigation_amount

                    print(
                        f"{self.name} with Health Buff trait mitigates {self_mitigation_amount} of the population loss.")
                    print(
                        f"{other_tribe.name} with Health Buff trait mitigates {other_mitigation_amount} of the population loss.")

                self.population -= self_population_loss
                other_tribe.population -= other_population_loss

                # Calculate happiness loss based on the percentage of population lost
                self_happiness_loss = int((self_population_loss / self.population) * 100) if self.population > 0 else 0
                other_happiness_loss = int(
                    (other_population_loss / other_tribe.population) * 100) if other_tribe.population > 0 else 0

                # Update losers' happiness
                self.happiness = max(0, self.happiness - self_happiness_loss)
                other_tribe.happiness = max(0, other_tribe.happiness - other_happiness_loss)

                # Calculate happiness gain for the winners
                self_happiness_gain = int(0.2 * self_happiness_loss)  # Example: 20% of happiness loss as gain
                other_happiness_gain = int(0.2 * other_happiness_loss)  # Example: 20% of happiness loss as gain

                # Update winners' happiness
                self.happiness = max(0, min(100, self.happiness + self_happiness_gain))
                other_tribe.happiness = max(0, min(100, other_tribe.happiness + other_happiness_gain))

                print(f"{self.name} and {other_tribe.name} engage in conflict.")
                print(f"{self.name} loses {self_population_loss} population and {self_happiness_loss} happiness.")
                print(
                    f"{other_tribe.name} loses {other_population_loss} population and {other_happiness_loss} happiness.")
                print(f"{self.name} gains {self_happiness_gain} happiness as a winner.")
                print(f"{other_tribe.name} gains {other_happiness_gain} happiness as a winner.")
            else:
                print(f"{self.name} and {other_tribe.name} can't engage in conflict due to zero total population.")

    def form_alliance(self, other_tribe):
        # Implement alliance action
        alliance_bonus = 0.2  # Adjust based on traits
        self.resources += int(other_tribe.resources * alliance_bonus)
        other_tribe.resources += int(self.resources * alliance_bonus)

        # Calculate happiness gained from forming the alliance
        happiness_gain_self = int(10 * alliance_bonus)  # Example: 10% of resources gained as happiness
        happiness_gain_other = int(10 * alliance_bonus)  # Example: 10% of resources gained as happiness

        # Update happiness for both tribes
        self.happiness += happiness_gain_self
        other_tribe.happiness += happiness_gain_other

        print(
            f"{self.name} and {other_tribe.name} form an alliance. {self.name} gains {happiness_gain_self} happiness, and {other_tribe.name} gains {happiness_gain_other} happiness.")


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
