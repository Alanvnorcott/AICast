# tribe.py

import random
from traits import TraitsHandler

class Tribe:
    def __init__(self, traits, name):
        self.traits = traits
        self.name = name
        self.population = 1500
        self.resources = 3000
        self.turns_without_enough_resources = 0
        self.happiness = 100
        self.health_multiplier = 1.0
        self.damage_multiplier = 1.0
        self.resource_gain_multiplier = 1.0
        self.attack_multiplier = 1.0
        self.alliance_bonus = 0.0
        self.happiness_multiplier = 1.0
        self.relationship_score = 5
        TraitsHandler.apply_trait_multipliers(self)
        self.season = None

    @staticmethod
    def create_and_initialize_tribes(num_tribes):
        traits_list = ["Health Buff", "Damage Buff", "Resourceful", "Aggressive", "Nomadic", "Cooperative", "Cautious"]
        tribes = []
        for i in range(num_tribes):
            traits = random.sample(traits_list, 2)
            random_letter = random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
            name = f"Tribe {random_letter}"
            tribe = Tribe(traits, name)
            tribes.append(tribe)
        Tribe.initialize_tribes(tribes)
        return tribes

    @staticmethod
    def initialize_tribes(tribes):
        for tribe in tribes:
            print(f"Initialized {tribe.name} with traits {tribe.traits}")

    @staticmethod
    def apply_trait_multipliers(tribe):
        for trait in tribe.traits:
            if trait == "Health Buff":
                tribe.health_multiplier += 0.2
            elif trait == "Damage Buff":
                tribe.damage_multiplier += 0.2
            elif trait == "Resourceful":
                tribe.resource_gain_multiplier += 0.2
            elif trait == "Aggressive":
                tribe.attack_multiplier += 0.2
            elif trait == "Nomadic":
                tribe.resource_gain_multiplier += 0.1
            elif trait == "Cooperative":
                tribe.alliance_bonus += 0.1
            elif trait == "Cautious":
                tribe.happiness_multiplier += 0.1

    def create_sub_tribe(self):
        if self.name.startswith("Sub Tribe") or self.population < 1500:
            return None
        random_letter = random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        sub_tribe_name = f"Sub Tribe {random_letter} ({self.name})"
        sub_tribe_traits = random.sample(self.traits, min(len(self.traits), 2))
        sub_tribe = Tribe(sub_tribe_traits, sub_tribe_name)
        sub_tribe.population = int(self.population / 2)
        TraitsHandler.apply_trait_multipliers(sub_tribe)
        return sub_tribe

    def perform_actions(self, other_tribes, ai_decision):
        attack_weight = ai_decision.get("attack", 0.0)
        collect_weight = ai_decision.get("collect", 0.0)
        trade_weight = ai_decision.get("trade", 0.0)
        conflict_weight = ai_decision.get("conflict", 0.0)
        form_alliance_weight = ai_decision.get("form_alliance", 0.0)
        pass_weight = ai_decision.get("pass", 0.0)
        total_weight = attack_weight + collect_weight + trade_weight + conflict_weight + form_alliance_weight + pass_weight
        if total_weight == 0.0:
            pass_weight = 1.0
        action_choices = ["Attack", "Collect", "Trade", "Conflict", "Form_Alliance", "Pass"]
        action_weights = [attack_weight, collect_weight, trade_weight, conflict_weight, form_alliance_weight,
                          pass_weight]
        action = random.choices(action_choices, weights=action_weights)[0]
        for other_tribe in other_tribes:
            if other_tribe is not None:
                if action == "Attack":
                    self.attack(other_tribe)
                elif action == "Collect":
                    self.collect_resources()
                elif action == "Pass":
                    self.pass_action()

        if self.population > 0 and self.resources > 0:
            consumed_resources = self.population * 0.2
            if self.resources >= int(consumed_resources):
                self.resources -= int(consumed_resources)
                divisor = max(1, self.population * 2)
                happiness_factor = max(0, min(1, int(self.resources / divisor)))
                if self.resources >= self.population / 3:
                    happiness_factor += 0.1
                self.happiness = int(happiness_factor * 100)

    def attack(self, other_tribe):
        attack_strength = 10
        if other_tribe.population > 0:
            initial_loser_population_loss = min(other_tribe.population, int(attack_strength * (other_tribe.population / (self.population + other_tribe.population))))
            initial_loser_population_loss = max(1, initial_loser_population_loss)
            initial_happiness_loss = int((initial_loser_population_loss / other_tribe.population) * 100)
        else:
            initial_loser_population_loss = 0
            initial_happiness_loss = 0
        other_tribe.happiness = max(0, other_tribe.happiness - initial_happiness_loss)
        if "Aggressive" in self.traits:
            attack_strength += 0.2
        if "Damage Buff" in self.traits:
            damage_multiplier = 0.2
            attack_strength *= (1 + damage_multiplier)
            instant_kill_chance = 0.1
            instant_kill_amount = min(other_tribe.population, int(instant_kill_chance * attack_strength))
            if "Health Buff" in self.traits:
                mitigation_amount = min(instant_kill_amount, int(0.2 * instant_kill_amount))
                instant_kill_amount -= mitigation_amount
                print(f"{self.name} with Health Buff trait mitigates {mitigation_amount} of the population loss.")
            other_tribe.population -= instant_kill_amount
            print(f"{self.name} with Damage Buff trait has a chance to instantly kill {instant_kill_amount} of {other_tribe.name}'s population.")
        if other_tribe.population > 0:
            final_loser_population_loss = min(other_tribe.population, int(attack_strength * (other_tribe.population / (self.population + other_tribe.population))))
            final_loser_population_loss = max(1, final_loser_population_loss)
        else:
            final_loser_population_loss = 0
        if "Health Buff" in self.traits:
            mitigation_amount = min(final_loser_population_loss, int(0.2 * final_loser_population_loss))
            final_loser_population_loss -= mitigation_amount
            print(f"{self.name} with Health Buff trait mitigates {mitigation_amount} of the population loss.")
        resource_steal_chance = 0.3
        if random.random() < resource_steal_chance:
            stolen_resources = min(other_tribe.resources, 100)
            other_tribe.resources -= stolen_resources
            self.resources += stolen_resources

            print(f"{self.name} steals {stolen_resources} resources from {other_tribe.name}.")

        # Update loser's population and happiness
        other_tribe.population -= final_loser_population_loss
        final_happiness_loss = int(1.5 * final_loser_population_loss)
        other_tribe.happiness = max(0, other_tribe.happiness - final_happiness_loss)

        # Calculate happiness gain for the attacker
        self_happiness_gain = int(0.5 * final_happiness_loss)  # Example: 50% of happiness loss as gain

        # Update attacker's happiness with a big boost
        self.happiness = min(100, self.happiness + self_happiness_gain + 20)  # Adding a big boost of 20 happiness

        print(
            f"{self.name} attacks {other_tribe.name}. {other_tribe.name} loses {final_loser_population_loss} population and {final_happiness_loss} happiness.")
        print(f"{self.name} gains {self_happiness_gain + 20} happiness as the attacker.")

        # Ensure the winner has at least two survivors
        remaining_population = max(2, self.population - final_loser_population_loss)
        self.population = remaining_population

    def collect_resources(self):
        # Constants for realistic resource gain
        base_resource_gain = int(random.randint(100, int(900 * .4))) # Adjust as needed
        if base_resource_gain >= 500:
            print(f"{self.name} struck gold!")
        resource_multiplier = 1.0  # No trait multiplier by default

        # Apply resourceful and nomadic trait multipliers
        if "Resourceful" in self.traits:
            resource_multiplier += int(0.2)  # Example: 20% increase in resource gain
        if "Nomadic" in self.traits and random.randint(1, 4) == 2:
            resource_multiplier += int(0.1)  # Example: 10% increase in resource gain
            print(f"{self.name} is full of Nomads! Additional resources collected!")

        # Adjust resource gain based on season
        season_multiplier = self.get_season_multiplier()
        resource_multiplier *= season_multiplier

        # Calculate realistic resource gain
        resource_gain = int(base_resource_gain * resource_multiplier)

        self.resources += resource_gain

        # Calculate happiness gained from collecting resources
        happiness_gain = int(
            (resource_gain/10)) # Example: Adjusted for a more realistic happiness gain

        # Update happiness
        self.happiness = max(0, min(100, self.happiness + happiness_gain))

        print(f"{self.name} collects {resource_gain} resources and gains {happiness_gain} happiness.")

    def get_season_multiplier(self):
        # Define season multipliers
        season_multipliers = {
            "Spring": 1.2,
            "Summer": 1.5,
            "Fall": 0.8,
            "Winter": 0.6
        }

        # Randomly select current season or implement logic to determine the season
        self.season = random.choice(list(season_multipliers.keys()))

        return season_multipliers[self.season]

    def pass_action(self):
        # Implement pass action
        pass

    def reproduce(self):
        if self.population > 2 and self.resources > self.population * 3:
            added_individuals = random.randint(1, 2) * int((self.population / 2))
            self.population += added_individuals
        pass

    def break_off(self):

        if self.happiness < random.uniform(0, 50):
            sub_tribe = self.create_sub_tribe()
            if sub_tribe:
                print(f"{self.name} breaks off into a sub-tribe: {sub_tribe.name}.")
                return sub_tribe
            else:
                print(f"{self.name} cannot create a sub-tribe due to low population or existing sub-tribe status.")
        return None

    def interact(self, other_tribe):
        self.update_relationship_score()
        other_tribe.update_relationship_score()
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
        if self.population > 0:
            self.relationship_score += random.randint(-1, 1)
            self.relationship_score = max(1, min(10, self.relationship_score))

    def trade_resources(self, other_tribe):
        # Implement trade action
        trade_amount = 50
        if self.resources >= trade_amount and other_tribe.resources >= trade_amount:
            self.resources -= trade_amount
            other_tribe.resources += trade_amount
            print(f"{self.name} and {other_tribe.name} engage in trade.")

    def conflict(self, other_tribe):
        conflict_strength_self = 20
        conflict_strength_other = 20
        if "Aggressive" in self.traits:
            conflict_strength_self += 0.2  # Example: 20% increase in conflict strength
        if "Damage Buff" in self.traits:
            damage_multiplier = 0.2  # Example: 20% increase in damage
            conflict_strength_self *= (1 + damage_multiplier)
            instant_kill_chance = 0.1  # Example: 10% chance
            instant_kill_amount = min(other_tribe.population, int(instant_kill_chance * conflict_strength_self))
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
                self_happiness_loss = int((self_population_loss / self.population) * 100) if self.population > 0 else 0
                other_happiness_loss = int(
                    (other_population_loss / other_tribe.population) * 100) if other_tribe.population > 0 else 0
                self.happiness = max(0, self.happiness - self_happiness_loss)
                other_tribe.happiness = max(0, other_tribe.happiness - other_happiness_loss)
                self_happiness_gain = int(0.2 * self_happiness_loss)
                other_happiness_gain = int(0.2 * other_happiness_loss)
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


        happiness_gain_self = int(10 * alliance_bonus)
        happiness_gain_other = int(10 * alliance_bonus)


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
