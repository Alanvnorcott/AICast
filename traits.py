class TraitsHandler:
    @staticmethod
    def apply_trait_multipliers(tribe):
        # Assign multipliers based on traits
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
