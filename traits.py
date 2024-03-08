class TraitsHandler:
    @staticmethod
    def apply_trait_multipliers(tribe):
        # Assign multipliers based on traits
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
