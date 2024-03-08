# simulate.py

import random
from tribe import create_tribes, initialize_tribes
import time

def simulate(tribes):
    generations = 50  # Number of generations to simulate

    # Initialize tribes
    initialize_tribes(tribes)

    for generation in range(generations):
        print(f"\nGeneration {generation + 1}")

        # Check if there are no tribes left
        if len(tribes) == 0:
            print("No tribes left. Simulation ends.")
            break

        # Check if there is only one surviving tribe
        if len(tribes) == 1:
            print(f"{tribes[0].name} is the last surviving tribe. Simulation continues with the lone tribe.")

        for i, tribe in enumerate(tribes):
            other_tribe = random.choice(tribes[:i] + tribes[i + 1:]) if len(tribes) > 1 else None
            tribe.perform_actions(other_tribe)
            tribe.reproduce()
            tribe.break_off()

        # Interactions between tribes
        for i in range(len(tribes)):
            for j in range(i + 1, len(tribes)):
                tribes[i].interact(tribes[j])

        # Remove tribes with zero population
        eliminated_tribes = [tribe for tribe in tribes if tribe.population <= 0]
        for eliminated_tribe in eliminated_tribes:
            print(f"{eliminated_tribe.name} has been eliminated.")
        tribes = [tribe for tribe in tribes if tribe.population > 0]

        # Check for resource constraints and eliminate less successful tribes
        tribes.sort(key=lambda x: x.resources, reverse=True)
        surviving_tribes = tribes[:2]

        # Display tribe information
        for tribe in surviving_tribes:
            print(f"\n{tribe.name} with Traits {tribe.traits}:")
            print(f"Population: {tribe.population}")
            print(f"Resources: {tribe.resources}")
            print(f"Happiness: {tribe.happiness}")

        # Introduce a 1-second delay between turns
        time.sleep(4)

# Create tribes
initial_tribes = create_tribes(4)

# Simulate
simulate(initial_tribes)
