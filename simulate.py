# simulate.py

import random
from tribe import Tribe
import time
import tensorflow as tf


def display_tribe_info(tribes):
    print("\nCurrent Tribe Information:")
    for tribe in tribes:
        print(f"\n{tribe.name} with Traits {tribe.traits}:")
        print(f"Population: {tribe.population}")
        print(f"Resources: {tribe.resources}")
        print(f"Happiness: {tribe.happiness}")

def simulate(tribes, agent):
    generations = 50  # Number of generations to simulate

    # Initialize tribes
    Tribe.initialize_tribes(tribes)

    for generation in range(generations):
        print(f"\nGeneration {generation + 1}")

        # Main tribes perform actions, reproduce, and break off
        for i, tribe in enumerate(tribes):
            other_tribe = random.choice(tribes[:i] + tribes[i + 1:]) if len(tribes) > 1 else None

            # Get the current state of the tribe (you need to implement this)
            current_state = {
                'population': tribe.population,
                'resources': tribe.resources,
                'happiness': tribe.happiness,
            }

            # Convert the current state to a format suitable for the agent
            current_state = tf.convert_to_tensor(current_state, dtype=tf.float32)

            # Use the DQN agent's policy to select an action
            action_step = agent.policy.action(current_state)

            # Extract the chosen action from the action step
            ai_decision = action_step.action.numpy()[0]

            # Perform actions using the determined AI decision
            tribe.perform_actions(other_tribe, ai_decision)
            tribe.reproduce()

        # Interactions between all tribes
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
        display_tribe_info(surviving_tribes)

        # Check if there are no tribes left
        if len(tribes) == 0:
            print("No tribes left. Simulation ends.")
            break

        # Check if there is only one surviving tribe
        if len(tribes) == 1:
            print(f"{tribes[0].name} is the last surviving tribe. Simulation ends.")
            break

        # Introduce a 1-second delay between turns


