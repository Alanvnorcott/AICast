#simulate.py
import random
from tribe import Tribe
import tensorflow as tf
import numpy as np

current_state_dict = {'population': 1500, 'resources': 3000, 'happiness': 100}
current_state_data = [current_state_dict['population'], current_state_dict['resources'], current_state_dict['happiness']]
current_state_np = np.array(current_state_data)
current_state_tensor = tf.convert_to_tensor(current_state_np, dtype=tf.float32)

def display_tribe_info(tribes):
    print("\nCurrent Tribe Information:")
    for tribe in tribes:
        print(f"\n{tribe.name} with Traits {tribe.traits}:")
        print(f"Population: {tribe.population}")
        print(f"Resources: {tribe.resources}")
        print(f"Happiness: {tribe.happiness}")

def simulate(tribes, agent):
    generations = 50  

    Tribe.initialize_tribes(tribes)

    for generation in range(generations):
        print(f"\nGeneration {generation + 1}")

        for i, tribe in enumerate(tribes):
            other_tribe = random.choice(tribes[:i] + tribes[i + 1:]) if len(tribes) > 1 else None
            current_state = {
                'population': tribe.population,
                'resources': tribe.resources,
                'happiness': tribe.happiness,
            }
            current_state = tf.convert_to_tensor(current_state, dtype=tf.float32)

            action_step = agent.policy.action(current_state)
            ai_decision = action_step.action.numpy()[0]

            tribe.perform_actions(other_tribe, ai_decision)
            tribe.reproduce()

        for i in range(len(tribes)):
            for j in range(i + 1, len(tribes)):
                tribes[i].interact(tribes[j])

        eliminated_tribes = [tribe for tribe in tribes if tribe.population <= 0]
        for eliminated_tribe in eliminated_tribes:
            print(f"{eliminated_tribe.name} has been eliminated.")
        tribes = [tribe for tribe in tribes if tribe.population > 0]

        tribes.sort(key=lambda x: x.resources, reverse=True)
        surviving_tribes = tribes[:2]

        display_tribe_info(surviving_tribes)

        if len(tribes) == 0:
            print("No tribes left. Simulation ends.")
            break

        if len(tribes) == 1:
            print(f"{tribes[0].name} is the last surviving tribe. Simulation ends.")
            break