# simulate.py

import random
from tribe import Tribe
import time
import numpy as np
from tf_agents.trajectories import time_step as ts
import tensorflow as tf
def simulate(tribes, trained_agent):
    generations = 50  # Number of generations to simulate

    # Initialize tribes
    Tribe.initialize_tribes(tribes)

    for generation in range(generations):
        print(f"\nGeneration {generation + 1}")

        # Main tribes perform actions, reproduce, and break off
        for i, tribe in enumerate(tribes):
            other_tribe = random.choice(tribes[:i] + tribes[i + 1:]) if len(tribes) > 1 else None
            ai_decision = tribe.determine_ai_decision(trained_agent)

            # Ensure the observation structure matches the time_step_spec
            observation = np.array([tribe.population, tribe.resources, tribe.happiness], dtype=np.float32)
            time_step = ts.restart(observation)

            # Ensure the time_step structure matches the time_step_spec
            time_step_spec = trained_agent.collect_data_spec.time_step_spec
            time_step = time_step._replace(
                step_type=tf.constant(ts.StepType.MID, dtype=tf.int32),
                reward=tf.constant(0.0, dtype=tf.float32),
                discount=tf.constant(1.0, dtype=tf.float32),
                observation=tf.constant(observation, dtype=tf.float32)
            )

            # Adjusting the structure of the time_step observation
            observation_spec_shape = time_step_spec.observation.shape
            if observation.shape != observation_spec_shape:
                observation = np.array([tribe.population, tribe.resources, tribe.happiness], dtype=np.float32)
                observation = observation.reshape(observation_spec_shape)

            time_step = time_step._replace(observation=tf.constant(observation, dtype=tf.float32))

            tribe.perform_actions(other_tribe, ai_decision, time_step)

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
        for tribe in surviving_tribes:
            print(f"\n{tribe.name} with Traits {tribe.traits}:")
            print(f"Population: {tribe.population}")
            print(f"Resources: {tribe.resources}")
            print(f"Happiness: {tribe.happiness}")

        # Check if there are no tribes left
        if len(tribes) == 0:
            print("No tribes left. Simulation ends.")
            break

        # Check if there is only one surviving tribe
        if len(tribes) == 1:
            print(f"{tribes[0].name} is the last surviving tribe. Simulation ends.")
            break

        # Introduce a 1-second delay between turns
        time.sleep(4)

# Create and initialize tribes using the shared function
initial_tribes = Tribe.create_and_initialize_tribes(4)
