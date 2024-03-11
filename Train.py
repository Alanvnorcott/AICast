# Train.py

import tensorflow as tf
from tf_agents.networks import q_network
from tf_agents.agents.dqn import dqn_agent
from tf_agents.environments import tf_py_environment
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.drivers import dynamic_step_driver
from TribeEnvironment import TribeEnvironment
from tf_agents.utils import common
import os
from tribe import Tribe
import matplotlib.pyplot as plt
import signal
import numpy as np
from simulate import simulate

# Initialize lists to store training information for plotting
steps = []
losses = []
# Define the actions mapping
actions_mapping = {0: "attack", 1: "collect", 2: "pass"}
# Create the Tribe environment
num_actions = 3
num_features = 3
initial_tribes = Tribe.create_and_initialize_tribes(4)
environment = TribeEnvironment(tribes=initial_tribes, num_actions=num_actions, num_features=num_features)
# Initialize a counter for actions
action_counter = {action: 0 for action in actions_mapping.values()}

# Wrap the environment in a TF PyEnvironment
tf_environment = tf_py_environment.TFPyEnvironment(environment)

# Ensure observation_spec dtype matches Q-network input dtype
obs_spec = tf_environment.observation_spec()
obs_spec = tf.TensorSpec(shape=obs_spec.shape, dtype=tf.float32)  # Change dtype to tf.float32


# Define the Q-network with fc_layer_params
fc_layer_params = (100,)
q_net = q_network.QNetwork(
    input_tensor_spec=obs_spec,  # Pass obs_spec here
    action_spec=tf_environment.action_spec(),
    fc_layer_params=fc_layer_params
)



# Define the DQN agent
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-3)
train_step_counter = tf.Variable(0)
agent = dqn_agent.DqnAgent(
    tf_environment.time_step_spec(),
    tf_environment.action_spec(),
    q_network=q_net,
    optimizer=optimizer,
    td_errors_loss_fn=common.element_wise_squared_loss,
    train_step_counter=train_step_counter
)
agent.initialize()

# Define the replay buffer
replay_buffer_capacity = 10000
replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=tf_environment.batch_size,
    max_length=replay_buffer_capacity
)

# Define the data collection
collect_driver = dynamic_step_driver.DynamicStepDriver(
    tf_environment,
    agent.collect_policy,
    observers=[replay_buffer.add_batch],
    num_steps=len(initial_tribes)  # Use the number of tribes as the number of steps
)

# Define the dataset
dataset = replay_buffer.as_dataset(
    sample_batch_size=64,
    num_steps=2,
    num_parallel_calls=3
).prefetch(3)

# Initialize the iterator
iterator = iter(dataset)

# Define the training step function
def train_step():
    trajectories, _ = next(iterator)
    return agent.train(experience=trajectories)


# Define the training loop
num_iterations = 10000
sample_batch_size = 64  # Define the sample batch size

# Set the initial saved_model_path
saved_model_path = 'C:\\Users\\Alan\\PycharmProjects\\AICast\\models'

# Function to handle KeyboardInterrupt and save the model
def save_and_exit(signal, frame, saved_model_path):
    print("Training interrupted. Saving the model...")
    tf.saved_model.save(agent.policy, saved_model_path)
    print("Model saved. Exiting.")
    exit()

# Register the signal handler for KeyboardInterrupt
signal.signal(signal.SIGINT, lambda signal, frame: save_and_exit(signal, frame, saved_model_path))

# Function to clip population and happiness values to valid range
def clip_values(value, min_value, max_value):
    return max(min(value, max_value), min_value)


# Function to check if any tribe's population is zero
def all_tribes_depleted(tribes):
    return all(tribe.population <= 0 for tribe in tribes)


try:
    for iteration in range(num_iterations):
        simulate(initial_tribes, trained_agent=agent)

        ai_decision = agent.policy.action(observations=initial_tribes.get_observation())
        collect_driver.run()

        # Sample a batch of data from the buffer only if it has enough items
        if replay_buffer.num_frames().numpy() >= sample_batch_size:
            experience, _ = next(iterator)

            # Train the agent
            train_loss = train_step()

            # Print training information
            if agent.train_step_counter.numpy() % 100 == 0:
                print(f"Step: {agent.train_step_counter.numpy()}, Loss: {train_loss.loss.numpy()}")

                # Store training information for plotting
                steps.append(agent.train_step_counter.numpy())
                losses.append(train_loss.loss.numpy())

                # Print actions taken during training
                print("Actions taken:")
                for i, step in enumerate(experience.action):
                    for j, tribe_action in enumerate(step):
                        tribe_name = f"Tribe {chr(ord('A') + j)}"
                        tribe_action_name = actions_mapping.get(int(tribe_action), "unknown")
                        action_counter[tribe_action_name] += 1

                        # Ensure 'j' is within the bounds of the initial_tribe list
                        if j < len(initial_tribes):
                            tribe = initial_tribes[j]
                            print(f"{tribe_name}: {tribe_action_name}")
                            print(f"{tribe.name} - Stats:")



                            # Update the tribe's action based on the AI decision
                            if tribe_action_name == "attack":
                                # Example: Use the AI agent's decision for attack
                                if ai_decision["attack"] > 0.5:  # Replace 0.5 with your desired threshold
                                    tribe.attack(tribe.other_tribe)  # Pass the appropriate arguments
                                else:
                                    tribe.pass_action()
                            elif tribe_action_name == "collect":
                                # Example: Use the AI agent's decision for collect
                                if ai_decision["collect"] > 0.5:  # Replace 0.5 with your desired threshold
                                    tribe.collect_resources()
                                else:
                                    tribe.pass_action()
                            elif tribe_action_name == "pass":
                                # Example: Use the AI agent's decision for pass
                                if ai_decision["pass"] > 0.5:  # Replace 0.5 with your desired threshold
                                    tribe.pass_action()

                            # Update the tribe's state and attributes
                            tribe.population = clip_values(tribe.population, 0, np.inf)
                            tribe.resources = clip_values(tribe.resources, 0, np.inf)
                            tribe.happiness = clip_values(tribe.happiness, 0, 100)

                            print(f"Population: {tribe.population}")
                            print(f"Resources: {tribe.resources}")
                            print(f"Happiness: {tribe.happiness}")
                            print()
                        else:
                            print(f"Invalid index 'j' for initial_tribe list.")

                # Print tribe statistics after each iteration
                print("Tribe Statistics:")
                for tribe in initial_tribes:
                    print(f"{tribe.name}:")
                    print(f"Population: {tribe.population}")
                    print(f"Resources: {tribe.resources}")
                    print(f"Happiness: {tribe.happiness}")
                    print()

                # Plot the training loss
                plt.plot(steps, losses, label='Training Loss')
                plt.xlabel('Training Steps')
                plt.ylabel('Loss')
                plt.legend()
                plt.show()

                # Check if all tribes are depleted
                if all_tribes_depleted(initial_tribes):
                    print("All tribes are depleted. Training complete.")
                    break  # Terminate training loop


except KeyboardInterrupt:
    # Save the model when interrupted
    save_and_exit(None, None, saved_model_path)

most_taken_action = max(action_counter, key=action_counter.get)
print(f"Most taken action during training: {most_taken_action} (count: {action_counter[most_taken_action]})")
# Save the trained model using tf.saved_model.save
if not os.path.exists(saved_model_path):
    os.makedirs(saved_model_path)

# Save the model
tf.saved_model.save(agent.policy, saved_model_path)


