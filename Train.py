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
import matplotlib.pyplot as plt
import signal
import numpy as np
import random
from traits import TraitsHandler

# Initialize lists to store training information for plotting
steps = []
losses = []
# Define the actions mapping
actions_mapping = {0: "attack", 1: "collect", 2: "pass"}
# Create the Tribe environment
num_actions = 3
num_features = 3
initial_tribe = Tribe.create_and_initialize_tribes(4)
environment = TribeEnvironment(tribes=initial_tribe, num_actions=num_actions, num_features=num_features)

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
    num_steps=1
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

# ...

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

# Function to display tribe information during training
def display_tribe_info(tribes):
    print("\nCurrent Tribe Information:")
    for tribe in tribes:
        print(f"\n{tribe.name} with Traits {tribe.traits}:")
        print(f"Population: {tribe.population}")
        print(f"Resources: {tribe.resources}")
        print(f"Happiness: {tribe.happiness}")

# Register the signal handler for KeyboardInterrupt
signal.signal(signal.SIGINT, lambda signal, frame: save_and_exit(signal, frame, saved_model_path))

try:
    for iteration in range(num_iterations):
        # Display tribe information before collecting steps
        display_tribe_info(tf_environment.pyenv.envs[0]._tribes)

        # Collect a few steps using the random policy
        collect_driver.run()

        # Sample a batch of data from the buffer only if it has enough items
        if replay_buffer.num_frames().numpy() >= sample_batch_size:
            experience, _ = next(iterator)

            # Print actions taken during training
            print("Actions taken during training:")
            for step in experience.action:
                tribe_0_action = actions_mapping.get(int(step[0]), "unknown")
                tribe_1_action = actions_mapping.get(int(step[1]), "unknown")
                print(f"Tribe A: {tribe_0_action}, Tribe B: {tribe_1_action}")


            # Train the agent
            train_loss = train_step()

            # Print training information
            if agent.train_step_counter.numpy() % 1000 == 0:
                print(f"Step: {agent.train_step_counter.numpy()}, Loss: {train_loss.loss.numpy()}")

                # Store training information for plotting
                steps.append(agent.train_step_counter.numpy())
                losses.append(train_loss.loss.numpy())

                # Display tribe information after training
                display_tribe_info(tf_environment.pyenv.envs[0]._tribes)

                # Plot the training loss
                plt.plot(steps, losses, label='Training Loss')
                plt.xlabel('Training Steps')
                plt.ylabel('Loss')
                plt.legend()
                plt.show()

            if all(tribe.population == 0 for tribe in tf_environment.pyenv.envs[0]._tribes):
                print("Training ended as there are no tribes left.")
                break

except KeyboardInterrupt:
    # Save the model when interrupted
    save_and_exit(None, None, saved_model_path)

# Save the trained model using tf.saved_model.save
if not os.path.exists(saved_model_path):
    os.makedirs(saved_model_path)

# Save the model
tf.saved_model.save(agent.policy, saved_model_path)

