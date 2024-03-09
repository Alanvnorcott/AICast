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

# Create the Tribe environment
num_actions = 3
num_features = 3
initial_tribe = Tribe.create_and_initialize_tribes(4)
environment = TribeEnvironment(tribes=initial_tribe, num_actions=num_actions, num_features=num_features)

# Wrap the environment in a TF PyEnvironment
tf_environment = tf_py_environment.TFPyEnvironment(environment)

# Define the Q-network
fc_layer_params = (100,)
q_net = q_network.QNetwork(
    tf_environment.observation_spec(),
    tf_environment.action_spec(),
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

# Define the training loop
num_iterations = 10000
for _ in range(num_iterations):
    # Collect a few steps using the random policy
    collect_driver.run()

    # Sample a batch of data from the buffer
    experience, _ = next(iterator)

    # Train the agent
    train_loss = train_step()

    # Print training information
    if agent.train_step_counter.numpy() % 1000 == 0:
        print(f"Step: {agent.train_step_counter.numpy()}, Loss: {train_loss.loss.numpy()}")

# Save the trained model
saved_model_path = './AICast/models'
if not os.path.exists(saved_model_path):
    os.makedirs(saved_model_path)
agent.save_policy(saved_model_path)

