import tensorflow as tf
from tf_agents.networks import q_network
from tf_agents.agents.dqn import dqn_agent
from tf_agents.environments import tf_py_environment
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.drivers import dynamic_step_driver
from TribeEnvironment import TribeEnvironment
from tf_agents.utils import common
import os
import signal
import matplotlib.pyplot as plt
import numpy as np
import random
from traits import TraitsHandler
from tribe import Tribe

steps = []
losses = []
actions_mapping = {0: "attack", 1: "collect", 2: "trade", 3: "conflict", 4: "form_alliance", 5: "pass"}
num_tribes = 4
num_actions = 6
num_features = 3
initial_tribe = Tribe.create_and_initialize_tribes(num_tribes)
environment = TribeEnvironment(num_tribes=num_tribes, num_actions=num_actions, num_features=num_features)
tf_environment = tf_py_environment.TFPyEnvironment(environment)
obs_spec = tf_environment.observation_spec()
obs_spec = tf.TensorSpec(shape=obs_spec.shape, dtype=tf.float32)

fc_layer_params = (100,)
q_net = q_network.QNetwork(
    input_tensor_spec=obs_spec,
    action_spec=tf_environment.action_spec(),
    fc_layer_params=fc_layer_params
)

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

replay_buffer_capacity = 10000
replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=tf_environment.batch_size,
    max_length=replay_buffer_capacity
)

collect_driver = dynamic_step_driver.DynamicStepDriver(
    tf_environment,
    agent.collect_policy,
    observers=[replay_buffer.add_batch],
    num_steps=1
)

dataset = replay_buffer.as_dataset(
    sample_batch_size=64,
    num_steps=2,
    num_parallel_calls=3
).prefetch(3)

iterator = iter(dataset)

def train_step():
    trajectories, _ = next(iterator)
    return agent.train(experience=trajectories)

num_iterations = 10000
sample_batch_size = 64
saved_model_path = 'C:\\Users\\Alan\\PycharmProjects\\AICast\\models'

def save_and_exit(signal, frame, saved_model_path):
    print("Training interrupted. Saving the model...")
    tf.saved_model.save(agent.policy, saved_model_path)
    print("Model saved. Exiting.")
    exit()

def display_tribe_info(tribes):
    print("\nCurrent Tribe Information:")
    for tribe in tribes:
        print(f"\n{tribe.name} with Traits {tribe.traits}:")
        print(f"Population: {tribe.population}")
        print(f"Resources: {tribe.resources}")
        print(f"Happiness: {tribe.happiness}")

signal.signal(signal.SIGINT, lambda signal, frame: save_and_exit(signal, frame, saved_model_path))

try:
    for iteration in range(num_iterations):
        display_tribe_info(tf_environment.pyenv.envs[0]._tribes)
        collect_driver.run()
        if replay_buffer.num_frames().numpy() >= sample_batch_size:
            experience, _ = next(iterator)
            train_loss = train_step()
            if agent.train_step_counter.numpy() % 1000 == 0:
                print(f"Step: {agent.train_step_counter.numpy()}, Loss: {train_loss.loss.numpy()}")
                steps.append(agent.train_step_counter.numpy())
                losses.append(train_loss.loss.numpy())
                display_tribe_info(tf_environment.pyenv.envs[0]._tribes)
                plt.plot(steps, losses, label='Training Loss')
                plt.xlabel('Training Steps')
                plt.ylabel('Loss')
                plt.legend()
                plt.show()
            if all(tribe.population == 0 for tribe in tf_environment.pyenv.envs[0]._tribes):
                print("Training ended as there are no tribes left.")
                break

except KeyboardInterrupt:
    save_and_exit(None, None, saved_model_path)

if not os.path.exists(saved_model_path):
    os.makedirs(saved_model_path)

tf.saved_model.save(agent.policy, saved_model_path)
