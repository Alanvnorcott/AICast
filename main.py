# main.py

from tribe import create_tribes, initialize_tribes
from simulate import simulate
import Train
from Train import agent, save_and_exit, saved_model_path
import matplotlib.pyplot as plt
import tensorflow as tf

action_counts = {"attack": 0, "collect": 0, "pass": 0}

def convert_tribe_state_to_tensor(tribe_state):
    # Convert dictionary values to a list or numpy array
    tensor_data = [tribe_state['population'], tribe_state['resources'], tribe_state['happiness']]
    # Convert list to TensorFlow tensor
    tensor = tf.convert_to_tensor(tensor_data, dtype=tf.float32)
    return tensor

# Example usage of the function
tribe_state = {'population': 1500, 'resources': 3000, 'happiness': 100}
tribe_state_tensor = convert_tribe_state_to_tensor(tribe_state)
def perform_actions(actions):
    for turn, step in enumerate(actions):
        for i, tribe_action in enumerate(step):
            if Train.environment.pyenv.envs[0]._tribes[i].population > 0:
                action_name = list({"attack": 0.2, "collect": 0.6, "pass": 0.2}.keys())[list({"attack": 0.2, "collect": 0.6, "pass": 0.2}.values()).index(tribe_action)]
                action_counts[action_name] += 1  # Update action count

                # Perform the action and update tribe information
                Train.environment.pyenv.envs[0]._tribes[i].perform_actions(None, {"attack": 0.2, "collect": 0.6, "pass": 0.2})


    Train.environment.pyenv.envs[0]._tribes = [tribe for tribe in Train.environment.pyenv.envs[0]._tribes if tribe.population > 0]
    if not Train.environment.pyenv.envs[0]._tribes:
        print("All tribes are dead. Reinitializing new tribes.")
        new_tribes = create_tribes(4)
        initialize_tribes(new_tribes)
        simulate(new_tribes, agent)
        return new_tribes

    return None

    # Remove tribes with 0 population
    Train.environment.pyenv.envs[0]._tribes = [tribe for tribe in Train.environment.pyenv.envs[0]._tribes if tribe.population > 0]

    # Check if there are no tribes left
    if not Train.environment.pyenv.envs[0]._tribes:
        print("All tribes are dead. Reinitializing new tribes.")
        new_tribes = create_tribes(4)
        initialize_tribes(new_tribes)
        simulate(new_tribes, agent)
        return new_tribes

    return None



def print_training_info(iteration, train_loss):
    print(f"Step: {Train.agent.train_step_counter.numpy()}, Loss: {train_loss.loss.numpy()}")

    # Store training information for plotting
    Train.steps.append(Train.agent.train_step_counter.numpy())
    Train.losses.append(train_loss.loss.numpy())


def plot_training_and_actions():
    # Create a figure with two subplots
    fig, (ax1, ax2) = Train.plt.subplots(2, 1, figsize=(8, 6))

    # Plot training loss
    ax1.plot(Train.steps, Train.losses, label='Training Loss')
    ax1.set_xlabel('Training Steps')
    ax1.set_ylabel('Loss')
    ax1.legend()

    # Plot action counts
    actions = list(action_counts.keys())
    counts = list(action_counts.values())
    ax2.bar(actions, counts)
    ax2.set_xlabel('Actions')
    ax2.set_ylabel('Counts')
    ax2.set_title('Distribution of Actions')

    # Adjust spacing between subplots
    Train.plt.subplots_adjust(hspace=0.5)

    # Show the combined plot
    Train.plt.show()
def save_model_and_exit():
    # Save the model when interrupted
    Train.save_and_exit(None, None, Train.saved_model_path)

    # Save the trained model using tf.saved_model.save
    if not Train.os.path.exists(Train.saved_model_path):
        Train.os.makedirs(Train.saved_model_path)

    # Save the model
    Train.tf.saved_model.save(Train.agent.policy, Train.saved_model_path)


# Create tribes
initial_tribes = create_tribes(4)
initialize_tribes(initial_tribes)
simulate(initial_tribes, agent)

# Train
try:
    for iteration in range(Train.num_iterations):
        # Reset action counts
        action_counts = {"attack": 0, "collect": 0, "pass": 0}


        train_loss = Train.train_step()
        if Train.agent.train_step_counter.numpy() % 10000 == 0:
            print_training_info(iteration, train_loss)
        print("\nTribe information after training:")
        plot_training_and_actions()
        if all(tribe.population == 0 for tribe in Train.environment.pyenv.envs[0]._tribes):
            print("All tribes are eliminated. Training has ended.")
            save_model_and_exit()
            break



except KeyboardInterrupt:
    save_model_and_exit()
