# main.py

from tribe import create_tribes, initialize_tribes
from simulate import simulate
import Train
from Train import agent, save_and_exit, saved_model_path
import matplotlib.pyplot as plt

action_counts = {"attack": 0, "collect": 0, "pass": 0}

def perform_actions(actions):
    for turn, step in enumerate(actions):
        for i, tribe_action in enumerate(step):
            if Train.environment.pyenv.envs[0]._tribes[i].population > 0:
                action_name = list({"attack": 0.2, "collect": 0.6, "pass": 0.2}.keys())[list({"attack": 0.2, "collect": 0.6, "pass": 0.2}.values()).index(tribe_action)]
                action_counts[action_name] += 1  # Update action count

                # Perform the action and update tribe information
                Train.environment.pyenv.envs[0]._tribes[i].perform_actions(None, {"attack": 0.2, "collect": 0.6, "pass": 0.2})

    # Remove tribes with 0 population
    Train.environment.pyenv.envs[0]._tribes = [tribe for tribe in Train.environment.pyenv.envs[0]._tribes if tribe.population > 0]

    # Check if there are no tribes left
    if not Train.environment.pyenv.envs[0]._tribes:
        print("All tribes are dead. Reinitializing new tribes.")

        # Create and initialize a new set of tribes
        new_tribes = create_tribes(4)
        initialize_tribes(new_tribes)

        # Simulate with the new tribes
        simulate(new_tribes, agent)

        # Return the new tribes for the next iteration
        return new_tribes

    return None

    # Remove tribes with 0 population
    Train.environment.pyenv.envs[0]._tribes = [tribe for tribe in Train.environment.pyenv.envs[0]._tribes if tribe.population > 0]

    # Check if there are no tribes left
    if not Train.environment.pyenv.envs[0]._tribes:
        print("All tribes are dead. Reinitializing new tribes.")

        # Create and initialize a new set of tribes
        new_tribes = create_tribes(4)
        initialize_tribes(new_tribes)

        # Simulate with the new tribes
        simulate(new_tribes, agent)

        # Return the new tribes for the next iteration
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

# Initialize tribes
initialize_tribes(initial_tribes)

# Simulate and pass the 'agent' object
simulate(initial_tribes, agent)

# Train
try:
    for iteration in range(Train.num_iterations):
        # Reset action counts
        action_counts = {"attack": 0, "collect": 0, "pass": 0}

        # Train the agent
        train_loss = Train.train_step()

        if Train.agent.train_step_counter.numpy() % 10000 == 0:
            # Print training information
            print_training_info(iteration, train_loss)

        # Print tribe information after training
        print("\nTribe information after training:")

        plot_training_and_actions()

        # Check if there are no tribes left
        if all(tribe.population == 0 for tribe in Train.environment.pyenv.envs[0]._tribes):
            print("All tribes are eliminated. Training has ended.")
            save_model_and_exit()
            break



except KeyboardInterrupt:
    # Save the model when interrupted
    save_model_and_exit()
