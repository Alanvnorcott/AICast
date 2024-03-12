# main.py

from tribe import create_tribes, initialize_tribes
from simulate import simulate
import Train
from Train import agent, save_and_exit, saved_model_path
import time


def perform_actions(actions):
    for turn, step in enumerate(actions):
        for i, tribe_action in enumerate(step):
            # Perform the action and update tribe information
            Train.environment.pyenv.envs[0]._tribes[i].perform_actions(None, {"attack": 0.0, "collect": 1.0, "pass": 0.0})

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


def plot_training_loss():
    Train.plt.plot(Train.steps, Train.losses, label='Training Loss')
    Train.plt.xlabel('Training Steps')
    Train.plt.ylabel('Loss')
    Train.plt.legend()
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
        # Print tribe information before each action

        new_tribes = perform_actions(Train.experience.action)

        # Train the agent
        train_loss = Train.train_step()

        if Train.agent.train_step_counter.numpy() % 10000 == 0:
            # Print training information
            print_training_info(iteration, train_loss)

        # Print tribe information after training
        print("\nTribe information after training:")

        # Plot the training loss
        plot_training_loss()

        # Check if there are no tribes left
        if all(tribe.population == 0 for tribe in Train.environment.pyenv.envs[0]._tribes):
            print("Training ended as there are no tribes left.")
            break

        if new_tribes is not None:
            # If new tribes are reinitialized, use them for the next iteration
            initial_tribes = new_tribes

except KeyboardInterrupt:
    # Save the model when interrupted
    save_model_and_exit()
