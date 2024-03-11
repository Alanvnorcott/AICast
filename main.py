# main.py

from tribe import create_tribes, initialize_tribes
from simulate import simulate
import Train
from Train import agent
import time

def print_tribe_information(prefix, iteration):
    for i, tribe in enumerate(Train.environment.pyenv.envs[0]._tribes):
        tribe_name = chr(ord('A') + i)
        traits = ', '.join(trait.name for trait in tribe.traits)
        print(
            f"{prefix}Tribe {tribe_name} with Traits [{traits}]: Population: {tribe.population} Resources: {tribe.resources} Happiness: {tribe.happiness} Generation {iteration + 1}")

def perform_actions_and_print_info(actions):
    for turn, step in enumerate(actions):
        print(f"\n----- Turn {turn + 1} -----")
        for i, tribe_action in enumerate(step):
            tribe_name = chr(ord('A') + i)
            tribe_action_name = Train.actions_mapping.get(int(tribe_action), "unknown")
            print(f"Tribe {tribe_name}: {tribe_action_name}")

            # Perform the action and update tribe information
            Train.environment.pyenv.envs[0]._tribes[i].perform_actions(None, {"attack": 0.0, "collect": 1.0, "pass": 0.0})

            # Print tribe information after each action
            tribe = Train.environment.pyenv.envs[0]._tribes[i]
            traits = ', '.join(trait.name for trait in tribe.traits)
            print(
                f"Tribe {tribe_name} with Traits [{traits}]: Population: {tribe.population} Resources: {tribe.resources} Happiness: {tribe.happiness}")

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
        print_tribe_information("Before action ", iteration)

        # Print actions for all tribes during each turn
        print("\nActions taken during training:")
        perform_actions_and_print_info(Train.experience.action)

        # Train the agent
        train_loss = Train.train_step()

        if Train.agent.train_step_counter.numpy() % 10000 == 0:
            # Print training information
            print_training_info(iteration, train_loss)

        # Print tribe information after training
        print("\nTribe information after training:")
        print_tribe_information("After training ", iteration)

        # Plot the training loss
        plot_training_loss()

        # Check if there are no tribes left
        if all(tribe.population == 0 for tribe in Train.environment.pyenv.envs[0]._tribes):
            print("Training ended as there are no tribes left.")
            break


except KeyboardInterrupt:
    # Save the model when interrupted
    Train.save_and_exit(None, None, Train.saved_model_path)

    # Save the trained model using tf.saved_model.save
    if not Train.os.path.exists(Train.saved_model_path):
        Train.os.makedirs(Train.saved_model_path)

    # Save the model
    Train.tf.saved_model.save(Train.agent.policy, Train.saved_model_path)
