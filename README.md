# Multi-Player Hybrid Monopoly Reinforcement Learning Environment (Gymnasium)

A **multi-player** and **hybrid** reinforcement learning environment for the game of Monopoly, designed for training agents that interact within a standard Monopoly game context and implemented using the [Gymnasium](https://gymnasium.farama.org/) API structure.

## 🎮 About the Project

This project aims to create a **multi-player** Monopoly game environment compliant with the [Gymnasium](https://gymnasium.farama.org/) interface, allowing for training Reinforcement Learning (RL) agents to play as one or more players within a standard Monopoly game. The environment separates the base game logic (`MonopolyGame`) from the specific interface for RL training (`MonopolyRLEnv`).

It is designed to support **multi-agent scenarios**, such as training a single RL agent to play against simulated (heuristic) opponents, or potentially serving as groundwork for more complex multi-agent RL setups. The environment's **hybrid** nature stems from the design where the RL agent controls specific player *management* actions (like building, trading), while the environment handles the core game flow (dice rolls, movement, landing consequences, managing other players' turns). It includes handling of observation and action spaces, as well as action masks to guide the agent's valid choices during its turn or phase.

**Note:** This project is under active development. The current implementation of the core game logic within `MonopolyRLEnv`'s `step` function is incomplete, and the trading system is simplified. Refer to the [Project Status](#project-status) section and the [Improvement Plan](#improvement-plan) for more details.

## ✨ Current Features

* Designed for a **multi-player game structure**.
* Compliance with the [Gymnasium](https://gymnasium.farama.org/) API structure (`gym.Env`), adapting it for a multi-player, turn-based context.
* Separation between the RL environment (`MonopolyRLEnv`) and the core game logic (`MonopolyGame`).
* Definition of observation and action spaces (`gym.spaces.Dict`, `Box`, `Discrete`, `MultiBinary`) which **include information about all players**.
* Support for `action_masks` in the observation to indicate **valid management actions for the current player**.
* Partial data normalization for properties.

## 🚀 Installation

1.  Clone this repository:
    ```bash
    git clone [https://github.com/AdamArdent/Monopoly_AI_RL.git](https://github.com/AdamArdent/Monopoly_AI_RL.git)
    cd Monopoly_AI_RL
    ```
2.  Install dependencies (make sure you have Python 3.7+ and pip):
    ```bash
    pip install -r requirements.txt
    ```

## 💡 Usage

Here's a simple example showing interaction with the environment, typically controlling one player's actions per step within their turn or phase:

```python
import gymnasium as gym
# Make sure the path to your environment is correct
# This depends on your project structure and the defined entry_point
# in setup.py or how you register the environment.
# If the environment is not yet registered, you might need to import it directly:
# from your_module.monopoly_env import MonopolyRLEnv

# Example if the environment is registered as 'MonopolyRLEnv-v0'
# Note: The environment handles multiple players internally,
# but the standard Gymnasium step interface typically returns
# observation/reward for the *current* agent whose turn it is to act.
env = gym.make('MonopolyRLEnv-v0', num_players=4) # Example: set number of players

observation, info = env.reset(seed=42)

for _ in range(1000): # Limit the number of steps for the example
    # The agent needs to process the observation (which includes other players' states)
    # and choose a valid management action for the current player based on action_mask
    action_mask = observation['action_mask']
    valid_actions = [i for i, mask in enumerate(action_mask) if mask == 1]
    if not valid_actions:
        # Exceptional case if no valid action is available (e.g., stuck),
        # should not happen normally if 'do nothing' is always valid.
        print("No valid actions available for the current player, episode ends.")
        break

    # Example of choosing a valid random management action for the current player
    import random
    action = random.choice(valid_actions)

    # The step function processes this action for the current player,
    # advances the game state (which might include dice rolls, movement,
    # landing consequences for the current player, and potentially
    # transitioning to the next player's turn or phase).
    observation, reward, terminated, truncated, info = env.step(action)

    print(f"Step {_}: Player {info.get('current_player_idx', 'N/A')} Action={action}, Reward={reward}, Terminated={terminated}, Truncated={truncated}")
    # Note: Reward and observation are typically for the player who just acted.
    # The 'info' dictionary might contain details about whose turn it is next.


    if terminated or truncated:
        print("Episode finished.")
        break

env.close()
