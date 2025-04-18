import random
from typing import List, Optional, Dict, Any, Tuple
import numpy as np
import gymnasium as gym
from environment.board import Board
from environment.player import Player

NUM_PROPERTIES = 28
MAX_MONEY = 10000
NUM_CASE = 40

#TODO: modifier les méthodes déjà existante pour l'utilisation de l'ia.
class MonopolyRLEnv(gym.Env):
    """
    Monopoly environment for reinforcement learning that follows the Gymnasium interface.
    This class focuses solely on the RL interface, separating it from human-playable game logic.
    """

    def __init__(self):
        print("Initializing Monopoly RL Environment")
        # Initialize game components
        self.players = self._initialize_players()
        self.board = Board()
        self._init_property_states()

        # Property tracking
        self.property_order = self.board.property_order
        self.property_data = self.board.property_data
        self.property_data_norm = (self.property_data - self.board.property_min) / (
                self.board.property_max - self.board.property_min + 1e-8
        )

        # Define observation space
        self.observation_space = gym.spaces.Dict({
            "self_money": gym.spaces.Box(low=0, high=MAX_MONEY, shape=(1,), dtype=np.int32),
            "self_position": gym.spaces.Discrete(NUM_CASE),
            "self_properties": gym.spaces.MultiBinary(NUM_PROPERTIES),
            "others_money": gym.spaces.Box(low=0, high=MAX_MONEY, shape=(3,), dtype=np.int32),
            "active_players": gym.spaces.MultiBinary(4),
            "others_properties": gym.spaces.Box(
                low=0, high=5, shape=(3, NUM_PROPERTIES), dtype=np.int8
            ),
            "others_positions": gym.spaces.Box(low=0, high=NUM_CASE, shape=(3,), dtype=np.int32),
            "self_houses": gym.spaces.Box(low=0, high=5, shape=(NUM_PROPERTIES,), dtype=np.int8),
            "all_properties": gym.spaces.Box(
                low=0.0, high=1.0, shape=(NUM_PROPERTIES, 11), dtype=np.float32
            ),
            "others_houses": gym.spaces.Box(
                low=0, high=5, shape=(3, NUM_PROPERTIES), dtype=np.int8
            ),
            "action_masks": gym.spaces.Dict({
                "mortgage": gym.spaces.MultiBinary(NUM_PROPERTIES),
                "build": gym.spaces.MultiBinary(NUM_PROPERTIES),
                "can_trade": gym.spaces.MultiBinary(1)
            }),
        })

        # Define action space
        self.action_space = gym.spaces.Dict({
            "action_type": gym.spaces.Discrete(5),  # 0-4 for different actions
            "property_idx": gym.spaces.Discrete(NUM_PROPERTIES),
            "trade_partner": gym.spaces.Discrete(3),
            "trade_amount": gym.spaces.Box(low=0, high=MAX_MONEY, shape=(1,), dtype=np.int32)
        })

        # Track current player
        self.current_player_idx = 0

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[Dict, Dict]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        self.players = self._initialize_players()
        self.board = Board()
        self._init_property_states()
        self.current_player_idx = 0

        observation = self._get_obs_for_player(self.players[self.current_player_idx])
        info = self._get_info()

        return observation, info

    def step(self, action: Dict[str, Any]) -> Tuple[Dict, float, bool, bool, Dict]:
        """
        Execute one step in the environment.

        Args:
            action: Dictionary containing the action to take

        Returns:
            observation: The next observation
            reward: The reward for taking this action
            terminated: Whether the episode is terminated
            truncated: Whether the episode was truncated
            info: Additional information
        """
        player = self.players[self.current_player_idx]
        reward = 0
        done = False
        info = {}

        try:
            action_type = action["action_type"]

            if action_type == 0:  # Mortgage
                property_idx = action["property_idx"]
                if property_idx < len(self.property_order):
                    prop = self.property_order[property_idx]
                    if prop in self._get_mortgageable_properties(player):
                        self._handle_mortgage(player, prop)
                        reward += 5  # Small reward for freeing up cash
                    else:
                        reward -= 2  # Penalty for invalid mortgage attempt

            elif action_type == 1:  # Build
                property_idx = action["property_idx"]
                if property_idx < len(self.property_order):
                    prop = self.property_order[property_idx]
                    if prop in self._get_buildable_properties(player):
                        self._handle_build(player, prop)
                        reward += 10  # Reward for development
                    else:
                        reward -= 2  # Penalty for invalid build attempt

            elif action_type == 2:  # Trade money for property
                partner_idx = action["trade_partner"]
                property_idx = action["property_idx"]
                amount = action["trade_amount"][0]

                other_players = self._get_other_players(player)
                if partner_idx < len(other_players) and property_idx < len(self.property_order):
                    partner = other_players[partner_idx]
                    prop = self.property_order[property_idx]

                    if prop in partner.properties and player.money >= amount:
                        self._handle_trade(player, partner, prop, amount)
                        reward += 15  # Good reward for successful trade
                    else:
                        reward -= 2  # Penalty for invalid trade

            elif action_type == 3:  # Trade property for property
                partner_idx = action["trade_partner"]
                property_idx = action["property_idx"]

                other_players = self._get_other_players(player)
                if partner_idx < len(other_players) and property_idx < len(self.property_order):
                    partner = other_players[partner_idx]
                    self._handle_property_swap(player, partner, property_idx)
                    reward += 15  # Good reward for successful trade

            elif action_type == 4:  # Do nothing
                reward -= 1  # Small negative reward for doing nothing

            # Add reward based on player's overall state
            reward += self._calculate_reward(player)

        except Exception as e:
            reward -= 10  # Larger penalty for errors
            info["error"] = str(e)

        # Move to next player
        self._cycle_to_next_player()
        next_obs = self._get_obs_for_player(self.players[self.current_player_idx])

        # Check if game is over (only one player left)
        active_players = [p for p in self.players if not p.bankrupt]
        terminated = len(active_players) <= 1
        truncated = False

        return next_obs, reward, terminated, truncated, info

    def _get_info(self) -> Dict[str, Any]:
        """Return information about the current state of the game."""
        active_players = [p for p in self.players if not p.bankrupt]
        return {
            "active_players_count": len(active_players),
            "current_player": self.current_player_idx,
            "player_money": [p.money for p in self.players],
            "player_properties_count": [len(p.properties) for p in self.players],
            "bankrupt_players": [p.bankrupt for p in self.players]
        }

    def _get_obs_for_player(self, player: Player) -> Dict[str, Any]:
        """
        Generate observation for the current player.
        Includes player state and information about other players.
        """
        other_players = [p for p in self.players if p != player and not p.bankrupt]

        # Calculate valid actions
        mortgageable = [int(prop in self._get_mortgageable_properties(player)) for prop in self.property_order]
        buildable = [int(prop in self._get_buildable_properties(player)) for prop in self.property_order]

        # Get data for other players
        others_money = np.zeros(3, dtype=np.int32)
        others_properties = np.zeros((3, NUM_PROPERTIES), dtype=np.int8)
        others_positions = np.zeros(3, dtype=np.int32)
        others_houses = np.zeros((3, NUM_PROPERTIES), dtype=np.int8)
        active_players = np.array([int(not p.bankrupt) for p in self.players], dtype=np.int8)

        for i, other in enumerate(other_players[:3]):
            others_money[i] = other.money
            others_properties[i] = self._properties_to_binary(other)
            others_positions[i] = other.position
            others_houses[i] = self._get_houses_vector(other)

        return {
            # Personal state
            "self_money": np.array([player.money], dtype=np.int32),
            "self_position": player.position,
            "self_properties": self._properties_to_binary(player),
            "self_houses": self._get_houses_vector(player),
            "action_masks": {
                "mortgage": np.array(mortgageable, dtype=np.int8),
                "build": np.array(buildable, dtype=np.int8),
                "can_trade": np.array([int(len(player.properties) > 0)], dtype=np.int8)
            },

            # Other players' states
            "others_money": others_money,
            "others_properties": others_properties,
            "others_positions": others_positions,
            "others_houses": others_houses,
            "active_players": active_players,

            # Property data
            "all_properties": self.property_data_norm,
        }

    def _get_houses_vector(self, player: Player) -> np.ndarray:
        """Convert player's houses to a vector format."""
        houses = np.zeros(NUM_PROPERTIES, dtype=np.int8)
        for idx, prop_name in enumerate(self.property_order):
            if prop_name in player.properties:
                prop = self._get_board_property(prop_name)
                houses[idx] = min(prop.get("houses", 0), 5)
            else:
                houses[idx] = 0  # No houses since player doesn't own property
        return houses

    def _properties_to_binary(self, player: Player) -> np.ndarray:
        """Convert player's properties to binary vector."""
        binary_vector = np.zeros(NUM_PROPERTIES, dtype=np.int8)
        for idx, prop_name in enumerate(self.property_order):
            if prop_name in player.properties:
                binary_vector[idx] = 1
        return binary_vector

    def _get_mortgageable_properties(self, player: Player) -> List[str]:
        """Get list of properties player can mortgage."""
        return [
            prop for prop in player.properties
            if not self._get_board_property(prop).get("mortgaged", False)
        ]

    def _get_buildable_properties(self, player: Player) -> List[str]:
        """Get list of properties player can build on."""
        buildable = []
        for prop in player.properties:
            prop_data = self._get_board_property(prop)
            if prop_data["type"] == "property" and not prop_data.get("mortgaged", False):
                color_group = self._get_color_group(prop_data["color_code"])
                if all(p in player.properties for p in color_group):
                    buildable.append(prop)
        return buildable

    def _handle_mortgage(self, player: Player, property_name: str) -> None:
        """Handle mortgaging a property."""
        prop = self._get_board_property(property_name)
        if not prop:
            raise ValueError(f"Property {property_name} not found")

        if property_name not in player.properties:
            raise ValueError(f"Player doesn't own {property_name}")

        if prop.get("mortgaged", False):
            raise ValueError(f"{property_name} is already mortgaged")

        prop["mortgaged"] = True
        player.receive(prop["hypothèque"])

    def _handle_build(self, player: Player, property_name: str) -> None:
        """Handle building a house on a property."""
        prop = self._get_board_property(property_name)
        if not prop:
            raise ValueError(f"Property {property_name} not found")

        if property_name not in player.properties:
            raise ValueError(f"Player doesn't own {property_name}")

        if prop["type"] != "property":
            raise ValueError(f"Cannot build on {property_name} (not a property)")

        if prop.get("mortgaged", False):
            raise ValueError(f"Cannot build on mortgaged property {property_name}")

        # Check color group ownership
        color_group = self._get_color_group(prop["color_code"])
        if any(p not in player.properties for p in color_group):
            raise ValueError("Must own all properties in color group to build")

        # Check funds
        house_cost = prop["price"] // 2
        if player.money < house_cost:
            raise ValueError(f"Not enough money to build (need {house_cost})")

        # Add house
        prop["houses"] = min(prop.get("houses", 0) + 1, 5)
        player.pay(house_cost)

    @staticmethod
    def _handle_trade(buyer: Player, seller: Player, property_name: str, amount: int) -> None:
        """Handle trading money for property between players."""
        if property_name not in seller.properties:
            raise ValueError(f"Seller doesn't own {property_name}")

        if buyer.money < amount:
            raise ValueError(f"Buyer doesn't have enough money ({amount})")

        buyer.pay(amount)
        seller.receive(amount)
        seller.properties.remove(property_name)
        buyer.properties.append(property_name)

    @staticmethod
    def _handle_property_swap(player1: Player, player2: Player, property_idx: int) -> None:
        """Handle swapping properties between players."""
        # For simplicity, swap player1's first property with player2's property at index
        if not player1.properties:
            raise ValueError("Player 1 has no properties to trade")

        player1_prop = player1.properties[0]

        if property_idx >= len(player2.properties):
            raise ValueError("Invalid property index for Player 2")

        player2_prop = player2.properties[property_idx]

        player1.properties.remove(player1_prop)
        player2.properties.append(player1_prop)

        player2.properties.remove(player2_prop)
        player1.properties.append(player2_prop)

    def _cycle_to_next_player(self) -> None:
        """Move to next active player."""
        for _ in range(len(self.players)):
            self.current_player_idx = (self.current_player_idx + 1) % len(self.players)
            if not self.players[self.current_player_idx].bankrupt:
                break

    def _calculate_reward(self, player: Player) -> float:
        """Calculate reward based on player's state."""
        reward = 0

        # Money reward
        reward += player.money * 0.01

        # Properties reward
        reward += len(player.properties) * 5

        # Houses reward
        for prop in player.properties:
            case = self._get_board_property(prop)
            reward += case.get("houses", 0) * 10

        # Bankruptcy penalty
        if player.bankrupt:
            reward -= 1000

        return reward

    def _get_board_property(self, property_name: str) -> Dict:
        """Get property data from board by name."""
        for case in self.board.board:
            if case["name"] == property_name:
                return case
        raise ValueError(f"Property {property_name} not found on board")

    def _get_color_group(self, color_code: str) -> List[str]:
        """Get all properties in a color group."""
        return [
            p["name"] for p in self.board.board
            if p.get("color_code") == color_code and p["type"] == "property"
        ]

    def _get_other_players(self, current_player: Player) -> List[Player]:
        """Get list of other active players."""
        return [p for p in self.players if p != current_player and not p.bankrupt]

    @staticmethod
    def _initialize_players() -> List[Player]:
        """Initialize player objects."""
        return [Player(name=f"Player {i + 1}") for i in range(4)]

    def _init_property_states(self) -> None:
        """Initialize property states on the board."""
        for case in self.board.board:
            if case["type"] in ["property", "station", "utility"]:
                case.setdefault("houses", 0)
                case.setdefault("mortgaged", False)

#TODO: implementer méthodes manquante de Game.py à ici
class MonopolyGame:
    """
    Human-playable Monopoly game class.
    This class is separate from the RL environment and handles human interaction.
    """

    def __init__(self):
        print("Initializing Human-Playable Monopoly Game")
        self.players = self._initialize_players()
        self.board = Board()
        self._init_property_states()

    def start(self):
        """Start the game and run until completion."""
        round_number = 1
        active_players = [p for p in self.players if not p.bankrupt]

        while len(active_players) > 1:
            print(f"\n====== Round {round_number} ======")
            for player in active_players:
                print(f"\n--- Turn of {player.name} ---")
                self._handle_player_turn(player)

                # Update active players
                active_players = [p for p in self.players if not p.bankrupt]
                if len(active_players) <= 1:
                    break

            round_number += 1

        if active_players:
            print(f"\nCongratulations, {active_players[0].name} has won the game!")
        else:
            print("The game ended without a winner.")

    def _handle_player_turn(self, player: Player):
        """Handle a player's turn in the game."""
        input(f"{player.name}, press Enter to roll the dice...")
        dice_roll = self._roll_dice()
        print(f"{player.name} rolled {dice_roll}.")

        # Update player position
        player.position = self.board.move_player(player.position, dice_roll)
        current_case = self.board.get_case(player.position)
        print(f"{player.name} moves to '{current_case['name']}'.")

        # Handle landing on case
        self._handle_landing_on_case(player, current_case)

    def _handle_landing_on_case(self, player: Player, current_case: Dict):
        """Handle the actions when a player lands on a specific case."""
        case_type = current_case.get("type", "")

        if case_type == "property":
            self._handle_property_case(player, current_case)
        elif case_type == "station":
            self._handle_station_case(player, current_case)
        elif case_type == "utility":
            self._handle_utility_case(player, current_case)
        elif case_type == "tax":
            self._handle_tax_case(player, current_case)
        elif case_type == "chest" or case_type == "chance":
            self._handle_card_case(player, case_type)
        elif case_type == "go_to_jail":
            self._handle_go_to_jail(player)
        # Add other case types as needed

        print(f"{player.name} now has ${player.money}.")

    def _handle_property_case(self, player: Player, property_case: Dict):
        """Handle landing on a property case."""
        property_name = property_case["name"]
        owner = self._find_property_owner(property_name)

        if owner is None:
            # No owner, player can buy it
            if player.money >= property_case["price"]:
                buy_choice = input(f"Do you want to buy {property_name} for ${property_case['price']}? (y/n): ")
                if buy_choice.lower() == 'y':
                    player.pay(property_case["price"])
                    player.properties.append(property_name)
                    print(f"{player.name} now owns {property_name}!")
                else:
                    self._auction_property(property_name, property_case["price"])
            else:
                print(f"{player.name} doesn't have enough money to buy {property_name}.")
                self._auction_property(property_name, property_case["price"])
        elif owner != player:
            # Property is owned by another player, pay rent
            rent = self._calculate_rent(property_case, owner)
            print(f"{player.name} pays ${rent} rent to {owner.name}.")
            player.pay(rent)
            owner.receive(rent)

    def _handle_station_case(self, player: Player, station_case: Dict):
        """Handle landing on a station case."""
        station_name = station_case["name"]
        owner = self._find_property_owner(station_name)

        if owner is None:
            if player.money >= station_case["price"]:
                buy_choice = input(f"Do you want to buy {station_name} for ${station_case['price']}? (y/n): ")
                if buy_choice.lower() == 'y':
                    player.pay(station_case["price"])
                    player.properties.append(station_name)
                    print(f"{player.name} now owns {station_name}!")
                else:
                    self._auction_property(station_name, station_case["price"])
            else:
                print(f"{player.name} doesn't have enough money to buy {station_name}.")
                self._auction_property(station_name, station_case["price"])
        elif owner != player:
            # Station is owned by another player, pay rent
            stations_owned = sum(1 for prop in owner.properties if self._get_board_property(prop)["type"] == "station")
            rent = station_case["rent"] * (2 ** (stations_owned - 1))
            print(f"{player.name} pays ${rent} rent to {owner.name}.")
            player.pay(rent)
            owner.receive(rent)

    def _handle_utility_case(self, player: Player, utility_case: Dict):
        """Handle landing on a utility case."""
        utility_name = utility_case["name"]
        owner = self._find_property_owner(utility_name)

        if owner is None:
            if player.money >= utility_case["price"]:
                buy_choice = input(f"Do you want to buy {utility_name} for ${utility_case['price']}? (y/n): ")
                if buy_choice.lower() == 'y':
                    player.pay(utility_case["price"])
                    player.properties.append(utility_name)
                    print(f"{player.name} now owns {utility_name}!")
                else:
                    self._auction_property(utility_name, utility_case["price"])
            else:
                print(f"{player.name} doesn't have enough money to buy {utility_name}.")
                self._auction_property(utility_name, utility_case["price"])
        elif owner != player:
            # Utility is owned by another player, pay rent based on dice roll
            utilities_owned = sum(1 for prop in owner.properties if self._get_board_property(prop)["type"] == "utility")
            dice_roll = self._roll_dice()
            print(f"{player.name} rolls {dice_roll} for utility payment.")

            if utilities_owned == 1:
                rent = dice_roll * 4
            else:
                rent = dice_roll * 10

            print(f"{player.name} pays ${rent} to {owner.name}.")
            player.pay(rent)
            owner.receive(rent)

    @staticmethod
    def _handle_tax_case(player: Player, tax_case: Dict):
        """Handle landing on a tax case."""
        tax_amount = tax_case.get("amount", 0)
        print(f"{player.name} pays ${tax_amount} in tax.")
        player.pay(tax_amount)

    # TODO : méthode déjà défini dans `Game.py`, besoins de l'implementer.

    @staticmethod
    def _handle_card_case(player: Player, card_type: str):
        """Handle landing on a chance or community chest case."""
        print(f"{player.name} draws a {card_type} card.")
        # Implementation for cards would go here
        # For simplicity, just a placeholder
        card_effects = [
            {"description": "Bank pays you dividend of $50.", "action": lambda p: p.receive(50)},
            {"description": "Pay hospital fees of $100.", "action": lambda p: p.pay(100)},
            {"description": "Advance to GO.", "action": lambda p: setattr(p, "position", 0)},
        ]
        card = random.choice(card_effects)
        print(f"Card says: {card['description']}")
        card["action"](player)

    def _handle_go_to_jail(self, player: Player):
        """Handle landing on the Go To Jail case."""
        print(f"{player.name} goes to jail!")
        jail_position = next(i for i, case in enumerate(self.board.board) if case["name"] == "Jail")
        player.position = jail_position

    def _find_property_owner(self, property_name: str) -> Optional[Player]:
        """Find which player owns a property."""
        for player in self.players:
            if property_name in player.properties:
                return player
        return None

    def _calculate_rent(self, property_case: Dict, owner: Player) -> int:
        """Calculate rent for a property based on houses/hotels."""
        houses = property_case.get("houses", 0)

        if houses == 0:
            # Check if owner has monopoly (owns all properties in color group)
            color_code = property_case.get("color_code")
            color_group = self._get_color_group(color_code)
            has_monopoly = all(prop in owner.properties for prop in color_group)

            base_rent = property_case["rent"]
            return base_rent * (2 if has_monopoly else 1)
        else:
            # Rent with houses
            rent_key = f"rent{houses}"
            if rent_key in property_case:
                return property_case[rent_key]
            else:
                # Fallback if specific house rent not defined
                return property_case["rent"] * (houses + 1)

    def _auction_property(self, property_name: str, starting_price: int):
        """Auction a property to the highest bidder."""
        print(f"\nAuction for {property_name} starting at ${starting_price}")

        current_price = starting_price // 2  # Start at half price
        active_bidders = [p for p in self.players if not p.bankrupt and p.money >= current_price]
        highest_bidder = None

        while len(active_bidders) > 0:
            for player in active_bidders[:]:
                print(f"Current bid: ${current_price}")
                bid_choice = input(f"{player.name}, do you want to bid ${current_price + 10}? (y/n): ")

                if bid_choice.lower() == 'y':
                    current_price += 10
                    highest_bidder = player
                else:
                    active_bidders.remove(player)

                if len(active_bidders) == 1:
                    break

            if len(active_bidders) == 0 and highest_bidder is None:
                print(f"No one bought {property_name}.")
                return

        if highest_bidder:
            print(f"{highest_bidder.name} won the auction for {property_name} at ${current_price}")
            highest_bidder.pay(current_price)
            highest_bidder.properties.append(property_name)

    def _get_color_group(self, color_code: str) -> List[str]:
        """Get all properties in a color group."""
        return [
            p["name"] for p in self.board.board
            if p.get("color_code") == color_code and p["type"] == "property"
        ]

    def _get_board_property(self, property_name: str) -> Dict:
        """Get property data from board by name."""
        for case in self.board.board:
            if case["name"] == property_name:
                return case
        raise ValueError(f"Property {property_name} not found on board")

    @staticmethod
    def _initialize_players() -> List[Player]:
        """Initialize player objects."""
        return [Player(name=f"Player {i + 1}") for i in range(4)]

    def _init_property_states(self):
        """Initialize property states on board."""
        for case in self.board.board:
            if case["type"] in ["property", "station", "utility"]:
                case.setdefault("houses", 0)
                case.setdefault("mortgaged", False)

    @staticmethod
    def _roll_dice() -> int:
        """Roll dice and return total."""
        return random.randint(1, 6) + random.randint(1, 6)


# Register the environment with Gymnasium
gym.register(
    id="MonopolyRL-v0",
    entry_point="environment.game:MonopolyRLEnv",
)

if __name__ == "__main__":
    # For human play, use this:
    game = MonopolyGame()
    game.start()

    # For RL training, use this:
    env = MonopolyRLEnv()
    initial_obs, initial_info = env.reset()

    # Example of taking a random action
    action = env.action_space.sample()
    next_obs, reward, terminated, truncated, step_info = env.step(action)
    print(f"Reward: {reward}")