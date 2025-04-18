import random
from typing import List, Optional
from gym.error import InvalidAction
from environment.board import Board
from environment.player import Player
import gymnasium as gym
import random
import numpy as np

# Number of properties in the Monopoly game
NUM_PROPERTIES = 28
# Maximum amount of money a player can have
MAX_MONEY = 10000
# Number of cases on the Monopoly board
NUM_CASE = 40


class Game(gym.Env):
    """
    Monopoly game environment implementing the Gymnasium interface.
    This class represents a complete Monopoly game that can be used for
    reinforcement learning training or human play.
    """

    def __init__(self):
        print("Initializing Game")
        self.players = self._initialize_players()
        self.board = Board()
        self._init_property_states()
        self.property_order = [
            case['name'] for case in self.board.board
            if case['type'] in ['property', 'station', 'utility']
        ]
        # Verify consistency with NUM_PROPERTIES
        assert len(self.property_order) == NUM_PROPERTIES, "Inconsistency in properties count"
        # Direct retrieval from the Board
        self.property_order = self.board.property_order
        self.property_data = self.board.property_data
        # Min-Max normalization
        self.property_data_norm = (self.board.property_data - self.board.property_min) / (
                    self.board.property_max - self.board.property_min + 1e-8)
        # Define the enhanced observation space
        self.observation_space = gym.spaces.Dict({
            "self_money": gym.spaces.Box(low=0, high=MAX_MONEY, shape=(1,), dtype=np.int32),
            "self_position": gym.spaces.Discrete(NUM_CASE),
            "self_properties": gym.spaces.MultiBinary(NUM_PROPERTIES),
            "others_money": gym.spaces.Box(low=0, high=MAX_MONEY, shape=(3,), dtype=np.int32),  # Max 3 other players
            "active_players": gym.spaces.MultiBinary(4),
            "others_properties": gym.spaces.Box(
                low=0,
                high=5,  # 0-5 houses/hotel
                shape=(3, NUM_PROPERTIES),
                dtype=np.int8
            ),  # 3x28 matrix
            "others_positions": gym.spaces.Box(low=0, high=NUM_CASE, shape=(3,), dtype=np.int32),
            "self_houses": gym.spaces.Box(low=0, high=5, shape=(NUM_PROPERTIES,), dtype=np.int8),
            "all_properties": gym.spaces.Box(
                low=0.0,
                high=1.0,
                shape=(NUM_PROPERTIES, 11),
                dtype=np.float32
            ),
            "others_houses": gym.spaces.Box(
                low=0,
                high=5,
                shape=(3, NUM_PROPERTIES),
                dtype=np.int8
            ),
            "action_masks": gym.spaces.Dict({
                "mortgage": gym.spaces.MultiBinary(NUM_PROPERTIES),
                "build": gym.spaces.MultiBinary(NUM_PROPERTIES),
                "can_trade": gym.spaces.MultiBinary(1)
            }),
        })
        # Track active player for multi-agent mode
        self.current_player_idx = 0

        self.action_space = gym.spaces.Dict({
            "action_type": gym.spaces.Discrete(5),  # 0-4 corresponding to choices
            "property_idx": gym.spaces.Discrete(NUM_PROPERTIES),  # For property-related actions,
            # whether to buy property or not, choosing property to build on --> int
            "trade_partner": gym.spaces.Discrete(3),  # Index of other players
            "trade_amount": gym.spaces.Box(low=0, high=MAX_MONEY, shape=(1,), dtype=np.int32)  # For auctions/trades
        })

    def _get_obs_for_player(self, player: Player):
        """
        Creates observation for a player, including data about other players and property stats.

        Args:
            player: The player for whom to generate the observation

        Returns:
            Dictionary containing the complete observation state
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
        active_players = [int(not p.bankrupt) for p in self.players]

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

            # Other players' state
            "others_money": others_money,
            "others_properties": others_properties,
            "others_positions": others_positions,
            "others_houses": others_houses,
            "active_players": np.array(active_players, dtype=np.int8),

            # Static property data (prices, rents, etc.)
            "all_properties": self.board.property_data,  # Directly from the Board
        }

    def _get_houses_vector(self, player: Player):
        """
        Creates a vector representing houses owned by the player for each property.

        Args:
            player: The player whose houses to count

        Returns:
            NumPy array of house counts (-1 if property not owned)
        """
        houses = np.zeros(NUM_PROPERTIES, dtype=np.int8)
        for idx, prop_name in enumerate(self.property_order):
            # Check if player owns the property
            if prop_name in player.properties:
                case = next(c for c in self.board.board if c["name"] == prop_name)
                houses[idx] = min(case.get("houses", 0), 5)
            else:
                houses[idx] = -1  # Marker for unowned property
        return houses

    def _properties_to_binary(self, player):
        """
        Converts player's properties to a binary vector.

        Args:
            player: The player whose properties to convert

        Returns:
            Binary vector where 1 indicates ownership
        """
        binary_vector = np.zeros(NUM_PROPERTIES, dtype=np.int8)
        for idx, prop_name in enumerate(self.property_order):
            if prop_name in player.properties:
                binary_vector[idx] = 1
        return binary_vector

    def _get_info(self):
        pass

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """
        Resets the environment to an initial state.

        Args:
            seed: Random seed for reproducibility
            options: Additional options for reset

        Returns:
            Initial observation for the current player
        """
        super().reset(seed=seed)
        observation = self._get_obs_for_player(self.players[self.current_player_idx])
        return observation
        pass

    def _select_mortgage_property(self, player: Player) -> str:
        """
        Automatically selects a property to mortgage based on agent policy.

        Args:
            player: The player who needs to mortgage

        Returns:
            Property name to mortgage

        Raises:
            InvalidAction: If no mortgageable property is available
        """
        mortgageable = [
            prop for prop in player.properties
            if not self._get_board_property(prop)["mortgaged"]
        ]

        # Here you would use agent policy to choose
        # For example, take the first mortgageable property
        if mortgageable:
            return mortgageable[0]
        raise self.InvalidAction("No mortgageable property available")

    def _validate_property_ownership(self, player: Player, property_name: str, owner: Player = None) -> dict:
        """
        Validates property ownership (reusable helper method).

        Args:
            player: The player to validate
            property_name: The property to check
            owner: Optional specific owner to check against

        Returns:
            Property dictionary if valid

        Raises:
            InvalidAction: If ownership validation fails
        """
        prop = self._get_board_property(property_name)
        if not prop:
            raise self.InvalidAction(f"Property {property_name} not found")
        if owner and property_name not in owner.properties:
            raise self.InvalidAction(f"{owner.name} does not own {property_name}")
        if property_name not in player.properties and not owner:
            raise self.InvalidAction(f"You do not own {property_name}")
        return prop

    def _select_build_property(self, player: Player) -> str:
        """
        Automatically selects a property to build on based on agent policy.

        Args:
            player: The player who wants to build

        Returns:
            Property name to build on

        Raises:
            InvalidAction: If no buildable property is available
        """
        buildable = []
        for prop in player.properties:
            case = self._get_board_property(prop)
            if case["type"] == "property" and not case["mortgaged"]:
                color_group = self._get_color_group(case["color_code"])
                if all(p in player.properties for p in color_group):
                    buildable.append(prop)

        # Prioritize properties with fewest houses
        if buildable:
            return min(buildable, key=lambda p: self._get_board_property(p)["houses"])
        raise self.InvalidAction("No buildable property available")

    def _cycle_to_next_player(self):
        """
        Cycles to the next non-bankrupt player.
        """
        for _ in range(len(self.players)):
            self.current_player_idx = (self.current_player_idx + 1) % len(self.players)
            if not self.players[self.current_player_idx].bankrupt:
                break

    def step(self, action):
        """
        Executes an action and updates the environment.

        Args:
            action: The action to execute

        Returns:
            Tuple of (observation, reward, done, info)
        """
        player = self.players[self.current_player_idx]
        reward = 0
        done = False
        info = {}

        try:
            if action == 0:  # Mortgage
                prop = self._select_mortgage_property(player)
                self._handle_mortgage(player, prop)

            elif action == 1:  # Build
                prop = self._select_build_property(player)
                self._handle_build(player, prop)

            elif action["action_type"] == 2:  # Trade money for property
                partner = self._get_other_players(player)[action["trade_partner"]]
                prop = self.property_order[action["property_idx"]]
                self._handle_trade(player, partner, prop, action["trade_amount"])

            elif action["action_type"] == 3:  # Trade property for property
                partner = self._get_other_players(player)[action["trade_partner"]]
                self._handle_property_swap(player, partner, action["property_idx"])

            elif action["action_type"] == 4:  # Do nothing
                pass

            reward = self._calculate_reward(player)

        except self.InvalidAction as e:
            reward = -10
            info["error"] = str(e)

        self._cycle_to_next_player()
        next_obs = self._get_obs_for_player(self.players[self.current_player_idx])

        return next_obs, reward, done, info

    def _handle_mortgage(self, player: Player, property_name: str):
        """
        Handles mortgage action using the helper method for validation.

        Args:
            player: Player mortgaging the property
            property_name: Property to mortgage

        Raises:
            InvalidAction: If property can't be mortgaged
        """
        prop = self._validate_property_ownership(player, property_name)

        if prop.get("mortgaged", False):
            raise self.InvalidAction(f"{property_name} is already mortgaged")

        prop["mortgaged"] = True
        player.receive(prop["hypothèque"])
        print(f"{player.name} has mortgaged {property_name}")

    def _handle_build(self, player: Player, property_name: str):
        """
        Handles building a house on a property.

        Args:
            player: Player building the house
            property_name: Property to build on

        Raises:
            InvalidAction: If building conditions are not met
        """
        prop = self._get_board_property(property_name)
        if not prop:
            raise InvalidAction(f"Property {property_name} not found")

        # Check for complete color group ownership
        color_group = self._get_color_group(prop["color_code"])
        if any(p not in player.properties for p in color_group):
            raise InvalidAction("Incomplete color group ownership")

        # Check funds
        house_cost = prop["price"] // 2
        if player.money < house_cost:
            raise InvalidAction("Insufficient funds")

        prop["houses"] = min(prop.get("houses", 0) + 1, 5)
        player.pay(house_cost)

    def _get_other_players(self, current_player: Player) -> List[Player]:
        """
        Returns a list of other active players.

        Args:
            current_player: The current player to exclude

        Returns:
            List of other non-bankrupt players
        """
        return [p for p in self.players if p != current_player and not p.bankrupt]

    @staticmethod
    def _handle_trade(buyer: Player, seller: Player, property_name: str, amount: int):
        """
        Handles money for property trade (static method).

        Args:
            buyer: Player buying the property
            seller: Player selling the property
            property_name: Property being traded
            amount: Transaction amount

        Raises:
            InvalidAction: If trade conditions are not met
        """
        if property_name not in seller.properties:
            raise Game.InvalidAction(f"{seller.name} does not own {property_name}")
        if buyer.money < amount:
            raise Game.InvalidAction(f"{buyer.name} doesn't have enough money")

        buyer.pay(amount)
        seller.receive(amount)
        seller.properties.remove(property_name)
        buyer.properties.append(property_name)

    def _handle_property_swap(self, player: Player, partner: Player, property_name: str):
        """
        Uses helper method for property swap validation.

        Args:
            player: First player in the swap
            partner: Second player in the swap
            property_name: Property to swap

        Raises:
            InvalidAction: If swap conditions are not met
        """
        player_prop = self._validate_property_ownership(player, property_name)
        partner_prop = self._validate_property_ownership(partner, property_name, owner=partner)

        if player_prop not in player.properties:
            raise InvalidAction("Invalid property")

        player.properties.remove(player_prop)
        partner.properties.append(player_prop)
        partner.properties.remove(partner_prop)
        player.properties.append(partner_prop)

    def _get_color_group(self, color_code: str) -> List[str]:
        """
        Returns all properties in a color group.

        Args:
            color_code: Color code to search for

        Returns:
            List of property names in the same color group
        """
        return [
            p["name"] for p in self.board.board
            if p.get("color_code") == color_code
               and p["type"] == "property"
        ]

    # Internal exception class
    class InvalidAction(Exception):
        """
        Custom exception for invalid player actions.
        """

        def __init__(self, message: str):
            super().__init__(message)
            self.message = message

    def _get_mortgageable_properties(self, player):
        """
        Returns properties that can be mortgaged.

        Args:
            player: Player whose properties to check

        Returns:
            List of mortgageable property names
        """
        return [prop for prop in player.properties
                if not self._get_board_property(prop)["mortgaged"]]

    def _get_buildable_properties(self, player):
        """
        Returns properties where houses can be built.

        Args:
            player: Player whose properties to check

        Returns:
            List of buildable property names
        """
        buildable = []
        for prop in player.properties:
            case = self._get_board_property(prop)
            if case["type"] == "property" and not case["mortgaged"]:
                color_group = self._get_color_group(case["color_code"])
                if all(p in player.properties for p in color_group):
                    buildable.append(prop)
        return buildable

    def _calculate_reward(self, player):
        """
        Reward function combining multiple factors.

        Args:
            player: Player to calculate reward for

        Returns:
            Calculated reward value
        """
        reward = 0

        # Reward for money
        reward += player.money * 0.01

        # Reward for properties
        reward += len(player.properties) * 5

        # Reward for built houses
        for prop in player.properties:
            case = self._get_board_property(prop)
            reward += case.get("houses", 0) * 10

        # Penalty for bankruptcy
        if player.bankrupt:
            reward -= 1000

        return reward

    @staticmethod
    def _initialize_players() -> List[Player]:
        """
        Initializes players for the game.

        Returns:
            List of Player objects
        """
        return [Player(name=f"Player {i + 1}") for i in range(4)]

    def _init_property_states(self):
        """
        Initializes house counts for all properties.
        """
        for case in self.board.board:
            if case["type"] == "property":
                case.setdefault("houses", 0)

    def start(self):
        """
        Main game loop. Continues until only one player remains (not bankrupt).
        Each player takes their turn in sequence.
        """
        round_number = 1
        while len(self.players) > 1:
            print(f"\n====== Round {round_number} ======")
            for player in self.players.copy():
                if player.bankrupt:
                    continue
                print(f"\n--- {player.name}'s turn ---")
                self._handle_player_turn(player)
                if len(self.players) == 1:
                    break
            round_number += 1

        if self.players:
            print(f"\nCongratulations, {self.players[0].name} won the game!")
        else:
            print("The game ended without a winner.")

    def _handle_player_turn(self, player: Player):
        """
        Handles a player's turn:
          - Player presses Enter to roll dice
          - Player rolls dice and moves on the board
          - The arrival space and corresponding action are applied

        Args:
            player: Player whose turn it is
        """
        if player.bankrupt:
            print(f"{player.name} is bankrupt and cannot play.")
            return

        input(f"{player.name}, press Enter to roll the dice...")
        dice_roll = self._roll_dice()
        print(f"{player.name} rolled {dice_roll}.")

        # Update player position
        player.position = self.board.move_player(player.position, dice_roll)
        current_case = self.board.get_case(player.position)
        print(f"{player.name} moves to '{current_case['name']}'.")

        # Process arrival space
        self._handle_case_action(player, current_case)

    @staticmethod
    def _roll_dice() -> int:
        """
        Rolls the dice.

        Returns:
            Total dice roll value
        """
        return random.randint(1, 12)

    def _handle_case_action(self, player: Player, case: dict):
        """
        Handles action based on the case type.

        Args:
            player: Player landing on the case
            case: Case dictionary with type and data
        """
        if case["type"] == "property":
            self._handle_property_case(player, case)
        elif case["type"] == "tax":
            self._handle_tax_case(player, case)
        elif case["type"] == "start":
            player.money += 200
            print(f"{player.name} receives 200€ for passing Start.")
        elif case["type"] == "community_chest":
            self._handle_action_case_community_chest(player, case)
        elif case["type"] == "chance":
            self._handle_action_case_chance(player, case)
        elif case["type"] == "go_to_jail":
            self._handle_action_case_jail(player, case)
        elif case["type"] == "free_parking":
            player.money += 200
            print(f"{player.name} receives 200€ on Free Parking.")
        elif case["type"] == "utility":
            # Specific logic for utilities (not implemented here)
            pass
        else:
            print(f"No action defined for type '{case['type']}'.")

    def _handle_property_case(self, player: Player, case: dict):
        """
        Handles landing on a property space.

        Args:
            player: Player landing on the property
            case: Property case dictionary
        """
        owner = Game.find_property_owner(self.players, case["name"])
        if not owner:
            # Offer player to buy the property
            choice = input(
                f"{player.name}, do you want to buy {case['name']} for {case['price']}€? (y/n) : ").strip().lower()
            if choice == "o":
                Game._handle_property_purchase(player, case)
            else:
                print(f"{player.name} declined to buy {case['name']}. Auction begins.")
                self.auction_property(case["name"], starting_bid=case["price"])
        else:
            self._handle_rent_payment(player, case, owner)

    @staticmethod
    def find_property_owner(players: List[Player], property_name: str) -> Optional[Player]:
        """
        Finds the owner of a property.

        Args:
            players: List of all players
            property_name: Property to find owner for

        Returns:
            Player who owns the property or None
        """
        return next((p for p in players if property_name in p.properties), None)

    @staticmethod
    def _handle_property_purchase(player: Player, case: dict):
        """
        Handles purchasing a property.

        Args:
            player: Player buying the property
            case: Property case dictionary
        """
        if player.buy_property(case["name"], case["price"]):
            print(f"✅ {player.name} bought {case['name']}! Balance: {player.money}€")
        else:
            print(f"❌ {player.name} cannot afford {case['name']}")

    def _handle_rent_payment(self, player: Player, case: dict, owner: Player):
        """
        Handles rent payment when landing on another player's property.

        Args:
            player: Player paying rent
            case: Property case dictionary
            owner: Owner of the property
        """
        if owner == player:
            print(f"🌟 {player.name} already owns this property")
            return

        if case["type"] == "station":
            station_count = sum(1 for prop in owner.properties if prop.startswith("Gare"))
            station_count = max(station_count, 1)
            base_rent = case.get("rent", 25)
            multiplier = 2 ** (station_count - 1)
            rent = base_rent * multiplier
            print(f"{owner.name} owns {station_count} station(s) - rent is now {rent}€.")
        else:
            rent = Game.calculate_rent(case)

        print(f"💸 Rent due to {owner.name}: {rent}€")
        if player.money < rent:
            self.action_in_game(player)
        if player.money < rent:
            self.handle_bankruptcy(player, creditor=owner)
            return

        player.pay(rent)
        owner.receive(rent)
        print(f"Balance {player.name}: {player.money}€ → {owner.name}: {owner.money}€")

    @staticmethod
    def calculate_rent(case: dict) -> int:
        """
        Calculates rent for a property based on houses.

        Args:
            case: Property case dictionary

        Returns:
            Calculated rent amount
        """
        try:
            houses = case.get("houses", 0)
            return {
                0: case["rent"],
                5: case["hotel"]
            }.get(houses, case.get(f"H{houses}", case["rent"]))
        except KeyError as e:
            print(f"Configuration error: {e}")
            return case["rent"]

    def _handle_tax_case(self, player: Player, case: dict):
        """
        Handles landing on a tax space.

        Args:
            player: Player landing on tax space
            case: Tax case dictionary
        """
        tax = case["price"]
        if player.money < tax:
            self.action_in_game(player)
        if player.money < tax:
            self.handle_bankruptcy(player)
            return
        player.pay(tax)
        print(f"⚖️ {player.name} pays {tax}€ in taxes. Balance: {player.money}€")

    def _handle_action_case_jail(self, player: Player, case: dict, jail_price: int = 50):
        """
        Handles jail-related actions.

        Args:
            player: Player in jail
            case: Jail case dictionary
            jail_price: Cost to get out of jail
        """
        if case["type"] == "go_to_jail":
            jail_position = self.board.get_position("Prison/Simple visite")
            player.position = jail_position
            print(f"{player.name} is sent to jail!")
            print("Choose an action:")
            print(f"1. Pay {jail_price}€ to get out immediately")
            print("2. Roll dice to try to get out")
            choice = input("Your choice: ").strip()
            if choice == "1":
                if player.money < jail_price:
                    self.handle_bankruptcy(player)
                    return
                else:
                    player.pay(jail_price)
                    dice_roll = self._roll_dice()
                    player.position = self.board.move_player(player.position, dice_roll)
                    current_case = self.board.get_case(player.position)
                    print(f"{player.name} advances {dice_roll} spaces and lands on {current_case['name']}.")
                    self._handle_case_action(player, current_case)
            elif choice == "2":
                if not hasattr(player, "jail_turns"):
                    player.jail_turns = 0
                die1 = random.randint(1, 6)
                die2 = random.randint(1, 6)
                print(f"{player.name} rolls: {die1} and {die2}.")
                if die1 == die2:
                    print(f"{player.name} rolled doubles and is released from jail!")
                    player.jail_turns = 0
                    movement = die1 + die2
                    player.position = self.board.move_player(player.position, movement)
                    current_case = self.board.get_case(player.position)
                    print(f"{player.name} advances {movement} spaces and lands on {current_case['name']}.")
                    self._handle_case_action(player, current_case)
                else:
                    player.jail_turns += 1
                    if player.jail_turns >= 3:
                        print(f"{player.name} didn't roll doubles in 3 attempts and is released from jail.")
                        player.jail_turns = 0
                        new_die1 = random.randint(1, 6)
                        new_die2 = random.randint(1, 6)
                        movement = new_die1 + new_die2
                        print(f"{player.name} rolls again: {new_die1} and {new_die2}.")
                        player.position = self.board.move_player(player.position, movement)
                        current_case = self.board.get_case(player.position)
                        print(f"{player.name} advances {movement} spaces and lands on {current_case['name']}.")
                        self._handle_case_action(player, current_case)
                    else:
                        print(f"{player.name} didn't roll doubles and stays in jail (attempt {player.jail_turns}/3).")
            else:
                print("Invalid choice. Player remains in jail for this turn.")

    def _handle_random_card_action(self, player: Player, actions: List[dict]):
        """
        Executes a random action from the provided list.
        Logic is common to Chance and Community Chest cards.

        Args:
            player: Player drawing the card
            actions: List of possible card actions
        """
        chosen_action = random.choice(actions)
        print(chosen_action["message"])
        action_type = chosen_action["type"]

        if action_type == "advance_to_go":
            player.position = self.board.get_position("Go")
            player.receive(chosen_action["amount"])
            print(
                f"{player.name} is now on Go and receives {chosen_action['amount']}€. New balance: {player.money}€.")
        elif action_type == "gain_money":
            player.receive(chosen_action["amount"])
            print(f"The new balance of {player.name} is {player.money}€.")
        elif action_type == "lose_money":
            if player.money < chosen_action["amount"]:
                self.action_in_game(player)
            if player.money < chosen_action["amount"]:
                self.handle_bankruptcy(player)
                return
            player.pay(chosen_action["amount"])
            print(f"The new balance of {player.name} is {player.money}€.")
        elif action_type == "advance":
            player.position = self.board.move_player(player.position, chosen_action["spaces"])
            current_case = self.board.get_case(player.position)
            print(f"{player.name} moves forward {chosen_action['spaces']} spaces and lands on {current_case['name']}.")
            self._handle_case_action(player, current_case)
        elif action_type == "go_to_jail":
            player.position = self.board.get_position("Go to Jail")
            print(f"{player.name} is sent to jail!")
        elif action_type == "nothing":
            print(f"No additional action for {player.name}.")

    def _handle_action_case_chance(self, player: Player, case: dict):
        if case["type"] == "chance":
            actions = [
                {
                    "type": "advance_to_go",
                    "amount": 200,
                    "message": f"{player.name} advances to Go and collects 200€!"
                },
                {
                    "type": "gain_money",
                    "amount": 50,
                    "message": f"{player.name} receives a dividend of 50€ from the bank."
                },
                {
                    "type": "lose_money",
                    "amount": 15,
                    "message": f"{player.name} must pay a fine of 15€ for speeding."
                },
                {
                    "type": "advance",
                    "spaces": 2,
                    "message": f"{player.name} moves forward 2 spaces."
                },
                {
                    "type": "go_to_jail",
                    "message": f"{player.name} goes directly to jail!"
                },
                {
                    "type": "nothing",
                    "message": f"No special action for {player.name} this time."
                }
            ]
            self._handle_random_card_action(player, actions)

    def _handle_action_case_community_chest(self, player: Player, case: dict):
        if case["type"] == "community_chest":
            actions = [
                {
                    "type": "gain_money",
                    "amount": 200,
                    "message": f"{player.name} receives 200€ from the community chest!"
                },
                {
                    "type": "lose_money",
                    "amount": 100,
                    "message": f"{player.name} must pay 100€ to the community chest."
                },
                {
                    "type": "advance",
                    "spaces": 3,
                    "message": f"{player.name} moves forward 3 spaces."
                },
                {
                    "type": "go_to_jail",
                    "message": f"{player.name} goes directly to jail!"
                },
                {
                    "type": "nothing",
                    "message": f"No special action for {player.name} this time."
                }
            ]
            self._handle_random_card_action(player, actions)

    def _get_board_property(self, property_name: str) -> Optional[dict]:
        for case in self.board.board:
            if case["name"] == property_name:
                return case
        return None

    def action_in_game(self, player: Player):
        """
        Allows the player to perform an in-game action to improve their financial situation.
        Options include:
          1. Mortgage a property
          2. Build a house
          3. Trade: Buy a property (money for property)
          4. Trade: Exchange a property for another property
          5. Quit (no action)
        """
        print("\n--- Available actions ---")
        print("1. Mortgage a property")
        print("2. Build a house")
        print("3. Trade: Buy a property from another player (money for property)")
        print("4. Trade: Exchange a property for another property")
        print("5. Quit (no action)")
        choice = input("Your choice: ").strip()

        if choice == "1":
            eligible_props = []
            for prop in player.properties:
                board_prop = self._get_board_property(prop)
                if board_prop is not None and board_prop["type"] in ["property", "station", "utility"]:
                    if not board_prop.get("mortgaged", False):
                        eligible_props.append(prop)
            if not eligible_props:
                print("You have no properties eligible for mortgage.")
                return

            print("\nProperties eligible for mortgage:")
            for i, prop in enumerate(eligible_props, start=1):
                board_prop = self._get_board_property(prop)
                print(f"{i}. {prop} (Mortgage value: {board_prop['mortgage']}€)")
            selection = input("Select the property to mortgage (number): ").strip()
            try:
                sel = int(selection)
                if sel < 1 or sel > len(eligible_props):
                    print("Invalid selection.")
                    return
                chosen_prop = eligible_props[sel - 1]
                board_prop = self._get_board_property(chosen_prop)
                board_prop["mortgaged"] = True
                mortgage_value = board_prop["mortgage"]
                player.receive(mortgage_value)
                print(
                    f"{player.name} mortgaged {chosen_prop} and receives {mortgage_value}€. New balance: {player.money}€.")
            except ValueError:
                print("Invalid input.")

        elif choice == "2":
            eligible_props = []
            for prop in player.properties:
                board_prop = self._get_board_property(prop)
                if board_prop is not None and board_prop["type"] == "property":
                    if not board_prop.get("mortgaged", False):
                        eligible_props.append(prop)
            if not eligible_props:
                print("You have no properties eligible for building houses.")
                return

            print("\nProperties eligible for building a house:")
            for i, prop in enumerate(eligible_props, start=1):
                board_prop = self._get_board_property(prop)
                houses = board_prop.get("houses", 0)
                print(f"{i}. {prop} (Current houses: {houses})")
            selection = input("Select the property to build a house on (number): ").strip()
            try:
                sel = int(selection)
                if sel < 1 or sel > len(eligible_props):
                    print("Invalid selection.")
                    return
                chosen_prop = eligible_props[sel - 1]
                board_prop = self._get_board_property(chosen_prop)
                # Check full ownership of the color group
                if board_prop["type"] == "property":
                    color_group = [
                        p["name"] for p in self.board.board
                        if p.get("color_code") == board_prop["color_code"]
                           and p["type"] == "property"
                    ]
                    missing = [prop for prop in color_group if prop not in player.properties]

                    if missing:
                        print(
                            f"❌ Cannot build! You must own the entire {board_prop['color_code'].upper()} group.")
                        print(f"Missing properties: {', '.join(missing)}")
                        return
                current_houses = board_prop.get("houses", 0)
                if current_houses >= 4:
                    print(f"You already have the maximum number of houses on {chosen_prop} (4 max).")
                    return
                house_cost = int(board_prop["price"] / 2)
                if player.money < house_cost:
                    print(
                        f"{player.name} does not have enough money to build a house on {chosen_prop} (cost: {house_cost}€).")
                    return
                player.pay(house_cost)
                board_prop["houses"] = current_houses + 1
                print(f"A house has been built on {chosen_prop}. Houses now: {board_prop['houses']}.")
                print(f"New balance of {player.name}: {player.money}€.")
            except ValueError:
                print("Invalid input.")

        elif choice == "3":
            # Trade: Buying a property from another player (money for property)
            seller_name = input("Enter the name of the selling player: ").strip()
            seller = next((p for p in self.players if p.name.lower() == seller_name.lower() and p != player), None)
            if not seller:
                print("Player not found or you cannot trade with yourself.")
                return
            property_name = input(f"{seller.name}, enter the name of the property you want to sell: ").strip()
            Game.trade_action_money_to_card(player, seller, property_name)

        elif choice == "4":
            # Trade: Exchanging one property for another property
            partner_name = input("Enter the name of the player to trade with: ").strip()
            partner = next((p for p in self.players if p.name.lower() == partner_name.lower() and p != player), None)
            if not partner:
                print("Player not found or you cannot trade with yourself.")
                return
            Game.trade_action_card_to_card(player, partner)

        elif choice == "5":
            print("No action taken.")
        else:
            print("Invalid choice.")

    @staticmethod
    def trade_action_money_to_card(buyer: Player, seller: Player, property_name: str):
        if property_name not in seller.properties:
            print(f"Error: {seller.name} does not own the property {property_name}.")
            return

        amount_input = input(
            f"{buyer.name}, how much would you like to pay to buy {property_name} from {seller.name}?\nAmount: "
        )
        if not amount_input.isnumeric():
            print("Error: please enter a valid number.")
            return

        amount = int(amount_input)

        if buyer.money < amount:
            print(f"Error: {buyer.name} does not have enough money to pay {amount}€.")
            return

        print(f"{buyer.name} pays {amount}€ to {seller.name} to purchase {property_name}.")
        buyer.pay(amount)
        seller.receive(amount)

        seller.properties.remove(property_name)
        buyer.properties.append(property_name)
        print(f"Transaction successful! {buyer.name} now owns {property_name}.")
        print(f"Balance of {buyer.name}: {buyer.money}€")
        print(f"Balance of {seller.name}: {seller.money}€")

    @staticmethod
    def trade_action_card_to_card(buyer: Player, seller: Player):
        buyer_property = input(
            f"{buyer.name}, enter the name of the property you want to offer: "
        ).strip()
        if not buyer_property:
            print("Error: you must enter a valid property name.")
            return

        if buyer_property not in buyer.properties:
            print(f"Error: {buyer.name} does not own '{buyer_property}'.")
            return

        seller_property = input(
            f"{seller.name}, enter the name of the property you want to offer in exchange for '{buyer_property}': "
        ).strip()
        if not seller_property:
            print("Error: you must enter a valid property name.")
            return

        if seller_property not in seller.properties:
            print(f"Error: {seller.name} does not own '{seller_property}'.")
            return

        print(
            f"Proposed trade: {buyer.name} offers '{buyer_property}' in exchange for '{seller_property}' from {seller.name}.")

        buyer.properties.remove(buyer_property)
        seller.properties.append(buyer_property)

        seller.properties.remove(seller_property)
        buyer.properties.append(seller_property)

        print("Trade successful!")
        print(f"{buyer.name} now owns: {buyer.properties}")
        print(f"{seller.name} now owns: {seller.properties}")

    def handle_bankruptcy(self, player: Player, creditor: Optional[Player] = None):
        """
        Manages a player's bankruptcy.
        - If a creditor is specified, they receive the remaining money and the player's properties.
        - Otherwise, the properties are returned to the market.
        The player is then removed from the game.
        """
        print(f"🚨 {player.name} is bankrupt!")
        if creditor:
            print(f"{player.name}'s assets are transferred to {creditor.name}.")
            creditor.receive(player.money)
            for prop in player.properties:
                creditor.properties.append(prop)
        else:
            print(f"{player.name}'s properties are returned to the market.")

        player.money = 0
        player.properties = []
        player.bankrupt = True

        if player in self.players:
            self.players.remove(player)
        print(f"{player.name} has been removed from the game.")

    def auction_property(self, property_name: str, starting_bid: int = 0):
        """
        Organizes an auction for the property named property_name.
        All non-bankrupt players participate.
        """
        print(f"\nStarting auction for {property_name} (starting bid: {starting_bid}€)")
        eligible_players = [p for p in self.players if not p.bankrupt]
        if not eligible_players:
            print("No players are eligible to participate in the auction.")
            return

        current_bid = starting_bid
        highest_bidder = None
        active_bidders = eligible_players.copy()

        while len(active_bidders) > 1:
            any_bid_made = False
            for player in active_bidders.copy():
                print(f"\n{player.name}, the current bid is {current_bid}€.")
                bid_str = input(f"{player.name}, enter your bid (or press Enter to pass): ").strip()
                if bid_str == "":
                    print(f"{player.name} passes and is out of this auction.")
                    active_bidders.remove(player)
                else:
                    try:
                        bid = int(bid_str)
                        if bid <= current_bid:
                            print("Your bid must be higher than the current bid.")
                        elif bid > player.money:
                            print("You don’t have enough money to make that bid.")
                        else:
                            current_bid = bid
                            highest_bidder = player
                            any_bid_made = True
                            print(f"{player.name} bids {bid}€.")
                    except ValueError:
                        print("Invalid input. You skip this round.")
                        active_bidders.remove(player)
            if not any_bid_made:
                break

        if highest_bidder is not None:
            print(f"\n{highest_bidder.name} wins the auction for {property_name} with a bid of {current_bid}€.")
            highest_bidder.pay(current_bid)
            highest_bidder.properties.append(property_name)
            print(f"{highest_bidder.name}’s new balance is {highest_bidder.money}€.")
        else:
            print("No bids were made for this auction.")


gym.register(
    id="MyMonopolyEnv-v0",
    entry_point="Monopoly_AI.environement.game:Game",
)
if __name__ == "__main__":
    game = Game()
    game.start()