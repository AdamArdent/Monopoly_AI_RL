import numpy as np  # Import the numpy library, commonly used for numerical operations.
from typing import List, Dict, Any # Import typing hints for better code readability and maintainability.

class Board:
    """
    Represents the game board for a Monopoly-like game.

    Attributes:
        board (List[Dict[str, Any]]): A list of dictionaries, where each dictionary represents a square on the board
                                       with its properties (name, type, price, rent, etc.).
        property_order (List[str]): A list containing the names of all purchasable properties, stations, and utilities
                                    in their order on the board.
        property_data (np.ndarray): A numpy array containing structured data for purchasable properties,
                                    including price, rent levels, mortgage value, house cost, color ID, and color group size.
        property_max (np.ndarray): A numpy array containing the maximum values for each data point in `property_data`.
        property_min (np.ndarray): A numpy array containing the minimum values for each data point in `property_data`.
    """
    def __init__(self):
        """
        Initializes the game board with all squares and their associated data.
        """
        print("Initializing Board") # Console output to indicate board initialization.
        self.board = [
            # Define each square on the board as a dictionary
            # 'name': Name of the square
            # 'type': Type of square (start, property, community_chest, tax, station, jail, free_parking, go_to_jail, utility, chance)
            # 'price': Purchase price (0 for non-purchasable squares)
            # 'rent': Base rent or tax amount (0 for non-rentable squares)
            # 'hypothèque': Mortgage value (0 for non-mortgageable squares)
            # Additional keys for properties: color_code, H1-H4 (rent with houses), hotel (rent with hotel)

            {"name": "Départ", "type": "start", "price": 0, "rent": 0, "hypothèque": 0},
            {"name": "Boulevard de Belleville", "type": "property", "price": 60, "rent": 2, "color_code": "brown",
             "H1": 10, "H2": 30, "H3": 90, "H4": 160, "hotel": 250, "hypothèque": 30},
            {"name": "Caisse de communauté", "type": "community_chest", "price": 0, "rent": 0, "hypothèque": 0},
            {"name": "Rue Lecourbe", "type": "property", "price": 60, "rent": 4, "color_code": "brown",
             "H1": 20, "H2": 60, "H3": 180, "H4": 320, "hotel": 450, "hypothèque": 30},
            {"name": "Impôt sur le revenu", "type": "tax", "price": 200, "rent": 0, "hypothèque": 0},
            {"name": "Gare Montparnasse", "type": "station", "price": 200, "rent": 25, "color_code": "station",
             "hypothèque": 100},
            {"name": "Rue de Vaugirard", "type": "property", "price": 100, "rent": 6, "color_code": "light_blue",
             "H1": 30, "H2": 90, "H3": 270, "H4": 400, "hotel": 550, "hypothèque": 50},
            {"name": "Chance", "type": "chance", "price": 0, "rent": 0, "hypothèque": 0},
            {"name": "Rue de Courcelles", "type": "property", "price": 100, "rent": 6, "color_code": "light_blue",
             "H1": 30, "H2": 90, "H3": 270, "H4": 400, "hotel": 550, "hypothèque": 50},
            {"name": "Avenue de la République", "type": "property", "price": 120, "rent": 8, "color_code": "light_blue",
             "H1": 40, "H2": 100, "H3": 300, "H4": 450, "hotel": 600, "hypothèque": 60},
            {"name": "Prison/Simple visite", "type": "jail", "price": 0, "rent": 0, "hypothèque": 0},
            {"name": "Boulevard de la Villette", "type": "property", "price": 140, "rent": 10, "color_code": "pink",
             "H1": 50, "H2": 150, "H3": 450, "H4": 625, "hotel": 750, "hypothèque": 70},
            {"name": "Compagnie d'électricité", "type": "utility", "price": 150, "rent": 4, "hypothèque": 75},
            {"name": "Avenue de Neuilly", "type": "property", "price": 140, "rent": 10, "color_code": "pink",
             "H1": 50, "H2": 150, "H3": 450, "H4": 625, "hotel": 750, "hypothèque": 70},
            {"name": "Rue de Paradis", "type": "property", "price": 160, "rent": 12, "color_code": "pink",
             "H1": 60, "H2": 180, "H3": 500, "H4": 700, "hotel": 900, "hypothèque": 80},
            {"name": "Gare de Lyon", "type": "station", "price": 200, "rent": 25, "color_code": "station",
             "hypothèque": 100},
            {"name": "Avenue Mozart", "type": "property", "price": 180, "rent": 14, "color_code": "orange",
             "H1": 70, "H2": 200, "H3": 550, "H4": 750, "hotel": 950, "hypothèque": 90},
            {"name": "Caisse de communauté", "type": "community_chest", "price": 0, "rent": 0, "hypothèque": 0},
            {"name": "Boulevard Saint-Michel", "type": "property", "price": 180, "rent": 14, "color_code": "orange",
             "H1": 70, "H2": 200, "H3": 550, "H4": 750, "hotel": 950, "hypothèque": 90},
            {"name": "Place Pigalle", "type": "property", "price": 200, "rent": 16, "color_code": "orange",
             "H1": 80, "H2": 220, "H3": 600, "H4": 800, "hotel": 1000, "hypothèque": 100},
            {"name": "Parc gratuit", "type": "free_parking", "price": 0, "rent": 0, "hypothèque": 0},
            {"name": "Avenue Matignon", "type": "property", "price": 220, "rent": 18, "color_code": "red",
             "H1": 90, "H2": 250, "H3": 700, "H4": 875, "hotel": 1050, "hypothèque": 110},
            {"name": "Chance", "type": "chance", "price": 0, "rent": 0, "hypothèque": 0},
            {"name": "Boulevard Malesherbes", "type": "property", "price": 220, "rent": 18, "color_code": "red",
             "H1": 90, "H2": 250, "H3": 700, "H4": 875, "hotel": 1050, "hypothèque": 110},
            {"name": "Avenue Henri-Martin", "type": "property", "price": 240, "rent": 20, "color_code": "red",
             "H1": 100, "H2": 300, "H3": 750, "H4": 925, "hotel": 1100, "hypothèque": 120},
            {"name": "Gare du Nord", "type": "station", "price": 200, "rent": 25, "color_code": "station",
             "hypothèque": 100},
            {"name": "Faubourg Saint-Honoré", "type": "property", "price": 260, "rent": 22, "color_code": "yellow",
             "H1": 110, "H2": 330, "H3": 800, "H4": 975, "hotel": 1150, "hypothèque": 130},
            {"name": "Place de la Bourse", "type": "property", "price": 260, "rent": 22, "color_code": "yellow",
             "H1": 110, "H2": 330, "H3": 800, "H4": 975, "hotel": 1150, "hypothèque": 130},
            {"name": "Compagnie des eaux", "type": "utility", "price": 150, "rent": 4, "hypothèque": 75},
            {"name": "Rue La Fayette", "type": "property", "price": 280, "rent": 24, "color_code": "yellow",
             "H1": 120, "H2": 360, "H3": 850, "H4": 1025, "hotel": 1200, "hypothèque": 140},
            {"name": "Allez en prison", "type": "go_to_jail", "price": 0, "rent": 0, "hypothèque": 0},
            {"name": "Avenue de Breteuil", "type": "property", "price": 300, "rent": 26, "color_code": "green",
             "H1": 130, "H2": 390, "H3": 900, "H4": 1100, "hotel": 1275, "hypothèque": 150},
            {"name": "Avenue Foch", "type": "property", "price": 300, "rent": 26, "color_code": "green",
             "H1": 130, "H2": 390, "H3": 900, "H4": 1100, "hotel": 1275, "hypothèque": 150},
            {"name": "Caisse de communauté", "type": "community_chest", "price": 0, "rent": 0, "hypothèque": 0},
            {"name": "Boulevard des Capucines", "type": "property", "price": 320, "rent": 28, "color_code": "green",
             "H1": 150, "H2": 450, "H3": 1000, "H4": 1200, "hotel": 1400, "hypothèque": 160},
            {"name": "Gare Saint-Lazare", "type": "station", "price": 200, "rent": 25, "color_code": "station",
             "hypothèque": 100},
            {"name": "Chance", "type": "chance", "price": 0, "rent": 0, "hypothèque": 0},
            {"name": "Avenue des Champs-Élysées", "type": "property", "price": 350, "rent": 35,
             "color_code": "dark_blue",
             "H1": 175, "H2": 500, "H3": 1100, "H4": 1300, "hotel": 1500, "hypothèque": 175},
            {"name": "Taxe de luxe", "type": "tax", "price": 100, "rent": 0, "hypothèque": 0},
            {"name": "Rue de la Paix", "type": "property", "price": 400, "rent": 50, "color_code": "dark_blue",
             "H1": 200, "H2": 600, "H3": 1400, "H4": 1700, "hotel": 2000, "hypothèque": 200}
        ]
        self._init_property_data() # Call helper method to initialize structured property data.

        # This part seems redundant as it's also done in _init_property_data,
        self.property_order = [
            case['name'] for case in self.board
            if case['type'] in ['property', 'station', 'utility']
        ]

    def get_position(self, property_name: str) -> int:
        """
        Returns the index of a square by its name.

        Args:
            property_name (str): The name of the square to find.

        Returns:
            int: The index of the square in the board list, or -1 if not found.
        """
        for index, case in enumerate(self.board):
            if case["name"] == property_name:
                return index
        return -1 # Return -1 if the property name is not found.

    def get_case(self, index: int) -> Dict[str, Any]:
        """
        Returns the square at a given index.

        Args:
            index (int): The index of the square to retrieve.

        Returns:
            Dict[str, Any]: The dictionary representing the square.

        Raises:
            ValueError: If the provided index is out of the valid range for the board.
        """
        if 0 <= index < len(self.board):
            return self.board[index]
        else:
            raise ValueError("Invalid case index.") # Raise error for out-of-bounds index.

    def move_player(self, current_position: int, dice_roll: int) -> int:
        """
        Calculates the new position of a player after a dice roll, wrapping around the board.

        Args:
            current_position (int): The player's current position (index) on the board.
            dice_roll (int): The result of the dice roll.

        Returns:
            int: The player's new position (index) after the move.
        """
        # Calculate new position, using modulo to wrap around the board length
        new_position = (current_position + dice_roll) % len(self.board)
        return new_position

    def get_property(self, name: str) -> Dict[str, Any] | None:
        """
        Finds and returns a specific property, station, or utility by its name.

        Args:
            name (str): The name of the property, station, or utility.

        Returns:
            Dict[str, Any] | None: The dictionary representing the square if found, otherwise None.
        """
        # Use a generator expression and next() to find the first matching square
        return next((c for c in self.board if c["name"] == name), None)

    def get_color_group(self, color_code: str) -> List[Dict[str, Any]]:
        """
        Retrieves all properties belonging to a specific color group or type (like stations).

        Args:
            color_code (str): The color code or type identifier (e.g., 'brown', 'station').

        Returns:
            List[Dict[str, Any]]: A list of dictionaries for squares matching the color code.
        """
        # Filter the board list to find all squares with the given color code
        return [c for c in self.board if c.get("color_code") == color_code]

    def _init_property_data(self):
        """
        Initializes structured data for properties, stations, and utilities into a numpy array
        for easier numerical processing.
        """
        self.property_order = [] # List to store names of purchasable properties in order
        self.property_data = [] # List to temporarily store property data before converting to numpy

        # Mapping of color codes to integer IDs
        color_mapping = {
            'brown': 0,
            'light_blue': 1,
            'pink': 2,
            'orange': 3,
            'red': 4,
            'yellow': 5,
            'green': 6,
            'dark_blue': 7,
            'station': 8,
            'utility': 9,
            "special": 10 # Default ID for squares without standard color groups
        }

        for case in self.board:
            # Process only purchasable types (properties, stations, utilities)
            if case["type"] in ['property', 'station', 'utility']:
                self.property_order.append(case["name"])
                # Append a list of numerical data points for the current property
                self.property_data.append([
                    # Basic Data / Données de base
                    case["price"],  # Purchase Price / Prix d'achat
                    case.get("rent", 0),  # Base Rent / Loyer de base
                    case.get("H1", 0),  # Rent with 1 House / Loyer avec 1 maison
                    case.get("H2", case.get("rent", 0)),  # Rent with 2 Houses / Loyer avec 2 maisons (default to base rent if not specified)
                    case.get("H3", case.get("rent", 0)),  # Rent with 3 Houses / Loyer avec 3 maisons
                    case.get("H4", case.get("rent", 0)),  # Rent with 4 Houses / Loyer avec 4 maisons
                    case.get("hotel", case.get("rent", 0)),  # Rent with Hotel / Loyer avec hôtel
                    case.get("hypothèque", 0),  # Mortgage Value / Valeur hypothécaire
                    # House cost is often half the property price, but can be specified.
                    case.get("house_cost", case["price"] // 2),  # Cost per House / Coût par maison
                    color_mapping.get(case.get("color_code", "special"), 10),  # Color Group ID / ID couleur
                    # Size of the color group the property belongs to
                    len([c for c in self.board if c.get("color_code") == case.get("color_code") if case.get("color_code") is not None])
                    # Color Group Size / Taille du groupe de couleur (Only count properties within the same color group)
                ])
        # Convert the list of property data into a numpy array for efficient processing
        self.property_data = np.array(self.property_data, dtype=np.int32)

        # Calculate global maximum and minimum values for each data column in property_data
        self.property_max = np.max(self.property_data, axis=0) # Global Maxima / Maxima globaux
        self.property_min = np.min(self.property_data, axis=0) # Global Minima / Minima globaux