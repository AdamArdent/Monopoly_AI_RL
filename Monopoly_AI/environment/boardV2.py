import numpy as np
from typing import List, Dict, Any
class Board:
    def __init__(self):
        print("Initializing Board")
        self.board = [
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
        self._init_property_data()

        self.property_order = [
            case['name'] for case in self.board
            if case['type'] in ['property', 'station', 'utility']
        ]

    def get_position(self, property_name):
        """Retourne l'index d'une propriété par son nom"""
        for index, case in enumerate(self.board):
            if case["name"] == property_name:
                return index
        return -1

    def get_case(self, index):
        """Retourne la case à un indice donné"""
        if 0 <= index < len(self.board):
            return self.board[index]
        else:
            raise ValueError("Indice de case invalide.")

    def move_player(self, current_position, dice_roll):
        """Déplace un joueur sur le plateau"""
        current_position = (current_position + dice_roll) % len(self.board)
        return current_position

    def _init_property_data(self):
        """Initialise les données brutes des propriétés"""
        self.property_order = []
        self.property_data = []

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
            'utility': 9
        }

        for case in self.board:
            if case["type"] in ['property', 'station', 'utility']:
                self.property_order.append(case["name"])
                self.property_data.append([
                    # Données brutes
                    case["price"],
                    case.get("rent", 0),
                    case.get("H1", 0),
                    case.get("H2", case.get("rent", 0)),
                    case.get("H3", case.get("rent", 0)),
                    case.get("H4", case.get("rent", 0)),
                    case.get("hotel", case.get("rent", 0)),
                    case.get("hypothèque", 0),
                    case.get("house_cost", case["price"] // 2),
                    color_mapping.get(case.get("color_code", "special"), 10),
                    len([c for c in self.board if c.get("color_code") == case.get("color_code")])
                ])

        # Conversion simple en numpy array
        self.property_data = np.array(self.property_data, dtype=np.int32)

    def get_property(self, name: str) -> Dict[str, Any]:
        return next((c for c in self.board if c["name"] == name), None)

    def get_color_group(self, color_code: str) -> List[Dict[str, Any]]:
        return [c for c in self.board if c.get("color_code") == color_code]