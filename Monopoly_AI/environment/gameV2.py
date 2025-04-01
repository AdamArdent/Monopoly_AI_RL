from typing import List, Optional
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from environment.boardV2 import Board
from environment.player import Player

NUM_PROPERTIES = 28
MAX_MONEY = 10000
NUM_CASE = 40


class MonopolyEnv(gym.Env):
    """Environnement Monopoly pour l'apprentissage par renforcement."""

    class InvalidAction(Exception):
        """Exception pour les actions invalides."""

        def __init__(self, message: str):
            super().__init__(message)
            self.message = message

    def __init__(self):
        super().__init__()
        super().__init__()
        self.players = self._initialize_players()
        self.board = Board()
        self.property_order = self.board.property_order

        # Définition des espaces
        self.observation_space = self._create_observation_space()
        self.action_space = self._create_action_space()
        self.current_player_idx = 0

    @staticmethod
    def _create_observation_space():
        """Espace d'observation avec données brutes"""
        return spaces.Dict({
            "self_money": spaces.Box(0, MAX_MONEY, (1,), dtype=np.int32),
            "self_position": spaces.Discrete(NUM_CASE),
            "self_properties": spaces.MultiBinary(NUM_PROPERTIES),
            "others_money": spaces.Box(0, MAX_MONEY, (3,), dtype=np.int32),
            "active_players": spaces.MultiBinary(4),
            "others_properties": spaces.Box(0, 1, (3, NUM_PROPERTIES), dtype=np.int8),
            "others_positions": spaces.Box(0, NUM_CASE, (3,), dtype=np.int32),
            "self_houses": spaces.Box(0, 5, (NUM_PROPERTIES,), dtype=np.int8),
            "all_properties": spaces.Box(
                low=0,
                high=MAX_MONEY,
                shape=(NUM_PROPERTIES, 11),
                dtype=np.int32
            ),
            "action_masks": spaces.Dict({
                "mortgage": spaces.MultiBinary(NUM_PROPERTIES),
                "build": spaces.MultiBinary(NUM_PROPERTIES),
                "can_trade": spaces.MultiBinary(1)
            }),
        })
    @staticmethod
    def _create_action_space():
        """Espace d'action inchangé"""
        return spaces.Dict({
            "action_type": spaces.Discrete(5),
            "property_idx": spaces.Discrete(NUM_PROPERTIES),
            "trade_partner": spaces.Discrete(3),
            "trade_amount": spaces.Box(0, MAX_MONEY, (1,), dtype=np.int32)
        })

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        return self._get_obs_for_player(self.players[self.current_player_idx])

    def step(self, action):
        player = self.players[self.current_player_idx]
        reward = 0
        done = False
        info = {}

        try:
            if action["action_type"] == 0:
                self._handle_mortgage(player, action["property_idx"])
            elif action["action_type"] == 1:
                self._handle_build(player, action["property_idx"])
            elif action["action_type"] == 2:
                self._handle_trade(player, action)
            elif action["action_type"] == 3:
                self._handle_property_swap(player, action)

            reward = self._calculate_reward(player)
            done = self._check_game_over()

        except self.InvalidAction as e:
            reward = -10
            info["error"] = str(e)

        self._cycle_to_next_player()
        return self._get_obs_for_player(self.players[self.current_player_idx]), reward, done, info

    def _get_obs_for_player(self, player: Player):
        """Génère l'observation pour un joueur donné."""
        other_players = [p for p in self.players if p != player and not p.bankrupt]

        return {
            "self_money": np.array([player.money], dtype=np.int32),
            "self_position": player.position,
            "self_properties": self._properties_to_binary(player),
            "self_houses": self._get_houses_vector(player),
            "action_masks": self._get_action_masks(player),
            "others_money": self._get_others_money(other_players),
            "others_properties": self._get_others_properties(other_players),
            "others_positions": self._get_others_positions(other_players),
            "active_players": self._get_active_players(),
            "all_properties": self.board.property_data.astype(np.int32),
        }

    def _get_action_masks(self, player):
        return {
            "mortgage": np.array([
                int(prop in self._get_mortgageable_properties(player))
                for prop in self.property_order
            ], dtype=np.int8),
            "build": np.array([
                int(prop in self._get_buildable_properties(player))
                for prop in self.property_order
            ], dtype=np.int8),
            "can_trade": np.array([int(len(player.properties) > 0)], dtype=np.int8)
        }

    def _handle_mortgage(self, player: Player, property_idx: int):
        """Gère l'hypothèque d'une propriété."""
        prop_name = self.property_order[property_idx]
        prop = self._validate_property_ownership(player, prop_name)

        if prop["mortgaged"]:
            raise self.InvalidAction(f"{prop_name} déjà hypothéquée")

        prop["mortgaged"] = True
        player.receive(prop["hypothèque"])

    def _handle_build(self, player: Player, property_idx: int):
        """Gère la construction de maisons."""
        prop_name = self.property_order[property_idx]
        prop = self._validate_property_ownership(player, prop_name)

        if prop["type"] != "property":
            raise self.InvalidAction("Type de propriété invalide")

        if not self._has_full_color_group(player, prop["color_code"]):
            raise self.InvalidAction("Groupe de couleur incomplet")

        house_cost = prop["price"] // 2
        if player.money < house_cost:
            raise self.InvalidAction("Fonds insuffisants")

        prop["houses"] = min(prop.get("houses", 0) + 1, 5)
        player.pay(house_cost)

    def _validate_property_ownership(self, player: Player, property_name: str):
        """Valide la possession d'une propriété."""
        if property_name not in player.properties:
            raise self.InvalidAction(f"Joueur ne possède pas {property_name}")
        return self.board.get_property(property_name)

    def _has_full_color_group(self, player: Player, color_code: str):
        """Vérifie la possession complète d'un groupe de couleur."""
        return all(
            p["name"] in player.properties
            for p in self.board.get_color_group(color_code)
        )

    def _calculate_reward(self, player: Player):
        """Calcule la récompense basée sur l'état du joueur."""
        reward = player.money * 0.01
        reward += len(player.properties) * 5
        reward += sum(
            self.board.get_property(p).get("houses", 0) * 10
            for p in player.properties
        )
        return reward

    def _check_game_over(self):
        """Vérifie si la partie est terminée."""
        return sum(not p.bankrupt for p in self.players) <= 1

    def _cycle_to_next_player(self):
        """Passe au prochain joueur actif."""
        for _ in range(len(self.players)):
            self.current_player_idx = (self.current_player_idx + 1) % len(self.players)
            if not self.players[self.current_player_idx].bankrupt:
                break

    def _properties_to_binary(self, player: Player) -> np.ndarray:
        """Convertit les propriétés du joueur en vecteur binaire."""
        return np.array([
            int(prop in player.properties)
            for prop in self.property_order
        ], dtype=np.int8)

    def _get_houses_vector(self, player: Player) -> np.ndarray:
        """Retourne le vecteur des maisons/hôtels pour les propriétés."""
        houses = np.zeros(NUM_PROPERTIES, dtype=np.int8)
        for idx, prop_name in enumerate(self.property_order):
            prop = self.board.get_property(prop_name)
            if prop_name in player.properties and prop["type"] == "property":
                houses[idx] = min(prop.get("houses", 0), 5)
            else:
                houses[idx] = -1  # -1 pour les propriétés non possédées
        return houses

    @staticmethod
    def _get_others_money(other_players: List[Player]) -> np.ndarray:
        """Récupère l'argent des autres joueurs."""
        return np.array([p.money for p in other_players[:3]], dtype=np.int32)

    @staticmethod
    def _get_others_positions(other_players: List[Player]) -> np.ndarray:
        """Récupère les positions des autres joueurs."""
        return np.array([p.position for p in other_players[:3]], dtype=np.int32)

    @staticmethod
    def _execute_trade(buyer: Player, seller: Player, prop: str, amount: int):
        """Exécute la transaction commerciale."""
        buyer.pay(amount)
        seller.receive(amount)
        seller.properties.remove(prop)
        buyer.properties.append(prop)

    def _get_active_players(self) -> np.ndarray:
        """Génère le masque des joueurs actifs."""
        return np.array([
            int(not p.bankrupt)
            for p in self.players
        ], dtype=np.int8)

    def _get_others_properties(self, other_players: List[Player]) -> np.ndarray:
        """Matrice des propriétés des autres joueurs."""
        properties = np.zeros((3, NUM_PROPERTIES), dtype=np.int8)
        for i, p in enumerate(other_players[:3]):
            properties[i] = self._properties_to_binary(p)
        return properties

    def _get_others_houses(self, other_players: List[Player]) -> np.ndarray:
        """Matrice des maisons des autres joueurs."""
        houses = np.zeros((3, NUM_PROPERTIES), dtype=np.int8)
        for i, p in enumerate(other_players[:3]):
            houses[i] = self._get_houses_vector(p)
        return houses

    def _get_mortgageable_properties(self, player: Player) -> List[str]:
        """Liste des propriétés hypothécables."""
        return [
            prop for prop in player.properties
            if not self.board.get_property(prop)["mortgaged"]
        ]

    def _get_buildable_properties(self, player: Player) -> List[str]:
        """Liste des propriétés constructibles."""
        buildable = []
        for prop in player.properties:
            board_prop = self.board.get_property(prop)
            if (board_prop["type"] == "property" and
                    not board_prop["mortgaged"] and
                    self._has_full_color_group(player, board_prop["color_code"])):
                buildable.append(prop)
        return buildable

    def _get_color_group_properties(self, color_code: str) -> List[str]:
        """Récupère toutes les propriétés d'un groupe de couleur."""
        return [p["name"] for p in self.board.get_color_group(color_code)]

    def _handle_trade(self, player: Player, action: dict):
        """Gère les échanges entre joueurs."""
        partner = self._get_other_players(player)[action["trade_partner"]]
        prop_idx = action["property_idx"]
        amount = action["trade_amount"][0]
        prop_name = self.property_order[prop_idx]

        if prop_name not in partner.properties:
            raise self.InvalidAction("Le partenaire ne possède pas cette propriété")

        if player.money < amount:
            raise self.InvalidAction("Fonds insuffisants pour l'échange")

        self._execute_trade(player, partner, prop_name, amount)

    def _handle_property_swap(self, player: Player, action: dict):
        """Gère l'échange de propriété contre propriété."""
        partner = self._get_other_players(player)[action["trade_partner"]]
        prop_idx = action["property_idx"]
        player_prop = self.property_order[prop_idx]
        partner_prop = self.property_order[action.get("partner_prop_idx", 0)]

        if player_prop not in player.properties:
            raise self.InvalidAction("Propriété du joueur invalide")
        if partner_prop not in partner.properties:
            raise self.InvalidAction("Propriété du partenaire invalide")

        player.properties.remove(player_prop)
        partner.properties.append(player_prop)
        partner.properties.remove(partner_prop)
        player.properties.append(partner_prop)

    def _get_other_players(self, current_player: Player) -> List[Player]:
        """Retourne la liste des autres joueurs actifs."""
        return [p for p in self.players if p != current_player and not p.bankrupt]

    def get_state_for_ai(self, player: Player) -> dict:
        """Interface pour les systèmes d'IA externes."""
        return {
            'money': player.money,
            'position': player.position,
            'properties': self._properties_to_binary(player),
            'houses': self._get_houses_vector(player),
            'others': [{
                'money': p.money,
                'position': p.position,
                'properties': self._properties_to_binary(p)
            } for p in self._get_other_players(player)]
        }

    @staticmethod
    def _initialize_players() -> List[Player]:
        return [Player(name=f"Player {i + 1}") for i in range(4)]

    def _init_property_states(self):
        for case in self.board.board:
            if case["type"] == "property":
                case.setdefault("houses", 0)


# Enregistrement de l'environnement
gym.register(
    id="MonopolyEnv-v1",
    entry_point="environment.game:MonopolyEnv",
    kwargs={}
)