import random
from typing import List, Optional
from gym.error import InvalidAction
from environment.board import Board
from environment.player import Player
import gymnasium as gym
import random
import numpy as np
NUM_PROPERTIES = 28
MAX_MONEY = 10000
NUM_CASE = 40

class Game(gym.Env):
    def __init__(self):
        print("Initializing Game")
        self.players = self._initialize_players()
        self.board = Board()
        self._init_property_states()
        self.property_order = [
            case['name'] for case in self.board.board
            if case['type'] in ['property', 'station', 'utility']
        ]
        # Vérification de la cohérence avec NUM_PROPERTIES
        assert len(self.property_order) == NUM_PROPERTIES, "Incohérence dans les propriétés"
        # Récupération directe depuis le Board
        self.property_order = self.board.property_order
        self.property_data = self.board.property_data
        # Normalisation Min-Max
        self.property_data_norm = (self.board.property_data - self.board.property_min) / (self.board.property_max - self.board.property_min + 1e-8)
        # Définir l'espace d'observation enrichi
        self.observation_space = gym.spaces.Dict({
            "self_money": gym.spaces.Box(low=0, high=MAX_MONEY, shape=(1,), dtype=np.int32),
            "self_position": gym.spaces.Discrete(NUM_CASE),
            "self_properties": gym.spaces.MultiBinary(NUM_PROPERTIES),
            "others_money": gym.spaces.Box(low=0, high=MAX_MONEY, shape=(3,), dtype=np.int32),  # 3 autres joueurs max
            "active_players": gym.spaces.MultiBinary(4),
            "others_properties": gym.spaces.Box(
                low=0,
                high=5,  # 0-5 maisons/hôtel
                shape=(3, NUM_PROPERTIES),
                dtype=np.int8
            ),  # Matrice 3x28
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
        # Track le joueur actif pour le multi-agent
        self.current_player_idx = 0

        self.action_space = gym.spaces.Dict({
            "action_type": gym.spaces.Discrete(5),  # 0-4 correspondant aux choix
            "property_idx": gym.spaces.Discrete(NUM_PROPERTIES),  # Pour les actions liées aux propriétés,
            # veux acheter prop ou non, choisis la prop sur laquelle il veut build --> int
            "trade_partner": gym.spaces.Discrete(3),  # Index des autres joueurs
            "trade_amount": gym.spaces.Box(low=0, high=MAX_MONEY, shape=(1,), dtype=np.int32)#pour auction,
        })

    def _get_obs_for_player(self, player: Player):
        """Observation pour un joueur, incluant les données des autres et les stats des propriétés."""
        other_players = [p for p in self.players if p != player and not p.bankrupt]
        # Calcul des actions valides
        mortgageable = [int(prop in self._get_mortgageable_properties(player)) for prop in self.property_order]
        buildable = [int(prop in self._get_buildable_properties(player)) for prop in self.property_order]
        # Récupérer les données des autres joueurs
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
            # État personnel
            "self_money": np.array([player.money], dtype=np.int32),
            "self_position": player.position,
            "self_properties": self._properties_to_binary(player),
            "self_houses": self._get_houses_vector(player),
            "action_masks": {
                "mortgage": np.array(mortgageable, dtype=np.int8),
                "build": np.array(buildable, dtype=np.int8),
                "can_trade": np.array([int(len(player.properties) > 0)], dtype=np.int8)
            },

            # État des autres joueurs
            "others_money": others_money,
            "others_properties": others_properties,
            "others_positions": others_positions,
            "others_houses": others_houses,
            "active_players": np.array(active_players, dtype=np.int8),

            # Données statiques des propriétés (prix, loyers, etc.)
            "all_properties": self.board.property_data,  # Directement depuis le Board
        }
    def _get_houses_vector(self, player: Player):
        """Version améliorée avec vérification de propriété"""
        houses = np.zeros(NUM_PROPERTIES, dtype=np.int8)
        for idx, prop_name in enumerate(self.property_order):
            # Vérifier si le joueur possède la propriété
            if prop_name in player.properties:
                case = next(c for c in self.board.board if c["name"] == prop_name)
                houses[idx] = min(case.get("houses", 0), 5)
            else:
                houses[idx] = -1  # Marqueur pour propriété non possédée
        return houses

    def _properties_to_binary(self, player):
        """Convertit les propriétés du joueur en vecteur binaire."""
        binary_vector = np.zeros(NUM_PROPERTIES, dtype=np.int8)
        for idx, prop_name in enumerate(self.property_order):
            if prop_name in player.properties:
                binary_vector[idx] = 1
        return binary_vector

    def _get_info(self):
       pass

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        observation = self._get_obs_for_player(self.players[self.current_player_idx])
        return observation
        pass

    def _select_mortgage_property(self, player: Player) -> str:
        """Sélectionne automatiquement une propriété à hypothéquer selon la politique de l'agent"""
        mortgageable = [
            prop for prop in player.properties
            if not self._get_board_property(prop)["mortgaged"]
        ]

        # Ici vous utiliserez la politique de l'agent pour choisir
        # Pour l'exemple, on prend la première propriété hypothécable
        if mortgageable:
            return mortgageable[0]
        raise self.InvalidAction("Aucune propriété hypothécable disponible")

    def _validate_property_ownership(self, player: Player, property_name: str, owner: Player = None) -> dict:
        """Valide la possession d'une propriété (méthode helper réutilisable)"""
        prop = self._get_board_property(property_name)
        if not prop:
            raise self.InvalidAction(f"Propriété {property_name} introuvable")
        if owner and property_name not in owner.properties:
            raise self.InvalidAction(f"{owner.name} ne possède pas {property_name}")
        if property_name not in player.properties and not owner:
            raise self.InvalidAction(f"Vous ne possédez pas {property_name}")
        return prop

    def _select_build_property(self, player: Player) -> str:
        """Sélectionne automatiquement une propriété où construire selon la politique de l'agent"""
        buildable = []
        for prop in player.properties:
            case = self._get_board_property(prop)
            if case["type"] == "property" and not case["mortgaged"]:
                color_group = self._get_color_group(case["color_code"])
                if all(p in player.properties for p in color_group):
                    buildable.append(prop)

        # Prioriser les propriétés avec le moins de maisons
        if buildable:
            return min(buildable, key=lambda p: self._get_board_property(p)["houses"])
        raise self.InvalidAction("Aucune propriété constructible disponible")

    def _cycle_to_next_player(self):
        """Passe au prochain joueur non bankrupt."""
        for _ in range(len(self.players)):
            self.current_player_idx = (self.current_player_idx + 1) % len(self.players)
            if not self.players[self.current_player_idx].bankrupt:
                break

    def step(self, action):
        player = self.players[self.current_player_idx]
        reward = 0
        done = False
        info = {}

        try:
            if action == 0:  # Hypothèque
                prop = self._select_mortgage_property(player)
                self._handle_mortgage(player, prop)

            elif action == 1:  # Construction
                prop = self._select_build_property(player)
                self._handle_build(player, prop)

            elif action["action_type"] == 2:  # Trade argent/propriété
                partner = self._get_other_players(player)[action["trade_partner"]]
                prop = self.property_order[action["property_idx"]]
                self._handle_trade(player, partner, prop, action["trade_amount"])

            elif action["action_type"] == 3:  # Trade propriété/propriété
                partner = self._get_other_players(player)[action["trade_partner"]]
                self._handle_property_swap(player, partner, action["property_idx"])

            elif action["action_type"] == 4:  # Ne rien faire
                pass

            reward = self._calculate_reward(player)

        except self.InvalidAction as e:
            reward = -10
            info["error"] = str(e)

        self._cycle_to_next_player()
        next_obs = self._get_obs_for_player(self.players[self.current_player_idx])

        return next_obs, reward, done, info

    def _handle_mortgage(self, player: Player, property_name: str):
        """Utilise la méthode helper pour la validation"""
        prop = self._validate_property_ownership(player, property_name)

        if prop.get("mortgaged", False):
            raise self.InvalidAction(f"{property_name} est déjà hypothéquée")

        prop["mortgaged"] = True
        player.receive(prop["hypothèque"])
        print(f"{player.name} a hypothéqué {property_name}")

    def _handle_build(self, player: Player, property_name: str):
        """Gère la construction d'une maison"""
        prop = self._get_board_property(property_name)
        if not prop:
            raise InvalidAction(f"Propriété {property_name} introuvable")

        # Vérification de la possession complète du groupe
        color_group = self._get_color_group(prop["color_code"])
        if any(p not in player.properties for p in color_group):
            raise InvalidAction("Possession incomplète du groupe de couleurs")

        # Vérification des fonds
        house_cost = prop["price"] // 2
        if player.money < house_cost:
            raise InvalidAction("Fonds insuffisants")

        prop["houses"] = min(prop.get("houses", 0) + 1, 5)
        player.pay(house_cost)

    def _get_other_players(self, current_player: Player) -> List[Player]:
        """Retourne la liste des autres joueurs actifs"""
        return [p for p in self.players if p != current_player and not p.bankrupt]

    @staticmethod
    def _handle_trade(buyer: Player, seller: Player, property_name: str, amount: int):
        """Gère un échange argent contre propriété (méthode static)"""
        if property_name not in seller.properties:
            raise Game.InvalidAction(f"{seller.name} ne possède pas {property_name}")
        if buyer.money < amount:
            raise Game.InvalidAction(f"{buyer.name} n'a pas assez d'argent")

        buyer.pay(amount)
        seller.receive(amount)
        seller.properties.remove(property_name)
        buyer.properties.append(property_name)

    def _handle_property_swap(self, player: Player, partner: Player, property_name: str):
        """Utilise la méthode helper pour la validation"""
        player_prop = self._validate_property_ownership(player, property_name)
        partner_prop = self._validate_property_ownership(partner, property_name, owner=partner)

        if player_prop not in player.properties:
            raise InvalidAction("Propriété invalide")

        player.properties.remove(player_prop)
        partner.properties.append(player_prop)
        partner.properties.remove(partner_prop)
        player.properties.append(partner_prop)

    def _get_color_group(self, color_code: str) -> List[str]:
        """Retourne toutes les propriétés d'un même groupe de couleur"""
        return [
            p["name"] for p in self.board.board
            if p.get("color_code") == color_code
               and p["type"] == "property"
        ]

    # Ajouter cette classe d'exception interne
    class InvalidAction(Exception):
        def __init__(self, message: str):
            super().__init__(message)
            self.message = message

    def _get_mortgageable_properties(self, player):
        return [prop for prop in player.properties
                if not self._get_board_property(prop)["mortgaged"]]

    def _get_buildable_properties(self, player):
        buildable = []
        for prop in player.properties:
            case = self._get_board_property(prop)
            if case["type"] == "property" and not case["mortgaged"]:
                color_group = self._get_color_group(case["color_code"])
                if all(p in player.properties for p in color_group):
                    buildable.append(prop)
        return buildable

    def _calculate_reward(self, player):
        """Reward function combinant plusieurs facteurs"""
        reward = 0

        # Récompense pour l'argent gagné
        reward += player.money * 0.01

        # Récompense pour les propriétés
        reward += len(player.properties) * 5

        # Récompense pour les maisons construites
        for prop in player.properties:
            case = self._get_board_property(prop)
            reward += case.get("houses", 0) * 10

        # Pénalité pour faillite
        if player.bankrupt:
            reward -= 1000

        return reward

    @staticmethod
    def _initialize_players() -> List[Player]:
        return [Player(name=f"Player {i + 1}") for i in range(4)]

    def _init_property_states(self):
        for case in self.board.board:
            if case["type"] == "property":
                case.setdefault("houses", 0)

    def start(self):
        """
        Boucle principale du jeu. Tant qu'il reste plus d'un joueur (non en faillite),
        chaque joueur joue à son tour.
        """
        round_number = 1
        while len(self.players) > 1:
            print(f"\n====== Round {round_number} ======")
            for player in self.players.copy():
                if player.bankrupt:
                    continue
                print(f"\n--- Tour de {player.name} ---")
                self._handle_player_turn(player)
                if len(self.players) == 1:
                    break
            round_number += 1

        if self.players:
            print(f"\nFélicitations, {self.players[0].name} a gagné la partie!")
        else:
            print("La partie s'est terminée sans vainqueur.")

    def _handle_player_turn(self, player: Player):
        """
        Gère le tour d'un joueur :
          - Le joueur appuie sur Entrée pour lancer les dés.
          - Le joueur lance les dés et avance sur le plateau.
          - On récupère la case d'arrivée et on y applique l'action correspondante.
        """
        if player.bankrupt:
            print(f"{player.name} est en faillite et ne peut plus jouer.")
            return

        input(f"{player.name}, appuyez sur Entrée pour lancer les dés...")
        dice_roll = self._roll_dice()
        print(f"{player.name} a lancé les dés et obtenu {dice_roll}.")

        # Mise à jour de la position du joueur
        player.position = self.board.move_player(player.position, dice_roll)
        current_case = self.board.get_case(player.position)
        print(f"{player.name} se déplace vers la case '{current_case['name']}'.")

        # Traitement de la case d'arrivée
        self._handle_case_action(player, current_case)

    @staticmethod
    def _roll_dice() -> int:
        return random.randint(1, 12)

    def _handle_case_action(self, player: Player, case: dict):
        if case["type"] == "property":
            self._handle_property_case(player, case)
        elif case["type"] == "tax":
            self._handle_tax_case(player, case)
        elif case["type"] == "start":
            player.money += 200
            print(f"{player.name} reçoit 200€ en passant par la case 'Départ'.")
        elif case["type"] == "community_chest":
            self._handle_action_case_community_chest(player, case)
        elif case["type"] == "chance":
            self._handle_action_case_chance(player, case)
        elif case["type"] == "go_to_jail":
            self._handle_action_case_jail(player, case)
        elif case["type"] == "free_parking":
            player.money += 200
            print(f"{player.name} reçoit 200€ sur 'Parc gratuit'.")
        elif case["type"] == "utility":
            # Logique spécifique aux compagnies (non implémentée ici)
            pass
        else:
            print(f"Aucune action définie pour le type '{case['type']}'.")

    def _handle_property_case(self, player: Player, case: dict):
        owner = Game.find_property_owner(self.players, case["name"])
        if not owner:
            # Propose au joueur d'acheter la propriété
            choice = input(f"{player.name}, voulez-vous acheter {case['name']} pour {case['price']}€ ? (o/n) : ").strip().lower()
            if choice == "o":
                Game._handle_property_purchase(player, case)
            else:
                print(f"{player.name} a refusé d'acheter {case['name']}. L'enchère démarre.")
                self.auction_property(case["name"], starting_bid=case["price"])
        else:
            self._handle_rent_payment(player, case, owner)

    @staticmethod
    def find_property_owner(players: List[Player], property_name: str) -> Optional[Player]:
        return next((p for p in players if property_name in p.properties), None)

    @staticmethod
    def _handle_property_purchase(player: Player, case: dict):
        if player.buy_property(case["name"], case["price"]):
            print(f"✅ {player.name} a acheté {case['name']} ! Solde: {player.money}€")
        else:
            print(f"❌ {player.name} ne peut pas acheter {case['name']}")

    def _handle_rent_payment(self, player: Player, case: dict, owner: Player):
        if owner == player:
            print(f"🌟 {player.name} est déjà propriétaire")
            return

        if case["type"] == "station":
            station_count = sum(1 for prop in owner.properties if prop.startswith("Gare"))
            station_count = max(station_count, 1)
            base_rent = case.get("rent", 25)
            multiplier = 2 ** (station_count - 1)
            rent = base_rent * multiplier
            print(f"{owner.name} possède {station_count} station(s) - le loyer est maintenant de {rent}€.")
        else:
            rent = Game.calculate_rent(case)

        print(f"💸 Loyer dû à {owner.name}: {rent}€")
        if player.money < rent:
            self.action_in_game(player)
        if player.money < rent:
            self.handle_bankruptcy(player, creditor=owner)
            return

        player.pay(rent)
        owner.receive(rent)
        print(f"Solde {player.name}: {player.money}€ → {owner.name}: {owner.money}€")

    @staticmethod
    def calculate_rent(case: dict) -> int:
        try:
            houses = case.get("houses", 0)
            return {
                0: case["rent"],
                5: case["hotel"]
            }.get(houses, case.get(f"H{houses}", case["rent"]))
        except KeyError as e:
            print(f"Erreur configuration: {e}")
            return case["rent"]

    def _handle_tax_case(self, player: Player, case: dict):
        tax = case["price"]
        if player.money < tax:
            self.action_in_game(player)
        if player.money < tax:
            self.handle_bankruptcy(player)
            return
        player.pay(tax)
        print(f"⚖️ {player.name} paie {tax}€ de taxes. Solde: {player.money}€")

    def _handle_action_case_jail(self, player: Player, case: dict, jail_price: int = 50):
        if case["type"] == "go_to_jail":
            jail_position = self.board.get_position("Prison/Simple visite")
            player.position = jail_position
            print(f"{player.name} est envoyé en prison !")
            print("Choisissez une action :")
            print(f"1. Payer {jail_price}€ pour sortir immédiatement")
            print("2. Lancer les dés pour tenter de sortir")
            choice = input("Votre choix : ").strip()
            if choice == "1":
                if player.money < jail_price:
                    self.handle_bankruptcy(player)
                    return
                else:
                    player.pay(jail_price)
                    dice_roll = self._roll_dice()
                    player.position = self.board.move_player(player.position, dice_roll)
                    current_case = self.board.get_case(player.position)
                    print(f"{player.name} avance de {dice_roll} cases et arrive sur {current_case['name']}.")
                    self._handle_case_action(player, current_case)
            elif choice == "2":
                if not hasattr(player, "jail_turns"):
                    player.jail_turns = 0
                die1 = random.randint(1, 6)
                die2 = random.randint(1, 6)
                print(f"{player.name} lance les dés: {die1} et {die2}.")
                if die1 == die2:
                    print(f"{player.name} a fait un double et est libéré de prison !")
                    player.jail_turns = 0
                    movement = die1 + die2
                    player.position = self.board.move_player(player.position, movement)
                    current_case = self.board.get_case(player.position)
                    print(f"{player.name} avance de {movement} cases et arrive sur {current_case['name']}.")
                    self._handle_case_action(player, current_case)
                else:
                    player.jail_turns += 1
                    if player.jail_turns >= 3:
                        print(f"{player.name} n'a pas fait de double en 3 tentatives et est libéré de prison.")
                        player.jail_turns = 0
                        new_die1 = random.randint(1, 6)
                        new_die2 = random.randint(1, 6)
                        movement = new_die1 + new_die2
                        print(f"{player.name} lance à nouveau les dés: {new_die1} et {new_die2}.")
                        player.position = self.board.move_player(player.position, movement)
                        current_case = self.board.get_case(player.position)
                        print(f"{player.name} avance de {movement} cases et arrive sur {current_case['name']}.")
                        self._handle_case_action(player, current_case)
                    else:
                        print(f"{player.name} n'a pas fait de double et reste en prison (tentative {player.jail_turns}/3).")
            else:
                print("Choix invalide. Le joueur reste en prison pour ce tour.")

    def _handle_random_card_action(self, player: Player, actions: List[dict]):
        """
        Exécute une action aléatoire à partir de la liste fournie.
        La logique est commune aux cartes Chance et Community Chest.
        """
        chosen_action = random.choice(actions)
        print(chosen_action["message"])
        action_type = chosen_action["type"]

        if action_type == "advance_to_go":
            player.position = self.board.get_position("Départ")
            player.receive(chosen_action["amount"])
            print(
                f"{player.name} est maintenant sur Départ et reçoit {chosen_action['amount']}€. Nouveau solde : {player.money}€.")
        elif action_type == "gain_money":
            player.receive(chosen_action["amount"])
            print(f"Le nouveau solde de {player.name} est de {player.money}€.")
        elif action_type == "lose_money":
            if player.money < chosen_action["amount"]:
                self.action_in_game(player)
            if player.money < chosen_action["amount"]:
                self.handle_bankruptcy(player)
                return
            player.pay(chosen_action["amount"])
            print(f"Le nouveau solde de {player.name} est de {player.money}€.")
        elif action_type == "advance":
            player.position = self.board.move_player(player.position, chosen_action["spaces"])
            current_case = self.board.get_case(player.position)
            print(f"{player.name} avance de {chosen_action['spaces']} cases et se retrouve sur {current_case['name']}.")
            self._handle_case_action(player, current_case)
        elif action_type == "go_to_jail":
            player.position = self.board.get_position("Allez en prison")
            print(f"{player.name} est envoyé en prison !")
        elif action_type == "nothing":
            print(f"Pas d'action supplémentaire pour {player.name}.")

    def _handle_action_case_chance(self, player: Player, case: dict):
        if case["type"] == "chance":
            actions = [
                {
                    "type": "advance_to_go",
                    "amount": 200,
                    "message": f"{player.name} avance jusqu'à Départ et reçoit 200€ !"
                },
                {
                    "type": "gain_money",
                    "amount": 50,
                    "message": f"{player.name} reçoit un dividende de 50€ de la banque."
                },
                {
                    "type": "lose_money",
                    "amount": 15,
                    "message": f"{player.name} doit payer une amende de 15€ pour excès de vitesse."
                },
                {
                    "type": "advance",
                    "spaces": 2,
                    "message": f"{player.name} avance de 2 cases."
                },
                {
                    "type": "go_to_jail",
                    "message": f"{player.name} va directement en prison !"
                },
                {
                    "type": "nothing",
                    "message": f"Aucune action particulière pour {player.name} cette fois."
                }
            ]
            self._handle_random_card_action(player, actions)

    def _handle_action_case_community_chest(self, player: Player, case: dict):
        if case["type"] == "community_chest":
            actions = [
                {
                    "type": "gain_money",
                    "amount": 200,
                    "message": f"{player.name} reçoit 200€ de la communauté !"
                },
                {
                    "type": "lose_money",
                    "amount": 100,
                    "message": f"{player.name} doit payer 100€ à la communauté."
                },
                {
                    "type": "advance",
                    "spaces": 3,
                    "message": f"{player.name} avance de 3 cases."
                },
                {
                    "type": "go_to_jail",
                    "message": f"{player.name} va directement en prison !"
                },
                {
                    "type": "nothing",
                    "message": f"Aucune action particulière pour {player.name} cette fois."
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
        Permet au joueur de réaliser une action en jeu pour améliorer sa situation financière.
        Les options incluent :
          1. Hypothéquer une propriété
          2. Construire une maison
          3. Trade: Acheter une propriété (argent contre propriété)
          4. Trade: Échanger une propriété contre une autre propriété
          5. Quitter (aucune action)
        """
        print("\n--- Actions disponibles ---")
        print("1. Hypothéquer une propriété")
        print("2. Construire une maison")
        print("3. Trade: Acheter une propriété à un autre joueur (argent contre propriété)")
        print("4. Trade: Échanger une propriété contre une autre propriété")
        print("5. Quitter (aucune action)")
        choice = input("Votre choix : ").strip()

        if choice == "1":
            eligible_props = []
            for prop in player.properties:
                board_prop = self._get_board_property(prop)
                if board_prop is not None and board_prop["type"] in ["property", "station", "utility"]:
                    if not board_prop.get("mortgaged", False):
                        eligible_props.append(prop)
            if not eligible_props:
                print("Vous n'avez aucune propriété éligible à l'hypothèque.")
                return

            print("\nPropriétés éligibles à l'hypothèque :")
            for i, prop in enumerate(eligible_props, start=1):
                board_prop = self._get_board_property(prop)
                print(f"{i}. {prop} (Valeur hypothécaire : {board_prop['hypothèque']}€)")
            selection = input("Sélectionnez la propriété à hypothéquer (numéro) : ").strip()
            try:
                sel = int(selection)
                if sel < 1 or sel > len(eligible_props):
                    print("Sélection invalide.")
                    return
                chosen_prop = eligible_props[sel - 1]
                board_prop = self._get_board_property(chosen_prop)
                board_prop["mortgaged"] = True
                mortgage_value = board_prop["hypothèque"]
                player.receive(mortgage_value)
                print(f"{player.name} a hypothéqué {chosen_prop} et reçoit {mortgage_value}€. Nouveau solde : {player.money}€.")
            except ValueError:
                print("Entrée invalide.")

        elif choice == "2":
            eligible_props = []
            for prop in player.properties:
                board_prop = self._get_board_property(prop)
                if board_prop is not None and board_prop["type"] == "property":
                    if not board_prop.get("mortgaged", False):
                        eligible_props.append(prop)
            if not eligible_props:
                print("Vous n'avez aucune propriété éligible pour construire des maisons.")
                return

            print("\nPropriétés éligibles pour construire des maisons :")
            for i, prop in enumerate(eligible_props, start=1):
                board_prop = self._get_board_property(prop)
                houses = board_prop.get("houses", 0)
                print(f"{i}. {prop} (Maisons actuelles : {houses})")
            selection = input("Sélectionnez la propriété sur laquelle construire une maison (numéro) : ").strip()
            try:
                sel = int(selection)
                if sel < 1 or sel > len(eligible_props):
                    print("Sélection invalide.")
                    return
                chosen_prop = eligible_props[sel - 1]
                board_prop = self._get_board_property(chosen_prop)
                # Vérification de la possession complète du groupe de couleur
                if board_prop["type"] == "property":
                    color_group = [
                        p["name"] for p in self.board.board
                        if p.get("color_code") == board_prop["color_code"]
                           and p["type"] == "property"
                    ]
                    missing = [prop for prop in color_group if prop not in player.properties]

                    if missing:
                        print(
                            f"❌ Construction impossible ! Vous devez posséder tout le groupe {board_prop['color_code'].upper()}")
                        print(f"Propriétés manquantes : {', '.join(missing)}")
                        return
                current_houses = board_prop.get("houses", 0)
                if current_houses >= 4:
                    print(f"Vous avez déjà le maximum de maisons sur {chosen_prop} (4 maisons maximum).")
                    return
                house_cost = int(board_prop["price"] / 2)
                if player.money < house_cost:
                    print(f"{player.name} n'a pas assez d'argent pour construire une maison sur {chosen_prop} (coût : {house_cost}€).")
                    return
                player.pay(house_cost)
                board_prop["houses"] = current_houses + 1
                print(f"Une maison a été construite sur {chosen_prop}. Nombre de maisons maintenant : {board_prop['houses']}.")
                print(f"Nouveau solde de {player.name} : {player.money}€.")
            except ValueError:
                print("Entrée invalide.")

        elif choice == "3":
            # Trade: Acheter une propriété à un autre joueur (argent contre propriété)
            seller_name = input("Entrez le nom du joueur vendeur : ").strip()
            seller = next((p for p in self.players if p.name.lower() == seller_name.lower() and p != player), None)
            if not seller:
                print("Joueur introuvable ou vous ne pouvez pas trader avec vous-même.")
                return
            property_name = input(f"{seller.name}, entrez le nom de la propriété que vous souhaitez vendre : ").strip()
            Game.trade_action_money_to_card(player, seller, property_name)

        elif choice == "4":
            # Trade: Échanger une propriété contre une autre propriété
            partner_name = input("Entrez le nom du joueur avec qui vous souhaitez échanger : ").strip()
            partner = next((p for p in self.players if p.name.lower() == partner_name.lower() and p != player), None)
            if not partner:
                print("Joueur introuvable ou vous ne pouvez pas trader avec vous-même.")
                return
            Game.trade_action_card_to_card(player, partner)

        elif choice == "5":
            print("Aucune action effectuée.")
        else:
            print("Choix invalide.")

    @staticmethod
    def trade_action_money_to_card(buyer: Player, seller: Player, property_name: str):
        if property_name not in seller.properties:
            print(f"Erreur : {seller.name} ne possède pas la propriété {property_name}.")
            return

        amount_input = input(
            f"{buyer.name}, combien voulez-vous payer pour acheter {property_name} de {seller.name} ?\nMontant : "
        )
        if not amount_input.isnumeric():
            print("Erreur : veuillez entrer un montant valide (nombre).")
            return

        amount = int(amount_input)

        if buyer.money < amount:
            print(f"Erreur : {buyer.name} n'a pas assez d'argent pour payer {amount}€.")
            return

        print(f"{buyer.name} paie {amount}€ à {seller.name} pour acheter {property_name}.")
        buyer.pay(amount)
        seller.receive(amount)

        seller.properties.remove(property_name)
        buyer.properties.append(property_name)
        print(f"Transaction réussie ! {buyer.name} possède maintenant {property_name}.")
        print(f"Solde de {buyer.name} : {buyer.money}€")
        print(f"Solde de {seller.name} : {seller.money}€")

    @staticmethod
    def trade_action_card_to_card(buyer: Player, seller: Player):
        buyer_property = input(
            f"{buyer.name}, entrez le nom de la propriété que vous souhaitez offrir pour l'échange : "
        ).strip()
        if not buyer_property:
            print("Erreur : vous devez entrer un nom de propriété valide.")
            return

        if buyer_property not in buyer.properties:
            print(f"Erreur : {buyer.name} ne possède pas la propriété '{buyer_property}'.")
            return

        seller_property = input(
            f"{seller.name}, entrez le nom de la propriété que vous souhaitez offrir en échange de '{buyer_property}' : "
        ).strip()
        if not seller_property:
            print("Erreur : vous devez entrer un nom de propriété valide.")
            return

        if seller_property not in seller.properties:
            print(f"Erreur : {seller.name} ne possède pas la propriété '{seller_property}'.")
            return

        print(f"Transaction proposée : {buyer.name} offre '{buyer_property}' en échange de '{seller_property}' de {seller.name}.")

        buyer.properties.remove(buyer_property)
        seller.properties.append(buyer_property)

        seller.properties.remove(seller_property)
        buyer.properties.append(seller_property)

        print("Transaction réussie !")
        print(f"{buyer.name} possède maintenant : {buyer.properties}")
        print(f"{seller.name} possède maintenant : {seller.properties}")

    def handle_bankruptcy(self, player: Player, creditor: Optional[Player] = None):
        """
        Gère la faillite d'un joueur.
        - Si un créancier (creditor) est précisé, ce dernier reçoit l'argent restant et les propriétés du joueur.
        - Sinon, les propriétés sont remises sur le marché.
        Le joueur est ensuite retiré de la partie.
        """
        print(f"🚨 {player.name} est en faillite !")
        if creditor:
            print(f"Les actifs de {player.name} sont transférés à {creditor.name}.")
            creditor.receive(player.money)
            for prop in player.properties:
                creditor.properties.append(prop)
        else:
            print(f"Les propriétés de {player.name} sont remises sur le marché.")

        player.money = 0
        player.properties = []
        player.bankrupt = True

        if player in self.players:
            self.players.remove(player)
        print(f"{player.name} a été retiré de la partie.")

    def auction_property(self, property_name: str, starting_bid: int = 0):
        """
        Organise une enchère pour la propriété dont le nom est property_name.
        Tous les joueurs non en faillite participent.
        """
        print(f"\nLancement de l'enchère pour {property_name} (enchère initiale : {starting_bid}€)")
        eligible_players = [p for p in self.players if not p.bankrupt]
        if not eligible_players:
            print("Aucun joueur n'est éligible pour participer à l'enchère.")
            return

        current_bid = starting_bid
        highest_bidder = None
        active_bidders = eligible_players.copy()

        while len(active_bidders) > 1:
            any_bid_made = False
            for player in active_bidders.copy():
                print(f"\n{player.name}, le montant actuel est de {current_bid}€.")
                bid_str = input(f"{player.name}, entrez votre offre (ou appuyez sur Entrée pour passer) : ").strip()
                if bid_str == "":
                    print(f"{player.name} passe et n'est plus éligible pour cette enchère.")
                    active_bidders.remove(player)
                else:
                    try:
                        bid = int(bid_str)
                        if bid <= current_bid:
                            print("Votre offre doit être supérieure à l'offre actuelle.")
                        elif bid > player.money:
                            print("Vous n'avez pas assez d'argent pour proposer cette offre.")
                        else:
                            current_bid = bid
                            highest_bidder = player
                            any_bid_made = True
                            print(f"{player.name} propose {bid}€.")
                    except ValueError:
                        print("Entrée invalide. Vous passez cette ronde.")
                        active_bidders.remove(player)
            if not any_bid_made:
                break

        if highest_bidder is not None:
            print(f"\n{highest_bidder.name} remporte l'enchère pour {property_name} avec une offre de {current_bid}€.")
            highest_bidder.pay(current_bid)
            highest_bidder.properties.append(property_name)
            print(f"Le nouveau solde de {highest_bidder.name} est de {highest_bidder.money}€.")
        else:
            print("Aucune offre n'a été faite pour cette enchère.")
gym.register(
    id="MyMonopolyEnv-v0",
    entry_point="Monopoly_AI.environement.game:Game",
)
if __name__ == "__main__":
    game = Game()
    game.start()