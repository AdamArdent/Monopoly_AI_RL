# env/mojo_game.py

import random


class MojoGame:
    """
    Classe représentant la logique du jeu de cartes Mojo, adaptée pour une interaction avec un agent.

    Dans cette version simplifiée, on joue avec un seul joueur dont la main est limitée à hand_size cartes.
    Deux actions sont possibles :
      - 0 : Tirer une carte et l’ajouter à la main (récompense = valeur de la carte).
      - 1 : Tirer une carte, l’ajouter à la main, puis retirer la carte la plus faible
            (récompense = différence entre la carte tirée et la carte retirée).
    """

    def __init__(self, hand_size=5):
        self.hand_size = hand_size
        self.reset_game()

    def reset_game(self):
        """Réinitialise l’état du jeu."""
        self.deck = self._create_deck()
        self.player = {"hand": [], "score": 0}
        # Distribution initiale : le joueur reçoit jusqu’à hand_size cartes
        for _ in range(self.hand_size):
            if self.deck:
                self.player["hand"].append(self.deck.pop(0))
        self.game_over = False

    def _create_deck(self):
        """Crée et mélange un deck de 52 cartes (valeurs de 1 à 52)."""
        deck = list(range(1, 53))
        random.shuffle(deck)
        return deck

    def get_state(self):
        """
        Retourne l’état du jeu sous forme de liste :
          - La main du joueur (hand_size valeurs normalisées, complétées par 0 si nécessaire)
          - Le score normalisé (score / (52 * hand_size))
          - Le ratio de cartes restantes dans le deck (len(deck)/52)

        La taille totale de l’observation est hand_size + 2.
        """
        hand = self.player["hand"]
        hand_padded = hand + [0] * (self.hand_size - len(hand))
        normalized_hand = [card / 52.0 for card in hand_padded]
        normalized_score = self.player["score"] / (52 * self.hand_size)
        deck_ratio = len(self.deck) / 52.0
        return normalized_hand + [normalized_score, deck_ratio]

    def apply_action(self, action):
        """
        Applique une action pour le joueur et retourne (observation, reward, done).

        Paramètres:
          action (int): 0 ou 1.

        Renvoie:
          obs (list): Observation de l’état après l’action.
          reward (float): Récompense obtenue.
          done (bool): True si le jeu est terminé.
        """
        if self.game_over:
            return self.get_state(), 0, True

        reward = 0

        # Vérifier la disponibilité d'une carte à piocher
        if self.deck:
            card = self.deck.pop(0)
        else:
            self.game_over = True
            return self.get_state(), 0, True

        if action == 0:
            # Action 0 : ajouter la carte et incrémenter le score de sa valeur
            self.player["hand"].append(card)
            self.player["score"] += card
            reward = card
        elif action == 1:
            # Action 1 : ajouter la carte puis retirer la carte la plus faible (si possible)
            self.player["hand"].append(card)
            if len(self.player["hand"]) > 1:
                removed = min(self.player["hand"])
                self.player["hand"].remove(removed)
                diff = card - removed
                self.player["score"] += diff
                reward = diff
            else:
                self.player["score"] += card
                reward = card
        else:
            # Action invalide
            reward = -1

        # Si la main dépasse hand_size, retirer la carte la plus ancienne
        if len(self.player["hand"]) > self.hand_size:
            self.player["hand"].pop(0)

        # Si le deck est épuisé, fin du jeu
        if not self.deck:
            self.game_over = True

        return self.get_state(), reward, self.game_over
