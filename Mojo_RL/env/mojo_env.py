# env/mojo_env.py

import gym
from gym import spaces
import numpy as np
from .mojo_game import MojoGame  # Import relatif dans le package


class MojoEnv(gym.Env):
    """
    Environnement Gym pour le jeu Mojo.
    Cet environnement encapsule la logique de jeu via la classe MojoGame.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(MojoEnv, self).__init__()
        # Deux actions possibles : 0 et 1
        self.action_space = spaces.Discrete(2)
        # Observation : hand_size cartes + score + deck_ratio
        # Pour hand_size=5, la taille de l'observation est 7
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(7,), dtype=np.float32)
        self.game = MojoGame(hand_size=5)
        self.state = None

    def reset(self):
        """Réinitialise l’environnement et retourne l’état initial."""
        self.game.reset_game()
        self.state = np.array(self.game.get_state(), dtype=np.float32)
        return self.state

    def step(self, action):
        """
        Exécute l’action dans la logique du jeu et renvoie (observation, reward, done, info).
        """
        obs, reward, done = self.game.apply_action(action)
        self.state = np.array(obs, dtype=np.float32)
        info = {"score": self.game.player["score"]}
        return self.state, reward, done, info

    def render(self, mode='human'):
        """Affiche l’état actuel du jeu."""
        state = self.game.get_state()
        score = self.game.player["score"]
        deck_count = len(self.game.deck)
        print(f"Etat: {state} | Score: {score} | Cartes restantes: {deck_count}")

    def close(self):
        pass

# Pour tester l’environnement indépendamment, vous pouvez décommenter le bloc suivant :
# if __name__ == "__main__":
#     env = MojoEnv()
#     obs = env.reset()
#     done = False
#     while not done:
#         action = env.action_space.sample()
#         obs, reward, done, info = env.step(action)
#         env.render()
#     env.close()
