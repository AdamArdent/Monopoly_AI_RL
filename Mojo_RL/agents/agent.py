# agents/agent.py

import numpy as np
import random

class Agent:
    """
    Agent utilisant une approche de Q-learning sur table pour l’environnement Mojo.
    La stratégie epsilon-greedy est utilisée pour le choix d’action.
    """
    def __init__(self, action_space, observation_space, learning_rate=0.1, discount_factor=0.99,
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.action_space = action_space
        self.observation_space = observation_space
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Pour la Q-table, nous discrétisons chaque dimension de l'observation en 10 bins.
        self.num_bins = 10
        self.q_table = {}

    def discretize(self, obs):
        """
        Convertit l'observation (vecteur de réels) en un tuple d'entiers.
        """
        bins = np.linspace(0, 1, self.num_bins)
        discretized = tuple(np.digitize(x, bins) for x in obs)
        return discretized

    def get_q(self, state):
        """
        Renvoie les valeurs Q pour un état discrétisé.
        Si l’état n’existe pas dans la table, il est initialisé à 0 pour chaque action.
        """
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_space.n)
        return self.q_table[state]

    def choose_action(self, obs):
        """
        Choisit une action suivant la stratégie epsilon-greedy.
        """
        state = self.discretize(obs)
        if random.random() < self.epsilon:
            return self.action_space.sample()
        else:
            q_values = self.get_q(state)
            return int(np.argmax(q_values))

    def learn(self, obs, action, reward, next_obs, done):
        """
        Met à jour la Q-table selon la règle de Q-learning.
        """
        state = self.discretize(obs)
        next_state = self.discretize(next_obs)
        q_values = self.get_q(state)
        next_q_values = self.get_q(next_state)
        target = reward if done else reward + self.gamma * np.max(next_q_values)
        q_values[action] += self.lr * (target - q_values[action])
        # Mise à jour de epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
