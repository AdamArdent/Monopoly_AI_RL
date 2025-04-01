# main.py

from Mojo_RL.training.train import train_agent
from Mojo_RL.env.mojo_env import MojoEnv
def main():
    num_episodes = 50  # Vous pouvez également charger ce paramètre depuis config/config.yaml
    print("Début de l'entraînement de l'agent sur le jeu Mojo...")
    rewards = train_agent(num_episodes=num_episodes)
    print("Entraînement terminé.")
    test= MojoEnv()
    test.render()
if __name__ == "__main__":
    main()
