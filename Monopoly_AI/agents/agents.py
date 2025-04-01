import gymnasium as gym

# Assurez-vous que le code de registration a déjà été exécuté
env = gym.make("MonopolyEnv-v1")


class AgentPolicy:
    def choose_mortgage_property(self, mortgageable_props: List[str], state: dict) -> int:
        """À connecter à votre modèle d'IA"""
        # Exemple : choisir la propriété la plus chère
        values = [self._get_property_value(prop) for prop in mortgageable_props]
        return np.argmax(values)

    def choose_build_property(self, buildable_props: List[str], state: dict) -> int:
        """À connecter à votre modèle d'IA"""
        # Exemple : choisir la propriété avec le meilleur ROI
        roi = [self._calculate_roi(prop) for prop in buildable_props]
        return np.argmax(roi)