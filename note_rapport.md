# Rapport de Revue de Code : Environnement MonopolyRLEnv

**Projet:** Environnement MonopolyRLEnv pour l'apprentissage par renforcement sur Monopoly

---

## 1. Introduction / Résumé Exécutif

Ce rapport détaille l'analyse de la classe `MonopolyRLEnv`, conçue comme un environnement d'apprentissage par renforcement pour le jeu de Monopoly, utilisant l'interface Gymnasium.

Le code présente une structure de base solide, notamment la séparation entre l'environnement RL (`MonopolyRLEnv`) et une version jouable par l'homme (`MonopolyGame`), ainsi qu'une adhésion correcte à l'API Gymnasium (définition des espaces, méthodes `reset`, `step`). L'utilisation de masques d'action (`action_masks`) est également un point positif notable.

Cependant, l'environnement souffre de lacunes majeures qui empêchent son utilisation efficace pour l'entraînement d'un agent RL performant. Les problèmes principaux sont :

* **Implémentation très incomplète** de la logique de jeu fondamentale au sein de la fonction `step` de `MonopolyRLEnv`.
* Un système d'échange (**trading**) extrêmement simplifié et peu réaliste.

Ce rapport détaille ces points et propose un ensemble de recommandations priorisées pour guider le développement futur.

## 2. Analyse Détaillée

### 2.1. Points Forts

* **Conformité Gymnasium**: L'environnement respecte l'interface `gym.Env`, facilitant l'intégration avec les bibliothèques RL standards.
* **Séparation des préoccupations**: La distinction entre `MonopolyRLEnv` et `MonopolyGame` est une bonne pratique.
* **Structure des Espaces**: L'utilisation de `gym.spaces.Dict` et des types d'espaces (`Box`, `Discrete`, `MultiBinary`) est globalement adaptée.
* **Masques d'Action**: L'inclusion de `action_masks` dans l'observation est une excellente initiative pour guider l'agent.
* **Normalisation**: La normalisation des données des propriétés (`property_data_norm`) est présente.

### 2.2. Problèmes Majeurs et Points Critiques

* **Logique de Jeu Incomplète (Flux de Tour - Priorité Haute)**:
    * La fonction `MonopolyRLEnv.step` ne gère actuellement que des actions de gestion (hypothéquer, construire, échanger, ne rien faire).
    * **Absence Totale du Cœur de Jeu**: Le lancer de dés, le déplacement du pion, l'atterrissage sur une case (avec ses conséquences : achat, paiement de loyer, pioche de carte, aller en prison, case départ, etc.) ne sont pas simulés dans `MonopolyRLEnv`.
    * **Incohérence**: La classe `MonopolyGame` contient cette logique, mais elle est absente de l'environnement RL, le rendant non fonctionnel pour simuler une partie complète.
    * Impact: L'agent ne peut pas apprendre les stratégies liées au déplacement, à l'acquisition de propriétés ou au paiement/collecte de loyers, qui sont centrales au Monopoly.

* **Système d'Échange (Trade) Simpliste (Priorité Haute)**:
    * L'action `action_type == 3` (Propriété contre Propriété) est irréaliste : elle échange la première propriété du joueur contre une propriété spécifique de l'adversaire, sans permettre au joueur de choisir quelle propriété offrir.
    * Les échanges (types 2 et 3) ne modélisent aucune forme de négociation ou d'acceptation par l'adversaire. L'échange est implicitement accepté s'il est techniquement possible (possession, fonds).
    * Impact: L'agent ne peut pas apprendre de stratégies d'échange complexes et réalistes, un aspect crucial du jeu de haut niveau.

### 2.3. Autres Points d'Amélioration

* Gestion des Actions Valides et Masques: La validité de certaines actions (ex: construire sur une propriété valide) est vérifiée après le choix de l'agent, entraînant une pénalité. Les masques (`action_masks`) pourraient être utilisés plus en amont pour restreindre directement les choix de l'agent (par exemple, ne proposer que les `property_idx` valides pour l'action sélectionnée).
* Granularité des Actions: `step` exécute une seule action de gestion puis passe au joueur suivant. Un joueur devrait pouvoir effectuer plusieurs actions de gestion durant son tour. Le flux actuel ne le permet pas.
* Fonction de Récompense (`_calculate_reward`): La récompense actuelle basée sur l'état (argent, nb propriétés, nb maisons) donnée à chaque step peut introduire des biais (ex: pénaliser la construction à court terme) et nécessite probablement un ajustement fin (reward shaping). Envisager des récompenses plus "sparses" ou basées sur le changement de la valeur nette.
* Gestion des Adversaires (Observation/Action): L'utilisation d'index relatifs (0-2 pour `trade_partner`) peut être source de confusion si des joueurs font faillite. L'observation de taille fixe pour les adversaires est fonctionnelle mais nécessite que l'agent utilise bien le masque `active_players`.
* Règles de Monopoly Manquantes: Des règles importantes sont absentes de `MonopolyRLEnv` : gestion détaillée de la prison, effet des lancers de doubles, cartes Chance/Caisse de Communauté, enchères lorsqu'un joueur refuse d'acheter, processus complet de faillite.
* Absence de Truncation: L'environnement n'implémente pas de condition `truncated = True`, risquant des épisodes infiniment longs.
* Précision des Espaces (Observation): Mineur: La définition de `others_properties` comme `Box(low=0, high=5, ...)` est fonctionnelle mais `high=1` serait sémantiquement plus précis si l'encodage est binaire (0/1).

## 3. Recommandations Priorisées

Il est recommandé d'aborder les points suivants, par ordre de priorité décroissante :
* (voir `Game.py` pour l'utilisation de méthodes déjà existante). 

**Priorité Haute:**

* **Refondre `MonopolyRLEnv.step` pour inclure la Logique de Jeu Fondamentale**:
    * Décider du périmètre exact de `step` (un tour complet ? une phase ?).
    * Implémenter impérativement: Lancer de dés, déplacement, logique d'atterrissage sur les cases (achat/loyer/carte/prison/etc.), gestion des tours (doubles, fin de tour).
    * Intégrer la possibilité d'effectuer des actions de gestion (construire, hypothéquer...) après la phase de déplacement/atterrissage, au moment approprié du tour.
* **Concevoir un Système d'Échange (Trade) Réaliste**:
    * Modifier l'espace d'action pour permettre de spécifier clairement l'offre (propriétés offertes/demandées, argent offert/demandé).
    * Implémenter un mécanisme d'acceptation/refus (même simple au début, basé sur une heuristique pour l'adversaire simulé).
* **Ajouter et completer l'environement de jeu**:
    * Ajout d'un max de maison.
    * Triple double dés = prison (excés de vitesse).

**Priorité Moyenne:**

* Définir la Granularité des Actions par Tour: Permettre potentiellement plusieurs actions de gestion par tour et définir une action ou condition claire de "fin de tour".
* Itérer sur la Fonction de Récompense: Expérimenter avec des récompenses basées sur le changement de valeur nette, des récompenses "sparse" pour événements majeurs, et ajuster les pondérations.
* Implémenter les Règles Manquantes: Ajouter progressivement la gestion détaillée de la prison, des doubles, des cartes, des enchères, et du processus de faillite dans `MonopolyRLEnv`.
* Améliorer l'Utilisation des Masques d'Action: Utiliser les masques pour restreindre plus finement les choix de l'agent avant qu'une action soit sélectionnée, réduisant les actions invalides.
* Ajouter la Truncation: Implémenter une limite maximale de tours/steps pour terminer les épisodes (`truncated = True`).

**Priorité Basse:**

* Fiabiliser la Gestion des Adversaires: Envisager d'utiliser des ID de joueur fixes (0-3) pour les actions comme `trade_partner`.
* Ajuster la Définition des Espaces d'Observation: Rendre les limites des `Box` plus précises si nécessaire (ex: `high=1` pour `others_properties` si binaire).

---