# MLOps Project: The ML Factory

### Vision du projet

L'objectif est de construire une infrastructure "Zero-Downtime". Vous allez créer une chaîne où le modèle de Machine Learning est totalement découplé de l'application qui le consomme. Grâce à l'utilisation d'un **Model Registry** et d'un **Object Storage**, vous apprendrez à mettre à jour l'intelligence d'une API sans jamais redémarrer un seul conteneur.

Projet_2_1_MLFactory.md
4 Ko
﻿
# MLOps Project: The ML Factory

### Vision du projet

L'objectif est de construire une infrastructure "Zero-Downtime". Vous allez créer une chaîne où le modèle de Machine Learning est totalement découplé de l'application qui le consomme. Grâce à l'utilisation d'un **Model Registry** et d'un **Object Storage**, vous apprendrez à mettre à jour l'intelligence d'une API sans jamais redémarrer un seul conteneur.

### Architecture du Workspace

Le projet repose sur une isolation stricte des services. Chaque composant possède son propre environnement et sa propre image Docker.

* **`src/train/`** : Le laboratoire. C'est ici que vous expérimentez et publiez vos modèles vers le registre (script lancé hors Docker).
* **`src/api/`** : L'usine. Une API FastAPI qui sert les prédictions en interrogeant dynamiquement MLflow.
* **`src/front/`** : La vitrine. Une interface Streamlit pour permettre aux utilisateurs de tester les modèles.
* **Infrastructure** : Pilotée par un `docker-compose.yml` incluant **MLflow** (le catalogue) et **MinIO** (le hangar de stockage S3).

```text
ml-factory-project/
├── data/ iris_test.csv      
├── src/
│   ├── api/ (main.py + Dockerfile)
│   ├── front/ (app.py + Dockerfile)
│   └── train/ (train.py)
├── docker-compose.yml       
├── pyproject.toml (UV)           
└── .env                     

```

### Configuration de l'Infrastructure

Avant de coder, vous devez monter l'usine. Votre fichier `docker-compose.yml` est le plan de montage.

* **Stockage Persistant** : Configurez **MinIO** avec un volume Docker pour ne pas perdre vos modèles au premier redémarrage.
* **Registre central** : **MLflow** doit être configuré pour pointer vers MinIO via les variables d'environnement S3 standard (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`).
* **Variables d'environnement** : Centralisez tout dans un `.env`. L'API et le Front doivent connaître l'URL interne de MLflow (`http://mlflow:5000`).

### Développement du Serving (FastAPI)

L'API est la pièce maîtresse. Elle ne doit pas charger un modèle local, mais utiliser le **MlflowClient** pour être "réactive".

* **Inférence Dynamique** : À chaque appel, l'API vérifie si l'alias `"Production"` a été déplacé sur une nouvelle version.
* **Hot-reloading** : Si une nouvelle version est détectée, le modèle est rechargé à chaud depuis MinIO.
* **Transparence** : La réponse de l'API doit inclure la classe prédite ET l'ID de la version du modèle utilisé pour garantir la traçabilité.

### Interface de Test (Streamlit)

Le front-end permet de simuler une utilisation réelle.

* Proposez de charger une ligne de test du dataset `iris_test.csv`.
* Affichez un indicateur visuel clair (une bannière ou un badge) indiquant la **version actuelle** du modèle en ligne.
* Affichez les probabilités de prédiction pour rendre l'interface plus vivante.

---

### Le Scénario de Validation : Deux approches

Le TP se valide en observant la transition entre deux modèles de complexité différente.

#### Phase 1 : Le passage en force (Automation)

1. Entraînez une **Régression Logistique** avec `train.py`.
2. Dans votre code, utilisez `client.set_registered_model_alias()` pour forcer automatiquement ce modèle en `"Production"`.
3. Vérifiez sur Streamlit que la **Version 1** est active et fonctionnelle.

#### Phase 2 : Le choix du chef (Manuel)

1. Modifiez votre script pour entraîner un **RandomForestClassifier**.
2. **Attention** : Commentez la ligne qui force l'alias en Production. Lancez le script.
3. Constatez que le modèle est bien dans MLflow (Version 2) mais que Streamlit utilise toujours la Version 1.
4. **Action Manuelle** : Allez sur l'UI MLflow (`localhost:5000`), sélectionnez la Version 2 et attribuez-lui manuellement l'alias `"Production"`.
5. Observez le changement instantané sur votre application Streamlit.
Projet_2_1_MLFactory.md
4 Ko