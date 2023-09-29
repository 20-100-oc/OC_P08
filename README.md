# Projet: Participez à la conception d'une voiture autonome

## Contexte
Au sein d'un département R&D pour voiture autonome, une équipe d'ingénieurs est chargée de concevoir le stystème de vision par ordinateur.
Cette tâche se sépare en plusieurs parties:
- acquisition des images en temps réel
- traitement des images
- segmentation des images
- système de décision

Je suis chargé d'intégrer la segmentation d'image en facilitant l'inclusion du travail des collègues suivants et précédents dans la chaîne du système.
Le but est de créer un modèle de segmentation qui classifie dans une catégorie donnée chaque pixel d'une photo, en l'occurence prise depuis une voiture à caméra embarquée.
Il faut ensuite permettre aux autres membres de l'équipe d'intéragir avec ce modèle à travers une API déployée.

## Objectifs
- Pré-traîtement des données (images et masques)
- Élaboration d'un modèle d'apprentissage pronfond adapté au images
- Augmentation des données pour améliorer les performances du modèle
- Déploiement cloud du modèle sous forme d'API avec Azure
- Création d'une application web de présentation pour intéragir avec l'API
- Rédaction d'un document détaillant la démarche de modélisation

## Livrables
- Notebooks et scripts du pipeline
- Modèle déployé sur le cloud en tant qu'API
- Application web d'interface avec l'API
- Note technique
- Présentation PowerPoint

## Outils
- Python
- Git / Github
- Jupyter notebook / Python IDE
- PowerPoint
- Streamlit
- Azure web app
- Google Storage

### Python : libraires additionnelles
- pandas
- numpy
- tensorflow
- albumentations
- h5py
- google-auth
- fastapi
- streamlit
- requests