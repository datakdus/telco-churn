# 📉 Prédiction du Churn Client - Projet Freelance

Bienvenue dans ce projet de **data science appliquée à la fidélisation client**, visant à prédire la résiliation (`churn`) d'abonnés à partir de données comportementales et contractuelles.

---

## 🧠 Objectif du projet

Ce projet a pour but de :

- Prédire si un client risque de résilier son contrat (churn)
- Comprendre les variables influentes sur le churn
- Fournir des recommandations métier
- Proposer une **application web interactive** développée avec Streamlit

---

## 📂 Contenu du dossier

```
churn/
├── app.py                    ← Application Streamlit
├── model.pkl                 ← Modèle RandomForest entraîné
├── scaler.pkl                ← Scaler StandardScaler pour les prédictions
├── df_model.csv              ← Données préparées pour la modélisation
├── Telco-Customer-Churn.csv ← Données brutes (optionnel)
├── train_pipeline.py         ← Pipeline complet d'entraînement
├── churn_notebook.ipynb      ← Analyse exploratoire + modélisation
├── churn_notebook.html       ← Export HTML du notebook
├── requirements.txt          ← Dépendances Python
├── logo.png                  ← Logo affiché dans l'app Streamlit
└── README.md                 ← Ce fichier
```

---

## 📊 Données utilisées

Les données proviennent d’un jeu de données public de type Telco :  
- **Colonnes** : informations client, contrat, paiements, services souscrits  
- **Target** : `Churn` (binaire : 1 = résiliation, 0 = client fidèle)  
- **Nettoyage** : gestion des valeurs manquantes, encodage LabelEncoder, capping  
- **Rééquilibrage** : sur-échantillonnage avec SMOTE

---

## ⚙️ Pipeline de modélisation

- 🧹 Nettoyage des données (doublons, valeurs aberrantes)
- 🔢 Encodage des variables catégorielles
- ⚖️ Redressement du déséquilibre avec SMOTE
- 📏 Mise à l’échelle des variables numériques
- 🌲 Entraînement d’un modèle Random Forest
- 💾 Sauvegarde du modèle et des artefacts

📁 Script : `train_pipeline.py`

---

## 🖥️ Application Streamlit

L'app `app.py` permet :

- 📁 **Prédiction en masse** (batch) via fichier CSV
- 👤 **Prédiction manuelle** via formulaire interactif
- 📊 **Visualisation des résultats** (camembert, histogrammes)
- 📌 **Recommandations métier personnalisées**
- 📘 **Page "À propos"** avec détails techniques et auteur

---

## 🚀 Lancer l’application

Assurez-vous d’avoir Python 3.8+ installé, puis :

```bash
pip install -r requirements.txt
streamlit run app.py
```

> L'application s'ouvrira automatiquement dans votre navigateur.

---

## 🧑‍💻 Auteur

- **Nom** : Ulrich Kouassi  
- **Profil GitHub** : [@datakdus](https://github.com/datakdus)  
- **LinkedIn** : [linkedin.com/in/ulrich-kouassi-275081266](https://linkedin.com/in/ulrich-kouassi-275081266)

---

## 📌 À venir

- ✅ Déploiement sur Streamlit Cloud
- ✅ Ajout d’une prédiction manuelle plus complète
- ✅ Interprétabilité avec SHAP ou LIME

---
