# 📊 Prédiction du churn client dans le secteur des télécommunications

## 📌 Contexte

La rétention client est un enjeu majeur dans le secteur très concurrentiel des télécommunications. Le **churn** (ou attrition) représente la perte de clients et engendre des coûts importants. Pouvoir **anticiper les départs** permet à l'entreprise de mettre en place des actions préventives ciblées.

## 🎯 Objectif du projet

Ce projet vise à construire un système de machine learning capable de :

- Prédire si un client est à risque de churn (oui/non).
- Identifier les facteurs explicatifs les plus importants.
- Proposer une stratégie de rétention adaptée aux profils prédits à risque.

## 📂 Données utilisées

Le jeu de données contient 7043 clients avec :

- Des variables démographiques (sexe, senior, personnes à charge).
- Des données de service (Internet, sécurité, streaming, contrat).
- Des données financières (factures mensuelles et totales).
- Une variable cible `Churn` (Yes/No).

## 🔍 Étapes du projet

1. **Exploration et préparation des données**
   - Nettoyage, encodage, traitement des valeurs manquantes.
   - Analyse univariée et bivariée des variables explicatives.

2. **Modélisation**
   - Entraînement de plusieurs modèles : Logistic Regression, Decision Tree, Random Forest, XGBoost.
   - Gestion du déséquilibre des classes avec SMOTE.
   - Évaluation via F1-score, matrice de confusion, AUC/ROC.

3. **Application et stratégie**
   - Prédiction sur nouveaux clients (jeu de test).
   - Segmentation selon le risque (faible / moyen / élevé).
   - Recommandations d'actions ciblées pour chaque groupe.

## ✅ Résultats

- Le meilleur modèle est **Random Forest** avec :
  - F1-score (CV) : 0.85
  - AUC : 0.81
- Variables les plus importantes : type de contrat, ancienneté, sécurité en ligne, méthode de paiement.

## 💡 Recommandations business

- Prioriser les offres de rétention pour les clients mensuels sans services de sécurité.
- Inciter à des contrats longue durée via des avantages.
- Suivre mensuellement les churners potentiels via tableau de bord.

## 🚀 Perspectives

- Intégration dans une **application Streamlit** interne pour usage régulier.

## ⚙️ Usage

1. Ouvre le notebook `notebooks/churn.ipynb` avec **Jupyter Notebook** ou **Visual Studio Code**.
2. Charge les données depuis `data/telco_customer_churn.csv`.
3. Suis les étapes d’analyse et de modélisation décrites dans le notebook.

## 📁 Ressources

- Notebook : `notebooks/churn.ipynb`
- (À venir) Application Streamlit : lien ou description

## 🛠️ Installation

Pour installer les dépendances nécessaires, exécute la commande suivante dans ton terminal :

```bash
pip install -r requirements.txt
