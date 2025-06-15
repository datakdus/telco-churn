import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# 🔧 Chargement du modèle et du scaler
# ----------------------------
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
df_ref = pd.read_csv("df_model.csv")
feature_names = df_ref.drop("Churn", axis=1).columns.tolist()

# ----------------------------
# 🎨 Configuration générale
# ----------------------------
st.set_page_config(page_title="Prédiction du Churn", page_icon="📉", layout="centered")

# 🌐 CSS personnalisé
st.markdown("""
<style>
/* 🌍 Arrière-plan général */
body {
    background-color: #f8f9fa;
    font-family: 'Segoe UI', sans-serif;
    color: #333;
}

/* 📌 Titres */
h1, h2, h3, h4 {
    color: #003366;
    font-weight: 700;
}

/* 🔘 Boutons */
.stButton > button {
    background-color: #003366;
    color: white;
    border-radius: 8px;
    padding: 0.5em 1em;
    font-weight: bold;
    border: none;
    transition: background-color 0.3s ease;
}
.stButton > button:hover {
    background-color: #0055a5;
}

/* 📥 Fichier uploader */
.css-1fcdlhz {
    background-color: #ffffff;
    border: 1px solid #ccc;
    border-radius: 10px;
}

/* 📊 Conteneurs de métriques */
[data-testid="metric-container"] {
    background-color: #ffffff;
    border: 1px solid #ddd;
    border-radius: 12px;
    padding: 1rem;
    margin: 0.5rem 0;
}

/* 📎 Tableaux */
.stDataFrame {
    border: none;
    border-radius: 10px;
    overflow: hidden;
}

/* 📈 Graphiques */
div[data-testid="stPyplot"] {
    background-color: white;
    padding: 1rem;
    border-radius: 12px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
}

/* 🔗 Liens */
a {
    color: #0055a5;
    text-decoration: none;
}
a:hover {
    text-decoration: underline;
}

/* 📋 Sidebar */
.stSidebar {
    background-color: #f0f4f8;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------
# 🖼️ Affichage du logo dans la sidebar + Informations
# ----------------------------
with st.sidebar:
    st.image("logo.png", width=180)
    st.markdown("### 📉 Prédiction du Churn")
    st.markdown("Anticipez les départs clients grâce à la data science.")

    st.markdown("---")
    st.markdown("👨‍💼 [LinkedIn](https://www.linkedin.com/in/ulrich-kouassi-275081266)")
    st.markdown("📫 zoubradaya@gmail.com")
    st.markdown("🧑‍💻 par **Ulrich Kouassi**")
    st.markdown("📅 v1.0 – 2025")

    st.markdown("---")
    with st.expander("ℹ️ Comment le modèle prédit ?"):
        st.markdown("""
        - Chaque client est analysé en fonction de ses caractéristiques (durée, services, montant, etc.).
        - Le modèle estime la probabilité de départ (`Churn_Proba`) et donne une prédiction (`Churn_Pred`).
        - Les seuils de recommandation sont ajustés selon les meilleures pratiques métiers.
        """)

# ----------------------------
# 📌 Barre latérale de navigation
# ----------------------------
st.sidebar.title("🔍 Navigation")
menu = st.sidebar.radio("Choisir une page :", [
    "📁 Prédiction via CSV",
    "👤 Prédiction manuelle",
    "📘 À propos"
])

# ----------------------------
# 📁 Prédiction via fichier CSV
# ----------------------------
if menu == "📁 Prédiction via CSV":
    st.title("📁 Prédiction en batch (via fichier CSV)")
    st.markdown("""
    Chargez un fichier avec les données des clients pour estimer leur **probabilité de churn** (résiliation).
    """)

    # 🔧 Création automatique du fichier exemple s’il n’existe pas
    import os
    if not os.path.exists("exemple_input_100.csv"):
        data = pd.DataFrame({
            "gender": np.random.choice(["Male", "Female"], 100),
            "SeniorCitizen": np.random.choice([0, 1], 100),
            "Partner": np.random.choice(["Yes", "No"], 100),
            "Dependents": np.random.choice(["Yes", "No"], 100),
            "tenure": np.random.randint(1, 72, 100),
            "PhoneService": np.random.choice(["Yes", "No"], 100),
            "MultipleLines": np.random.choice(["Yes", "No", "No phone service"], 100),
            "InternetService": np.random.choice(["DSL", "Fiber optic", "Dial-up", "No"], 100),
            "OnlineSecurity": np.random.choice(["Yes", "No"], 100),
            "OnlineBackup": np.random.choice(["Yes", "No"], 100),
            "DeviceProtection": np.random.choice(["Yes", "No"], 100),
            "TechSupport": np.random.choice(["Yes", "No"], 100),
            "StreamingTV": np.random.choice(["Yes", "No"], 100),
            "StreamingMovies": np.random.choice(["Yes", "No"], 100),
            "Contract": np.random.choice(["Month-to-month", "One year", "Two year"], 100),
            "PaperlessBilling": np.random.choice(["Yes", "No"], 100),
            "PaymentMethod": np.random.choice(["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"], 100),
            "MonthlyCharges": np.round(np.random.uniform(20, 120, 100), 2),
            "TotalCharges": np.round(np.random.uniform(20, 8000, 100), 2),
        })

        # Remplacer "No internet service" par "No"
        for col in ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]:
            data.loc[data["InternetService"] == "No", col] = "No"

        # Remplacer "No phone service" par "No"
        data.loc[data["PhoneService"] == "No", "MultipleLines"] = "No"

        data.to_csv("exemple_input_100.csv", index=False)

    # 📥 Exemple CSV téléchargeable (100 clients réalistes)
    with open("exemple_input_100.csv", "r") as f:
        st.download_button(
            label="📥 Télécharger un fichier CSV exemple (100 clients)",
            data=f.read(),
            file_name="exemple_input_100.csv",
            mime="text/csv"
        )

    uploaded_file = st.file_uploader("📤 Charger un fichier CSV", type="csv", key="file_uploader_batch")

    if uploaded_file:
        try:
            # Lecture du fichier CSV
            df_input = pd.read_csv(uploaded_file)
            st.success("✅ Fichier chargé avec succès.")
            st.dataframe(df_input.head())

            # Nettoyage des cellules vides ou contenant uniquement des espaces
            df_input.replace(r'^\s*$', np.nan, regex=True, inplace=True)

            # Alerte si valeurs manquantes
            nb_missing = df_input.isnull().sum().sum()
            if nb_missing > 0:
                st.warning(f"⚠️ {nb_missing} valeurs manquantes détectées. Les lignes incomplètes seront supprimées.")
                df_input.dropna(inplace=True)

            # Standardisation des colonnes binaires
            cols_to_fix = ["MultipleLines", "OnlineSecurity", "OnlineBackup",
                           "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
            for col in cols_to_fix:
                if col in df_input.columns:
                    df_input[col] = df_input[col].replace({
                        "No internet service": "No",
                        "No phone service": "No"
                    })

            # Dictionnaire d'encodage corrigé
            label_dict = {
                "Yes": 1, "No": 0,
                "Male": 1, "Female": 0,
                "Month-to-month": 0, "One year": 1, "Two year": 2,
                "Electronic check": 0, "Mailed check": 1,
                "Bank transfer (automatic)": 2, "Credit card (automatic)": 3,
                "DSL": 0, "Fiber optic": 1, "Dial-up": 2, "No": 3
            }

            df_input.replace(label_dict, inplace=True)

            # Vérification des colonnes nécessaires
            missing_cols = [col for col in feature_names if col not in df_input.columns]
            if missing_cols:
                st.error(f"❌ Colonnes manquantes : {missing_cols}")
            else:
                try:
                    df_input[feature_names] = df_input[feature_names].astype(float)
                except Exception as e:
                    st.error(f"⚠️ Certaines colonnes contiennent des types non numériques : {e}")
                    st.stop()

                # Prédiction
                X_scaled = scaler.transform(df_input[feature_names])
                proba = model.predict_proba(X_scaled)[:, 1]
                prediction = (proba > 0.5).astype(int)

                # Résultats
                df_input["Churn_Proba"] = np.round(proba, 2)
                df_input["Churn_Pred"] = prediction

                st.subheader("📊 Résultats")
                st.dataframe(df_input[["Churn_Proba", "Churn_Pred"]].head())

                # Top 10 clients à risque
                st.subheader("📋 Top 10 clients à risque")
                top_clients = df_input.sort_values("Churn_Proba", ascending=False).head(10)
                st.dataframe(top_clients)

                # Statistiques
                churn_rate = df_input["Churn_Pred"].mean()
                nb_clients = len(df_input)
                nb_churn = df_input["Churn_Pred"].sum()
                nb_fidèles = nb_clients - nb_churn

                st.subheader("📌 Résumé global")
                st.write(f"👥 Nombre total de clients analysés : **{nb_clients}**")
                st.write(f"📉 Nombre de clients à risque de churn : **{nb_churn}**")
                st.write(f"✅ Clients fidèles : **{nb_fidèles}**")
                st.metric("🔥 Taux de churn prédit", f"{churn_rate:.2%}")

                # Recommandations globales
                st.subheader("💡 Recommandation globale")
                if churn_rate > 0.30:
                    st.error("🚨 Taux de churn élevé")
                    st.markdown("""
                    **Actions recommandées :**
                    - Identifier les segments les plus à risque
                    - Lancer une campagne de rétention
                    - Offrir des avantages ciblés aux clients à risque
                    """)
                elif churn_rate > 0.10:
                    st.warning("⚠️ Taux de churn modéré")
                    st.markdown("""
                    **Actions recommandées :**
                    - Suivre les clients à risque
                    - Tester des offres personnalisées
                    """)
                else:
                    st.success("🎉 Taux de churn faible")
                    st.markdown("""
                    ✅ Continuer les bonnes pratiques :
                    - Maintenir la qualité de service
                    - Fidéliser les clients
                    """)

                # Visualisations
                st.subheader("📊 Visualisations")
                fig1, ax1 = plt.subplots()
                ax1.pie([nb_fidèles, nb_churn], labels=["Fidèles", "Churn"], autopct='%1.1f%%',
                        colors=["green", "red"], startangle=90)
                ax1.axis("equal")
                st.pyplot(fig1)

                fig2, ax2 = plt.subplots()
                ax2.hist(df_input["Churn_Proba"], bins=10, color="skyblue", edgecolor="black")
                ax2.set_xlabel("Probabilité de churn")
                ax2.set_ylabel("Nombre de clients")
                st.pyplot(fig2)

                # Téléchargement
                st.download_button(
                    "📥 Télécharger les résultats (CSV)",
                    data=df_input.to_csv(index=False),
                    file_name="resultats_churn.csv",
                    mime="text/csv"
                )

                st.success("✅ Prédiction batch terminée avec succès.")

        except Exception as e:
            st.error(f"❌ Erreur inattendue lors du traitement : {e}")

# ----------------------------
# 👤 Prédiction manuelle
# ----------------------------
elif menu == "👤 Prédiction manuelle":
    st.title("👤 Prédiction manuelle")
    st.markdown("Remplissez les champs pour obtenir une prédiction personnalisée.")

    # Champs à remplir
    gender = st.sidebar.selectbox("🧑 Genre", ["Male", "Female"])
    senior = st.sidebar.selectbox("🧓 Le client est-il senior ?", [0, 1])
    partner = st.sidebar.selectbox("👫 A-t-il un(e) partenaire ?", ["Yes", "No"])
    dependents = st.sidebar.selectbox("👶 A-t-il des personnes à charge ?", ["Yes", "No"])
    tenure = st.sidebar.slider("📆 Ancienneté (mois)", 0, 72, 12)
    phoneservice = st.sidebar.selectbox("📞 A-t-il un service téléphonique ?", ["Yes", "No"])
    multiplelines = st.sidebar.selectbox("📱 Lignes multiples ?", ["Yes", "No"])
    internet = st.sidebar.selectbox("🌐 Type d'internet", ["DSL", "Fiber optic", "No"])
    onlinesecurity = st.sidebar.selectbox("🔐 Sécurité en ligne ?", ["Yes", "No"])
    onlinebackup = st.sidebar.selectbox("💾 Sauvegarde en ligne ?", ["Yes", "No"])
    deviceprotection = st.sidebar.selectbox("📱 Protection d'appareil ?", ["Yes", "No"])
    techsupport = st.sidebar.selectbox("🛠️ Support technique ?", ["Yes", "No"])
    streamingtv = st.sidebar.selectbox("📺 Streaming TV ?", ["Yes", "No"])
    streamingmovies = st.sidebar.selectbox("🎬 Streaming films ?", ["Yes", "No"])
    contract = st.sidebar.selectbox("📝 Type de contrat", ["Month-to-month", "One year", "Two year"])
    paperless = st.sidebar.selectbox("📧 Facturation sans papier ?", ["Yes", "No"])
    payment = st.sidebar.selectbox("💳 Méthode de paiement", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    monthly = st.sidebar.number_input("💰 Frais mensuels", 0.0, 1000.0, 70.0)
    total = st.sidebar.number_input("💵 Frais totaux", 0.0, 10000.0, 1200.0)

    if st.sidebar.button("🔮 Prédire"):
        try:
            # 1. Dictionnaire brut
            input_dict = {
                "gender": gender,
                "SeniorCitizen": senior,
                "Partner": partner,
                "Dependents": dependents,
                "tenure": tenure,
                "PhoneService": phoneservice,
                "MultipleLines": multiplelines,
                "InternetService": internet,
                "OnlineSecurity": onlinesecurity,
                "OnlineBackup": onlinebackup,
                "DeviceProtection": deviceprotection,
                "TechSupport": techsupport,
                "StreamingTV": streamingtv,
                "StreamingMovies": streamingmovies,
                "Contract": contract,
                "PaperlessBilling": paperless,
                "PaymentMethod": payment,
                "MonthlyCharges": monthly,
                "TotalCharges": total
            }

            # 2. Encodage
            label_dict = {
                "Yes": 1, "No": 0,
                "Male": 1, "Female": 0,
                "Month-to-month": 0, "One year": 1, "Two year": 2,
                "Electronic check": 0, "Mailed check": 1,
                "Bank transfer (automatic)": 2, "Credit card (automatic)": 3,
                "DSL": 0, "Fiber optic": 1, "No": 2
            }

            input_encoded = {k: label_dict.get(v, v) for k, v in input_dict.items()}

            # 3. Liste des variables dans le bon ordre
            feature_names = [
                "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
                "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
                "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
                "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
                "MonthlyCharges", "TotalCharges"
            ]

            # 4. Création du DataFrame
            X_input = pd.DataFrame([input_encoded])[feature_names]

            # 5. Mise à l’échelle
            X_scaled = scaler.transform(X_input)

            # 6. Prédiction
            proba = model.predict_proba(X_scaled)[0, 1]
            prediction = int(proba > 0.5)

            st.metric("📈 Probabilité de churn", f"{proba:.2%}")

            if prediction == 1:
                st.error("⚠️ Risque élevé de churn")
                st.markdown("""
                ### 💡 Recommandations :
                - Offrir une réduction ou un bonus de fidélité
                - Proposer une réévaluation du forfait
                - Appeler le client pour comprendre ses besoins
                """)
            else:
                st.success("🎉 Client fidèle")
                st.markdown("""
                ### ✅ Recommandations :
                - Continuer à offrir un service de qualité
                - Envoyer une enquête de satisfaction
                - Récompenser la fidélité avec un avantage
                """)
        except Exception as e:
            st.error(f"❌ Erreur pendant la prédiction : {e}")

# ----------------------------
# 📘 À propos
# ----------------------------
else:
    st.title("📘 À propos du projet")
    st.markdown("""
    ### 🎯 Objectif
    Cette application vise à prédire la probabilité qu’un client quitte son abonnement (churn) à partir de ses informations contractuelles et comportementales.

    ### 🧠 Modèle utilisé
    - **Algorithme** : Random Forest Classifier
    - **Traitement des données** :
        - Encodage des variables catégorielles
        - Redressement du déséquilibre avec SMOTE
        - Mise à l’échelle des variables numériques avec StandardScaler

    ### 📊 Fonctionnalités prises en compte
    - Données client : `SeniorCitizen`, `tenure`, `MonthlyCharges`, `TotalCharges`
    - Services souscrits : Internet, Sécurité, Support, etc.
    - Type de contrat, méthode de paiement, engagement...

    ### 💼 Modes d'utilisation
    - **📁 Prédiction en masse** via un fichier CSV contenant plusieurs clients
    - **👤 Prédiction manuelle** d’un client via formulaire interactif
    - **📊 Visualisation** des résultats (histogrammes, métriques)
    - **📌 Recommandations métier** en fonction du profil du client

    ---
    🔗 **Code source** : [github.com/datakdus](https://github.com/datakdus)  
    🧑‍💼 **Développé par** : Ulrich / [datakdus](https://github.com/datakdus)  
    📅 **Année** : 2025
    """)
