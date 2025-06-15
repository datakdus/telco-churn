import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE

# 1. Chargement des données
df = pd.read_csv("Telco-Customer-Churn.csv")

# 2. Nettoyage de base
df.drop_duplicates(inplace=True)

# Suppression des espaces blancs dans les noms de colonnes
df.columns = df.columns.str.strip()

# Suppression des lignes avec des valeurs manquantes dans TotalCharges
df = df[df["TotalCharges"] != " "]
df["TotalCharges"] = df["TotalCharges"].astype(float)

# 3. Encodage des variables catégorielles
df = df.drop(["customerID"], axis=1)
for col in df.select_dtypes("object").columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# 4. Séparation X / y
X = df.drop("Churn", axis=1)
y = df["Churn"]

# 5. Traitement du déséquilibre des classes
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# 6. Scaling des données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_resampled)

# 7. Entraînement du modèle
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_scaled, y_resampled)

# 8. Sauvegarde du modèle et du scaler
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

# 9. Sauvegarde du dataset préparé
df_model = pd.DataFrame(X_scaled, columns=X.columns)
df_model["Churn"] = y_resampled
df_model.to_csv("df_model.csv", index=False)

print("✅ Pipeline terminé : modèle, scaler et données sauvegardés.")
