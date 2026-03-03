import pandas as pd

# 1. Chargement de tes fichiers
df_features = pd.read_csv("global_features.csv")  # Ton gros CSV
df_xtrain_ids = pd.read_csv("../../data/X_train.csv",sep=';')        # Contient les IDs pour le train
df_xtest_ids = pd.read_csv("../../data/X_test.csv",sep=';')          # Contient les IDs pour le test
df_ytrain = pd.read_csv("../../data/y_train.csv",sep=';')            # Contient video_id et le score

# 2. Séparation des features
# On filtre le gros CSV en fonction des IDs présents dans X_train et X_test
X_train_features = df_features[df_features['video_id'].isin(df_xtrain_ids['id'])]
X_test_features = df_features[df_features['video_id'].isin(df_xtest_ids['id'])]

# 3. Alignement de Y_train avec X_train_features
# Important : il faut que l'ordre des lignes soit le même entre X et Y !
# Le plus sûr est de faire un merge pour coller le score directement aux features
df_ytrain = df_ytrain.rename(columns={'ID': 'video_id'})
train_complete = pd.merge(X_train_features, df_ytrain, on='video_id')

# Maintenant on sépare à nouveau pour l'entraînement
y_train_final = train_complete['popularity'] # Remplace par le nom exact de ta colonne
X_train_final = train_complete.drop(columns=['popularity']) # On garde l'ID pour l'instant ou on le drop aussi

# --- Sauvegarde en CSV (Classique) ---
X_train_final.to_csv("X_train.csv", index=False)
y_train_final.to_csv("y_train.csv", index=False)
X_test_features.to_csv("X_test.csv", index=False)
