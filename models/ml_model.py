import pandas as pd, numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error # Ou MSE selon ta compète

# 1. Chargement des données (Utilise tes noms de fichiers sauvegardés)
X_train = pd.read_csv("../features/global_features/X_train.csv")
y_train = pd.read_csv("../features/global_features/y_train.csv")
X_test = pd.read_csv("../features/global_features/X_test.csv")

# 2. Préparation
y_train = y_train.values.flatten()
test_ids = X_test['video_id']
X = X_train.drop(columns=['video_id'], errors='ignore')
X_test_final = X_test.drop(columns=['video_id'], errors='ignore')

# 3. Configuration du K-Fold
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# Listes pour stocker les scores de validation et les prédictions finales
oof_preds = np.zeros(len(X)) # Out-of-fold predictions
test_preds = np.zeros(len(X_test_final))
cv_scores = []

print(f"Début de la Cross-Validation ({n_splits} folds)...")

# 4. Boucle d'entraînement
for fold, (train_idx, val_idx) in enumerate(kf.split(X, y_train)):
    X_t, X_v = X.iloc[train_idx], X.iloc[val_idx]
    y_t, y_v = y_train[train_idx], y_train[val_idx]
    
    # Modèle pour ce fold
    model = lgb.LGBMRegressor(
        n_estimators=3000,
        learning_rate=0.03,
        num_leaves=63,
        objective='regression',
        random_state=42 + fold, # Un peu de variation
        n_jobs=-1
    )
    
    # Entraînement avec arrêt précoce
    model.fit(
        X_t, y_t,
        eval_set=[(X_v, y_v)],
        eval_metric='rmse',
        callbacks=[lgb.early_stopping(stopping_rounds=100), lgb.log_evaluation(period=0)]
    )
    
    
    # 5. Calcul des scores et affichage du RMSE
    val_preds = model.predict(X_v)
    oof_preds[val_idx] = val_preds
    
    # Calcul du RMSE pour ce fold : on prend la racine carrée du MSE
    from sklearn.metrics import mean_squared_error
    fold_rmse = mean_squared_error(y_v, val_preds) 
    cv_scores.append(fold_rmse)
    
    print(f"Fold {fold+1} RMSE: {fold_rmse:.4f}")
    
    # Moyennage des prédictions pour le test set
    test_preds += model.predict(X_test_final) / n_splits


print("-" * 30)
rmse_moyen = np.mean(cv_scores)
print(f"RMSE Moyen CV: {rmse_moyen:.4f} (+/- {np.std(cv_scores):.4f})")

# 6. Création du fichier de soumission
submission = pd.DataFrame({
    'ID': test_ids,
    'popularity': test_preds
})

submission.to_csv("submission.csv", index=False)
print("Fichier 'submission.csv' généré ! 🏆")