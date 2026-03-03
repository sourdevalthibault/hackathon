import pandas as pd
from functools import reduce
import os

def merge_tiktok_datasets(file_list, output_name="keyframes_final_features.csv"):
    """
    Fusionne plusieurs CSV sur la base de leur première colonne.
    """
    dfs = []
    
    for file in file_list:
        print(f"Chargement de {file}...")
        # On charge le CSV
        df = pd.read_csv(file)
        
        # On s'assure que la première colonne (ID) est bien traitée comme un string 
        # (évite les problèmes de formatage scientifique sur les IDs longs)
        id_col = df.columns[0]
        df[id_col] = df[id_col].astype(str)
        
        dfs.append(df)

    # Fusion itérative (Merge)
    # 'outer' permet de garder toutes les vidéos même si elles manquent dans certains fichiers
    # 'inner' ne garderait que les vidéos présentes dans ABSOLUMENT tous les fichiers
    print("Fusion en cours...")
    df_final = reduce(lambda left, right: pd.merge(
        left, 
        right, 
        on=left.columns[0], 
        how='outer'
    ), dfs)

    # 2. Modifier l'ID (Enlever le préfixe 'VIDEO_')
    # On cible la première colonne (index 0)
    id_col_name = df_final.columns[0]

    # Méthode rapide avec .str.replace
    df_final[id_col_name] = df_final[id_col_name].str.replace('VIDEO_', '', regex=False)

    # Vérification
    print("Colonnes restantes :", df_final.columns.tolist())
    print("Aperçu des IDs :")
    print(df_final[id_col_name].head())

    # Sauvegarde du résultat
    df_final.to_csv(output_name, index=False)
    print(f"Terminé ! Fichier sauvegardé sous : {output_name}")
    print(f"Dimensions finales : {df_final.shape}")
    return df_final

# --- CONFIGURATION ---
# Liste tes fichiers ici
my_files = ['features_statics.csv', 'features_dynamics.csv', 'features_semantics.csv', 'features_humans.csv']

# Exécution
df_competition = merge_tiktok_datasets(my_files)