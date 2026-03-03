import pandas as pd
from functools import reduce
import os

def merge_tiktok_datasets(file_list, output_name="global_features.csv"):
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


    # Sauvegarde du résultat
    df_final.to_csv(output_name, index=False)
    print(f"Terminé ! Fichier sauvegardé sous : {output_name}")
    print(f"Dimensions finales : {df_final.shape}")
    return df_final

# --- CONFIGURATION ---
# Liste tes fichiers ici
my_files = ['../video_features/keyframes_final_features.csv', '../others/audio.csv', '../others/metadata.csv', '../others/top_colors.csv', '../others/video_text.csv']

# Exécution
df_competition = merge_tiktok_datasets(my_files)