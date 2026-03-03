import pandas as pd

# Charger les deux fichiers
df1 = pd.read_csv("metadata_train.csv")
df2 = pd.read_csv("metadata_test.csv")

# Coller les lignes (l'un en dessous de l'autre)
# ignore_index=True permet de refaire une numérotation propre des lignes
df_final = pd.concat([df1, df2], ignore_index=True)
df_final = df_final.rename(columns={'id': 'video_id'})
# Sauvegarder
df_final.to_csv("metadata.csv", index=False)
print(f"✅ Fusion terminée : {len(df_final)} lignes au total.")

# Charger les deux fichiers
df1 = pd.read_csv("vggish_pooled_train.csv")
df2 = pd.read_csv("vggish_pooled_test.csv")

# Coller les lignes (l'un en dessous de l'autre)
# ignore_index=True permet de refaire une numérotation propre des lignes
df_final = pd.concat([df1, df2], ignore_index=True)
df_final = df_final.rename(columns={'ID': 'video_id'})
# Sauvegarder
df_final.to_csv("vggish.csv", index=False)
print(f"✅ Fusion terminée : {len(df_final)} lignes au total.")


# Charger les deux fichiers
df1 = pd.read_csv("attention_train.csv")
df2 = pd.read_csv("attention_test.csv")

# Coller les lignes (l'un en dessous de l'autre)
# ignore_index=True permet de refaire une numérotation propre des lignes
df_final = pd.concat([df1, df2], ignore_index=True)
df_final = df_final.rename(columns={'ID': 'video_id'})
# Sauvegarder
df_final.to_csv("attention.csv", index=False)
print(f"✅ Fusion terminée : {len(df_final)} lignes au total.")


# Charger les deux fichiers
df1 = pd.read_csv("text_train.csv")
df2 = pd.read_csv("text_test.csv")

# Coller les lignes (l'un en dessous de l'autre)
# ignore_index=True permet de refaire une numérotation propre des lignes
df_final = pd.concat([df1, df2], ignore_index=True)
df_final = df_final.iloc[:, 1:]
# 2. Modifier l'ID (Enlever le préfixe 'VIDEO_')
# On cible la première colonne (index 0)
id_col_name = df_final.columns[0]

# Méthode rapide avec .str.replace
df_final[id_col_name] = df_final[id_col_name].str.replace('VIDEO_', '', regex=False)

# Sauvegarder
df_final.to_csv("video_text.csv", index=False)
print(f"✅ Fusion terminée : {len(df_final)} lignes au total.")

# Charger les deux fichiers
df1 = pd.read_csv("top_colors_train.csv")
df2 = pd.read_csv("top_colors_test.csv")

# Coller les lignes (l'un en dessous de l'autre)
# ignore_index=True permet de refaire une numérotation propre des lignes
df_final = pd.concat([df1, df2], ignore_index=True)
df_final = df_final.iloc[:, 1:]
# 2. Modifier l'ID (Enlever le préfixe 'VIDEO_')
# On cible la première colonne (index 0)
id_col_name = df_final.columns[0]

# Méthode rapide avec .str.replace
df_final[id_col_name] = df_final[id_col_name].str.replace('VIDEO_', '', regex=False)

# Sauvegarder
df_final.to_csv("top_colors.csv", index=False)
print(f"✅ Fusion terminée : {len(df_final)} lignes au total.")

#----------------AUDIO--------------------------
df = pd.read_csv("audio_features.csv")
cols = list(df.columns)
# 2. Réorganiser : la dernière + tout le reste sauf la dernière
cols = [cols[-1]] + cols[:-1]
# 3. Appliquer le nouvel ordre au DataFrame
df = df[cols]
df = df.rename(columns={'ID': 'video_id'})
df.to_csv("audio.csv", index=False)
print(f"✅ Fusion terminée : {len(df)} lignes au total.")

