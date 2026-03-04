from pathlib import Path
import pandas as pd

ROOT = Path("/home/mohamed/SDD/hackathon/sia-predicting-short-form-video-popularity")
DATA_DIR = ROOT / "Data"

# -------- TRAIN --------
x_train = pd.read_csv(DATA_DIR / "X_train_thib.csv")
merged_train = pd.read_csv(DATA_DIR / "merged_train.csv")

# supprimer les colonnes dupliquées dans merged_train
merged_train = merged_train.drop(columns=[c for c in merged_train.columns if c in x_train.columns], errors="ignore")

# merge horizontal
train_final = pd.concat([x_train, merged_train], axis=1)

# sauvegarde
train_final.to_csv(DATA_DIR / "X_train_merged.csv", index=False)


# -------- TEST --------
x_test = pd.read_csv(DATA_DIR / "X_test_thib.csv")
merged_test = pd.read_csv(DATA_DIR / "merged_test.csv")

merged_test = merged_test.drop(columns=[c for c in merged_test.columns if c in x_test.columns], errors="ignore")

test_final = pd.concat([x_test, merged_test], axis=1)

test_final.to_csv(DATA_DIR / "X_test_merged.csv", index=False)

print("Merge terminé.")
print("Train shape:", train_final.shape)
print("Test shape:", test_final.shape)
from pathlib import Path
import pandas as pd

ROOT = Path("/home/mohamed/SDD/hackathon/sia-predicting-short-form-video-popularity")
DATA_DIR = ROOT / "Data"

def col_report(train_path, test_path, name=""):
    train = pd.read_csv(train_path, nrows=5)  # on lit juste l'entête + qq lignes (rapide)
    test  = pd.read_csv(test_path,  nrows=5)

    train_cols = set(train.columns)
    test_cols  = set(test.columns)

    only_train = sorted(train_cols - test_cols)
    only_test  = sorted(test_cols - train_cols)
    common     = sorted(train_cols & test_cols)

    print(f"\n=== {name} ===")
    print(f"Train cols: {len(train_cols)} | Test cols: {len(test_cols)} | Common: {len(common)}")
    print(f"Only in TRAIN: {len(only_train)}")
    print(f"Only in TEST : {len(only_test)}")

    # Affiche un aperçu
    if only_train:
        print("Exemples only_train:", only_train[:20])
    if only_test:
        print("Exemples only_test :", only_test[:20])

    return only_train, only_test, common

# 1) Comparer tes 2 gros fichiers séparément
col_report(DATA_DIR/"X_train_thib.csv",    DATA_DIR/"X_test_thib.csv",    name="X_thib")
col_report(DATA_DIR/"merged_train.csv",    DATA_DIR/"merged_test.csv",    name="merged")
col_report(DATA_DIR/"X_train_merged.csv",  DATA_DIR/"X_test_merged.csv",  name="final merged output")

# 2) Check de colonnes dupliquées dans chaque fichier (au cas où)
def dup_cols(path, name=""):
    df = pd.read_csv(path, nrows=5)
    dups = df.columns[df.columns.duplicated()].tolist()
    print(f"\nDup columns in {name}: {len(dups)}")
    if dups:
        print("Exemples:", dups[:50])

dup_cols(DATA_DIR/"X_train_merged.csv", "train_final")
dup_cols(DATA_DIR/"X_test_merged.csv",  "test_final")