from pathlib import Path
import pandas as pd

def merge_all_by_id(
    data_dir: str | Path,
    out_train_name: str = "merged_train.csv",
    out_test_name: str = "merged_test.csv",
    how: str = "inner",   # "inner" recommandé pour train ; pour test souvent "inner" aussi si tout a les mêmes IDs
) -> dict:
    """
    Cherche dans data_dir tous les fichiers *_train.csv et *_test.csv,
    les merge successivement sur la colonne ID, puis sauvegarde 2 fichiers.
    Met toujours ID en 1ère colonne.
    """
    data_dir = Path(data_dir)
    train_files = sorted(data_dir.glob("*_train.csv"))
    test_files = sorted(data_dir.glob("*_test.csv"))

    def _merge_files(files: list[Path], out_path: Path) -> tuple[str, tuple[int, int] | None]:
        if not files:
            return ("no_files", None)

        df = None
        for f in files:
            d = pd.read_csv(f)

            if "ID" not in d.columns:
                raise ValueError(f"{f.name} ne contient pas la colonne 'ID'.")

            # Assure ID en string (important)
            d["ID"] = d["ID"].astype(str)

            # Si jamais ID n'est pas 1ere col dans ce fichier, on le remet devant
            cols = ["ID"] + [c for c in d.columns if c != "ID"]
            d = d[cols]

            if df is None:
                df = d
            else:
                # Eviter collisions de noms de colonnes
                common = (set(df.columns) & set(d.columns)) - {"ID"}
                if common:
                    # On suffixe les colonnes du dataframe entrant
                    d = d.rename(columns={c: f"{c}__{f.stem}" for c in common})

                df = df.merge(d, on="ID", how=how)

        # Remet ID en 1ère colonne
        df = df[["ID"] + [c for c in df.columns if c != "ID"]]

        df.to_csv(out_path, index=False)
        return ("saved", df.shape)

    out_train = data_dir / out_train_name
    out_test = data_dir / out_test_name

    train_status, train_shape = _merge_files(train_files, out_train)
    test_status, test_shape = _merge_files(test_files, out_test)

    return {
        "train": {"status": train_status, "files": [f.name for f in train_files], "out": str(out_train), "shape": train_shape},
        "test": {"status": test_status, "files": [f.name for f in test_files], "out": str(out_test), "shape": test_shape},
    }


# Exemple d'utilisation
if __name__ == "__main__":
    ROOT = Path("/home/mohamed/SDD/hackathon/sia-predicting-short-form-video-popularity")
    res = merge_all_by_id(ROOT / "Data", how="inner")
    print(res)