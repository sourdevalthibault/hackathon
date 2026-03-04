from pathlib import Path
import pandas as pd
import re

def read_csv_robust(path: str | Path) -> pd.DataFrame:
    """
    Lit un CSV avec détection automatique du séparateur (, ou ;) et fallback engine=python.
    """
    path = Path(path)

    # 1) Essai auto-sep (marche très bien dans la majorité des cas)
    try:
        return pd.read_csv(path, sep=None, engine="python")
    except Exception:
        pass

    # 2) Essai virgule
    try:
        return pd.read_csv(path, sep=",", engine="python")
    except Exception:
        pass

    # 3) Essai point-virgule
    try:
        return pd.read_csv(path, sep=";", engine="python")
    except Exception as e:
        raise RuntimeError(f"Impossible de lire le CSV: {path}") from e


def prepare_features_csv(
    csv_path: str | Path,
    x_train_path: str | Path = "X_train.csv",
    x_test_path: str | Path = "X_test.csv",
    out_dir: str | Path = "Data",
    train_size: int = 1349,
    mixed_size: int = 1687,
    id_candidates: tuple[str, ...] = ("ID", "id", "video_id"),
) -> dict:
    csv_path = Path(csv_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = read_csv_robust(csv_path)

    # 1) Trouver/renommer la colonne ID
    found_id_col = next((c for c in id_candidates if c in df.columns), None)
    if found_id_col is None:
        raise ValueError(
            f"Aucune colonne ID trouvée dans {csv_path.name}. "
            f"Attendu parmi {id_candidates}. Colonnes: {list(df.columns)[:30]}..."
        )
    if found_id_col != "ID":
        df = df.rename(columns={found_id_col: "ID"})

    # 2) Normaliser ID
    def normalize_id(x):
        if pd.isna(x):
            return x
        s = str(x).strip()
        s = re.sub(r"^VIDEO_", "", s)
        digits = re.findall(r"\d+", s)
        return max(digits, key=len) if digits else s

    df["ID"] = df["ID"].map(normalize_id)

    # 3) Drop colonnes index inutiles
    unnamed = [c for c in df.columns if c.lower().startswith("unnamed")]
    if unnamed:
        df = df.drop(columns=unnamed)

    first_col = df.columns[0]
    if first_col != "ID":
        col_num = pd.to_numeric(df[first_col], errors="coerce")
        if col_num.notna().all():
            n = len(df)
            if col_num.equals(pd.Series(range(n))) or col_num.equals(pd.Series(range(1, n + 1))):
                df = df.drop(columns=[first_col])

    # 4) Charger X_train / X_test en robuste
    x_train = read_csv_robust(x_train_path)
    x_test = read_csv_robust(x_test_path)

    # Harmonisation colonne ID côté X_*
    def ensure_id_col(dfx: pd.DataFrame) -> pd.DataFrame:
        if "ID" in dfx.columns:
            pass
        elif "video_id" in dfx.columns:
            dfx = dfx.rename(columns={"video_id": "ID"})
        elif "id" in dfx.columns:
            dfx = dfx.rename(columns={"id": "ID"})
        else:
            raise ValueError(f"X_* ne contient pas de colonne ID/video_id/id. Colonnes: {list(dfx.columns)[:30]}...")
        dfx["ID"] = dfx["ID"].map(normalize_id)
        return dfx

    x_train = ensure_id_col(x_train)
    x_test = ensure_id_col(x_test)

    train_ids = set(x_train["ID"].astype(str))
    test_ids = set(x_test["ID"].astype(str))

    # 5) Détection mix/train/test
    n = len(df)
    df_ids = set(df["ID"].astype(str))
    inter_train = len(df_ids & train_ids)
    inter_test = len(df_ids & test_ids)

    ft = inter_train / len(train_ids) if train_ids else 0.0
    ftest = inter_test / len(test_ids) if test_ids else 0.0

    is_mixed = (inter_train > 0 and inter_test > 0 and (ft > 0.10 and ftest > 0.10)) or (n == mixed_size)

    base_name = csv_path.stem
    saved, shapes = {}, {}

    if is_mixed:
        df_train = df[df["ID"].astype(str).isin(train_ids)].copy()
        df_test = df[df["ID"].astype(str).isin(test_ids)].copy()

        out_train = out_dir / f"{base_name}_train.csv"
        out_test = out_dir / f"{base_name}_test.csv"
        df_train.to_csv(out_train, index=False)
        df_test.to_csv(out_test, index=False)

        saved["train"], saved["test"] = str(out_train), str(out_test)
        shapes["train"], shapes["test"] = df_train.shape, df_test.shape
        return {"status": "mixed_split", "saved": saved, "shapes": shapes}

    is_train = (n == train_size) or (inter_train >= inter_test)
    if is_train:
        out_train = out_dir / f"{base_name}_train.csv"
        df.to_csv(out_train, index=False)
        return {"status": "train_only", "saved": {"train": str(out_train)}, "shapes": {"train": df.shape}}
    else:
        out_test = out_dir / f"{base_name}_test.csv"
        df.to_csv(out_test, index=False)
        return {"status": "test_only", "saved": {"test": str(out_test)}, "shapes": {"test": df.shape}}

from pathlib import Path

ROOT = Path("/home/mohamed/SDD/hackathon/sia-predicting-short-form-video-popularity")

res = prepare_features_csv(
    csv_path=ROOT / "vggish_pooled.csv",
    x_train_path=ROOT / "X_train.csv",
    x_test_path=ROOT / "X_test.csv",
    out_dir=ROOT / "Data",
)

print(res)