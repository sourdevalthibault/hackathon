import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# ======== CONFIG ========
ROOT = Path("/home/mohamed/SDD/hackathon/sia-predicting-short-form-video-popularity")
INPUT_CSV = ROOT / "audio_features.csv"
OUTPUT_CSV = ROOT / "audio_features_scored.csv"
FAILED_TXT = ROOT / "audio_score_failed.txt"
ID_COL = "ID"
EPS = 1e-8
# ========================

def safe_numeric_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def safe_zscore(s: pd.Series, eps: float = EPS) -> pd.Series:
    x = safe_numeric_series(s)
    mu = x.mean(skipna=True)
    std = x.std(skipna=True)
    if pd.isna(std) or std < eps:
        return pd.Series(np.zeros(len(x)), index=x.index)
    return (x - mu) / (std + eps)

def clip_series(s: pd.Series, lo: float = -5.0, hi: float = 5.0) -> pd.Series:
    return s.clip(lo, hi)

def weighted_sum(parts: dict, weights: dict, eps: float = EPS) -> pd.Series:
    used = [(name, ser) for name, ser in parts.items() if ser is not None]
    if not used:
        return None
    wsum = 0.0
    out = 0.0
    for name, ser in used:
        w = weights.get(name, 1.0 / len(used))
        out = out + w * ser
        wsum += w
    return out / (wsum + eps)

def add_audio_scores_one_row(row: pd.Series) -> dict:
    """
    Prend une ligne (Series) et retourne un dict avec les nouvelles features.
    Important: conserve ID.
    """
    d = row.to_dict()

    # Force ID
    if ID_COL not in d:
        raise ValueError(f"Missing {ID_COL} in row")
    d[ID_COL] = str(d[ID_COL])

    # Build z_ for numeric fields
    # Ici on ne calcule pas z-score row-wise (ça n'a pas de sens).
    # Donc on fera les z-scores au niveau DataFrame (plus bas).
    return d

def add_scores_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Force ID to str
    if ID_COL not in out.columns:
        raise ValueError(f"Colonne '{ID_COL}' introuvable.")
    out[ID_COL] = out[ID_COL].astype(str)

    # Z-scores (toutes colonnes numériques sauf ID)
    for c in out.columns:
        if c == ID_COL:
            continue
        col = safe_numeric_series(out[c])
        if col.notna().any():
            out[f"z_{c}"] = clip_series(safe_zscore(col))

    # --- Scores composites (FIX: pas de "or 0.0")
    s_energy = weighted_sum(
        {
            "z_rms_mean": out["z_rms_mean"] if "z_rms_mean" in out.columns else None,
            "z_tempo": out["z_tempo"] if "z_tempo" in out.columns else None,
            "z_onset_rate": out["z_onset_rate"] if "z_onset_rate" in out.columns else None,
        },
        {"z_rms_mean": 0.45, "z_tempo": 0.35, "z_onset_rate": 0.20},
    )
    out["audio_energy_score"] = s_energy if s_energy is not None else 0.0

    s_bright = weighted_sum(
        {
            "z_centroid_mean": out["z_centroid_mean"] if "z_centroid_mean" in out.columns else None,
            "z_rolloff_mean": out["z_rolloff_mean"] if "z_rolloff_mean" in out.columns else None,
            "z_bandwidth_mean": out["z_bandwidth_mean"] if "z_bandwidth_mean" in out.columns else None,
        },
        {"z_centroid_mean": 0.45, "z_rolloff_mean": 0.35, "z_bandwidth_mean": 0.20},
    )
    out["audio_brightness_score"] = s_bright if s_bright is not None else 0.0

    s_act = weighted_sum(
        {
            "z_onset_rate": out["z_onset_rate"] if "z_onset_rate" in out.columns else None,
            "z_zcr_mean": out["z_zcr_mean"] if "z_zcr_mean" in out.columns else None,
            "z_rms_std": out["z_rms_std"] if "z_rms_std" in out.columns else None,
        },
        {"z_onset_rate": 0.50, "z_zcr_mean": 0.35, "z_rms_std": 0.15},
    )
    out["audio_activity_score"] = s_act if s_act is not None else 0.0

    # --- Interactions
    out["tempo_x_onset"] = safe_numeric_series(out["tempo"]) * safe_numeric_series(out["onset_rate"])
    out["rms_x_tempo"] = safe_numeric_series(out["rms_mean"]) * safe_numeric_series(out["tempo"])
    out["centroid_x_bandwidth"] = safe_numeric_series(out["centroid_mean"]) * safe_numeric_series(out["bandwidth_mean"])

    # --- MFCC agregés
    mfcc_mean_cols = [c for c in out.columns if c.startswith("mfcc_") and c.endswith("_mean")]
    mfcc_std_cols  = [c for c in out.columns if c.startswith("mfcc_") and c.endswith("_std")]

    if mfcc_mean_cols:
        out["mfcc_means_avg"] = out[mfcc_mean_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
        out["mfcc_means_std"] = out[mfcc_mean_cols].apply(pd.to_numeric, errors="coerce").std(axis=1)
    else:
        out["mfcc_means_avg"] = np.nan
        out["mfcc_means_std"] = np.nan

    if mfcc_std_cols:
        out["mfcc_stds_avg"] = out[mfcc_std_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    else:
        out["mfcc_stds_avg"] = np.nan

    # --- Score global
    out["audio_pop_audio_score"] = (
        0.50 * out["audio_energy_score"] +
        0.30 * out["audio_activity_score"] +
        0.20 * out["audio_brightness_score"]
    )

    # ID en première colonne
    cols = [ID_COL] + [c for c in out.columns if c != ID_COL]
    out = out[cols]
    return out
def main():
    print("INPUT_CSV =", INPUT_CSV)
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"❌ Fichier introuvable: {INPUT_CSV}")

    # ✅ CRITIQUE: lire ID comme string dès le départ
    df = pd.read_csv(INPUT_CSV, dtype={ID_COL: "string"})
    print("Lignes lues:", len(df))
    print("Colonnes lues:", df.columns.tolist()[:8], "...")

    # ✅ Force ID en string (au cas où)
    df[ID_COL] = df[ID_COL].astype("string")

    failed = []
    ok_rows = []

    # Row-wise check (style similaire au tien) juste pour logger les erreurs d'ID
    for i, row in tqdm(df.iterrows(), total=len(df)):
        try:
            ok_rows.append(add_audio_scores_one_row(row))  # conserve ID
        except Exception as e:
            rid = row.get(ID_COL, f"row_{i}")
            failed.append(f"{rid}\t{repr(e)}")

    base_df = pd.DataFrame(ok_rows)

    # ✅ Re-force ID ici aussi (au cas où Pandas l'infère mal)
    if ID_COL in base_df.columns:
        base_df[ID_COL] = base_df[ID_COL].astype("string")
    else:
        raise ValueError(f"❌ '{ID_COL}' absent de base_df après ok_rows.")

    out_df = add_scores_dataframe(base_df)

    # ✅ ID en string + en première colonne (très important pour debug/merge)
    out_df[ID_COL] = out_df[ID_COL].astype("string")
    cols = [ID_COL] + [c for c in out_df.columns if c != ID_COL]
    out_df = out_df[cols]

    print("ID présent ?", ID_COL in out_df.columns)
    print("Premières colonnes:", out_df.columns.tolist()[:5])

    # ✅ Vérif anti e+18 : on affiche 3 IDs
    print("Exemples ID:", out_df[ID_COL].head(3).tolist())

    out_df.to_csv(OUTPUT_CSV, index=False)

    with open(FAILED_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(failed))

    print("✅ CSV écrit:", OUTPUT_CSV)
    print("⚠️ Lignes en échec:", len(failed), "->", FAILED_TXT)
    print(out_df[[ID_COL, "audio_pop_audio_score", "tempo_x_onset", "rms_x_tempo", "centroid_x_bandwidth"]].head())

if __name__ == "__main__":
    main()