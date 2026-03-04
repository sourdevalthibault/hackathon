import pandas as pd
import numpy as np
from pathlib import Path
import re
import warnings
warnings.filterwarnings("ignore")

# =========================
# CONFIG
# =========================
ROOT = Path("/home/mohamed/SDD/hackathon/sia-predicting-short-form-video-popularity")
DATA_DIR = ROOT / "Data"
DATA_DIR.mkdir(exist_ok=True)

X_TRAIN_PATH = ROOT / "X_train.csv"
X_TEST_PATH  = ROOT / "X_test.csv"

# =========================
# Utils
# =========================
def read_csv_robust(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, sep=None, engine="python")
    except Exception:
        try:
            return pd.read_csv(path, sep=";", engine="python")
        except Exception:
            return pd.read_csv(path, sep=",", engine="python")

def normalize_id_series(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip()
    s = s.str.replace(r"^(VIDEO_|video_|Video_)", "", regex=True)
    s = s.str.extract(r"(\d+)", expand=False).fillna(s)
    return s.str.strip()

def safe_value_counts_map(train_series: pd.Series) -> dict:
    # map value -> count, NaN excluded
    vc = train_series.dropna().astype(str).value_counts()
    return vc.to_dict()

def apply_count_feature(df: pd.DataFrame, col: str, count_map: dict, new_name: str, default: int = 0):
    vals = df[col].astype(str)
    df[new_name] = vals.map(count_map).fillna(default).astype(int)

def try_get_id_col(df: pd.DataFrame) -> pd.DataFrame:
    if "ID" in df.columns:
        return df
    if "id" in df.columns:
        return df.rename(columns={"id": "ID"})
    if "video_id" in df.columns:
        return df.rename(columns={"video_id": "ID"})
    # fallback: first column
    return df.rename(columns={df.columns[0]: "ID"})

# =========================
# Load
# =========================
X_train = read_csv_robust(X_TRAIN_PATH)
X_test  = read_csv_robust(X_TEST_PATH)

X_train = try_get_id_col(X_train)
X_test  = try_get_id_col(X_test)

X_train["ID"] = normalize_id_series(X_train["ID"])
X_test["ID"]  = normalize_id_series(X_test["ID"])

# =========================
# Feature engineering (NO TARGET)
# =========================

# --- A) Count/Frequency features from categorical columns (computed on TRAIN only)
# columns available in metadata (adapt if some missing)
cat_cols = ["uploader", "uploader_short", "channel", "artist", "track", "album", "uid"]
for c in cat_cols:
    if c not in X_train.columns:
        X_train[c] = np.nan
    if c not in X_test.columns:
        X_test[c] = np.nan

# counts (train-derived)
maps = {c: safe_value_counts_map(X_train[c]) for c in cat_cols}

# apply counts + log-counts (stabilize)
for c in cat_cols:
    apply_count_feature(X_train, c, maps[c], f"{c}_count", default=0)
    apply_count_feature(X_test,  c, maps[c], f"{c}_count", default=0)
    X_train[f"{c}_log_count"] = np.log1p(X_train[f"{c}_count"])
    X_test[f"{c}_log_count"]  = np.log1p(X_test[f"{c}_count"])

# --- B) Music presence / original sound style flags (safe)
# has_music
X_train["has_music"] = X_train["artist"].notna().astype(int)
X_test["has_music"]  = X_test["artist"].notna().astype(int)

# "original sound" heuristic: many datasets label it like "original sound" / "оригинальный звук"
def is_original_sound(s: pd.Series) -> pd.Series:
    t = s.fillna("").astype(str).str.lower()
    return (
        t.str.contains("original") |
        t.str.contains("original sound") |
        t.str.contains("son original") |
        t.str.contains("son d'origine") |
        t.str.contains("оригиналь")  # russe (ex: "оригинальный звук")
    ).astype(int)

if "track" in X_train.columns:
    X_train["is_original_sound"] = is_original_sound(X_train["track"])
    X_test["is_original_sound"]  = is_original_sound(X_test["track"])
else:
    X_train["is_original_sound"] = 0
    X_test["is_original_sound"]  = 0

# --- C) Format / resolution features (safe)
def extract_format_features(df: pd.DataFrame) -> pd.DataFrame:
    if "format" not in df.columns:
        df["format"] = np.nan
    wh = df["format"].fillna("").astype(str).str.extract(r"(\d+)\s*x\s*(\d+)", expand=True)
    df["width"]  = pd.to_numeric(wh[0], errors="coerce")
    df["height"] = pd.to_numeric(wh[1], errors="coerce")
    df["resolution_area"] = df["width"] * df["height"]
    df["is_vertical"] = (df["height"] > df["width"]).astype(int)
    df["is_full_hd"] = (df["resolution_area"] >= 1920 * 1080).astype(int)
    df["aspect_w_over_h"] = (df["width"] / df["height"]).replace([np.inf, -np.inf], np.nan)
    return df

X_train = extract_format_features(X_train)
X_test  = extract_format_features(X_test)

# --- D) Text features beyond what you already have (no sentiment; no hashtags/emojis count)
# (You already have text_len/hashtags/emojis densities + sentiment)
def extract_text_features(df: pd.DataFrame) -> pd.DataFrame:
    if "description" not in df.columns:
        df["description"] = ""
    desc = df["description"].fillna("").astype(str)

    # words
    tokens = desc.str.findall(r"\b\w+\b")
    df["nb_words"] = tokens.apply(len).astype(int)

    # avg word length
    def avg_len(ws):
        if not ws:
            return 0.0
        return float(np.mean([len(w) for w in ws]))
    df["avg_word_length"] = tokens.apply(avg_len)

    # punctuation flags
    df["has_exclamation"] = desc.str.contains("!").astype(int)
    df["has_question"]    = desc.str.contains(r"\?").astype(int)

    # uppercase ratio (safe)
    letters = desc.str.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ]")
    uppers  = desc.str.findall(r"[A-ZÀ-Ö]")
    df["uppercase_ratio"] = [
        (len(u) / len(l)) if len(l) else 0.0
        for u, l in zip(uppers, letters)
    ]

    # mentions / links / numbers presence
    df["has_mention"] = desc.str.contains(r"@\w+").astype(int)
    df["has_url"]     = desc.str.contains(r"http|www\.").astype(int)
    df["has_number"]  = desc.str.contains(r"\d").astype(int)

    # length-per-second style features (needs video_duration; you already use duration but this is interaction)
    if "video_duration" in df.columns:
        vd = pd.to_numeric(df["video_duration"], errors="coerce").replace(0, np.nan)
        df["words_per_second"] = (df["nb_words"] / vd).replace([np.inf, -np.inf], np.nan)
    else:
        df["words_per_second"] = np.nan

    return df

X_train = extract_text_features(X_train)
X_test  = extract_text_features(X_test)

# --- E) Cross-categorical diversity features (safe, no target)
# Example: how many distinct artists a uploader uses (computed on train only)
def add_uploader_diversity(train_df: pd.DataFrame, test_df: pd.DataFrame):
    # distinct artists per uploader
    ua = (
        train_df[["uploader", "artist"]]
        .dropna(subset=["uploader"])
        .astype({"uploader": str, "artist": str})
        .groupby("uploader")["artist"]
        .nunique()
        .rename("uploader_unique_artist")
        .reset_index()
    )
    train_df = train_df.merge(ua, on="uploader", how="left")
    test_df  = test_df.merge(ua, on="uploader", how="left")
    train_df["uploader_unique_artist"] = train_df["uploader_unique_artist"].fillna(0).astype(int)
    test_df["uploader_unique_artist"]  = test_df["uploader_unique_artist"].fillna(0).astype(int)

    # distinct tracks per uploader
    ut = (
        train_df[["uploader", "track"]]
        .dropna(subset=["uploader"])
        .astype({"uploader": str, "track": str})
        .groupby("uploader")["track"]
        .nunique()
        .rename("uploader_unique_track")
        .reset_index()
    )
    train_df = train_df.merge(ut, on="uploader", how="left")
    test_df  = test_df.merge(ut, on="uploader", how="left")
    train_df["uploader_unique_track"] = train_df["uploader_unique_track"].fillna(0).astype(int)
    test_df["uploader_unique_track"]  = test_df["uploader_unique_track"].fillna(0).astype(int)

    return train_df, test_df

X_train, X_test = add_uploader_diversity(X_train, X_test)

# =========================
# Keep ONLY the new metadata features (not already in your feature list)
# =========================
# You already have: aspect_ratio, video_duration, release_year, text_len, nb_hashtags, nb_emojis, densities, sentiment, uploader one-hot
# So we output only the new engineered ones:

out_cols = [
    "ID",

    # counts / log-counts
    "uploader_count", "uploader_log_count",
    "uploader_short_count", "uploader_short_log_count",
    "channel_count", "channel_log_count",
    "artist_count", "artist_log_count",
    "track_count", "track_log_count",
    "album_count", "album_log_count",
    "uid_count", "uid_log_count",

    # music flags
    "has_music",
    "is_original_sound",

    # format features
    "width", "height", "resolution_area",
    "is_vertical", "is_full_hd", "aspect_w_over_h",

    # text features (new)
    "nb_words", "avg_word_length",
    "has_exclamation", "has_question",
    "uppercase_ratio",
    "has_mention", "has_url", "has_number",
    "words_per_second",

    # diversity
    "uploader_unique_artist",
    "uploader_unique_track",
]

# Ensure columns exist even if missing upstream
for c in out_cols:
    if c not in X_train.columns:
        X_train[c] = np.nan
    if c not in X_test.columns:
        X_test[c] = np.nan

train_out = X_train[out_cols].copy()
test_out  = X_test[out_cols].copy()

# Save
train_path = DATA_DIR / "metadata_features_train.csv"
test_path  = DATA_DIR / "metadata_features_test.csv"
train_out.to_csv(train_path, index=False)
test_out.to_csv(test_path, index=False)

print("✅ Saved:")
print(" -", train_path, train_out.shape)
print(" -", test_path, test_out.shape)
print("\nPreview train:")
print(train_out.head(3).to_string(index=False))