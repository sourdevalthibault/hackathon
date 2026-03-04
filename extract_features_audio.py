import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import librosa

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
### Part 1 ###
# ======== CONFIG ========
ROOT = Path("/home/mohamed/SDD/hackathon/sia-predicting-short-form-video-popularity")
AUDIO_ROOT = ROOT / "audio"
OUTPUT_CSV = ROOT / "audio_features.csv"
FAILED_TXT = ROOT / "audio_failed.txt"
# ========================

def safe_float(x, default=np.nan):
    """Convertit x en float même si c'est un array/list."""
    try:
        x = np.asarray(x)
        if x.size == 0:
            return default
        return float(x.reshape(-1)[0])
    except Exception:
        try:
            return float(x)
        except Exception:
            return default

def extract_features(wav_path: Path) -> dict:
    y, sr = librosa.load(wav_path, sr=16000, mono=True)

    feats = {}

    # RMS
    try:
        rms = librosa.feature.rms(y=y)[0]
        feats["rms_mean"] = float(np.mean(rms))
        feats["rms_std"]  = float(np.std(rms))
    except Exception:
        feats["rms_mean"] = np.nan
        feats["rms_std"]  = np.nan

    # ZCR
    try:
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        feats["zcr_mean"] = float(np.mean(zcr))
        feats["zcr_std"]  = float(np.std(zcr))
    except Exception:
        feats["zcr_mean"] = np.nan
        feats["zcr_std"]  = np.nan

    # Spectral
    try:
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        feats["centroid_mean"] = float(np.mean(centroid))
        feats["centroid_std"]  = float(np.std(centroid))
    except Exception:
        feats["centroid_mean"] = np.nan
        feats["centroid_std"]  = np.nan

    try:
        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        feats["bandwidth_mean"] = float(np.mean(bandwidth))
        feats["bandwidth_std"]  = float(np.std(bandwidth))
    except Exception:
        feats["bandwidth_mean"] = np.nan
        feats["bandwidth_std"]  = np.nan

    try:
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        feats["rolloff_mean"] = float(np.mean(rolloff))
        feats["rolloff_std"]  = float(np.std(rolloff))
    except Exception:
        feats["rolloff_mean"] = np.nan
        feats["rolloff_std"]  = np.nan

    # MFCC
    try:
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        for i in range(13):
            feats[f"mfcc_{i}_mean"] = float(np.mean(mfcc[i]))
            feats[f"mfcc_{i}_std"]  = float(np.std(mfcc[i]))
    except Exception:
        for i in range(13):
            feats[f"mfcc_{i}_mean"] = np.nan
            feats[f"mfcc_{i}_std"]  = np.nan

    # Tempo
    try:
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        feats["tempo"] = safe_float(tempo)
    except Exception:
        feats["tempo"] = np.nan

    # Onset rate
    try:
        onsets = librosa.onset.onset_detect(y=y, sr=sr)
        feats["onset_rate"] = float(len(onsets) / (len(y)/sr + 1e-9))
    except Exception:
        feats["onset_rate"] = np.nan

    return feats


wav_files = sorted(AUDIO_ROOT.rglob("*.wav"))
print("AUDIO_ROOT =", AUDIO_ROOT)
print("WAV trouvés =", len(wav_files))

if len(wav_files) == 0:
    raise RuntimeError("❌ Aucun .wav trouvé. Vérifie le chemin AUDIO_ROOT.")

rows = []
failed = []

for wav in tqdm(wav_files):
    vid = wav.stem.split("_")[-1]  # VIDEO_123 -> 123
    try:
        feats = extract_features(wav)
        feats["ID"] = str(vid)
        rows.append(feats)
    except Exception as e:
        failed.append(f"{wav}\t{repr(e)}")

df = pd.DataFrame(rows)

print("Rows extraites:", len(df))
print("Colonnes:", df.shape[1])

df.to_csv(OUTPUT_CSV, index=False)

with open(FAILED_TXT, "w", encoding="utf-8") as f:
    f.write("\n".join(failed))

print("✅ CSV écrit:", OUTPUT_CSV)
print("⚠️ Fichiers en échec:", len(failed), "->", FAILED_TXT)
print(df.head())


