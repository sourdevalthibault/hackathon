import subprocess
from pathlib import Path

# ====== À MODIFIER ======
ROOT = Path("/home/mohamed/SDD/hackathon/sia-predicting-short-form-video-popularity")
# ========================

VIDEOS_ROOT = ROOT / "videos_mp4" / "downloads"
AUDIO_ROOT  = ROOT / "audio"   # créé dans le même dossier "sia-..."

AUDIO_ROOT.mkdir(parents=True, exist_ok=True)

SAMPLE_RATE = "16000"
CHANNELS = "1"

def extract_audio(mp4_path: Path, wav_out: Path) -> bool:
    wav_out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel", "error",
        "-i", str(mp4_path),
        "-vn",
        "-ac", CHANNELS,
        "-ar", SAMPLE_RATE,
        "-acodec", "pcm_s16le",
        str(wav_out),
    ]
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError:
        return False

mp4_files = sorted(VIDEOS_ROOT.rglob("*.mp4"))
print(f"Trouvé {len(mp4_files)} vidéos dans: {VIDEOS_ROOT}")

ok = failed = skipped = 0

for mp4 in mp4_files:
    # Reproduit la même arborescence sous audio/
    # USER_xxx/VIDEO_....mp4 -> audio/USER_xxx/VIDEO_....wav
    rel = mp4.relative_to(VIDEOS_ROOT)
    out_wav = (AUDIO_ROOT / rel).with_suffix(".wav")

    if out_wav.exists() and out_wav.stat().st_size > 0:
        skipped += 1
        continue

    if extract_audio(mp4, out_wav):
        ok += 1
    else:
        failed += 1
        print(f"[ERREUR] {mp4}")

print(f"✅ OK: {ok} | ⏭️ Skipped: {skipped} | ❌ Failed: {failed}")
print("Audios dans:", AUDIO_ROOT)