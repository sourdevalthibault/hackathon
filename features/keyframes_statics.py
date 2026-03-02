import cv2
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict

# --- CONFIGURATION ---
KEYFRAMES_DIR = "../data/keyframes/" # Ton dossier de JPG
OUTPUT_CSV = "keyframes_features_statics.csv"

def analyze_single_image(image_path):
    """Calcule la netteté, luminosité et saturation d'une image fixe."""
    frame = cv2.imread(image_path)
    if frame is None:
        return None
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    return {
        'sharpness': cv2.Laplacian(gray, cv2.CV_64F).var(),
        'brightness': np.mean(gray),
        'saturation': np.mean(hsv[:,:,1])
    }

# 1. On regroupe les images par vidéo (car une vidéo a entre 2 et 5 images)
video_to_frames = defaultdict(list)
all_files = sorted([f for f in os.listdir(KEYFRAMES_DIR) if f.lower().endswith(('.jpg', '.jpeg'))])

for filename in all_files:
    # On extrait le video_id à partir de "VIDEO_ID_kf0.jpg" -> "VIDEO_ID"
    # On split par le dernier "_" pour être sûr
    video_id = filename.rsplit('_kf', 1)[0]
    video_to_frames[video_id].append(filename)

print(f"Analyse de {len(all_files)} images pour {len(video_to_frames)} vidéos...")

# 2. On traite chaque groupe de frames
final_results = []

for video_id, frames in tqdm(video_to_frames.items()):
    video_data = {'video_id': video_id}

    # 1. On initialise TOUTES les colonnes possibles avec np.nan
    # Cela garantit que chaque ligne a la même structure
    for i in range(1, 6): # Pour f1 à f5
        video_data[f'f{i}_sharpness'] = np.nan
        video_data[f'f{i}_brightness'] = np.nan
        video_data[f'f{i}_saturation'] = np.nan
    
    # 2. On remplit uniquement les frames existantes
    for i, frame_name in enumerate(frames):
        if i >= 5: break 
        
        img_path = os.path.join(KEYFRAMES_DIR, frame_name)
        stats = analyze_single_image(img_path)
        
        if stats:
            # On écrase le np.nan par la vraie valeur
            for key, value in stats.items():
                video_data[f'f{i+1}_{key}'] = round(float(value), 2)
    
    final_results.append(video_data)    

# 3. Création du DataFrame et export
df = pd.DataFrame(final_results)

# On s'assure que video_id est en premier
cols = ['video_id'] + [c for c in df.columns if c != 'video_id']
df = df[cols]

df.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Terminé ! Features sauvegardées dans {OUTPUT_CSV}")