import cv2
import os
import pandas as pd
import numpy as np
from tqdm import tqdm

def extract_smart_frames(video_path, num_frames=3):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0
    
    # On calcule les indices des images (ex: à 25%, 50%, 75% de la vidéo)
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    collected_data = []
    
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx) # SAUT IMMÉDIAT À L'IMAGE
        ret, frame = cap.read()
        if not ret: continue
        
        # --- CALCULS LÉGERS ICI ---
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        data = {
            'sharpness': cv2.Laplacian(gray, cv2.CV_64F).var(),
            'brightness': np.mean(gray),
            'saturation': np.mean(hsv[:,:,1])
        }
        collected_data.append(data)
    
    cap.release()
    
    if len(collected_data) == 0: return None

    # On aplatit les résultats : frame1_sharpness, frame2_sharpness...
    flat_features = {'duration': duration}
    for i, d in enumerate(collected_data):
        for key, value in d.items():
            flat_features[f'f{i+1}_{key}'] = value
            
    return flat_features

# --- SCAN ET EXECUTION SUR KAGGLE ---
INPUT_DIR = "../data/videos/"
results = []

# Récupération récursive des chemins
all_videos = []
for root, _, files in os.walk(INPUT_DIR):
    for f in files:
        if f.lower().endswith(('.mp4', '.mov')):
            all_videos.append(os.path.join(root, f))

print(f"Analyse flash de {len(all_videos)} vidéos...")

for path in tqdm(all_videos):

    video_id = os.path.splitext(os.path.basename(path))[0]
    feats = extract_smart_frames(path, num_frames=3)
    if feats:
        feats['video_id'] = video_id
        results.append(feats)


df = pd.DataFrame(results)
# Réorganiser les colonnes pour que video_id soit bien la première
cols = ['video_id'] + [c for c in df.columns if c != 'video_id']
df = df[cols]
df.to_csv("features_net_luminosité_saturation.csv", index=False)