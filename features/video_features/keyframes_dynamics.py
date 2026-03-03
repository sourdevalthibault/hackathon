import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

# --- CONFIGURATION ---
KEYFRAMES_DIR = "../data/keyframes/"
VIDEO_DIR = "../data/videos/"
OUTPUT_CSV = "features_dynamics.csv"

def get_video_path(video_id, root_dir):
    for root, _, files in os.walk(root_dir):
        for f in files:
            if f.startswith(video_id) and f.lower().endswith(('.mp4', '.mov')):
                return os.path.join(root, f)
    return None

def analyze_motion_at_indices(video_path, indices, fps):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return []
    
    motion_scores = []
    shake_scores = []
    
    limit_2s = int(2.0 * fps)
    
    hook_mots, hook_shakes = [], []
    content_mots, content_shakes = [], []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, prev = cap.read()
        if not ret or prev is None: continue
        
        # Analyse locale sur 4 frames pour capter la tendance du mouvement à cet instant
        prev_g = cv2.cvtColor(cv2.resize(prev, (80, 45)), cv2.COLOR_BGR2GRAY)
        
        local_mots = []
        local_shakes = []
        
        for _ in range(4):
            ret, curr = cap.read()
            if not ret or curr is None: break
            curr_g = cv2.cvtColor(cv2.resize(curr, (80, 45)), cv2.COLOR_BGR2GRAY)
            
            # Détection de points d'intérêt
            p0 = cv2.goodFeaturesToTrack(prev_g, maxCorners=20, qualityLevel=0.3, minDistance=7)
            if p0 is not None:
                p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_g, curr_g, p0, None)
                if p1 is not None and len(p1[st==1]) > 2:
                    diff = p1[st==1] - p0[st==1]
                    dist = np.linalg.norm(diff, axis=1)
                    local_mots.append(np.mean(dist))
                    local_shakes.append(np.var(diff))
            prev_g = curr_g
            
        # Dispatching entre HOOK et CONTENU
        avg_loc_mot = np.mean(local_mots) if local_mots else 0
        avg_loc_shk = np.mean(local_shakes) if local_shakes else 0
        
        if idx < limit_2s:
            hook_mots.append(avg_loc_mot)
            hook_shakes.append(avg_loc_shk)
        else:
            content_mots.append(avg_loc_mot)
            content_shakes.append(avg_loc_shk)

    
    return {
            'hook_motion': np.mean(hook_mots) if hook_mots else 0,
            'hook_shake': np.mean(hook_shakes) if hook_shakes else 0,
            
            # Si pas de scènes après 2s, on met NaN pour dire "Information absente"
            'content_motion': np.mean(content_mots) if content_mots else np.nan,
            'content_shake': np.mean(content_shakes) if content_shakes else np.nan,
            
            'max_peak_motion': np.max(hook_mots + content_mots) if (hook_mots + content_mots) else 0
        }

# 1. Grouper les keyframes par vidéo pour connaître les indices déjà extraits
# Note : On recalcule les indices à partir des noms de fichiers si on ne les a pas stockés, 
# mais ici on va simplement ré-extraire les indices via la logique du script précédent.
video_ids = sorted(list(set([f.rsplit('_kf', 1)[0] for f in os.listdir(KEYFRAMES_DIR) if "_kf" in f])))

results = []

print(f"🚀 Analyse du dynamisme intelligent sur {len(video_ids)} vidéos...")

for v_id in tqdm(video_ids):
    path = get_video_path(v_id, VIDEO_DIR)
    if not path: continue
    
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    # On reconstruit les indices (Hook + Scenes) pour savoir où regarder
    # C'est la même logique que ton script d'extraction de frames
    from scenedetect import detect, ContentDetector
    scene_list = detect(path, ContentDetector(threshold=27.0))
    
    indices = [int(0.5 * fps), int(1.5 * fps)] # Hook systématique
    for scene in scene_list:
        mid = (scene[0].get_frames() + scene[1].get_frames()) // 2
        if mid >= int(2.0 * fps):
            indices.append(mid)
    
    # On analyse le mouvement uniquement à ces endroits précis
    motion_data = analyze_motion_at_indices(path, indices[:5], fps)
    
    if motion_data:
        motion_data['video_id'] = v_id
        motion_data['num_cuts'] = len(scene_list)
        results.append(motion_data)

# 2. Sauvegarde
df = pd.DataFrame(results)
cols = ['video_id', 'hook_motion', 'hook_shake', 'content_motion', 'content_shake', 'max_peak_motion']
df = df[cols]
df.to_csv(OUTPUT_CSV, index=False)

print(f"✅ Terminé ! Features de dynamisme intelligent prêtes dans {OUTPUT_CSV}")