import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from scenedetect import SceneManager, ContentDetector, VideoStreamCv2

def process_video_single_pass(video_path):
    # 1. On utilise l'adaptateur natif de Scenedetect pour OpenCV (Plus stable)
    video_stream = VideoStreamCv2(video_path)
    cap = video_stream._cap # On récupère l'objet cap interne pour nos calculs de mouvement
    
    if not cap.isOpened():
        return None

    # 2. Infos de base
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    
    # 3. Setup Scenedetect
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=27.0))
    
    # --- VITESSE : On saute 4 images sur 5 ---
    scene_manager.detect_scenes(video_stream, frame_skip=4) 
    scenes = scene_manager.get_scene_list()
    
    # 4. Mouvement (On fait un check au milieu de la vidéo)
    mid_frame = total_frames // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
    
    mots, shakes = [], []
    ret, prev = cap.read()
    if ret and prev is not None:
        # On réduit la taille pour que ce soit instantané
        prev_g = cv2.cvtColor(cv2.resize(prev, (80, 45)), cv2.COLOR_BGR2GRAY)
        for _ in range(5): # On analyse 5 frames pour le dynamisme
            ret, curr = cap.read()
            if not ret or curr is None: break
            curr_g = cv2.cvtColor(cv2.resize(curr, (80, 45)), cv2.COLOR_BGR2GRAY)
            p0 = cv2.goodFeaturesToTrack(prev_g, maxCorners=20, qualityLevel=0.3, minDistance=7)
            if p0 is not None:
                p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_g, curr_g, p0, None)
                if p1 is not None and len(p1[st==1]) > 2:
                    diff = p1[st==1] - p0[st==1]
                    mots.append(np.mean(np.linalg.norm(diff, axis=1)))
                    shakes.append(np.var(diff))
            prev_g = curr_g


    
    return {
        'video_id': os.path.splitext(os.path.basename(video_path))[0],
        'num_cuts': len(scenes),
        'avg_motion': round(np.mean(mots), 4) if mots else 0,
        'camera_shake': round(np.mean(shakes), 6) if shakes else 0
    }

# --- BOUCLE DE TEST ---
INPUT_DIR = "../data/videos/" 
video_paths = []
for root, dirs, files in os.walk(INPUT_DIR):
    for f in files:
        if f.lower().endswith(('.mp4', '.mov')):
            video_paths.append(os.path.join(root, f))

print(f"🚀 Test sur {len(video_paths)} vidéos...")

results = []
for i, path in enumerate(tqdm(video_paths)):
    data = process_video_single_pass(path)
    if data:
        results.append(data)
    #if i == 9: break # On s'arrête à 10 pour le crash-test

df = pd.DataFrame(results)
df.to_csv("features_dynamisme.csv", index=False)
print("✅ Succès ! CSV créé.")
print(df.head())