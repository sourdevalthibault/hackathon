import os
import cv2
import numpy as np
from tqdm import tqdm
from scenedetect import detect, ContentDetector

# --- CONFIGURATION ---
VIDEO_DIR = "../data/videos/"
OUTPUT_DIR = "../data/keyframes/"
MAX_FRAMES = 5 

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def get_keyframes_indices(video_path, cap):
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0: fps = 30 
    
    indices = []
    
    # --- 1. LE HOOK (FIXE : 0.5s et 1.5s) ---
    hook_frames = [int(0.5 * fps), int(1.5 * fps)]
    indices.extend([f for f in hook_frames if f < total_frames])
    
    # --- 2. LES SCÈNES (UNIQUEMENT APRÈS 2 SECONDES) ---
    limit_2s = int(2.0 * fps)
    
    scene_list = detect(video_path, ContentDetector(threshold=27.0))
    
    for scene in scene_list:
        start, end = scene
        mid = (start.get_frames() + end.get_frames()) // 2
        
        # On n'ajoute la frame que si elle se situe APRES les 2 premières secondes
        if mid >= limit_2s:
            indices.append(mid)
            
    # --- 3. TRI ET LIMITE ---
    indices = sorted(list(set(indices)))
    
    # On garde les 2 frames du hook + les 3 premières scènes après 2s (total 5)
    return indices[:MAX_FRAMES]

def process_video_final(video_path, output_folder):
    try:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        cap = cv2.VideoCapture(video_path)
        
        # Calcul des indices avec la règle des 2 secondes
        frame_indices = get_keyframes_indices(video_path, cap)
        
        saved_count = 0
        for i, idx in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            
            if ret and frame is not None:
                filename = f"{video_name}_kf{i}.jpg"
                save_path = os.path.join(output_folder, filename)
                cv2.imwrite(save_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                saved_count += 1
        
        cap.release()
        return saved_count
        
    except Exception as e:
        print(f"\nErreur sur {os.path.basename(video_path)}: {e}")
        return 0

# --- BOUCLE DE TRAITEMENT ---
video_paths = [os.path.join(r, f) for r, _, fs in os.walk(VIDEO_DIR) for f in fs if f.lower().endswith(('.mp4', '.mov'))]

print(f"📸 Extraction Hook (<2s) + Scènes (>2s) sur {len(video_paths)} vidéos...")

for path in tqdm(video_paths):
    process_video_final(path, OUTPUT_DIR)

print(f"\n✅ Dossier de frames terminé ! {OUTPUT_DIR}")