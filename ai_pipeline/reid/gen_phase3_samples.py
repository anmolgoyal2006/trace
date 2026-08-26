"""
gen_phase3_samples.py — Phase 3.6
Generate a contact-sheet verification image of 16 accepted crops
from the phase3_final dataset, spread across different track IDs.
Output: dataset/crop_verification/phase3_final_samples.jpg
"""
import cv2
import json
import math
import random
import numpy as np
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
META_PATH = REPO_ROOT / "dataset" / "crops_metadata_phase3_final.json"
OUT_PATH  = REPO_ROOT / "dataset" / "crop_verification" / "phase3_final_samples.jpg"

THUMB_W  = 120
THUMB_H  = 200
COLS     = 8
PAD      = 4
LABEL_H  = 16
N_SAMPLE = 16

random.seed(42)

meta = json.loads(META_PATH.read_text())

# Group by track_id
by_track: dict[int, list] = {}
for rec in meta:
    by_track.setdefault(rec["track_id"], []).append(rec)

track_ids = sorted(by_track.keys())
chosen_tracks = random.sample(track_ids, min(N_SAMPLE, len(track_ids)))
samples = [random.choice(by_track[tid]) for tid in sorted(chosen_tracks)]

ROWS = math.ceil(len(samples) / COLS)
canvas_w = COLS * (THUMB_W + PAD) + PAD
canvas_h = ROWS * (THUMB_H + LABEL_H + PAD) + PAD
canvas = np.full((canvas_h, canvas_w, 3), 40, dtype=np.uint8)

FONT = cv2.FONT_HERSHEY_SIMPLEX
FS   = 0.35
ok   = 0

for idx, rec in enumerate(samples):
    img_path = REPO_ROOT / rec["crop_path"]
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"[WARN] unreadable: {img_path}")
        continue
    thumb = cv2.resize(img, (THUMB_W, THUMB_H))
    row = idx // COLS
    col = idx  % COLS
    x = PAD + col * (THUMB_W + PAD)
    y = PAD + row * (THUMB_H + LABEL_H + PAD)
    canvas[y : y + THUMB_H, x : x + THUMB_W] = thumb
    label = f"t{rec['track_id']} f{rec['frame']}"
    cv2.putText(canvas, label, (x, y + THUMB_H + LABEL_H - 2),
                FONT, FS, (200, 200, 200), 1)
    ok += 1

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
cv2.imwrite(str(OUT_PATH), canvas)
print(f"Saved {ok} thumbnails -> {OUT_PATH}")
print(f"Canvas size: {canvas_w}x{canvas_h}")
