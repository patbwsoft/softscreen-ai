import cv2
import numpy as np
from ultralytics import YOLO
import os
import glob
from collections import deque

# === Output auto-naming ===
existing_tests = glob.glob("outputs/softscreen_multi_medium_one_output_*.mp4")
next_index = len(existing_tests) + 1
output_path = f"outputs/softscreen_multi_medium_one_output_{next_index:03}.mp4"

# === Load model and video ===
model = YOLO("scripts/yolov8m-seg.pt")
video_path = "videos/dog_test_video.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("❌ Could not open video.")
    exit()

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

print(f"✅ Processing {frame_count} frames with YOLOv8m + Persistence...")

# === Persistence settings ===
mask_history = deque(maxlen=5)  # holds last 5 masks

frame_num = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame, verbose=False)[0]
    combined_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)

    if results.masks is not None:
        for seg, cls in zip(results.masks.data, results.boxes.cls):
            class_id = int(cls.item())
            if model.names[class_id].lower() == "dog":
                single_mask = seg.cpu().numpy()
                single_mask = cv2.resize(single_mask, (frame.shape[1], frame.shape[0]))
                single_mask = (single_mask > 0.3).astype(np.uint8)

                single_mask = cv2.dilate(single_mask, np.ones((7, 7), np.uint8), iterations=1)
                single_mask = cv2.GaussianBlur(single_mask.astype(np.float32), (21, 21), 0)
                single_mask = (single_mask > 0.4).astype(np.uint8)

                combined_mask = np.maximum(combined_mask, single_mask)

    # === Add current or empty mask to history ===
    mask_history.append(combined_mask)

    # === Combine with previous masks ===
    persistent_mask = np.zeros_like(combined_mask)
    for past_mask in mask_history:
        persistent_mask = np.maximum(persistent_mask, past_mask)

    if np.any(persistent_mask):
        blurred_frame = cv2.GaussianBlur(frame, (151, 151), 0)
        mask_3ch = np.repeat(persistent_mask[:, :, np.newaxis], 3, axis=2)
        frame = np.where(mask_3ch == 1, blurred_frame, frame)

    out.write(frame)

    # === Debug Visual: Show mask overlay ===
    debug_mask = np.repeat(persistent_mask[:, :, np.newaxis] * 255, 3, axis=2)
    combined_view = np.hstack((frame, debug_mask.astype(np.uint8)))
    cv2.imshow("SoftScreen Debug", combined_view)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_num += 1
    print(f"🌀 Frame {frame_num}/{frame_count}", end="\r")

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"\n✅ Done! Output saved to: {output_path}")
