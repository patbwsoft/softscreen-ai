from mello_brain.memory import TemporalMemory
from mello_brain.introspect import run_pre_session_adjustments
from ultralytics import YOLO

import cv2
import os
import time

# === Pre-run introspection ===
run_pre_session_adjustments()

# === Test file ===
TEST_MEDIA = r"C:\Projects\SoftScreen\videos\dog_playing2.mp4"

# === Runtime options ===
SAVE_OUTPUT = False
MAX_TEST_FRAMES = None

print("Loading Mello's memory...")
memory = TemporalMemory()
print("Memory restored successfully.")

model = YOLO("models/yolov8x-seg.pt")

video_filename = os.path.basename(TEST_MEDIA)
video_name_no_ext = os.path.splitext(video_filename)[0]
OUTPUT_PATH = rf"C:\Projects\SoftScreen\outputs\{video_name_no_ext}_output.mp4"

frame_counter = 0
fps_start_time = time.time()
paused = False

if TEST_MEDIA.endswith(".mp4"):
    cap = cv2.VideoCapture(TEST_MEDIA)

    if not cap.isOpened():
        print(f"Failed to open video: {TEST_MEDIA}")
        exit()

    out = None
    if SAVE_OUTPUT:
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(OUTPUT_PATH, fourcc, 30.0,
                              (int(cap.get(3)), int(cap.get(4))))
        print(f"Saving output to: {OUTPUT_PATH}")

    print("Starting SoftScreen Test...")

    try:
        while cap.isOpened():
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("Video ended naturally.")
                    break

                detections = []
                results = model(frame)[0]
                for r in results.boxes:
                    label = model.names[int(r.cls[0])]
                    conf = float(r.conf[0])
                    box = [int(x) for x in r.xyxy[0]]
                    detections.append({
                        'label': label,
                        'confidence': conf,
                        'bbox': box
                    })

                output_frame = memory.process_frame(frame, detections)
                frame_counter += 1

                if frame_counter % 30 == 0:
                    elapsed_time = time.time() - fps_start_time
                    fps = 30 / elapsed_time
                    print(f"Frame {frame_counter} | Estimated FPS: {fps:.2f}")
                    fps_start_time = time.time()

                if SAVE_OUTPUT:
                    out.write(output_frame)

                cv2.imshow("SoftScreen Brain Output", output_frame)

                if MAX_TEST_FRAMES and frame_counter >= MAX_TEST_FRAMES:
                    print(f"Max test frame count ({MAX_TEST_FRAMES}) reached.")
                    break
            else:
                paused_frame = output_frame.copy()
                cv2.putText(paused_frame, "PAUSED", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 8, cv2.LINE_AA)
                cv2.imshow("SoftScreen Brain Output", paused_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Manual quit detected.")
                break
            elif key == ord('p'):
                paused = not paused
                print("Paused." if paused else "Resumed.")

    except Exception as e:
        print(f"Unexpected error: {e}")

    finally:
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()
        print(f"Test completed. Total frames processed: {frame_counter}")
        memory.save_brain()

elif TEST_MEDIA.lower().endswith((".jpg", ".png")):
    frame = cv2.imread(TEST_MEDIA)
    results = model(frame)[0]
    detections = []
    for r in results.boxes:
        label = model.names[int(r.cls[0])]
        conf = float(r.conf[0])
        box = [int(x) for x in r.xyxy[0]]
        detections.append({
            'label': label,
            'confidence': conf,
            'bbox': box
        })
    output_frame = memory.process_frame(frame, detections)
    cv2.imshow("SoftScreen Brain Output", output_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    memory.save_brain()

else:
    print("Unsupported media format.")