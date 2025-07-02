import cv2
import numpy as np

# Load your test image
image_path = "frame_00509.jpg"
output_path = "blur_test_output.jpg"

frame = cv2.imread(image_path)
if frame is None:
    print("❌ Failed to load image.")
    exit()

print(f"✅ Loaded image {frame.shape}")

# === Manually place blur target over Buffy ===
h, w = frame.shape[:2]
mask = np.zeros((h, w), dtype=np.uint8)

# Estimated position of Buffy in the frame (adjust as needed)
center = (500, 1300)  # (x, y)
radius = 120  # play with this

cv2.circle(mask, center, radius, 255, -1)
print("✅ Created circular mask over Buffy")

# Apply blur only where the mask is nonzero
blurred = cv2.GaussianBlur(frame, (191, 191), 0)
output = np.where(mask[:, :, None] == 255, blurred, frame)

cv2.imwrite(output_path, output)
print(f"✅ Saved blurred image to: {output_path}")