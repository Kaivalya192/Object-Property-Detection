import os
import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("yolov8l-seg.pt")

input_dir = "input"
output_dir = "objects"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def get_center(mask):
    """Calculate the center of the mask."""
    M = cv2.moments(mask)
    if M['m00'] == 0:
        return None
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    return (cx, cy)

def distance_from_center(center, image_center):
    """Calculate the distance from the center of the image."""
    return np.sqrt((center[0] - image_center[0]) ** 2 + (center[1] - image_center[1]) ** 2)

for image_name in os.listdir(input_dir):
    if image_name.endswith((".jpg", ".jpeg", ".png")):
        image_path = os.path.join(input_dir, image_name)
        frame = cv2.imread(image_path)
        
        height, width = frame.shape[:2]
        image_center = (width // 2, height // 2)

        results = model(frame)
        masks = results[0].masks.data.cpu().numpy() if results[0].masks is not None else None

        if masks is None or len(masks) == 0:
            print(f"No objects found in {image_name}")
            continue

        centers = [get_center(mask) for mask in masks]
        distances = [distance_from_center(c, image_center) if c is not None else float('inf') for c in centers]
        min_index = np.argmin(distances)
        mask = masks[min_index]

        mask_resized = cv2.resize(mask, (width, height))
        mask_applied = np.where(mask_resized > 0.5, 1, 0).astype(np.uint8)

        colored_mask = cv2.bitwise_and(frame, frame, mask=mask_applied)

        output_path = os.path.join(output_dir, image_name)
        cv2.imwrite(output_path, colored_mask)
        print(f"Processed {image_name}, saved output to {output_path}")

print("Processing complete.")