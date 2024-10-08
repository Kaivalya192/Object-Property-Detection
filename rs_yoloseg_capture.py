import pyrealsense2 as rs
import cv2
from ultralytics import YOLO
from collections import deque
import numpy as np

model = YOLO("yolov8l-seg.pt")
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)
frame_history = deque(maxlen=100)

def get_center(mask):
    M = cv2.moments(mask)
    if M['m00'] == 0:
        return None
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    return (cx, cy)

def distance_from_center(center, image_center):
    return np.sqrt((center[0] - image_center[0]) ** 2 + (center[1] - image_center[1]) ** 2)

image_center = (320, 240)
mask_file_name = "object.png"

while True:
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    if not color_frame:
        continue

    frame = np.asanyarray(color_frame.get_data())
    results = model(frame)
    
    masks = results[0].masks.data.cpu().numpy() if results[0].masks is not None else None
    
    if masks is None or len(masks) == 0:
        # Show blank screen
        blank_frame = np.zeros_like(frame)
        cv2.imshow('Segmented Object', blank_frame)
        continue

    centers = [get_center(mask) for mask in masks]

    if centers:
        distances = [distance_from_center(c, image_center) if c is not None else float('inf') for c in centers]
        min_index = np.argmin(distances)
        frame_history.append(min_index)

        # Check for most common index and ensure it's within bounds
        if frame_history:
            most_common_index = max(set(frame_history), key=frame_history.count)
            if most_common_index < len(masks):  # Ensure index is valid
                mask = masks[most_common_index]

                # Resize and apply the mask
                mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
                mask_applied = np.where(mask_resized > 0.5, 1, 0).astype(np.uint8)  # Create binary mask

                # Create a colored output using the original frame
                colored_mask = cv2.bitwise_and(frame, frame, mask=mask_applied)

                cv2.imshow('Segmented Object', colored_mask)

                # Save the colored mask image
                cv2.imwrite(mask_file_name, colored_mask)  # Overwrite the previous mask
                print(f"Saved colored mask to {mask_file_name}")

            else:
                # If the most common index is out of bounds, show a blank screen
                blank_frame = np.zeros_like(frame)
                cv2.imshow('Segmented Object', blank_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

pipeline.stop()
cv2.destroyAllWindows()
