"""
ISL Data Collection Tool
========================
Captures hand landmarks using MediaPipe and saves them to a CSV file.

Usage:
    python collect_data.py

Instructions:
    1. Run the script — webcam opens
    2. Show your hand gesture to the camera
    3. Press a KEY (A-Z or 0-9) to label the current gesture
    4. Each keypress captures ~20 samples of that gesture
    5. Press 'Q' to quit and save

The output file (landmark_data.csv) can be used with train_classifier.py
to train a custom gesture classifier.
"""

import os, cv2, csv, time
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

import mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions

# Setup
MODEL_PATH = os.path.join(script_dir, 'hand_landmarker.task')
OUTPUT_FILE = os.path.join(script_dir, 'landmark_data.csv')

hand_options = vision.HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
)
landmarker = vision.HandLandmarker.create_from_options(hand_options)

# Hand connections for drawing
CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17),
]

def normalize_landmarks(landmarks):
    """Normalize landmarks relative to wrist."""
    wrist = landmarks[0]
    features = []
    for lm in landmarks:
        features.extend([lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z])
    return features

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("ERROR: Cannot open camera!")
        return

    # Load existing data if any
    existing_data = []
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                existing_data.append(row)
        print(f"Loaded {len(existing_data)} existing samples")

    # CSV header: label, x0, y0, z0, x1, y1, z1, ..., x20, y20, z20
    header = ['label'] + [f'{c}{i}' for i in range(21) for c in ['x', 'y', 'z']]

    samples = existing_data.copy()
    label_counts = {}
    for row in existing_data:
        lbl = row[0]
        label_counts[lbl] = label_counts.get(lbl, 0) + 1

    print("\n" + "=" * 50)
    print("  ISL Data Collection Tool")
    print("=" * 50)
    print("Show your hand gesture to the camera, then")
    print("press a KEY (A-Z, 0-9) to label it.")
    print("Each keypress captures 20 samples.")
    print("Press Q to quit and save.")
    print("=" * 50)
    if label_counts:
        print("\nExisting labels:", dict(sorted(label_counts.items())))

    capturing_label = None
    capture_count = 0
    capture_target = 20

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        result = landmarker.detect(mp_image)

        hand_detected = False
        current_features = None

        if result.hand_landmarks:
            hand_lms = result.hand_landmarks[0]
            hand_detected = True
            current_features = normalize_landmarks(hand_lms)

            # Draw landmarks
            h, w = frame.shape[:2]
            points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_lms]

            for s, e in CONNECTIONS:
                cv2.line(frame, points[s], points[e], (0, 255, 128), 2)
            for i, (px, py) in enumerate(points):
                color = (0, 0, 255) if i in [4,8,12,16,20] else (255, 255, 0)
                cv2.circle(frame, (px, py), 5, color, -1)

            # Bounding box
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            cv2.rectangle(frame, (min(xs)-15, min(ys)-15), (max(xs)+15, max(ys)+15), (0, 255, 0), 2)

        # Capture if in progress
        if capturing_label and hand_detected and current_features and capture_count < capture_target:
            samples.append([capturing_label] + [f"{f:.6f}" for f in current_features])
            capture_count += 1
            label_counts[capturing_label] = label_counts.get(capturing_label, 0) + 1
            cv2.putText(frame, f"Capturing '{capturing_label}': {capture_count}/{capture_target}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        if capturing_label and capture_count >= capture_target:
            print(f"  Captured {capture_target} samples for '{capturing_label}'")
            capturing_label = None
            capture_count = 0

        # Status display
        status = "HAND DETECTED" if hand_detected else "No hand - show your hand"
        color = (0, 255, 0) if hand_detected else (0, 0, 255)
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, f"Total samples: {len(samples)}", (10, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(frame, "Press A-Z/0-9 to capture, Q to quit", (10, frame.shape[0] - 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        cv2.imshow('ISL Data Collector', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            break
        elif key != 255 and capturing_label is None:
            char = chr(key).upper()
            if char.isalnum():
                capturing_label = char
                capture_count = 0
                print(f"  Recording gesture '{char}' — hold your hand steady...")

    # Save
    cap.release()
    cv2.destroyAllWindows()

    if samples:
        with open(OUTPUT_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(samples)
        print(f"\nSaved {len(samples)} samples to {OUTPUT_FILE}")
        print("Label distribution:", dict(sorted(label_counts.items())))
        print("\nNext step: Run 'python train_classifier.py' to train your model!")
    else:
        print("\nNo samples collected.")


if __name__ == '__main__':
    main()
