import os, cv2, base64, math, pickle, json
import numpy as np
from flask import Flask, render_template, jsonify, request

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir:
    os.chdir(script_dir)

app = Flask(__name__, template_folder='templates')

# ── MediaPipe Hand Landmarker Setup ───────────────────────────
import mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions

MODEL_PATH = os.path.join(script_dir, 'hand_landmarker.task')

# Create the HandLandmarker (reusable instance)
hand_options = vision.HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)
hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)
print("[OK] MediaPipe HandLandmarker loaded")

# ── Optional: Load trained sklearn classifier ─────────────────
sklearn_model = None
sklearn_labels = None
CLASSIFIER_PATH = os.path.join(script_dir, 'isl_classifier.pkl')

if os.path.exists(CLASSIFIER_PATH):
    try:
        with open(CLASSIFIER_PATH, 'rb') as f:
            data = pickle.load(f)
            sklearn_model = data['model']
            sklearn_labels = data['labels']
        print(f"[OK] Loaded custom classifier with {len(sklearn_labels)} classes")
    except Exception as e:
        print(f"[WARN] Could not load classifier: {e}")

# ── Finger Position Classifier (Geometric Heuristics) ─────────
# Works without any training data — purely based on finger geometry

# MediaPipe hand landmark indices
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20


def get_finger_states(landmarks, handedness='Right'):
    """
    Determine which fingers are up (extended) based on landmark positions.
    Returns a list of 5 booleans: [thumb, index, middle, ring, pinky]
    """
    fingers = []

    # Thumb: compare x-coordinate (direction depends on handedness)
    if handedness == 'Right':
        fingers.append(landmarks[THUMB_TIP].x < landmarks[THUMB_IP].x)
    else:
        fingers.append(landmarks[THUMB_TIP].x > landmarks[THUMB_IP].x)

    # Other 4 fingers: tip is above (lower y) than PIP joint = finger is up
    for tip, pip in [(INDEX_TIP, INDEX_PIP),
                     (MIDDLE_TIP, MIDDLE_PIP),
                     (RING_TIP, RING_PIP),
                     (PINKY_TIP, PINKY_PIP)]:
        fingers.append(landmarks[tip].y < landmarks[pip].y)

    return fingers


def classify_isl_single_hand(finger_states, landmarks, handedness='Right'):
    """
    Classify ISL (Indian Sign Language) gestures from a single hand.
    ISL one-handed signs: C, I, L, O, U, V and numbers 0-9.
    Returns (label, confidence).
    """
    thumb, index, middle, ring, pinky = finger_states
    total_up = sum(finger_states)

    # ── ISL Numbers (0-9) ─────────────────────────────────
    if total_up == 0:
        # Check if it's a fist (0) or curved C/O shape
        thumb_tip = landmarks[THUMB_TIP]
        index_tip = landmarks[INDEX_TIP]
        dist = math.sqrt((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)
        if dist < 0.05:
            return "ISL: O (ओ)", 85.0
        elif dist > 0.08:
            return "ISL: C (सी)", 82.0
        else:
            return "ISL: 0 (शून्य)", 88.0

    if total_up == 1:
        if index:
            # Index finger pointing up = ISL 1
            return "ISL: 1 (एक)", 92.0
        if thumb:
            # Thumb up = ISL "good" / thumbs up gesture
            return "ISL: अच्छा (Good)", 90.0
        if pinky:
            # Pinky up = ISL I
            return "ISL: I (आई)", 88.0

    if total_up == 2:
        if index and middle:
            idx_x = landmarks[INDEX_TIP].x
            mid_x = landmarks[MIDDLE_TIP].x
            spread = abs(idx_x - mid_x)
            if spread > 0.05:
                return "ISL: V (वी) / 2 (दो)", 92.0
            else:
                return "ISL: U (यू) / 2 (दो)", 88.0
        if thumb and index:
            return "ISL: L (एल)", 90.0
        if thumb and pinky:
            return "ISL: Y (वाय)", 88.0
        if index and pinky:
            return "ISL: 🤘", 82.0

    if total_up == 3:
        if index and middle and ring:
            return "ISL: W (डब्ल्यू) / 3 (तीन)", 88.0
        if thumb and index and middle:
            return "ISL: 3 (तीन)", 85.0

    if total_up == 4:
        if not thumb:
            return "ISL: 4 (चार)", 88.0

    if total_up == 5:
        return "ISL: 5 (पाँच) / खुला हाथ", 92.0

    # Thumb alongside fist = ISL A (one-hand variant)
    if total_up == 0 or (total_up == 1 and thumb):
        thumb_y = landmarks[THUMB_TIP].y
        index_mcp_y = landmarks[INDEX_MCP].y
        if thumb_y < index_mcp_y:
            return "ISL: A (ए)", 82.0

    return f"ISL: ? ({total_up} उँगलियाँ)", 45.0


def classify_isl_two_hands(hand1_fingers, hand2_fingers, hand1_lms, hand2_lms):
    """
    Classify ISL two-handed alphabet gestures.
    Many ISL letters (A, B, D, E, F, G, H, K, M, N, P, Q, R, S, T, X, Z)
    are two-handed signs where both hands form specific shapes together.
    """
    total1 = sum(hand1_fingers)
    total2 = sum(hand2_fingers)
    total_both = total1 + total2

    # Both fists touching = ISL: A (ए)
    if total1 == 0 and total2 == 0:
        return "ISL: A (ए) [2-Handed]", 88.0

    # Both hands open = ISL: B (बी)
    if total1 == 5 and total2 == 5:
        return "ISL: B (बी) [2-Handed]", 90.0

    # One fist + one open = ISL: D (डी)
    if (total1 == 0 and total2 == 5) or (total1 == 5 and total2 == 0):
        return "ISL: D (डी) [2-Handed]", 85.0

    # Both hands index pointing = ISL: H (एच)
    if (total1 == 1 and hand1_fingers[1]) and (total2 == 1 and hand2_fingers[1]):
        return "ISL: H (एच) [2-Handed]", 86.0

    # One hand index + one fist = ISL: G (जी)
    if (total1 == 1 and hand1_fingers[1] and total2 == 0) or \
       (total2 == 1 and hand2_fingers[1] and total1 == 0):
        return "ISL: G (जी) [2-Handed]", 84.0

    # One hand V + one fist = ISL: K (के)
    t1_idx_mid = hand1_fingers[1] and hand1_fingers[2] and total1 == 2
    t2_idx_mid = hand2_fingers[1] and hand2_fingers[2] and total2 == 2
    if (t1_idx_mid and total2 == 0) or (t2_idx_mid and total1 == 0):
        return "ISL: K (के) [2-Handed]", 83.0

    # Both V signs = ISL: X (एक्स)
    if t1_idx_mid and t2_idx_mid:
        return "ISL: X (एक्स) [2-Handed]", 84.0

    # One open hand + one index = ISL: P (पी)
    if (total1 == 5 and total2 == 1 and hand2_fingers[1]) or \
       (total2 == 5 and total1 == 1 and hand1_fingers[1]):
        return "ISL: P (पी) [2-Handed]", 83.0

    # Both hands 3 fingers = ISL: M (एम)
    if total1 == 3 and total2 == 3:
        return "ISL: M (एम) [2-Handed]", 82.0

    # Both hands 4 fingers = ISL: N (एन)  
    if total1 == 4 and total2 == 4:
        return "ISL: N (एन) [2-Handed]", 82.0

    # One open + one V = ISL: R (आर)
    if (total1 == 5 and t2_idx_mid) or (total2 == 5 and t1_idx_mid):
        return "ISL: R (आर) [2-Handed]", 82.0

    # Fallback for unrecognized 2-hand combo
    return f"ISL: 2-हाथ ({total1}+{total2} उँगलियाँ)", 50.0


def normalize_landmarks(landmarks):
    """Normalize landmarks relative to wrist for ML model input."""
    wrist = landmarks[WRIST]
    features = []
    for lm in landmarks:
        features.extend([lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z])
    return features


def classify_gesture(landmarks, handedness='Right'):
    """
    Main classification function for single hand.
    Uses trained model if available, otherwise falls back to ISL geometric heuristics.
    """
    # Try sklearn model first (if trained)
    if sklearn_model is not None and sklearn_labels is not None:
        try:
            features = normalize_landmarks(landmarks)
            prediction = sklearn_model.predict([features])[0]
            probabilities = sklearn_model.predict_proba([features])[0]
            confidence = float(max(probabilities)) * 100
            label = sklearn_labels[prediction]
            if confidence > 40:
                return label, confidence
        except Exception:
            pass

    # Fallback: ISL geometric finger-state classifier
    finger_states = get_finger_states(landmarks, handedness)
    return classify_isl_single_hand(finger_states, landmarks, handedness)


# ── Drawing Utilities ─────────────────────────────────────────

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),       # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),       # Index
    (0, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (0, 13), (13, 14), (14, 15), (15, 16), # Ring
    (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
    (5, 9), (9, 13), (13, 17),             # Palm
]


def draw_hand_landmarks(frame, landmarks, label, confidence):
    """Draw hand skeleton, bounding box, and label on the frame."""
    h, w = frame.shape[:2]

    # Convert normalized landmarks to pixel coordinates
    points = []
    for lm in landmarks:
        px, py = int(lm.x * w), int(lm.y * h)
        points.append((px, py))

    # Draw connections (skeleton)
    for start, end in HAND_CONNECTIONS:
        if start < len(points) and end < len(points):
            cv2.line(frame, points[start], points[end], (0, 255, 128), 2)

    # Draw landmark dots
    for i, (px, py) in enumerate(points):
        color = (0, 0, 255) if i in [4, 8, 12, 16, 20] else (255, 255, 0)  # Red for tips, cyan for others
        cv2.circle(frame, (px, py), 5, color, -1)
        cv2.circle(frame, (px, py), 5, (0, 0, 0), 1)  # Black outline

    # Compute bounding box
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    margin = 20
    x1, y1 = max(0, min(xs) - margin), max(0, min(ys) - margin)
    x2, y2 = min(w, max(xs) + margin), min(h, max(ys) + margin)

    # Draw GREEN bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

    # Draw label above the box
    label_text = f"{label} ({confidence:.0f}%)"
    text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
    cv2.rectangle(frame, (x1, y1 - text_size[1] - 15), (x1 + text_size[0] + 10, y1), (0, 255, 0), -1)
    cv2.putText(frame, label_text, (x1 + 5, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    return frame


# ── Flask Routes ──────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        img_data = data['image'].split(',')[1]
        nparr = np.frombuffer(base64.b64decode(img_data), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Flip horizontally (mirror effect)
        img_flip = cv2.flip(img, 1)
        img_resize = cv2.resize(img_flip, (640, 480))

        # Convert to RGB for MediaPipe
        img_rgb = cv2.cvtColor(img_resize, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

        # Detect hands
        result = hand_landmarker.detect(mp_image)

        predictions = []

        if result.hand_landmarks:
            # Collect finger states for all hands
            all_finger_states = []
            all_handedness = []
            for i, hand_lms in enumerate(result.hand_landmarks):
                handedness = 'Right'
                if result.handedness and i < len(result.handedness):
                    handedness = result.handedness[i][0].category_name
                all_handedness.append(handedness)
                all_finger_states.append(get_finger_states(hand_lms, handedness))

            # ── 2-HAND ISL DETECTION ──────────────────────────
            if len(result.hand_landmarks) >= 2:
                label, confidence = classify_isl_two_hands(
                    all_finger_states[0], all_finger_states[1],
                    result.hand_landmarks[0], result.hand_landmarks[1]
                )

                # Draw both hands
                for i, hand_lms in enumerate(result.hand_landmarks[:2]):
                    img_resize = draw_hand_landmarks(
                        img_resize, hand_lms, label if i == 0 else "", confidence if i == 0 else 0
                    )

                predictions.append({
                    'label': label,
                    'confidence': confidence,
                    'handedness': '2-Handed ISL'
                })

            # ── 1-HAND ISL DETECTION ──────────────────────────
            else:
                hand_lms = result.hand_landmarks[0]
                handedness = all_handedness[0]
                label, confidence = classify_gesture(hand_lms, handedness)
                img_resize = draw_hand_landmarks(img_resize, hand_lms, label, confidence)

                predictions.append({
                    'label': label,
                    'confidence': confidence,
                    'handedness': handedness
                })

        # Encode annotated frame
        _, buf_frame = cv2.imencode('.jpg', img_resize, [cv2.IMWRITE_JPEG_QUALITY, 80])
        frame_b64 = base64.b64encode(buf_frame).decode('utf-8')

        # Primary prediction (highest confidence)
        best_pred = ""
        best_conf = 0.0
        if predictions:
            best = max(predictions, key=lambda p: p['confidence'])
            best_pred = best['label']
            best_conf = best['confidence']

        return jsonify({
            'pred_text': best_pred,
            'confidence': best_conf,
            'frame_img': frame_b64,
            'hands_detected': len(predictions),
            'all_predictions': predictions,
            'model_type': 'MediaPipe HandLandmarker + Finger Classifier'
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/gestures')
def get_gestures():
    """Return list of recognizable ISL gestures."""
    gestures = [
        "── 1-Hand ISL Signs ──",
        "0 (शून्य) - Fist",
        "1 (एक) - Index up",
        "2 (दो) / V - Index+Middle",
        "3 (तीन) - Thumb+Index+Middle",
        "4 (चार) - Four fingers",
        "5 (पाँच) - Open hand",
        "C (सी) - Curved hand",
        "I (आई) - Pinky up",
        "L (एल) - Thumb+Index",
        "O (ओ) - Circle shape",
        "U (यू) - Index+Middle together",
        "Y (वाय) - Thumb+Pinky",
        "── 2-Hand ISL Signs ──",
        "A (ए) - Both fists",
        "B (बी) - Both open",
        "D (डी) - Fist+Open",
        "G (जी) - Index+Fist",
        "H (एच) - Both index",
        "K (के) - V+Fist",
        "M (एम) - Both 3 fingers",
        "N (एन) - Both 4 fingers",
        "P (पी) - Open+Index",
        "R (आर) - Open+V",
        "X (एक्स) - Both V signs",
    ]
    if sklearn_labels:
        gestures = list(sklearn_labels.values()) if isinstance(sklearn_labels, dict) else list(sklearn_labels)
    return jsonify(gestures)


if __name__ == '__main__':
    print("=" * 55)
    print("  Indian Sign Language (ISL) Interpreter")
    print("  Powered by MediaPipe HandLandmarker")
    print(f"  Custom classifier: {'YES' if sklearn_model else 'NO (using finger geometry)'}")
    print("  Server: http://localhost:5000")
    print("=" * 55)
    app.run(host='0.0.0.0', port=5000, debug=False)
