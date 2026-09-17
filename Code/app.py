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


import numpy as np

def get_distance(lm1, lm2):
    return math.sqrt((lm1.x - lm2.x)**2 + (lm1.y - lm2.y)**2)

def get_angle_3d(a, b, c):
    """
    Calculate the 3D angle between vector AB and vector BC.
    If the finger is straight, the angle is close to 0 degrees.
    If the finger is curled, the angle is large (e.g. > 45 degrees).
    """
    v1 = np.array([b.x - a.x, b.y - a.y, b.z - a.z])
    v2 = np.array([c.x - b.x, c.y - b.y, c.z - b.z])
    
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
        
    cosine_angle = np.dot(v1, v2) / (norm_v1 * norm_v2)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    return np.degrees(np.arccos(cosine_angle))

def get_finger_states(landmarks, handedness='Right'):
    """
    Determine which fingers are extended using true 3D joint angles.
    This makes detection completely immune to camera rotation or hand tilt!
    """
    fingers = []
    
    # Thumb: angle between (CMC->MCP) and (MCP->TIP)
    # Using 1, 2, 4
    thumb_angle = get_angle_3d(landmarks[1], landmarks[2], landmarks[4])
    fingers.append(thumb_angle < 35)
    
    # Index: MCP(5) -> PIP(6) -> TIP(8)
    idx_angle = get_angle_3d(landmarks[5], landmarks[6], landmarks[8])
    fingers.append(idx_angle < 45)
    
    # Middle: MCP(9) -> PIP(10) -> TIP(12)
    mid_angle = get_angle_3d(landmarks[9], landmarks[10], landmarks[12])
    fingers.append(mid_angle < 45)
    
    # Ring: MCP(13) -> PIP(14) -> TIP(16)
    ring_angle = get_angle_3d(landmarks[13], landmarks[14], landmarks[16])
    fingers.append(ring_angle < 45)
    
    # Pinky: MCP(17) -> PIP(18) -> TIP(20)
    pinky_angle = get_angle_3d(landmarks[17], landmarks[18], landmarks[20])
    fingers.append(pinky_angle < 45)
    
    return fingers


def classify_isl_single_hand(finger_states, landmarks, handedness='Right'):
    """
    True ISL rules for 1-handed signs (Numbers 1-9, C, L).
    Highly forgiving logic.
    """
    up_count = sum(finger_states)
    
    # ── Forgiving Letters ──
    # Letter C (Thumb & Index curved, distance is moderate)
    dist_c = get_distance(landmarks[THUMB_TIP], landmarks[INDEX_TIP])
    if not finger_states[2] and not finger_states[3] and not finger_states[4]:
        if 0.03 < dist_c < 0.12:
            return "ISL: C (सी)", 85.0
            
    # Letter L (Thumb & Index extended, distance is large)
    if finger_states[0] and finger_states[1] and not finger_states[2] and not finger_states[3]:
        if dist_c > 0.15:
            return "ISL: L (एल)", 88.0

    # ── Numbers ──
    if up_count == 1 and finger_states[1]: return "ISL: 1 (एक)", 95.0
    if up_count == 2 and finger_states[1] and finger_states[2]:
        spread = get_distance(landmarks[INDEX_TIP], landmarks[MIDDLE_TIP])
        if spread > 0.05: return "ISL: 2 (दो)", 92.0
    if up_count == 3 and finger_states[1] and finger_states[2] and finger_states[0]: 
        return "ISL: 3 (तीन)", 90.0
    if up_count == 4 and not finger_states[0]: 
        return "ISL: 4 (चार)", 90.0
    if up_count == 5: 
        return "ISL: 5 (पाँच)", 92.0

    if up_count == 0:
        return "ISL: 0 (शून्य)", 85.0
        
    return f"ISL: ? ({up_count} उँगलियाँ)", 45.0


def get_palm_size(landmarks):
    # Distance from WRIST (0) to MIDDLE_MCP (9)
    return get_distance(landmarks[0], landmarks[9])

def classify_isl_two_hands(f1, f2, lms1, lms2):
    """
    True ISL 2-handed alphabet logic (Scale-Invariant Model).
    Uses relative distances so it works perfectly regardless of camera distance!
    """
    for base_lms, ptr_lms, base_f, ptr_f in [(lms1, lms2, f1, f2), (lms2, lms1, f2, f1)]:
        
        ptr_tip = ptr_lms[INDEX_TIP]
        palm_size = get_palm_size(base_lms)
        if palm_size == 0: palm_size = 0.01 # prevent division by zero
        
        def rel_dist(lm1, lm2):
            return get_distance(lm1, lm2) / palm_size
        
        # ── MISSING NUMBERS (6, 7, 8, 9) ──
        # ISL 6-9 are often signed as 5 on one hand, and 1-4 on the other.
        if sum(base_f) == 5:
            ptr_up = sum(ptr_f)
            if ptr_up == 1: return "ISL: 6 (छह)", 90.0
            if ptr_up == 2: return "ISL: 7 (सात)", 90.0
            if ptr_up == 3: return "ISL: 8 (आठ)", 90.0
            if ptr_up == 4: return "ISL: 9 (नौ)", 90.0
        
        # ── VOWELS (A, E, I, O, U) ──
        if rel_dist(ptr_tip, base_lms[THUMB_TIP]) < 0.7: return "ISL: A (ए)", 90.0
        if rel_dist(ptr_tip, base_lms[INDEX_TIP]) < 0.7: return "ISL: E (ई)", 90.0
        if rel_dist(ptr_tip, base_lms[MIDDLE_TIP]) < 0.7: return "ISL: I (आई)", 90.0
        if rel_dist(ptr_tip, base_lms[RING_TIP]) < 0.7: return "ISL: O (ओ)", 90.0
        if rel_dist(ptr_tip, base_lms[PINKY_TIP]) < 0.7: return "ISL: U (यू)", 90.0
            
        # ── CONSONANTS ──
        base_palm = base_lms[9]
        d_idx = rel_dist(ptr_lms[INDEX_TIP], base_palm)
        d_mid = rel_dist(ptr_lms[MIDDLE_TIP], base_palm)
        d_ring = rel_dist(ptr_lms[RING_TIP], base_palm)
        
        # M, N, V
        if d_idx < 1.2 and d_mid < 1.2 and d_ring < 1.2:
            return "ISL: M (एम)", 88.0
        elif d_idx < 1.2 and d_mid < 1.2:
            spread = rel_dist(ptr_lms[INDEX_TIP], ptr_lms[MIDDLE_TIP])
            if spread > 0.5: return "ISL: V (वी)", 88.0
            else: return "ISL: N (एन)", 88.0
                
        # D and P (Pointer touching Base Index)
        if rel_dist(ptr_lms[INDEX_TIP], base_lms[INDEX_TIP]) < 1.0 and \
           rel_dist(ptr_lms[THUMB_TIP], base_lms[INDEX_MCP]) < 1.0:
            return "ISL: D (डी)", 85.0
            
        # B (Hands open, sides touching)
        if rel_dist(base_lms[INDEX_MCP], ptr_lms[INDEX_MCP]) < 0.8 and sum(base_f) >= 4 and sum(ptr_f) >= 4:
            return "ISL: B (बी)", 85.0
            
        # X (Index fingers crossing/touching)
        if rel_dist(ptr_lms[INDEX_TIP], base_lms[INDEX_TIP]) < 0.8 and ptr_f[1] and base_f[1] and not ptr_f[2] and not base_f[2]:
            return "ISL: X (एक्स)", 85.0
            
        # S (Pinkies hooking/touching)
        if rel_dist(ptr_lms[PINKY_TIP], base_lms[PINKY_TIP]) < 0.8 and ptr_f[4] and base_f[4]:
            return "ISL: S (एस)", 85.0
            
        # T (Pointer index resting on base palm)
        if rel_dist(ptr_lms[INDEX_TIP], base_palm) < 0.8 and ptr_f[1] and sum(base_f) >= 4:
            return "ISL: T (टी)", 85.0
            
        # W (Fingers interlocked - Index to Index, Middle to Middle)
        if rel_dist(ptr_lms[INDEX_TIP], base_lms[INDEX_TIP]) < 0.8 and rel_dist(ptr_lms[MIDDLE_TIP], base_lms[MIDDLE_TIP]) < 0.8:
            return "ISL: W (डब्लू)", 80.0
            
    return "Unknown", 0.0


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

    # Fallback: ISL exact geometric spatial classifier
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
            predictions = []
            
            # Pre-calculate finger states for all hands
            all_finger_states = [get_finger_states(lms, result.handedness[i][0].category_name) 
                                 for i, lms in enumerate(result.hand_landmarks)]
            all_handedness = [result.handedness[i][0].category_name for i in range(len(result.hand_landmarks))]

            # ── 2-Hand Logic ────────────────────────────────────
            if len(result.hand_landmarks) >= 2:
                # Try to classify as a 2-handed interacting ISL sign
                label, confidence = classify_isl_two_hands(
                    all_finger_states[0], all_finger_states[1], 
                    result.hand_landmarks[0], result.hand_landmarks[1]
                )
                
                if confidence > 50:
                    # Successfully found a 2-handed interaction! Draw on both hands.
                    for i, hand_lms in enumerate(result.hand_landmarks[:2]):
                        img_resize = draw_hand_landmarks(
                            img_resize, hand_lms, label if i == 0 else "-->", confidence if i == 0 else 0
                        )
                    predictions.append({'label': label, 'confidence': confidence, 'handedness': '2-Handed ISL'})
                else:
                    # 2 hands present, but NO interaction. Treat as two separate 1-handed signs!
                    for i, hand_lms in enumerate(result.hand_landmarks):
                        h_label, h_conf = classify_gesture(hand_lms, all_handedness[i])
                        img_resize = draw_hand_landmarks(img_resize, hand_lms, h_label, h_conf)
                        predictions.append({'label': h_label, 'confidence': h_conf, 'handedness': all_handedness[i]})
            
            # ── 1-Hand Logic ────────────────────────────────────
            else:
                # Only 1 hand on screen
                hand_lms = result.hand_landmarks[0]
                handedness = all_handedness[0]
                label, confidence = classify_gesture(hand_lms, handedness)
                img_resize = draw_hand_landmarks(img_resize, hand_lms, label, confidence)
                predictions.append({'label': label, 'confidence': confidence, 'handedness': handedness})

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
        "2 (दो) - Index+Middle up",
        "3 (तीन) - Thumb+Index+Middle",
        "4 (चार) - Four fingers",
        "5 (पाँच) - Open hand",
        "C (सी) - Curved hand",
        "L (एल) - Thumb+Index",
        "── 2-Hand ISL Signs (True Spatial) ──",
        "6-9 (छह-नौ) - Base 5 + Pointer 1-4",
        "A (ए) - Index points to Thumb tip",
        "B (बी) - Open hands touching sides",
        "D (डी) - D shape (Pointer touches Base)",
        "E (ई) - Index points to Index tip",
        "I (आई) - Index points to Middle tip",
        "M (एम) - 3 fingers on palm",
        "N (एन) - 2 fingers on palm",
        "O (ओ) - Index points to Ring tip",
        "S (एस) - Pinkies hooked together",
        "T (टी) - Pointer index on base palm",
        "U (यू) - Index points to Pinky tip",
        "V (वी) - V fingers on palm",
        "W (डब्लू) - Fingers interlocked",
        "X (एक्स) - Index fingers crossed",
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
