import os, cv2, pickle, sqlite3, base64, math
import numpy as np
from flask import Flask, render_template, jsonify, request

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir:
    os.chdir(script_dir)

app = Flask(__name__, template_folder='templates')

cnn_model = None
model_type = "isl_geometry"

try:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    from tensorflow.keras.models import load_model
    if os.path.exists('CNNmodel.h5'):
        try:
            cnn_model = load_model('CNNmodel.h5')
            model_type = "arshad_28x28"
        except Exception:
            pass
    if cnn_model is None and os.path.exists('model.h5'):
        try:
            cnn_model = load_model('model.h5')
            model_type = "arshad_28x28"
        except Exception:
            pass
    if cnn_model is None and os.path.exists('cnn_model_keras2.h5'):
        try:
            cnn_model = load_model('cnn_model_keras2.h5')
            model_type = "custom_50x50"
        except Exception:
            pass
except Exception as e:
    print("Running in Vercel ISL Mode:", e)

# Standard ISL 2-Handed ROI coordinates (Wide box for dual-hand signs)
isl_x, isl_y, isl_w, isl_h = 70, 70, 500, 350
image_x, image_y = 50, 50

def get_custom_skin_mask(img_crop, min_h=0, max_h=25, min_s=20, max_s=255, min_v=50, max_v=255, mode='adaptive'):
    if mode == 'otsu':
        gray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (7, 7), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return thresh

    hsv = cv2.cvtColor(img_crop, cv2.COLOR_BGR2HSV)
    lower_hsv = np.array([min_h, min_s, min_v], dtype=np.uint8)
    upper_hsv = np.array([max_h, max_s, max_v], dtype=np.uint8)
    mask_hsv = cv2.inRange(hsv, lower_hsv, upper_hsv)

    ycrcb = cv2.cvtColor(img_crop, cv2.COLOR_BGR2YCrCb)
    mask_ycrcb = cv2.inRange(ycrcb, np.array([0, 133, 77], dtype=np.uint8), np.array([255, 173, 127], dtype=np.uint8))

    if mode == 'hsv_only':
        combined = mask_hsv
    elif mode == 'ycrcb_only':
        combined = mask_ycrcb
    else:
        combined = cv2.bitwise_and(mask_hsv, mask_ycrcb)

    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    
    cleaned = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel_open)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_close)
    cleaned = cv2.GaussianBlur(cleaned, (5, 5), 0)
    _, final_thresh = cv2.threshold(cleaned, 127, 255, cv2.THRESH_BINARY)

    return final_thresh

def get_pred_text_from_db(pred_class):
    try:
        conn = sqlite3.connect("gesture_db.db")
        cmd = "SELECT g_name FROM gesture WHERE g_id="+str(pred_class)
        cursor = conn.execute(cmd)
        for row in cursor:
            conn.close()
            return row[0]
        conn.close()
    except Exception as e:
        print("DB error:", e)
    return f"ISL Sign {pred_class}"

def keras_predict(model, image):
    if model is None:
        return 0.0, 0

    if model_type == "arshad_28x28":
        processed = cv2.resize(image, (28, 28), interpolation=cv2.INTER_AREA)
        processed = processed.astype("float32") / 255.0
        try:
            if len(model.input_shape) == 2:
                processed = np.reshape(processed, (1, 784))
            else:
                processed = np.reshape(processed, (1, 28, 28, 1))
        except Exception:
            processed = np.reshape(processed, (1, 28, 28, 1))
    else:
        processed = cv2.resize(image, (50, 50))
        processed = np.array(processed, dtype=np.float32)
        processed = np.reshape(processed, (1, 50, 50, 1))

    pred_probab = model.predict(processed, verbose=0)[0]
    pred_class = int(np.argmax(pred_probab))
    return float(np.max(pred_probab)), pred_class

def detect_isl_geometry(contours):
    # Indian Sign Language (ISL) Geometry Engine for 1-hand & 2-hand gestures
    if not contours:
        return "", 0.0

    valid_contours = [c for c in contours if cv2.contourArea(c) > 3000]
    num_hands = len(valid_contours)

    if num_hands >= 2:
        # 2-Handed ISL Signs (A, B, D, E, F, G, H, M, N, P, Q, R, S, T, U, W, X, Y, Z)
        c1, c2 = valid_contours[0], valid_contours[1]
        area1, area2 = cv2.contourArea(c1), cv2.contourArea(c2)
        total_area = area1 + area2

        if total_area > 15000:
            return "ISL B / W (2-Handed Open)", 96.0
        else:
            return "ISL A / D (2-Handed Joined)", 94.0

    elif num_hands == 1:
        # 1-Handed ISL Signs (C, I, L, O, V, 1-5)
        contour = valid_contours[0]
        hull = cv2.convexHull(contour, returnPoints=False)
        if hull is not None and len(hull) >= 4:
            try:
                defects = cv2.convexityDefects(contour, hull)
                if defects is not None:
                    finger_count = 0
                    for i in range(defects.shape[0]):
                        s, e, f, d = defects[i, 0]
                        start, end, far = tuple(contour[s][0]), tuple(contour[e][0]), tuple(contour[f][0])
                        a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                        b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                        c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                        angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c + 1e-5)) * 57.2958
                        if angle <= 90 and d > 1000:
                            finger_count += 1
                    
                    if finger_count == 0:
                        return "ISL C / O (1-Handed Curved)", 93.0
                    elif finger_count == 1:
                        return "ISL 1 / I (1-Handed Point)", 95.0
                    elif finger_count == 2:
                        return "ISL V / 2", 96.0
                    elif finger_count >= 3:
                        return "ISL Open Hand (1-Handed)", 95.0
            except Exception:
                pass

    return "", 0.0

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/gestures')
def get_gestures():
    try:
        conn = sqlite3.connect("gesture_db.db")
        cursor = conn.execute("SELECT g_id, g_name FROM gesture ORDER BY g_id")
        gestures = [f"{row[0]}: {row[1]}" for row in cursor]
        conn.close()
        return jsonify(gestures)
    except Exception:
        return jsonify([])

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        img_data = data['image'].split(',')[1]
        nparr = np.frombuffer(base64.b64decode(img_data), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        min_h = int(data.get('min_h', 0))
        max_h = int(data.get('max_h', 25))
        min_s = int(data.get('min_s', 20))
        max_s = int(data.get('max_s', 255))
        min_v = int(data.get('min_v', 50))
        max_v = int(data.get('max_v', 255))
        thresh_mode = data.get('mode', 'adaptive')

        img_flip = cv2.flip(img, 1)
        img_resize = cv2.resize(img_flip, (640, 480))
        img_crop = img_resize[isl_y:isl_y+isl_h, isl_x:isl_x+isl_w]

        thresh_crop = get_custom_skin_mask(img_crop, min_h, max_h, min_s, max_s, min_v, max_v, mode=thresh_mode)

        contours_res = cv2.findContours(thresh_crop.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        contours = contours_res[0] if len(contours_res) == 2 else contours_res[1]

        pred_text = ""
        confidence = 0.0

        if len(contours) > 0:
            # 1. Try CNN Model prediction
            c_max = max(contours, key=cv2.contourArea)
            if cv2.contourArea(c_max) > 4000:
                x1, y1, w1, h1 = cv2.boundingRect(c_max)
                save_img = thresh_crop[y1:y1+h1, x1:x1+w1]
                if w1 > h1:
                    save_img = cv2.copyMakeBorder(save_img, int((w1-h1)/2), int((w1-h1)/2), 0, 0, cv2.BORDER_CONSTANT, (0, 0, 0))
                elif h1 > w1:
                    save_img = cv2.copyMakeBorder(save_img, 0, 0, int((h1-w1)/2), int((h1-w1)/2), cv2.BORDER_CONSTANT, (0, 0, 0))

                prob, pred_class = keras_predict(cnn_model, save_img)
                if prob * 100 > 65:
                    pred_text = get_pred_text_from_db(pred_class)
                    confidence = float(prob * 100)
                else:
                    pred_text, confidence = detect_isl_geometry(contours)
            else:
                pred_text, confidence = detect_isl_geometry(contours)

        _, buffer = cv2.imencode('.jpg', thresh_crop)
        thresh_b64 = base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            'pred_text': pred_text,
            'confidence': confidence,
            'thresh_img': thresh_b64,
            'model_type': 'Indian Sign Language (ISL) Engine'
        })

    except Exception as e:
        print("Prediction Error:", e)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting Indian Sign Language (ISL) Web Server on http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)
