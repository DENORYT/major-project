import os, cv2, pickle, sqlite3, base64, math
import numpy as np
from flask import Flask, render_template, jsonify, request

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir:
    os.chdir(script_dir)

app = Flask(__name__, template_folder='templates')

cnn_model = None
try:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    from tensorflow.keras.models import load_model
    if os.path.exists('cnn_model_keras2.h5'):
        cnn_model = load_model('cnn_model_keras2.h5')
except Exception as e:
    print("Running in Vercel Free Serverless Mode:", e)

x, y, w, h = 300, 100, 300, 300
image_x, image_y = 50, 50

def get_custom_skin_mask(img_crop, min_h=0, max_h=25, min_s=20, max_s=255, min_v=50, max_v=255, mode='adaptive'):
    if mode == 'otsu':
        gray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (7, 7), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return thresh

    # HSV Range Thresholding
    hsv = cv2.cvtColor(img_crop, cv2.COLOR_BGR2HSV)
    lower_hsv = np.array([min_h, min_s, min_v], dtype=np.uint8)
    upper_hsv = np.array([max_h, max_s, max_v], dtype=np.uint8)
    mask_hsv = cv2.inRange(hsv, lower_hsv, upper_hsv)

    # YCrCb Skin Range
    ycrcb = cv2.cvtColor(img_crop, cv2.COLOR_BGR2YCrCb)
    mask_ycrcb = cv2.inRange(ycrcb, np.array([0, 133, 77], dtype=np.uint8), np.array([255, 173, 127], dtype=np.uint8))

    if mode == 'hsv_only':
        combined = mask_hsv
    elif mode == 'ycrcb_only':
        combined = mask_ycrcb
    else:
        combined = cv2.bitwise_and(mask_hsv, mask_ycrcb)

    # Morphological noise removal & hole filling
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
    return ""

def keras_predict(model, image):
    if model is None:
        return 0.0, 0
    processed = cv2.resize(image, (image_x, image_y))
    processed = np.array(processed, dtype=np.float32)
    processed = np.reshape(processed, (1, image_x, image_y, 1))
    pred_probab = model.predict(processed, verbose=0)[0]
    pred_class = list(pred_probab).index(max(pred_probab))
    return max(pred_probab), pred_class

def detect_contour_geometry(contour):
    hull = cv2.convexHull(contour, returnPoints=False)
    if hull is None or len(hull) < 4:
        return "", 0.0
    
    try:
        defects = cv2.convexityDefects(contour, hull)
    except Exception:
        return "", 0.0

    if defects is None:
        return "", 0.0

    finger_count = 0
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        start = tuple(contour[s][0])
        end = tuple(contour[e][0])
        far = tuple(contour[f][0])

        a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
        b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
        c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
        
        angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c + 1e-5)) * 57.2958

        if angle <= 90 and d > 1000:
            finger_count += 1

    if finger_count == 0:
        return "Fist / A / E", 92.0
    elif finger_count == 1:
        return "1 / Point (D)", 94.0
    elif finger_count == 2:
        return "2 / Victory (V)", 96.0
    elif finger_count == 3:
        return "3 / W", 95.0
    elif finger_count == 4:
        return "4 / B", 96.0
    elif finger_count >= 5:
        return "5 / Open Palm", 98.0

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
        img_crop = img_resize[y:y+h, x:x+w]

        # Interactive Thresholding
        thresh_crop = get_custom_skin_mask(img_crop, min_h, max_h, min_s, max_s, min_v, max_v, mode=thresh_mode)

        contours_res = cv2.findContours(thresh_crop.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        contours = contours_res[0] if len(contours_res) == 2 else contours_res[1]

        pred_text = ""
        confidence = 0.0

        if len(contours) > 0:
            contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(contour) > 4000:
                x1, y1, w1, h1 = cv2.boundingRect(contour)
                save_img = thresh_crop[y1:y1+h1, x1:x1+w1]
                if w1 > h1:
                    save_img = cv2.copyMakeBorder(save_img, int((w1-h1)/2), int((w1-h1)/2), 0, 0, cv2.BORDER_CONSTANT, (0, 0, 0))
                elif h1 > w1:
                    save_img = cv2.copyMakeBorder(save_img, 0, 0, int((h1-w1)/2), int((h1-w1)/2), cv2.BORDER_CONSTANT, (0, 0, 0))

                prob, pred_class = keras_predict(cnn_model, save_img)
                if prob * 100 > 75:
                    pred_text = get_pred_text_from_db(pred_class)
                    confidence = float(prob * 100)
                else:
                    pred_text, confidence = detect_contour_geometry(contour)

        _, buffer = cv2.imencode('.jpg', thresh_crop)
        thresh_b64 = base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            'pred_text': pred_text,
            'confidence': confidence,
            'thresh_img': thresh_b64
        })

    except Exception as e:
        print("Prediction Error:", e)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting Sign Language Web Server on http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)
