import os
import cv2
import numpy as np
import mediapipe as mp
import pickle
import time
import json
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

with open('model.p', 'rb') as f:
    model_data = pickle.load(f)

model = model_data.get('model', None)
if model is None:
    raise ValueError("Model object not found in the pickle file.")

labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E',
    5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O',
    15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T',
    20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y',
    25: 'Z', 26: '0', 27: '1', 28: '2', 29: '3',
    30: '4', 31: '5', 32: '6', 33: '7', 34: '8', 35: '9'
}

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

@app.route('/')
def index():
    return jsonify({"status": "Sign Language API is running", "endpoints": ["/predict_video_batch", "/health"]})

@app.route('/health')
def health_check():
    return jsonify({"status": "healthy"}), 200

@app.route('/predict_video_batch', methods=['POST'])
def predict_video_batch():
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    video = request.files['video']
    if video.filename == '':
        return jsonify({"error": "No selected video"}), 400

    filename = secure_filename(video.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    video.save(filepath)

    cap = cv2.VideoCapture(filepath)
    raw_predictions = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                x_, y_ = [], []
                data_aux = []

                for lm in hand_landmarks.landmark:
                    x_.append(lm.x)
                    y_.append(lm.y)

                for lm in hand_landmarks.landmark:
                    data_aux.append(lm.x - min(x_))
                    data_aux.append(lm.y - min(y_))

                while len(data_aux) < 84:
                    data_aux.append(0)

                prediction = model.predict([np.asarray(data_aux)])
                predicted_char = labels_dict[int(prediction[0])]
                raw_predictions.append(predicted_char)

    cap.release()
    os.remove(filepath)
    
    # Remove consecutive duplicates
    predictions = []
    if raw_predictions:
        predictions.append(raw_predictions[0])
        for i in range(1, len(raw_predictions)):
            if raw_predictions[i] != raw_predictions[i-1]:
                predictions.append(raw_predictions[i])
    
    # Option to return both raw and filtered predictions
    return jsonify({
        "predictions": predictions,
        "raw_predictions": raw_predictions
    })

if __name__ == '__main__':
    # Use environment variable for port with a default of 5000
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
