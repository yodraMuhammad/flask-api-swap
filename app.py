import os
from flask import Flask, request, jsonify
import cv2
import numpy as np
import insightface

# Inisialisasi Flask dan model face swap
app = Flask(__name__)

# Path ke file model yang telah diunduh
model_path = 'models/inswapper_128.onnx'
if not os.path.exists(model_path):
    raise FileNotFoundError(f'Model file {model_path} does not exist. Please download it and place it in the correct directory.')

# Memuat model
model = insightface.model_zoo.get_model(model_path)

def face_swap(source_img, target_img):
    # Inisialisasi detektor wajah
    detector = insightface.app.FaceAnalysis()
    detector.prepare(ctx_id=0, det_size=(640, 640))

    # Deteksi wajah di source_img
    source_faces = detector.get(source_img)
    if len(source_faces) == 0:
        raise ValueError("No face detected in the source image.")

    # Mengambil wajah pertama yang terdeteksi
    source_face = source_faces[0]

    # Deteksi wajah di target_img
    target_faces = detector.get(target_img)
    if len(target_faces) == 0:
        raise ValueError("No face detected in the target image.")

    # Mengambil wajah pertama yang terdeteksi di target_img
    target_face = target_faces[0]

    # Melakukan face swapping
    swapped_img = model.get(target_img, target_face, source_face, paste_back=True)
    return swapped_img

@app.route('/', methods=['GET'])
def home():
    return 'Create by yodra.muhamad@gmail.com'

@app.route('/swap', methods=['POST'])
def swap_faces():
    # Mengambil gambar dari request
    source_file = request.files['source']
    target_file = request.files['target']

    # Membaca gambar sebagai numpy array
    source_img = cv2.imdecode(np.frombuffer(source_file.read(), np.uint8), cv2.IMREAD_COLOR)
    target_img = cv2.imdecode(np.frombuffer(target_file.read(), np.uint8), cv2.IMREAD_COLOR)

    # Melakukan face swapping
    swapped_img = face_swap(source_img, target_img)

    # Encode gambar hasil swapping ke format jpg
    _, img_encoded = cv2.imencode('.jpg', swapped_img)

    # Mengembalikan hasil sebagai respons
    return img_encoded.tobytes(), 200, {'Content-Type': 'image/jpeg'}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
