from flask import Flask, render_template, request, send_from_directory
import os
import cv2
import numpy as np
from flask import send_file
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from tensorflow.keras.models import load_model
from flask import session
from PIL import Image
import io


# === Paramètres ===
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
EXTRACTED_FOLDER = os.path.join(UPLOAD_FOLDER, 'extracted_rois')
COORDS_FILE = 'coords.txt'  # Ce fichier doit contenir x,y,w,h
MODEL_PATH = 'cnn_conformite_model.h5'
CLASS_NAMES = ["conforme", "non_conforme"]
IMG_SIZE = 28
app.secret_key = 'supersecret'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
@app.route('/uploads/<path:filename>')
def upload_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


# === Chargement du modèle ===
model = load_model(MODEL_PATH)

# === Fonctions ===
def extract_rois(image_path, output_dir, coord_file):
    image = cv2.imread(image_path)
    os.makedirs(output_dir, exist_ok=True)

    with open(coord_file, "r") as f:
        coords = [tuple(map(int, line.strip().split(","))) for line in f]

    roi_paths = []

    for i, (x, y, w, h) in enumerate(coords):
        roi = image[y:y+h, x:x+w]
        roi_path = os.path.join(output_dir, f"roi_{i:02}.jpg")
        cv2.imwrite(roi_path, roi)
        roi_paths.append(roi_path)

    return roi_paths

def prepare_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype("float32") / 255.0
    return img.reshape(1, IMG_SIZE, IMG_SIZE, 1)

def predict_image(path):
    img = prepare_image(path)
    prediction = model.predict(img, verbose=0)
    class_index = np.argmax(prediction)
    confidence = prediction[0][class_index]
    return CLASS_NAMES[class_index], round(confidence * 100, 2)

def generate_pdf(results):
    buffer = io.BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    y = height - 40

    p.setFont("Helvetica-Bold", 14)
    p.drawString(30, y, "Rapport de prédiction")
    y -= 30

    p.setFont("Helvetica", 12)
    for item in results:
        line = f"{item['image']} ➤ {item['label']} ({item['confidence']}%)"
        p.drawString(30, y, line)
        y -= 20
        if y < 50:
            p.showPage()
            y = height - 40

    p.save()
    buffer.seek(0)
    return buffer


def extraire_Couleur_conforme(image_path): 
    # Ouvrir l'image et convertir en RGB
    img = Image.open(image_path).convert("RGB")
    # Récupérer les pixels et supprimer les doublons
    unique_colors = set(img.getdata())
    print(f"Nombre de couleurs uniques dans l'image : {len(unique_colors)}")
    # Afficher chaque couleur unique
    print("Couleurs RGB uniques dans l'image :")
    return unique_colors


# === Routes ===
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    if not file:
        return "❌ Aucun fichier uploadé"

    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(EXTRACTED_FOLDER, exist_ok=True)

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    roi_paths = extract_rois(file_path, EXTRACTED_FOLDER, COORDS_FILE)

    results = []
    for path in roi_paths:
        label, confidence = predict_image(path)
        results.append({
            'image': os.path.basename(path),
            'label': label,
            'confidence': confidence
        })

    session['results'] = results
    return render_template(
        'result.html',
        results=results,
        input_image=file.filename  # 🟢 Passe le nom de l'image ici
    )

@app.route('/download')
def download_report():
    results = session.get('results', [])
    if not results:
        return "Aucun résultat à inclure."
    pdf = generate_pdf(results)
    return send_file(pdf, as_attachment=True, download_name='rapport_predictions.pdf', mimetype='application/pdf')


# === Exécution ===
if __name__ == '__main__':
    app.run(debug=True)
