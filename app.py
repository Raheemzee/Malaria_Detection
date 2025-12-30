from flask import Flask, request, render_template, send_from_directory
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
from fastai.vision.all import load_learner, PILImage

app = Flask(__name__)

# ------------------ Paths ------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
RESULT_FOLDER = os.path.join(BASE_DIR, "static", "results")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["RESULT_FOLDER"] = RESULT_FOLDER

# ------------------ Load Blood Cell Model ------------------

model_path = os.path.join(BASE_DIR, "blood_cell_classifier.pkl")

import pathlib, sys
if sys.platform != "win32":
    pathlib.WindowsPath = pathlib.PosixPath

learn = load_learner(model_path)

# ------------------ BLOOD CELL FUNCTIONS ------------------

def segment_blood_cells(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    cell_images, boxes = [], []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 10 and h > 10:
            cell_images.append(img[y:y+h, x:x+w])
            boxes.append((x, y, w, h))

    return cell_images, boxes, img


def classify_and_count_cells(cell_images):
    cell_counts = {}

    for cell in cell_images:
        pil_img = PILImage.create(cv2.cvtColor(cell, cv2.COLOR_BGR2RGB))
        pred_class, _, _ = learn.predict(pil_img)
        cell_counts[pred_class] = cell_counts.get(pred_class, 0) + 1

    return cell_counts


def process_smear_image(image_path, filename):
    cell_images, boxes, img = segment_blood_cells(image_path)
    cell_counts = classify_and_count_cells(cell_images)

    for (x, y, w, h) in boxes:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    result_path = os.path.join(app.config["RESULT_FOLDER"], filename)
    cv2.imwrite(result_path, img)

    return cell_counts, f"/static/results/{filename}"

# ------------------ MALARIA DETECTION ------------------

def detect_malaria_parasites(image_path, filename):
    img = cv2.imread(image_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Giemsa-stained parasite color (purple/blue)
    lower = np.array([120, 40, 40])
    upper = np.array([160, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)

    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=1)

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    parasite_count = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 15 < area < 300:
            parasite_count += 1
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)

    result_path = os.path.join(app.config["RESULT_FOLDER"], filename)
    cv2.imwrite(result_path, img)

    return parasite_count, f"/static/results/{filename}"

# ------------------ ROUTES ------------------

@app.route("/", methods=["GET", "POST"])
def upload_file():
    results = []
    results_single = []

    if request.method == "POST":
        mode = request.form.get("mode")  # blood | malaria
        uploaded_files = request.files.getlist("files")

        for file in uploaded_files:
            if not file:
                continue

            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(file_path)

            if mode == "blood":
                cell_counts, processed_image_url = process_smear_image(file_path, filename)
                results.append({
                    "type": "blood",
                    "filename": filename,
                    "cell_counts": cell_counts,
                    "processed_image": processed_image_url
                })

            elif mode == "malaria":
                parasites, processed_image_url = detect_malaria_parasites(file_path, filename)
                results.append({
                    "type": "malaria",
                    "filename": filename,
                    "parasites": parasites,
                    "processed_image": processed_image_url,
                    "risk": (
                        "High" if parasites > 50 else
                        "Moderate" if parasites > 10 else
                        "Low"
                    )
                })

    return render_template("index.html", results=results, results_single=results_single)


@app.route("/classify-single", methods=["POST"])
def classify_single_cells():
    results = []
    results_single = []
    uploaded_files = request.files.getlist("single_files")

    for file in uploaded_files:
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(file_path)

            img = PILImage.create(file_path)
            pred_class, pred_idx, probs = learn.predict(img)
            results_single.append({
                "filename": filename,
                "predicted_class": str(pred_class),
                "confidence": f"{probs[pred_idx]:.4f}",
                "image_url": f"/uploads/{filename}"
            })

    return render_template("index.html", results=results, results_single=results_single)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# ------------------ Run ------------------

if __name__ == "__main__":
    app.run(debug=True)
