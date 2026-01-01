from flask import Flask, request, render_template, send_from_directory
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np

app = Flask(__name__)

# ------------------ Paths ------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
RESULT_FOLDER = os.path.join(BASE_DIR, "static", "results")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["RESULT_FOLDER"] = RESULT_FOLDER

# ------------------ MALARIA DETECTION ------------------

def detect_malaria_parasites(image_path, filename):
    img = cv2.imread(image_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Giemsa-stained malaria parasite (purple/blue)
    lower = np.array([120, 40, 40])
    upper = np.array([160, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=1)

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    parasite_count = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 1 < area < 30:
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

    if request.method == "POST":
        uploaded_files = request.files.getlist("files")

        for file in uploaded_files:
            if not file:
                continue

            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(file_path)

            parasites, processed_image_url = detect_malaria_parasites(
                file_path, filename
            )

            results.append({
                "filename": filename,
                "parasites": parasites,
                "processed_image": processed_image_url,
                "risk": (
                    "High" if parasites > 50 else
                    "Moderate" if parasites > 10 else
                    "Low"
                )
            })

    return render_template("index.html", results=results)


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

# ------------------ Run ------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
