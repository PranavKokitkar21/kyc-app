from flask import Flask, render_template, request
import os
import cv2
import pytesseract
import re
import base64
import numpy as np
import uuid
import requests
from rapidfuzz import fuzz
from skimage.metrics import structural_similarity as ssim

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# ---------------- ENV VARIABLES ----------------
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

HEADERS = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json"
}

# ---------------- FACE DETECTOR ----------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():

    name_input = request.form["name"]
    dob_input = request.form["dob"]
    aadhaar_file = request.files["aadhaar"]
    selfie_data = request.form["selfie_data"]

    # Save Aadhaar locally
    aadhaar_path = os.path.join(UPLOAD_FOLDER, aadhaar_file.filename)
    aadhaar_file.save(aadhaar_path)

    # Decode selfie
    header, encoded = selfie_data.split(",", 1)
    selfie_bytes = base64.b64decode(encoded)
    selfie_array = np.frombuffer(selfie_bytes, np.uint8)
    selfie_img = cv2.imdecode(selfie_array, cv2.IMREAD_COLOR)

    selfie_path = os.path.join(UPLOAD_FOLDER, "selfie.jpg")
    cv2.imwrite(selfie_path, selfie_img)

    # ---------------- OCR ----------------
    image = cv2.imread(aadhaar_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    extracted_text = pytesseract.image_to_string(gray)

    pattern = r"\d{4}\s?\d{4}\s?\d{4}"
    match = re.search(pattern, extracted_text)

    if match:
        aadhaar_number = match.group()
        aadhaar_clean = aadhaar_number.replace(" ", "")
        aadhaar_valid = len(aadhaar_clean) == 12 and aadhaar_clean.isdigit()
        masked_aadhaar = "XXXX-XXXX-" + aadhaar_clean[-4:]
    else:
        aadhaar_valid = False
        masked_aadhaar = "Not Found"

    # Name Matching
    name_score = fuzz.partial_ratio(name_input.lower(), extracted_text.lower())

    # DOB Matching
    dob_match = dob_input in extracted_text

    # ---------------- FACE VERIFICATION ----------------
    face_score = 0

    aadhaar_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    selfie_gray = cv2.cvtColor(selfie_img, cv2.COLOR_BGR2GRAY)

    aadhaar_faces = face_cascade.detectMultiScale(aadhaar_gray, 1.3, 5)
    selfie_faces = face_cascade.detectMultiScale(selfie_gray, 1.3, 5)

    if len(aadhaar_faces) > 0 and len(selfie_faces) > 0:
        (x, y, w, h) = aadhaar_faces[0]
        aadhaar_face = aadhaar_gray[y:y+h, x:x+w]

        (x2, y2, w2, h2) = selfie_faces[0]
        selfie_face = selfie_gray[y2:y2+h2, x2:x2+w2]

        aadhaar_face = cv2.resize(aadhaar_face, (200, 200))
        selfie_face = cv2.resize(selfie_face, (200, 200))

        similarity, _ = ssim(aadhaar_face, selfie_face, full=True)
        face_score = round(similarity * 100, 2)

    # ---------------- FINAL SCORING ----------------
    score = 0

    if aadhaar_valid:
        score += 30

    if name_score > 70:
        score += 20

    if dob_match:
        score += 20

    face_points = int(face_score * 0.3)
    score += face_points

    if score >= 80:
        final_status = "KYC VERIFIED ✅"
        color = "green"

        user_id = str(uuid.uuid4())

        data = {
            "id": user_id,
            "full_name": name_input,
            "dob": dob_input,
            "masked_aadhaar": masked_aadhaar,
            "aadhaar_image_url": aadhaar_path,
            "selfie_image_url": selfie_path,
            "face_score": face_score,
            "name_score": name_score,
            "total_score": score,
            "kyc_status": "VERIFIED"
        }

        try:
            response = requests.post(
                f"{SUPABASE_URL}/rest/v1/verified_users",
                json=data,
                headers=HEADERS,
                timeout=10
            )
            print("Supabase response:", response.status_code)
        except Exception as e:
            print("Supabase error:", e)

    else:
        final_status = "KYC REJECTED ❌"
        color = "red"

    return f"""
    <div style='font-family:Arial;padding:30px;'>
    <h2 style='color:{color};'>{final_status}</h2>
    <hr>
    <p><b>Masked Aadhaar:</b> {masked_aadhaar}</p>
    <p><b>Aadhaar Valid:</b> {aadhaar_valid}</p>
    <p><b>Name Match Score:</b> {name_score}%</p>
    <p><b>DOB Match:</b> {dob_match}</p>
    <p><b>Face Similarity Score:</b> {face_score}%</p>
    <p><b>Total Score:</b> {score}/100</p>
    </div>
    """

if __name__ == "__main__":
    app.run()