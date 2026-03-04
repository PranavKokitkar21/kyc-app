import os
import numpy as np
import cv2
import pytesseract
from flask import Flask, request, render_template_string
from supabase import create_client
from rapidfuzz import fuzz
from skimage.metrics import structural_similarity as ssim

app = Flask(__name__)

# ---------------- SUPABASE ----------------
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ---------------- UI ----------------
HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
<title>KYC Verification</title>
<style>
body {
    font-family: Arial;
    background: linear-gradient(135deg,#0f2027,#203a43,#2c5364);
    color: white;
    text-align: center;
    padding: 40px;
}
.container {
    background: rgba(255,255,255,0.1);
    padding: 30px;
    border-radius: 15px;
    width: 420px;
    margin: auto;
}
input, button {
    width: 100%;
    padding: 10px;
    margin: 10px 0;
    border-radius: 8px;
    border: none;
}
button {
    background: #00c6ff;
    color: black;
    font-weight: bold;
    cursor: pointer;
}
label {
    display: block;
    text-align: left;
    margin-top: 10px;
    font-weight: bold;
}
small {
    display: block;
    text-align: left;
    font-size: 12px;
    color: #ddd;
}
</style>
</head>
<body>

<h1>KYC Verification System</h1>

<div class="container">
<form method="POST" enctype="multipart/form-data">

<label>Full Name (as in Aadhaar)</label>
<input type="text" name="name" placeholder="Enter Full Name" required>

<label>Date of Birth (exact format as in Aadhaar)</label>
<input type="text" name="dob" placeholder="Example: 2003 or 12/05/2003" required>

<label>Upload Aadhaar Card Image</label>
<small>Please upload clear front side image of Aadhaar</small>
<input type="file" name="aadhaar" accept="image/*" required>

<label>Capture Live Selfie</label>
<small>On mobile this will open front camera automatically</small>
<input type="file" name="selfie" accept="image/*" capture="user" required>

<button type="submit">Verify KYC</button>

</form>
</div>

</body>
</html>
"""

# ---------------- ROUTE ----------------
@app.route("/", methods=["GET", "POST"])
def home():

    if request.method == "POST":

        name = request.form["name"]
        dob = request.form["dob"]

        aadhaar_file = request.files["aadhaar"]
        selfie_file = request.files["selfie"]

        # Read images in memory (NOT saved on disk)
        aadhaar_np = np.frombuffer(aadhaar_file.read(), np.uint8)
        selfie_np = np.frombuffer(selfie_file.read(), np.uint8)

        aadhaar_img = cv2.imdecode(aadhaar_np, cv2.IMREAD_COLOR)
        selfie_img = cv2.imdecode(selfie_np, cv2.IMREAD_COLOR)

        # -------- OCR ----------
        gray = cv2.cvtColor(aadhaar_img, cv2.COLOR_BGR2GRAY)
        extracted_text = pytesseract.image_to_string(gray)

        name_score = fuzz.partial_ratio(name.lower(), extracted_text.lower())
        dob_score = fuzz.partial_ratio(dob.lower(), extracted_text.lower())

        # -------- Face Match ----------
        aadhaar_resized = cv2.resize(aadhaar_img, (200, 200))
        selfie_resized = cv2.resize(selfie_img, (200, 200))

        aadhaar_gray = cv2.cvtColor(aadhaar_resized, cv2.COLOR_BGR2GRAY)
        selfie_gray = cv2.cvtColor(selfie_resized, cv2.COLOR_BGR2GRAY)

        face_score = ssim(aadhaar_gray, selfie_gray) * 100

        # -------- Decision ----------
        if name_score > 60 and dob_score > 60 and face_score > 40:

            supabase.table("verified_users").insert({
                "name_score": name_score,
                "face_score": round(face_score, 2),
                "status": "Verified"
            }).execute()

            return "<h2 style='color:lightgreen;text-align:center;'>KYC Verified Successfully ✅</h2>"

        else:
            return "<h2 style='color:red;text-align:center;'>KYC Rejected ❌</h2>"

    return render_template_string(HTML_PAGE)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)