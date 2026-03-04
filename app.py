import os
import numpy as np
import cv2
import pytesseract
from flask import Flask, request, render_template_string
from supabase import create_client
from rapidfuzz import fuzz
from skimage.metrics import structural_similarity as ssim

# -------------------------------
# Flask Setup
# -------------------------------
app = Flask(__name__)

# -------------------------------
# Supabase Setup
# -------------------------------
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# -------------------------------
# HTML UI
# -------------------------------
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
    width: 400px;
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
}
</style>
</head>
<body>
<h1>KYC Verification System</h1>
<div class="container">
<form method="POST" enctype="multipart/form-data">
<input type="text" name="name" placeholder="Enter Name" required>
<input type="text" name="dob" placeholder="Enter DOB (as in Aadhaar)" required>
<input type="file" name="aadhaar" required>
<input type="file" name="selfie" required>
<button type="submit">Verify</button>
</form>
</div>
</body>
</html>
"""

# -------------------------------
# Home Route
# -------------------------------
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":

        name = request.form["name"]
        dob = request.form["dob"]

        aadhaar_file = request.files["aadhaar"]
        selfie_file = request.files["selfie"]

        # Convert to OpenCV format (memory processing)
        aadhaar_np = np.frombuffer(aadhaar_file.read(), np.uint8)
        selfie_np = np.frombuffer(selfie_file.read(), np.uint8)

        aadhaar_img = cv2.imdecode(aadhaar_np, cv2.IMREAD_COLOR)
        selfie_img = cv2.imdecode(selfie_np, cv2.IMREAD_COLOR)

        # ---------------- OCR ----------------
        gray = cv2.cvtColor(aadhaar_img, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(gray)

        name_match = fuzz.partial_ratio(name.lower(), text.lower())
        dob_match = fuzz.partial_ratio(dob.lower(), text.lower())

        # ---------------- Face Match ----------------
        aadhaar_face = cv2.resize(aadhaar_img, (200, 200))
        selfie_face = cv2.resize(selfie_img, (200, 200))

        aadhaar_gray = cv2.cvtColor(aadhaar_face, cv2.COLOR_BGR2GRAY)
        selfie_gray = cv2.cvtColor(selfie_face, cv2.COLOR_BGR2GRAY)

        score = ssim(aadhaar_gray, selfie_gray)
        face_match = score * 100

        # ---------------- Decision ----------------
        if name_match > 60 and dob_match > 60 and face_match > 40:

            # Store in Supabase
            supabase.table("kyc_records").insert({
                "name": name,
                "dob": dob,
                "status": "Verified"
            }).execute()

            return "<h2 style='color:green;text-align:center;'>KYC Verified Successfully</h2>"

        else:
            return "<h2 style='color:red;text-align:center;'>KYC Rejected</h2>"

    return render_template_string(HTML_PAGE)


# -------------------------------
# Render Entry
# -------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)