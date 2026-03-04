import os
import io
from flask import Flask, render_template, request
from supabase import create_client
from PIL import Image
import numpy as np
from skimage.metrics import structural_similarity as ssim

app = Flask(__name__)

# 🔒 Prevent large uploads (important for Render free tier)
app.config['MAX_CONTENT_LENGTH'] = 3 * 1024 * 1024  # 3MB max

# 🔑 Supabase Config (DO NOT hardcode in production, use env vars)
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# 🎯 Face matching thresholds
STRONG_MATCH_THRESHOLD = 0.65
MANUAL_REVIEW_THRESHOLD = 0.55


# ================================
# Home Route
# ================================
@app.route("/")
def home():
    return render_template("index.html")


# ================================
# Image Preprocessing
# ================================
def preprocess_image(file):
    image = Image.open(file).convert("L")  # convert to grayscale
    image = image.resize((300, 300))       # resize to fixed size
    return np.array(image)


# ================================
# Face Similarity Function
# ================================
def calculate_similarity(img1, img2):
    score, _ = ssim(img1, img2, full=True)
    return score


# ================================
# Upload & Verify Route
# ================================
@app.route("/upload", methods=["POST"])
def upload():

    name = request.form.get("name")
    aadhaar_number = request.form.get("aadhaar_number")
    dob = request.form.get("dob")

    aadhaar_image = request.files.get("aadhaar_image")
    live_image = request.files.get("live_image")

    if not aadhaar_image or not live_image:
        return "Images missing"

    try:
        # 🔹 Preprocess both images
        img1 = preprocess_image(aadhaar_image)
        img2 = preprocess_image(live_image)

        similarity_score = calculate_similarity(img1, img2)

        # 🔹 Decide verification status
        if similarity_score >= STRONG_MATCH_THRESHOLD:
            verification_status = "VERIFIED"
            is_verified = True

        elif similarity_score >= MANUAL_REVIEW_THRESHOLD:
            verification_status = "MANUAL_REVIEW"
            is_verified = False

        else:
            verification_status = "REJECTED"
            is_verified = False

        # 🔹 Only store if VERIFIED
        if is_verified:
            supabase.table("verified_users").insert({
                "name": name,
                "aadhaar_number": aadhaar_number,
                "dob": dob,
                "similarity_score": float(similarity_score),
                "status": verification_status
            }).execute()

        return render_template(
            "result.html",
            status=verification_status,
            score=round(similarity_score * 100, 2)
        )

    except Exception as e:
        print("Error:", e)
        return "Something went wrong. Please try again."


# ================================
# Run App
# ================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)