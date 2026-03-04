import os
import base64
import numpy as np
import cv2
import pytesseract
import re
from flask import Flask, request, render_template_string
from supabase import create_client
from rapidfuzz import fuzz
from skimage.metrics import structural_similarity as ssim
from datetime import datetime

app = Flask(__name__)

# ---------------- SUPABASE ----------------

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ---------------- HTML UI ----------------

HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
<title>KYC Verification System</title>

<style>

body{
font-family:Arial;
background:linear-gradient(135deg,#0f2027,#203a43,#2c5364);
color:white;
text-align:center;
padding:40px;
}

.container{
background:rgba(255,255,255,0.1);
padding:30px;
border-radius:15px;
width:420px;
margin:auto;
}

input,button{
width:100%;
padding:10px;
margin:10px 0;
border-radius:8px;
border:none;
}

button{
background:#00c6ff;
color:black;
font-weight:bold;
cursor:pointer;
}

video{
width:100%;
border-radius:10px;
margin-top:10px;
}

</style>
</head>

<body>

<h1>KYC Verification System</h1>

<div class="container">

<form method="POST" enctype="multipart/form-data">

<input type="text" name="name" placeholder="Full Name (as in Aadhaar)" required>

<input type="text" name="dob" placeholder="DOB (as in Aadhaar)" required>

<label>Upload Aadhaar Image</label>
<input type="file" name="aadhaar" accept="image/*" required>

<label>Capture Live Selfie</label>

<video id="video" autoplay></video>

<button type="button" onclick="capture()">Capture Selfie</button>

<input type="hidden" name="selfie_data" id="selfie_data">

<button type="submit">Verify KYC</button>

</form>

</div>

<script>

const video=document.getElementById('video');

navigator.mediaDevices.getUserMedia({video:{facingMode:"user"}})

.then(stream=>{
video.srcObject=stream;
})

.catch(err=>{
alert("Camera permission denied");
});

function capture(){

const canvas=document.createElement("canvas");

canvas.width=video.videoWidth;
canvas.height=video.videoHeight;

const ctx=canvas.getContext("2d");

ctx.drawImage(video,0,0);

const dataURL=canvas.toDataURL("image/jpeg");

document.getElementById("selfie_data").value=dataURL;

alert("Selfie captured successfully");

}

</script>

</body>
</html>
"""

# ---------------- FACE MATCH ----------------

def compare_faces(img1,img2):

    img1=cv2.resize(img1,(200,200))
    img2=cv2.resize(img2,(200,200))

    gray1=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    gray2=cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

    score,_=ssim(gray1,gray2,full=True)

    return score*100


# ---------------- AADHAAR EXTRACTION ----------------

def extract_aadhaar(text):

    pattern = r"\d{4}\s?\d{4}\s?\d{4}"
    match = re.search(pattern,text)

    if match:

        aadhaar = match.group().replace(" ","")

        masked = "XXXX-XXXX-" + aadhaar[-4:]

        return masked

    return "XXXX-XXXX-XXXX"


# ---------------- MAIN ROUTE ----------------

@app.route("/",methods=["GET","POST"])

def home():

    if request.method=="POST":

        name=request.form["name"]
        dob=request.form["dob"]

        aadhaar_file=request.files["aadhaar"]
        selfie_data=request.form.get("selfie_data")

        if not selfie_data:
            return "<h2>Please capture selfie first</h2>"

        # Decode selfie
        header,encoded=selfie_data.split(",",1)

        selfie_bytes=base64.b64decode(encoded)

        selfie_np=np.frombuffer(selfie_bytes,np.uint8)

        selfie_img=cv2.imdecode(selfie_np,cv2.IMREAD_COLOR)

        # Decode Aadhaar
        aadhaar_np=np.frombuffer(aadhaar_file.read(),np.uint8)

        aadhaar_img=cv2.imdecode(aadhaar_np,cv2.IMREAD_COLOR)

        # OCR
        gray=cv2.cvtColor(aadhaar_img,cv2.COLOR_BGR2GRAY)

        extracted_text=pytesseract.image_to_string(gray)

        # Name score
        name_score=fuzz.partial_ratio(name.lower(),extracted_text.lower())

        # DOB score
        dob_score=fuzz.partial_ratio(dob.lower(),extracted_text.lower())

        # Face score
        face_score=compare_faces(aadhaar_img,selfie_img)

        # Aadhaar masking
        masked_aadhaar=extract_aadhaar(extracted_text)

        # Total score
        total_score=int((name_score+face_score)/2)

        # Verification decision
        if name_score>60 and dob_score>60 and face_score>15:

            status="Verified"

        else:

            status="Rejected"

        # Save in Supabase
        supabase.table("verified_users").insert({

            "full_name":name,
            "dob":dob,
            "masked_aadhaar":masked_aadhaar,
            "aadhaar_image_url":None,
            "selfie_image_url":None,
            "face_score":round(face_score,2),
            "name_score":round(name_score,2),
            "total_score":total_score,
            "kyc_status":status,
            "verified_at":datetime.utcnow().isoformat()

        }).execute()

        return f"""

        <h2>KYC {status}</h2>

        <h3>Verification Scores</h3>

        <p>Name Score: {name_score}</p>
        <p>DOB Score: {dob_score}</p>
        <p>Face Score: {round(face_score,2)}</p>

        """

    return render_template_string(HTML_PAGE)


if __name__=="__main__":
    app.run(host="0.0.0.0",port=10000)