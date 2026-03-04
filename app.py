import os
import base64
import numpy as np
import cv2
import pytesseract
from flask import Flask, request, render_template_string
from supabase import create_client
from rapidfuzz import fuzz

app = Flask(__name__)

# Supabase setup
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
<title>KYC Verification</title>
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
alert("Camera access denied");
});

function capture(){

const canvas=document.createElement("canvas");

canvas.width=video.videoWidth;
canvas.height=video.videoHeight;

const ctx=canvas.getContext("2d");

ctx.drawImage(video,0,0);

const dataURL=canvas.toDataURL("image/jpeg");

document.getElementById("selfie_data").value=dataURL;

alert("Selfie captured");

}

</script>

</body>
</html>
"""

# Histogram face comparison
def compare_faces_histogram(img1,img2):

    img1=cv2.resize(img1,(200,200))
    img2=cv2.resize(img2,(200,200))

    img1_gray=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    img2_gray=cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

    hist1=cv2.calcHist([img1_gray],[0],None,[256],[0,256])
    hist2=cv2.calcHist([img2_gray],[0],None,[256],[0,256])

    cv2.normalize(hist1,hist1)
    cv2.normalize(hist2,hist2)

    score=cv2.compareHist(hist1,hist2,cv2.HISTCMP_CORREL)

    return score*100


@app.route("/",methods=["GET","POST"])
def home():

    if request.method=="POST":

        name=request.form["name"]
        dob=request.form["dob"]

        aadhaar_file=request.files["aadhaar"]
        selfie_data=request.form["selfie_data"]

        if selfie_data=="":
            return "<h2 style='color:red;text-align:center'>Please capture selfie</h2>"

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

        name_score=fuzz.partial_ratio(name.lower(),extracted_text.lower())

        dob_score=fuzz.partial_ratio(dob.lower(),extracted_text.lower())

        # Face comparison
        face_score=compare_faces_histogram(aadhaar_img,selfie_img)

        if name_score>60 and dob_score>60 and face_score>30:

            supabase.table("verified_users").insert({
                "name_score":name_score,
                "face_score":round(face_score,2),
                "status":"Verified"
            }).execute()

            return f"""
            <h2 style='color:lightgreen;text-align:center'>KYC Verified Successfully</h2>

            <h3>Scores</h3>

            <p>Name Score: {name_score}</p>
            <p>DOB Score: {dob_score}</p>
            <p>Face Score: {round(face_score,2)}</p>
            """

        else:

            return f"""
            <h2 style='color:red;text-align:center'>KYC Rejected</h2>

            <h3>Scores</h3>

            <p>Name Score: {name_score}</p>
            <p>DOB Score: {dob_score}</p>
            <p>Face Score: {round(face_score,2)}</p>
            """

    return render_template_string(HTML_PAGE)


if __name__=="__main__":
    app.run(host="0.0.0.0",port=10000)