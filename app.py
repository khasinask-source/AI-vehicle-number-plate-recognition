import streamlit as st
import cv2
import numpy as np
from PIL import Image
import easyocr
from ultralytics import YOLO
import re

# ---------------- LOAD MODELS ----------------
@st.cache_resource
def load_models():
    model = YOLO("yolov8n.pt")
    reader = easyocr.Reader(['en'])
    return model, reader

model, reader = load_models()

# ---------------- TITLE ----------------
st.title("🚗 Vehicle Number Plate Detection (YOLO + EasyOCR)")

# ---------------- LOAD DEFAULT IMAGE ----------------
image = Image.open("sample.jpg")

# Convert to OpenCV
img = np.array(image)
img_copy = img.copy()

# ---------------- DISPLAY ORIGINAL ----------------
st.subheader("Original Image")
st.image(image, use_container_width=True)

# ---------------- YOLO DETECTION ----------------
results = model(img)

plate_img = None

for result in results:
    boxes = result.boxes.xyxy.cpu().numpy()

    for box in boxes:
        x1, y1, x2, y2 = map(int, box)

        crop = img_copy[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        h, w, _ = crop.shape

        # Focus bottom-center region (plate area)
        if w > h:
            # Focus tighter on plate area
        plate_img = crop[int(h*0.65):int(h*0.9), int(w*0.25):int(w*0.75)]

            cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
            break

# ---------------- SHOW DETECTION ----------------
st.subheader("Detected Region")
st.image(img_copy, use_container_width=True)

# ---------------- OCR ----------------
if plate_img is not None:
    st.subheader("Extracted Plate")
    st.image(plate_img)

    # Preprocess
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2)

    # OCR
    result = reader.readtext(gray)

candidates = []

for detection in result:
    candidate = detection[1]
    
    # Keep only alphanumeric
    candidate = re.sub(r'[^A-Z0-9]', '', candidate.upper())
    
    if len(candidate) >= 4:
        candidates.append(candidate)

# Join all parts
text = " ".join(candidates)

    # Clean text
    text = re.sub(r'[^A-Z0-9 ]', '', text)
    text = text.strip()

    # Extra cleanup (remove duplicates / noise words)
    words = text.split()
    final_text = ""

    for w in words:
        if any(char.isdigit() for char in w):
            final_text = w
            break

    st.success(f"Detected Number: {final_text}")

else:
    st.error("No license plate detected")
