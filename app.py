import streamlit as st
import cv2
import numpy as np
from PIL import Image
import easyocr
from ultralytics import YOLO

# ---------------- LOAD MODELS ----------------
@st.cache_resource
def load_models():
    model = YOLO("yolov8n.pt")  # general model
    reader = easyocr.Reader(['en'])
    return model, reader

model, reader = load_models()

# ---------------- TITLE ----------------
st.title("🚗 Vehicle Number Plate Detection (YOLO + EasyOCR)")

# ---------------- SIDEBAR ----------------
uploaded_file = st.sidebar.file_uploader("Upload Vehicle Image", type=["jpg", "png", "jpeg"])
use_sample = st.sidebar.checkbox("Use Sample Image")

# ---------------- LOAD IMAGE ----------------
if uploaded_file is not None:
    image = Image.open(uploaded_file)
elif use_sample:
    image = Image.open("sample.jpg")
else:
    st.info("Upload an image or select sample image")
    st.stop()

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

        # Crop detected region
        crop = img_copy[y1:y2, x1:x2]

        # Heuristic: assume plate is wide rectangle
        h, w, _ = crop.shape
        if w > h:  
            plate_img = crop

            # Draw box
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0,255,0), 2)
            break

# ---------------- SHOW DETECTION ----------------
st.subheader("Detected Region")
st.image(img_copy, use_container_width=True)

# ---------------- OCR ----------------
if plate_img is not None:
    st.subheader("Extracted Plate")
    st.image(plate_img)

    # Convert to grayscale for OCR
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)

    # EasyOCR
    result = reader.readtext(gray)

    text = ""
    for detection in result:
        text += detection[1] + " "

    text = text.strip()

    st.success(f"Detected Number: {text}")

else:
    st.error("No license plate detected")
