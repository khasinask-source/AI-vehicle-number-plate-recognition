import streamlit as st
import cv2
import numpy as np
import pytesseract
from PIL import Image

# Title
st.title("🚗 Vehicle Number Plate Detection App")

# Sidebar
st.sidebar.header("Options")
uploaded_file = st.sidebar.file_uploader("Upload Vehicle Image", type=["jpg", "png", "jpeg"])
use_sample = st.sidebar.checkbox("Use Sample Image")

# ---------------- DETECTION FUNCTION ----------------
def detect_plate(image):
    img = np.array(image)

    # Resize
    img = cv2.resize(img, (600, 400))

    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Noise reduction
    gray = cv2.bilateralFilter(gray, 11, 17, 17)

    # Edge detection
    edged = cv2.Canny(gray, 30, 200)

    # Find contours
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]

    plate = None

    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.018 * peri, True)

        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(c)

            # 🔥 FIX 1: Remove border noise (important)
            padding = 5
            plate = img[y+padding:y+h-padding, x+padding:x+w-padding]

            cv2.drawContours(img, [approx], -1, (0, 255, 0), 3)
            break

    return img, plate

# ---------------- PREPROCESSING ----------------
def preprocess_plate(plate):
    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)

    # 🔥 FIX 2: Enlarge image (OCR works better)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # 🔥 FIX 3: Light blur (remove noise)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # 🔥 FIX 4: Adaptive threshold (better than fixed)
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        15, 5
    )

    return thresh

# ---------------- OCR ----------------
def extract_text(plate):
    processed = preprocess_plate(plate)

    # 🔥 FIX 5: Correct OCR mode
    config = '--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'

    text = pytesseract.image_to_string(processed, config=config)

    # Clean text
    text = "".join(e for e in text if e.isalnum())

    return text, processed

# ---------------- IMAGE INPUT ----------------
if uploaded_file is not None:
    image = Image.open(uploaded_file)
elif use_sample:
    image = Image.open("sample.jpg")
else:
    st.info("Upload an image or select sample image")
    st.stop()

# ---------------- DISPLAY ORIGINAL ----------------
st.subheader("Original Image")
st.image(image, use_container_width=True)

# ---------------- PROCESS ----------------
processed_img, plate = detect_plate(image)

st.subheader("Detected Plate Region")
st.image(processed_img, use_container_width=True)

# ---------------- OCR OUTPUT ----------------
if plate is not None:
    st.subheader("Extracted Plate")
    st.image(plate)

    text, processed = extract_text(plate)

    st.subheader("Processed Plate for OCR")
    st.image(processed, channels="GRAY")

    st.success(f"Detected Number: {text}")

else:
    st.error("No license plate detected")
