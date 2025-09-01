import streamlit as st
import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO
import easyocr

def load_yolo_model(model_path: str):
    """Loads the YOLO model from the given path, with error handling."""
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading YOLO model: {str(e)}")
        return None

def yolo_predict(model, image, conf: float = 0.25):
    """Runs YOLO prediction on the given image. Returns results or None on error."""
    try:
        results = model.predict(image, conf=conf)
        return results
    except Exception as e:
        st.error(f"Error during YOLO prediction: {str(e)}")
        return None

def extract_license_plate_text_easyocr(plate_img: np.ndarray, reader):
    """Extracts text from the license plate image using EasyOCR."""
    try:
        plate_rgb = cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB)
        results = reader.readtext(plate_rgb)
        # Join all detected strings for license plate
        text = " ".join([res[1] for res in results])
        return text.strip()
    except Exception as e:
        st.error(f"Error during OCR (EasyOCR): {str(e)}")
        return ""

def annotate_image_with_boxes_and_conf(image: np.ndarray, boxes, confs):
    """Draws bounding boxes and confidence scores on the image."""
    annotated_img = image.copy()
    for (box, conf) in zip(boxes, confs):
        x1, y1, x2, y2 = box
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Draw confidence score in blue
        cv2.putText(
            annotated_img,
            f"{conf*100:.2f}%",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 0, 0), 2
        )
    return annotated_img

def get_license_plate_boxes_and_confs(yolo_results):
    """Extracts bounding box coordinates and confidence scores from YOLO results."""
    boxes = []
    confs = []
    for result in yolo_results:
        for box in result.boxes:
            coords = box.xyxy[0]
            x1, y1, x2, y2 = map(int, coords)
            boxes.append((x1, y1, x2, y2))
            confs.append(float(box.conf[0]))
    return boxes, confs

# Streamlit UI
st.title('License Plate Detection and Text Recognition')

# Model loading
MODEL_PATH = 'best_license_plate_model.pt'
model = load_yolo_model(MODEL_PATH)
if not model:
    st.stop()

# EasyOCR reader initialization
try:
    reader = easyocr.Reader(['en'], gpu=False)
except Exception as e:
    st.error(f"Error initializing EasyOCR: {str(e)}")
    st.stop()

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    try:
        # Read image
        image = Image.open(uploaded_file).convert('RGB')
        img_np = np.array(image)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # YOLO detection
        yolo_results = yolo_predict(model, img_bgr, conf=0.25)
        if yolo_results is None:
            st.stop()
        
        boxes, confs = get_license_plate_boxes_and_confs(yolo_results)
        plate_texts = []

        # Extract text for each detected license plate using EasyOCR
        for (x1, y1, x2, y2) in boxes:
            plate_roi = img_bgr[y1:y2, x1:x2]
            if plate_roi.size == 0:
                plate_texts.append("No ROI found")
                continue
            text = extract_license_plate_text_easyocr(plate_roi, reader)
            plate_texts.append(text if text else "No text detected")

        # Annotate image with boxes and confidence scores (not text)
        annotated_img = annotate_image_with_boxes_and_conf(img_bgr, boxes, confs)

        # Display results
        st.image(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB),
                 caption="Detected License Plate(s)", use_container_width=True)
        st.write("Detected Plate Texts:")
        for text in plate_texts:
            st.success(text)
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")