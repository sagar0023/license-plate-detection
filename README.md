# License Plate Detection and Recognition Streamlit App

This project is a Streamlit web application for detecting license plates in images and recognizing their text. It uses a YOLO model (trained on a Kaggle dataset) for detection and EasyOCR for text extraction.

## Dataset Source

YOLO model building and dataset can be found here on my kaggle:  
**[License Plate Dataset YOLO V8 | Kaggle](https://www.kaggle.com/code/assprophet/license-plate-detection)**

## Features

- Upload an image and detect license plate(s) using YOLO.
- Extract the license plate text using EasyOCR.
- Visualize bounding boxes with confidence scores.
- Display detected text for each license plate.

---

## Step-by-Step Guide: Run Locally

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
```

### 2. Set Up Virtual Environment

```bash
python -m venv venv
```
Activate the environment:

- **Windows**
  ```bash
  venv\Scripts\activate
  ```
- **Mac/Linux**
  ```bash
  source venv/bin/activate
  ```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

If you don't have a `requirements.txt`, use:

```bash
pip install streamlit ultralytics==8.2.38 opencv-python-headless numpy pandas matplotlib Pillow easyocr dill
```

### 4. Download Trained YOLO Model

- Place your trained YOLO `.pt` file (e.g., `best_license_plate_model.pt`) in the project directory.
- You can train your own model using the Kaggle dataset above and the [Ultralytics YOLO documentation](https://docs.ultralytics.com/).

### 5. Run the Streamlit App

```bash
streamlit run app.py
```

### 6. Usage

- Open the local Streamlit URL (usually `http://localhost:8501`) in your browser.
- Upload an image containing a license plate.
- View detected bounding boxes and confidence scores on the image.
- See extracted plate text below the image.

---

## Troubleshooting

- **EasyOCR is slow or downloading weights?**  
  The first run may take a minute as EasyOCR downloads its detection model. Subsequent runs will be faster.
- **No GPU detected?**  
  The app defaults to CPU, but you can enable GPU if available.
- **YOLO model not loading?**  
  Make sure your `.pt` file is correct and matches Ultralytics YOLO version.

---

## References

- [Ultralytics YOLO Documentation](https://docs.ultralytics.com/)
- [EasyOCR Documentation](https://github.com/JaidedAI/EasyOCR)
- [Kaggle License Plate Dataset](https://www.kaggle.com/datasets/dataturks/indian-license-plates-object-detection)

---

## License

This project is provided under the MIT License.
