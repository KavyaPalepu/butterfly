import os
import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# === Flask App Setup ===
app = Flask(__name__)
MODEL_PATH = "vgg16_model.h5"
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# === Load the trained model ===
model = load_model(MODEL_PATH)

# ✅ Load actual class names from CSV, not folder names
df = pd.read_csv("archive/Training_set.csv")
df.rename(columns={"Filename": "filename", "Class": "label"}, inplace=True)
class_names = sorted(df["label"].unique())  # List of butterfly species

# === Routes ===

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/preview', methods=["POST"])
def preview():
    file = request.files['image']
    if not file or file.filename == '':
        return "No image selected"

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    return render_template("input.html", image_path=filepath)

@app.route('/predict', methods=["POST"])
def predict():
    image_path = request.form['image_path']

    # Preprocess the image
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    confidence = np.max(prediction) * 100
    predicted_class = class_names[class_index]

    return render_template("output.html",
                           image_path=image_path,
                           label=predicted_class,
                           confidence=confidence)

# === Start Flask App ===
if __name__ == "__main__":
    app.run(debug=True)
