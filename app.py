import os
import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# === Flask App Setup ===
app = Flask(__name__)
MODEL_PATH = "vgg16_model.h5"  # name is fine, even if it's MobileNetV2
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# === Load Trained Model ===
model = load_model(MODEL_PATH)

# === Load class labels from CSV ===
df = pd.read_csv("archive/Training_set.csv")
df.rename(columns={"Filename": "filename", "Class": "label"}, inplace=True)
class_names = sorted(df["label"].unique())

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

    # ✅ Resize image to (160, 160) for MobileNetV2
    img = image.load_img(image_path, target_size=(160, 160))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict class probabilities
    prediction = model.predict(img_array)[0]  # shape: (75,)

    # Top-3 predictions
    top_indices = prediction.argsort()[-3:][::-1]
    top_labels = [class_names[i] for i in top_indices]
    top_confidences = [prediction[i] * 100 for i in top_indices]
    top_preds = list(zip(top_labels, top_confidences))

    return render_template("output.html",
                           image_path=image_path,
                           top_preds=top_preds)

# === Run the App ===
if __name__ == "__main__":
    app.run(debug=True)
