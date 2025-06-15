import os
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)

# Path to the model and upload folder
MODEL_PATH = 'covid19_xray_model.h5'
UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the model
model = load_model(MODEL_PATH)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file uploaded!', 400

    file = request.files['file']
    if file.filename == '':
        return 'No file selected!', 400

    # Save the uploaded image
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    # Preprocess the image
    img = image.load_img(filepath, target_size=(256, 256))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Predict
    prediction = model.predict(img_array)[0][0]
    label = "Positive for COVID-19" if prediction > 0.5 else "Negative for COVID-19"
    confidence = f"{prediction * 100:.2f}%" if prediction > 0.5 else f"{(1 - prediction) * 100:.2f}%"

    return render_template('result.html', label=label, confidence=confidence, image=file.filename)

if __name__ == '__main__':
    app.run(debug=True)
