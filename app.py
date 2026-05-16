from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load trained model
model = tf.keras.models.load_model("model/potato_model.h5", compile=False)



class_names = [
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy'
]

# Load test dataset (for testing button)
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "dataset/test",
    image_size=(256, 256),
    batch_size=32
)

normalization_layer = tf.keras.layers.Rescaling(1./255)
test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/train')
def train():
    return "<h2>Model already trained using train_model.py</h2>"


@app.route('/test')
def test():
    loss, acc = model.evaluate(test_ds)
    return f"<h2>Test Accuracy: {round(acc*100,2)}%</h2>"


@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    img = image.load_img(filepath, target_size=(256, 256))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = round(100 * np.max(prediction), 2)

    return render_template(
        'index.html',
        prediction=predicted_class,
        confidence=confidence
    )


if __name__ == '__main__':
    app.run(debug=True)
