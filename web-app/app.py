import os
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the TensorFlow model
model = tf.keras.models.load_model('model.h5')

# Function to preprocess the image
def preprocess_image(image):
    # Preprocess the image as needed (e.g., resize, normalize, reshape)
    # Example: Resize the image to 28x28
    image = tf.image.resize(image, [28, 28])
    image = tf.reshape(image, (1, 28, 28, 3))
    return image.numpy()  # Convert to numpy array

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    try:
        # Check if a file was uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})

        file = request.files['file']

        # Check if the file has a valid filename
        if file.filename == '':
            return jsonify({'error': 'No selected file'})

        # Preprocess the image
        image = tf.io.decode_image(file.read(), channels=3)  # Update channels to 3 for color images
        preprocessed_image = preprocess_image(image)

        # Perform prediction
        prediction = model.predict(preprocessed_image)
        predicted_label = np.argmax(prediction)
        if predicted_label.tolist() == 0 :
            return jsonify({'prediction': 'Cyst'})
        elif predicted_label.tolist()== 1 :
            return jsonify({'prediction': 'Normal'})
        elif predicted_label.tolist()== 2 :
            return jsonify({'prediction': 'Stone'})
        elif predicted_label.tolist()== 3 :
            return jsonify({'prediction': 'Tumor'})
        #return jsonify({'prediction': predicted_label.tolist()})
    except Exception as e:
        print('Exception:', str(e))
        return jsonify({'error': 'An error occurred during image processing', 'details': str(e)}), 500  # HTTP status 500 for internal server error

if __name__ == '__main__':
    app.run(debug=True)
