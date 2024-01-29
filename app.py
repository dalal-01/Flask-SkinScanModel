from flask import Flask, request, jsonify
import os
import base64
import io
from PIL import Image
import tensorflow as tf
import cv2
import numpy as np

new_model = tf.keras.models.load_model("SkinScanModelVGG16.h5")
app = Flask(__name__)

@app.route('/api', methods=['PUT'])
def hello_world():
    try:
        data = request.get_json()

        if 'image' in data:
            base64_image = data['image']
            imgdata = base64.b64decode(base64_image)
            image = Image.open(io.BytesIO(imgdata))

            # Save the image to the server
            filename = 'received_image.png'
            image.save(filename)

            # Load and preprocess the input image
            img = image.resize((224, 224))  # Adjust size based on your model input size
            img_array = np.array(img)  # Convert PIL image to NumPy array
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0  # Normalize the image data if your model was trained with normalized data

            # Make predictions
            predictions = new_model.predict(img_array)

            # Interpret the predictions
            class_labels = ["Acne", "Eczema", "Melanoma", "Normal Skin", "Psoriasis"]
            predicted_class_index = np.argmax(predictions)
            predicted_class_label = class_labels[predicted_class_index]

            # Print the result
            print(f'Predicted class: {predicted_class_label}')
            print(f'Raw predictions: {predictions}')

            # Return the result without the "message" field
            return jsonify({'predicted_class': str(predicted_class_label), 'raw_predictions': predictions.tolist()})
        else:
            return jsonify({'error': 'No image provided in the request'}), 400

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=False)
