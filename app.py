import cv2
import numpy as np
from PIL import Image
import io
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf


app = Flask(__name__)
CORS(app)

def image_processing(img):
    image_data = img.read()
    image = Image.open(io.BytesIO(image_data))
    img = np.array(image)

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_resized = cv2.resize(img_gray, (224,224))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(7,7))
    img_clahe = clahe.apply(img_resized)
    norm_img = cv2.normalize(img_clahe, None, alpha=0.0, beta=1.0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    final_img = np.expand_dims(norm_img, axis=0)
    return final_img
    

NonVSVeryMild_model = tf.keras.models.load_model('model_NonVSVeryMild.h5')
NonVSMild_model = tf.keras.models.load_model('model_NonVSMild.h5')
NonVSModerate_model = tf.keras.models.load_model('model_NonVSModerate.h5')
VeryMildVSMild_model = tf.keras.models.load_model('model_VeryMildVSMild.h5')
VeryMildVSModerate_model = tf.keras.models.load_model('model_VeryMildVSModerate.h5')
MildVSModerate_model = tf.keras.models.load_model('model_MildVSModerate.h5')

@app.route('/api/classify', methods=['POST'])
def classify():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        image = request.files['image']

        processed_image = image_processing(image)
        prediction = [0,0,0,0]  
        NonVSVeryMild_Predict = float(NonVSVeryMild_model.predict(processed_image, verbose=0))
        NonVSMild_Predict = float(NonVSMild_model.predict(processed_image, verbose=0))
        NonVSModerate_Predict = float(NonVSModerate_model.predict(processed_image, verbose=0))
        VeryMildVSMild_Predict = float(VeryMildVSMild_model.predict(processed_image, verbose=0))
        VeryMildVSModerate_Predict = float(VeryMildVSModerate_model.predict(processed_image, verbose=0))
        MildVSModerate_Predict = float(MildVSModerate_model.predict(processed_image, verbose=0))

        confidence = {
            'NonVSVeryMild': 0.9264,
            'NonVSMild': 0.9684,
            'NonVSModerate': 0.9947,
            'VeryMildVSMild': 0.9449,
            'VeryMildVSModerate': 0.9952,
            'MildVSModerate': 0.9820
        }

        prediction[0] = ((NonVSVeryMild_Predict * confidence['NonVSVeryMild']) + (NonVSMild_Predict * confidence['NonVSMild']) + (NonVSModerate_Predict * confidence['NonVSModerate'])) / 3
        prediction[1] = (((1 - NonVSVeryMild_Predict) * confidence['NonVSVeryMild']) + (VeryMildVSMild_Predict * confidence['VeryMildVSMild']) + (VeryMildVSModerate_Predict * confidence['VeryMildVSModerate'])) / 3
        prediction[2] = (((1 - NonVSMild_Predict) * confidence['NonVSMild']) + ((1 - VeryMildVSMild_Predict) * confidence['VeryMildVSMild']) + (MildVSModerate_Predict * confidence['MildVSModerate'])) / 3
        prediction[3] = (((1 - NonVSModerate_Predict) * confidence['NonVSModerate']) + ((1 - VeryMildVSModerate_Predict) * confidence['VeryMildVSModerate']) + ((1 - MildVSModerate_Predict) * confidence['MildVSModerate'])) / 3
    
        classes = ['Non Demented', 'Very Mild Demented', 'Mild Demented', 'Moderate Demented']
        confidence = float(np.max(prediction))
        result = classes[np.argmax(prediction)]
        
        return jsonify({
        'result': result,
        'confidence': confidence
    })
        
        return jsonify(result)
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': 'Failed to process image'}), 500

if __name__ == '__main__':
    app.run(port=5000)
