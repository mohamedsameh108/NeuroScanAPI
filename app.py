from flask import Flask, request, jsonify
from flask_cors import CORS
from preprocessing import image_processing
from tensorflow.keras.models import load_model
import numpy as np
import os

app = Flask(__name__)
CORS(app)

NonVSVeryMild_model = load_model(os.getcwd() + '\\models\\model_NonVSVeryMild.h5')
NonVSMild_model = load_model(os.getcwd() + '\\models\\model_NonVSMild.h5')
NonVSModerate_model = load_model(os.getcwd() + '\\models\\model_NonVSModerate.h5')
VeryMildVSMild_model = load_model(os.getcwd() + '\\models\\model_VeryMildVSMild.h5')
VeryMildVSModerate_model = load_model(os.getcwd() + '\\models\\model_VeryMildVSModerate.h5')
MildVSModerate_model = load_model(os.getcwd() + '\\models\\model_MildVSModerate.h5')

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
