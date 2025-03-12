import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS



app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/predict_diabetes', methods=['POST'])
def predict_diabetes():
    try:
        # Load the pre-trained model and scaler
        with open('diabetes/diabetes_model.pkl', 'rb') as model_file:
            loaded_model = pickle.load(model_file)

        with open('diabetes/diabetes_scaler.pkl', 'rb') as scaler_file:
            loaded_scaler = pickle.load(scaler_file)

        # Get data from the request
        data = request.json

        # Prepare input features in the correct order
        feature_columns = [
            'Pregnancies', 'Glucose', 'BloodPressure', 
            'SkinThickness', 'Insulin', 'BMI', 
            'DiabetesPedigreeFunction', 'Age'
        ]

        # Create DataFrame from input
        input_data = pd.DataFrame([data])

        # Scale the input features
        scaled_data = loaded_scaler.transform(input_data[feature_columns])

        # Make prediction
        prediction = loaded_model.predict(scaled_data)

        # Return prediction result
        return jsonify({
            'prediction': int(prediction[0]),
            'result_text': 'Diabetes Positive' if prediction[0] == 1 else 'Diabetes Negative'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400
    

@app.route('/predict_alzheimer', methods=['POST'])
def predict_alzheimer():
    try:
        # Load the pre-trained model and scaler
        with open('alzheimer/alzheimer_model.pkl', 'rb') as model_file:
            loaded_model = pickle.load(model_file)

        with open('alzheimer/alzheimer_scaler.pkl', 'rb') as scaler_file:
            loaded_scaler = pickle.load(scaler_file)

        columns_to_normalize = [
            'Age', 'BMI', 'AlcoholConsumption', 'PhysicalActivity', 
            'DietQuality', 'SleepQuality', 'SystolicBP', 'DiastolicBP', 
            'CholesterolTotal', 'CholesterolLDL', 'CholesterolHDL', 
            'CholesterolTriglycerides', 'MMSE', 'FunctionalAssessment', 'ADL'
        ]       

        # Get data from request
        data = request.json
        
        # Convert to DataFrame with appropriate structure
        input_df = pd.DataFrame([data])
        
        # Only normalize the columns that were normalized during training
        normalized_features = input_df[columns_to_normalize].copy()
        normalized_features = pd.DataFrame(loaded_scaler.transform(normalized_features), 
                                         columns=columns_to_normalize,
                                         index=input_df.index)
        
        # Replace the original columns with the normalized ones
        for col in columns_to_normalize:
            input_df[col] = normalized_features[col]
        
        # Make prediction
        prediction = loaded_model.predict(input_df)
        
        # Map prediction to binary result
        stage_mapping = {
            0: "Non-Demented",
            1: "Risk of Dementia"
        }
        
        result_text = stage_mapping.get(prediction[0], "Unknown")
        
        return jsonify({
            "prediction": int(prediction[0]),
            "result_text": result_text
        })
    
    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 400

if __name__ == '__main__':
    app.run()