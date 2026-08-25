from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load serialized model pipeline components
try:
    scaler = joblib.load('scaler.joblib')
    model = joblib.load('model.joblib')
except FileNotFoundError:
    print("Error: Serialized joblib files not found. Run train_and_serialize.py first.")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract inputs from request JSON payload
        data = request.get_json(force=True)
        
        # Expected format: {"features": [Experience_Years, Test_Score, Certifications_Count]}
        raw_features = np.array(data['features']).reshape(1, -1)
        
        # Scale and predict
        scaled_features = scaler.transform(raw_features)
        prediction = model.predict(scaled_features)[0]
        probability = model.predict_proba(scaled_features)[0][prediction]
        
        return jsonify({
            'status': 'success',
            'prediction': int(prediction),
            'confidence': float(probability),
            'interpretation': 'Hired' if prediction == 1 else 'Not Hired'
        })
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

if __name__ == '__main__':
    # Launch local development API server
    app.run(host='0.0.0.0', port=5000, debug=True)
