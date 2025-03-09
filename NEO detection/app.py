from joblib import load
from flask import Flask, jsonify, request
from tensorflow.keras.models import load_model
import numpy as np

# Load the saved models once at the beginning
trajectory_model = load_model('trajectory_model.keras')
object_model = load('object_model_pipeline.joblib')

# Print model types to verify loading
print(type(trajectory_model))
print(hasattr(trajectory_model, 'predict'))  # Should return True

# Create Flask app
app = Flask(__name__)

@app.route('/predict_trajectory', methods=['POST'])
def predict_trajectory():
    try:
        data = request.json
        input_features = np.array(data['features']).reshape(1, 4, -1)  # Reshape to match the model input
        prediction = trajectory_model.predict(input_features)
        return jsonify({'trajectory': prediction.tolist()})
    except Exception as e:
        return jsonify({"error": f"Error predicting trajectory: {str(e)}"}), 500

@app.route('/predict_object', methods=['POST'])
def predict_object():
    try:
        # Get the input JSON
        data = request.json
        input_features = data.get('features')  # Expecting a list of feature values

        if not input_features:
            return jsonify({"error": "No features provided"}), 400

        # Predict the cluster
        cluster_id = object_model.predict([input_features])[0]

        # Map the cluster ID to an object type
        cluster_labels_mapped = {
            0: 'Other',
            1: 'Comet',
            2: 'Asteroid'  # Adjust based on your analysis
        }
        object_type = cluster_labels_mapped.get(cluster_id, 'Unknown')

        return jsonify({"cluster_id": int(cluster_id), "object_type": object_type})
    except Exception as e:
        return jsonify({"error": f"Error predicting object: {str(e)}"}), 500

if __name__ == '__main__':
    # Run the Flask app only once
    app.run(debug=True)