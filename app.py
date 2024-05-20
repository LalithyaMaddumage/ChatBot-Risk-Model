from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import pickle

app = Flask(__name__)

# Load the trained model
model = load_model('my_model.keras')

# Load the tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Define the mapping from predicted class index to risk level label
risk_level_mapping = {0: "High Risk", 1: "Medium Risk", 2: "Low Risk", 3: "No Risk"}

# Placeholder for the actual maxlen used during your model's training
maxlen = 100  # Replace with your actual value

@app.route('/')
def read_root():
    return jsonify({"message": "Welcome to the risk level prediction API"})

@app.route('/predict/', methods=['POST'])
def predict_risk_level():
    data = request.get_json()
    if 'text' not in data:
        return jsonify({"error": "No text provided"}), 400

    # Preprocess text
    preprocessed_text = data['text']  # Placeholder, replace with actual preprocessing if needed

    # Convert text to sequence
    sequence = tokenizer.texts_to_sequences([preprocessed_text])
    # Pad the sequence
    padded_sequence = pad_sequences(sequence, maxlen=maxlen)
    
    # Predict
    prediction_prob = model.predict(padded_sequence)
    predicted_class = np.argmax(prediction_prob)
    
    # Map the predicted class to its corresponding label
    predicted_label = risk_level_mapping.get(predicted_class, "Unknown Risk Level")

    return jsonify({
        "Predicted_Risk_Level": predicted_label,
        "Prediction_Probabilities": prediction_prob.tolist()
    })

if __name__ == '__main__':
    app.run(debug=True)
