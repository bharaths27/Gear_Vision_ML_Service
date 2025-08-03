# ml-service/app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

app = Flask(__name__)

# UPDATED: Specific CORS configuration for your deployed frontend
# This tells your ML service to only accept requests from your portfolio website.
CORS(app, resources={r"/predict": {"origins": "https://bharaths27.github.io"}})

model = joblib.load('performance_model.pkl')

car_base_stats = {
    'Audi_A4': {'hp': 261, 'zero_to_60': 5.2}, 'Audi_A7': {'hp': 335, 'zero_to_60': 5.2},
    'Audi_R8': {'hp': 602, 'zero_to_60': 3.1}, 'Audi_RS5': {'hp': 444, 'zero_to_60': 3.7},
    'Audi_RS6': {'hp': 591, 'zero_to_60': 3.5},
}

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    model_name = data.get('modelName')
    mod_cost = data.get('mod_cost')
    mod_hp_gain = data.get('mod_hp_gain')

    if not all([model_name, mod_cost is not None, mod_hp_gain is not None]):
        return jsonify({'error': 'Missing required fields'}), 400

    base_stats = car_base_stats.get(model_name, {'hp': 400, 'zero_to_60': 4.5})

    input_data = pd.DataFrame([[
        base_stats['hp'], base_stats['zero_to_60'], mod_cost, mod_hp_gain
    ]], columns=['base_hp', 'base_0_to_60', 'mod_cost', 'mod_hp_gain'])

    prediction = model.predict(input_data)
    
    response = {
        'predicted_0_to_60': round(prediction[0], 2),
        'total_cost': mod_cost,
        'new_hp': base_stats['hp'] + mod_hp_gain
    }
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True, port=5002)