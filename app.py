from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import os

app = Flask(__name__)

# ─────────────────────────────────────────────
#  Load all trained models
# ─────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_model(filename):
    path = os.path.join(BASE_DIR, filename)
    with open(path, 'rb') as f:
        return pickle.load(f)

heart_model   = load_model('heart_model.pkl')
kidney_model  = load_model('Kidney_disease_model.pkl')
liver_model   = load_model('liver_disease_model.pkl')
lung_model    = load_model('lung_cancer_model.pkl')
stroke_model  = load_model('stroke_model.pkl')


# ─────────────────────────────────────────────
#  Page Routes
# ─────────────────────────────────────────────

@app.route('/')
def home():
    return render_template('index__home.html')

@app.route('/heart')
def heart():
    return render_template('index_heart.html')

@app.route('/kidney')
def kidney():
    return render_template('index_kidney.html')

@app.route('/liver')
def liver():
    return render_template('index_liver.html')

@app.route('/lung')
def lung():
    return render_template('index_lung.html')

@app.route('/brain')
def brain():
    return render_template('index_brain.html')


# ─────────────────────────────────────────────
#  Helper: build prediction response
# ─────────────────────────────────────────────

def make_prediction(model, features):
    """
    Run model prediction and return a JSON-serialisable dict with:
      - prediction : 0 or 1
      - confidence : percentage string (e.g. "87.00")
    """
    features_array = np.array([features], dtype=float)
    prediction     = int(model.predict(features_array)[0])
    confidence     = round(float(model.predict_proba(features_array)[0][prediction]) * 100, 2)
    return {"prediction": prediction, "confidence": confidence}



@app.route('/predict_heart', methods=['POST'])
def predict_heart():
    try:
        data = request.get_json()
        features = [
            float(data['age']),
            float(data['sex']),
            float(data['cp']),
            float(data['trestbps']),
            float(data['chol']),
            float(data['fbs']),
            float(data['restecg']),
            float(data['thalach']),
            float(data['exang']),
            float(data['oldpeak']),
            float(data['slope']),
            float(data['ca']),
            float(data['thal']),
        ]
        result = make_prediction(heart_model, features)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400



@app.route('/predict_kidney', methods=['POST'])
def predict_kidney():
    try:
        data = request.get_json()
        features = [
            float(data['Age']),
            float(data['Gender']),
            float(data['Water_Intake']),
            float(data['Urination_Frequency']),
            float(data['Physical_Activity']),
            float(data['Sitting_Hours']),
            float(data['Salt_Intake_Level']),
            float(data['Protein_Intake']),
            float(data['Sleep_Hours']),
            float(data['Smoking']),
            float(data['Alcohol']),
            float(data['BP_Level']),
            float(data['Diabetes']),
        ]
        result = make_prediction(kidney_model, features)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400



@app.route('/predict_liver', methods=['POST'])
def predict_liver():
    try:
        data = request.get_json()
        features = [
            float(data['Age']),
            float(data['Gender']),
            float(data['BMI']),
            float(data['AlcoholConsumption']),
            float(data['Smoking']),
            float(data['GeneticRisk']),
            float(data['PhysicalActivity']),
            float(data['Diabetes']),
            float(data['Hypertension']),
            float(data['LiverFunctionTest']),
        ]
        result = make_prediction(liver_model, features)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400




@app.route('/predict_lung', methods=['POST'])
def predict_lung():
    try:
        data = request.get_json()
        features = [
            float(data['GENDER']),
            float(data['AGE']),
            float(data['SMOKING']),
            float(data['YELLOW_FINGERS']),
            float(data['ANXIETY']),
            float(data['PEER_PRESSURE']),
            float(data['CHRONIC DISEASE']),
            float(data['FATIGUE ']),         
            float(data['ALLERGY ']),           
            float(data['WHEEZING']),
            float(data['ALCOHOL CONSUMING']),
            float(data['COUGHING']),
            float(data['SHORTNESS OF BREATH']),
            float(data['SWALLOWING DIFFICULTY']),
            float(data['CHEST PAIN']),
        ]
        result = make_prediction(lung_model, features)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400



@app.route('/predict_stroke', methods=['POST'])
def predict_stroke():
    try:
        data = request.get_json()
        features = [
            float(data['gender']),
            float(data['age']),
            float(data['hypertension']),
            float(data['heart_disease']),
            float(data['ever_married']),
            float(data['work_type']),
            float(data['Residence_type']),      
            float(data['avg_glucose_level']),
            float(data['bmi']),
            float(data['smoking_status']),
        ]
        result = make_prediction(stroke_model, features)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400


# ─────────────────────────────────────────────
#  Run
# ─────────────────────────────────────────────

if __name__ == '__main__':
    app.run(debug=True)