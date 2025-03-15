import pickle
import numpy as np
from flask import Flask, render_template, request
import os 
import mlflow
import mlflow.pyfunc
from algorithm import LinearRegression  

app = Flask(__name__)



# -----------------------------
# Load the first and second models from local pickle files
# -----------------------------
# First Model (Local - Regression)
first_model = pickle.load(open('./models/car_prediction.model', 'rb'))
first_scaler = pickle.load(open('./models/first_scaler.pkl', 'rb'))

# Second Model (Local - Regression)
second_model = pickle.load(open('./models/second_model.pkl', 'rb'))
second_scaler = pickle.load(open('./models/second_scaler.pkl', 'rb'))
second_poly = pickle.load(open('./models/second_poly.pkl', 'rb'))

# -----------------------------
# Load the third model from MLflow Model Registry (Classification)
# -----------------------------
mlflow.set_tracking_uri("https://admin:password@mlflow.ml.brain.cs.ait.ac.th")
mlflow.set_experiment("ST125064-a3")
staged_model_uri = "models:/ST125064-a3-model/Staging"
third_model = mlflow.pyfunc.load_model(staged_model_uri)
third_scaler = pickle.load(open('./models/third_scaler.pkl', 'rb'))

# -----------------------------
# Prediction history dictionary for each model
# -----------------------------
prediction_history = {
    "first": [],
    "second": [],
    "third": []
}

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict_first", methods=["GET", "POST"])
def predict_first():
    return predict(model=first_model, scaler=first_scaler, poly=None, history_key="first", model_name="Old Model")

@app.route("/predict_second", methods=["GET", "POST"])
def predict_second():
    return predict(model=second_model, scaler=second_scaler, poly=second_poly, history_key="second", model_name="New Model")

@app.route("/predict_third", methods=["GET", "POST"])
def predict_third():
    return predict(model=third_model, scaler=third_scaler, poly=None, history_key="third", model_name="Third Model (Deployed)")

def predict(model, scaler, poly, history_key, model_name):
    if request.method == "GET":
        return render_template("predict.html", prediction_history=prediction_history[history_key], model_type=model_name)

    # Retrieve form inputs with default values
    year = request.form.get("year", "").strip()
    max_power = request.form.get("max_power", "").strip()
    engine = request.form.get("engine", "").strip()
    owner = request.form.get("owner", "1").strip()
    fuel = request.form.get("fuel", "0").strip()
    transmission = request.form.get("transmission", "0").strip()
    action = request.form.get("action")  # e.g., "Clear"

    if action == "Clear":
        prediction_history[history_key] = []
        return render_template("predict.html", prediction_history=prediction_history[history_key], model_type=model_name)

    # Convert inputs safely
    year = int(year) if year.isdigit() else 2015
    max_power = float(max_power) if max_power.replace(".", "", 1).isdigit() else 80.0
    engine = int(engine) if engine.isdigit() else 1500
    owner = int(owner) if owner.isdigit() else 1
    fuel = int(fuel) if fuel.isdigit() else 0
    transmission = int(transmission) if transmission.isdigit() else 0

    input_data = np.array([[year, max_power, engine, owner, fuel, transmission]])
    # Scale input
    scaled_input = scaler.transform(input_data)
    # Apply polynomial transformation if provided
    if poly:
        transformed_input = poly.transform(scaled_input)
    else:
        transformed_input = scaled_input

    # For the third model (classification), convert bucket to descriptive label.
    if history_key == "third":
        predicted_class = model.predict(transformed_input)[0]
        bucket_mapping = {0: "Cheap", 1: "Affordable", 2: "Expensive", 3: "Luxury"}
        prediction_str = bucket_mapping.get(predicted_class, "Unknown")
    else:
        # For regression models, convert log price back to price.
        predicted_price_log = model.predict(transformed_input)
        predicted_price = np.exp(predicted_price_log[0])
        prediction_str = f"${predicted_price:,.2f}"

    # Prepare parameters for display
    parameters = (f"Year: {year}, Max Power: {max_power}, Engine: {engine}, "
                  f"Owner: {['First', 'Second', 'Third', 'Fourth & Above'][owner - 1]}, "
                  f"Fuel: {['Petrol', 'Diesel'][fuel]}, Transmission: {['Manual', 'Automatic'][transmission]}")
    
    # Append prediction details to history (using unified key "prediction")
    prediction_history[history_key].append({
        "prediction": prediction_str,
        "parameters": parameters
    })
    return render_template("predict.html", prediction_history=prediction_history[history_key], model_type=model_name)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
