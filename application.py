from flask import Flask, request, render_template, jsonify
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, OrdinalEncoder

from src.exception import MyException
from src.logger import logging
from src.utils import save_object, evaluate_models
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)  # refer python.config file

app=application

## Route for a home page
@app.route('/') 

def index():
    return render_template('index.html')

## Test dummy data
@app.route('/test')
def test_prediction():
    try:
        # Dummy values (make sure they exist in your encoder training)
        data = CustomData(
            airline="Indigo",
            source_city="Delhi",
            departure_time="Morning",
            stops="zero",
            arrival_time="Evening",
            destination_city="Mumbai",
            flight_class="Economy",
            duration=1,
            days_left=15
        )

        pred_df = data.get_data_as_data_frame()
        print("Dummy Input DF:\n", pred_df)

        predict_pipeline = PredictPipeline()
        result = predict_pipeline.predict(pred_df)

        return jsonify({"test_predicted_price": float(result[0])})
    
    except Exception as e:
        return jsonify({"error": str(e)})

# Predict route (just print data for now)
@app.route('/predict', methods=['GET', 'POST'])

def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        # Collect data from the form
        airline = request.form.get('airline')
        source_city = request.form.get('source_city')
        departure_time = request.form.get('departure_time')
        stops = request.form.get('stops')
        arrival_time = request.form.get('arrival_time')
        destination_city = request.form.get('destination_city')
        flight_class = request.form.get('flight_class')
        duration = request.form.get('duration')
        selected_date = request.form.get('days_left')

        # Convert date -> days_left
        try:
            today = datetime.today().date()
            journey_date = datetime.strptime(selected_date, "%Y-%m-%d").date()
            days_left = (journey_date - today).days
        except Exception as e:
            days_left = 0

        logging.info("Data received..")

        data = CustomData(
            airline = airline,
            source_city = source_city,
            departure_time = departure_time,
            stops = stops,
            arrival_time = arrival_time,
            destination_city = destination_city,
            flight_class = flight_class,
            duration = duration,
            days_left = days_left
        )
        
        pred_df = data.get_data_as_data_frame()
        print(pred_df)
        
        predict_pipeline = PredictPipeline()
        result = predict_pipeline.predict(pred_df)

        print(f"predicted_price: {result}")
        logging.info(f"Price predicted: {round(result[0],2)}")

        # Return a response to frontend
        return jsonify({"predicted_price": round(float(result[0]), 2)})

if __name__=="__main__":
    app.run(host="0.0.0.0")  #remove debug=True