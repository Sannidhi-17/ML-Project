import pickle
from flask import Flask, request, render_template
import numpy as np
import pandas as pd
# import logging
from src.logger import logging
from sklearn.preprocessing import StandardScaler
from pipeline.predict_pipeline import CustomData, PredictPipeline

application=Flask(__name__)
app = application

## route for a home page
@app.route("/")
def index():
    return render_template("index.html")


@app.route('/predict_data', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'POST':
        logging.info(f"Form data: {request.form}")
        
        data = CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('reading_score')),
            writing_score=float(request.form.get('writing_score')),
        )


        logging.info("######")
        logging.info(request.form.get('gender'), request.form.get('race_ethnicity'), request.form.get('parental_level_of_education'), request.form.get('lunch'))
        # logging.info(test_preparation_course, reading_score, writing_score)

        pred_df = data.get_data_as_data_frame()
        logging.info(f"Dataframe: {pred_df}")  # Log the DataFrame
        predict_pipeline = PredictPipeline() 
        try:
            results = predict_pipeline.predict(pred_df)
            return render_template("home.html", results=results[0])
        except Exception as e:
            logging.error(f"Prediction failed: {str(e)}")
            return render_template("home.html", error=str(e))

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=True)   

