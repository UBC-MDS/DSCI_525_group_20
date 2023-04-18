import tempfile
import boto3
import joblib
import numpy as np
import pandas as pd

from flask import Flask, request, jsonify


s3_client = boto3.client('s3')
bucket_name = "mds-s3-20-lauren"
key = "output/model.joblib"

# 1. Load your model here
with tempfile.TemporaryFile() as fp:
    s3_client.download_fileobj(Fileobj=fp, Bucket=bucket_name, Key=key)
    fp.seek(0)
    model = joblib.load(fp)
# model = joblib.load("s3://mds-s3-20-lauren/output/model.joblib")

app = Flask(__name__)


# 2. Define a prediction function
def return_prediction(data):
    data = np.array([data])
    return model.predict(data).tolist()

# 3. Set up home page using basic html
@app.route("/")
def index():
    return """
    <h1>Welcome to our rain prediction service</h1>
    <h3>We are group 20!</h3>
    To use this service, make a JSON post request to the /predict url with 25 climate model outputs.
    """

# 4. define a new route which will accept POST requests and return model predictions
@app.route('/predict', methods=['POST'])
def rainfall_prediction():
    content = request.json  # this extracts the JSON content we sent
    input_data = content['data']
    prediction = return_prediction(input_data)
    results = {
        'input': input_data,
        'output': prediction
    }
    return jsonify(results)