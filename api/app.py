from flask import Flask, request, jsonify, make_response
import mlflow
import mlflow.pyfunc
import os
import pandas as pd

BASE_PATH = os.path.expandvars("$PROJECT_BASE_PATH")

# Load the model from the specified directory
model = mlflow.pyfunc.load_model(os.path.join(BASE_PATH, "api", "model_dir"))

app = Flask(__name__)

@app.route("/info", methods=["GET"])
def info():
    # Get model metadata
    metadata = model.metadata.to_dict()
    response = make_response(jsonify(metadata), 200)
    response.content_type = "application/json"
    return response

@app.route("/", methods=["GET"])
def home():
    msg = """
    Welcome to our ML service to predict Customer satisfaction\n\n

    This API has two main endpoints:\n
    1. /info: to get info about the deployed model.\n
    2. /predict: to send predict requests to our deployed model.\n
    """

    response = make_response(msg, 200)
    response.content_type = "text/plain"
    return response

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get the JSON data from the request
        input_data = request.get_json()

        # Ensure the input data is valid
        if input_data is None:
            raise ValueError("No input data provided")

        # Convert the input data into the format expected by the model
        input_df = pd.DataFrame([input_data])

        # Make the prediction using the loaded model
        prediction = model.predict(input_df)

        # Return the prediction result
        return jsonify({'result': prediction.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# This will run a local server to accept requests to the API.
if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5001))
    app.run(debug=True, host='0.0.0.0', port=port)
