from flask import Flask, request, jsonify
import pickle
import pandas as pd
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


# Load the pre-trained model
with open('loan_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Define a function for data preprocessing
def preprocess_data(data):
    # Convert the JSON data into a DataFrame
    df = pd.DataFrame([data])

    # Apply the same data preprocessing steps as in your script
    code_numeric = {'Male': 1, 'Female': 2, 'Yes': 1, 'No': 2,
                    'Graduate': 1, 'Not Graduate': 2, 'Urban': 3,
                    'Semiurban': 2, 'Rural': 1, 'Y': 1, 'N': 0, '3+': 3}

    df = df.applymap(lambda s: code_numeric.get(s) if s in code_numeric else s)

    # Additional preprocessing steps can be added here if needed

    return df

@app.route('/', methods=['POST'])
def predict():
    try:
        
        # Get JSON data from the request
        data = request.json

        # Perform data preprocessing on 'data'
        preprocessed_data = preprocess_data(data)

        # Make predictions using the pre-trained model
        prediction = model.predict(preprocessed_data)


        # Return the prediction as a JSON response
        response = {'prediction': int(prediction[0])}
        if prediction[0] == 1:
            result = "Approved"
        else:
            result = "Not Approved"

        return jsonify({"result": result})
        

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
