from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

# Load the model
model_path = os.path.join('model', 'titanic_survival_model.pkl')
model = joblib.load(model_path)

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction_text = ""
    status = None # 'survived' or 'not-survived' for styling

    if request.method == 'POST':
        try:
            # Get values from form
            pclass = int(request.form['Pclass'])
            sex = request.form['Sex']
            age = float(request.form['Age'])
            fare = float(request.form['Fare'])
            embarked = request.form['Embarked']

            # Create DataFrame
            input_data = pd.DataFrame([{
                'Pclass': pclass,
                'Sex': sex,
                'Age': age,
                'Fare': fare,
                'Embarked': embarked
            }])

            # Predict
            prediction = model.predict(input_data)[0]

            if prediction == 1:
                prediction_text = "SURVIVED"
                status = "survived"
            else:
                prediction_text = "DID NOT SURVIVE"
                status = "not-survived"

        except Exception as e:
            prediction_text = f"Error: {e}"
            status = "error"

    return render_template('index.html', result=prediction_text, status=status)

if __name__ == '__main__':
    app.run(debug=True)
