import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [int (x) for x in request.form.values()]
    final_features = [np.array(features)]
    print(final_features, '\n\n\n')
    prediction = model.predict(final_features)

    # data = request.get_json(force=True)

    # # Create a Pandas DataFrame from the JSON data
    # df = pd.DataFrame(data, index=[0])

    # # Predict the output using your model
    # prediction = model.predict(df)

    if prediction[0] == 1:
        output = "You are more likely to have a heart disease."
    else:
        output = "You are less likely to have a heart disease."

    return render_template('index.html', prediction_txt=output)

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values())).astype(int)])
    print("\n\n" , prediction.dtype)

    output = prediction[0]
    return jsonify(output)


if __name__=='__main__':
    app.run(debug=True)
