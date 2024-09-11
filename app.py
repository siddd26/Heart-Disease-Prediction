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
    print(features)
    final_features = [np.array(features)]
    print(f"Input Features for the model: {final_features}")
    prediction = model.predict(final_features)

    if prediction[0] == 1:
        output = "You are more likely to have a heart disease."
    else:
        output = "You are less likely to have a heart disease."

    return render_template('index.html', prediction_txt=output)

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values())).astype(int)])
    # print("\n\n" , prediction.dtype)

    if prediction[0] == 1:
        output = "You are more likely to have a heart disease."
    else:
        output = "You are less likely to have a heart disease."
    return jsonify({'prediction': output})


if __name__=='__main__':
    app.run(host='0.0.0.0', port=5000)
    # app.run(debug=True)
