import numpy as np
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
    prediction = model.predict(final_features)

    output = "You are less likely to have a heart disease." if prediction[0]==0 else "You are more likely to have a heart disease."

    return render_template('index.html', prediction_txt=output)

if __name__=='__main__':
    app.run(debug=True)
