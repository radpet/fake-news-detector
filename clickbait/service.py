import pandas as pd
from flask import Flask, request, jsonify
from clickbait.clickbait_predictor import ClickbaitPredictor

app = Flask(__name__, static_folder='static', static_url_path='/static')

clickbait_pred = ClickbaitPredictor("model/tokenizer.pkl", "model/svc.pkl")

@app.route("/clickbait")
def get_clickbait():
    query = request.args.get('q')
    prediction = clickbait_pred.predict([query])
    return jsonify(prediction[0])

@app.route("/")
def get_home():
    return "zdr"

if __name__ == '__main__':
    app.run('0.0.0.0', 8006)