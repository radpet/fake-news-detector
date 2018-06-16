import pandas as pd
from flask import Flask, request, jsonify

from news_aggregator.news_aggregator_predictor import CategoryPredictor
from stance_index import StanceIndex

app = Flask(__name__, static_folder='static', static_url_path='/static')

category_TOKENIZER = '../../news_aggregator/checkpoints/2018-06-16_09:25:03/tokenizer'
category_MODEL = '../../news_aggregator/checkpoints/2018-06-16_09:25:03/weights.11-0.16.hdf5'

category_clf = CategoryPredictor(tokenizer_path=category_TOKENIZER,
                                 weights_path=category_MODEL)

stance_index = StanceIndex()


@app.route("/category")
def get_category():
    query = request.args.get('q')
    df = pd.DataFrame({'TITLE': [query]})
    prediction = category_clf.predict(df)
    return jsonify(prediction)


@app.route("/stance")
def get_stance():
    query = request.args.get('q')
    result = stance_index.eval(query)
    return jsonify(result)


@app.route('/')
def root():
    return app.send_static_file('index.html')


if __name__ == '__main__':
    app.run('0.0.0.0', 8005)
