import pandas as pd
from flask import Flask, request, jsonify

from stance.stance_index import StanceIndex

app = Flask(__name__, static_folder='static', static_url_path='/static')

from news_aggregator.news_aggregator_service import category_clf

stance_index = StanceIndex()


@app.route("/category")
def get_category():
    query = request.args.get('q')
    df = pd.DataFrame({'TITLE': [query]})
    prediction = category_clf.predict(df)
    att = category_clf.explain(df)
    result = {
        "text": query,
        "pred": prediction,
        "att": att
    }
    return jsonify(result)


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
