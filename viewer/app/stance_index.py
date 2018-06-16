import json
import os
import pandas as pd

from stance.stance_predictor import StancePredictor

BODY_PATH = '../../stance/checkpoints/2018-05-13_16:54:37/tokenizer_body.pkl'
HEADLINE_PATH = '../../stance/checkpoints/2018-05-13_16:54:37/tokenizer_headline.pkl'
WEIGHTS_PATH = '../../stance/checkpoints/2018-05-13_16:54:37/weights.14-0.74.hdf5'
LABEL_TO_ID = '../../stance/checkpoints/2018-05-13_16:54:37/label_to_id.pkl'


class StanceIndex():

    def __init__(self):
        self.stance_clf = StancePredictor(tokenizer_body_path=BODY_PATH,
                                          tokenizer_headline_path=HEADLINE_PATH,
                                          weights_path=WEIGHTS_PATH,
                                          label_to_id_path=LABEL_TO_ID)
        self.news = self._load()

    def _load_category(self, category):
        news = []
        for filename in os.listdir(category):
            headline = None
            body = ''
            with open(os.path.join(category, filename), 'r') as f:
                for i, line in enumerate(f):
                    if i == 0:
                        headline = line.strip()
                    else:
                        body += line.strip()
            news.append(News(headline, body, category, filename))
        return news

    def _load(self):
        return self._load_category('business') + self._load_category('science') + self._load_category('health')

    def eval(self, fact):
        results = {}
        for _news in self.news:
            df = pd.DataFrame({
                'Headline': [fact],
                'articleBody': [_news.body]
            })
            prediction = self.stance_clf.predict(df)
            result = {
                'source': _news.source,
                'stance': prediction
            }
            if prediction['label'] in results:
                results[prediction['label']].append(result)
            else:
                results[prediction['label']] = [result]
        return results


class News():
    def __init__(self, headline, body, category, source):
        self.headline = headline
        self.body = body
        self.category = category
        self.source = source
