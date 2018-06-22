import os

from stance.stance_predictor import StanceBowPredictor

BOW_VECT_PATH = '../../stance/checkpoints/2018-06-19_22:01:12/bow_vect.pkl'
TF_VECT_PATH = '../../stance/checkpoints/2018-06-19_22:01:12/tf_vect.pkl'
IDF_VECT_PATH = '../../stance/checkpoints/2018-06-19_22:01:12/idf_vect.pkl'
WEIGHTS_PATH = '../../stance/model_weights.hdf5'


class StanceIndex():

    def __init__(self):
        self.stance_clf = StanceBowPredictor(bow_vect_path=BOW_VECT_PATH,
                                             tf_vect_path=TF_VECT_PATH,
                                             idf_vect_path=IDF_VECT_PATH,
                                             weights_path=WEIGHTS_PATH)
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
            df = type('', (), {})()
            setattr(df, 'headlines', [{'Headline': fact, 'Body ID': 1}])
            setattr(df, 'id_to_body', {1: _news.body})
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
