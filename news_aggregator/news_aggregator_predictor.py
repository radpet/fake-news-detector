import numpy as np
from keras.models import load_model

from common.TokenizerSerializer import TokenizerSerializer
from common.attention import AttentionWithContext
from common.predictor import Predictor
from news_aggregator.bi_gru_classificator_baseline import get_features, LABEL_DICT_REV, LABEL_DICT_FULL


class CategoryPredictor(Predictor):

    def __init__(self, tokenizer_path, weights_path):
        self.tokenizer_path = tokenizer_path
        self.weights_path = weights_path

        self._load_model()
        pass

    def _load_checkpoint(self, path):
        return load_model(path, custom_objects={'AttentionWithContext': AttentionWithContext})

    def _load_model(self):
        self.model = self._load_checkpoint(self.weights_path)
        self.tokenizer = TokenizerSerializer.load(self.tokenizer_path)

    def explain(self, X):
        pass

    def predict(self, X):
        features = get_features(X, self.tokenizer, maxlen=32)
        pred = self.model.predict(features)
        y_pred = np.argmax(pred, axis=1)
        return {
            'label': LABEL_DICT_FULL[LABEL_DICT_REV[y_pred[0]]],
            'score': str(pred[0, y_pred[0]])
        }
