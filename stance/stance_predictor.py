import pickle

import numpy as np

from keras.models import load_model

from common.TokenizerSerializer import TokenizerSerializer
from common.attention import AttentionWithContext
from common.predictor import Predictor
from stance.bi_lstm_baseline import preprocess_features


class StancePredictor(Predictor):

    def __init__(self, tokenizer_headline_path, tokenizer_body_path, weights_path, label_to_id_path):
        self.tokenizer_headline_path = tokenizer_headline_path
        self.tokenizer_body_path = tokenizer_body_path
        self.weights_path = weights_path
        self.label_to_id_path = label_to_id_path
        self._load_checkpoint()

    def _load_checkpoint(self):
        self.model = load_model(self.weights_path, custom_objects={'AttentionWithContext': AttentionWithContext})
        self.tokenizer_headline = TokenizerSerializer.load(self.tokenizer_headline_path)
        self.tokenizer_body = TokenizerSerializer.load(self.tokenizer_body_path)

        with open(self.label_to_id_path, 'rb') as f:
            self.label_to_id = pickle.load(f)
            self.id_to_label = {val: key for key, val in self.label_to_id.items()}

    def predict(self, X):
        features = preprocess_features(X, self.tokenizer_headline, self.tokenizer_body)

        pred = self.model.predict([features['headline'], features['body']])
        y_pred = np.argmax(pred, axis=1)
        return {
            'label': self.id_to_label[y_pred[0]],
            'score': str(pred[0, y_pred[0]])
        }

        def explain(self, X):
            pass
