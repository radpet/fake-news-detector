import numpy as np
from keras import Model
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

        att_layer_mod = AttentionWithContext(return_att=True)
        att_layer_mod.set_weights(self.model.layers[3].weights)
        att_layer_mod(self.model.layers[2].output)
        self.att_layer = Model(inputs=self.model.input,
                               outputs=att_layer_mod.output)

        self.reverse_word_map = dict(map(reversed, self.tokenizer.word_index.items()))
        self.reverse_word_map[0] = '<PAD>'

    def _load_checkpoint(self, path):
        return load_model(path, custom_objects={'AttentionWithContext': AttentionWithContext})

    def _load_model(self):
        self.model = self._load_checkpoint(self.weights_path)
        self.model.summary()
        self.tokenizer = TokenizerSerializer.load(self.tokenizer_path)

    def explain(self, X):
        features = get_features(X, self.tokenizer, maxlen=16)
        att_scores, att_cv = self.att_layer.predict(features)
        att_scores = np.reshape(att_scores, [-1, 16])

        input_text = [self.reverse_word_map[s] for s in features[0]]

        return {
            'text': input_text,
            'att_score': att_scores[0].tolist(),
        }

    def predict(self, X):
        features = get_features(X, self.tokenizer, maxlen=16)
        pred = self.model.predict(features)
        y_pred = np.argmax(pred, axis=1)
        return {
            'label': LABEL_DICT_FULL[LABEL_DICT_REV[y_pred[0]]],
            'score': str(pred[0, y_pred[0]])
        }
