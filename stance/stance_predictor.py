import numpy as np

from common.predictor import Predictor
from stance.tf_dense import get_model
from stance.utils import FeatureExtractor, ID_TO_LABEL


class StanceBowPredictor(Predictor):

    def __init__(self, bow_vect_path, tf_vect_path, idf_vect_path, weights_path):
        self.fe = FeatureExtractor()
        self.fe.load_vect(bow_vect_path, tf_vect_path, idf_vect_path)
        self.weights_path = weights_path
        self._load_checkpoint()

    def _load_checkpoint(self):
        self.model = get_model()
        self.model.load_weights(self.weights_path)

    def predict(self, test):
        test_set = np.array(self.fe.transform(test))

        pred = self.model.predict(test_set)

        y_pred = np.argmax(pred, axis=1)
        return {
            'label': ID_TO_LABEL[y_pred[0]],
            'score': str(pred[0, y_pred[0]])
        }

    def explain(self, X):
        pass
