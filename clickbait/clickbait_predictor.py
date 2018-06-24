import pickle
import os
import numpy as np
from common.predictor import Predictor

class ClickbaitPredictor(Predictor):
    def __init__(self, tokenizer_path="model/tokenizer.pkl", svc_path="model/svc.pkl"):
        self.tokenizer = self.deserialize(tokenizer_path)
        self.model = self.deserialize(svc_path)
        pass

    def deserialize(self, path):
        with open(path, 'rb') as f:
            return pickle.load(f)

    def explain(self, X):
        pass

    def predict(self, X):
        tokenized = self.tokenizer.transform(X)
        pred = self.model.predict(tokenized)
        return pred
    
#predictor = ClickbaitPredictor("model/tokenizer.pkl", "./model/svc.pkl")
#print(predictor.predict(["Luke Cage is the first Netflix MCU show with a strong season 2", "Google Chrome on Android can now cache news articles on Wi-Fi for offline reading"]))
    