import os

import pandas as pd
import numpy as np
from keras.models import load_model

from common.TokenizerSerializer import TokenizerSerializer
from common.attention import AttentionWithContext
from news_aggregator.bi_gru_classificator_baseline import get_features, get_labels, LABEL_DICT
from sklearn.metrics import classification_report


def load_checkpoint(path):
    return load_model(path, custom_objects={'AttentionWithContext': AttentionWithContext})


def load_data(path):
    return pd.read_csv(path)


def run():
    checkpoint_folder = './checkpoints/2018-06-15_00:21:17'
    weights = 'weights.14-0.16.hdf5'

    model = load_checkpoint(os.path.join(checkpoint_folder, weights))

    tokenizer = TokenizerSerializer.load(os.path.join(checkpoint_folder, 'tokenizer'))

    test = load_data('./data/test.csv')
    print('Loaded {} for testing'.format(test.shape))

    test_x = get_features(test, tokenizer)
    test_y = get_labels(test)

    y_pred = model.predict(test_x)

    y_pred = np.argmax(y_pred, axis=1)

    LABEL_DICT_REV = {val: key for key, val in LABEL_DICT.items()}

    y_pred = np.array([LABEL_DICT_REV[x] for x in y_pred]).reshape((-1, 1))

    report = classification_report(y_true=test_y, y_pred=y_pred)

    with open(os.path.join(checkpoint_folder, 'classification_report_test_{}.txt'.format(weights)), 'w') as f:
        f.write(report)
    print(report)


if __name__ == '__main__':
    run()
