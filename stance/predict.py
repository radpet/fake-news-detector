from keras.preprocessing.sequence import pad_sequences

from common.TokenizerSerializer import TokenizerSerializer
from keras.models import load_model
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

from common.attention import AttentionWithContext
from stance.bi_lstm_baseline import MAXLEN_HEADLINE, MAXLEN_BODY

import os
import numpy as np
import pickle


def load_checkpoint(path):
    return load_model(path, custom_objects={'AttentionWithContext': AttentionWithContext})


def load_data(path):
    return pd.read_csv(path)


def get_features(data, tokenizer_headline, tokenizer_body):
    X = {
        'headline': data['Headline'],
        'body': data['articleBody']
    }

    X['headline'] = tokenizer_headline.texts_to_sequences(X['headline'])
    X['headline'] = pad_sequences(X['headline'], maxlen=MAXLEN_HEADLINE)

    X['body'] = tokenizer_body.texts_to_sequences(X['body'])
    X['body'] = pad_sequences(X['body'], maxlen=MAXLEN_BODY)

    return X['headline'], X['body']


def get_labels(data, label_to_id):
    y = data['Stance']

    y = y.map(lambda label: label_to_id[label]).values
    one_hot = np.zeros((y.shape[0], len(label_to_id)))
    one_hot[np.arange(y.shape[0]), y] = 1

    return one_hot


def run():
    checkpoint_folder = './checkpoints/2018-05-13_16:54:37'
    tokenizer_headline = TokenizerSerializer.load(os.path.join(checkpoint_folder, 'tokenizer_headline.pkl'))
    tokenizer_body = TokenizerSerializer.load(os.path.join(checkpoint_folder, 'tokenizer_body.pkl'))

    with open(os.path.join(checkpoint_folder, 'label_to_id.pkl'), 'rb') as f:
        label_to_id = pickle.load(f)

    id_to_label = {i: label for label, i in label_to_id.items()}

    data = load_data(path='./data/train/split/train.csv')
    X_headline, X_body = get_features(tokenizer_body=tokenizer_body, tokenizer_headline=tokenizer_headline, data=data)
    y = get_labels(data, label_to_id)

    print(X_headline.shape)
    print(X_body.shape)
    model = load_checkpoint(os.path.join(checkpoint_folder, 'weights.14-0.74.hdf5'))
    model.summary()

    predictions = model.predict([X_headline, X_body])

    df_predictions = data.copy()
    df_predictions['stance_pred'] = np.array([id_to_label[pred] for pred in np.argmax(predictions, axis=1)]) \
        .reshape((-1, 1))

    # df_predictions.to_csv(os.path.join(checkpoint_folder, 'test_predicted.csv'), index=False)
    print('Predicted {}'.format(predictions.shape))

    print(confusion_matrix(y_true=df_predictions['Stance'], y_pred=df_predictions['stance_pred']))
    print(classification_report(y_true=df_predictions['Stance'], y_pred=df_predictions['stance_pred']))


if __name__ == '__main__':
    run()
