# simple idea with trick based on https://arxiv.org/pdf/1707.03264.pdf
import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
from keras import Input, Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Concatenate, Dense, Dropout
from numpy.linalg import norm
from numpy.ma import dot
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.metrics import confusion_matrix, classification_report

from stance.bi_lstm_baseline import preprocess_labels, LABELS_DICT

VECT_DIM = 5000
NUM_CLASSES = 4


def create_count_tokenizers(train):
    corpus = train['Headline'].append(train['articleBody'])
    count_tokenizer = CountVectorizer(max_features=VECT_DIM, stop_words='english').fit(corpus)
    return count_tokenizer


def create_idf_tokenizers(train, test):
    corpus = train['Headline'].append(test['Headline']).append(train['articleBody']).append(test['articleBody'])
    idf_tokenizer = TfidfVectorizer(max_features=VECT_DIM, stop_words='english').fit(corpus)

    return idf_tokenizer


def get_model():
    input_headline = Input(shape=(VECT_DIM,))
    input_body = Input(shape=(VECT_DIM,))

    input_cosine = Input(shape=(1,))

    input = Concatenate()([input_headline, input_cosine, input_body])

    dense = Dense(100, activation='relu')(input)
    dropout = Dropout(0.3)(dense)
    output = Dense(NUM_CLASSES, activation='softmax')(dropout)

    model = Model(inputs=[input_headline, input_cosine, input_body], outputs=[output])

    return model


def get_features(train, c_tokenizer):
    headline_vectorized = c_tokenizer.transform(train['Headline'])
    body_vectorized = c_tokenizer.transform(train['articleBody'])
    return TfidfTransformer(use_idf=False).fit_transform(headline_vectorized).toarray(), TfidfTransformer(
        use_idf=False).fit_transform(body_vectorized).toarray()


def get_cos_sim(train, i_tokenizer):
    headline_vectorized = i_tokenizer.transform(train['Headline']).toarray()
    body_vectorized = i_tokenizer.transform(train['articleBody']).toarray()
    cos = []
    for i in range(train.shape[0]):
        cos.append(dot(headline_vectorized[i], body_vectorized[i]) / (
                norm(headline_vectorized[i]) * norm(body_vectorized[i])))
    return np.array(cos).reshape((-1, 1))


def train():
    currentTime = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    os.mkdir('checkpoints/' + currentTime)

    currentCheckointFolder = os.path.join('checkpoints', currentTime)

    data_train = pd.read_csv('data/train/train.csv')
    c_tokenizer = create_count_tokenizers(data_train)

    with open(os.path.join(currentCheckointFolder, 'c_tokenizer.pkl'), 'wb') as f:
        pickle.dump(c_tokenizer, f)

    data_test = pd.read_csv('data/test/test.csv')
    i_tokenizer = create_idf_tokenizers(data_train, data_test)

    with open(os.path.join(currentCheckointFolder, 'i_tokenizer.pkl'), 'wb') as f:
        pickle.dump(i_tokenizer, f)

    train_split = pd.read_csv('data/train/split/train.csv')
    y_train = preprocess_labels(train_split, LABELS_DICT)

    vect_headline_train, vect_body_train = get_features(train_split, c_tokenizer)
    cosine_sim_train = get_cos_sim(train_split, i_tokenizer)

    dev_split = pd.read_csv('data/train/split/dev.csv')
    y_dev = preprocess_labels(dev_split, LABELS_DICT)

    vect_headline_dev, vect_body_dev = get_features(dev_split, c_tokenizer)
    cosine_sim_dev = get_cos_sim(dev_split, i_tokenizer)

    model = get_model()

    model.summary()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    class_weights = {
        'unrelated': 1 / 0.73131,
        'discuss': 1 / 0.17828,
        'agree': 1 / 0.0736012,
        'disagree': 1 / 0.0168094
    }

    class_weights = {LABELS_DICT[label]: val for (label, val) in class_weights.items()}

    early_stopping = EarlyStopping(patience=5)
    model_checkpoint = ModelCheckpoint(
        os.path.join(currentCheckointFolder, 'weights.{epoch:02d}-{val_loss:.2f}.hdf5'), monitor='val_loss')

    model.fit([vect_headline_train, cosine_sim_train, vect_body_train], y_train,
              validation_data=([vect_headline_dev, cosine_sim_dev, vect_body_dev], y_dev),
              class_weight=class_weights, batch_size=100, epochs=40, callbacks=[early_stopping, model_checkpoint])

    def show_eval_metrics(X, y_true, name='dev'):
        preds = model.predict(X)
        preds_y = np.argmax(preds, axis=1).reshape((-1, 1))

        conf_matrix = confusion_matrix(y_true=y_true, y_pred=preds_y)
        print(conf_matrix)

        with open(os.path.join(currentCheckointFolder, 'conf_matrix_{}.txt'.format(name)), 'w') as f:
            f.write(str(conf_matrix))

        report = classification_report(y_true=y_true, y_pred=preds_y)

        with open(os.path.join(currentCheckointFolder, 'classification_report_{}.txt'.format(name)), 'w') as f:
            f.write(str(report))

        print(report)

    show_eval_metrics([vect_headline_train, cosine_sim_train, vect_body_train],
                      np.argmax(y_train, axis=1).reshape((-1, 1)), name='train')

    show_eval_metrics([vect_headline_dev, cosine_sim_dev, vect_body_dev], np.argmax(y_dev, axis=1).reshape((-1, 1)),
                      name='dev')


if __name__ == '__main__':
    train()
