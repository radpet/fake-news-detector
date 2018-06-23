import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
from keras import Input
from keras import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Embedding, Bidirectional, Concatenate, Dense, Dropout, BatchNormalization, GRU, CuDNNGRU
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# set based on the dataset expl
from common.GloveEmbeddings import GloveEmbeddings
from common.attention import AttentionWithContext
from stance.utils import LABEL_TO_ID, NUM_CLASSES, Loader, FeatureExtractor2, f_scorer

NUM_WORDS_HEADLINE = 20000
MAXLEN_HEADLINE = 50

NUM_WORDS_BODY = 20000
MAXLEN_BODY = 300

NUM_WORDS = 50000


def get_pretrained_embeddings(path, tokenizer, embedding_dim=200):
    embeddings_index = {}
    with open(os.path.join(path, 'glove.6B.{}d.txt'.format(embedding_dim)), encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    print('Found %s word vectors.' % len(embeddings_index))
    word_index = tokenizer.word_index
    print('Found %s words in the tokenizer index' % len(word_index))
    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


def encode_with_bi_lstm(embedding_headline_weights, embedding_body_weights):
    # encode the headline and the body each with bi_lstm then concat the context vectors and classify
    # (this is my own idea that just want to try ;P )

    input_headline = Input(shape=(MAXLEN_HEADLINE,), name='input_headline')
    embedding_headline = Embedding(embedding_headline_weights.shape[0], embedding_headline_weights.shape[1],
                                   weights=[embedding_headline_weights],
                                   name='embedding_headline', trainable=False)(input_headline)
    headline_context_vector = Bidirectional(CuDNNGRU(50, return_sequences=True), name='bi_context_headline')(
        embedding_headline)
    headline_context_vector_att = AttentionWithContext()(headline_context_vector)

    input_body = Input(shape=(MAXLEN_BODY,), name='input_body')
    embedding_body = Embedding(embedding_body_weights.shape[0], embedding_body_weights.shape[1],
                               weights=[embedding_body_weights],
                               name='embedding_body', trainable=False)(input_body)
    body_context_vector = Bidirectional(CuDNNGRU(100, return_sequences=True), name='bi_context_body')(
        embedding_body)
    body_context_vector_att = AttentionWithContext()(body_context_vector)

    concat = Concatenate()([headline_context_vector_att, body_context_vector_att])
    out = Dense(32, activation='relu', name='dense1')(concat)
    out = BatchNormalization()(out)
    out = Dropout(0.4)(out)
    out = Dense(8, activation='relu', name='dense2')(out)
    out = Dense(NUM_CLASSES, activation='softmax')(out)

    model = Model(inputs=[input_headline, input_body], outputs=out)

    model.summary()
    return model


def save_tokenizer(name, tokenizer):
    with open(name, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)


def run():
    class_weights = {
        'unrelated': 1 / 0.73131,
        'discuss': 1 / 0.17828,
        'agree': 1 / 0.0736012,
        'disagree': 1 / 0.0168094
    }

    class_weights = {LABEL_TO_ID[label]: val for (label, val) in class_weights.items()}

    train = Loader().load_dataset('./data/train/train_stances.csv', './data/train/train_bodies.csv')
    test = Loader().load_dataset('./data/test/test_stances.csv', './data/test/test_bodies.csv')

    fe = FeatureExtractor2(headline_len=MAXLEN_HEADLINE, body_len=MAXLEN_BODY,
                           max_words=NUM_WORDS).fit(train, test)

    train_headlines, train_bodies, train_stances = fe.transform(train, labels=True)

    test_headlines, test_bodies, test_stances = fe.transform(test, labels=True)

    embedding_headline_weights = GloveEmbeddings(
        '/media/radoslav/ce763dbf-b2a6-4110-960f-2ef10c8c6bde/MachineLearning/glove.6B/glove.6B.100d.txt',
        100).load().get_embedding_matrix_for_tokenizer(fe.tokenizer)

    embedding_body_weights = GloveEmbeddings(
        '/media/radoslav/ce763dbf-b2a6-4110-960f-2ef10c8c6bde/MachineLearning/glove.6B/glove.6B.300d.txt',
        300).load().get_embedding_matrix_for_tokenizer(fe.tokenizer)

    model = encode_with_bi_lstm(embedding_headline_weights=embedding_headline_weights,
                                embedding_body_weights=embedding_body_weights)

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    currentTime = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    os.mkdir('checkpoints/' + currentTime)

    currentCheckointFolder = os.path.join('checkpoints', currentTime)

    early_stopping = EarlyStopping(patience=5)
    model_checkpoint = ModelCheckpoint(
        os.path.join(currentCheckointFolder, 'weights.{epoch:02d}-{val_loss:.2f}.hdf5'), monitor='val_loss')

    model.fit([train_headlines, train_bodies], train_stances, class_weight=class_weights,
              batch_size=64, epochs=20)

    fe.save(currentCheckointFolder)

    def show_eval_metrics(X, y_true, name='dev'):
        preds = model.predict(X)
        preds_y = np.argmax(preds, axis=1)

        conf_matrix = confusion_matrix(y_true=y_true, y_pred=preds_y)
        print(conf_matrix)

        with open(os.path.join(currentCheckointFolder, 'conf_matrix_{}.txt'.format(name)), 'w') as f:
            f.write(str(conf_matrix))

        report = classification_report(y_true=y_true, y_pred=preds_y)

        with open(os.path.join(currentCheckointFolder, 'classification_report_{}.txt'.format(name)), 'w') as f:
            f.write(str(report))

        print(report)

        print(f_scorer(y_pred=preds_y, y_true=y_true))

    show_eval_metrics([train_headlines, train_bodies], train_stances, name='train')

    show_eval_metrics([test_headlines, test_bodies], test_stances, name='dev')


if __name__ == '__main__':
    run()
