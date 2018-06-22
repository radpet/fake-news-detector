import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
from keras import Input
from keras import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Embedding, Bidirectional, Concatenate, Dense, Dropout, BatchNormalization, GRU
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# set based on the dataset expl
from common.GloveEmbeddings import GloveEmbeddings
from common.attention import AttentionWithContext
from stance.utils import LABEL_TO_ID, NUM_CLASSES

NUM_WORDS_HEADLINE = 20000
MAXLEN_HEADLINE = 32

NUM_WORDS_BODY = 20000
MAXLEN_BODY = 500


def load_full():
    return pd.read_csv("data/train/train.csv")


def load_train():
    return pd.read_csv('data/train/split/train.csv')


def load_dev():
    return pd.read_csv('data/train/split/dev.csv')


def load_test():
    return pd.read_csv('data/train/split/test.csv')


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


def preprocess_features(data, tokenizer_headline, tokenizer_body):
    X = {}
    X['headline'] = data['Headline']
    X['body'] = data['articleBody']

    X['headline'] = tokenizer_headline.texts_to_sequences(X['headline'])
    X['headline'] = pad_sequences(X['headline'], maxlen=MAXLEN_HEADLINE)

    X['body'] = tokenizer_body.texts_to_sequences(X['body'])
    X['body'] = pad_sequences(X['body'], maxlen=MAXLEN_BODY)

    return X


def create_tokenizers(data):
    tokenizer_headline = Tokenizer(num_words=NUM_WORDS_HEADLINE)
    tokenizer_headline.fit_on_texts(data['Headline'])

    tokenizer_body = Tokenizer(num_words=NUM_WORDS_BODY)
    tokenizer_body.fit_on_texts(data['articleBody'])

    return tokenizer_headline, tokenizer_body


def preprocess_labels(data, label_to_id):
    # print(label_to_id, id_to_label)
    y = data['Stance']
    y = y.map(lambda label: label_to_id[label]).values
    one_hot = np.zeros((y.shape[0], len(label_to_id)))
    one_hot[np.arange(y.shape[0]), y] = 1
    # print(one_hot[:5])
    return one_hot


def map_labels_to_id():
    return LABEL_TO_ID


def encode_with_bi_lstm(embedding_headline_weights, embedding_body_weights):
    # encode the headline and the body each with bi_lstm then concat the context vectors and classify
    # (this is my own idea that just want to try ;P )

    input_headline = Input(shape=(MAXLEN_HEADLINE,), name='input_headline')
    embedding_headline = Embedding(embedding_headline_weights.shape[0], embedding_headline_weights.shape[1],
                                   weights=[embedding_headline_weights],
                                   name='embedding_headline', trainable=False)(input_headline)
    headline_context_vector = Bidirectional(GRU(10, dropout=0.2, return_sequences=True), name='bi_context_headline')(
        embedding_headline)
    headline_context_vector_att = AttentionWithContext()(headline_context_vector)

    input_body = Input(shape=(MAXLEN_BODY,), name='input_body')
    embedding_body = Embedding(embedding_body_weights.shape[0], embedding_body_weights.shape[1],
                               weights=[embedding_body_weights],
                               name='embedding_body', trainable=False)(input_body)
    body_context_vector = Bidirectional(GRU(50, dropout=0.2, return_sequences=True), name='bi_context_body')(
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
    data = load_full()
    tokenizer_headline, tokenizer_body = create_tokenizers(data)

    del data

    train = load_train()

    X_train = preprocess_features(train, tokenizer_headline, tokenizer_body)

    label_to_id = map_labels_to_id()
    y_train = preprocess_labels(train, label_to_id)

    dev = load_dev()
    X_dev = preprocess_features(dev, tokenizer_headline, tokenizer_body)
    y_dev = preprocess_labels(dev, label_to_id)

    class_weights = {
        'unrelated': 1 / 0.73131,
        'discuss': 1 / 0.17828,
        'agree': 1 / 0.0736012,
        'disagree': 1 / 0.0168094
    }

    class_weights = {label_to_id[label]: val for (label, val) in class_weights.items()}

    X_train_headline = X_train['headline']
    X_train_body = X_train['body']

    print("Train shapes", X_train_headline.shape, X_train_body.shape)

    X_dev_headline = X_dev['headline']
    X_dev_body = X_dev['body']

    print("Test shapes", X_dev_headline.shape, X_dev_body.shape)

    embedding_headline_weights = GloveEmbeddings(
        '/media/radoslav/ce763dbf-b2a6-4110-960f-2ef10c8c6bde/MachineLearning/glove.6B/glove.6B.100d.txt',
        100).load().get_embedding_matrix_for_tokenizer(tokenizer_headline)

    embedding_body_weights = GloveEmbeddings(
        '/media/radoslav/ce763dbf-b2a6-4110-960f-2ef10c8c6bde/MachineLearning/glove.6B/glove.6B.300d.txt',
        300).load().get_embedding_matrix_for_tokenizer(tokenizer_body)

    model = encode_with_bi_lstm(embedding_headline_weights=embedding_headline_weights,
                                embedding_body_weights=embedding_body_weights)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    currentTime = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    os.mkdir('checkpoints/' + currentTime)

    currentCheckointFolder = os.path.join('checkpoints', currentTime)

    save_tokenizer(os.path.join(currentCheckointFolder, 'tokenizer_headline.pkl'), tokenizer_headline)
    save_tokenizer(os.path.join(currentCheckointFolder, 'tokenizer_body.pkl'), tokenizer_body)
    save_tokenizer(os.path.join(currentCheckointFolder, 'label_to_id.pkl'), label_to_id)

    early_stopping = EarlyStopping(patience=5)
    model_checkpoint = ModelCheckpoint(
        os.path.join(currentCheckointFolder, 'weights.{epoch:02d}-{val_loss:.2f}.hdf5'), monitor='val_loss')

    model.fit([X_train_headline, X_train_body], y_train, validation_data=([X_dev_headline, X_dev_body], y_dev),
              class_weight=class_weights, batch_size=128, epochs=20, callbacks=[early_stopping, model_checkpoint])

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

    show_eval_metrics([X_train_headline, X_train_body], np.argmax(y_train, axis=1).reshape((-1, 1)), name='train')

    show_eval_metrics([X_dev_headline, X_dev_body], np.argmax(y_dev, axis=1).reshape((-1, 1)), name='dev')

    # test = load_test()
    # X_test = preprocess_features(test, tokenizer_headline, tokenizer_body)
    # y_test = preprocess_labels(test, label_to_id)
    #
    # X_test_headline = X_test['headline']
    # X_test_body = X_test['body']
    #
    # show_eval_metrics([X_test_headline, X_test_body], np.argmax(y_test, axis=1).reshape((-1, 1)), name='test')


if __name__ == '__main__':
    run()
