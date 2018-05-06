from datetime import datetime

import pandas as pd
import numpy as np
from keras import Input
from keras import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Embedding, Bidirectional, LSTM, Concatenate, Dense, Dropout
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import os
import pickle

# set based on the dataset expl
NUM_WORDS_HEADLINE = 10000
MAXLEN_HEADLINE = 32

NUM_WORDS_BODY = 10000
MAXLEN_BODY = 360

NUM_CLASSES = 4


def load():
    return pd.read_csv("data/train/train.csv")


def get_pretrained_embeddings(path, tokenizer):
    EMBEDDING_DIM = 200
    embeddings_index = {}
    with open(os.path.join(path, 'glove.6B.200d.txt'),encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    print('Found %s word vectors.' % len(embeddings_index))
    word_index = tokenizer.word_index
    embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


def preprocess_features(data):
    X = {}
    X['headline'] = data['Headline']
    X['body'] = data['articleBody']

    tokenizer_headline = Tokenizer(num_words=NUM_WORDS_HEADLINE)
    tokenizer_headline.fit_on_texts(X['headline'])  # todo fit on a bigger corpus
    X['headline'] = tokenizer_headline.texts_to_sequences(X['headline'])
    X['headline'] = pad_sequences(X['headline'], maxlen=MAXLEN_HEADLINE)

    tokenizer_body = Tokenizer(num_words=NUM_WORDS_BODY)
    tokenizer_body.fit_on_texts(X['body'])
    X['body'] = tokenizer_body.texts_to_sequences(X['body'])
    X['body'] = pad_sequences(X['body'], maxlen=MAXLEN_BODY)

    return X, tokenizer_headline, tokenizer_body


def preprocess_labels(data):
    y = data['Stance']
    labels = list(set(y))
    label_to_id = {x: i for i, x in enumerate(labels)}
    id_to_label = {i: x for i, x in enumerate(labels)}

    # print(label_to_id, id_to_label)

    y = y.map(lambda label: label_to_id[label]).values
    one_hot = np.zeros((y.shape[0], len(labels)))
    one_hot[np.arange(y.shape[0]), y] = 1
    # print(one_hot[:5])
    return one_hot, label_to_id, id_to_label


def encode_with_bi_lstm(embedding_headline_weights, embedding_body_weights):
    # encode the headline and the body each with bi_lstm then concat the context vectors and classify
    # (this is my own idea that just want to try ;P )

    input_headline = Input(shape=(MAXLEN_HEADLINE,), name='input_headline')
    embedding_headline = Embedding(embedding_headline_weights.shape[0], embedding_headline_weights.shape[1],
                                   weights=[embedding_headline_weights],
                                   name='embedding_headline', trainable=False)(input_headline)
    headline_context_vector = Bidirectional(LSTM(100), name='bi_context_headline')(embedding_headline)

    input_body = Input(shape=(MAXLEN_BODY,), name='input_body')
    embedding_body = Embedding(embedding_body_weights.shape[0], embedding_body_weights.shape[1],
                               weights=[embedding_body_weights],
                               name='embedding_body', trainable=False)(input_body)
    body_context_vector = Bidirectional(LSTM(100), name='bi_context_body')(embedding_body)

    concat = Concatenate()([headline_context_vector, body_context_vector])
    out = Dense(64, activation='relu', name='dense1')(concat)
    out = Dropout(0.4)(out)
    out = Dense(32, activation='relu', name='dense2')(out)
    out = Dropout(0.3)(out)
    out = Dense(NUM_CLASSES, activation='softmax')(out)

    model = Model(inputs=[input_headline, input_body], outputs=out)

    model.summary()
    return model


def save_tokenizer(name, tokenizer):
    with open(name, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)


def run():
    data = load()
    # todo save the tokenizers when doing the final training
    X, tokenizer_headline, tokenizer_body = preprocess_features(data)
    y, label_to_id, id_to_label = preprocess_labels(data)

    class_weights = {
        'unrelated': 1 / 0.73131,
        'discuss': 1 / 0.17828,
        'agree': 1 / 0.0736012,
        'disagree': 1 / 0.0168094
    }

    class_weights = {label_to_id[label]: val for (label, val) in class_weights.items()}
    # print(label_to_id)
    # print(class_weights)
    zipped = list(zip(X['headline'], X['body']))
    X_train, X_test, y_train, y_test = train_test_split(zipped, y, stratify=y, random_state=123)

    def extract_from_zipped(zipped, idx):
        return np.array([pair[idx] for pair in zipped])

    X_train_headline = extract_from_zipped(X_train, 0)
    X_train_body = extract_from_zipped(X_train, 1)

    print("Train shapes", X_train_headline.shape, X_train_body.shape)

    X_test_headline = extract_from_zipped(X_test, 0)
    X_test_body = extract_from_zipped(X_test, 1)

    print("Test shapes", X_test_headline.shape, X_test_body.shape)

    embedding_headline_weights = get_pretrained_embeddings(
        '/media/radoslav/ce763dbf-b2a6-4110-960f-2ef10c8c6bde/MachineLearning/glove.6B', tokenizer_headline)
    embedding_body_weights = get_pretrained_embeddings(
        '/media/radoslav/ce763dbf-b2a6-4110-960f-2ef10c8c6bde/MachineLearning/glove.6B', tokenizer_body)

    model = encode_with_bi_lstm(embedding_headline_weights=embedding_headline_weights,
                                embedding_body_weights=embedding_body_weights)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    currentTime = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    os.mkdir('checkpoints/' + currentTime)

    currentCheckointFolder = os.path.join('checkpoints', currentTime)

    save_tokenizer(os.path.join(currentCheckointFolder, 'tokenizer_headline.pkl'), tokenizer_headline)
    save_tokenizer(os.path.join(currentCheckointFolder, 'tokenizer_body.pkl'), tokenizer_body)
    save_tokenizer(os.path.join(currentCheckointFolder, 'label_to_id.pkl'), label_to_id)

    early_stopping = EarlyStopping(patience=3)
    model_checkpoint = ModelCheckpoint(
        os.path.join(currentCheckointFolder, 'weights.{epoch:02d}-{val_loss:.2f}.hdf5'), monitor='val_loss')

    model.fit([X_train_headline, X_train_body], y_train, validation_data=([X_test_headline, X_test_body], y_test),
              class_weight=class_weights, batch_size=128, epochs=20, callbacks=[early_stopping, model_checkpoint])

    preds = model.predict([X_test_headline, X_test_body])
    preds_y = [np.argmax(pred) for pred in preds]

    y_test_label = [np.argmax(one_hot) for one_hot in y_test]

    conf_matrix = confusion_matrix(y_true=y_test_label, y_pred=preds_y)
    print(conf_matrix)

    report = classification_report(y_true=y_test_label, y_pred=preds_y)
    print(report)


if __name__ == '__main__':
    run()
