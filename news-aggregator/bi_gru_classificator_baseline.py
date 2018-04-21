import os

from datetime import datetime
import numpy as np
import pandas as pd
from keras import Input
from keras import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras.layers import Embedding, Bidirectional, GRU, Dense, Dropout
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

from common.GloveEmbeddings import GloveEmbeddings
from common.TokenizerSerializer import TokenizerSerializer

NUM_CATEGORIES = 4
# TODO tune based on data exploration results
MAX_LEN_HEADING = 32
MAX_NUM_WORDS = 30000

LABEL_DICT = {
    'b': 0,  # business
    't': 1,  # science
    'e': 2,  # fun
    'm': 3  # health
}


def load():
    train = pd.read_csv('data/train.csv')
    dev = pd.read_csv('data/dev.csv')
    return train, dev


def get_labels(data):
    return data['CATEGORY']


def get_text(data):
    return data['TITLE']


def preprocess_text(text):
    modified = text.str.replace('\'', '')
    modified = modified.str.lower()
    return modified


def get_labels_one_hot(data):
    labels = get_labels(data)
    labels = labels.map(lambda l: LABEL_DICT[l])
    one_hot = np.zeros((labels.shape[0], NUM_CATEGORIES))
    one_hot[np.arange(labels.shape[0]), labels] = 1
    return one_hot


def get_model(input_dim, embeddings_matrix):
    input = Input(shape=(input_dim,))
    embedding = Embedding(embeddings_matrix.shape[0], embeddings_matrix.shape[1], weights=[embeddings_matrix],
                          trainable=False)(input)
    rec = Bidirectional(GRU(200, unroll=True, dropout=0.2))(embedding)
    dense = Dense(100, activation='relu')(rec)
    dropout = Dropout(0.4)(dense)
    output = Dense(NUM_CATEGORIES, activation='softmax')(dropout)

    model = Model(inputs=[input], outputs=[output])
    model.summary()
    return model


def create_tokenizer(data):
    texts = get_text(data)
    texts = texts.fillna("")
    tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
    tokenizer.fit_on_texts(texts)

    return tokenizer


def to_sequence(data, tokenizer):
    tokenized = tokenizer.texts_to_sequences(data)
    tokenized = pad_sequences(tokenized, maxlen=MAX_LEN_HEADING)
    return tokenized


def get_features(data, tokenizer):
    preprocessed_text = preprocess_text(get_text(data))
    as_sequence = to_sequence(preprocessed_text, tokenizer)
    return as_sequence


def main(current_checkpoint_folder=None, weights=None):
    train, dev = load()
    if current_checkpoint_folder is None:
        current_time = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
        os.mkdir('checkpoints/' + current_time)
        current_checkpoint_folder = os.path.join('checkpoints', current_time)

        tokenizer = create_tokenizer(train + dev)
        tokenizer_serializer = TokenizerSerializer(tokenizer)
        tokenizer_serializer.serialize(os.path.join(current_checkpoint_folder, 'tokenizer'))
    else:
        tokenizer = TokenizerSerializer.load(os.path.join(current_checkpoint_folder, 'tokenizer'))

    train_X = get_features(train, tokenizer)
    train_y = get_labels_one_hot(train)
    print(train_X[0])
    print(train_y[0])

    dev_X = get_features(dev, tokenizer)
    dev_y = get_labels_one_hot(dev)

    embeddings = GloveEmbeddings(
        '/media/radoslav/ce763dbf-b2a6-4110-960f-2ef10c8c6bde/MachineLearning/glove.6B/glove.6B.100d.txt',
        100).load().get_embedding_matrix_for_tokenizer(tokenizer)

    # callbacks
    early_stopping = EarlyStopping(patience=3)
    learning_rate_reducer = ReduceLROnPlateau()
    model_checkpoint = ModelCheckpoint(
        os.path.join(current_checkpoint_folder, 'weights.{epoch:02d}-{val_loss:.2f}.hdf5'), monitor='val_loss')
    tensorboard = TensorBoard(log_dir=current_checkpoint_folder)

    initial_epoch = 0

    if weights is None:
        model = get_model(MAX_LEN_HEADING, embeddings_matrix=embeddings)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    else:
        model = load_model(os.path.join(current_checkpoint_folder, weights))
        initial_epoch = int(float(weights.split('.')[1][:2]))

    model.fit(train_X, train_y, validation_data=(dev_X, dev_y), batch_size=64, epochs=initial_epoch + 15,
              callbacks=[learning_rate_reducer, early_stopping, model_checkpoint, tensorboard],
              initial_epoch=initial_epoch)


import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", help="continue training from this checkpoint (folder) ")
    parser.add_argument("--weights", help="continue training with this weights at the checkpoint")
    args = parser.parse_args()
    if 'checkpoint' in args:
        main(args.checkpoint, args.weights)
    else:
        main()
