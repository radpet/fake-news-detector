# simple idea with trick based on https://arxiv.org/pdf/1707.03264.pdf
import os
from datetime import datetime

import keras
import numpy as np
from keras import Input, Model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, classification_report

from stance.utils import Loader, FeatureExtractor, ID_TO_LABEL, f_scorer, class_weights

vocab_size = 4000
feature_size = 2 * vocab_size + 1
target_size = 4
hidden_size = 50
l2_alpha = 0.0001
learn_rate = 0.01
clip_ratio = 5
batch_size_train = 500
epochs = 90


def get_model():
    features = Input(shape=(feature_size,))
    dense = Dense(hidden_size, activation='relu', kernel_regularizer=keras.regularizers.l2(l2_alpha))(features)
    dropout = Dropout(0.3)(dense)
    output = Dense(target_size, activation='softmax', kernel_regularizer=keras.regularizers.l2(l2_alpha))(dropout)
    model = Model(inputs=[features], outputs=[output])

    model.summary()
    return model


def train():
    file_train_instances = './data/train/train_stances.csv'
    file_train_bodies = './data/train/train_bodies.csv'

    file_test_instances = './data/test/test_stances.csv'
    file_test_bodies = './data/test/test_bodies.csv'

    raw_train = Loader().load_dataset(file_train_instances, file_train_bodies)
    raw_test = Loader().load_dataset(file_test_instances, file_test_bodies)

    fe = FeatureExtractor(vocab_size)

    train_set, train_stances, valid_set, valid_stances = fe.get_train_fit_vect(raw_train, raw_test)
    train_set = train_set + valid_set
    train_stances = train_stances + valid_stances
    train_set = np.array(train_set)
    train_stances = np.array(train_stances)

    valid_set = np.array(valid_set)
    valid_stances = np.array(valid_stances)

    test_set, test_stances = fe.transform(raw_test, labels=True)
    test_set = np.array(test_set)
    test_stances = np.array(test_stances)

    currentTime = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    os.mkdir('checkpoints/' + currentTime)

    currentCheckointFolder = os.path.join('checkpoints', currentTime)

    fe.save_vect(currentCheckointFolder)

    model = get_model()

    model.summary()
    optimizer = Adam(lr=learn_rate, clipvalue=clip_ratio)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    early_stopping = EarlyStopping(patience=3)
    model_checkpoint = ModelCheckpoint(os.path.join(currentCheckointFolder, 'weights.{epoch:02d}-{val_loss:.2f}.hdf5'),
                                       monitor='val_loss',
                                       save_weights_only=True, save_best_only=True)

    model.fit(train_set, train_stances, validation_data=(test_set, test_stances),
              batch_size=batch_size_train, epochs=epochs, callbacks=[early_stopping, model_checkpoint])

    model.save_weights(os.path.join(currentCheckointFolder, 'weights_saved.hdf5'))

    def show_eval_metrics(X, y_true, name='', persist=False):
        preds = model.predict(X)
        preds = np.argmax(preds, axis=1)
        conf_matrix = confusion_matrix(y_true=y_true, y_pred=preds)
        print(conf_matrix)

        with open(os.path.join(currentCheckointFolder, 'conf_matrix_{}.txt'.format(name)), 'w') as f:
            f.write(str(conf_matrix))

        report = classification_report(y_true=y_true, y_pred=preds)
        print(report)

        with open(os.path.join(currentCheckointFolder, 'classification_report_{}.txt'.format(name)), 'w') as f:
            f.write(str(report))

        if persist:
            with open(os.path.join(currentCheckointFolder, 'predictions_{}.csv'.format(name)), 'w') as f:
                f.write('Stance\n')
                for pred in preds:
                    f.write(str(ID_TO_LABEL[pred]))
                    f.write('\n')

        print("******SCORE VIA SCORER*********")
        print(f_scorer(y_true=y_true, y_pred=preds))

    show_eval_metrics(train_set, train_stances, name='train')

    show_eval_metrics(valid_set, valid_stances, name='valid')

    show_eval_metrics(test_set, test_stances, name='test', persist=True)


if __name__ == '__main__':
    train()
