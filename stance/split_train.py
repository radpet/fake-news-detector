import pandas as pd
import numpy as np


def load_train():
    data = pd.read_csv('data/train/train.csv')
    data['articleBody'] = data['articleBody'].str.lower()
    data['Headline'] = data['Headline'].str.lower()

    return data


def split_train(data, training=0.8):
    '''
    Splits the data into train and hold out sets
    :return:
    '''

    np.random.seed(123456)

    article_ids = list(set(data['Body ID'].values))

    np.random.shuffle(article_ids)

    training_ids = article_ids[:int(training * len(article_ids))]
    hold_out_ids = article_ids[int(training * len(article_ids)):]

    train = data[data['Body ID'].isin(training_ids)]
    hold_out = data[data['Body ID'].isin(hold_out_ids)]
    return train, hold_out


def run():
    train, dev_test = split_train(data=load_train(), training=0.8)

    print('Train shape', train.shape)
    print('Train unique articles', len(set(train['Body ID'].values)))
    print('Train stances', train['Stance'].value_counts())
    dev, test = split_train(data=dev_test, training=0.8)

    print('Dev shape', dev.shape)
    print('Dev unique articles', len(set(dev['Body ID'].values)))
    print('Dev stances', dev['Stance'].value_counts())

    # print('Test shape', test.shape)
    # print('Test unique articles', len(set(test['Body ID'].values)))
    # print('Tev stances', test['Stance'].value_counts())

    train.to_csv('data/train/split/train.csv', index=False)
    dev.to_csv('data/train/split/dev.csv', index=False)
    # test.to_csv('data/train/split/test.csv', index=False)


if __name__ == '__main__':
    run()
