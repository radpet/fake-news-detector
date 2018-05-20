import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    data = pd.read_csv('data/uci-news-aggregator.csv')
    print(data.keys())
    print(data.shape)

    train, dev_test = train_test_split(data, stratify=data['CATEGORY'], random_state=123)

    print('Train shape', train.shape)

    dev, test = train_test_split(dev_test, stratify=dev_test['CATEGORY'], test_size=0.3, random_state=123)

    print('Dev shape', dev.shape)
    print('Test shape', test.shape)

    train.to_csv('data/train.csv', index=False)
    dev.to_csv('data/dev.csv', index=False)
    test.to_csv('data/test.csv', index=False)


if __name__ == '__main__':
    main()
