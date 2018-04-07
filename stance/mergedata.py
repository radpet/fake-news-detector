import pandas as pd


def load_train():
    return pd.read_csv('data/train/train_stances.csv'), pd.read_csv('data/train/train_bodies.csv')


def merge_train():
    print('Merging train_stances and train_bodies....')
    stances, bodies = load_train()

    merged = pd.merge(stances, bodies, on='Body ID', how='outer')

    print('....Stances frame shape', stances.shape)
    print('....Bodies frame shape', bodies.shape)
    print('....Merged frame shape', merged.shape)

    merged.to_csv('data/train/train.csv', index=False)

    print('Merged was successful')


def run():
    merge_train()


if __name__ == '__main__':
    run()
