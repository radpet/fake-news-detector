import pandas as pd


def merge_train():
    stances, bodies = load('data/train/train_stances.csv', 'data/train/train_bodies.csv')
    return merge(stances, bodies, 'data/train/train.csv')


def merge_test():
    stances, bodies = load('data/test/competition_test_stances.csv', 'data/test/competition_test_bodies.csv')
    return merge(stances, bodies, 'data/test/test.csv')


def load(stances, bodies):
    return pd.read_csv(stances), pd.read_csv(bodies)


def merge(stances, bodies, output):
    print('Merging {} and {}....'.format(stances, bodies))

    merged = pd.merge(stances, bodies, on='Body ID', how='outer')

    print('....Stances frame shape', stances.shape)
    print('....Bodies frame shape', bodies.shape)
    print('....Merged frame shape', merged.shape)

    merged.to_csv(output, index=False)

    print('Merge was successful')


def run():
    # merge_train()
    merge_test()


if __name__ == '__main__':
    run()
