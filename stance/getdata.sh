
train_stances='https://raw.githubusercontent.com/FakeNewsChallenge/fnc-1/master/train_stances.csv'
train_bodies='https://raw.githubusercontent.com/FakeNewsChallenge/fnc-1/master/train_bodies.csv'

mkdir data/train

curl -o data/train/train_stances.csv $train_stances
curl -o data/train/train_bodies.csv $train_bodies

python3 mergedata.py