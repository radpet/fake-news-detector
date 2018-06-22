
train_stances='https://raw.githubusercontent.com/FakeNewsChallenge/fnc-1/master/train_stances.csv'
train_bodies='https://raw.githubusercontent.com/FakeNewsChallenge/fnc-1/master/train_bodies.csv'

mkdir data/train

curl -o data/train/train_stances.csv $train_stances
curl -o data/train/train_bodies.csv $train_bodies


test_stances='https://raw.githubusercontent.com/FakeNewsChallenge/fnc-1/master/competition_test_stances.csv'
test_bodies='https://raw.githubusercontent.com/FakeNewsChallenge/fnc-1/master/competition_test_bodies.csv'

mkdir data/test

curl -o data/test/test_stances.csv $test_stances
curl -o data/test/test_bodies.csv $test_bodies


scorer='https://raw.githubusercontent.com/FakeNewsChallenge/fnc-1/master/scorer.py'
curl -o data/scorer.py $scorer