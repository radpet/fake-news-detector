# Download dataset

1. run `bash getdata.sh`
2. The dataset should be now in data/train/train.csv


# Model

## BI_LSTM_Baseline

### Summary
The idea of this model is to encode the heading and the body with Bidirectional LSTM each. Then concat the resulting output vectors and run them through dense layers for classifying.

### Performance
