data='https://www.kaggle.com/uciml/news-aggregator-dataset/downloads/news-aggregator-dataset.zip/1'

source ../env/bin/activate
kaggle datasets download -d uciml/news-aggregator-dataset -p data/

python3 splitdata.py