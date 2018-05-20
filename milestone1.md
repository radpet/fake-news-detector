# Abstract
  The purpose of this document is to introduce the reader to our current research and development state
for fake news detection and anylisis.


# Introduction

We are of the opinion that the task of detecting fake news is complex even for humans. That is why we decided to approach
the problem not by automatically detecting whether an article is fake or not but to analize the content and provide
meaningful insights such as sentiment, category, summary and many others.


# Stance Detection

## Motivation
Our main motives towards implementing a stance detection solution is pretty well said in the introduction of the [http://www.fakenewschallenge.org/](http://www.fakenewschallenge.org/) competion
> In particular, a good stance detection solution would allow a human fact checker to enter a claim or headline and instantly retrieve the top articles that agree, disagree or discuss the claim/headline in question. 
> They could then look at the arguments for and against the claim, and use their human judgment and reasoning skills to assess the validity of the claim in question. 
>Such a tool would enable human fact checkers to be fast and effective.

Our goal is not only to output whether the headline agree, disagree or discuss the main content but to find those phrases
in the text that contribute most to the result.

## [Implementation](https://github.com/radpet/fake-news-detector/tree/master/stance)

The dataset we used for training was one from the [http://www.fakenewschallenge.org/](http://www.fakenewschallenge.org/) competion. We did a brief data exploration in order to obtain intuition about the data inside. [Link to data exploration notebook](https://github.com/radpet/fake-news-detector/blob/master/stance/Data%20Exploration.ipynb)

We split the provided train data into 3 parts - 80/10/10 Train, Dev (validation) and Test. There was a catch with the splitting we found the hard way. After splitting in random with stratify and training a model we received exceptionally good result on our test set. Later when tested on the provided test set the model score was alwful. This was caused by duplicating articles both in train and test split. Our hypothesis is that the network seems to have learned the relationship between the exact feature vectors rather than generalized. We then split the train set again in 80/10/10 manner but there were no duplicate articles in the other splits. One can view the splitting script [here](https://github.com/radpet/fake-news-detector/blob/master/stance/split_train.py)

We tried two deep learning approaches. We experimented both with LSTM and GRU 
* Tokenize the headline and use glove word embeddings. Encode headline with bidirectional reccurent network (GRU).
Tokenize the article and use glove word embeddings. Encode the article using bidirectional reccurent network(GRU). Concatanate the output vectors of the headline and article and run the extracted features through 2 dense layers.

This model did not have good score (*Should train again because I forgot what the scores were)

* The second approach adds attention layers after each of the bidirectional GRU and uses the 'weighted' output vector for classification.

A model checkpoint can be found here [here](https://github.com/radpet/fake-news-detector/tree/master/stance/checkpoints/2018-05-13_16:54:37) and its source [here](https://github.com/radpet/fake-news-detector/blob/master/stance/bi_lstm_baseline.py)

Remaining tasks:
* Experiment adding more features derived from other models.
* Since we introduced attention mechanism we hope that using the learned weights we can visualize which phrases the network emphasize on.

## Performance

2BI_GRU_2_DENSE - TBA

2BI_GRU_ATT_2_DENSE - [classification report](https://github.com/radpet/fake-news-detector/blob/master/stance/checkpoints/2018-05-13_16:54:37/classification_report_dev.txt)


# Headline categorization

## Motivation

Nowadays more and more article publishing websites use tricky techniques to bring more visitors and expand its audience. Some of this techniques include classifying an article in a currently trending category in order to bring more viewes. Our approach to battle this is to introduce solution that assigns one of the 4 categories (business; science and technology; entertainment; health) to the headline of the article. Assigning categories to headlines have a lot of benefits such as allowing the search of an article not only by keywords but by choosing one of these 4 domains. Another positive is that if the predicted domain differs a lot from the one given by the source the article might not be that trustworthy.

## Implementation

The dataset we used was taken from [https://www.kaggle.com/uciml/news-aggregator-dataset](https://www.kaggle.com/uciml/news-aggregator-dataset). We split the data in train/dev/test(75/16.67/8.33) with stratify.

Models:
* We experimented with bidirectional recurrent network and pretrained glove embeddings. The model can be seen [here](https://github.com/radpet/fake-news-detector/blob/master/news_aggregator/bi_gru_classificator_baseline.py). We provide a [checkpoint](https://github.com/radpet/fake-news-detector/tree/master/news-aggregator/checkpoints).

## Performance

TBA

# Sentiment (Irony Detection)

## Motivation
We all want the news and articles we read to be straightforward and unambiguous. Humor, irony and sarcasm can be good tools to tell a story or bring up a point, but often they cause misunderstanding and confusion if taken seriously.
By detecting if a statement is ironic or not we can give insight whether the article is serious or satirical.

## [Implementation](https://github.com/radpet/fake-news-detector/tree/master/irony)
The dataset we used was provided by [SemEval](https://competitions.codalab.org/competitions/17468). Data exploration can be found [here](https://github.com/radpet/fake-news-detector/blob/master/irony/Data.ipynb).

#### [Baseline](https://github.com/radpet/fake-news-detector/blob/master/irony/baseline.ipynb)
We used Tf-idf vectorizer to encode the tweets and Logistic regression to classify whether they are ironic or not. This approach achieves 0.65 F1 score.

#### Models
* We used pretrained Glove embeddings, trained on english tweets. The vectors are encoded through a Bidirectional LSTM layer and an Attention layer. Afterwards we use 2 dense layers to classify the extracted features. This model achieves 0.66 F1 score.





