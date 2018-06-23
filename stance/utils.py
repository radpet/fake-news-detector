import os
import pickle
from csv import DictReader

import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

NUM_CLASSES = 4

LABEL_TO_ID = {
    'agree': 0,
    'discuss': 1,
    'unrelated': 2,
    'disagree': 3
}

ID_TO_LABEL = {
    0: 'agree',
    1: 'discuss',
    2: 'unrelated',
    3: 'disagree'
}


def get_labels(data):
    y = data['Stance']
    y = y.map(lambda label: LABEL_TO_ID[label]).values
    return y


def load_dataset(path_headline, path_bodies):
    headlines = []
    with open(path_headline, 'r') as f:
        csv_reader = DictReader(f)
        for line in csv_reader:
            headlines.append(line)

    bodies = []
    with open(path_bodies, 'r') as f:
        csv_reader = DictReader(f)
        for line in csv_reader:
            bodies.append(line)

    idToBody = {}
    for body in bodies:
        idToBody[int(body['Body ID'])] = body['articleBody']
    return headlines, bodies, idToBody


class Loader():

    def load_dataset(self, path_headline, path_bodies):
        headlines = []
        with open(path_headline, 'r') as f:
            csv_reader = DictReader(f)
            for line in csv_reader:
                line['Body ID'] = int(line['Body ID'])
                headlines.append(line)

        bodies = []
        with open(path_bodies, 'r') as f:
            csv_reader = DictReader(f)
            for line in csv_reader:
                bodies.append(line)

        idToBody = {}
        for body in bodies:
            idToBody[int(body['Body ID'])] = body['articleBody']

        self.headlines = headlines
        self.bodies = bodies
        self.id_to_body = idToBody
        return self


class FeatureExtractor():
    def __init__(self, max_features=None):
        self.max_features = max_features
        self.bow_vect = None
        self.tfreq_vect = None
        self.tfidf_vect = None

    def get_train_fit_vect(self, train, test):
        heads = []
        heads_track = {}
        bodies = []
        bodies_track = {}
        body_ids = []
        id_ref = {}
        train_set = []
        train_stances = []
        valid_set = []
        valid_stances = []
        cos_track = {}
        test_heads = []
        test_heads_track = {}
        test_bodies = []
        test_bodies_track = {}
        head_tfidf_track = {}
        body_tfidf_track = {}

        for instance in train.headlines:
            headline = instance['Headline']
            body_id = instance['Body ID']
            if headline not in heads_track:
                heads.append(headline)
                heads_track[headline] = 1
            if body_id not in bodies_track:
                bodies.append(train.id_to_body[body_id])
                bodies_track[body_id] = 1
                body_ids.append(body_id)

        np.random.seed(123456)
        np.random.shuffle(train.headlines)

        training_r = 0.8
        training_heads = train.headlines[:int(training_r * len(train.headlines))]
        valid_heads = train.headlines[int(training_r * len(train.headlines)):]

        for instance in test.headlines:
            headline = instance['Headline']
            body_id = instance['Body ID']
            if headline not in test_heads_track:
                test_heads.append(headline)
                test_heads_track[headline] = 1
            if body_id not in test_bodies_track:
                test_bodies.append(test.id_to_body[body_id])
                test_bodies_track[body_id] = 1

        for i, elem in enumerate(heads + body_ids):
            id_ref[elem] = i

        bow_vectorizer = CountVectorizer(max_features=self.max_features, stop_words='english')
        bow = bow_vectorizer.fit_transform(heads + bodies)

        tfreq_vectorizer = TfidfTransformer(use_idf=False).fit(bow)
        tfreq = tfreq_vectorizer.transform(bow).toarray()

        tfidf_vectorizer = TfidfVectorizer(max_features=self.max_features, stop_words='english'). \
            fit(heads + bodies + test_heads + test_bodies)

        for instance in training_heads:
            head = instance['Headline']
            body_id = instance['Body ID']
            head_tf = tfreq[id_ref[head]].reshape(1, -1)
            body_tf = tfreq[id_ref[body_id]].reshape(1, -1)
            if head not in head_tfidf_track:
                head_tfidf = tfidf_vectorizer.transform([head]).toarray()
                head_tfidf_track[head] = head_tfidf
            else:
                head_tfidf = head_tfidf_track[head]
            if body_id not in body_tfidf_track:
                body_tfidf = tfidf_vectorizer.transform([train.id_to_body[body_id]]).toarray()
                body_tfidf_track[body_id] = body_tfidf
            else:
                body_tfidf = body_tfidf_track[body_id]
            if (head, body_id) not in cos_track:
                tfidf_cos = cosine_similarity(head_tfidf, body_tfidf)[0].reshape(1, 1)
                cos_track[(head, body_id)] = tfidf_cos
            else:
                tfidf_cos = cos_track[(head, body_id)]
            feat_vec = np.squeeze(np.concatenate([head_tf, body_tf, tfidf_cos], axis=1))
            train_set.append(feat_vec)
            train_stances.append(LABEL_TO_ID[instance['Stance']])

        for instance in valid_heads:
            head = instance['Headline']
            body_id = instance['Body ID']
            head_tf = tfreq[id_ref[head]].reshape(1, -1)
            body_tf = tfreq[id_ref[body_id]].reshape(1, -1)
            if head not in head_tfidf_track:
                head_tfidf = tfidf_vectorizer.transform([head]).toarray()
                head_tfidf_track[head] = head_tfidf
            else:
                head_tfidf = head_tfidf_track[head]
            if body_id not in body_tfidf_track:
                body_tfidf = tfidf_vectorizer.transform([train.id_to_body[body_id]]).toarray()
                body_tfidf_track[body_id] = body_tfidf
            else:
                body_tfidf = body_tfidf_track[body_id]
            if (head, body_id) not in cos_track:
                tfidf_cos = cosine_similarity(head_tfidf, body_tfidf)[0].reshape(1, 1)
                cos_track[(head, body_id)] = tfidf_cos
            else:
                tfidf_cos = cos_track[(head, body_id)]
            feat_vec = np.squeeze(np.concatenate([head_tf, body_tf, tfidf_cos], axis=1))
            valid_set.append(feat_vec)
            valid_stances.append(LABEL_TO_ID[instance['Stance']])

        self.bow_vect = bow_vectorizer
        self.tfreq_vect = tfreq_vectorizer
        self.tfidf_vect = tfidf_vectorizer
        return train_set, train_stances, valid_set, valid_stances

    def transform(self, X, labels=False):
        heads = []
        heads_track = {}
        bodies = []
        bodies_track = {}
        body_ids = []
        id_ref = {}
        valid_set = []
        valid_stances = []
        cos_track = {}
        head_tfidf_track = {}
        body_tfidf_track = {}

        for instance in X.headlines:
            head = instance['Headline']
            body_id = instance['Body ID']
            if head not in heads_track:
                heads.append(head)
                heads_track[head] = 1
            if body_id not in bodies_track:
                bodies.append(X.id_to_body[body_id])
                bodies_track[body_id] = 1
                body_ids.append(body_id)

        for i, elem in enumerate(heads + body_ids):
            id_ref[elem] = i

        bow = self.bow_vect.transform(heads + bodies)
        tfreq = self.tfreq_vect.transform(bow).toarray()

        for instance in X.headlines:
            head = instance['Headline']
            body_id = instance['Body ID']
            head_tf = tfreq[id_ref[head]].reshape(1, -1)
            body_tf = tfreq[id_ref[body_id]].reshape(1, -1)
            if head not in head_tfidf_track:
                head_tfidf = self.tfidf_vect.transform([head]).toarray()
                head_tfidf_track[head] = head_tfidf
            else:
                head_tfidf = head_tfidf_track[head]
            if body_id not in body_tfidf_track:
                body_tfidf = self.tfidf_vect.transform([X.id_to_body[body_id]]).toarray()
                body_tfidf_track[body_id] = body_tfidf
            else:
                body_tfidf = body_tfidf_track[body_id]
            if (head, body_id) not in cos_track:
                tfidf_cos = cosine_similarity(head_tfidf, body_tfidf)[0].reshape(1, 1)
                cos_track[(head, body_id)] = tfidf_cos
            else:
                tfidf_cos = cos_track[(head, body_id)]
            feat_vec = np.squeeze(np.concatenate([head_tf, body_tf, tfidf_cos], axis=1))
            valid_set.append(feat_vec)
            if labels:
                valid_stances.append(LABEL_TO_ID[instance['Stance']])

        if labels:
            return valid_set, valid_stances
        else:
            return valid_set

    def save_vect(self, dirpath):
        with open(os.path.join(dirpath, 'bow_vect.pkl'), 'wb') as f:
            pickle.dump(self.bow_vect, f)

        with open(os.path.join(dirpath, 'tfreq_vect.pkl'), 'wb') as f:
            pickle.dump(self.tfreq_vect, f)

        with open(os.path.join(dirpath, 'tfidf_vect.pkl'), 'wb') as f:
            pickle.dump(self.tfidf_vect, f)

    def load_vect(self, bow_vect_path, tfreq_vect_path, tfidf_vect_path):
        with open(os.path.join(bow_vect_path, 'bow_vect.pkl'), 'rb') as f:
            self.bow_vect = pickle.load(f)

        with open(os.path.join(tfreq_vect_path, 'tfreq_vect.pkl'), 'rb') as f:
            self.tfreq_vect = pickle.load(f)

        with open(os.path.join(tfidf_vect_path, 'tfidf_vect.pkl'), 'rb') as f:
            self.tfidf_vect = pickle.load(f)


class FeatureExtractor2():
    def __init__(self, max_words, headline_len, body_len):
        self.headline_len = headline_len
        self.body_len = body_len
        self.max_words = max_words
        self.tokenizer = None

    def fit(self, train, test):
        heads = []
        heads_track = {}
        bodies = []
        bodies_track = {}

        for instance in train.headlines:
            headline = instance['Headline']
            body_id = instance['Body ID']
            if headline not in heads_track:
                heads.append(headline)
                heads_track[headline] = 1
            if body_id not in bodies_track:
                bodies.append(train.id_to_body[body_id])
                bodies_track[body_id] = 1

        for instance in test.headlines:
            headline = instance['Headline']
            body_id = instance['Body ID']
            if headline not in heads_track:
                heads.append(headline)
                heads_track[headline] = 1
            if body_id not in bodies_track:
                bodies.append(test.id_to_body[body_id])
                bodies_track[body_id] = 1
        tokenizer = Tokenizer(num_words=self.max_words)
        tokenizer.fit_on_texts(bodies + heads)

        self.tokenizer = tokenizer

        return self

    def transform(self, X, labels=False):
        heads_track = {}
        heads = []
        bodies_track = {}
        bodies = []
        body_ids = []

        for instance in X.headlines:
            headline = instance['Headline']
            body_id = instance['Body ID']
            if headline not in heads_track:
                heads.append(headline)
                heads_track[headline] = 1
            if body_id not in bodies_track:
                bodies.append(X.id_to_body[body_id])
                bodies_track[body_id] = 1
                body_ids.append(body_id)

        id_ref = {}

        for i, elem in enumerate(heads + body_ids):
            id_ref[elem] = i

        seq = self.tokenizer.texts_to_sequences(heads + bodies)

        transformed_headlines = []
        transformed_bodies = []
        stances = []
        for instance in X.headlines:
            headline = instance['Headline']
            body_id = instance['Body ID']
            head_seq = seq[id_ref[headline]]
            transformed_headlines.append(head_seq)
            body_seq = seq[id_ref[body_id]]
            transformed_bodies.append(body_seq)
            if labels:
                stances.append(LABEL_TO_ID[instance['Stance']])

        transformed_headlines = pad_sequences(transformed_headlines, maxlen=self.headline_len)
        transformed_bodies = pad_sequences(transformed_bodies, maxlen=self.body_len)
        if labels:
            return transformed_headlines, transformed_bodies, np.array(stances)
        return transformed_headlines, transformed_bodies

    def save(self, currentCheckpointFolder):
        with open(os.path.join(currentCheckpointFolder, 'tokenizer.pkl'), 'wb') as f:
            pickle.dump(self.tokenizer, f)

    def load(self, path):
        with open(path, 'rb') as f:
            self.tokenizer = pickle.load(f)


def f_scorer(y_true, y_pred, labels=False):
    score = 0
    RELATED = ['agree', 'disagree', 'discuss']
    for i, (g, t) in enumerate(zip(y_true, y_pred)):
        g_stance = g
        t_stance = t
        if not labels:
            g_stance = ID_TO_LABEL[g]
            t_stance = ID_TO_LABEL[t]
        if g_stance == t_stance:
            score += 0.25
            if g_stance != 'unrelated':
                score += 0.50
        if g_stance in RELATED and t_stance in RELATED:
            score += 0.25
    return score


class_weights = {
    'unrelated': 1 / 0.73131,
    'discuss': 1 / 0.17828,
    'agree': 1 / 0.0736012,
    'disagree': 1 / 0.0168094
}

class_weights = {LABEL_TO_ID[label]: val for (label, val) in class_weights.items()}
