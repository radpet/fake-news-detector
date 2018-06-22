import pandas as pd
import numpy as np
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.svm import LinearSVC

def parse_dataset():
    clickbait = []
    non_clickbait = []
    with open("data/clickbait_data", 'rt') as data_in:
        for line in data_in:
            if line.strip():
                clickbait.append(line.strip())
                
                
    with open("data/non_clickbait_data", 'rt') as data_in:
        for line in data_in:
            if line.strip():
                non_clickbait.append(line.strip())

    return clickbait, non_clickbait

def preprocess_titles(titles): 
    return list(map(lambda x: x.lower(), titles))
    
def create_dataframe(clickbait, non_clickbait):
    cb_df = pd.DataFrame({'clickbait': np.ones(len(clickbait)), 'title': preprocess_titles(clickbait)})
    n_cb_df = pd.DataFrame({'clickbait': np.zeros(len(non_clickbait)), 'title': preprocess_titles(non_clickbait)})
    return pd.concat([cb_df, n_cb_df], ignore_index=True)

def train():
    clickbait, non_clickbait = parse_dataset()
    titles = create_dataframe(clickbait, non_clickbait)
    
    idf_tokenizer = TfidfVectorizer(max_features=30000, stop_words='english').fit(titles['title'])
    with open('model/tokenizer.pkl', 'wb') as f:
        pickle.dump(idf_tokenizer, f)
    
    
    X_train, X_test, y_train, y_test = train_test_split(titles['title'], titles['clickbait'],
                                                    stratify=titles['clickbait'], 
                                                    test_size=0.25, random_state=9)
    

    train_tokenized = idf_tokenizer.transform(X_train)
    test_tokenized = idf_tokenizer.transform(X_test)
    
    
    svc = LinearSVC()
    svc.fit(train_tokenized, y_train)
    with open('model/svc.pkl', 'wb') as f:
        pickle.dump(svc, f)
        

    predict = svc.predict(test_tokenized)
    with open('model/report.txt', 'w') as f:
        f.write("***** Classification Report *****\n")
        f.write(str(classification_report(y_pred=predict, y_true=y_test)))
        f.write("***** Confusion Matrix *****\n")
        f.write(str(confusion_matrix(y_test, predict)))
    
    
train()
    
