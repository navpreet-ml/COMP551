# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 13:06:59 2019

@author: nsingh30
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 17:57:21 2019

@author: nsingh30
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 10:53:15 2019

@author: nsingh30
"""

import re
import glob
import csv
import pandas as pd
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer
import nltk


def get_labeled_data(type_):

    examples = []
    labels = []

    file_names = glob.glob('./{0}/{0}/pos/*.txt'.format(type_))
    for n in file_names:
        f = open(n, encoding = "utf8")
        examples.append(f.read())
        labels.append('positive')
        f.close()

    file_names = glob.glob('./{0}/{0}/neg/*.txt'.format(type_))
    for n in file_names:
        f = open(n, encoding = "utf8")
        examples.append(f.read())
        labels.append('negative')
        f.close()
    return examples, labels

numbers = re.compile(r'(\d+)')

def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def get_test_data(type_):
    examples = []
#    file_names = glob.glob('./{0}/*.txt'.format(type_))
    file_names = sorted(glob.glob('./{0}/{0}/*.txt'.format(type_)), key=numericalSort) # sorted file name
    for n in file_names:
        f = open(n, encoding = "utf8")
        examples.append(f.read())
        f.close()
    return examples

reviews_train, target  = get_labeled_data('train')
reviews_test = get_test_data('test')

REPLACE_NO_SPACE = re.compile("(\.)|(\;)|(\:)|(\!)|(\')|(\?)|(\,)|(\")|(\()|(\))|(\[)|(\])")
REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")
#
def preprocess_reviews(reviews):
    reviews = [REPLACE_NO_SPACE.sub("", line.lower()) for line in reviews]
    reviews = [REPLACE_WITH_SPACE.sub(" ", line) for line in reviews]
    return reviews
#
english_stopwords = stopwords.words('english')
def remove_stopwords(reviews):
    removed_stopwords = []
    for review in reviews:
        removed_stopwords.append(' '.join([word for word in review.split() if word not in english_stopwords]))
    return removed_stopwords
#
#def get_stemmed_text(reviews):
#    stemmer = PorterStemmer()
#    return [' '.join([stemmer.stem(word) for word in review.split()]) for review in reviews]

def get_lemmatized_text(reviews):
    lemmatizer = WordNetLemmatizer()
    return [' '.join([lemmatizer.lemmatize(word) for word in review.split()]) for review in reviews]

#def spelling_correct(reviews):
#    for review in reviews:
#        review.apply(lambda x: str(TextBlob(x).correct()))
#    return reviews

 
#reviews_train_clean = spelling_correct(reviews_train)
#reviews_test_clean = spelling_correct(reviews_test)
reviews_train_clean = remove_stopwords(reviews_train)
reviews_test_clean = remove_stopwords(reviews_test)
reviews_train_clean = preprocess_reviews(reviews_train_clean)
reviews_test_clean = preprocess_reviews(reviews_test_clean)
reviews_train_clean = get_lemmatized_text(reviews_train_clean)
reviews_test_clean = get_lemmatized_text(reviews_test_clean)
#reviews_train_clean = get_stemmed_text(reviews_train)
#reviews_test_clean = get_stemmed_text(reviews_test)




from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import RidgeClassifier


#wc_vectorizer = CountVectorizer(max_features=1000, lowercase=True, ngram_range=(1,2),analyzer = "word")
#wc_vectorizer.fit(reviews_train_clean)
#tfidf_vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), sublinear_tf=True, smooth_idf=False, max_features=None)
#        dtype=<class 'numpy.int64'>, encoding='utf-8', input='content',
#        ngram_range=(1, 2), norm='l2', preprocessor=None, smooth_idf=True)
tfidf_vectorizer = TfidfVectorizer(binary = 'True', ngram_range=(1, 3))
tfidf_vectorizer.fit(reviews_train)

X = tfidf_vectorizer.transform(reviews_train)
X_test = tfidf_vectorizer.transform(reviews_test)
#X_train, X_val, y_train, y_val = train_test_split(X, target, train_size = 0.75)
X_train = X
y_train = target

#classifier_LR = LogisticRegression(C = 10.0, penalty = 'l2', tol = 0.0001)
#classifier_LR.fit(X_train, y_train)
#y_pred_LR = classifier_LR.predict(X_val)
#cm_LR = confusion_matrix(y_val, y_pred_LR)
#accuracies_LR = cross_val_score(estimator = classifier_LR, X = X_train, y = y_train, cv = 10)
#accuracies_LR.mean()
#accuracies_LR.std()
#parameters_LR = [{'C': [0.1, 0.5, 1.0, 10.0], 'penalty': ['l2'], 'tol': [0.0001, 0.00001]}]
#grid_search_LR = GridSearchCV(estimator = classifier_LR,
#                           param_grid = parameters_LR,
#                           scoring = 'accuracy',
#                           cv = 10,
#                           n_jobs = -1)
#
#grid_search_LR = grid_search_LR.fit(X_train, y_train)
#best_accuracy_LR = grid_search_LR.best_score_
#best_parameters_LR = grid_search_LR.best_params_ 



classifier_SVC = LinearSVC(C = 10, penalty = 'l2', max_iter = 1000, tol = 0.0001)
classifier_SVC.fit(X_train, y_train)
y_pred_SVC = classifier_SVC.predict(X_test)
#cm_SVC = confusion_matrix(y_val, y_pred_SVC)
#accuracies_SVC = cross_val_score(estimator = classifier_SVC, X = X_train, y = y_train, cv = 10)
#accuracies_SVC.mean()
#accuracies_SVC.std()
#parameters_SVC = [{'C': [0.1, 1, 10], 'penalty': ['l2'], 'max_iter': [1000, 10000], 'tol':[0.0001, 0.00001]}]
#grid_search_SVC = GridSearchCV(estimator = classifier_SVC,
#                           param_grid = parameters_SVC,
#                           scoring = 'accuracy',
#                           cv = 10,
#                           n_jobs = -1)
#
#grid_search_SVC = grid_search_SVC.fit(X_train, y_train)
#best_accuracy_SVC = grid_search_SVC.best_score_
#best_parameters_SVC = grid_search_SVC.best_params_


#classifier_RC = RidgeClassifier(normalize = 'False', alpha = 0.0001)
#classifier_RC.fit(X_train, y_train)
#y_pred_RC = classifier_RC.predict(X_val)
#cm_RC = confusion_matrix(y_val, y_pred_RC)
#accuracies_RC = cross_val_score(estimator = classifier_RC, X = X_train, y = y_train, cv = 10)
#accuracies_RC.mean()
#accuracies_RC.std()
#parameters_RC = [{'normalize': ['True', 'False'], 'alpha': [0.001, 0.0001]}]
#grid_search_RC = GridSearchCV(estimator = classifier_RC,
#                           param_grid = parameters_RC,
#                           scoring = 'accuracy',
#                           cv = 10,
#                           n_jobs = -1)
#
#grid_search_RC = grid_search_RC.fit(X_train, y_train)
#best_accuracy_RC = grid_search_RC.best_score_
#best_parameters_RC = grid_search_RC.best_params_

label_num = [None]*25000
for i in range(0,len(y_pred_SVC)):
    if y_pred_SVC[i] == 'positive':
        label_num[i] = 1
    else:
        label_num[i] = 0

id = list(range(0,25000))
submission1 = pd.DataFrame({'Id':id,'Category':label_num})

filename = 'Submission Final.csv'

submission1.to_csv(filename,index=False)





