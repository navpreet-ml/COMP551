import re
import glob
import sklearn
import numpy as np
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords 
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from collections import Counter

# Here are the data loading =======================================================================
def get_positive_data(type_):

    examples = []
    labels = []

    file_names = glob.glob('./{0}/pos/*.txt'.format(type_))
    for n in file_names:
        f = open(n, encoding = "utf8")
        examples.append(f.read())
        labels.append(1)
        f.close()
    return examples, labels

def get_negative_data(type_):

    examples = []
    labels = []
    
    file_names = glob.glob('./{0}/neg/*.txt'.format(type_))
    for n in file_names:
        f = open(n, encoding = "utf8")
        examples.append(f.read())
        labels.append(0)
        f.close()
    return examples, labels

# Load test data in order
numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts
def get_test_data(type_):
    examples = []
#    file_names = glob.glob('./{0}/*.txt'.format(type_))
    file_names = sorted(glob.glob('./{0}/*.txt'.format(type_)), key=numericalSort) # sorted file name
    for n in file_names:
        f = open(n, encoding = "utf8")
        examples.append(f.read())
        f.close()
    return examples

# Data pre-processing =============================================================================
pos_reviews, pos_target  = get_positive_data('train')
neg_reviews, neg_target  = get_negative_data('train')
reviews_test = get_test_data('test')

REPLACE_NO_SPACE = re.compile("(\.)|(\;)|(\:)|(\!)|(\')|(\?)|(\,)|(\")|(\()|(\))|(\[)|(\])")
REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")
# Lower case
def preprocess_reviews(reviews):
    reviews = [REPLACE_NO_SPACE.sub("", line.lower()) for line in reviews]
    reviews = [REPLACE_WITH_SPACE.sub(" ", line) for line in reviews]
    return reviews
# Remove stop words
english_stopwords = stopwords.words('english')
def remove_stopwords(reviews):
    removed_stopwords = []
    for review in reviews:
        removed_stopwords.append(' '.join([word for word in review.split() if word not in english_stopwords]))
    return removed_stopwords
# Stemming and lemmatizing
def get_stemmed_text(reviews):
    stemmer = PorterStemmer()
    return [' '.join([stemmer.stem(word) for word in review.split()]) for review in reviews]
def get_lemmatized_text(reviews):
    lemmatizer = WordNetLemmatizer()
    return [' '.join([lemmatizer.lemmatize(word) for word in review.split()]) for review in reviews]

pos_reviews = remove_stopwords(pos_reviews)
neg_reviews = remove_stopwords(neg_reviews)
reviews_test = remove_stopwords(reviews_test)

pos_reviews = preprocess_reviews(pos_reviews)
neg_reviews = preprocess_reviews(neg_reviews)
reviews_test = preprocess_reviews(reviews_test)

pos_reviews = get_lemmatized_text(pos_reviews) # train pos
neg_reviews = get_lemmatized_text(neg_reviews) # train neg
reviews_test = get_lemmatized_text(reviews_test) # test

# List of string to dictionary (Bag of words count) ===============================================
def count_text(reviews):
    word_counts = {}
    for review in reviews:
        words = re.split("\s+", review)
        for word in words:
            word_counts[word] = word_counts.get(word, 0.0) + 1.0
    return word_counts
    
# Naive Bayes =====================================================================================
def make_class_prediction(X, neg_reviews, pos_reviews):
    result = []

    negative_counts = count_text(neg_reviews) # total word counts dict in the neg
    positive_counts = count_text(pos_reviews) #                               pos

    prob_negative = np.log( len(neg_reviews) / (len(pos_reviews) + len(neg_reviews)) )
    prob_positive = np.log( len(pos_reviews) / (len(pos_reviews) + len(neg_reviews)) ) # target is y

    for review in X:
        word_counts = Counter(re.split("\s+", review))
        neg_score = 0
        pos_score = 0
        for word in word_counts:
            neg_score += np.log( (negative_counts.get(word, 0) + 1) / (len(neg_reviews) + 2) )
            pos_score += np.log( (positive_counts.get(word, 0) + 1) / (len(pos_reviews) + 2) )
    
        neg_score += prob_negative
        pos_score += prob_positive
        
        if pos_score > neg_score:
            result.append(1)
        else:
            result.append(0)
    
    return result   

# Seperate training and validation sets ===========================================================
X_train_pos, X_val_pos, y_train_pos, y_val_pos = train_test_split(pos_reviews, pos_target, train_size=0.8, test_size=0.2)
X_train_neg, X_val_neg, y_train_neg, y_val_neg = train_test_split(neg_reviews, neg_target, train_size=0.8, test_size=0.2)

X_val = X_val_pos + X_val_neg
y_val = y_val_pos + y_val_neg

y_pred = make_class_prediction(X_val, X_train_neg, X_train_pos)
print(sklearn.metrics.accuracy_score(y_val, y_pred))