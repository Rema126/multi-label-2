# Load EDA Pkgs
import pandas as pd
import numpy as np
import skmultilearn
from sklearn.model_selection import train_test_split
import sys
import warnings
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
# ML Pkgs
from nltk import corpus
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, hamming_loss
# Split Dataset into Train and Text
from sklearn.model_selection import train_test_split
# Feature engineering
from sklearn.feature_extraction.text import TfidfVectorizer
# from skmultilearn.problem_transform import BinaryRelevance
# Multi Label Pkgs
from skmultilearn.problem_transform import BinaryRelevance
from skmultilearn.problem_transform import ClassifierChain
from skmultilearn.problem_transform import LabelPowerset
from skmultilearn.adapt import MLkNN

patients = pd.read_csv('PatientsDiets.csv')
patients.head()

col_names = ['ID', 'Age', 'Sex', 'Diagnosis', 'Diet', 'Type', 'Weight', 'Height', 'Renal', 'DM',
             'Liquid', 'NGT', 'LowSalt', 'SemiSolid', 'Normal', 'LowFat']
patients = pd.read_csv('PatientsDiets.csv', header=None, names=col_names)
patients.head()

print("Number of rows in data =",patients.shape[0])
print("Number of columns in data =",patients.shape[1])
print("\n")
print("**Sample data:**")
print(patients.head())

categories = list(patients.columns.values)
categories = categories[8:]
print(categories)

data = patients.loc[np.random.choice(patients.index, size=2000)]

if not sys.warnoptions:
    warnings.simplefilter("ignore")

data['Diagnosis'] = data['Diagnosis'].str.lower()
data.head()

tfidf = TfidfVectorizer()

# Build Features
data = tfidf.fit_transform(corpus).toarray()
data
patients.head()

y = patients[['Renal','DM','Liquid','NGT','LowSalt','SemiSolid','Normal','LowFat', 'FatFree']]

# Split Data
X_train, X_test = train_test_split(data, random_state=42, test_size=0.3)
Y_train, Y_test = train_test_split(data, random_state=42, test_size=0.3)

print(patients['Diagnosis'].shape)
print(X_train.shape)

# Building Our Model
# Estimator + Multilabel Estimator
# Problem Transform
dir(skmultilearn)

# Convert Our Multi-Label Prob to Multi-Class
# binary classficiation
binary_rel_clf = BinaryRelevance(MultinomialNB())
binary_rel_clf.fit(X_train, Y_train)

# Predictions
br_prediction = binary_rel_clf.predict(X_test)

# Convert to Array  To See Result
br_prediction.toarray()

# Accuracy
accuracy_score(Y_test, br_prediction)

# Hamming Loss :Incorrect Predictions
# The Lower the result the better
hamming_loss(Y_test, br_prediction)
