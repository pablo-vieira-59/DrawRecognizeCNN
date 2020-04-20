import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split


# Loading dataset
X = np.load('training_data/X_train.npy')
Y = np.load('training_data/Y_train.npy')
testX = np.load('training_data/X_test.npy')
testY = np.load('training_data/Y_test.npy')

gnb = GaussianNB()

gnb.fit(X, Y)

gnb_S = round(gnb.score(testX, testY) * 100, 2)
print(round(gnb_S, 2,), '%')
