from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
from dataloader import *
import numpy as np

if __name__ == '__main__':
    X, y = read_data()
    testX, testY = read_data("test.csv")

    clf_lr = LogisticRegression()
    clf_lr.fit(X, y)
    pred_lr = clf_lr.predict(testX)
    accuracy_lr = accuracy_score(pred_lr, testY)

    clf_svm = SVC()
    clf_svm.fit(X, y)
    pred_svm = clf_svm.predict(testX)
    accuracy_svm = accuracy_score(pred_svm, testY)

    clf_rf = RandomForestClassifier()
    clf_rf.fit(X, y)
    pred_rf = clf_rf.predict(testX)
    accuracy_rf = accuracy_score(pred_rf, testY)

    print("lr accuracy is: ", accuracy_lr)
    print("svm accuracy is: ", accuracy_svm)
    print("rf accuracy is: ", accuracy_rf)
