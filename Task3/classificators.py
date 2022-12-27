import numpy as np
from scipy.sparse.construct import kron

from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def Logistic_Regression(x_train_crossvalid, y_train_crossvalid, x_train, y_train, x_test, y_test):
    hyperParameters = {
        'penalty':['l1','l2'],
        'tol':[0.01, 0.1, 1],
        'solver':['liblinear', 'saga']
    }
    
    model = LogisticRegression()
    clf = GridSearchCV(model, hyperParameters, n_jobs=-1)
    clf.fit(x_train_crossvalid, y_train_crossvalid)

    best_params = clf.best_params_
    best_acc_score = clf.best_score_

    print("\rBest model params: \nPenalty: {0} | Tol: {1} | Solver: {2} - > Accuracy = {3}".format(best_params['penalty'], best_params['tol'], best_params['solver'], best_acc_score))
    model = LogisticRegression(penalty=best_params['penalty'], tol=best_params['tol'], solver=best_params['solver'], max_iter=2000)
    model.fit(x_train, y_train)
    print("Accuracy: ", accuracy_score(y_test, model.predict(x_test)))

def KNN(x_train_crossvalid, y_train_crossvalid, x_train, y_train, x_test, y_test):
    hyperParameters = {
        'algorithm' : ['ball_tree', 'kd_tree', 'brute'],
        'p' : [1, 2],
        'n_neighbors':[1, 3, 5, 7]
    }
 
    model = KNeighborsClassifier()
    clf = GridSearchCV(model, hyperParameters, n_jobs=-1)
    clf.fit(x_train_crossvalid, y_train_crossvalid)

    best_params = clf.best_params_
    best_acc_score = clf.best_score_

    print("\rBest model params: \nAlgorithm: {0} | P: {1} | N_neighbors: {2} - > Accuracy = {3}".format(best_params['algorithm'], best_params['p'], best_params['n_neighbors'], best_acc_score))
    model = KNeighborsClassifier(algorithm=best_params['algorithm'], p=best_params['p'], n_neighbors=best_params['n_neighbors'])
    model.fit(x_train, y_train)
    print("Accuracy: ", accuracy_score(y_test, model.predict(x_test)))

def DesicionTree(x_train_crossvalid, y_train_crossvalid, x_train, y_train, x_test, y_test):
    hyperParameters = {
        'criterion' : ["gini", "entropy"],
        'splitter': ["best", "random"],
        'max_depth':[i for i in range(5, 20)]
    }

    model = DecisionTreeClassifier()
    clf = GridSearchCV(model, hyperParameters, n_jobs=-1)
    clf.fit(x_train_crossvalid, y_train_crossvalid)

    best_params = clf.best_params_
    best_acc_score = clf.best_score_

    print("\rBest model params: \nCriterion: {0} | Splitter: {1} | Max_depth: {2} - > Accuracy = {3}".format(best_params['criterion'], best_params['splitter'], best_params['max_depth'], best_acc_score))
    model = DecisionTreeClassifier(criterion=best_params['criterion'], splitter=best_params['splitter'], max_depth=best_params['max_depth'])
    model.fit(x_train, y_train)
    print("Accuracy: ", accuracy_score(y_test, model.predict(x_test)))

def SVM(x_train_crossvalid, y_train_crossvalid, x_train, y_train, x_test, y_test):
    hyperParameters = {
        'C' : [0.1, 0.2, 0.5, 1, 2, 5],
        'kernel' : [ 'poly', 'rbf', 'sigmoid'],
        'tol':[0.001, 0.01, 0.1, 1]
    }

    model = SVC()
    clf = GridSearchCV(model, hyperParameters, n_jobs=-1)
    clf.fit(x_train_crossvalid, y_train_crossvalid)

    best_params = clf.best_params_
    best_acc_score = clf.best_score_

    print("\rBest model params: \nC: {0} | Kernel: {1} | Tol: {2} - > Accuracy = {3}".format(best_params['C'], best_params['kernel'], best_params['tol'], best_acc_score))
    model = SVC(C=best_params['C'], kernel=best_params['kernel'], tol=best_params['tol'])
    model.fit(x_train, y_train)
    print("Accuracy: ", accuracy_score(y_test, model.predict(x_test)))
    return model