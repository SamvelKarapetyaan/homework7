from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier, XGBRFClassifier
from sklearn.svm import SVC
import pandas as pd
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

class Model:
    # define a list of valid model names
    valid_name = ['Logistic', 'NB', 'KNN', 'Tree', 'AdaBoost', 'GBoost', 'RandomForest', 'SVC', 'LDA', 'QDA']

    def __init__(self, algorithm):
        """
        Initialize the Model object with a given algorithm.

        :param algorithm: str - the name of the algorithm to use
        """
        # store the input algorithm name

        ##################################
        # Checking validation, MAY BE CHANGED
        if algorithm == None:
            algorithm = "SVC"
        elif algorithm not in Model.valid_name:
            raise ValueError(f"Model must be from given list: {Model.valid_name}")
        

        # threshold initialization
        self.threshold = 0.5 # Must be changed
        ##################################

        self.algorithm = algorithm

        # define a dictionary of classifier algorithms with their respective hyperparameters for GridSearchCV
        self.classifiers = {
            "Logistic": (LogisticRegression(max_iter=100000), {
                'C': [0.1, 10, 11, 9]
            }),
            "NB": (GaussianNB(), {
                'priors': [None, [0.1, 0.9], [0.3, 0.7], [0.5, 0.5], [0.7, 0.3], [0.9, 0.1]]
            }),
            "KNN": (KNeighborsClassifier(), {
                'n_neighbors': [3, 5, 7],
                'weights': ['uniform', 'distance'],
                'leaf_size': [10, 20]
            }),
            "Tree": (DecisionTreeClassifier(), {
                'max_depth': [100, 150, 200]
            }),
            "AdaBoost": (AdaBoostClassifier(), {
                'n_estimators': [200, 250],
                'learning_rate': [0.5, 1.0]
            }),
            "GBoost": (XGBClassifier(), {
                'n_estimators': [100, 200],
                'max_depth': [3, 5],
                'colsample_bytree': [0.5, 1.0]
            }),
            "RandomForest": (XGBRFClassifier(), {
                'n_estimators': [100, 200],
                'max_depth': [3, 7],
                'learning_rate': [0.1, 0.5]
            }),
            "SVC": (SVC(probability=True), {
                'C': [1, 10],
                'kernel': ['poly', 'rbf']
            }),
            "LDA": (LinearDiscriminantAnalysis(), {
                'solver': ['svd', 'lsqr']
            }),
            "QDA": (QuadraticDiscriminantAnalysis(), {
                'reg_param': [0.0, 0.001, 0.5]
            })
        }
    def fit(self, X, y):
        # Get the classifier and parameter grid based on the algorithm name
        clf, params = self.classifiers[self.algorithm]
        # Perform grid search to find the best hyperparameters
        grid_search = GridSearchCV(clf, param_grid=params, cv=5, scoring='recall', n_jobs=4)
        grid_search.fit(X, y)
        # Set the algorithm attribute to the best estimator found by grid search
        self.algorithm = grid_search.best_estimator_

    def predict(self, X_test):
        # Use the fitted model to make predictions on new data
        return self.algorithm.predict(X_test)

    def predict_proba(self, X_test):
        # Use the fitted model to make predictions probability on new data
        return self.algorithm.predict_proba(X_test)

    def score(self, X_test, y_test):
        # Calculate the score (Accuracy)
        return self.algorithm.score(X_test, y_test)
