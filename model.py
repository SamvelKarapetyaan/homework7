from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier, XGBRFClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis


class Model:
    """
    A class for classification using various machine learning algorithms.
    It provides the methods written in valid_name:

    """
    # define a list of valid model names
    valid_name = ['Logistic', 'NB', 'KNN', 'Tree', 'AdaBoost', 'GBoost', 'RandomForest', 'SVC', 'LDA', 'QDA']

    def __init__(self, algorithm):
        """
        Initialize the Model object with a given algorithm.

        :param algorithm: str - the name of the algorithm to use
        """
        # store the input algorithm name
        self.algorithm = algorithm

        # define a dictionary of classifier algorithms with their respective hyperparameters for GridSearchCV
        self.classifiers = {
            "Logistic": (LogisticRegression(max_iter=100000, class_weight='balanced'), {
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
            "Tree": (DecisionTreeClassifier(class_weight='balanced'), {
                'max_depth': [100, 150, 200]
            }),
            "AdaBoost": (AdaBoostClassifier(), {
                'n_estimators': [200, 250],
                'learning_rate': [0.5, 1.0]
            }),
            "GBoost": (XGBClassifier(class_weight='balanced'), {
                'n_estimators': [100, 200],
                'max_depth': [3, 5],
                'colsample_bytree': [0.5, 1.0]
            }),
            "RandomForest": (XGBRFClassifier(class_weight='balanced'), {
                'n_estimators': [100, 200],
                'max_depth': [3, 7],
                'learning_rate': [0.1, 0.5]
            }),
            "SVC": (SVC(probability=True, class_weight='balanced'), {
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
        return self.algorithm.predict_proba(X_test)[:, 1]

    def score(self, X_test, y_test):
        # Calculate the score (Accuracy)
        return self.algorithm.score(X_test, y_test)

    def threshold(self, x_train, y_train):
        y_pred = self.predict_proba(x_train)
        fpr, tpr, thresholds = roc_curve(y_train, y_pred)
        roc_distances = np.sqrt(np.sum(np.square(1 - tpr) + np.square(fpr)))
        best_threshold_index = np.argmin(roc_distances)
        return thresholds[best_threshold_index]

