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


# ploting the scores in a matrix
def print_metrics(tp, fp, tn, fn):
    accuracy = (tp + tn) / (tp + tn + fn + fp)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    specificity = tn / (tn + fp)
    f1_score = tp / (tp + (fp + fn) / 2)
    return accuracy, precision, recall, specificity, f1_score


# Define the algorithms to evaluate
models = ['LogisticRegression', 'GaussianNB', 'KNeighborsClassifier', 'DecisionTreeClassifier', 'AdaBoostClassifier',
          'XGBClassifier', 'XGBRFClassifier', 'SVC', 'LinearDiscriminantAnalysis', 'QuadraticDiscriminantAnalysis']

# Define an empty dictionary to store the evaluation metrics for each algorithm
results = {'Algorithm': [], 'Accuracy': [], 'Precision': [], 'Recall': [], 'Specificity': [], 'F1-Score': [], 'AUC': []}

# Evaluate each algorithm
for model in models:
    clf = Model(model)
    clf.fit(X_train_resampled, y_train_resampled)
    y_pred = clf.predict(xtest)

    # Calculate evaluation metrics
    tn, fp, fn, tp = confusion_matrix(ytest, y_pred).ravel()
    Accuracy, Precision, Recall, Specificity, F1_Score = print_metrics(tp, fp, tn, fn)
    auc = roc_auc_score(ytest, clf.predict_proba(xtest)[:, 1])

    # Store the evaluation metrics in the results dictionary
    results['Algorithm'].append(model)
    results['Accuracy'].append(Accuracy)
    results['Precision'].append(Precision)
    results['Recall'].append(Recall)
    results['Specificity'].append(Specificity)
    results['F1-Score'].append(F1_Score)
    results['AUC'].append(auc)

# Convert the results dictionary to a DataFrame
results_df = pd.DataFrame(results)

# Set the algorithm name as the index
results_df.set_index('Algorithm', inplace=True)

print(results_df)
