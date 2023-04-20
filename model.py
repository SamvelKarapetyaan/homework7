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
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer


data = pd.read_csv('hospital_deaths_train.csv')
# Replaces missing values with the mean value of the non-missing elements along each column.
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(data)
df_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
df_imputed = df_imputed

# Seperate the target and dataset
X = df_imputed.drop('In-hospital_death', axis=1)
y =df_imputed['In-hospital_death']

# split the data into test and dat
xtrain, xtest, ytrain, ytest= train_test_split(X , y , test_size=0.3)

#  DO StandardScaler on xtest and xtrain
scaler = StandardScaler()
scaler.fit(xtrain)
xtrain = scaler.transform(xtrain)
xtest = scaler.transform(xtest)

# generates new samples in the minority class by randomly sampling with replacement from the existing samples
ros = RandomOverSampler(random_state=10)
X_train_resampled, y_train_resampled = ros.fit_resample(xtrain, ytrain)

class Model:
    def __init__(self, algorithm):
        self.algorithm = algorithm

    def fit(self, X, y):
        #  do GridsearCV on Logistic Regression and fit it
        if self.algorithm == "LogisticRegression":
            logistic_reg_params = {
                'C': [0.1, 10, 11, 9]}

            logistic_reg = GridSearchCV(LogisticRegression(max_iter=100000), param_grid=logistic_reg_params, cv=5,
                                        scoring='recall')
            logistic_reg.fit(X, y)
            param = logistic_reg.best_estimator_
            self.algorithm = LogisticRegression(max_iter=100000, C=param.C)
        #  do GridsearCV on GaussianNB and fit it
        elif self.algorithm == "GaussianNB":
            param_grid_gnb = {'priors': [None, [0.1, 0.9], [0.3, 0.7], [0.5, 0.5], [0.7, 0.3], [0.9, 0.1]]}

            Gausian = GridSearchCV(GaussianNB(), param_grid=param_grid_gnb, cv=5, scoring='recall')
            Gausian.fit(X, y)
            param = Gausian.best_estimator_
            self.algorithm = GaussianNB(priors=param.priors)
        #  do GridsearCV on KNeighborsClassifier and fit it
        elif self.algorithm == "KNeighborsClassifier":
            knn_params = {
                'n_neighbors': [3, 5, 7],
                'weights': ['uniform', 'distance'],
                'leaf_size': [10, 20]}

            Gausian = GridSearchCV(KNeighborsClassifier(), param_grid=knn_params, cv=5, scoring='recall')
            Gausian.fit(X, y)
            param = Gausian.best_estimator_
            self.algorithm = KNeighborsClassifier(n_neighbors=param.n_neighbors, weights=param.weights,
                                                  leaf_size=param.leaf_size)
        #  do GridsearCV on DecisionTreeClassifier and fit it
        elif self.algorithm == "DecisionTreeClassifier":
            decision_tree_params = {
                'max_depth': [100, 150, 200]}

            Gausian = GridSearchCV(DecisionTreeClassifier(), param_grid=decision_tree_params, cv=5, scoring='recall')
            Gausian.fit(X, y)
            param = Gausian.best_estimator_
            self.algorithm = DecisionTreeClassifier(max_depth=param.max_depth)

        #  do GridsearCV on AdaBoostClassifier and fit it
        elif self.algorithm == "AdaBoostClassifier":
            AdaBoost_param_grid = {
                'n_estimators': [200, 250],
                'learning_rate': [0.5, 1.0]}

            Gausian = GridSearchCV(AdaBoostClassifier(), param_grid=AdaBoost_param_grid, cv=5, scoring='recall',
                                   n_jobs=1)
            Gausian.fit(X, y)
            param = Gausian.best_estimator_
            self.algorithm = AdaBoostClassifier(n_estimators=param.n_estimators, learning_rate=param.learning_rate)

        #  do GridsearCV on XGBClassifier and fit it
        elif self.algorithm == "XGBClassifier":
            XGBoost_param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [3, 5],
                'colsample_bytree': [0.5, 1.0]}

            Gausian = GridSearchCV(XGBClassifier(), param_grid=XGBoost_param_grid, cv=5, scoring='recall', n_jobs=4)
            Gausian.fit(X, y)
            param = Gausian.best_estimator_
            self.algorithm = XGBClassifier(n_estimators=param.n_estimators, max_depth=param.max_depth,
                                           colsample_bytree=param.colsample_bytree)
        #  do GridsearCV on XGBRFClassifier and fit it
        elif self.algorithm == "XGBRFClassifier":
            XGRBoost_param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [3, 7],
                'learning_rate': [0.1, 0.5]
            }
            Gausian = GridSearchCV(XGBRFClassifier(), param_grid=XGRBoost_param_grid, cv=5, scoring='recall', n_jobs=4)
            Gausian.fit(X, y)
            param = Gausian.best_estimator_
            self.algorithm = XGBRFClassifier(n_estimators=param.n_estimators, max_depth=param.max_depth,
                                             learning_rate=param.learning_rate)
        #  do GridsearCV on SVC and fit it
        elif self.algorithm == "SVC":
            SVC_param_grid = {
                'C': [1, 10],
                'kernel': ['poly', 'rbf']}

            Gausian = GridSearchCV(SVC(probability=True), param_grid=SVC_param_grid, cv=5, scoring='recall', n_jobs=4)

            Gausian.fit(X, y)
            param = Gausian.best_estimator_
            self.algorithm = SVC(probability=True, C=param.C, kernel=param.kernel)
        #  do GridsearCV on LinearDiscriminantAnalysis and fit it
        elif self.algorithm == "LinearDiscriminantAnalysis":
            lda_param_grid = {'solver': ['svd', 'lsqr']}

            Gausian = GridSearchCV(LinearDiscriminantAnalysis(), param_grid=lda_param_grid, cv=5, scoring='recall')

            Gausian.fit(X, y)
            param = Gausian.best_estimator_
            self.algorithm = LinearDiscriminantAnalysis(solver=param.solver)
        #  do GridsearCV on QuadraticDiscriminantAnalysis and fit it
        elif self.algorithm == "QuadraticDiscriminantAnalysis":
            qda_param_grid = {'reg_param': [0.0, 0.001, 0.5]}

            Gausian = GridSearchCV(QuadraticDiscriminantAnalysis(), param_grid=qda_param_grid, cv=5, scoring='recall')

            Gausian.fit(X, y)
            param = Gausian.best_estimator_
            self.algorithm = QuadraticDiscriminantAnalysis(reg_param=param.reg_param)
        self.algorithm = self.algorithm.fit(X, y)

    def predict(self, X_test):
        return self.algorithm.predict(X_test)

    def predict_proba(self, X_test):
        return self.algorithm.predict_proba(X_test)

    def score(self, X_test, y_test):
        return self.algorithm.score(X_test, y_test)


# ploting the scores in a matrix
def print_metrics(tp, fp, tn, fn):
    Accuracy =  (tp + tn) / (tp + tn + fn + fp)
    Precision= tp / (tp + fp)
    Recall= tp / (tp + fn)
    Specificity =  tn / (tn + fp)
    F1_Score= (tp)/(tp+(fp+fn)/2)
    return Accuracy,  Precision , Recall , Specificity , F1_Score


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


