from sklearn.tree import DecisionTreeClassifier

class Model:
    def __init__(self):
        self.model = DecisionTreeClassifier()
        self.threshold = 0.5

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict_proba(X)