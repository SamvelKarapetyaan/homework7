import pandas as pd

class Preprocessor:
    def __init__(self):
        self.means = None
        self.target = 'In-hospital_death'

    def fit(self, data):
        self.means = data.mean()

    def transform(self, data):
        X = data.fillna(self.means)
        
        if self.target in X:
            X = X.drop(self.target, axis=1)
            y = X[self.target]
            return X, y
        

        return X