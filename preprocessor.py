import numpy as np
import pandas as pd

class Preprocessor:
    def __init__(self):
        self.column_means = None
        self.min_values = None
        self.max_values = None

        # Initialize target name
        self.target_column = "In-hospital_death"

    def fit(self, X):
        # Calculate column means for filling NaN values
        self.column_means = X.mean()

        # Calculate min and max values for scaling
        self.min_values = np.nanmin(X, axis=0)
        self.max_values = np.nanmax(X, axis=0)

    def transform(self, X):
        # Fill NaN values with column means
        # X_filled = np.where(np.isnan(X_none_to_nan), self.column_means, X_none_to_nan)
        # X_filled = np.where(np.isnan(X), self.column_means, X)

        # # Scale features using min-max scaling
        # X_scaled = (X_filled - self.min_values) / (self.max_values - self.min_values)
        # X_filled = np.where(np.isnan(X_scaled), self.column_means, X_scaled)


        # Pandas initialization
        X.fillna(self.column_means, inplace=True)

        # TODO: Must be written scaling part:
        
        ...

        #################3

        # Train mode checking and return
        if self.target_column in X.columns:
            X_transformed = X.drop(self.target_column, axis=1)
            y = X[self.target_column]
            return X_transformed, y

        print(X.columns)
        return X
        ###################

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
