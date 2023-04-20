import numpy as np

class Preprocessor:
    def __init__(self):
        self.first_mean = None
        self.second_mean = None
        self.min_values = None
        self.max_values = None
        
    def fit(self, X):
        # Calculate separate means for each group based on the first column
        first_group = X[X[:, 0] == 0]
        second_group = X[X[:, 0] == 1]
        
        self.first_mean = np.nanmean(first_group, axis=0)
        self.second_mean = np.nanmean(second_group, axis=0)
        
        # Calculate min and max values for scaling
        self.min_values = np.nanmin(X, axis=0)
        self.max_values = np.nanmax(X, axis=0)
        
        
        
    def transform(self, X, poly=True):
        
        # Create a copy of the input data to avoid modifying the original data
        X_filled = X.copy()
        
        # Fill NaN values based on the first column
        for i in range(X.shape[0]):
            if X[i, 0] == 0:
                X_filled[i, np.isnan(X[i])] = self.first_mean[np.isnan(X[i])]
            elif X[i, 0] == 1:
                X_filled[i, np.isnan(X[i])] = self.second_mean[np.isnan(X[i])]
        
        # Scale features using min-max scaling
        X_scaled = (X_filled - self.min_values) / (self.max_values - self.min_values)
        
        
        if poly:
            # Identify non-boolean columns
            non_boolean_columns = np.apply_along_axis(lambda x: len(np.unique(x)) > 2, axis=0, arr=X_scaled)

            # Generate polynomial features for non-boolean columns
            poly_features = []
            for degree in [2, 3]:
                poly_features.append(np.power(X_scaled[:, non_boolean_columns], degree))
            poly_features = np.hstack(poly_features)

            # Concatenate the generated polynomial features with the original scaled data
            X_extended = np.hstack((X_scaled, poly_features))
        else:
            X_extended=X_scaled
        
        return X_extended
