import argparse
import joblib
import json
import pandas as pd

from model import Model
from preprocessor import Preprocessor


class Pipeline:
    """
    Pipeline
    =====================
    The Pipeline class provides a way to preprocess and model data for training and testing. 
    It has following method.

     - run(data: pd.DataFrame, test: bool = False):
    
    Parameters:
    ---------------------
    1. data (pandas DataFrame): The input data to preprocess and model.
    2. test (bool): If True, load the preprocessor and model from their saved files and use them to predict the output for the input data. Otherwise, fit the preprocessor and model to the input data and save them to their respective files for future testing.
    
    
    Returns:
    ---------------------
    If test is True, a JSON file containing the predicted probabilities and threshold. Otherwise, nothing is returned.
    The run method performs the following steps:

    If test is True, the preprocessor and model are loaded from their saved files and used to predict the output for the input data. The predicted probabilities and threshold are saved to a JSON file.
    Otherwise, the preprocessor is fit to the input data, then the data is transformed using the preprocessor and the transformed data is used to fit the model. Finally, the preprocessor and model are saved to their respective files.
    Conclusion
    The Pipeline class provides a convenient way to preprocess and model data for training and testing. By using this class, the user can easily preprocess the data, fit the model, and save the preprocessor and model for future testing.
    """

    def __init__(self):
        # Initialize some parameters.
        self.model_filename = "model.sav"
        self.preprocessor_filename = "preprocessor.sav"
        self.model = Model()
        self.preprocessor = Preprocessor()

    def run(self, data, test=False):
        if test:
            # Model and Preprocessor loading process.
            self.model = joblib.load(self.model_filename)
            self.preprocessor = joblib.load(self.preprocessor_filename)

            # Preprocessing and get predictions
            X = self.preprocessor.transform(data)
            
            predictions = self.model.predict(X)[:, 1]
            predictions = predictions.tolist()

            threshold = self.model.threshold

            # Make dictionaries for saving.
            json_file = {
                "predict_probas": predictions, 
                "threshold": threshold
            }

            # Saving JSON file.
            with open("predictions.json", "w") as f:
                json.dump(json_file, f)

        else:
            # Preprocessor and model fitting.
            self.preprocessor.fit(data)

            X, y = self.preprocessor.transform(data)

            self.model.fit(X, y)

            # Saving fitted model and preprocessor.
            joblib.dump(self.model, self.model_filename)
            joblib.dump(self.preprocessor, self.preprocessor_filename)


def main():
    """
    main()
    --------------------
    The main() function serves as the entry point of the program.

    It uses the argparse module to parse command line arguments and the preprocessor 
    module from preprocessor.py to preprocess the data. Additionally, 
    it uses the Model module from model.py to train the model.

    If the program is run in test mode, it imports the pre-trained 
    model and generates predictions using the imported model.
    """

    # Define argparse.ArgumentParse object for argument manipulating
    parser = argparse.ArgumentParser(
        description="""
        It defines two arguments that can be passed to the program:

        1. "--data_path": a required argument of type 
            string that specifies the path to the data file.

        2. "--inference": an optional argument of type string 
            that activates test mode if set to "True".
        """,    
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Add --data_path and --inference arguments to parser
    parser.add_argument("--data_path", type=str, help="Path to data file.", required=True)
    parser.add_argument("--inference", type=str, help="Test mode activation", required=False, default=False)

    # --inference -> default = False.
    # By default activates train mode.

    # Get arguments as dictionary from parser
    args = parser.parse_args() # returns dictionary-like object

    possible_falses = ["0", "false", "False"]

    path_of_data = args.data_path
    test_mode = args.inference not in possible_falses
 

    # Reading CSV file
    DataFrame = pd.read_csv(path_of_data)

    # Pipeline running
    pipeline = Pipeline()
    pipeline.run(DataFrame , test=test_mode)



if __name__ == "__main__":
    main()