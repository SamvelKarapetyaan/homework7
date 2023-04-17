import argparse
import joblib
import pandas as pd

# from model import Model
# from preprocessor import Preprocessor


class Pipeline:
    def __init__(self,):
        pass
        # self.model = Model()
        # self.preprocessor = Preprocessor()

    def run(self, X, test=False):
        if test:
        # load preprocessor and model for testing
        # save results to predictions.json file 
            pass
        else:
        # call preprocessor and model for training
        # save preprocessor and model for future testing
            pass


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
    args = parser.parse_args() # {"<argument_name>": "value", ...}

    # ========================================================

    path_of_data = args.data_path
    test_mode = args.inference
 
    DataFrame = pd.read_csv(path_of_data)
    # ========================================================


    # Import best model
    best_model = "<file>.sav"
    
    load_model = joblib.load(best_model)


    pipeline = Pipeline()
    pipeline.run(best_model, test=test_mode)



if __name__ == "__main__":
    main()