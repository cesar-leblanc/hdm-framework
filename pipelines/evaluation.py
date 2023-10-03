from models.mlp import MLP
from models.rfc import RFC
from models.tnc import TNC
from models.xgb import XGB
from models.ftt import FTT

from utils import get_model_parameters

from data.load_data import get_input_data, get_target_values, get_le_species, get_le_header

AVAILABLE_EVALUATIONS = {  # Dictionary of available models for evaluation
    "mlp": MLP, 
    "rfc": RFC,
    "tnc": TNC,
    "xgb": XGB,
    "ftt": FTT
}

class Evaluation:
    """
    Pipeline class for parameters evaluation.
    """
    def run(self, args):
        """
        Run the 'Evaluation' pipeline of the framework.

        Args:
            args (argparse.Namespace): Parsed command-line arguments.
        """
        X = get_input_data(args)  # Retrieve input data (X)
        y = get_target_values(args)  # Retrieve target values (y)
        le_species = get_le_species(args)  # Retrieve the label encoder for species
        le_header = get_le_header(args)  # Retrieve the label encoder for header

        model_parameters = get_model_parameters(args, X.shape[1], len(le_header.classes_))  # Get model parameters based on number of input features and number of output classes
        
        model = AVAILABLE_EVALUATIONS[args.model](**model_parameters)  # Create an instance of the selected model
        model.run(args, X, y, le_species, le_header)  # Run the model's evaluation
