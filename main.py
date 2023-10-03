import argparse

from cli import add_all_parsers
from utils import disable_warnings, print_parameters, set_seed, disable_bytecode_generation

from pipelines.check import Check
from pipelines.evaluation import Evaluation
from pipelines.dataset import Dataset
from pipelines.training import Training
from pipelines.prediction import Prediction

AVAILABLE_PIPELINES = {  # Dictionary of all available pipelines
    "check": Check,
    "dataset": Dataset,
    "evaluation": Evaluation,
    "training": Training,
    "prediction": Prediction
}

def run(args):
    """
    Run the framework for Habitat Distribution Modeling.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    disable_warnings(args)  # Ignore all warnings during the code's execution
    disable_bytecode_generation(args)  # Disable bytecode generation to improve startup performance
    print_parameters(args)  # Print the framework parameters
    set_seed(args)  # Set the random seed for reproducibility
    pipeline = AVAILABLE_PIPELINES[args.pipeline]()  # Create an instance of the selected pipeline
    pipeline.run(args)  # Run the pipeline with the provided arguments

if __name__ == "__main__":
    parser = argparse.ArgumentParser()  # Create an argument parser
    add_all_parsers(parser)  # Add parsers for all the available options
    args = parser.parse_args()  # Parse the command-line arguments
    run(args)  # Run the framework with the parsed arguments
