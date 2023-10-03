from models.mlp import MLP
from models.rfc import RFC
from models.tnc import TNC
from models.xgb import XGB
from models.ftt import FTT

from utils import get_model_parameters

from data.load_data import get_test_header, get_test_species, get_eva_to_gbif_species, get_le_species, get_altitudes, get_ohe_country, get_countries, get_ohe_ecoregion, get_ecoregions, get_ohe_dune, get_dunes, get_ohe_coast, get_coasts, get_le_header
from data.preprocess_data import add_gbif_normalization, add_species_encoding, add_input_data, add_longitude, add_latitude, add_altitude, add_country, add_ecoregion, add_dune, add_coast, add_species_decoding
from data.save_data import set_test_data, set_test_header, set_test_species

AVAILABLE_PREDICTIONS = {  # Dictionary of available models for predictions
    "mlp": MLP, 
    "rfc": RFC,
    "tnc": TNC,
    "xgb": XGB,
    "ftt": FTT
}

class Prediction:
    """
    Pipeline class for habitats prediction.
    """
    def run(self, args):
        """
        Run the 'Prediction' pipeline of the framework.

        Args:
            args (argparse.Namespace): Parsed command-line arguments.
        """
        test_header = get_test_header(args)  # Retrieve test header data
        test_species = get_test_species(args, test_header)  # Retrieve test species data
        if args.gbif_normalization:
            eva_to_gbif_species = get_eva_to_gbif_species(args)  # Retrieve GBIF species mapping
            test_species, _ = add_gbif_normalization(test_species, eva_to_gbif_species)  # Normalize test species names using GBIF
        le_species = get_le_species(args)  # Retrieve species encoder
        le_header = get_le_header(args)  # Retrieve header encoder
        test_species, _ = add_species_encoding(test_species, le_species)  # Encode test species labels
        test_header = test_header.reset_index(drop=True)  # Reset the index
        test_species = test_species.reset_index(drop=True)  # Reset the index
        X = add_input_data(test_header, test_species, le_species)  # Add input data (X)
        if args.location_features:
            X = add_longitude(X, test_header)  # Add longitudes
            X = add_latitude(X, test_header)  # Add latitudes
            altitudes = get_altitudes(args)  # Retrieve altitudes data
            X, test_header = add_altitude(X, test_header, altitudes)  # Add altitudes
            ohe_country = get_ohe_country(args)  # Retrieve the countries encoder
            countries = get_countries(args)  # Retrieve countries data
            X, _, _, test_header, test_species = add_country(args, X, None, test_header, test_species, countries, ohe_country)  # Add countries
            ohe_ecoregion = get_ohe_ecoregion(args)  # Retrieve the ecoregions encoder
            ecoregions = get_ecoregions(args)  # Retrieve ecoregions data
            X, _, test_header = add_ecoregion(X, test_header, ecoregions, ohe_ecoregion)  # Add ecoregions
            ohe_dune = get_ohe_dune(args)  # Retrieve the dunes encoder
            dunes = get_dunes(args)  # Retrieve dunes data
            X, _, test_header = add_dune(X, test_header, dunes, ohe_dune)  # Add dunes
            ohe_coast = get_ohe_coast(args)  # Retrieve the coasts encoder
            coasts = get_coasts(args)  # Retrieve coasts data
            X, _, test_header = add_coast(X, test_header, coasts, ohe_coast)  # Add coasts
        set_test_data(args, X)  # Save the test data
        test_species = add_species_decoding(test_species, le_species)  # Decode test species labels
        set_test_header(args, test_header)  # Save the test header data
        set_test_species(args, test_species)  # Save the test species data
        
        model_parameters = get_model_parameters(args, X.shape[1], len(le_header.classes_))  # Retrieve model parameters

        model = AVAILABLE_PREDICTIONS[args.model](**model_parameters)  # Initialize the prediction model
        model.run(args, X, None, le_species, le_header)  # Run the prediction model
