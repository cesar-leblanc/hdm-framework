from data.load_data import get_eunis_habitats, get_eva_header, get_eva_species, get_altitudes, get_ecoregions, get_dunes, get_coasts, get_countries
from data.preprocess_data import add_header_encoding, add_gbif_normalization, add_species_encoding, add_spatial_split, add_input_data, add_target_values, add_longitude, add_latitude, add_altitude, add_country, add_ecoregion, add_dune, add_coast, add_header_decoding, add_species_decoding
from data.save_data import set_le_header, set_eva_to_gbif_species, set_le_species, set_split_assignment, set_ohe_country, set_ohe_ecoregion, set_ohe_dune, set_ohe_coast, set_eva_header, set_eva_species, set_input_data, set_target_values

class Dataset:
    """
    Pipeline class for dataset preparation.
    """
    def run(self, args):
        """
        Run the 'Dataset' pipeline of the framework.

        Args:
            args (argparse.Namespace): Parsed command-line arguments.
        """
        eunis_habitats = get_eunis_habitats(args)  # Retrieve the current list of EUNIS habitats
        eva_header = get_eva_header(args, eunis_habitats)  # Retrieve header data
        eva_header, le_header = add_header_encoding(eva_header)  # Encode target labels with value between 0 and n_classes-1
        set_le_header(args, le_header)  # Save the header encoder
        eva_species = get_eva_species(args, eva_header)  # Retrieve the species data
        if args.gbif_normalization:
            eva_species, eva_to_gbif_species = add_gbif_normalization(eva_species, None)  # Normalize species names using GBIF
            set_eva_to_gbif_species(args, eva_to_gbif_species)  # Save the mapping dictionary 
        eva_species, le_species = add_species_encoding(eva_species, None)  # Encode species labels
        set_le_species(args, le_species)  # Save the species encoder
        eva_header = eva_header[eva_header['PlotObservationID'].isin(eva_species['PlotObservationID'])]  # Filter header and species based on common PlotObservationID
        eva_header = eva_header.reset_index(drop=True)  # Reset the index
        eva_species = eva_species.reset_index(drop=True)  # Reset the index
        split_assignment, eva_header = add_spatial_split(args, eva_header)  # Perform spatial split assignment
        set_split_assignment(args, split_assignment)  # Save the split assignments
        X = add_input_data(eva_header, eva_species, le_species)  # Add input data (X)
        y = add_target_values(eva_header)  # Add target values (y)
        if args.location_features:
            X = add_longitude(X, eva_header)  # Add longitudes
            X = add_latitude(X, eva_header)  # Add latitudes
            altitudes = get_altitudes(args)  # Retrieve altitudes data
            X, eva_header = add_altitude(X, eva_header, altitudes)
            countries = get_countries(args)  # Retrieve countries data
            X, y, ohe_country, eva_header, eva_species = add_country(args, X, y, eva_header, eva_species, countries, None)  # Add countries
            set_ohe_country(args, ohe_country)  # Save the countries encoder
            ecoregions = get_ecoregions(args)  # Retrieve ecoregions data
            X, ohe_ecoregion, eva_header = add_ecoregion(X, eva_header, ecoregions, None)  # Add ecoregions
            set_ohe_ecoregion(args, ohe_ecoregion)  # Save the ecoregions encoder
            dunes = get_dunes(args)  # Retrieve dunes data
            X, ohe_dune, eva_header = add_dune(X, eva_header, dunes, None)  # Add dunes
            set_ohe_dune(args, ohe_dune)  # Save the dunes encoder
            coasts = get_coasts(args)  # Retrieve coasts data
            X, ohe_coast, eva_header = add_coast(X, eva_header, coasts, None)  # Add coasts
            set_ohe_coast(args, ohe_coast)   # Save the coasts encoder
        eva_species = add_species_decoding(eva_species, le_species)  # Decode species labels
        eva_header = add_header_decoding(eva_header, le_header)  # Decode header labels
        set_eva_header(args, eva_header)   # Save the header data
        set_eva_species(args, eva_species)  # Save the species data
        set_input_data(args, X)  # Save the input data
        set_target_values(args, y)  # Save the target values
