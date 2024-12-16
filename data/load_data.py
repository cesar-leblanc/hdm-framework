import scipy.sparse
import numpy as np
import pickle
import pandas as pd
import geopandas as gpd
import shapely
import itertools
import rioxarray
import pyproj
import torch
import joblib

def get_input_data(args):
    """
    Load the input data from a compressed sparse matrix file.

    Args:
        args (argparse.Namespace): An object containing the necessary arguments.

    Returns:
        scipy.sparse._csr.csr_matrix: The input data.
    """
    input_data_filepath = args.data_filepath + 'input_data.npz'  # Filepath of the input data
    X = scipy.sparse.load_npz(input_data_filepath)  # Load the input data from the file
    return X

def get_target_values(args):
    """
    Load the target values from a NumPy file.

    Args:
        args (argparse.Namespace): An object containing the necessary arguments.

    Returns:
        numpy.ndarray: The target values.
    """
    target_values_filepath = args.data_filepath + 'target_values.npy'  # Filepath of the target values
    y = np.load(target_values_filepath)  # Load the target values from the file
    return y

def get_le_species(args):
    """
    Load the species label encoder from a pickle file.

    Args:
        args (argparse.Namespace): An object containing the necessary arguments.

    Returns:
        sklearn.preprocessing._label.LabelEncoder: The loaded species label encoder object.
    """
    le_species_filepath = args.data_filepath + 'le_species.pkl'  # Filepath of the species label encoder
    with open(le_species_filepath, 'rb') as f:  # Open the file in binary mode
        le_species = pickle.load(f)  # Load the species label encoder from the file
    return le_species

def get_le_header(args):
    """
    Load the header label encoder from a pickle file.

    Args:
        args (argparse.Namespace): An object containing the necessary arguments.

    Returns:
        sklearn.preprocessing._label.LabelEncoder: The loaded header label encoder object.
    """
    le_header_filepath = args.data_filepath + 'le_header.pkl'  # Filepath of the header label encoder
    with open(le_header_filepath, 'rb') as f:  # Open the file in binary mode
        le_header = pickle.load(f)  # Load the header label encoder from the file
    return le_header

def get_split_assignments(args):
    """
    Load the split assignments from a NumPy array file.

    Args:
        args (argparse.Namespace): An object containing the necessary arguments.

    Returns:
        numpy.ndarray: The loaded split assignments.
    """
    split_assignments_filepath = args.data_filepath + 'split_assignments.npy'  # Filepath of the split assignments
    split_assignments = np.load(split_assignments_filepath)  # Load the split assignments from the file
    return split_assignments

def get_eunis_habitats(args):
    """
    Load the EUNIS habitats from an Excel file and filter them based on specific criteria.

    Args:
        args (argparse.Namespace): An object containing the necessary arguments.

    Returns:
        list: The filtered EUNIS habitats.
    """
    eunis_habitats_filepath = args.datasets_filepath + 'eunis_habitats.xlsx'  # Filepath of the EUNIS habitats Excel file
    eunis_habitats = pd.read_excel(eunis_habitats_filepath)  # Read the Excel file into a pandas DataFrame
    eunis_habitats = eunis_habitats['EUNIS 2020 code'].values.tolist()  # Extract the 'EUNIS 2020 code' column as a list
    eunis_habitats = [habitat for habitat in eunis_habitats if (not habitat.startswith('MA2') and len(habitat) == args.level) or (habitat.startswith('MA2') and len(habitat) == args.level + 2)]  # Filter the habitats based on their code length and prefixes
    return eunis_habitats

def get_eva_header(args, eunis_habitats):
    """
    Load and preprocess the eva_header data, filtering and transforming it based on specific criteria.

    Args:
        args (argparse.Namespace): An object containing the necessary arguments.
        eunis_habitats (list): A list of EUNIS habitats to filter the eva_header data.

    Returns:
        pandas.core.frame.DataFrame: The preprocessed eva_header data.
    """
    eva_header_filepath = args.datasets_filepath + 'eva_header.csv'  # Construct the filepath to the eva_header.csv file
    header_columns = ['PlotObservationID', 'Cover abundance scale', 'Date of recording', 'Expert System', 'Longitude', 'Latitude']  # Define the column names of the eva_header dataframe
    header_types = {'PlotObservationID': int, 'Cover abundance scale': str, 'Date of recording': int, 'Expert System': str, 'Longitude': float, 'Latitude': float}  # Define the data types of the eva_header dataframe
    eva_header = pd.read_csv(eva_header_filepath, sep='\t', usecols=header_columns, dtype=str, na_values='?')  # Read in the eva_header.csv file as a pandas dataframe
    eva_header['Cover abundance scale'] = eva_header['Cover abundance scale'].fillna('0')  # Replace missing values in the 'Cover abundance scale' column with '0'
    eva_header['Date of recording'] = eva_header['Date of recording'].fillna('0')  # Replace missing values in the 'Date of recording' column with '0'
    eva_header = eva_header.dropna()  # Remove any rows containing missing values
    eva_header = eva_header[eva_header['Cover abundance scale'] != 'Presence/Absence']  # Remove any rows where the 'Cover abundance scale' column has a value of 'Presence/Absence'
    eva_header["Date of recording"] = eva_header["Date of recording"].str.replace(':', '', regex=True).apply(lambda x: int(x[-4:]))  # Extract the year from the 'Date of recording' column and convert it to an integer
    eva_header = eva_header.loc[eva_header['Date of recording'] >= args.min_year]  # Remove any rows where the year in the 'Date of recording' column is less than the minimum year specified by the 'args' argument
    split_eva_header = eva_header['Expert System'].str.split(',', expand=True)  # Create a new dataframe for each split value in the 'Expert System' column
    split_eva_header = split_eva_header.stack().reset_index(level=1, drop=True).rename('Expert System')  # Stack the resulting dataframes vertically and rename the columns
    eva_header = eva_header.drop('Expert System', axis=1).join(split_eva_header)  # Join the original dataframe with the split dataframe on the index
    eva_header = eva_header.loc[eva_header['Expert System'].isin(eunis_habitats)]  # Filter out rows where the 'Expert System' column is not in the list of EUNIS habitats
    counts = eva_header['Expert System'].value_counts()  # Count the number of occurrences of each unique value in the 'Expert System' column
    keep = counts[counts >= args.occurrences].index.tolist()  # Keep only the values that appear 10 times or more
    eva_header = eva_header[eva_header['Expert System'].isin(keep)]  # Filter the dataframe to keep only the rows with values in the 'Expert System' column that appear 10 times or more
    eva_header = eva_header.astype(header_types)  # Convert the data types of the dataframe columns to the specified types
    eva_header = eva_header.round({'Longitude': 4, 'Latitude': 4})  # Round the 'Longitude' and 'Latitude' columns
    eva_header = eva_header.drop(['Cover abundance scale', 'Date of recording'], axis=1)  # Drop 'Cover abundance scale' and 'Date of recording' columns
    return eva_header  # Return the filtered dataframe

def get_test_header(args):
    """
    Load and preprocess the test header data, filtering and transforming it based on specific criteria.

    Args:
        args (argparse.Namespace): An object containing the necessary arguments.

    Returns:
        pandas.core.frame.DataFrame: The preprocessed test header data.
    """
    test_header_filepath = args.datasets_filepath + 'test_header.csv'  # Filepath of the test header CSV file
    header_columns = ['PlotObservationID', 'Longitude', 'Latitude']  # Specify the columns to be read from the CSV file
    header_types = {'PlotObservationID': int, 'Longitude': float, 'Latitude': float}  # Specify the data types for each column
    test_header = pd.read_csv(test_header_filepath, sep='\t', usecols=header_columns, dtype=str, na_values='?')  # Read the CSV file into a pandas DataFrame, using specified column names, data types, and handling missing values
    test_header = test_header.dropna()  # Drop rows with missing values from the DataFrame
    test_header = test_header.astype(header_types)  # Convert the data types of columns in the DataFrame to the specified types
    test_header = test_header.round({'Longitude': 4, 'Latitude': 4})  # Round the values in the 'Longitude' and 'Latitude' columns to 4 decimal places
    return test_header

def get_eva_species(args, eva_header):
    """
    Load and preprocess the eva_species data, filtering and transforming it based on specific criteria.

    Args:
        args (argparse.Namespace): An object containing the necessary arguments.
        eva_header (pandas.core.frame.DataFrame): The preprocessed eva_header data.

    Returns:
        pandas.core.frame.DataFrame: The preprocessed eva_species data.
    """
    eva_species_filepath = args.datasets_filepath + 'eva_species.csv'  # Filepath of the eva_species CSV file
    species_columns = ['PlotObservationID', 'Matched concept', 'Cover %']  # Specify the columns to be read from the CSV file
    species_types = {'PlotObservationID': int, 'Matched concept': str, 'Cover %': float}  # Specify the data types for each column
    eva_species = pd.read_csv(eva_species_filepath, delimiter='\t', usecols=species_columns, dtype=str)  # Read the CSV file into a pandas DataFrame, using specified column names, data types, and delimiter
    eva_species = eva_species.dropna()  # Drop rows with missing values from the DataFrame
    eva_species = eva_species.astype(species_types)  # Convert the data types of columns in the DataFrame to the specified types
    eva_species = eva_species[eva_species['PlotObservationID'].isin(eva_header['PlotObservationID'])]  # Filter the eva_species DataFrame to keep only the rows with PlotObservationID present in the eva_header DataFrame
    eva_species = eva_species.groupby(['PlotObservationID', 'Matched concept']).sum().reset_index()  # Group the DataFrame by 'PlotObservationID' and 'Matched concept', and sum the 'Cover %' values
    eva_species = eva_species[eva_species["Cover %"] > 0]  # Filter the DataFrame to keep only the rows with 'Cover %' greater than 0
    counts = eva_species['Matched concept'].value_counts()  # Count the number of occurrences of each unique value in the 'Matched concept' column
    keep = counts[counts >= args.occurrences].index.tolist()  # Keep only the values that appear 'args.occurrences' times or more
    eva_species = eva_species[eva_species['Matched concept'].isin(keep)]  # Filter the dataframe to keep only the rows with values in the 'Matched concept' column that appear 10 times or more
    return eva_species

def get_test_species(args, test_header):
    """
    Load and preprocess the test_species data, filtering and transforming it based on specific criteria.

    Args:
        args (argparse.Namespace): An object containing the necessary arguments.
        test_header (pandas.core.frame.DataFrame): The preprocessed test_header data.

    Returns:
        pandas.core.frame.DataFrame: The preprocessed test_species.
    """
    test_species_filepath = args.datasets_filepath + 'test_species.csv'  # Filepath of the test_species CSV file
    species_columns = ['PlotObservationID', 'Matched concept', 'Cover %']  # Specify the columns to be read from the CSV file
    species_types = {'PlotObservationID': int, 'Matched concept': str, 'Cover %': float}  # Specify the data types for each column
    test_species = pd.read_csv(test_species_filepath, delimiter='\t', usecols=species_columns, dtype=str)  # Read the CSV file into a pandas DataFrame, using specified column names, data types, and delimiter
    test_species = test_species.dropna()  # Drop rows with missing values from the DataFrame
    test_species = test_species.astype(species_types)  # Convert the data types of columns in the DataFrame to the specified types
    test_species = test_species[test_species['PlotObservationID'].isin(test_header['PlotObservationID'])]  # Filter the test_species DataFrame to keep only the rows with PlotObservationID present in the test_header DataFrame
    test_species = test_species.groupby(['PlotObservationID', 'Matched concept']).sum().reset_index()  # Group the DataFrame by 'PlotObservationID' and 'Matched concept', and sum the 'Cover %' values
    test_species = test_species[test_species["Cover %"] > 0]  # Filter the DataFrame to keep only the rows with 'Cover %' greater than 0
    return test_species

def get_eva_to_gbif_species(args):
    """
    Load the eva_to_gbif_species data from a pickle file.

    Args:
        args (argparse.Namespace): An object containing the necessary arguments.

    Returns:
        dict: The loaded eva_to_gbif_species data.
    """
    eva_to_gbif_species_filepath = args.data_filepath + 'eva_to_gbif_species.pkl'  # Filepath of the eva_to_gbif_species pickle file
    with open(eva_to_gbif_species_filepath, 'rb') as f:  # Open the pickle file in binary mode for reading
        eva_to_gbif_species = pickle.load(f)  # Load the contents of the pickle file
    return eva_to_gbif_species

def get_altitudes(args):
    """
    Load and preprocess the digital elevation model (DEM) data.

    Args:
        args (argparse.Namespace): An object containing the necessary arguments.

    Returns:
        xarray.core.dataarray.DataArray: The processed DEM data.
    """
    altitudes_filepath = args.datasets_filepath + 'digital_elevation_model.tif'  # Construct the file path of the DEM file
    altitudes = rioxarray.open_rasterio(altitudes_filepath)  # Load the DEM data as an xarray.DataArray object
    altitudes = altitudes.rio.reproject("EPSG:3857")  # Reproject the DEM data to EPSG:3857 coordinate system
    altitudes.attrs['_FillValue'] = 0  # Set the fill value of the DEM to 0
    altitudes = altitudes.where(altitudes != 32767, 0)  # Replace nodata values with 0
    return altitudes  # Return the processed DEM data

def get_countries(args):
    """
    Load and preprocess the country geometries.

    Args:
        args (argparse.Namespace): An object containing the necessary arguments.

    Returns:
        geopandas.geodataframe.GeoDataFrame: The processed country geometries.
    """
    countries_columns = ['COUNTRY', 'geometry']  # Specify the columns to keep in the country geometries
    countries_filepath = args.datasets_filepath + 'world_countries.shp'  # Construct the file path of the country shapefile
    countries = gpd.read_file(countries_filepath)  # Load the country geometries as a GeoDataFrame object
    countries = countries[countries_columns]  # Select only the specified columns
    countries = countries.to_crs(crs=3857)  # Reproject the country geometries to EPSG:3857 coordinate system
    return countries  # Return the processed country geometries

def get_ecoregions(args):
    """
    Load and preprocess the ecoregion geometries.

    Args:
        args (argparse.Namespace): An object containing the necessary arguments.

    Returns:
        geopandas.geodataframe.GeoDataFrame: The processed ecoregion geometries.
    """
    ecoregions_columns = ['ECO_ID', 'geometry']  # Specify the columns to keep in the ecoregion geometries
    ecoregions_filepath = args.datasets_filepath + 'ecoregions.shp'  # Construct the file path of the ecoregion shapefile
    ecoregions = gpd.read_file(ecoregions_filepath)  # Load the ecoregion geometries as a GeoDataFrame object
    ecoregions = ecoregions[ecoregions_columns]  # Select only the specified columns
    ecoregions = ecoregions.to_crs(crs=3857)  # Reproject the ecoregion geometries to EPSG:3857 coordinate system
    return ecoregions  # Return the processed ecoregion geometries

def get_dunes(args):
    """
    Load and preprocess the dune geometries.

    Args:
        args (argparse.Namespace): An object containing the necessary arguments.

    Returns:
        shapely.geometry.multipolygon.MultiPolygon: The merged dune polygon.
    """
    dunes_columns = ['CODE', 'geometry']  # Specify the columns to keep in the dune geometries
    dunes_filepath = args.datasets_filepath + 'vegetation.shp'  # Construct the file path of the dune shapefile
    dunes = gpd.read_file(dunes_filepath)  # Load the dune geometries as a GeoDataFrame object
    dunes = dunes[dunes_columns]  # Select only the specified columns
    dunes = dunes[dunes['CODE'].isin([f'P{i}' for i in range(1,17)])]  # Filter the dunes to keep only those with CODE values matching P1 to P16
    dunes = dunes.to_crs(crs=3857)  # Reproject the dune geometries to EPSG:3857 coordinate system
    dunes = shapely.ops.unary_union(dunes['geometry'].tolist())  # Merge the dune geometries into a single polygon using unary union
    return dunes  # Return the merged dune polygon

def get_coasts(args):
    """
    Load and preprocess the coastal geometries.

    Args:
        args (argparse.Namespace): An object containing the necessary arguments.

    Returns:
        geopandas.geodataframe.GeoDataFrame: The dissolved coastal geometries.
    """
    categories = dict()  # Create an empty dictionary
    categories['ARC_COAST'] = ["White Sea", "Arctic Ocean", "Barentsz Sea", "Kara Sea", "Laptev Sea", "Norwegian Sea"]  # Select all the bodies of water that are part of the Arctic Ocean
    categories['ATL_COAST'] = ["Strait of Gibraltar", "Bristol Channel", "Irish Sea and St. George's Channel", "Inner Seas off the West Coast of Scotland", "North Atlantic Ocean", "Bay of Biscay", "Celtic Sea", "English Channel", "North Sea", "Skagerrak"]  # Select all the bodies of water that are part of the Atlantic Ocean
    categories['BAL_COAST'] = ["Gulf of Riga", "Baltic Sea", "Gulf of Finland", "Gulf of Bothnia", "Kattegat"]  # Select all the bodies of water that are part of the Baltic Sea
    categories['BLA_COAST'] = ["Black Sea", "Sea of Azov"]  # Select all the bodies of water that are part of the Black Sea
    categories['MED_COAST'] = ["Alboran Sea", "Ionian Sea", "Tyrrhenian Sea", "Adriatic Sea", "Mediterranean Sea - Eastern Basin", "Aegean Sea", "Sea of Marmara", "Balearic (Iberian Sea)", "Mediterranean Sea - Western Basin", "Ligurian Sea"]  # Select all the bodies of water that are part of the Mediterranean Sea
    coasts_columns = ['NAME', 'geometry']  # Specify the columns to keep in the coast geometries
    coasts_filepath = args.datasets_filepath + 'world_seas.shp'  # Construct the file path of the coast shapefile
    coasts = gpd.read_file(coasts_filepath)  # Load the coastal geometries as a GeoDataFrame object
    coasts = coasts[coasts_columns]  # Select only the specified columns
    coasts = coasts.to_crs(crs=3857)  # Reproject the coastal geometries to EPSG:3857 coordinate system
    coasts["geometry"] = coasts.boundary  # Extract the boundary of the coastal geometries
    for i in range(len(coasts)):  # Iterate through the bodies of water
        if coasts.loc[i, 'NAME'] in list(itertools.chain(*categories.values())):  # Check if the body of water is part of the Arctic Ocean, the Atlantic Ocean, the Baltic Sea, the Black Sea or the Mediterranean Sea
            coasts.loc[i, 'NAME'] = list({key: value for key, value in categories.items() if coasts.loc[i, 'NAME'] in value}.keys())[0]  # Replace the body of water by its larger part
    coasts = coasts.loc[coasts.loc[:, 'NAME'].isin(list(categories.keys()))]  # Filter the coastal geometries to keep only those with valid names
    coasts = coasts.dissolve(by='NAME')  # Dissolve the coastal geometries based on the 'NAME' column
    return coasts

def get_eunis_red_list_crosswalks(args):
    """
    Load and process the EUNIS to Red List crosswalk data.

    Args:
        args (argparse.Namespace): An object containing the necessary arguments.

    Returns:
        dict: A dictionary mapping EUNIS codes to Red List codes.
    """
    eunis_habitats_filepath = args.datasets_filepath + 'eunis_habitats.xlsx'  # Construct the filepath of the EUNIS habitats Excel file
    eunis_habitats = pd.read_excel(eunis_habitats_filepath)  # Read the Excel file into a pandas DataFrame
    eunis_red_list_crosswalks = dict(zip(eunis_habitats['EUNIS 2020 code'], eunis_habitats['Red List code']))  # Create a dictionary mapping EUNIS codes to Red List codes
    eunis_red_list_crosswalks = {key: eunis_red_list_crosswalks[key] for key in eunis_red_list_crosswalks if type(eunis_red_list_crosswalks[key]) is str}  # Filter the crosswalk dictionary to keep only string values
    return eunis_red_list_crosswalks

def get_endangered_red_list_habitats(args):
    """
    Load and process the endangered Red List habitats data.

    Args:
        args (argparse.Namespace): An object containing the necessary arguments.

    Returns:
        dict: A dictionary mapping Red List project habitat codes to endangered habitat categories.
    """
    red_list_habitats_filepath = args.datasets_filepath + 'red_list_habitats.xlsx'  # Construct the filepath of the Red List habitats Excel file
    red_list_habitats = pd.read_excel(red_list_habitats_filepath, sheet_name='Terrestrial')  # Read the Excel file into a pandas DataFrame
    red_list_habitats = dict(zip(red_list_habitats['RL project habitat code'], red_list_habitats['Overall category EU28']))  # Create a dictionary mapping Red List project habitat codes to overall categories
    endangered_red_list_habitats = {key: val for key, val in red_list_habitats.items() if 'Endangered' in val}  # Filter the habitats dictionary to keep only endangered categories
    return endangered_red_list_habitats

def get_endangered_eunis_habitats(eunis_red_list_crosswalks, endangered_red_list_habitats, le_header):
    """
    Get the encoded labels of endangered EUNIS habitats based on crosswalks and label encoder.

    Args:
        dict: A dictionary mapping EUNIS codes to Red List codes.
        dict: A dictionary mapping Red List project habitat codes to endangered habitat categories.
        sklearn.preprocessing._label.LabelEncoder: The label encoder object for encoding habitat labels.

    Returns:
        list: Encoded labels of endangered EUNIS habitats.
    """
    endangered_habitats_with_crosswalks = {key: value for key, value in endangered_red_list_habitats.items() if key in eunis_red_list_crosswalks.values()}  # Filter the endangered Red List habitats to keep only those with crosswalks
    endangered_eunis_habitats = [key for key in eunis_red_list_crosswalks.keys() if eunis_red_list_crosswalks[key] in endangered_habitats_with_crosswalks.keys()]  # Find the EUNIS codes of the endangered habitats
    endangered_eunis_habitats = [key for key in endangered_eunis_habitats if key in le_header.classes_]  # Filter the EUNIS codes to keep only those present in the label encoder classes
    endangered_eunis_habitats = le_header.transform(endangered_eunis_habitats)  # Encode the EUNIS habitat labels
    return endangered_eunis_habitats

def get_ohe_country(args):
    """
    Load the one-hot encoding representation of countries.

    Args:
        args (argparse.Namespace): An object containing the necessary arguments.

    Returns:
        sklearn.preprocessing._encoders.OneHotEncoder: The one-hot encoding representation of countries.
    """
    ohe_country_filepath = args.data_filepath + 'ohe_country.pkl'  # Filepath of the one-hot encoding pickle file
    with open(ohe_country_filepath, 'rb') as f:  # Open the pickle file in binary mode for reading
        ohe_country = pickle.load(f)  # Load the contents of the pickle file
    return ohe_country

def get_ohe_ecoregion(args):
    """
    Load the one-hot encoding representation of ecoregions.

    Args:
        args (argparse.Namespace): An object containing the necessary arguments.

    Returns:
        sklearn.preprocessing._encoders.OneHotEncoder: The one-hot encoding representation of ecoregions.
    """
    ohe_ecoregion_filepath = args.data_filepath + 'ohe_ecoregion.pkl'  # Filepath of the one-hot encoding pickle file
    with open(ohe_ecoregion_filepath, 'rb') as f:  # Open the pickle file in binary mode for reading
        ohe_ecoregion = pickle.load(f)  # Load the contents of the pickle file
    return ohe_ecoregion

def get_ohe_dune(args):
    """
    Load the one-hot encoding representation of dunes.

    Args:
        args (argparse.Namespace): An object containing the necessary arguments.

    Returns:
        sklearn.preprocessing._encoders.OneHotEncoder: The one-hot encoding representation of dunes.
    """
    ohe_dune_filepath = args.data_filepath + 'ohe_dune.pkl'  # Filepath of the one-hot encoding pickle file
    with open(ohe_dune_filepath, 'rb') as f:  # Open the pickle file in binary mode for reading
        ohe_dune = pickle.load(f)  # Load the contents of the pickle file
    return ohe_dune

def get_ohe_coast(args):
    """
    Load the one-hot encoding representation of coast categories.

    Args:
        args (argparse.Namespace): An object containing the necessary arguments.

    Returns:
        sklearn.preprocessing._encoders.OneHotEncoder: The one-hot encoding representation of coast categories.
    """
    ohe_coast_filepath = args.data_filepath + 'ohe_coast.pkl'  # Filepath of the one-hot encoding pickle file
    with open(ohe_coast_filepath, 'rb') as f:  # Open the pickle file in binary mode for reading
        ohe_coast = pickle.load(f)  # Load the contents of the pickle file
    return ohe_coast

def get_scaler(args):
    """
    Load the scaler object used for feature scaling.

    Args:
        args (argparse.Namespace): An object containing the necessary arguments.

    Returns:
        sklearn.preprocessing._data.StandardScaler: The scaler object used for feature scaling.
    """
    scaler_filepath = args.data_filepath + 'scaler.pkl'  # Filepath of the scaler pickle file
    with open(scaler_filepath, 'rb') as f:  # Open the pickle file in binary mode for reading
        scaler = pickle.load(f)  # Load the contents of the pickle file
    return scaler

def get_mlp_model(args, model):
    """
    Load the pre-trained MLP model.

    Args:
        args (argparse.Namespace): An object containing the necessary arguments.
        model (models.mlp.MLP | torch.nn.parallel.data_parallel.DataParallel): The MLP model object.

    Returns:
        models.mlp.MLP | torch.nn.parallel.data_parallel.DataParallel: The loaded pre-trained MLP model.
    """
    mlp_filepath = args.models_filepath + 'MLP.pth'  # Filepath of the pre-trained MLP model
    model.load_state_dict(torch.load(mlp_filepath))  # Load the weights of the pre-trained model
    return model

def get_rfc_model(args, model):
    """
    Load the pre-trained Random Forest Classifier (RFC) model.

    Args:
        args (argparse.Namespace): An object containing the necessary arguments.
        models.rfc.RFC: The RFC model object.

    Returns:
        models.rfc.RFC: The loaded pre-trained RFC model.
    """
    rfc_filepath = args.models_filepath + 'RFC.joblib'  # Filepath of the pre-trained RFC model
    model = joblib.load(rfc_filepath)  # Load the pre-trained RFC model
    return model

def get_xgb_model(args, model):
    """
    Load the pre-trained XGBoost model.

    Args:
        args (argparse.Namespace): An object containing the necessary arguments.
        models.xgb.XGB: The XGBoost model object.

    Returns:
        models.xgb.XGB: The loaded pre-trained XGBoost model.
    """
    xgb_filepath = args.models_filepath + 'XGB.json'  # Filepath of the pre-trained XGBoost model
    model.load_model(xgb_filepath)  # Load the pre-trained XGBoost model
    return model

def get_tnc_model(args, model):
    """
    Load the pre-trained TNC model.

    Args:
        args (argparse.Namespace): An object containing the necessary arguments.
        models.tnc.TNC: The TNC model object.

    Returns:
        models.tnc.TNC: The loaded pre-trained TNC model.
    """
    tnc_filepath = args.models_filepath + 'TNC.zip'  # Construct the filepath of the TNC model
    model.load_model(tnc_filepath)  # Load the pre-trained TNC model from the file path
    return model

def get_ftt_model(args, model):
    """
    Load the pre-trained FTT model.

    Args:
        args (argparse.Namespace): An object containing the necessary arguments.
        models.ftt.FTT: The FTT model object.

    Returns:
        models.ftt.FTT: The loaded pre-trained FTT model.
    """
    ftt_filepath = args.models_filepath + 'FTT.pth'  # Construct the filepath of the FTT model
    model.load_state_dict(torch.load(ftt_filepath))  # Load the pre-trained FTT model from the file path
    return model
