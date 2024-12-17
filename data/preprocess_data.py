import numpy as np
import sklearn.preprocessing
import random
import imblearn
import collections
import pandas as pd
import requests
import tqdm
import scipy.sparse
import rtree
import shapely
import geopandas as gpd
import torch
import rioxarray
import pyproj

from utils import assign_block_ids, SparseDataset, get_available_subprocesses

from data.load_data import get_le_species, get_ohe_country, get_ohe_ecoregion, get_ohe_dune, get_ohe_coast

def add_fold_assignments(X, y, split_assignments, fold):
    """
    Split the data into training and testing sets based on fold assignments.

    Args:
        X (scipy.sparse._csr.csr_matrix): The input features.
        y (numpy.ndarray): The target labels.
        split_assignments (numpy.ndarray): The fold assignments for each data point.
        fold (int): The fold number to use as the testing set.

    Returns:
        tuple: A tuple containing the training and testing sets for both X and y.
    """
    X_train = X[split_assignments != fold]  # Select data points where split_assignments is not equal to the given fold
    X_test = X[split_assignments == fold]  # Select data points where split_assignments is equal to the given fold
    y_train = y[split_assignments != fold]  # Select corresponding target labels for the training set
    y_test = y[split_assignments == fold]  # Select corresponding target labels for the testing set
    return X_train, X_test, y_train, y_test

def add_dataloader(args, X, y, shuffle=True):
    """
    Create a DataLoader for the given input features and target labels.

    Args:
        args (argparse.Namespace): Command-line arguments.
        X (scipy.sparse._csr.csr_matrix): The input features.
        y (numpy.ndarray): The target labels.
        shuffle (bool, optional): Whether to shuffle the data. Defaults to True.

    Returns:
        torch.utils.data.DataLoader: The DataLoader object for iterating over the data in batches.
    """
    dataset = SparseDataset(X, y)  # Create a dataset from the input features and target labels
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=shuffle, num_workers=get_available_subprocesses(args))  # Create a DataLoader for the dataset with the specified batch size, shuffle, and number of workers
    return dataloader

def add_standardization(X, scaler=None):
    """
    Apply standardization to the input data.

    Args:
        X (scipy.sparse._csr.csr_matrix): The input data.
        scaler (sklearn.preprocessing._data.StandardScaler, optional): The scaler object. If None, a new StandardScaler instance will be created and fitted on the data. Defaults to None.

    Returns:
        tuple: A tuple containing the standardized data (X) and the scaler object.
    """
    if not scaler:
        scaler = sklearn.preprocessing.StandardScaler(with_mean=False)  # Create a StandardScaler instance
        scaler.fit(X)  # Compute the mean and std on the training data to be used for later scaling
    X = scaler.transform(X)  # Scale the data using the scaler
    return X, scaler

def add_normalization(X, n_columns):
    """
    Apply normalization to the input data.

    Args:
        X (scipy.sparse._csr.csr_matrix): The input data in CSR sparse matrix format.
        n_columns (int): The number of columns in the input data.

    Returns:
        scipy.sparse._csr.csr_matrix: The normalized input data in CSR sparse matrix format.
    """
    normalizer = sklearn.preprocessing.Normalizer(norm='l1')  # Create a Normalizer instance with the l1 norm
    for i in range(X.shape[0]):  # Iterate over the rows
        start, end = X.indptr[i], X.indptr[i+1]  # The indptr attribute of the CSR matrix defines the start and end indices of each row
        indices = X.indices[start:end]  # For each row, retrieve the indices of the non-zero elements using the 'indices' attribute of the CSR matrix
        data = X.data[start:end]  # Retrieve the non-zero elements of the row using the 'data' attribute of the CSR matrix
        mask = indices < n_columns  # Create a mask for the selected columns
        selected_data = data[mask]  # Use the mask to retrieve the non-zero elements of the row that correspond to the columns we want
        normalizer.fit(selected_data.reshape(1, -1))  # Do nothing and return the estimator unchanged
        selected_data = normalizer.transform(selected_data.reshape(1, -1))  # Normalize the matrix (scale each non zero row to unit norm)
        data[mask] = selected_data[0]  # Update the non-zero elements of the row in the CSR matrix
    return X

def add_dropout(X, p, n_columns):
    """
    Apply dropout to the input data.

    Args:
        X (scipy.sparse._csr.csr_matrix): The input data in CSR sparse matrix format.
        p (float): The dropout probability, indicating the proportion of non-zero elements to be set to zero.
        n_columns (int): The number of columns for species in the input data.

    Returns:
        scipy.sparse._csr.csr_matrix: The input data with dropout applied in CSR sparse matrix format.
    """
    column_indices = np.where(np.isin(X.indices, list(range(n_columns))))[0]  # Get the indices of the non-zero elements in the columns that we want
    sample_size = int(len(column_indices) * p)  # Set the sample size to p% of the total number of non-zero elements
    sample_indices = random.sample(list(column_indices), sample_size)  # Select a random subset of these indices
    X.data[sample_indices] = 0  # Set the values at the selected indices to 0 
    X.eliminate_zeros()  # Remove explicit zero elements from the matrix
    return X

def add_binarization(X, n_columns):
    """
    Apply binarization to the input data.

    Args:
        X (scipy.sparse._csr.csr_matrix): The input data in CSR sparse matrix format.
        n_columns (int): The number of columns for species in the input data.

    Returns:
        scipy.sparse._csr.csr_matrix: The input data with binarization applied in CSR sparse matrix format.
    """
    binarizer = sklearn.preprocessing.Binarizer(threshold=1)  # Create a binarizer instance with threshold=1
    column_indices = list(range(n_columns))  # Select the columns we want to binarize
    X.data[np.isin(X.indices, column_indices)] = binarizer.transform(X.data[np.isin(X.indices, column_indices)].reshape(-1, 1)).flatten()  # Binarize the data in the selected columns
    X.eliminate_zeros()  # Remove explicit zero elements from the matrix
    return X

def add_log(X, n_columns):
    """
    Apply natural logarithm to the input data.

    Args:
        X (scipy.sparse._csr.csr_matrix): The input data in CSR sparse matrix format.
        n_columns (int): The number of columns for species in the input data.

    Returns:
        scipy.sparse._csr.csr_matrix: The input data with natural logarithm applied to the selected columns in CSR sparse matrix format.
    """
    column_indices = list(range(n_columns))  # Select the columns we want to apply the natural logarithm to
    mask = np.isin(X.indices, column_indices)  # Create a mask for the selected columns
    X.data[mask] = np.log(X.data[mask])  # Use the mask to select the columns and apply the natural logarithm
    return X

def add_noise(X, n_columns):
    """
    Add noise to the input data.

    Args:
        X (scipy.sparse._csr.csr_matrix): The input data in CSR sparse matrix format.
        n_columns (int): The number of columns for species in the input data.

    Returns:
        scipy.sparse._csr.csr_matrix: The input data with noise added to the selected columns in CSR sparse matrix format.
    """
    mean_column = np.zeros(n_columns)  # Initialize mean for each column
    std_column = np.zeros(n_columns)  # Initialize standard deviation for each column
    for j in range(n_columns):  # Iterate through the columns we will modify
        mean_column[j] = np.mean(X[:, j].data)  # Calculate and store the mean of the elements in the current column
        std_column[j] = np.std(X[:, j].data)  # Calculate and store the standard deviation of the elements in the current column
    for i in range(len(X.data)):  # Iterate through the non-zero elements of X
        if X.indices[i] < n_columns:  # Check if the element is in one of the columns we want to modify
            noise = np.random.normal(mean_column[X.indices[i]], std_column[X.indices[i]])  # Generate noise value from normal distribution with mean and standard deviation calculated from the same column
            noise *= np.random.choice([-1, 1])  # Randomly choose whether to add or subtract the noise
            X.data[i] += noise  # Add the noise value to the element
            X.data[i] = np.max(X.data[i], 0)  # Ensure that the element is non-negative (clip the value at zero)
    X.eliminate_zeros()  # Eliminate explicit zeros from the sparse matrix to save memory
    return X

def add_rank(X, n_columns):
    """
    Add rank-based transformation to the input data.

    Args:
        X (scipy.sparse._csr.csr_matrix): The input data in CSR sparse matrix format.
        n_columns (int): The number of columns for species in the input data.

    Returns:
        scipy.sparse._csr.csr_matrix: The input data with rank-based transformation applied to the selected columns in CSR sparse matrix format.
    """
    for i in range(X.shape[0]):  # Iterate over the rows
        start, end = X.indptr[i], X.indptr[i+1]  # The indptr attribute of the CSR matrix defines the start and end indices of each row
        indices = X.indices[start:end]  # For each row, retrieve the indices of the non-zero elements using the 'indices' attribute of the CSR matrix
        data = X.data[start:end]  # Retrieve the non-zero elements of the row using the 'data' attribute of the CSR matrix
        mask = indices < n_columns  # Create a mask for the selected columns
        selected_data = data[mask]  # Use the mask to retrieve the non-zero elements of the row that correspond to the columns we want
        sorted_data = sorted(np.unique(selected_data), reverse=True)  # Sort the unique elements of the array in descending order
        rank_dict = {val: rank for rank, val in enumerate(sorted_data, start=1)}  # Use the sorted array to create a dictionary that maps each value to its rank
        selected_data = [1/rank_dict[val] for val in selected_data]  # Transform the original array by replacing each value with its corresponding rank from the dictionary, and dividing 1 by the rank
        data[mask] = selected_data  # Update the non-zero elements of the row in the CSR matrix
    return X

def remove_features(args, X, features):
    """
    Remove specific features from the input data.

    Args:
        X (scipy.sparse._csr.csr_matrix): The input data in CSR sparse matrix format.
        features (list): A list of features to remove from the data.

    Returns:
        scipy.sparse._csr.csr_matrix: The input data with the specified features removed in CSR sparse matrix format.
    """
    le_species = get_le_species(args)  # Get the LabelEncoder for species
    ohe_country = get_ohe_country(args)  # Get the OneHotEncoder for country
    ohe_ecoregion = get_ohe_ecoregion(args)  # Get the OneHotEncoder for ecoregion
    ohe_dune = get_ohe_dune(args)  # Get the OneHotEncoder for dune
    ohe_coast = get_ohe_coast(args)  # Get the OneHotEncoder for coast
    
    select_columns = []  # Initialize a list to store the selected columns
    
    if 'species' in features:
        select_columns += [*range(0, len(le_species.classes_), 1)]  # Add the indices of species columns to the selected columns
    if 'location' in features:
        select_columns += [*range(len(le_species.classes_), len(le_species.classes_) + 2, 1)]  # Add the indices of location columns to the selected columns    
    if 'altitude' in features:
        select_columns += [*range(len(le_species.classes_) + 2, len(le_species.classes_) + 2 + 1, 1)]  # Add the index of altitude column to the selected columns
    if 'country' in features:
        select_columns += [*range(len(le_species.classes_) + 2 + 1, len(le_species.classes_) + 2 + 1 + len(ohe_country.categories_[0]), 1)]  # Add the indices of country columns to the selected columns
    if 'ecoregion' in features:
        select_columns += [*range(len(le_species.classes_) + 2 + 1 + len(ohe_country.categories_[0]), len(le_species.classes_) + 2 + 1 + len(ohe_country.categories_[0]) + len(ohe_ecoregion.categories_[0]), 1)]  # Add the indices of ecoregion columns to the selected columns
    if 'dune' in features:
        select_columns += [*range(len(le_species.classes_) + 2 + 1 + len(ohe_country.categories_[0]) + len(ohe_ecoregion.categories_[0]), len(le_species.classes_) + 2 + 1 + len(ohe_country.categories_[0]) + len(ohe_ecoregion.categories_[0]) + len(ohe_dune.categories_[0]), 1)]  # Add the indices of dune columns to the selected columns
    if 'coast' in features:
        select_columns += [*range(len(le_species.classes_) + 2 + 1 + len(ohe_country.categories_[0]) + len(ohe_ecoregion.categories_[0]) + len(ohe_dune.categories_[0]), len(le_species.classes_) + 2 + 1 + len(ohe_country.categories_[0]) + len(ohe_ecoregion.categories_[0]) + len(ohe_dune.categories_[0]) + len(ohe_coast.categories_[0]), 1)]  # Add the indices of coast columns to the selected columns
    if 'all' in features:
        select_columns = [*range(0, len(le_species.classes_) + 2 + 1 + len(ohe_country.categories_[0]) + len(ohe_ecoregion.categories_[0]) + len(ohe_dune.categories_[0]) + len(ohe_coast.categories_[0]), 1)]

    X = X[:,select_columns]  # Select the columns specified in select_columns from the input data

    return X

def add_augmentation(X, y, labels=None):
    """
    Apply augmentation to the input data by oversampling minority classes using BorderlineSMOTE.

    Args:
        X (scipy.sparse._csr.csr_matrix): The input data.
        y (numpy.ndarray): The target labels.
        labels (list, optional): The specific labels to augment. If None, all minority classes will be augmented.

    Returns:
        scipy.sparse._csr.csr_matrix: The augmented input data.
        numpy.ndarray: The augmented target labels.
    """
    n_samples_majority = max(np.bincount(y))  # Calculate the number of samples in the majority class
    n_samples_minority = int(n_samples_majority * 0.1)  # Set the target number of samples for the minority classes to 10% of the number of samples in the majority class
    class_counts = dict(collections.Counter(y))  # Get the number of samples for each class in the original data
    if labels is None:
        sampling_strategy = {i: max(n, n_samples_minority) for i, n in class_counts.items()}  # Define the sampling strategy as a dictionary where each class has at least the maximum of its original number of samples and n_samples_minority
    else:
        sampling_strategy = {i: max(n, n_samples_minority) for i, n in class_counts.items() if i in labels}
    k_neighbors = int(min(class_counts.values()) / 2)  # Define the number of neighbors to be considered for each sample
    m_neighbors = k_neighbors * 2  # Define the number of neighbors to consider to determine if a sample is danger
    sm = imblearn.over_sampling.BorderlineSMOTE(sampling_strategy=sampling_strategy, k_neighbors=k_neighbors, m_neighbors=m_neighbors)  # Define the oversampler
    X_resampled, y_resampled = sm.fit_resample(X, y)  # Fit the oversampler on the training data
    return X_resampled, y_resampled 

def add_rare_labels(y, X_test, y_test):
    """
    Filter out rare labels from the test data based on their occurrence in the training labels.

    Args:
        y (numpy.ndarray): The training labels.
        X_test (scipy.sparse._csr.csr_matrix): The test data.
        y_test (numpy.ndarray): The test labels.

    Returns:
        scipy.sparse._csr.csr_matrix: The filtered test data containing samples with rare labels.
        numpy.ndarray: The filtered test labels containing rare labels.
    """
    labels, counts = np.unique(y, return_counts=True)  # Get the unique labels and their respective counts in the training labels
    sorted_labels = labels[np.argsort(counts)]  # Sort the labels based on their counts in ascending order
    rare_labels = sorted_labels[:10]  # Select the 10 labels with the lowest counts (rare labels)
    rare_indices = np.isin(y_test, rare_labels)  # Create a boolean mask indicating whether each label in the test labels is a rare label
    X_test_rare = X_test[rare_indices]  # Filter the test data based on the rare label indices
    y_test_rare = y_test[rare_indices]  # Filter the test labels based on the rare label indices
    return X_test_rare, y_test_rare

def add_endangered_habitats(X, y, endangered_eunis_habitats):
    """
    Filter the data and labels to include only samples from endangered habitats.

    Args:
        X (scipy.sparse._csr.csr_matrix): The data matrix.
        y (numpy.ndarray): The labels.
        endangered_eunis_habitats (list): The list of endangered EUNIS habitats.

    Returns:
        scipy.sparse._csr.csr_matrix: The filtered data matrix containing samples from endangered habitats.
        numpy.ndarray: The filtered labels corresponding to the filtered data.
    """
    mask = np.isin(y, endangered_eunis_habitats)  # Create a Boolean mask indicating which samples belong to endangered habitats
    X = X[mask]  # Use the mask to extract only the desired samples from the X matrix
    y = y[mask]  # Use the mask to extract only the desired labels from the y array
    return X, y

def add_header_encoding(eva_header):
    """
    Encode the 'Expert System' column in the EVA header using LabelEncoder.

    Args:
        eva_header (pandas.core.frame.DataFrame): The EVA header DataFrame.

    Returns:
        pandas.core.frame.DataFrame: The modified EVA header DataFrame with the 'Expert System' column encoded.
        sklearn.preprocessing._label.LabelEncoder: The LabelEncoder object used for encoding.
    """
    le_header = sklearn.preprocessing.LabelEncoder()  # Create a LabelEncoder object
    le_header.fit(eva_header['Expert System'])  # Fit the LabelEncoder on the 'Expert System' column
    eva_header['Expert System'] = le_header.transform(eva_header['Expert System'])  # Encode the 'Expert System' column
    return eva_header, le_header

def add_header_decoding(eva_header, le_header):
    """
    Decode the encoded 'Expert System' column in the EVA header using a LabelEncoder.

    Args:
        eva_header (pandas.core.frame.DataFrame): The EVA header DataFrame.
        le_header (sklearn.preprocessing._label.LabelEncoder): The LabelEncoder object used for encoding.

    Returns:
        pandas.core.frame.DataFrame: The modified EVA header DataFrame with the 'Expert System' column decoded.
    """
    eva_header['Expert System'] = le_header.inverse_transform(eva_header['Expert System'])  # Decode the 'Expert System' column using the inverse_transform method of the LabelEncoder
    return eva_header

def add_predictions_decoding(predictions, le_header):
    """
    Decode the encoded predictions using a LabelEncoder.

    Args:
        predictions (list): The encoded predictions.
        le_header (sklearn.preprocessing._label.LabelEncoder): The LabelEncoder object used for encoding.

    Returns:
        numpy.ndarray: The decoded predictions.

    """
    predictions = le_header.inverse_transform(predictions)  # Decode the predictions using the inverse_transform method of the LabelEncoder
    return predictions

def add_gbif_normalization(df_species, eva_to_gbif_species):
    """
    Perform GBIF species name normalization on a DataFrame column.

    Args:
        df_species (pandas.core.frame.DataFrame): DataFrame containing the species data.
        eva_to_gbif_species (dict): Mapping of EVA species names to corresponding GBIF species names.

    Returns:
        pandas.core.frame.DataFrame: DataFrame with normalized species names.
        dict: Mapping of original species names to normalized GBIF species names.
    """
    base = "https://api.gbif.org/v1"  # Define the base URL
    api = "species"  # Define the GBIF API we want to query
    function = "match"  # Define the functionality we want to use
    parameter = "name"  # Define the parameters for our API call
    url = f"{base}/{api}/{function}?{parameter}="  # Define the URL
    species_df = pd.unique(df_species['Matched concept'])  # Get unique species names from the DataFrame column
    species_gbif = []  # Initialize an empty list to store GBIF species names
    print()
    for species in tqdm.tqdm(species_df, desc="GBIF normalization"):  # Iterate over species names, displaying progress with tqdm
        url = url.replace(url.partition('name')[2], f'={species}')  # Construct the URL for GBIF API with the species name
        r = requests.get(url)  # Send a GET request to GBIF API
        r = r.json()  # Parse the response as JSON
        if 'phylum' in r and r['phylum'] == 'Tracheophyta' and 'species' in r:
            r = r["species"]  # Retrieve the GBIF species name
            if eva_to_gbif_species and r not in eva_to_gbif_species.values():
                r = '?'  # If eva_to_gbif_species is provided and the species is not in the mapping, replace with '?'
        else:
            r = '?'  # If the species is not found in GBIF, replace with '?'
        species_gbif.append(r)  # Append the normalized species name to the list
    df_to_gbif_species = dict(zip(species_df, species_gbif))  # Create a mapping of original species names to normalized GBIF species names
    df_species['Matched concept'] = df_species['Matched concept'].map(df_to_gbif_species)  # Map the normalized species names to the DataFrame column
    df_species = df_species[df_species["Matched concept"] != '?']  # Remove rows with '?' as the normalized species name
    df_species = df_species.groupby(['PlotObservationID', 'Matched concept']).sum().reset_index()  # Group by plot observation ID and normalized species name, summing other columns
    return df_species, df_to_gbif_species

def add_species_encoding(df_species, le_species):
    """
    Encode species names in a DataFrame column using LabelEncoder.

    Args:
        df_species (pandas.core.frame.DataFrame): DataFrame containing the species data.
        le_species (sklearn.preprocessing._label.LabelEncoder, optional): Pre-initialized LabelEncoder object for species encoding.

    Returns:
        pandas.core.frame.DataFrame: DataFrame with encoded species names.
        sklearn.preprocessing._label.LabelEncoder: LabelEncoder object used for encoding.
    """
    if not le_species:
        le_species = sklearn.preprocessing.LabelEncoder()  # If LabelEncoder object is not provided, initialize a new one
        le_species.fit(df_species['Matched concept'])  # Fit the LabelEncoder on the species data
    df_species['Matched concept'] = le_species.transform(df_species['Matched concept'])  # Encode the species names using the LabelEncoder
    return df_species, le_species

def add_species_decoding(df_species, le_species):
    """
    Decode encoded species names in a DataFrame column using a given LabelEncoder.

    Args:
        df_species (pandas.core.frame.DataFrame): DataFrame containing the species data.
        le_species (sklearn.preprocessing._label.LabelEncoder): LabelEncoder object used for encoding species names.

    Returns:
        pandas.core.frame.DataFrame: DataFrame with decoded species names.
    """
    df_species['Matched concept'] = le_species.inverse_transform(df_species['Matched concept'])  # Decode the encoded species names using the inverse_transform method of the LabelEncoder
    return df_species

def add_spatial_split(args, eva_header):
    """
    Assign spatial blocks and split data into folds based on spatial coordinates.

    Args:
        args (argparse.Namespace): Command-line arguments or configuration parameters.
        eva_header (pandas.core.frame.DataFrame): DataFrame containing the habitat data.

    Returns:
        numpy.ndarray: Array representing the fold assignments for each data point.
        pandas.core.frame.DataFrame: DataFrame with an additional column for fold assignments.
    """
    lon_list = eva_header['Longitude'].to_numpy()
    lat_list = eva_header['Latitude'].to_numpy()
    east_west_bin_width_km = args.east_west_bin_width_km
    north_south_bin_width_km = args.north_south_bin_width_km
    origin = (0.0, 0.0)  # Define the origin
    block_id_assignments = assign_block_ids(lon_list, lat_list, east_west_bin_width_km, north_south_bin_width_km, origin)  # Assign spatial block IDs to each data point based on coordinates
    vals = np.unique(block_id_assignments)
    print(f"There are {len(vals)} different blocks.")
    block_ids = np.unique(block_id_assignments)
    folds = [len(block_ids) // args.folds + (1 if x < len(block_ids) % args.folds else 0) for x in range(args.folds)]  # Split the block IDs into folds
    idx_rand = np.random.permutation(len(block_ids))  # Randomly permute the block IDs
    idx = np.array_split(idx_rand, args.folds)  # Split the permuted block IDs into folds
    assert len(np.unique(np.concatenate(idx).ravel())) == len(block_ids)  # Ensure that the concatenated indices cover all block IDs
    folds_blocks_id = [block_ids[idx[i]] for i in range(len(idx))]  # Create a list of block IDs for each fold
    assert len(np.concatenate(folds_blocks_id)) == len(block_ids)  # Ensure that the concatenated block IDs cover all unique block IDs
    split_assignment = [fold for i in range(len(lon_list)) for fold in range(args.folds) if block_id_assignments[i] in folds_blocks_id[fold]]  # Assign fold numbers to each data point based on the block IDs
    split_assignment = np.array(split_assignment, dtype=int)
    print(f'{args.folds} different folds:')
    for i in range(args.folds):
        print(f'Fold {i}: {folds[i]} blocks')
    eva_header['Fold'] = split_assignment  # Add a column for fold assignments to the DataFrame
    return split_assignment, eva_header

def add_input_data(df_header, df_species, le_species):
    """
    Create input data matrix for habitat distribution modeling.

    Args:
        df_header (pandas.core.frame.DataFrame): DataFrame containing the header data.
        df_species (pandas.core.frame.DataFrame): DataFrame containing the species data.
        le_species (sklearn.preprocessing._label.LabelEncoder): LabelEncoder object for species encoding.

    Returns:
        scipy.sparse._csr.csr_matrix: Input data matrix with species coverage information.
    """
    data = []  # Initialize the data list
    indices = []  # Initialize the indices list
    X = df_header["PlotObservationID"].values
    for i in tqdm.tqdm(range(len(X)), desc="Input data"):
        species_cover = df_species[df_species["PlotObservationID"] == X[i]].values[:, 2]  # Extract species coverage for a specific observation ID
        species_ind = df_species[df_species["PlotObservationID"] == X[i]].values[:, 1]  # Extract species indices for a specific observation ID
        data.append(species_cover)  # Append species coverage to the data list
        indices.append(species_ind)  # Append species indices to the indices list
    indptr = np.empty(len(df_header) + 1)  # Create indptr array for efficient sparse matrix construction
    indptr[0] = 0
    for i in range(1, len(indptr)):
        indptr[i] = indptr[i-1] + len(data[i-1])
    data = np.concatenate(data)  # Concatenate the data array
    indices = np.concatenate(indices)  # Concatenate the indices array
    X = scipy.sparse.csr_matrix((data, indices, indptr))  # Create a compressed sparse row (CSR) matrix using the data, indices, and indptr arrays
    n = len(le_species.classes_)  # Desired number of columns
    num_rows, num_cols = X.shape
    if num_cols < n:  # Check if the number of columns in X is less than n
        num_zeros = n - num_cols
        zeros = scipy.sparse.csr_matrix((num_rows, num_zeros), dtype=X.dtype)
        X = scipy.sparse.hstack([X, zeros])
    assert X.shape == (len(df_header), len(le_species.classes_))  # Ensure the shape of X matches the expected shape
    return X

def add_target_values(eva_header):
    """
    Extract target values for habitat distribution modeling.

    Args:
        eva_header (pandas.core.frame.DataFrame): DataFrame containing the header data.

    Returns:
        numpy.ndarray: Array of target values.
    """
    y = eva_header['Expert System'].values  # Extract the target values by accessing the 'Expert System' column
    return y

def add_longitude(X, df_header):
    """
    Add longitude feature to the input data.

    Args:
        X (scipy.sparse._csr.csr_matrix): Input data matrix.
        df_header (pandas.core.frame.DataFrame): DataFrame containing the header data.

    Returns:
        scipy.sparse._csr.csr_matrix: Updated input data matrix with longitude feature.
    """
    X = scipy.sparse.hstack((X, np.array(df_header['Longitude'])[:,None]), format="csr")  # Concatenate the longitude array with the input data matrix horizontally
    return X

def add_latitude(X, df_header):
    """
    Add latitude feature to the input data.

    Args:
        X (scipy.sparse._csr.csr_matrix): Input data matrix.
        df_header (pandas.core.frame.DataFrame): DataFrame containing the header data.

    Returns:
        scipy.sparse._csr.csr_matrix: Updated input data matrix with latitude feature.
    """
    X = scipy.sparse.hstack((X, np.array(df_header['Latitude'])[:,None]), format="csr")  # Concatenate the latitude array with the input data matrix horizontally
    return X

def add_altitude(X, df_header, altitudes):
    """
    Add altitude feature to the input data.

    Args:
        X (scipy.sparse._csr.csr_matrix): Input data matrix.
        df_header (pandas.core.frame.DataFrame): DataFrame containing the header data.
        altitudes (xarray.core.dataarray.DataArray): DataArray containing altitude values.

    Returns:
        scipy.sparse._csr.csr_matrix: Updated input data matrix with altitude feature.
        pandas.core.frame.DataFrame: Updated DataFrame with altitude values.
    """
    transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857")  # Create a coordinate transformer from EPSG:4326 (latitude and longitude) to EPSG:3857 (Web Mercator)
    list_of_altitudes = []  # Initialize a list to store the altitude values
    for i in tqdm.tqdm(range(len(df_header)), desc="Altitudes"):  # Iterate over each row in the df_header DataFrame
        x, y = transformer.transform(df_header['Latitude'][i], df_header['Longitude'][i])  # Transform latitude and longitude to Web Mercator coordinates
        list_of_altitudes.append(altitudes.sel(x=x, y=y, method='nearest').item())  # Retrieve the nearest altitude value from the altitudes DataArray based on the transformed coordinates and append it to the list
    df_header['Altitude'] = list_of_altitudes   # Add the altitude values as a new column 'Altitude' to the df_header DataFrame
    list_of_altitudes = scipy.sparse.csr_matrix(list_of_altitudes).T  # Convert the list of altitudes to a sparse matrix with a single column (transpose)
    X = scipy.sparse.hstack((X, list_of_altitudes), format='csr')  # Horizontally stack the altitude sparse matrix with the input data matrix X
    return X, df_header

def add_country(args, X, y, df_header, df_species, countries, ohe_country):
    """
    Add country feature to the input data.

    Args:
        args (argparse.Namespace): Arguments for controlling country filtering.
        X (scipy.sparse._csr.csr_matrix): Input data matrix.
        y (numpy.ndarray): Array of target values.
        df_header (pandas.core.frame.DataFrame): DataFrame containing the header data.
        df_species (pandas.core.frame.DataFrame): DataFrame containing the species data.
        countries (geopandas.geodataframe.GeoDataFrame): GeoDataFrame containing country boundaries.
        ohe_country (sklearn.preprocessing._encoders.OneHotEncoder): OneHotEncoder for country labels.

    Returns:
        scipy.sparse._csr.csr_matrix: Updated input data matrix with country feature.
        numpy.ndarray: Array of target values.
        sklearn.preprocessing._encoders.OneHotEncoder: Updated OneHotEncoder for country labels.
        pandas.core.frame.DataFrame: Updated header DataFrame with country labels.
        pandas.core.frame.DataFrame: Updated species DataFrame with observations from specific countries.
    """
    country_index = rtree.index.Index()  # Create a spatial index for the country boundaries
    for index, country in countries.iterrows():  # Insert each country's index and bounding box into the spatial index
        country_index.insert(index, country["geometry"].bounds)
    list_of_countries = []   # Initialize a list to store the country labels
    for i in tqdm.tqdm(range(len(df_header)), desc="Countries"):  # Iterate over each row in the df_header DataFrame
        point = shapely.geometry.Point(df_header['Longitude'][i], df_header['Latitude'][i])
        point = {'geometry': [point]}  # Create a point geometry from the longitude and latitude
        point = gpd.GeoDataFrame(point, crs="EPSG:4326")  # Convert the point to a GeoDataFrame with EPSG:4326 CRS
        point = point.to_crs(crs=3857)  # Convert the point's CRS to EPSG:3857 (Web Mercator)
        point = pd.Series(data=point['geometry'][0], index=['geometry'])  # Extract the geometry of the point as a Pandas Series
        min_distance = float("inf")  # Initialize variables for finding the closest country
        closest_country = None
        for country_id in country_index.nearest((point["geometry"].x, point["geometry"].y, point["geometry"].x, point["geometry"].y), 1):  # Find countries that are within a certain distance of the point
            distance = point["geometry"].distance(countries.iloc[country_id]["geometry"])  # Calculate the distance between the point and the country
            if distance < min_distance:  # Update the closest country if the distance is smaller
                min_distance = distance
                closest_country = countries.iloc[country_id]
        closest_country = closest_country[0]  # Get the name of the closest country
        list_of_countries.append(closest_country)  # Append the closest country to the list of countries
    if not ohe_country:  # If not using OneHotEncoder for country labels
        countries_to_keep = args.countries
        if countries_to_keep == ['all']:  # If all countries are to be kept
            countries_to_keep = ['Afghanistan', 'Albania', 'Algeria', 'American Samoa', 'Andorra', 'Angola', 'Anguilla', 'Antarctica', 'Antigua and Barbuda', 'Argentina', 'Armenia', 'Aruba', 'Australia', 'Austria', 'Azerbaijan', 'Azores', 'Bahamas', 'Bahrain', 'Bangladesh', 'Barbados', 'Belarus', 'Belgium', 'Belize', 'Benin', 'Bermuda', 'Bhutan', 'Bolivia', 'Bonaire', 'Bosnia and Herzegovina', 'Botswana', 'Bouvet Island', 'Brazil', 'British Indian Ocean Territory', 'British Virgin Islands', 'Brunei Darussalam', 'Bulgaria', 'Burkina Faso', 'Burundi', 'Cabo Verde', 'Cambodia', 'Cameroon', 'Canada', 'Canarias', 'Cayman Islands', 'Central African Republic', 'Chad', 'Chile', 'China', 'Christmas Island', 'Cocos Islands', 'Colombia', 'Comoros', 'Congo', 'Congo DRC', 'Cook Islands', 'Costa Rica', "CÃ´te d'Ivoire", 'Croatia', 'Cuba', 'Curacao', 'Cyprus', 'Czech Republic', 'Denmark', 'Djibouti', 'Dominica', 'Dominican Republic', 'Ecuador', 'Egypt', 'El Salvador', 'Equatorial Guinea', 'Eritrea', 'Estonia', 'Eswatini', 'Ethiopia', 'Falkland Islands', 'Faroe Islands', 'Fiji', 'Finland', 'France', 'French Guiana', 'French Polynesia', 'French Southern Territories', 'Gabon', 'Gambia', 'Georgia', 'Germany', 'Ghana', 'Gibraltar', 'Glorioso Islands', 'Greece', 'Greenland', 'Grenada', 'Guadeloupe', 'Guam', 'Guatemala', 'Guernsey', 'Guinea', 'Guinea-Bissau', 'Guyana', 'Haiti', 'Heard Island and McDonald Islands', 'Honduras', 'Hungary', 'Iceland', 'India', 'Indonesia', 'Iran', 'Iraq', 'Ireland', 'Isle of Man', 'Israel', 'Italy', 'Jamaica', 'Japan', 'Jersey', 'Jordan', 'Juan De Nova Island', 'Kazakhstan', 'Kenya', 'Kiribati', 'Kuwait', 'Kyrgyzstan', 'Laos', 'Latvia', 'Lebanon', 'Lesotho', 'Liberia', 'Libya', 'Liechtenstein', 'Lithuania', 'Luxembourg', 'Madagascar', 'Madeira', 'Malawi', 'Malaysia', 'Maldives', 'Mali', 'Malta', 'Marshall Islands', 'Martinique', 'Mauritania', 'Mauritius', 'Mayotte', 'Mexico', 'Micronesia', 'Moldova', 'Monaco', 'Mongolia', 'Montenegro', 'Montserrat', 'Morocco', 'Mozambique', 'Myanmar', 'Namibia', 'Nauru', 'Nepal', 'Netherlands', 'New Caledonia', 'New Zealand', 'Nicaragua', 'Niger', 'Nigeria', 'Niue', 'Norfolk Island', 'North Korea', 'North Macedonia', 'Northern Mariana Islands', 'Norway', 'Oman', 'Pakistan', 'Palau', 'Palestinian Territory', 'Panama', 'Papua New Guinea', 'Paraguay', 'Peru', 'Philippines', 'Pitcairn', 'Poland', 'Portugal', 'Puerto Rico', 'Qatar', 'RÃ©union', 'Romania', 'Russian Federation', 'Rwanda', 'Saba', 'Saint Barthelemy', 'Saint Eustatius', 'Saint Helena', 'Saint Kitts and Nevis', 'Saint Lucia', 'Saint Martin', 'Saint Pierre and Miquelon', 'Saint Vincent and the Grenadines', 'Samoa', 'San Marino', 'Sao Tome and Principe', 'Saudi Arabia', 'Senegal', 'Serbia', 'Seychelles', 'Sierra Leone', 'Singapore', 'Sint Maarten', 'Slovakia', 'Slovenia', 'Solomon Islands', 'Somalia', 'South Africa', 'South Georgia and South Sandwich Islands', 'South Korea', 'South Sudan', 'Spain', 'Sri Lanka', 'Sudan', 'Suriname', 'Svalbard', 'Sweden', 'Switzerland', 'Syria', 'Tajikistan', 'Tanzania', 'Thailand', 'Timor-Leste', 'Togo', 'Tokelau', 'Tonga', 'Trinidad and Tobago', 'Tunisia', 'Turkiye', 'Turkmenistan', 'Turks and Caicos Islands', 'Tuvalu', 'Uganda', 'Ukraine', 'United Arab Emirates', 'United Kingdom', 'United States', 'United States Minor Outlying Islands', 'Uruguay', 'US Virgin Islands', 'Uzbekistan', 'Vanuatu', 'Vatican City', 'Venezuela', 'Vietnam', 'Wallis and Futuna', 'Yemen', 'Zambia', 'Zimbabwe']  # List of all country names
        else:
            countries_to_keep = ''.join(countries_to_keep).split(',')
        list_of_countries = ['?' if country not in countries_to_keep else country for country in list_of_countries]  # Replace non-kept countries with '?'
        mask = np.array([country != '?' for country in list_of_countries])  # Create a boolean mask for filtering the data
        X = X[mask, :]  # Apply the mask to X
        if y is not None:
            y = y[mask]  # Apply the mask to y
        df_header = df_header[mask]  # Apply the mask to df_header
        df_species = df_species[df_species['PlotObservationID'].isin(df_header['PlotObservationID'])]  # Filter the species based on vegetation plots left in the header
        df_header = df_header.reset_index(drop=True)  # Reset the index
        df_species = df_species.reset_index(drop=True)  # Reset the index
        list_of_countries = [country for country in list_of_countries if country != '?']  # Remove '?' entries from the list of countries
        ohe_country = sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore')  # Initialize the OneHotEncoder for country labels
        ohe_country.fit(np.reshape(list_of_countries, (-1, 1)))  # Fit the OneHotEncoder for country labels
    df_header['Country'] = list_of_countries  # Add the country labels to df_header
    list_of_countries = ohe_country.transform(np.reshape(list_of_countries, (-1, 1)))  # Transform the country labels using OneHotEncoder
    X = scipy.sparse.hstack((X, list_of_countries))  # Concatenate the country labels with X
    return X, y, ohe_country, df_header, df_species

def add_ecoregion(X, df_header, ecoregions, ohe_ecoregion):
    """
    Adds ecoregion information to the input data.

    Args:
        X (scipy.sparse._csr.csr_matrix): Input data matrix.
        df_header (pandas.core.frame.DataFrame): DataFrame containing header information.
        ecoregions (geopandas.GeoDataFrame): GeoDataFrame containing ecoregion polygons.
        ohe_ecoregion (sklearn.preprocessing._encoders.OneHotEncoder): Pre-initialized OneHotEncoder for ecoregions.

    Returns:
        scipy.sparse._csr.csr_matrix: Updated data matrix with ecoregion information.
        sklearn.preprocessing._encoders.OneHotEncoder: Updated OneHotEncoder for ecoregions.
        pandas.core.frame.DataFrame: DataFrame with added 'Ecoregion' column.
    """
    ecoregion_index = rtree.index.Index()  # Create a spatial index for efficient querying of ecoregions
    for index, ecoregion in ecoregions.iterrows():
        ecoregion_index.insert(index, ecoregion["geometry"].bounds)
    list_of_ecoregions = []  # Initialize a list to store the closest ecoregion for each data point
    for i in tqdm.tqdm(range(len(df_header)), desc="Ecoregions"):  # Iterate over the data points and find the closest ecoregion
        point = shapely.geometry.Point(df_header['Longitude'][i], df_header['Latitude'][i])  # Create a Point object from latitude and longitude
        point = {'geometry': [point]}   # Convert the Point object to a GeoDataFrame with the same CRS as ecoregions
        point = gpd.GeoDataFrame(point, crs="EPSG:4326")
        point = point.to_crs(crs=3857)
        point = pd.Series(data=point['geometry'][0], index=['geometry'])
        min_distance = float("inf")  # Find the closest ecoregion using the spatial index
        closest_ecoregion = None
        for ecoregion_id in ecoregion_index.nearest((point["geometry"].x, point["geometry"].y, point["geometry"].x, point["geometry"].y), 1):  # Find ecoregions that are within a certain distance of the point
            distance = point["geometry"].distance(ecoregions.iloc[ecoregion_id]["geometry"])  # Calculate the distance between the point and the ecoregion
            if distance < min_distance:
                min_distance = distance
                closest_ecoregion = ecoregions.iloc[ecoregion_id]
        closest_ecoregion = closest_ecoregion[0]  # Add the closest ecoregion to the list
        list_of_ecoregions.append(closest_ecoregion)
    if not ohe_ecoregion:  # If the OneHotEncoder for ecoregions is not provided 
        ohe_ecoregion = sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore')  # Initialize a new OneHotEncoder for ecoregions
        ohe_ecoregion.fit(np.reshape(list_of_ecoregions, (-1, 1)))  # Fit the new OneHotEncoder for ecoregions
    df_header['Ecoregion'] = list_of_ecoregions  # Add the ecoregion labels to df_header
    list_of_ecoregions = ohe_ecoregion.transform(np.reshape(list_of_ecoregions, (-1, 1)))  # Transform the ecoregion labels using the OneHotEncoder
    X = scipy.sparse.hstack((X, list_of_ecoregions))  # Concatenate the ecoregion labels to the input data matrix
    return X, ohe_ecoregion, df_header

def add_dune(X, df_header, dunes, ohe_dune):
    """
    Add dune information to the input data matrix.

    Args:
        X (scipy.sparse._csr.csr_matrix): Input data matrix.
        df_header (pandas.core.frame.DataFrame): DataFrame containing header information.
        dunes (geopandas.GeoDataFrame): GeoDataFrame containing dune geometry information.
        ohe_dune (sklearn.preprocessing._encoders.OneHotEncoder): OneHotEncoder for dune labels.

    Returns:
        scipy.sparse._csr.csr_matrix: Updated data matrix.
        sklearn.preprocessing._encoders.OneHotEncoder: Updated OneHotEncoder for dune labels.
        pandas.core.frame.DataFrame: Updated DataFrame.
    """
    list_of_dunes = []  # Initialize a list to store dune labels
    for i in tqdm.tqdm(range(len(df_header)), desc="Dunes"):  # Iterate over each data point
        point = shapely.geometry.Point(df_header['Longitude'][i], df_header['Latitude'][i])  # Create a point object from longitude and latitude coordinates
        point = {'geometry': [point]}  # Create a GeoDataFrame with the point
        point = gpd.GeoDataFrame(point, crs="EPSG:4326")
        point = point.to_crs(crs=3857)   # Convert the point to the desired coordinate reference system
        point = shapely.geometry.Point(point.centroid.x, point.centroid.y)  # Calculate the centroid of the point
        list_of_dunes.append(point.within(dunes))  # Check if the point is within any dune
    list_of_dunes = ['Y_DUNES' if dune else 'N_DUNES' for dune in list_of_dunes]  # Map boolean values to dune labels ('Y_DUNES' or 'N_DUNES')
    if not ohe_dune:  # If OneHotEncoder for dune labels is not provided, initialize and fit a new one
        ohe_dune = sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore')
        ohe_dune.fit(np.reshape(list_of_dunes, (-1, 1)))
    df_header['Dune'] = list_of_dunes  # Update the DataFrame with dune labels
    list_of_dunes = ohe_dune.transform(np.reshape(list_of_dunes, (-1, 1)))  # Transform dune labels using the OneHotEncoder
    X = scipy.sparse.hstack((X, list_of_dunes))  # Concatenate the dune labels to the input data matrix
    return X, ohe_dune, df_header

def add_coast(X, df_header, coasts, ohe_coast):
    """
    Add coast information to the input data matrix.

    Args:
        X (scipy.sparse._csr.csr_matrix): Input data matrix.
        df_header (pandas.core.frame.DataFrame): DataFrame containing header information.
        coasts (geopandas.GeoDataFrame): GeoDataFrame containing coast geometry information.
        ohe_coast (sklearn.preprocessing._encoders.OneHotEncoder): OneHotEncoder for coast labels.

    Returns:
        scipy.sparse._csr.csr_matrix: Updated data matrix.
        sklearn.preprocessing._encoders.OneHotEncoder: Updated OneHotEncoder for coast labels.
        pandas.core.frame.DataFrame: Updated DataFrame.
    """
    list_of_coasts = []  # Initialize a list to store coast labels
    for i in tqdm.tqdm(range(len(df_header)), desc="Coasts"):  # Iterate over each data point
        point = shapely.geometry.Point(df_header['Longitude'][i], df_header['Latitude'][i])  # Create a point object from longitude and latitude coordinates
        point = {'geometry': [point]}  # Create a GeoDataFrame with the point
        point = gpd.GeoDataFrame(point, crs="EPSG:4326")
        point = point.to_crs(crs=3857)  # Convert the point to the desired coordinate reference system
        point = shapely.geometry.Point(point.centroid.x, point.centroid.y)  # Calculate the centroid of the point
        if min(coasts.geometry.distance(point)) <= 5000:  # Calculate the distance between the point and each coast and check if the minimum distance to any coast is within a threshold (5000 units)
            list_of_coasts.append(coasts.geometry.distance(point).idxmin())  # Add the index of the coast with the minimum distance
        else:
            list_of_coasts.append("N_COAST")  # Add a label indicating no coast is within the threshold
    if not ohe_coast:  # If OneHotEncoder for coast labels is not provided, initialize and fit a new one
        ohe_coast = sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore')
        ohe_coast.fit(np.reshape(list_of_coasts, (-1, 1)))
    df_header['Coast'] = list_of_coasts  # Update the DataFrame with coast labels
    list_of_coasts = ohe_coast.transform(np.reshape(list_of_coasts, (-1, 1)))  # Transform coast labels using the OneHotEncoder
    X = scipy.sparse.hstack((X, list_of_coasts))  # Concatenate the coast labels to the input data matrix
    return X, ohe_coast, df_header
