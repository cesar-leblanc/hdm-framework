import pickle
import numpy as np
import scipy.sparse
import pandas as pd
import joblib
import torch

def set_le_header(args, le_header):
    """
    Save the label encoder for header information to a file.

    Args:
        args (argparse.Namespace): Arguments passed to the function.
        le_header (sklearn.preprocessing._label.LabelEncoder): LabelEncoder for header information.
    """
    le_header_filepath = args.data_filepath + 'le_header.pkl'  # Define the file path to save the label encoder
    with open(le_header_filepath, 'wb') as f:  # Open the file in write binary mode and save the label encoder using pickle
        pickle.dump(le_header, f)

def set_eva_to_gbif_species(args, eva_to_gbif_species):
    """
    Save the mapping of EVA species to GBIF species to a file.

    Args:
        args (argparse.Namespace): Arguments passed to the function.
        eva_to_gbif_species (dict): Mapping of EVA species to GBIF species.
    """
    eva_to_gbif_species_filepath = args.data_filepath + 'eva_to_gbif_species.pkl'  # Define the file path to save the mapping
    with open(eva_to_gbif_species_filepath, 'wb') as f:  # Open the file in write binary mode and save the mapping using pickle
        pickle.dump(eva_to_gbif_species, f)

def set_le_species(args, le_species):
    """
    Save the LabelEncoder object for species to a file.

    Args:
        args (argparse.Namespace): Arguments passed to the function.
        le_species (sklearn.preprocessing._label.LabelEncoder): LabelEncoder object for species.
    """
    le_species_filepath = args.data_filepath + 'le_species.pkl'  # Define the file path to save the LabelEncoder object
    with open(le_species_filepath, 'wb') as f:  # Open the file in write binary mode and save the LabelEncoder object using pickle
        pickle.dump(le_species, f)

def set_split_assignment(args, split_assignment):
    """
    Save the split assignment array to a file.

    Args:
        args (argparse.Namespace): Arguments passed to the function.
        split_assignment (numpy.ndarray): Array containing the split assignments.
    """
    split_assignment_filepath = args.data_filepath + 'split_assignments.npy'  # Define the file path to save the split assignment array
    np.save(split_assignment_filepath, split_assignment)  # Save the split assignment array using numpy's save function

def set_ohe_country(args, ohe_country):
    """
    Save the OneHotEncoder object for country to a file.

    Args:
        args (argparse.Namespace): Arguments passed to the function.
        ohe_country (sklearn.preprocessing._encoders.OneHotEncoder): OneHotEncoder object for country.
    """
    ohe_country_filepath = args.data_filepath + 'ohe_country.pkl'  # Define the file path to save the OneHotEncoder object for country
    with open(ohe_country_filepath, 'wb') as f:  # Save the OneHotEncoder object using pickle
        pickle.dump(ohe_country, f)

def set_ohe_ecoregion(args, ohe_ecoregion):
    """
    Save the OneHotEncoder object for ecoregion to a file.

    Args:
        args (argparse.Namespace): Arguments passed to the function.
        ohe_ecoregion (sklearn.preprocessing._encoders.OneHotEncoder): OneHotEncoder object for ecoregion.
    """
    ohe_ecoregion_filepath = args.data_filepath + 'ohe_ecoregion.pkl'  # Define the file path to save the OneHotEncoder object for ecoregion
    with open(ohe_ecoregion_filepath, 'wb') as f:  # Save the OneHotEncoder object using pickle
        pickle.dump(ohe_ecoregion, f)

def set_ohe_dune(args, ohe_dune):
    """
    Save the OneHotEncoder object for dune to a file.

    Args:
        args (argparse.Namespace): Arguments passed to the function.
        ohe_dune (sklearn.preprocessing._encoders.OneHotEncoder): OneHotEncoder object for dune.
    """
    ohe_dune_filepath = args.data_filepath + 'ohe_dune.pkl'  # Define the file path to save the OneHotEncoder object for dune
    with open(ohe_dune_filepath, 'wb') as f:  # Save the OneHotEncoder object using pickle
        pickle.dump(ohe_dune, f)

def set_ohe_coast(args, ohe_coast):
    """
    Save the OneHotEncoder object for coast to a file.

    Args:
        args (argparse.Namespace): Arguments passed to the function.
        ohe_coast (sklearn.preprocessing._encoders.OneHotEncoder): OneHotEncoder object for coast.
    """
    ohe_coast_filepath = args.data_filepath + 'ohe_coast.pkl'  # Define the file path to save the OneHotEncoder object for coast
    with open(ohe_coast_filepath, 'wb') as f:  # Save the OneHotEncoder object using pickle
        pickle.dump(ohe_coast, f)

def set_eva_header(args, eva_header):
    """
    Save the evaluation header DataFrame to a CSV file.

    Args:
        args (argparse.Namespace): Arguments passed to the function.
        eva_header (pandas.core.frame.DataFrame): Evaluation header DataFrame.
    """
    eva_header_filepath = args.datasets_filepath + 'eva_header.csv'  # Define the file path to save the evaluation header CSV file
    eva_header.to_csv(eva_header_filepath, index=False)  # Save the evaluation header DataFrame to a CSV file

def set_eva_species(args, eva_species):
    """
    Save the evaluation species DataFrame to a CSV file.

    Args:
        args (argparse.Namespace): Arguments passed to the function.
        eva_species (pandas.core.frame.DataFrame): Evaluation species DataFrame.
    """
    eva_species_filepath = args.datasets_filepath + 'eva_species.csv'  # Define the file path to save the evaluation species CSV file
    eva_species.to_csv(eva_species_filepath, index=False)  # Save the evaluation species DataFrame to a CSV file

def set_test_header(args, test_header):
    """
    Save the test header DataFrame to a CSV file.

    Args:
        args (argparse.Namespace): Arguments passed to the function.
        test_header (pandas.core.frame.DataFrame): Test header DataFrame.
    """
    test_header_filepath = args.datasets_filepath + 'test_header.csv'  # Define the file path to save the test header CSV file
    test_header.to_csv(test_header_filepath, index=False)  # Save the test header DataFrame to a CSV file

def set_test_species(args, test_species):
    """
    Save the test species DataFrame to a CSV file.

    Args:
        args (argparse.Namespace): Arguments passed to the function.
        test_species (pandas.core.frame.DataFrame): Test species DataFrame.
    """
    test_species_filepath = args.datasets_filepath + 'test_species.csv'  # Define the file path to save the test species CSV file
    test_species.to_csv(test_species_filepath, index=False)  # Save the test species DataFrame to a CSV file

def set_input_data(args, X):
    """
    Save the input data sparse matrix to a compressed NPZ file.

    Args:
        args (argparse.Namespace): Arguments passed to the function.
        X (scipy.sparse._csr.csr_matrix): Input data sparse matrix.
    """
    input_data_filepath = args.data_filepath + 'input_data.npz'  # Define the file path to save the input data NPZ file
    scipy.sparse.save_npz(input_data_filepath, X)  # Save the input data sparse matrix to a compressed NPZ file
    
def set_test_data(args, X):
    """
    Save the test data sparse matrix to a compressed NPZ file.

    Args:
        args (argparse.Namespace): Arguments passed to the function.
        X (scipy.sparse._csr.csr_matrix): Test data sparse matrix.
    """
    test_data_filepath = args.data_filepath + 'test_data.npz'  # Define the file path to save the test data NPZ file
    scipy.sparse.save_npz(test_data_filepath, X)  # Save the test data sparse matrix to a compressed NPZ file
    
def set_target_values(args, y):
    """
    Save the target values array to a NPY file.

    Args:
        args (argparse.Namespace): Arguments passed to the function.
        y (np.ndarray): Target values array.
    """
    target_values_filepath = args.data_filepath + 'target_values.npy'  # Define the file path to save the target values NPY file
    np.save(target_values_filepath, y)  # Save the target values array to a NPY file
    
def set_scaler(args, scaler):
    """
    Save the scaler object to a pickle file.

    Args:
        args (argparse.Namespace): Arguments passed to the function.
        scaler (sklearn.preprocessing._data.StandardScaler): Scaler object.
    """
    scaler_filepath = args.data_filepath + 'scaler.pkl'  # Define the file path to save the scaler pickle file
    with open(scaler_filepath, 'wb') as f:  # Save the scaler object to a pickle file
        pickle.dump(scaler, f)

def set_mlp_model(args, model):
    """
    Save the MLP model to a PyTorch model file.

    Args:
        args (argparse.Namespace): Arguments passed to the function.
        model (models.mlp.MLP | torch.nn.parallel.data_parallel.DataParallel): MLP model.
    """
    mlp_filepath = args.models_filepath + 'MLP.pth'  # Define the file path to save the MLP model file
    torch.save(model, mlp_filepath)  # Save the MLP model to a PyTorch model file
    
def set_rfc_model(args, model):
    """
    Save the RFC model to a joblib file.

    Args:
        args (argparse.Namespace): Arguments passed to the function.
        model (models.rfc.RFC): RFC model.
    """
    rfc_filepath = args.models_filepath + 'RFC.joblib'  # Define the file path to save the RFC model file
    joblib.dump(model, rfc_filepath, compress=True)  # Save the RFC model to a joblib file
    
def set_xgb_model(args, model):
    """
    Save the XGB model to a JSON file.

    Args:
        args (argparse.Namespace): Arguments passed to the function.
        model (models.xgb.XGB): XGB model.
    """
    xgb_filepath = args.models_filepath + 'XGB.json'  # Define the file path to save the XGB model file
    model.save_model(xgb_filepath)  # Save the XGB model to a JSON file

def set_tnc_model(args, model):
    """
    Save the TNC model to a file.

    Args:
        args (argparse.Namespace): Arguments passed to the function.
        model (models.tnc.TNC): TNC model.
    """
    tnc_filepath = args.models_filepath + 'TNC'  # Define the file path to save the TNC model file
    model.save_model(tnc_filepath)  # Save the TNC model to a file
    
def set_ftt_model(args, model):
    """
    Save the FTT model to a PyTorch model file.

    Args:
        args (argparse.Namespace): Arguments passed to the function.
        model (models.ftt.FTT | torch.nn.parallel.data_parallel.DataParallel): FTT model.
    """
    ftt_filepath = args.models_filepath + 'FTT.pth'  # Define the file path to save the FTT model file
    torch.save(model, ftt_filepath)  # Save the FTT model to a PyTorch model file

def set_predictions(args, predictions):
    """
    Save the predictions array to a text file.

    Args:
        args (argparse.Namespace): Arguments passed to the function.
        predictions (np.ndarray): Predictions array.
    """
    predictions_filepath = args.data_filepath + 'predictions.txt'  # Define the file path to save the predictions text file
    np.savetxt(predictions_filepath, predictions, fmt="%s")  # Save the predictions array to a text file
