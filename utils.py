import torch
import random
import numpy as np
import pytopk
import torchmetrics
import geopy.distance
import sys
import os
import shutil
import scipy.sparse
import multiprocessing
import sklearn.utils
import pandas as pd
import warnings

def bool_type(s):
    """
    Converts a string to a boolean value.

    Args:
        s (str): The input string to convert.

    Returns:
        bool: The converted boolean value.

    Raises:
        ValueError: If the input string is not a valid boolean value.
    """
    if s.lower() in ('true', 't', 'yes', 'y', '1'):
        return True
    elif s.lower() in ('false', 'f', 'no', 'n', '0'):
        return False
    else:
        raise ValueError('Invalid boolean value: %s' % s)
    
def disable_warnings(args):
    """
    Disable warnings based on the provided argument.

    Args:
        args (argparse.Namespace): Namespace containing command-line arguments.

    Note:
        If the 'disable_warnings' argument is True, this function will ignore all warnings to avoid cluttering the output with warning messages during the execution of the code.
    """
    if args.disable_warnings:
        warnings.filterwarnings("ignore")

def disable_bytecode_generation(args):
    """
    Disables bytecode generation by removing existing bytecode cache directories.

    Args:
        args (argparse.Namespace): The arguments containing the bytecode generation flag.

    Note:
        Bytecode generation can be enabled or disabled based on the value of the write_bytecode flag in args.
    """
    if args.write_bytecode:  # Check if bytecode generation is enabled
        sys.dont_write_bytecode = False  # Allow bytecode generation
    else:  # Check if bytecode generation is disabled
        sys.dont_write_bytecode = True  # Disable bytecode generation
        for root, dirs, files in os.walk(".", topdown=False):  # Remove existing bytecode cache directories
            for dir in dirs:
                if dir == "__pycache__":  # Check if the directory is "__pycache__"
                    shutil.rmtree(os.path.join(root, dir))  # Remove the "__pycache__" directory and its contents

def print_parameters(args):
    """
    Prints the parameters if the print_parameters flag is set.

    Args:
        args (argparse.Namespace): The arguments containing the parameters.

    Prints:
        The parameters and their corresponding values.
    """
    if args.print_parameters:  # Check if the print_parameters flag is set
        parameters = vars(args)  # Get the dictionary of parameters
        print('Parameters:')  # Print the heading for the parameters
        for k, v in parameters.items():  # Iterate over the parameters
            print(k + ': ' + str(v))  # Print the parameter name and value

def set_seed(args):
    """
    Sets the random seeds for reproducibility.

    Args:
        args (argparse.Namespace): The arguments containing the seed values.

    Note:
        The seed is set differently based on the pipeline and model selected.
        If the pipeline is 'evaluation', 'training', or 'prediction' and the model is 'mlp' or 'ftt', the seed is set for the torch module.
    """
    random.seed(args.global_seed)  # Set the random seed for the random module
    np.random.seed(args.global_seed)  # Set the random seed for the numpy module
    if args.pipeline in ['evaluation', 'training', 'prediction'] and args.model in ['mlp', 'ftt']:
        torch.manual_seed(args.seed)  # Set the seed for the torch module
        if args.use_gpu:
            torch.cuda.manual_seed(args.seed)  # Set the seed for the torch CUDA module
    else:
        torch.manual_seed(args.global_seed)  # Set the seed for the torch module
        if args.use_gpu:
            torch.cuda.manual_seed(args.global_seed)  # Set the seed for the torch CUDA module

def get_device(args):
    """
    Retrieves the device for computation.

    Args:
        args (argparse.Namespace): The arguments containing the device configuration.

    Returns:
        torch.device: The device object representing the computation device.
    """
    if args.use_gpu:
        device = torch.device('cuda')  # Set the device to GPU
    else:
        device = torch.device('cpu')  # Set the device to CPU
    return device

def get_criterion(args, y_train):
    """
    Retrieves the criterion for loss calculation.

    Args:
        args (argparse.Namespace): The arguments containing the criterion configuration.
        y_train (numpy.ndarray): The training labels.

    Returns:
        torch.nn.modules.loss.CrossEntropyLoss | pytopk.noised_losses.BalNoisedTopK | pytopk.noised_losses.ImbalNoisedTopK: The criterion for loss calculation.
    """
    if args.model == 'mlp' or args.model =='ftt':
        if args.criterion == 'cross-entropy':
            criterion = torch.nn.CrossEntropyLoss()  # Use CrossEntropyLoss for criterion
        elif args.criterion == 'bal-noised-top-k':
            criterion = pytopk.BalNoisedTopK(k=args.k, epsilon=3.0)  # Use BalNoisedTopK for criterion
        elif args.criterion == 'imbal-noised-top-k': 
            criterion = pytopk.ImbalNoisedTopK(k=args.k, epsilon=0.01, max_m=0.7, cls_num_list=np.bincount(y_train).tolist(), scale=50, n_sample=5)  # Use ImbalNoisedTopK for criterion
    elif args.model == 'tnc':
        if args.loss_fn == 'cross-entropy':
            criterion = torch.nn.CrossEntropyLoss()  # Use CrossEntropyLoss for criterion
    return criterion

def get_optimizer(args, model):
    """
    Retrieves the optimizer for model parameter updates.

    Args:
        args (argparse.Namespace): The arguments containing the optimizer configuration.
        model (torch.nn.Module): The model to optimize.

    Returns:
        torch.optim.sgd.SGD | torch.optim.adam.Adam | torch.optim.adamw.AdamW: The optimizer for model parameter updates.
    """
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate_init, momentum=args.momentum, nesterov=args.nesterov, weight_decay=args.weight_decay)  # Use Stochastic Gradient Descent (SGD) optimizer with specified parameters
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate_init, weight_decay=args.weight_decay)  # Use Adam optimizer with specified parameters
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate_init, weight_decay=args.weight_decay)  # Use AdamW optimizer with specified parameters
    return optimizer

def get_scheduler(args, optimizer):
    """
    Retrieves the scheduler for adjusting the learning rate.

    Args:
        args (argparse.Namespace): The arguments containing the scheduler configuration.
        optimizer (torch.optim): The optimizer for which the scheduler will adjust the learning rate.

    Returns:
        torch.optim.lr_scheduler.ReduceLROnPlateau: The scheduler for adjusting the learning rate.
    """
    if args.model == 'mlp' or args.model == 'ftt':
        if args.scheduler == 'plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=args.factor, patience=args.patience)  # Use ReduceLROnPlateau scheduler with specified parameters
    elif args.model == 'tnc':
        if args.scheduler_fn == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR
    return scheduler

def get_metric(args, num_classes):
    """
    Retrieves the metric for evaluating model performance.

    Args:
        args (argparse.Namespace): The arguments containing the metric configuration.
        num_classes (int): The number of classes for the metric.

    Returns:
        torchmetrics.classification.accuracy.Accuracy: The metric for evaluating model performance.
    """
    if args.metric == 'accuracy':
        metric = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes, average=args.average, top_k=args.k)  # Use Accuracy metric with specified parameters
    return metric

def get_available_subprocesses(args):
    """
    Get the number of available subprocesses (CPUs) based on the command-line arguments.

    Args:
        args (argparse.Namespace): Command-line arguments.

    Returns:
        int: The number of available subprocesses (CPUs).
    """
    if args.num_workers == -1:
        num_cpus = multiprocessing.cpu_count()  # Get the number of available CPUs
    else:
        num_cpus = args.num_workers  # Use the specified number of subprocesses from command-line arguments
    return num_cpus

class NormedLinear(torch.nn.Module):
    """
    Custom module for a normalized linear layer.
    """
    def __init__(self, in_features, out_features):
        """
        Initializes the NormedLinear module.

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
        """
        super(NormedLinear, self).__init__()
        self.weight = torch.nn.Parameter(torch.Tensor(in_features, out_features))  # Define the weight matrix as a trainable parameter
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)  # Initialize the weight matrix with random values and normalize it along dimension 1

    def forward(self, x):
        """
        Forward pass of the normalized linear layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        out = torch.nn.functional.normalize(x, dim=1).mm(torch.nn.functional.normalize(self.weight, dim=0))  # Normalize the input tensor along dimension 1 (row-wise) and perform matrix multiplication between the normalized input and normalized weight matrix
        return out

def coords_to_bin(x, y, x_bin_width, y_bin_width):
    """
    Convert coordinates to bin indices based on the specified bin widths.

    Args:
        x (numpy.ndarray): Array of x-coordinates.
        y (numpy.ndarray): Array of y-coordinates.
        x_bin_width (float): Width of the x-coordinate bins.
        y_bin_width (float): Width of the y-coordinate bins.

    Returns:
        tuple: Tuple containing two numpy arrays, (x_bin_list, y_bin_list), representing the bin indices.
    """
    assert np.all(x >= 0)  # Assert that all x-coordinates are non-negative
    assert np.all(y >= 0)  # Assert that all y-coordinates are non-negative
    assert x_bin_width > 0  # Assert that x_bin_width is positive
    assert y_bin_width > 0  # Assert that y_bin_width is positive
    x_bin_list = np.array(np.floor(x / x_bin_width), dtype=int)  # Compute x-coordinate bin indices
    y_bin_list = np.array(np.floor(y / y_bin_width), dtype=int)  # Compute y-coordinate bin indices
    return (x_bin_list, y_bin_list)

def lon_to_global_easting(lon, origin):
    """
    Convert longitude coordinates to global easting distances in kilometers.

    Args:
        lon (numpy.ndarray): Array of longitude coordinates.
        origin (tuple): Tuple containing the origin coordinates (latitude, longitude).

    Returns:
        numpy.ndarray: Array of global easting distances in kilometers.
    """
    WGS84_EQUATOR_CIRCUMFERENCE = 40075.017  # Equator circumference of WGS84 ellipsoid in kilometers
    global_eastings_km = np.zeros_like(lon)  # Initialize an array to store the global easting distances
    for i in range(len(lon)):
        lon_ofs = lon[i] - origin[1]  # Calculate the offset of longitude from the origin longitude
        if lon_ofs < -180:
            lon_ofs = 180 - (lon_ofs + 180)  # If the offset is less than -180 degrees, wrap it around to the range [-180, 180]
        elif lon_ofs > 180:
            lon_ofs = -180 - (lon_ofs - 180)  # If the offset is greater than 180 degrees, wrap it around to the range [-180, 180]
        global_eastings_km[i] = geopy.distance.geodesic(origin, (origin[0], lon[i])).km  # Calculate the geodesic distance between the origin and the current longitude coordinate, convert the distance to kilometers and store it in the global eastings array
        if lon_ofs < 0:
            global_eastings_km[i] = WGS84_EQUATOR_CIRCUMFERENCE - global_eastings_km[i]  # If the offset is negative, adjust the global easting distance by subtracting it from the equator circumference
    return global_eastings_km

def lat_to_global_northing(lat, origin):
    """
    Convert latitude coordinates to global northing distances.

    Args:
        lat (numpy.ndarray): Array of latitude coordinates.
        origin (tuple): Tuple containing the origin coordinates (latitude, longitude).

    Returns:
        numpy.ndarray: Array of global northing distances.
    """
    WGS84_MERIDIAN_CIRCUMFERENCE = 40007.863  # Circumference of the meridian in kilometers
    global_northings_km = np.zeros_like(lat)  # Initialize an array for global northing distances
    for i in range(len(lat)):
        lat_ofs = lat[i] - origin[0]  # Calculate the latitude offset from the origin
        if lat_ofs < -90:
            lat_ofs = 90 - (lat_ofs + 90)  # Adjust the latitude offset if it is less than -90 degrees
        elif lat_ofs > 90:
            lat_ofs = -90 - (lat_ofs - 90)  # Adjust the latitude offset if it is greater than 90 degrees
        global_northings_km[i] = geopy.distance.geodesic(origin, (lat[i], origin[1])).km  # Calculate the geodesic distance between the origin and the latitude coordinate, which represents the global northing distance in kilometers
        if lat_ofs < 0:
            global_northings_km[i] = WGS84_MERIDIAN_CIRCUMFERENCE - global_northings_km[i]  # Adjust the global northing distance if the latitude offset is negative
    return global_northings_km

def assign_block_ids(lon, lat, east_west_bin_km, north_south_bin_km, origin):
    """
    Assign block IDs to coordinates based on binning and origin.

    Args:
        lon (numpy.ndarray): Array of longitude coordinates.
        lat (numpy.ndarray): Array of latitude coordinates.
        east_west_bin_km (float): Width of the east-west bins in kilometers.
        north_south_bin_km (float): Height of the north-south bins in kilometers.
        origin (tuple): Tuple containing the origin coordinates (latitude, longitude).

    Returns:
        numpy.ndarray: Array of block IDs.
    """
    assert np.max(origin) <= 180  # Assert that the maximum value of origin is within [-180, 180]
    assert np.min(origin) >= -180  # Assert that the minimum value of origin is within [-180, 180]
    easting_list_km = lon_to_global_easting(lon, origin)  # Convert longitude coordinates to global easting distances
    northing_list_km = lat_to_global_northing(lat, origin)  # Convert latitude coordinates to global northing distances
    easting_bin_list, northing_bin_list = coords_to_bin(easting_list_km, northing_list_km, east_west_bin_km, north_south_bin_km)  # Convert the global easting and northing distances to bin indices
    block_id_list = []
    for i in range(len(easting_bin_list)):
        block_id_list.append("e" + str(easting_bin_list[i]).zfill(4) + "n" + str(northing_bin_list[i]).zfill(4))  # Generate block IDs by concatenating the easting and northing bin indices with appropriate formatting
    return np.array(block_id_list)

def get_model_parameters(args, n_features, n_outputs):
    """
    Get the model parameters based on the chosen model.

    Args:
        args (argparse.Namespace): Arguments object containing the framework configuration.
        n_features (int): Number of input features.
        n_outputs (int): Number of output classes.

    Returns:
        dict: Model parameters.
    """
    if args.model == 'mlp':
        model_parameters = {
            "input_dim": n_features,  # The number of input features
            "num_hidden_layers": args.num_hidden_layers,  # The number of hidden layers
            "num_neurons": args.num_neurons,  # The hidden dimension (number of neurons / number of non-linear activation functions)
            "output_dim": n_outputs,  # The number of output classes
            "activation_func": args.activation_func,  # The activtion function for the hidden layer
            "weight_init": args.weight_init,  # The choice of weight initializations
            "dropout_p": args.dropout_p,  # The dropout probability
            "use_batch_norm": args.use_batch_norm,  # The choice to enable or disable batch normalization
        }
    elif args.model == 'rfc':
        model_parameters = {
            "n_estimators": args.n_estimators,  # The number of trees in the forest
            "criterion": args.criterion,  # The function to measure the quality of a split
            "max_depth": args.max_depth,  # The maximum depth of the tree
            "min_samples_split": args.min_samples_split,  # The minimum number of samples required to split an internal node
            "min_samples_leaf": args.min_samples_leaf,  # The minimum number of samples required to be at a leaf node
            "min_weight_fraction_leaf": args.min_weight_fraction_leaf,  # The minimum weighted fraction of the sum total of weights (of all the input samples) required to be at a leaf node
            "max_features": args.max_features,  # The number of features to consider when looking for the best split
            "max_leaf_nodes": args.max_leaf_nodes,  # The value for which trees grow in best-first fashion
            "min_impurity_decrease": args.min_impurity_decrease,  # The value for which a node will be split if this split induces a decrease of the impurity greater than or equal to this value
            "bootstrap": args.bootstrap,  # The choice to use bootstrap samples when building trees
            "oob_score": args.oob_score,  # The choice to use out-of-bag samples to estimate the generalization score
            "n_jobs": args.n_jobs,  # The number of jobs to run in parallel
            "random_state": args.random_state,  # The value that controls both the randomness of the bootstrapping of the samples used when building trees and the sampling of the features to consider when looking for the best split at each node
            "verbose": args.n_estimators if args.verbose == 'n_estimators' else int(args.verbose),  # The value that controls the verbosity when fitting and predicting
            "warm_start": args.warm_start,  # The choice to reuse the solution of the previous call to fit and add mor estimators to the ensemble or to just fit a whole new forest
            "class_weight": args.class_weight,  # The weights associated with classes
            "ccp_alpha": args.ccp_alpha,  # The complexity parameter used for Minimal Cost-Complexity Pruning
            "max_samples": args.max_samples  # The number of samples to draw from X to train each base estimator if bootstrap is used
        }
    elif args.model == 'xgb':
        model_parameters = {
            "n_estimators": args.n_estimators,  # The number of boosting rounds
            "max_depth": args.max_depth,  # The maximum tree depth for base learners
            "max_leaves": args.max_leaves,  # The maximum number of leaves; 0 indicates no limit
            "max_bin": args.max_bin,  # The maximum number of bins per feature if using histogram-based algorithm
            "grow_policy": args.grow_policy,  # The tree growing policy
            "learning_rate": args.learning_rate,  # The boosting learning rate (xgb's 'eta')
            "verbosity": args.verbosity,  # The degree of verbosity
            "objective": args.objective,  # The learning task and the corresponding learning objective or a custom objective function to be used
            "booster": args.booster,  # The booster to use
            "tree_method": args.tree_method,  # The tree method to use
            "n_jobs": args.n_jobs,  # The number of parallel threads used to run xgboost
            "gamma": args.gamma,  # The minimum loss reduction required to make a further partition on a leaf node of the tree
            "min_child_weight": args.min_child_weight,  # The minimum sum of instance weight (hessian) needed in a child
            "max_delta_step": args.max_delta_step,  # The maximum delta step we allow each tree's weight estimation to be
            "subsample": args.subsample,  # The subsample ratio of the training instance
            "sampling_method": args.sampling_method,  # The sampling method
            "colsample_bytree": args.colsample_bytree,  # The subsample ratio of columns when constructing each tree
            "colsample_bylevel": args.colsample_bylevel,  # The subsample ratio of columns for each level
            "colsample_bynode": args.colsample_bynode,  # The subsample ratio of columns for each split
            "reg_alpha": args.reg_alpha,  # The L1 regularization term on weights (xgb's alpha)
            "reg_lambda": args.reg_lambda,  # The L2 regularization term on weights (xgb's lambda)
            "scale_pos_weight": args.scale_pos_weight,  # The balancing of positive and negative weights
            "base_score": args.base_score,  # The initial prediction score of all instances, global bias
            "random_state": args.random_state,  # The random number seed
            "missing": args.missing,  # The value in the data which needs to be present as a missing value
            "num_parallel_tree": args.num_parallel_tree,  # The value used for boosting random forest
            "monotone_constraints": args.monotone_constraints,  # The constraint of variable monotonicity
            "interaction_constraints": args.interaction_constraints,  # The constraints for interaction representing permitted interactions
            "importance_type": args.importance_type,  # The feature importance type for the feature importances property
            "gpu_id": args.gpu_id,  # The device ordinal
            "validate_parameters": args.validate_parameters,  # The choice to give warnings for unknown parameter
            "predictor": args.predictor,  # The value forces XGBoost to use specific predictor
            "enable_categorical": args.enable_categorical,  # The support for categorical data
            "feature_types": args.feature_types,  # The value used for specifying feature types without constructing a dataframe
            "max_cat_to_onehot": args.max_cat_to_onehot,  # The threshold for deciding whether XGBoost should use one-hot encoding based split for categorical data
            "max_cat_threshold": args.max_cat_threshold,  # The maximum number of categories considered for each split
            "eval_metric": args.eval_metric,  # The metric used for monitoring the training result and early stopping
            "early_stopping_rounds": args.early_stopping_rounds,  # The early stopping activation
            "callbacks": args.callbacks,  # The list of callback functions that are applied at end of each iteration
            "use_label_encoder": args.use_label_encoder  # The choice to use label encoder
        }
    elif args.model == 'tnc':
        model_parameters = {
            "n_d": args.n_d,  # The width of the decision prediction layer
            "n_a": args.n_a,  # The width of the attention embedding for each mask
            "n_steps": args.n_steps,  # The number of steps in the architecture (usually between 3 and 10)
            "gamma": args.gamma,  # The coefficient for feature reusage in the masks
            "cat_idxs": args.cat_idxs,  # The list of categorical features indices
            "cat_dims": args.cat_dims,  # The list of categorical features number of modalities (number of unique values for a categorical feature)
            "cat_emb_dim": args.cat_emb_dim,  # The list of embeddings size for each categorical features
            "n_independent": args.n_independent,  # The number of independent Gated Linear Units layers at each step
            "n_shared": args.n_shared,  # The number of shared Gated Linear Units at each step
            "epsilon": args.epsilon,  # The value should be left untouched
            "seed": args.seed,  # The random seed for reproducibility
            "momentum": args.momentum,  # The momentum for batch normalization
            "clip_value": args.clip_value,  # The gradient will be clipped at the given value
            "lambda_sparse": args.lambda_sparse,  # The extra sparsity loss coefficient
            "optimizer_fn": torch.optim.Adam if args.optimizer_fn == 'adam' else '?',  # The PyTorch optimizer function
            "optimizer_params": args.optimizer_params,  # The parameters compatible with the PyTorch optimizer function used to initialize the optimizer
            "scheduler_fn": get_scheduler(args, optimizer=None),  # The PyTorch scheduler to change learning rates during training
            "scheduler_params": args.scheduler_params,  # The dictionnary of parameters to apply to the PyTorch scheduler
            "model_name": args.model_name,  # The name of the model used for saving in disk
            "verbose": args.verbose,  # The verbosity for notebooks plots
            "device_name": args.device_name,  # The name of the device used for training
            "input_dim": n_features,  # The number of input features
            "output_dim": n_outputs,  # The number of output classes
            "mask_type": args.mask_type,  # The masking function to use for selection features
            "grouped_features": args.grouped_features,  # The groups to allow the model to share its attention across features inside a same group
            "n_shared_decoder": args.n_shared_decoder,  # The number of shared GLU block in decoder
            "n_indep_decoder": args.n_indep_decoder  # The number of independent GLU block in decoder
        }
    elif args.model == 'ftt':
        model_parameters = {
            "n_num_features": n_features,  # The number of input features
            "cat_cardinalities": args.cat_cardinalities,  # The number of unique values for each feature
            "d_token": args.d_token,  # The size of one token
            "n_blocks": args.n_blocks,  # The number of Transformer blocks
            "attention_n_heads": args.attention_n_heads,  # The number of attention heads
            "attention_dropout": args.attention_dropout,  # The dropout for attention blocks
            "attention_initialization": args.attention_initialization,  # The initialization for attention blocks
            "attention_normalization": args.attention_normalization,  # The normalization for attention blocks
            "ffn_d_hidden": args.ffn_d_hidden,  # The input size for the second linear layer in the Feed-Forward Network module
            "ffn_dropout": args.ffn_dropout,  # The dropout rate after the first linear layer in the Feed-Forward Network module
            "ffn_activation": args.ffn_activation,  # The activation used in the Feed-Forward Network
            "ffn_normalization": args.ffn_normalization,  # The normalization used in the Feed-Forward Network
            "residual_dropout": args.residual_dropout,  # The dropout rate rate for the output of each residual branch of all Transformer blocks
            "prenormalization": args.prenormalization,  # The choice to place normalizations at the beginning of each residual branch
            "first_prenormalization": args.first_prenormalization,  # The choice to keep the first normalization from the first Transformer layer
            "last_layer_query_idx": args.last_layer_query_idx,  # The indices of tokens that should be processed by the last Transformer block
            "n_tokens": None if int(args.kv_compression_ratio * n_features) == 0 else n_features + 1,  # The option for fast linear attention
            "kv_compression_ratio": None if int(args.kv_compression_ratio * n_features) == 0 else args.kv_compression_ratio,  # The choice to apply a technique to speed up attention modules when the number of features is large
            "kv_compression_sharing": None if int(args.kv_compression_ratio * n_features) == 0 else args.kv_compression_sharing,  # Weight sharing policy for the technique to speed up attention modules when the number of features is large
            "head_activation": args.head_activation,  # The activation used in the heads
            "head_normalization": args.head_normalization,  # The normalization used in the heads
            "d_out": n_outputs  # The number of output classes
        }
    return model_parameters

def get_fit_parameters(args, X_train, X_test, y_train, y_test):
    """
    Get the fit parameters based on the chosen model.

    Args:
        args (argparse.Namespace): Arguments object containing the framework configuration.
        X_train (scipy.sparse._csr.csr_matrix): Training input data.
        X_test (scipy.sparse._csr.csr_matrix): Testing input data.
        y_train (numpy.ndarray): Training target data.
        y_test (numpy.ndarray): Testing target data.

    Returns:
        dict: Fit parameters.
    """
    if args.model == 'tnc':
        fit_parameters = {
            'X_train': X_train,  # Training input data
            'y_train': y_train,  # Training target data
            'eval_set': [(X_test, y_test)],  # Evaluation set
            'eval_name': args.eval_name,  # The list of eval set names
            'eval_metric': args.eval_metric,  # The list of evaluation metrics
            'max_epochs': args.max_epochs,  # The maximum number of epochs for training
            'patience': args.patience,  # The number of consecutive epochs without improvement before performing early stopping
            'weights': args.weights,  # The sampling parameter
            'loss_fn': get_criterion(args, y_train),  # The loss function for training
            'batch_size': args.batch_size,  # The number of examples per batch
            'virtual_batch_size': args.virtual_batch_size,  # The size of the mini batches used for "Ghost Batch Normalization"
            'num_workers': get_available_subprocesses(args),  # The number or workers used in the data loader
            'drop_last': args.drop_last,  # The choice to drop last batch if not complete during training
            'callbacks': args.callbacks,  # The list of custom callbacks
            'pin_memory': args.pin_memory,  # Whether to set pin_memory to True or False during training
            'from_unsupervised': args.from_unsupervised,  # Use a previously self supervised model as starting weights
            'augmentations': args.augmentations,  # The choice to apply custom data augmentation pipeline during training
            'warm_start': args.warm_start,  # The choice to fit twice the same model and start from a warm start
            'compute_importance': args.compute_importance  # The choice to compute feature importance
        }
    return fit_parameters

class SparseDataset(torch.utils.data.Dataset):
    """
    Custom dataset class for sparse input data.
    """
    def __init__(self, X, y):
        """
        Initialize the SparseDataset.

        Args:
            X (scipy.sparse._csr.csr_matrix): Sparse input data.
            y (numpy.ndarray): Target data.
        """
        self.X = X  # Sparse input data
        self.y = y  # Target data

    def __getitem__(self, index):
        """
        Get a single item from the dataset.

        Args:
            index (int): Index of the item to retrieve.

        Returns:
            tuple: A tuple containing the input data and its corresponding target.
        """
        X = torch.from_numpy(self.X[index].toarray()[0]).float()  # Convert the sparse input data to a dense tensor
        y = self.y[index]  # Target data
        return X, y

    def __len__(self):
        """
        Get the total number of items in the dataset.

        Returns:
            int: Total number of items in the dataset.
        """
        return self.X.shape[0]  # Number of items in the dataset
