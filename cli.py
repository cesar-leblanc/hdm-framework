from utils import bool_type

def add_all_parsers(parser):
    """
    Add all the necessary parsers based on the selected pipeline and model.

    Args:
        parser (argparse.ArgumentParser): The argument parser object.
    """
    _add_pipeline_parser(parser)  # Add the pipeline parser
    if parser.parse_known_args()[0].pipeline == 'check':  # Check if the selected pipeline is "check"
        _add_check_parser(parser)  # Add corresponding parser
    if parser.parse_known_args()[0].pipeline == 'dataset':  # Check if the selected pipeline is "dataset"
        _add_dataset_parser(parser)  # Add corresponding parser
    elif parser.parse_known_args()[0].pipeline == 'evaluation':  # Check if the selected pipeline is "evaluation"
        _add_evaluation_parser(parser)  # Add corresponding parser
    elif parser.parse_known_args()[0].pipeline == 'training':  # Check if the selected pipeline is "training"
        _add_training_parser(parser)  # Add corresponding parser
    elif parser.parse_known_args()[0].pipeline == 'prediction':  # Check if the selected pipeline is "prediction"
        _add_prediction_parser(parser)  # Add corresponding parser
    if parser.parse_known_args()[0].pipeline in ['evaluation', 'training', 'prediction']:  # Check if the selected pipeline is "evaluation", "training" or "prediction"
        if parser.parse_known_args()[0].model == 'mlp':  # Check if the selected model is MLP
            _add_mlp_parser(parser)  # Add corresponding parser
        elif parser.parse_known_args()[0].model == 'rfc':  # Check if the selected model is RFC
            _add_rfc_parser(parser)  # Add corresponding parser
        elif parser.parse_known_args()[0].model == 'xgb':  # Check if the selected model is XGB
            _add_xgb_parser(parser)  # Add corresponding parser
        elif parser.parse_known_args()[0].model == 'tnc':  # Check if the selected model is TNC
            _add_tnc_parser(parser)  # Add corresponding parser
        elif parser.parse_known_args()[0].model == 'ftt':  # Check if the selected model is FTT
            _add_ftt_parser(parser)  # Add corresponding parser
        _add_preprocessing_parser(parser)  # Add parser for preprocessing
    _add_filepaths_parser(parser)  # Add parser for filepaths
    _add_miscellaneous_parser(parser)  # Add miscellaneous parser

def _add_pipeline_parser(parser):
    """
    Add the pipeline parser to the argument parser.

    Args:
        parser (argparse.ArgumentParser): The argument parser object.
    """
    group_pipeline = parser.add_argument_group('Pipeline parameters')  # Create argument group for pipeline parameters
    group_pipeline.add_argument('--pipeline', required=True, choices=['check', 'dataset', 'evaluation', 'training', 'prediction'], help='The pipeline to choose.')

def _add_check_parser(parser):
    """
    Add the check parser to the argument parser.

    Args:
        parser (argparse.ArgumentParser): The argument parser object.
    """
    group_check = parser.add_argument_group('Check parameters')  # Create argument group for check parameters
    group_check.add_argument('--check_dependencies', type=bool_type, default=True, help='The choice to check if all the required dependencies are installed.')
    group_check.add_argument('--check_files', type=bool_type, default=True, help='The choice to check if all the necessary files and directories exist.')
    group_check.add_argument('--check_environment', type=bool_type, default=True, help='The choice to check if the environment is properly configured.')

def _add_dataset_parser(parser):
    """
    Add the dataset parser to the argument parser.

    Args:
        parser (argparse.ArgumentParser): The argument parser object.
    """
    group_dataset = parser.add_argument_group('Dataset parameters')  # Create argument group for dataset parameters
    group_dataset.add_argument('--level', default=3, choices=[1, 2, 3], help='The level of the EUNIS hierarchy.')
    group_dataset.add_argument('--min_year', type=int, default=0, help='The year from which we start keeping plots.')
    group_dataset.add_argument('--occurrences', type=int, default=10, help='The minimum number of occurrences for habitats and species.')
    group_dataset.add_argument('--countries', type=list, default=['all'], help='The list of countries to keep.')
    group_dataset.add_argument('--gbif_normalization', type=bool_type, default=True, help='The choice to normalize species names against the GBIF backbone.')
    group_dataset.add_argument('--folds', type=int, default=10, help='The number of folds for cross-validation.')
    group_dataset.add_argument('--east_west_bin_width_km', type=float, default=10.0, help='The width in kilometers for the east-west bins to use for the spatial split.')
    group_dataset.add_argument('--north_south_bin_width_km', type=float, default=10.0, help='The width in kilometers for the north-south bins to use for the spatial split.')
    group_dataset.add_argument('--location_features', type=bool_type, default=True, help='The choice to include several location-based features.')

def _add_evaluation_parser(parser):
    """
    Add the evaluation parser to the argument parser.

    Args:
        parser (argparse.ArgumentParser): The argument parser object.
    """
    group_evaluation = parser.add_argument_group('Evaluation parameters')  # Create argument group for evaluation parameters
    group_evaluation.add_argument('--model', default='mlp', choices=['mlp', 'rfc', 'ftt', 'xgb', 'tnc'], help='The classifier model to evaluate.')
    group_evaluation.add_argument('--metric', default='accuracy', choices=['accuracy'], help='The classification score.')
    group_evaluation.add_argument('--average', default='micro', choices=['micro', 'macro'], help='The reduction that is applied over labels.')
    group_evaluation.add_argument('--k', type=int, default=1, help='The number of most likely outcomes considered to find the correct label.')
    group_evaluation.add_argument('--n_folds', type=int, default=10, help='The number of folds.')

def _add_training_parser(parser):
    """
    Add the training parser to the argument parser.

    Args:
        parser (argparse.ArgumentParser): The argument parser object.
    """
    group_training = parser.add_argument_group('Training parameters')  # Create argument group for training parameters
    group_training.add_argument('--model', default='mlp', choices=['mlp', 'rfc', 'ftt', 'xgb', 'tnc'], help='The classifier model to train.')
    group_training.add_argument('--metric', default='accuracy', choices=['accuracy'], help='The classification score.')
    group_training.add_argument('--average', default='micro', choices=['micro', 'macro'], help='The reduction that is applied over labels.')
    group_training.add_argument('--k', type=int, default=1, help='The number of most likely outcomes considered to find the correct label.')

def _add_prediction_parser(parser):
    """
    Add the prediction parser to the argument parser.

    Args:
        parser (argparse.ArgumentParser): The argument parser object.
    """
    group_prediction = parser.add_argument_group('Prediction parameters')  # Create argument group for prediction parameters
    group_prediction.add_argument('--model', default='mlp', choices=['mlp', 'rfc', 'ftt', 'xgb', 'tnc'], help='The classifier model to predict.')
    group_prediction.add_argument('--gbif_normalization', type=bool_type, default=True, help='The choice to normalize species names against the GBIF backbone.')
    group_prediction.add_argument('--location_features', type=bool_type, default=True, help='The choice to include several location-based features.')

def _add_mlp_parser(parser):
    """
    Add the MLP parser to the argument parser.

    Args:
        parser (argparse.ArgumentParser): The argument parser object.
    """
    group_mlp = parser.add_argument_group('Multi-layer perceptron classifier parameters')  # Create argument group for MLP parameters
    group_mlp.add_argument('--num_hidden_layers', type=int, default=1, help='The number of hidden layers.')
    group_mlp.add_argument('--num_neurons', type=int, default=200, help='The hidden dimension (number of neurons / number of non-linear activation functions).')
    group_mlp.add_argument('--activation_func', default='relu', choices=['relu', 'sigmoid', 'tanh'], help='The activation function for the hidden layer.')
    group_mlp.add_argument('--optimizer', default='adam', choices=['sgd', 'adam'], help='The optimizer class.')
    group_mlp.add_argument('--learning_rate_init', type=float, default=0.001, help='The initial learning rate to use (controls the step-size in updating the weights).')
    group_mlp.add_argument('--batch_size', type=int, default=256, help='The size of minibatches.')
    group_mlp.add_argument('--criterion', default='cross-entropy', choices=['cross-entropy', 'bal-noised-top-k', 'imbal-noised-top-k'], help='The loss class.')
    group_mlp.add_argument('--weight_decay', type=float, default=0.0001, help='The strength of the L2 regularization term.')
    group_mlp.add_argument('--scheduler', default='plateau', choices=['plateau'], help='The step learning scheduler class.')
    group_mlp.add_argument('--num_epochs', type=int, default=100, help='The number of epochs (how many times each data point will be used).')
    group_mlp.add_argument('--num_iter_no_change', type=int, default=10, help='The patience parameter (maximum number of epochs to not improve).')
    group_mlp.add_argument('--momentum', type=float, default=0.9, help='The momentum for gradient descent update.')
    group_mlp.add_argument('--weight_init', default=None, choices=['kaiming', 'xavier'], help='The choice of weight initializations.') 
    group_mlp.add_argument('--dropout_p', type=float, default=0, help='The dropout probability.')
    group_mlp.add_argument('--use_batch_norm', type=bool_type, default=False, help='The choice to enable or disable batch normalization.')
    group_mlp.add_argument('--nesterov', type=bool_type, default=True, help="The choice to use Nesterov's momentum.")
    group_mlp.add_argument('--seed', type=int, default=123, help='The random seed for reproducibility.')
    group_mlp.add_argument('--factor', type=float, default=0.1, help='The factor by which the learning rate will be reduced.')
    group_mlp.add_argument('--patience', type=int, default=5, help='The number of epochs with no improvement after which learning rate will be reduced.')
    group_mlp.add_argument('--num_workers', type=int, default=-1, help='The number of subprocesses to use for data loading.')

def _add_rfc_parser(parser):
    """
    Add the RFC parser to the argument parser.

    Args:
        parser (argparse.ArgumentParser): The argument parser object.
    """
    group_rfc = parser.add_argument_group('Random forest classifier parameters')  # Create argument group for RFC parameters
    group_rfc.add_argument('--n_estimators', type=int, default=200, help='The number of trees in the forest.')
    group_rfc.add_argument('--criterion', default='gini', choices=['gini', 'entropy', 'log_loss'], help='The function to measure the quality of a split.')
    group_rfc.add_argument('--max_depth', type=int, default=None, help='The maximum depth of the tree.')
    group_rfc.add_argument('--min_samples_split', type=int, default=2, help='The minimum number of samples required to split an internal node.')
    group_rfc.add_argument('--min_samples_leaf', type=int, default=1, help='The minimum number of samples required to be at a leaf node.')
    group_rfc.add_argument('--min_weight_fraction_leaf', type=float, default=0.0, help='The minimum weighted fraction of the sum total of weights (of all the input samples) required to be at a leaf node.')
    group_rfc.add_argument('--max_features', default='sqrt', choices=['sqrt', 'log2', None], help='The number of features to consider when looking for the best split.')
    group_rfc.add_argument('--max_leaf_nodes', type=int, default=None, help='The value for which trees grow in best-first fashion.')
    group_rfc.add_argument('--min_impurity_decrease', type=float, default=0.0, help='The value for which a node will be split if this split induces a decrease of the impurity greater than or equal to this value.')
    group_rfc.add_argument('--bootstrap', type=bool_type, default=True, help='The choice to use bootstrap samples when building trees.')
    group_rfc.add_argument('--oob_score', type=bool_type, default=False, help='The choice to use out-of-bag samples to estimate the generalization score.')
    group_rfc.add_argument('--n_jobs', type=int, default=-1, help='The number of jobs to run in parallel.')
    group_rfc.add_argument('--random_state', type=int, default=123, help='The value that controls both the randomness of the bootstrapping of the samples used when building trees and the sampling of the features to consider when looking for the best split at each node.')
    group_rfc.add_argument('--verbose', default='n_estimators', choices=['n_estimators'] + list(map(str, range(0, 1000001))), help='The value that controls the verbosity when fitting and predicting.')
    group_rfc.add_argument('--warm_start', type=bool_type, default=False, help='The choice to reuse the solution of the previous call to fit and add more estimators to the ensemble or to just fit a whole new forest.')
    group_rfc.add_argument('--class_weight', default=None, choices=['balanced', 'balanced_subsample', None], help='The weights associated with classes.')
    group_rfc.add_argument('--ccp_alpha', type=float, default=0.0, help='The complexity parameter used for Minimal Cost-Complexity Pruning.')
    group_rfc.add_argument('--max_samples', type=float, default=None, help='The number of samples to draw from X to train each base estimator if bootstrap is used.')

def _add_xgb_parser(parser):
    """
    Add the XGB parser to the argument parser.

    Args:
        parser (argparse.ArgumentParser): The argument parser object.
    """
    group_xgb = parser.add_argument_group('Extreme boosting classifier parameters')  # Create argument group for XGB parameters
    group_xgb.add_argument('--n_estimators', type=int, default=200, help='The number of boosting rounds.')
    group_xgb.add_argument('--max_depth', type=int, default=100, help='The maximum tree depth for base learners.')
    group_xgb.add_argument('--max_leaves', type=int, default=0, help='The maximum number of leaves; 0 indicates no limit.')
    group_xgb.add_argument('--max_bin', type=int, default=256, help='The maximum number of bins per feature if using histogram-based algorithm.')
    group_xgb.add_argument('--grow_policy', default='depthwise', choices=['depthwise', 'lossguide'], help='The tree growing policy.')
    group_xgb.add_argument('--learning_rate', type=float, default=0.1, help="The boosting learning rate (xgb's 'eta').")
    group_xgb.add_argument('--verbosity', type=int, default=1, help='The degree of verbosity.')
    group_xgb.add_argument('--objective', default='multi:softmax', choices=['multi:softmax', 'multi:softprob'], help='The learning task and the corresponding learning objective or a custom objective function to be used.')
    group_xgb.add_argument('--booster', default='gbtree', choices=['gbtree', 'gblinear', 'dart'], help='The booster to use.')
    group_xgb.add_argument('--tree_method', default='auto', choices=['auto', 'exact', 'approx', 'hist', 'gpu_hist'], help='The tree method to use.')
    group_xgb.add_argument('--n_jobs', type=int, default=-1, help='The number of parallel threads used to run xgboost.')
    group_xgb.add_argument('--gamma', type=float, default=0, help='The minimum loss reduction required to make a further partition on a leaf node of the tree.')
    group_xgb.add_argument('--min_child_weight', type=float, default=1, help='The minimum sum of instance weight (hessian) needed in a child.')
    group_xgb.add_argument('--max_delta_step', type=float, default=0, help="The maximum delta step we allow each tree's weight estimation to be.")
    group_xgb.add_argument('--subsample', type=float, default=0.8, help='The subsample ratio of the training instance.')
    group_xgb.add_argument('--sampling_method', default='uniform', choices=['uniform', 'gradient_based'], help='The sampling method.')
    group_xgb.add_argument('--colsample_bytree', type=float, default=1, help='The subsample ratio of columns when constructing each tree.')
    group_xgb.add_argument('--colsample_bylevel', type=float, default=1, help='The subsample ratio of columns for each level.')
    group_xgb.add_argument('--colsample_bynode', type=float, default=1, help='The subsample ratio of columns for each split.')
    group_xgb.add_argument('--reg_alpha', type=float, default=0, help="The L1 regularization term on weights (xgb's alpha).")
    group_xgb.add_argument('--reg_lambda', type=float, default=1, help="The L2 regularization term on weights (xgb's lambda).")
    group_xgb.add_argument('--scale_pos_weight', type=float, default=None, help='The balancing of positive and negative weights.')
    group_xgb.add_argument('--base_score', type=float, default=None, help='The initial prediction score of all instances, global bias.')
    group_xgb.add_argument('--random_state', type=int, default=123, help='The random number seed.')
    group_xgb.add_argument('--missing', type=float, default=float('nan'), help='The value in the data which needs to be present as a missing value.')
    group_xgb.add_argument('--num_parallel_tree', type=int, default=1, help='The value used for boosting random forest.')
    group_xgb.add_argument('--monotone_constraints', type=str, default=None, help='The constraint of variable monotonicity.')
    group_xgb.add_argument('--interaction_constraints', type=str, default=None, help='The constraints for interaction representing permitted interactions.')
    group_xgb.add_argument('--importance_type', type=str, default=None, help='The feature importance type for the feature importances property.')
    group_xgb.add_argument('--gpu_id', type=int, default=None, help='The device ordinal.')
    group_xgb.add_argument('--validate_parameters', type=bool_type, default=None, help='The choice to give warnings for unknown parameter.')
    group_xgb.add_argument('--predictor', default='auto', choices=['auto', 'cpu_predictor', 'gpu_predictor'], help='The value forces XGBoost to use specific predictor.')
    group_xgb.add_argument('--enable_categorical', type=bool_type, default=None, help='The support for categorical data.')
    group_xgb.add_argument('--feature_types', type=str, default=None, help='The value used for specifying feature types without constructing a dataframe.')
    group_xgb.add_argument('--max_cat_to_onehot', type=int, default=None, help='The threshold for deciding whether XGBoost should use one-hot encoding based split for categorical data.')
    group_xgb.add_argument('--max_cat_threshold', type=int, default=None, help='The maximum number of categories considered for each split.')
    group_xgb.add_argument('--eval_metric', type=str, default='merror', help='The metric used for monitoring the training result and early stopping.')
    group_xgb.add_argument('--early_stopping_rounds', type=int, default=10, help='The early stopping activation.')
    group_xgb.add_argument('--callbacks', type=list, default=None, help='The list of callback functions that are applied at end of each iteration.')
    group_xgb.add_argument('--use_label_encoder', type=bool_type, default=None, help='The choice to use label encoder.')

def _add_tnc_parser(parser):
    """
    Add the TNC parser to the argument parser.

    Args:
        parser (argparse.ArgumentParser): The argument parser object.
    """
    group_tnc = parser.add_argument_group('Tabnet classifier parameters')  # Create argument group for TNC parameters
    group_tnc.add_argument('--n_d', type=int, default=16, help='The width of the decision prediction layer.')
    group_tnc.add_argument('--n_a', type=int, default=16, help='The width of the attention embedding for each mask.')
    group_tnc.add_argument('--n_steps', type=int, default=3, help='The number of steps in the architecture (usually between 3 and 10).')
    group_tnc.add_argument('--gamma', type=float, default=1.3, help="The coefficient for feature reusage in the masks.")
    group_tnc.add_argument('--cat_idxs', type=list, default=[], help='The list of categorical features indices.')
    group_tnc.add_argument('--cat_dims', type=list, default=[], help='The list of categorical features number of modalities (number of unique values for a categorical feature).')
    group_tnc.add_argument('--cat_emb_dim', type=list, default=1, help='The list of embeddings size for each categorical features.')
    group_tnc.add_argument('--n_independent', type=int, default=1, help='The number of independent Gated Linear Units layers at each step.')
    group_tnc.add_argument('--n_shared', type=int, default=3, help='The number of shared Gated Linear Units at each step.')
    group_tnc.add_argument('--epsilon', type=float, default=1e-15, help='The value should be left untouched.')
    group_tnc.add_argument('--seed', type=int, default=123, help='The random seed for reproducibility.')
    group_tnc.add_argument('--momentum', type=float, default=0.02, help='The momentum for batch normalization.')
    group_tnc.add_argument('--clip_value', type=float, default=None, help='The gradient will be clipped at the given value.')
    group_tnc.add_argument('--lambda_sparse', type=float, default=1e-3, help='The extra sparsity loss coefficient.')
    group_tnc.add_argument('--optimizer_fn', default='adam', choices=['adam'], help='The PyTorch optimizer function.')
    group_tnc.add_argument('--optimizer_params', type=dict, default={'lr':0.02}, help='The parameters compatible with the PyTorch optimizer function used to initialize the optimizer.')
    group_tnc.add_argument('--scheduler_fn', default='step', choices=['step'], help='The PyTorch scheduler to change learning rates during training.')
    group_tnc.add_argument('--scheduler_params', type=dict, default={"step_size": 10, "gamma": 0.9}, help='The dictionary of parameters to apply to the PyTorch scheduler.')
    group_tnc.add_argument('--model_name', type=str, default='TabNetClassifier', help='The name of the model used for saving in disk.')
    group_tnc.add_argument('--verbose', type=int, default=1, help='The verbosity for notebooks plots.')
    group_tnc.add_argument('--device_name', type=str, default='auto', help='The name of the device used for training.')
    group_tnc.add_argument('--mask_type', default='sparsemax', choices=['sparsemax', 'entmax'], help='The masking function to use for selection features.')
    group_tnc.add_argument('--grouped_features', type=list, default=[], help='The groups to allow the model to share its attention across features inside a same group.')
    group_tnc.add_argument('--n_shared_decoder', type=int, default=1, help='The number of shared GLU block in decoder.')
    group_tnc.add_argument('--n_indep_decoder', type=int, default=1, help='The number of independent GLU block in decoder.')
    group_tnc.add_argument('--eval_name', type=list, default=['eval'], help='The list of eval set names.')
    group_tnc.add_argument('--eval_metric', default=['accuracy'], choices=["accuracy", "balanced_accuracy", "logloss"], help='The list of evaluation metrics.')
    group_tnc.add_argument('--max_epochs', type=int, default=100, help='The maximum number of epochs for training.')
    group_tnc.add_argument('--patience', type=int, default=10, help='The number of consecutive epochs without improvement before performing early stopping.')
    group_tnc.add_argument('--weights', default=0, choices = [0, 1], help='The sampling parameter.')
    group_tnc.add_argument('--loss_fn', default='cross-entropy', choices=['cross-entropy'], help='The loss function for training.')    
    group_tnc.add_argument('--batch_size', type=int, default=1024, help='The number of examples per batch.')
    group_tnc.add_argument('--virtual_batch_size', type=int, default=128, help='The size of the mini batches used for "Ghost Batch Normalization".')
    group_tnc.add_argument('--num_workers', type=int, default=-1, help='The number or workers used in the data loader.')
    group_tnc.add_argument('--drop_last', type=bool_type, default=False, help='The choice to drop last batch if not complete during training.')
    group_tnc.add_argument('--callbacks', type=list, default=[], help='The list of custom callbacks.')
    group_tnc.add_argument('--pin_memory', type=bool_type, default=True, help='Whether to set pin_memory to True or False during training.')
    group_tnc.add_argument('--from_unsupervised', type=str, default=None, help='Use a previously self supervised model as starting weights.')
    group_tnc.add_argument('--augmentations', type=str, default=None, help='The choice to apply custom data augmentation pipeline during training.')
    group_tnc.add_argument('--warm_start', type=bool_type, default=False, help='The choice to fit twice the same model and start from a warm start.')
    group_tnc.add_argument('--compute_importance', type=bool_type, default=False, help='The choice to compute feature importance.')

def _add_ftt_parser(parser):
    """
    Add the FTT parser to the argument parser.

    Args:
        parser (argparse.ArgumentParser): The argument parser object.
    """
    group_ftt = parser.add_argument_group('Feature-tokenizer + transformer parameters')  # Create argument group for FTT parameters
    group_ftt.add_argument('--cat_cardinalities', type=list, default=None, help='The number of unique values for each feature.')
    group_ftt.add_argument('--d_token', type=int, default=16, help='The size of one token.')
    group_ftt.add_argument('--n_blocks', type=int, default=1, help='The number of Transformer blocks.')
    group_ftt.add_argument('--attention_n_heads', type=int, default=4, help='The number of attention heads.')
    group_ftt.add_argument('--attention_dropout', type=float, default=0.3, help='The dropout for attention blocks.')
    group_ftt.add_argument('--attention_initialization', type=str, default='kaiming', help='The initialization for attention blocks.')
    group_ftt.add_argument('--attention_normalization', type=str, default='LayerNorm', help='The normalization for attention blocks.')
    group_ftt.add_argument('--ffn_d_hidden', type=int, default=16, help='The input size for the second linear layer in the Feed-Forward Network module.')
    group_ftt.add_argument('--ffn_dropout', type=float, default=0.1, help='The dropout rate after the first linear layer in the Feed-Forward Network module.')
    group_ftt.add_argument('--ffn_activation', type=str, default='ReGLU', help='The activation used in the Feed-Forward Network.')
    group_ftt.add_argument('--ffn_normalization', type=str, default='LayerNorm', help='The normalization used in the Feed-Forward Network.')
    group_ftt.add_argument('--residual_dropout', type=float, default=0.0, help='The dropout rate rate for the output of each residual branch of all Transformer blocks.')
    group_ftt.add_argument('--prenormalization', type=bool_type, default=True, help='The choice to place normalizations at the beginning of each residual branch.')
    group_ftt.add_argument('--first_prenormalization', type=bool_type, default=False, help='The choice to keep the first normalization from the first Transformer layer.')
    group_ftt.add_argument('--last_layer_query_idx', type=list, default=[-1], help='The indices of tokens that should be processed by the last Transformer block.')
    group_ftt.add_argument('--kv_compression_ratio', type=float, default=0.004, help='The choice to apply a technique to speed up attention modules when the number of features is large.')
    group_ftt.add_argument('--kv_compression_sharing', default='headwise', choices=[None, 'headwise', 'key-value', 'layerwise'], help='Weight sharing policy for the technique to speed up attention modules when the number of features is large.')
    group_ftt.add_argument('--head_activation', type=str, default='ReLU', help='The activation used in the heads.')
    group_ftt.add_argument('--head_normalization', type=str, default='LayerNorm', help='The normalization used in the heads.')
    group_ftt.add_argument('--learning_rate_init', type=float, default=0.001, help='The initial learning rate to use (controls the step-size in updating the weights).')
    group_ftt.add_argument('--optimizer', default='adamw', choices=['adamw'], help='The optimizer class.')
    group_ftt.add_argument('--batch_size', type=int, default=512, help='The size of minibatches.')
    group_ftt.add_argument('--criterion', default='cross-entropy', choices=['cross-entropy'], help='The loss class.')
    group_ftt.add_argument('--weight_decay', type=float, default=0.0001, help='The strength of the L2 regularization term.')
    group_ftt.add_argument('--scheduler', default='plateau', choices=['plateau'], help='The step learning scheduler class.')
    group_ftt.add_argument('--num_epochs', type=int, default=100, help='The number of epochs (how many times each data point will be used).')
    group_ftt.add_argument('--num_iter_no_change', type=int, default=10, help='The patience parameter (maximum number of epochs to not improve).')    
    group_ftt.add_argument('--seed', type=int, default=123, help='The random seed for reproducibility.')
    group_ftt.add_argument('--factor', type=float, default=0.1, help='The factor by which the learning rate will be reduced.')
    group_ftt.add_argument('--patience', type=int, default=5, help='The number of epochs with no improvement after which learning rate will be reduced.')
    group_ftt.add_argument('--num_workers', type=int, default=-1, help='The number of subprocesses to use for data loading.')

def _add_preprocessing_parser(parser):
    """
    Add the preprocessing parser to the argument parser.

    Args:
        parser (argparse.ArgumentParser): The argument parser object.
    """
    group_preprocessing = parser.add_argument_group('Preprocessing parameters')  # Create argument group for preprocessing parameters
    group_preprocessing.add_argument('--standardization', type=bool_type, default=True, help='The choice to standardize features by removing the mean and scaling to unit variance.')
    group_preprocessing.add_argument('--binarization', type=bool_type, default=False, help='The choice to binarize data (set feature values to 0 or 1) according to a threshold.')
    group_preprocessing.add_argument('--dropout', type=float, default=0.0, help='The dropout probability for dropping out the features.')
    group_preprocessing.add_argument('--noise', type=bool_type, default=False, help='The choice to add noise with random samples drawn from a normal distribution.')
    group_preprocessing.add_argument('--normalization', type=bool_type, default=False, help='The choice to normalize samples individually to unit norm.')
    group_preprocessing.add_argument('--log', type=bool_type, default=False, help='The choice to apply the natural logarithm to the features.')
    group_preprocessing.add_argument('--rank', type=bool_type, default=False, help='The choice to rank the features.')
    group_preprocessing.add_argument('--augmentation', type=bool_type, default=False, help='The choice to apply custom data augmentation pipeline during training.')    
    group_preprocessing.add_argument('--rare', type=bool_type, default=False, help='The choice to keep only the rarest labels.')
    group_preprocessing.add_argument('--endangered', type=bool_type, default=False, help='The choice to keep only the endangered habitats.')    
    group_preprocessing.add_argument('--features', type=list, default=['all'], help='The list of features the model will see during fit.')

def _add_filepaths_parser(parser):
    """
    Add the filepaths parser to the argument parser.

    Args:
        parser (argparse.ArgumentParser): The argument parser object.
    """
    group_data = parser.add_argument_group('Filepaths parameters')  # Create argument group for filepaths parameters
    group_data.add_argument('--data_filepath', type=str, default='Data/', help='The filepath to load/save the data from/to a disk file.')
    group_data.add_argument('--datasets_filepath', type=str, default='Datasets/', help='The filepath to load/save the datasets from/to a disk file.')
    group_data.add_argument('--models_filepath', type=str, default='Models/', help='The filepath to load/save the model from/to a disk file.')

def _add_miscellaneous_parser(parser):
    """
    Add the miscellaneous parser to the argument parser.

    Args:
        parser (argparse.ArgumentParser): The argument parser object.
    """
    group_miscellaneous = parser.add_argument_group('Miscellaneous parameters')  # Create argument group for miscellaneous parameters
    group_miscellaneous.add_argument('--print_parameters', type=bool_type, default=True, help='The choice to print the list of parameters.')
    group_miscellaneous.add_argument('--global_seed', type=int, default=123, help='The seed for reproducible output from run to run.')
    group_miscellaneous.add_argument('--use_gpu', type=bool_type, default=True, help='The choice to use GPU for the model.')
    group_miscellaneous.add_argument('--data_parallelism', type=bool_type, default=True, help='The choice to implement data parallelism at the module level.')
    group_miscellaneous.add_argument('--write_bytecode', type=bool_type, default=False, help='The choice to enable or disable bytecode generation.')
    group_miscellaneous.add_argument('--disable_warnings', type=bool_type, default=True, help='The choice to enable or disable warnings.')
