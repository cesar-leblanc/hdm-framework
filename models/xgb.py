import xgboost
import torch
import numpy as np
import time

from utils import get_metric, get_model_parameters

from data.load_data import get_split_assignments, get_eunis_red_list_crosswalks, get_endangered_red_list_habitats, get_endangered_eunis_habitats, get_xgb_model, get_scaler
from data.preprocess_data import add_fold_assignments, add_standardization, add_normalization, add_dropout, add_binarization, add_log, add_noise, add_rank, remove_features, add_augmentation, add_endangered_habitats, add_predictions_decoding
from data.save_data import set_scaler, set_xgb_model, set_predictions

class XGB(xgboost.XGBClassifier):
    """
    XGBoost Classifier for habitat distribution modeling.
    """
    def __init__(self, n_estimators=200, max_depth=100, max_leaves=0, max_bin=256, grow_policy='depthwise', learning_rate=0.1, verbosity=1, objective='multi:softmax', booster='gbtree', tree_method='auto', n_jobs=-1, gamma=0, min_child_weight=1, max_delta_step=0, subsample=0.8, sampling_method='uniform', colsample_bytree=1, colsample_bylevel=1, colsample_bynode=1, reg_alpha=0, reg_lambda=1, scale_pos_weight=None, base_score=None, random_state=123, missing=float('nan'), num_parallel_tree=1, monotone_constraints=None, interaction_constraints=None, importance_type=None, gpu_id=None, validate_parameters=None, predictor='auto', enable_categorical=None, feature_types=None, max_cat_to_onehot=None, max_cat_threshold=None, eval_metric='merror', early_stopping_rounds=10, callbacks=None, use_label_encoder=None):
        """
        Initialize the XGBoost Classifier.

        Args:
            n_estimators (int, optional): The number of boosting rounds. Defaults to 200.
            max_depth (int, optional): The maximum tree depth for base learners. Defaults to 100.
            max_leaves (int, optional): The maximum number of leaves; 0 indicates no limit. Defaults to 0.
            max_bin (int, optional): The maximum number of bins per feature if using histogram-based algorithm. Defaults to 256.
            grow_policy (str, optional): The tree growing policy. Defaults to 'depthwise'.
            learning_rate (float, optional): The boosting learning rate (xgb's 'eta'). Defaults to 0.1.
            verbosity (int, optional): The degree of verbosity. Defaults to 1.
            objective (str | callable, optional): The learning task and the corresponding learning objective or a custom objective function to be used. Defaults to 'multi:softmax'.
            booster (str, optional): The booster to use. Defaults to 'gbtree'.
            tree_method (str, optional): The tree method to use. Defaults to 'auto'.
            n_jobs (int, optional): The number of parallel threads used to run xgboost. Defaults to -1.
            gamma (float, optional): The minimum loss reduction required to make a further partition on a leaf node of the tree. Defaults to 0.
            min_child_weight (float, optional): The minimum sum of instance weight (hessian) needed in a child. Defaults to 1.
            max_delta_step (float, optional): The maximum delta step we allow each tree's weight estimation to be. Defaults to 0.
            subsample (float, optional): The subsample ratio of the training instance. Defaults to 0.8.
            sampling_method (str, optional): The sampling method. Defaults to 'uniform'.
            colsample_bytree (float, optional): The subsample ratio of columns when constructing each tree. Defaults to 1.
            colsample_bylevel (float, optional): The subsample ratio of columns for each level. Defaults to 1.
            colsample_bynode (float, optional): The subsample ratio of columns for each split. Defaults to 1.
            reg_alpha (float, optional): The L1 regularization term on weights (xgb's alpha). Defaults to 0.
            reg_lambda (float, optional): The L2 regularization term on weights (xgb's lambda). Defaults to 1.
            scale_pos_weight (float, optional): The balancing of positive and negative weights. Defaults to None.
            base_score (float, optional): The initial prediction score of all instances, global bias. Defaults to None.
            random_state (int, optional): The random number seed. Defaults to 123.
            missing (float | None, optional): The value in the data which needs to be present as a missing value. Defaults to float('nan').
            num_parallel_tree (int, optional): The value used for boosting random forest. Defaults to 1.
            monotone_constraints (str | list, optional): The constraint of variable monotonicity. Defaults to None.
            interaction_constraints (str | list, optional): The constraints for interaction representing permitted interactions. Defaults to None.
            importance_type (str, optional): The feature importance type for the feature importances property. Defaults to None.
            gpu_id (int, optional): The device ordinal. Defaults to None.
            validate_parameters (bool, optional): The choice to give warnings for unknown parameter. Defaults to None.
            predictor (str, optional): The value forces XGBoost to use a specific predictor. Defaults to 'auto'.
            enable_categorical (bool, optional): The support for categorical data. Defaults to None.
            feature_types (str | list, optional): The value used for specifying feature types without constructing a dataframe. Defaults to None.
            max_cat_to_onehot (int, optional): The threshold for deciding whether XGBoost should use one-hot encoding based split for categorical data. Defaults to None.
            max_cat_threshold (int, optional): The maximum number of categories considered for each split. Defaults to None.
            eval_metric (str, optional): The metric used for monitoring the training result and early stopping. Defaults to 'merror'.
            early_stopping_rounds (int, optional): The early stopping activation. Defaults to 10.
            callbacks (list, optional): The list of callback functions that are applied at end of each iteration. Defaults to None.
            use_label_encoder (bool, optional): The choice to use label encoder. Defaults to None.
        """
        super(XGB, self).__init__()  # Initialize the parent class, `xgboost.XGBClassifier`
        self.n_estimators = n_estimators  # The number of boosting rounds
        self.max_depth = max_depth  # The maximum tree depth for base learners
        self.max_leaves = max_leaves  # The maximum number of leaves; 0 indicates no limit
        self.max_bin = max_bin  # The maximum number of bins per feature if using histogram-based algorithm
        self.grow_policy = grow_policy  # The tree growing policy
        self.learning_rate = learning_rate  # The boosting learning rate (xgb's 'eta')
        self.verbosity = verbosity  # The degree of verbosity
        self.objective = objective  # The learning task and the corresponding learning objective or a custom objective function to be used
        self.booster = booster  # The booster to use
        self.tree_method = tree_method  # The tree method to use
        self.n_jobs = n_jobs  # The number of parallel threads used to run xgboost
        self.gamma = gamma  # The minimum loss reduction required to make a further partition on a leaf node of the tree
        self.min_child_weight = min_child_weight  # The minimum sum of instance weight (hessian) needed in a child
        self.max_delta_step = max_delta_step  # The maximum delta step we allow each tree's weight estimation to be
        self.subsample = subsample  # The subsample ratio of the training instance
        self.sampling_method = sampling_method  # The sampling method
        self.colsample_bytree = colsample_bytree  # The subsample ratio of columns when constructing each tree
        self.colsample_bylevel = colsample_bylevel  # The subsample ratio of columns for each level
        self.colsample_bynode = colsample_bynode  # The subsample ratio of columns for each split
        self.reg_alpha = reg_alpha  # The L1 regularization term on weights (xgb's alpha)
        self.reg_lambda = reg_lambda  # The L2 regularization term on weights (xgb's lambda)
        self.scale_pos_weight = scale_pos_weight  # The balancing of positive and negative weights
        self.base_score = base_score  # The initial prediction score of all instances, global bias
        self.random_state = random_state  # The random number seed
        self.missing = missing  # The value in the data which needs to be present as a missing value
        self.num_parallel_tree = num_parallel_tree  # The value used for boosting random forest
        self.monotone_constraints = monotone_constraints  # The constraint of variable monotonicity
        self.interaction_constraints = interaction_constraints  # The constraints for interaction representing permitted interactions
        self.importance_type = importance_type  # The feature importance type for the feature importances property
        self.gpu_id = gpu_id  # The device ordinal
        self.validate_parameters = validate_parameters  # The choice to give warnings for unknown parameter
        self.predictor = predictor  # The value forces XGBoost to use specific predictor
        self.enable_categorical = enable_categorical  # The support for categorical data
        self.feature_types = feature_types  # The value used for specifying feature types without constructing a dataframe
        self.max_cat_to_onehot = max_cat_to_onehot  # The threshold for deciding whether XGBoost should use one-hot encoding based split for categorical data
        self.max_cat_threshold = max_cat_threshold  # The maximum number of categories considered for each split
        self.eval_metric = eval_metric  # The metric used for monitoring the training result and early stopping
        self.early_stopping_rounds = early_stopping_rounds  # The early stopping activation
        self.callbacks = callbacks  # The list of callback functions that are applied at end of each iteration
        self.use_label_encoder = use_label_encoder  # The choice to use label encoder
        
    def evaluate_model(self, args, X, y, le_species, le_header):
        """
        Evaluate the model performance using cross-validation.

        Args:
            args (argparse.Namespace): Arguments passed to the framework.
            X (scipy.sparse._csr.csr_matrix): Input features.
            y (numpy.ndarray): Target labels.
            le_species (sklearn.preprocessing._label.LabelEncoder): Label encoder for species.
            le_header (sklearn.preprocessing._label.LabelEncoder): Label encoder for habitat classes.

        Returns:
            None
        """
        if args.rank:
            X = add_rank(X, len(le_species.classes_))  # Add rank information to the features
        if args.binarization:
            X = add_binarization(X, len(le_species.classes_))  # Perform binarization on the features
        if args.normalization:
            X = add_normalization(X, len(le_species.classes_))  # Perform normalization on the features
        std_acc = []
        avg_acc = 0.0  # Initialize variables to track the average accuracy
        n_samples = 0  # Initialize variables to track the number of samples
        split_assignments = get_split_assignments(args)  # Get assignments of samples to folds
        best_accuracy = -np.inf
        best_fold = -1
        start_evaluation = time.time()  # Start time of the evaluation
        for fold in range(args.n_folds):
            best_fold_accuracy = self.evaluate_fold(args, X, y, le_species, le_header, fold, split_assignments)  # Evaluate the model on the current fold
            if best_fold_accuracy > best_accuracy:
                best_accuracy = best_fold_accuracy  # Update the best accuracy
                best_fold = fold  # Update the best fold index
                print("\nFold {} did improve the model.".format(best_fold))
            else:
                print("\nFold {} did not improve the model.".format(fold))
            std_acc.append(best_fold_accuracy)  # Add the best fold accuracy to the list
            avg_acc += best_fold_accuracy  # Accumulate the accuracy
            n_samples += 1  # Accumulate the number of samples
        end_evaluation = time.time()  # End time of the evaluation
        time_evaluation = end_evaluation - start_evaluation  # Total evaluation time
        avg_acc /= n_samples  # Compute the average accuracy
        std_acc = torch.std(torch.FloatTensor(std_acc))  # Compute the standard deviation of accuracies
        print(f"\nMean {args.metric} and standard deviation after {args.n_folds} folds of {args.n_estimators} estimators: {avg_acc:.4f}% & {std_acc:.4f}.")
        print(f"Total time of the evaluation: {time_evaluation:.2f}s.")
        
    def train_model(self, args, X, y, le_species, le_header):
        """
        Train the model using the provided data.

        Args:
            args (argparse.Namespace): Arguments passed to the framework.
            X (scipy.sparse._csr.csr_matrix): Input features.
            y (numpy.ndarray): Target labels.
            le_species (sklearn.preprocessing._label.LabelEncoder): Label encoder for species.
            le_header (sklearn.preprocessing._label.LabelEncoder): Label encoder for habitat classes.

        Returns:
            None
        """
        if args.rank:
            X = add_rank(X, len(le_species.classes_))  # Add rank information to the features
        if args.binarization:
            X = add_binarization(X, len(le_species.classes_))  # Perform binarization on the features
        if args.normalization:
            X = add_normalization(X, len(le_species.classes_))  # Perform normalization on the features
        if args.endangered:
            eunis_red_list_crosswalks = get_eunis_red_list_crosswalks(args)
            endangered_red_list_habitats = get_endangered_red_list_habitats(args)
            endangered_eunis_habitats = get_endangered_eunis_habitats(eunis_red_list_crosswalks, endangered_red_list_habitats, le_header)
            X, y = add_endangered_habitats(X, y, endangered_eunis_habitats)  # Add endangered habitats to the data
        if args.augmentation:
            if not args.endangered:
                labels = None
            else:
                labels = endangered_eunis_habitats
            X, y = add_augmentation(X, y, labels)  # Perform data augmentation
        if args.dropout > 0:
            X = add_dropout(X, args.dropout, len(le_species.classes_))  # Apply dropout to the features
        if args.noise:
            X = add_noise(X, len(le_species.classes_))  # Add noise to the features
        if args.log:
            X = add_log(X, len(le_species.classes_))  # Apply logarithmic transformation to the features
        if args.features != ['all']:
            features = ''.join(args.features).replace(' ', '').split(',')
            X = remove_features(args, X, features)  # Remove specified features from the input
        if args.standardization:
            X, scaler = add_standardization(X, scaler=None)  # Perform feature standardization
            set_scaler(args, scaler)  # Save the scaler for future use
        model_parameters = get_model_parameters(args, X.shape[1], len(le_header.classes_))  # Get model parameters
        model = XGB(**model_parameters)  # Initialize the model
        print()
        start_training = time.time()  # Start time of the training
        model.fit(X, y, eval_set=[(X, y)])  # Train the model
        end_training = time.time()  # End time of the training
        time_training = end_training - start_training  # Total training time
        print(f"\nTotal time of the training: {time_training:.2f}s.")
        y_probas = model.predict_proba(X)  # Predict probabilities for the training data
        metric = get_metric(args, len(le_header.classes_))  # Get the evaluation metric
        accuracy = metric(torch.from_numpy(y_probas), torch.from_numpy(y)) * 100  # Compute the accuracy
        if model.best_iteration + args.early_stopping_rounds <= args.n_estimators:
            print(f'Early stopping occurred at iteration {model.best_iteration + args.early_stopping_rounds} with best iteration = {model.best_iteration} and best {args.metric} = {accuracy:.4f}%.')
        else:
            print(f'Early stopping did not occur as best iteration = {model.best_iteration} and best {args.metric} = {accuracy:.4f}')
        set_xgb_model(args, model)  # Set the trained model for future use
        print("\nSuccessfully saved model at Models/XGB.json")
        
    def predict_model(self, args, X, y, le_species, le_header):
        """
        Perform predictions using the trained model.

        Args:
            args (argparse.Namespace): Arguments passed to the framework.
            X (scipy.sparse._csr.csr_matrix): Input features.
            y (numpy.ndarray): Target labels.
            le_species (sklearn.preprocessing._label.LabelEncoder): Label encoder for species.
            le_header (sklearn.preprocessing._label.LabelEncoder): Label encoder for habitat classes.

        Returns:
            None
        """
        if args.rank:
            X = add_rank(X, len(le_species.classes_))  # Add rank information to the features
        if args.binarization:
            X = add_binarization(X, len(le_species.classes_))  # Perform binarization on the features
        if args.normalization:
            X = add_normalization(X, len(le_species.classes_))  # Perform normalization on the features
        if args.log:
            X = add_log(X, len(le_species.classes_))  # Apply logarithmic transformation to the features
        if args.features != ['all']:
            features = ''.join(args.features).replace(' ', '').split(',')
            X = remove_features(args, X, features)  # Remove specified features from the input
        if args.standardization:
            scaler = get_scaler(args)  # Get the scaler for feature standardization
            X, _ = add_standardization(X, scaler=scaler)  # Perform feature standardization
        model_parameters = get_model_parameters(args, X.shape[1], len(le_header.classes_))  # Get model parameters
        model = XGB(**model_parameters)  # Initialize the model
        model = get_xgb_model(args, model)  # Load the trained model
        start_prediction = time.time()  # Start time of the prediction
        predictions = model.predict(X)  # Perform predictions
        end_prediction = time.time()  # End time of the prediction
        time_prediction = end_prediction - start_prediction  # Total prediction time
        predictions = add_predictions_decoding(predictions, le_header)  # Decode the predicted labels
        set_predictions(args, predictions)  # Save the predictions
        print("\nSuccessfully saved predictions at Data/predictions.txt")
        print(f"Total time of the prediction: {time_prediction:.2f}s.")

    def evaluate_fold(self, args, X, y, le_species, le_header, fold, split_assignments):
        """
        Evaluate the model performance on a specific fold of the data.

        Args:
            args (argparse.Namespace): Arguments passed to the framework.
            X (scipy.sparse._csr.csr_matrix): Input features.
            y (numpy.ndarray): Target labels.
            le_species (sklearn.preprocessing._label.LabelEncoder): Label encoder for species.
            le_header (sklearn.preprocessing._label.LabelEncoder): Label encoder for habitat classes.
            fold (int): Fold number for evaluation.
            split_assignments (numpy.ndarray): Assigned data splits.

        Returns:
            float: Accuracy of the model on the given fold.
        """
        print('\n' + '*'*11) 
        print('* Fold: {} *'.format(fold))  # Print fold number
        print('*'*11 + '\n')
        X_train, X_test, y_train, y_test = add_fold_assignments(X, y, split_assignments, fold)  # Split data into train and test sets for the fold
        if args.endangered:
            eunis_red_list_crosswalks = get_eunis_red_list_crosswalks(args)  # Obtain EUNIS red list crosswalks
            endangered_red_list_habitats = get_endangered_red_list_habitats(args)  # Obtain endangered red list habitats
            endangered_eunis_habitats = get_endangered_eunis_habitats(eunis_red_list_crosswalks, endangered_red_list_habitats, le_header)  # Get endangered EUNIS habitats
            X_test, y_test = add_endangered_habitats(X_test, y_test, endangered_eunis_habitats)  # Add endangered habitats to the test data
        if args.augmentation:
            if not args.endangered:
                labels = None
            else:
                labels = endangered_eunis_habitats
            X_train, y_train = add_augmentation(X_train, y_train, labels)  # Perform data augmentation on the train data
        if args.dropout > 0:
            X_test = add_dropout(X_test, args.dropout, len(le_species.classes_))  # Apply dropout to the test data
        if args.noise:
            X_test = add_noise(X_test, len(le_species.classes_))  # Add noise to the test data
        if args.log:
            X_train = add_log(X_train, len(le_species.classes_))  # Apply logarithmic transformation to the train data
            X_test = add_log(X_test, len(le_species.classes_))  # Apply logarithmic transformation to the test data
        if args.features != ['all']:
            features = ''.join(args.features).replace(' ', '').split(',')
            X_train = remove_features(args, X_train, features)  # Remove specified features from the training set
            X_test = remove_features(args, X_test, features)  # Remove specified features from the test set
        if args.standardization:
            X_train, scaler = add_standardization(X_train, scaler=None)  # Perform feature standardization on the train data
            X_test, scaler = add_standardization(X_test, scaler=scaler)  # Perform feature standardization on the test data
        model_parameters = get_model_parameters(args, X_train.shape[1], len(le_header.classes_))  # Get model parameters
        model = XGB(**model_parameters)  # Initialize the model
        start_fold = time.time()  # Start time of the fold evaluation
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)])  # Train the model on the train data and evaluate on the test data
        end_fold = time.time()  # End time of the fold evaluation
        time_fold = end_fold - start_fold  # Total fold evaluation time
        y_probas = model.predict_proba(X_test)  # Predict class probabilities for the test data
        metric = get_metric(args, len(le_header.classes_))  # Get the evaluation metric
        accuracy = metric(torch.from_numpy(y_probas), torch.from_numpy(y_test)) * 100  # Calculate accuracy using the evaluation metric
        if model.best_iteration + args.early_stopping_rounds <= args.n_estimators:
            print(f'\nEarly stopping occurred at iteration {model.best_iteration + 10} with best iteration = {model.best_iteration} and best {args.metric} = {accuracy:.4f}.')
        else:
            print(f'\nEarly stopping did not occur as best iteration = {model.best_iteration} and best {args.metric} = {accuracy:.4f}')
        print(f"Time: {time_fold:.2f}s.")  # Print the time taken for the fold evaluation
        return accuracy

    def run(self, args, X, y, le_species, le_header):
        """
        Run the XGB model based on the specified pipeline.

        Args:
            args (argparse.Namespace): Arguments passed to the framework.
            X (scipy.sparse._csr.csr_matrix): Input features.
            y (numpy.ndarray): Target labels.
            le_species (sklearn.preprocessing._label.LabelEncoder): Label encoder for species.
            le_header (sklearn.preprocessing._label.LabelEncoder): Label encoder for habitat classes.
        """
        if args.pipeline == 'evaluation':
            self.evaluate_model(args, X, y, le_species, le_header)  # Run evaluation pipeline
        elif args.pipeline == 'training':
            self.train_model(args, X, y, le_species, le_header)  # Run training pipeline
        elif args.pipeline == 'prediction':
            self.predict_model(args, X, y, le_species, le_header)  # Run prediction pipeline
