import sklearn.ensemble
import torch
import numpy as np
import time

from utils import get_metric, get_model_parameters

from data.load_data import get_split_assignments, get_eunis_red_list_crosswalks, get_endangered_red_list_habitats, get_endangered_eunis_habitats, get_rfc_model, get_scaler
from data.preprocess_data import add_fold_assignments, add_standardization, add_normalization, add_dropout, add_binarization, add_log, add_noise, add_rank, remove_features, add_augmentation, add_endangered_habitats, add_predictions_decoding
from data.save_data import set_scaler, set_rfc_model, set_predictions

class RFC(sklearn.ensemble.RandomForestClassifier):
    """
    Random Forest Classifier (RFC) model for habitat distribution modeling.
    """
    def __init__(self, n_estimators=200, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='sqrt', max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=True, oob_score=False, n_jobs=-1, random_state=123, verbose='n_estimators', warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None):
        """
        Initialize the Random Forest Classifier.

        Args:
            n_estimators (int, optional): The number of trees in the forest. Defaults to 200.
            criterion (str, optional): The function to measure the quality of a split. Defaults to 'gini'.
            max_depth (int, optional): The maximum depth of the tree. Defaults to None.
            min_samples_split (int, optional): The minimum number of samples required to split an internal node. Defaults to 2.
            min_samples_leaf (int, optional): The minimum number of samples required to be at a leaf node. Defaults to 1.
            min_weight_fraction_leaf (float, optional): The minimum weighted fraction of the sum total of weights (of all the input samples) required to be at a leaf node. Defaults to 0.0.
            max_features (int | float, optional): The number of features to consider when looking for the best split. Defaults to 'sqrt'.
            max_leaf_nodes (int, optional): The value for which trees grow in a best-first fashion. Defaults to None.
            min_impurity_decrease (float, optional): The value for which a node will be split if this split induces a decrease of the impurity greater than or equal to this value. Defaults to 0.0.
            bootstrap (bool, optional): The choice to use bootstrap samples when building trees. Defaults to True.
            oob_score (bool, optional): The choice to use out-of-bag samples to estimate the generalization score. Defaults to False.
            n_jobs (int | None, optional): The number of jobs to run in parallel. Defaults to -1.
            random_state (int | RandomState, optional): The value that controls both the randomness of the bootstrapping of the samples used when building trees and the sampling of the features to consider when looking for the best split at each node. Defaults to 123.
            verbose (int, optional): The value that controls the verbosity when fitting and predicting. Defaults to 'n_estimators'.
            warm_start (bool, optional): The choice to reuse the solution of the previous call to fit and add more estimators to the ensemble or to just fit a whole new forest. Defaults to False.
            class_weight (str | dict | list | None, optional): The weights associated with classes. Defaults to None.
            ccp_alpha (float, optional): The complexity parameter used for Minimal Cost-Complexity Pruning. Defaults to 0.0.
            max_samples (int | float, optional): The number of samples to draw from X to train each base estimator if bootstrap is used. Defaults to None.
        """
        super(RFC, self).__init__()  # Initialize the parent class, `sklearn.ensemble.RandomForestClassifier`
        self.n_estimators = n_estimators  # The number of trees in the forest
        self.criterion = criterion  # The function to measure the quality of a split
        self.max_depth = max_depth  # The maximum depth of the tree
        self.min_samples_split = min_samples_split  # The minimum number of samples required to split an internal node
        self.min_samples_leaf = min_samples_leaf  # The minimum number of samples required to be at a leaf node
        self.min_weight_fraction_leaf = min_weight_fraction_leaf  # The minimum weighted fraction of the sum total of weights (of all the input samples) required to be at a leaf node
        self.max_features = max_features  # The number of features to consider when looking for the best split
        self.max_leaf_nodes = max_leaf_nodes  # The value for which trees grow in best-first fashion
        self.min_impurity_decrease = min_impurity_decrease  # The value for which a node will be split if this split induces a decrease of the impurity greater than or equal to this value
        self.bootstrap = bootstrap  # The choice to use bootstrap samples when building trees
        self.oob_score = oob_score  # The choice to use out-of-bag samples to estimate the generalization score
        self.n_jobs = n_jobs  # The number of jobs to run in parallel
        self.random_state = random_state  # The value that controls both the randomness of the bootstrapping of the samples used when building trees and the sampling of the features to consider when looking for the best split at each node
        self.verbose = verbose  # The value that controls the verbosity when fitting and predicting
        self.warm_start = warm_start  # The choice to reuse the solution of the previous call to fit and add more estimators to the ensemble or to just fit a whole new forest
        self.class_weight = class_weight  # The weights associated with classes
        self.ccp_alpha = ccp_alpha  # The complexity parameter used for Minimal Cost-Complexity Pruning
        self.max_samples = max_samples  # The number of samples to draw from X to train each base estimator if bootstrap is used
        
    def evaluate_model(self, args, X, y, le_species, le_header):
        """
        Evaluate the model's performance using the specified evaluation settings.

        Args:
            args (argparse.Namespace): Arguments passed to the framework.
            X (scipy.sparse._csr.csr_matrix): Input features.
            y (numpy.ndarray): Target labels.
            le_species (sklearn.preprocessing._label.LabelEncoder): Label encoder for species.
            le_header (sklearn.preprocessing._label.LabelEncoder): Label encoder for habitat classes.
        """
        if args.rank:
            X = add_rank(X, len(le_species.classes_))  # Add rank as a feature
        if args.binarization:
            X = add_binarization(X, len(le_species.classes_))  # Add binarized features
        if args.normalization:
            X = add_normalization(X, len(le_species.classes_))  # Add normalized features
        std_acc = []
        avg_acc = 0.0  # Initialize variable to track the average accuracy
        n_samples = 0  # Initialize variable to track the number of samples
        split_assignments = get_split_assignments(args)  # Get split assignments for cross-validation
        best_accuracy = -np.inf  # Initialize variable to track the best accuracy
        best_fold = -1  # Initialize variable to track the best fold
        start_evaluation = time.time()  # Start time of evaluation
        for fold in range(args.n_folds):
            best_fold_accuracy = self.evaluate_fold(args, X, y, le_species, le_header, fold, split_assignments)  # Perform evaluation for a fold
            if best_fold_accuracy > best_accuracy:  # Check if current fold improves the model
                best_accuracy = best_fold_accuracy  # Update the best accuracy
                best_fold = fold  # Update the best fold
                print("\nFold {} did improve the model.".format(best_fold))
            else:
                print("\nFold {} did not improve the model.".format(fold))
            std_acc.append(best_fold_accuracy)  # Add best fold accuracy to the list
            avg_acc += best_fold_accuracy  # Accumulate the accuracy
            n_samples += 1  # Accumulate the number of samples
        end_evaluation = time.time()  # End time of evaluation
        time_evaluation = end_evaluation - start_evaluation  # Total evaluation time
        avg_acc /= n_samples  # Compute the average accuracy
        std_acc = torch.std(torch.FloatTensor(std_acc))  # Compute the standard deviation of accuracies
        print(f"\nMean {args.metric} and standard deviation after {args.n_folds} folds of {args.n_estimators} estimators: {avg_acc:.4f}% & {std_acc:.4f}.")
        print(f"Total time of the evaluation: {time_evaluation:.2f}s.")

    def train_model(self, args, X, y, le_species, le_header):
        """
        Train the model using the specified training settings.

        Args:
            args (argparse.Namespace): Arguments passed to the framework.
            X (scipy.sparse._csr.csr_matrix): Input features.
            y (numpy.ndarray): Target labels.
            le_species (sklearn.preprocessing._label.LabelEncoder): Label encoder for species.
            le_header (sklearn.preprocessing._label.LabelEncoder): Label encoder for habitat classes.
        """
        if args.rank:
            X = add_rank(X, len(le_species.classes_))  # Add rank as a feature
        if args.binarization:
            X = add_binarization(X, len(le_species.classes_))  # Add binarized features
        if args.normalization:
            X = add_normalization(X, len(le_species.classes_))  # Add normalized features
        if args.endangered:
            eunis_red_list_crosswalks = get_eunis_red_list_crosswalks(args)  # Get EUNIS-Red list crosswalks
            endangered_red_list_habitats = get_endangered_red_list_habitats(args)  # Get endangered Red list habitats
            endangered_eunis_habitats = get_endangered_eunis_habitats(eunis_red_list_crosswalks, endangered_red_list_habitats, le_header)  # Get endangered EUNIS habitats
            X, y = add_endangered_habitats(X, y, endangered_eunis_habitats)  # Add endangered habitats to the data
        if args.augmentation:
            if not args.endangered:
                labels = None
            else:
                labels = endangered_eunis_habitats
            X, y = add_augmentation(X, y, labels)  # Perform data augmentation
        if args.dropout > 0:
            X = add_dropout(X, args.dropout, len(le_species.classes_))  # Add dropout regularization
        if args.noise:
            X = add_noise(X, len(le_species.classes_))  # Add noise to the features
        if args.log:
            X = add_log(X, len(le_species.classes_))  # Apply logarithm to the features
        if args.features != ['all']:
            features = ''.join(args.features).replace(' ', '').split(',')
            X = remove_features(args, X, features)  # Remove specified features
        if args.standardization:
            X, scaler = add_standardization(X, scaler=None)  # Standardize the features
            set_scaler(args, scaler)  # Set the scaler for future use
        model_parameters = get_model_parameters(args, X.shape[1], len(le_header.classes_))  # Get model parameters
        model = RFC(**model_parameters)  # Create an instance of the Random Forest Classifier
        print()
        start_training = time.time()  # Start time of training
        model.fit(X, y)  # Train the model
        end_training = time.time()  # End time of training
        time_training = end_training - start_training  # Total training time
        print(f"\nTotal time of the training: {time_training:.2f}s.\n")
        y_probas = model.predict_proba(X)  # Get predicted probabilities
        metric = get_metric(args, len(le_header.classes_))  # Get the evaluation metric
        accuracy = metric(torch.from_numpy(y_probas), torch.from_numpy(y)) * 100  # Compute accuracy using the metric
        print(f"\nMean {args.metric} with {args.n_estimators} estimators: {accuracy:.4f}%.")
        set_rfc_model(args, model)  # Set the trained model for future use
        print("\nSuccessfully saved model at Models/RFC.joblib")
        
    def predict_model(self, args, X, y, le_species, le_header):
        """
        Use the trained model to make predictions on the input data.

        Args:
            args (argparse.Namespace): Arguments passed to the framework.
            X (scipy.sparse._csr.csr_matrix): Input features.
            y (numpy.ndarray): Target labels.
            le_species (sklearn.preprocessing._label.LabelEncoder): Label encoder for species.
            le_header (sklearn.preprocessing._label.LabelEncoder): Label encoder for habitat classes.
        """
        if args.rank:
            X = add_rank(X, len(le_species.classes_))  # Add rank as a feature
        if args.binarization:
            X = add_binarization(X, len(le_species.classes_))  # Add binarized features
        if args.normalization:
            X = add_normalization(X, len(le_species.classes_))  # Add normalized features
        if args.log:
            X = add_log(X, len(le_species.classes_))  # Apply logarithm to the features
        if args.features != ['all']:
            features = ''.join(args.features).replace(' ', '').split(',')
            X = remove_features(args, X, features)  # Remove specified features
        if args.standardization:
            scaler = get_scaler(args)  # Get the scaler used during training
            X, _ = add_standardization(X, scaler=scaler)  # Standardize the features using the scaler
        model_parameters = get_model_parameters(args, X.shape[1], len(le_header.classes_))  # Get model parameters
        model = RFC(**model_parameters)  # Create an instance of the Random Forest Classifier
        model = get_rfc_model(args, model)  # Load the trained model
        print()
        start_prediction = time.time()  # Start time of prediction
        predictions = model.predict(X)  # Make predictions
        end_prediction = time.time()  # End time of prediction
        time_prediction = end_prediction - start_prediction  # Total prediction time
        predictions = add_predictions_decoding(predictions, le_header)  # Decode the predictions using label encoder
        set_predictions(args, predictions)  # Set the predictions for future use
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
            fold (int): Fold index.
            split_assignments (numpy.ndarray): Assignment of samples to folds.

        Returns:
            accuracy: Accuracy achieved on the fold.
        """
        print('\n' + '*'*11)
        print('* Fold: {} *'.format(fold))  # Print fold
        print('*'*11 + '\n')
        X_train, X_test, y_train, y_test = add_fold_assignments(X, y, split_assignments, fold)  # Split data into train and test sets
        if args.endangered:
            eunis_red_list_crosswalks = get_eunis_red_list_crosswalks(args)
            endangered_red_list_habitats = get_endangered_red_list_habitats(args)
            endangered_eunis_habitats = get_endangered_eunis_habitats(eunis_red_list_crosswalks, endangered_red_list_habitats, le_header)
            X_test, y_test = add_endangered_habitats(X_test, y_test, endangered_eunis_habitats)  # Add endangered habitats to the test set
        if args.augmentation:
            if not args.endangered:
                labels = None
            else:
                labels = endangered_eunis_habitats  
            X_train, y_train = add_augmentation(X_train, y_train, labels)  # Add data augmentation to the training set
        if args.dropout > 0:
            X_test = add_dropout(X_test, args.dropout, len(le_species.classes_))  # Apply dropout to the test set
        if args.noise:
            X_test = add_noise(X_test, len(le_species.classes_))  # Add noise to the test set
        if args.log:
            X_train = add_log(X_train, len(le_species.classes_))  # Apply logarithm to the training set
            X_test = add_log(X_test, len(le_species.classes_))  # Apply logarithm to the test set
        if args.features != ['all']:
            features = ''.join(args.features).replace(' ', '').split(',')
            X_train = remove_features(args, X_train, features)  # Remove specified features from the training set
            X_test = remove_features(args, X_test, features)  # Remove specified features from the test set
        if args.standardization:
            X_train, scaler = add_standardization(X_train, scaler=None)  # Standardize the training set
            X_test, scaler = add_standardization(X_test, scaler=scaler)  # Standardize the test set using the same scaler
        model_parameters = get_model_parameters(args, X_train.shape[1], len(le_header.classes_))  # Get model parameters
        model = RFC(**model_parameters)  # Create an instance of the Random Forest Classifier
        start_fold = time.time()  # Start time of the fold
        model.fit(X_train, y_train)  # Train the model
        end_fold = time.time()  # End time of the fold
        time_fold = end_fold - start_fold  # Total fold time
        y_probas = model.predict_proba(X_test)  # Get predicted probabilities for the test set
        metric = get_metric(args, len(le_header.classes_))  # Get the evaluation metric
        accuracy = metric(torch.from_numpy(y_probas), torch.from_numpy(y_test)) * 100  # Calculate accuracy
        print(f"\nAccuracy: {accuracy:.4f}%.")  # Print Accuracy
        print(f"Time: {time_fold:.2f}s.")  # Print Time
        return accuracy

    def run(self, args, X, y, le_species, le_header):
        """
        Run the RFC model based on the specified pipeline.

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
