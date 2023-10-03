import pytorch_tabnet.tab_model
import torch
import numpy as np
import time

from utils import get_metric, get_model_parameters, get_fit_parameters

from data.load_data import get_split_assignments, get_eunis_red_list_crosswalks, get_endangered_red_list_habitats, get_endangered_eunis_habitats, get_tnc_model, get_scaler
from data.preprocess_data import add_fold_assignments, add_standardization, add_normalization, add_dropout, add_binarization, add_log, add_noise, add_rank, remove_features, add_augmentation, add_endangered_habitats, add_predictions_decoding
from data.save_data import set_scaler, set_tnc_model, set_predictions

class TNC(pytorch_tabnet.tab_model.TabNetClassifier):
    """
    TabNet Classifier for habitat distribution modeling.
    """
    def __init__(self, n_d=16, n_a=16, n_steps=3, gamma=1.3, cat_idxs=[], cat_dims=[], cat_emb_dim=1, n_independent=1, n_shared=3, epsilon=1e-15, seed=123, momentum=0.02, clip_value=None, lambda_sparse=1e-3, optimizer_fn='adam', optimizer_params={'lr':0.02}, scheduler_fn='step', scheduler_params={"step_size": 10, "gamma": 0.9}, model_name='TabNetClassifier', verbose=1, device_name='auto', input_dim= None, output_dim=None, mask_type='sparsemax', grouped_features=[], n_shared_decoder=1, n_indep_decoder=1):
        """
        Initialize the TabNet Classifier.

        Args:
            n_d (int, optional): The width of the decision prediction layer. Defaults to 16.
            n_a (int, optional): The width of the attention embedding for each mask. Defaults to 16.
            n_steps (int, optional): The number of steps in the architecture (usually between 3 and 10). Defaults to 3.
            gamma (float, optional): The coefficient for feature reusage in the masks. Defaults to 1.3.
            cat_idxs (list, optional): The list of categorical features indices. Defaults to [].
            cat_dims (list, optional): The list of categorical features number of modalities (number of unique values for a categorical feature). Defaults to [].
            cat_emb_dim (list, optional): The list of embeddings size for each categorical feature. Defaults to 1.
            n_independent (int, optional): The number of independent Gated Linear Units layers at each step. Defaults to 1.
            n_shared (int, optional): The number of shared Gated Linear Units at each step. Defaults to 3.
            epsilon (float, optional): The value should be left untouched. Defaults to 1e-15.
            seed (int, optional): The random seed for reproducibility. Defaults to 123.
            momentum (float, optional): The momentum for batch normalization.Defaults to 0.02.
            clip_value (float | tuple, optional): The gradient will be clipped at the given value. Defaults to None.
            lambda_sparse (float, optional): The extra sparsity loss coefficient. Defaults to 1e-3.
            optimizer_fn (callable, optional): The PyTorch optimizer function. Defaults to 'adam'.
            optimizer_params (dict, optional): The parameters compatible with the PyTorch optimizer function used to initialize the optimizer. Defaults to {'lr':0.02}
            scheduler_fn (callable, optional): The PyTorch scheduler to change learning rates during training. Defaults to 'step'.
            scheduler_params (dict, optional): The dictionary of parameters to apply to the PyTorch scheduler. Defaults to {"step_size": 10, "gamma": 0.9}.
            model_name (str, optional): The name of the model used for saving in disk. Defaults to 'TabNetClassifier'.
            verbose (int, optional): The verbosity for notebooks plots. Defaults to 1.
            device_name (str, optional): The name of the device used for training. Defaults to 'auto'.
            input_dim (int, optional): The number of input features. Defaults to None.
            output_dim (int, optional): The number of output classes. Defaults to None.
            mask_type (str, optional): The masking function to use for selecting features. Defaults to 'sparsemax'.
            grouped_features (list): The groups to allow the model to share its attention across features inside the same group. Defaults to [].
            n_shared_decoder (int, optional): The number of shared GLU blocks in the decoder. Defaults to 1.
            n_indep_decoder (int, optional): The number of independent GLU blocks in the decoder. Defaults to 1.
        """
        super(TNC, self).__init__()  # Initialize the parent class, `pytorch_tabnet.tab_model.TabNetClassifier`
        self.n_d = n_d  # The width of the decision prediction layer
        self.n_a = n_a  # The width of the attention embedding for each mask
        self.n_steps = n_steps  # The number of steps in the architecture (usually between 3 and 10)
        self.gamma = gamma  # The coefficient for feature reusage in the masks
        self.cat_idxs = cat_idxs  # The list of categorical features indices
        self.cat_dims = cat_dims  # The list of categorical features number of modalities (number of unique values for a categorical feature)
        self.cat_emb_dim = cat_emb_dim  # The list of embeddings size for each categorical features
        self.n_independent = n_independent  # The number of independent Gated Linear Units layers at each step
        self.n_shared = n_shared  # The number of shared Gated Linear Units at each step
        self.epsilon = epsilon  # The value should be left untouched
        self.seed = seed  # The random seed for reproducibility
        self.momentum = momentum  # The momentum for batch normalization
        self.clip_value = clip_value  # The gradient will be clipped at the given value
        self.lambda_sparse = lambda_sparse  # The extra sparsity loss coefficient
        self.optimizer_fn = optimizer_fn  # The PyTorch optimizer function
        self.optimizer_params = optimizer_params  # The parameters compatible with the PyTorch optimizer function used to initialize the optimizer
        self.scheduler_fn = scheduler_fn  # The PyTorch scheduler to change learning rates during training
        self.scheduler_params = scheduler_params  # The dictionnary of parameters to apply to the PyTorch scheduler
        self.model_name = model_name  # The name of the model used for saving in disk
        self.verbose = verbose  # The verbosity for notebooks plots
        self.device_name = device_name  # The name of the device used for training
        self.input_dim = input_dim  # The number of input features
        self.output_dim = output_dim  # The number of output classes
        self.mask_type = mask_type  # The masking function to use for selection features
        self.grouped_features = grouped_features  # The groups to allow the model to share its attention across features inside a same group
        self.n_shared_decoder = n_shared_decoder  # The number of shared GLU block in decoder
        self.n_indep_decoder = n_indep_decoder  # The number of independent GLU block in decoder

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
        avg_acc = 0.0  # Initialize variables to track the average accuracy
        n_samples = 0  # Initialize variables to track the number of samples
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
        print(f"\nMean {args.metric} and standard deviation after {args.n_folds} folds of {args.max_epochs} epochs: {avg_acc:.4f}% & {std_acc:.4f}.")
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
            X = add_log(X, len(le_species.classes_))  # Apply logarithm to the feature
        if args.features != ['all']:
            features = ''.join(args.features).replace(' ', '').split(',')
            X = remove_features(args, X, features)  # Remove specified features
        if args.standardization:
            X, scaler = add_standardization(X, scaler=None)  # Standardize the features
            set_scaler(args, scaler)  # Set the scaler for future use
        model_parameters = get_model_parameters(args, X.shape[1], len(le_header.classes_))  # Get model parameters
        model = TNC(**model_parameters)  # Create an instance of the TabNet Classifier
        fit_parameters = get_fit_parameters(args, X, X, y, y)  # Get fit parameters
        print()
        start_training = time.time()  # Start time of training
        model.fit(**fit_parameters)  # Train the model
        end_training = time.time()  # End time of training
        time_training = end_training - start_training  # Total training time
        print(f"\nTotal time of the training: {time_training:.2f}s.")
        y_probas = model.predict_proba(X)  # Get predicted probabilities
        metric = get_metric(args, len(le_header.classes_))  # Get the evaluation metric
        accuracy = metric(torch.from_numpy(y_probas), torch.from_numpy(y)) * 100  # Compute accuracy using the metric
        print(f"\nMean {args.metric} with {args.n_steps} steps: {accuracy:.4f}%.")
        set_tnc_model(args, model)  # Set the trained model for future use
        
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
        model = TNC(**model_parameters)  # Create an instance of the TabNet Classifier
        model = get_tnc_model(args, model)  # Load the trained model
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
        model = TNC(**model_parameters)  # Create an instance of the TabNet Classifier
        fit_parameters = get_fit_parameters(args, X_train, X_test, y_train, y_test)  # Get fit parameters
        start_fold = time.time()  # Start time of the fold
        model.fit(**fit_parameters)  # Train the model
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
        Run the TNC model based on the specified pipeline.

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
