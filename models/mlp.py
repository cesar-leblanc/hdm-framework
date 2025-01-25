import torch
import numpy as np
import time
import scipy.special

from utils import get_device, get_criterion, get_optimizer, get_scheduler, get_metric, NormedLinear, get_model_parameters

from data.load_data import get_split_assignments, get_eunis_red_list_crosswalks, get_endangered_red_list_habitats, get_endangered_eunis_habitats, get_scaler, get_mlp_model
from data.preprocess_data import add_fold_assignments, add_dataloader, add_standardization, add_normalization, add_dropout, add_binarization, add_log, add_noise, add_rank, remove_features, add_augmentation, add_endangered_habitats, add_predictions_decoding
from data.save_data import set_scaler, set_mlp_model, set_predictions

class MLP(torch.nn.Module):
    """
    Multi-Layer Perceptron (MLP) model for habitat distribution modeling.
    """
    def __init__(self, input_dim=None, num_hidden_layers=1, num_neurons=200, output_dim=None, activation_func='relu', weight_init=None, dropout_p=0, use_batch_norm=False):
        """
        Initialize the MLP model.

        Args:
            input_dim (int, optional): Dimensionality of the input features. Defaults to None.
            num_hidden_layers (int, optional): Number of hidden layers. Defaults to 1.
            num_neurons (int, optional): Number of neurons in the hidden layers. Defaults to 200.
            output_dim (int, optional): Dimensionality of the output. Defaults to None.
            activation_func (str, optional): Activation function to use. Defaults to 'relu'.
            weight_init (str, optional): Weight initialization method.  Defaults to None.
            dropout_p (float, optional): Dropout probability. Defaults to 0.
            use_batch_norm (bool, optional): Whether to use batch normalization or not. Defaults to False.
        """
        super(MLP, self).__init__()  # Initialize the parent class, `torch.nn.Module`
        self.fc1 = torch.nn.Linear(input_dim, num_neurons)  # Create a linear layer with `input_dim` input units and `num_neurons` output units
        self.use_batch_norm = use_batch_norm  # Save whether batch normalization should be used or not
        if self.use_batch_norm:
            self.bn1 = torch.nn.BatchNorm1d(num_neurons)  # Create a batch normalization layer with `num_neurons` input units
        if activation_func == 'relu':  # Determine which activation function to use based on the `activation_func` parameter
            self.activation1 = torch.nn.ReLU()
        elif activation_func == 'sigmoid':
            self.activation1 = torch.nn.Sigmoid()
        elif activation_func == 'tanh':
            self.activation1 = torch.nn.Tanh()
        self.dropout1 = torch.nn.Dropout(p=dropout_p)  # Create a dropout layer with a probability of dropping out `dropout_p`

        self.hidden_layers = torch.nn.ModuleList()  # Create an empty list of hidden layers
        for i in range(num_hidden_layers - 1):  # Iterate through the number of hidden layers
            self.hidden_layers.append(torch.nn.Linear(num_neurons, num_neurons))  # Add a linear layer with `num_neurons` input and output units to the list
            if self.use_batch_norm:
                self.hidden_layers.append(torch.nn.BatchNorm1d(num_neurons))  # Add a batch normalization layer with `num_neurons` input units to the list
            if activation_func == 'relu':  # Determine which activation function to use based on the `activation_func` parameter
                self.hidden_layers.append(torch.nn.ReLU())  # Add a ReLU activation function to the list
            elif activation_func == 'sigmoid':
                self.hidden_layers.append(torch.nn.Sigmoid())  # Add a sigmoid activation function to the list
            elif activation_func == 'tanh':
                self.hidden_layers.append(torch.nn.Tanh())  # Add a tanh activation function to the list
            self.hidden_layers.append(torch.nn.Dropout(p=dropout_p))  # Add a dropout layer with a probability of dropping out `dropout_p` to the list

        self.fc2 = torch.nn.Linear(num_neurons, output_dim)  # Create a linear layer with `num_neurons` input units and `output_dim` output units

        if weight_init == 'xavier':  # If Xavier initialization is selected
            torch.nn.init.xavier_normal_(self.fc1.weight)  # Initialize the weights of the first linear layer using Xavier initialization
            torch.nn.init.xavier_normal_(self.fc2.weight)  # Initialize the weights of the second linear layer using Xavier initialization
            for layer in self.hidden_layers:  # Iterate through the hidden layers
                if isinstance(layer, torch.nn.Linear):  # If the layer is a linear layer
                    torch.nn.init.xavier_normal_(layer.weight)  # Initialize the weights using Xavier initialization
        elif weight_init == 'kaiming':  # If Kaiming initialization is selected
            torch.nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')  # Initialize the weights of the first linear layer using Kaiming initialization with a ReLU nonlinearity
            torch.nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')  # Initialize the weights of the second linear layer using Kaiming initialization with a ReLU nonlinearity
            for layer in self.hidden_layers:  # Iterate through the hidden layers
                if isinstance(layer, torch.nn.Linear):  # If the layer is a linear layer
                    torch.nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')  # Initialize the weights using Kaiming initialization

    def forward(self, x):
        """
        Forward pass of the MLP model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.fc1(x)
        if self.use_batch_norm:
            x = self.bn1(x)
        x = self.activation1(x)
        x = self.dropout1(x)
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.fc2(x)
        return x

    def evaluate_model(self, args, X, y, le_species, le_header):
        """
        Evaluate the model using the specified arguments and input data.

        Args:
            args (argparse.Namespace): Arguments for model evaluation.
            X (scipy.sparse._csr.csr_matrix): Input features.
            y (numpy.ndarray): Target labels.
            le_species (sklearn.preprocessing._label.LabelEncoder): LabelEncoder for species.
            le_header (sklearn.preprocessing._label.LabelEncoder): LabelEncoder for header.
        """
        if args.rank:
            X = add_rank(X, len(le_species.classes_))  # Add rank-based features to the input data
        if args.binarization:
            X = add_binarization(X, len(le_species.classes_))  # Add binarized features to the input data
        if args.normalization:
            X = add_normalization(X, len(le_species.classes_))  # Add normalized features to the input data
        std_acc = []
        avg_acc = 0.0  # Initialize variable to track the average accuracy
        n_samples = 0  # Initialize variable to track the number of samples
        split_assignments = get_split_assignments(args)
        best_accuracy = -np.inf
        best_fold = -1
        start_evaluation = time.time()
        for fold in range(args.n_folds):  # Perform evaluation for each fold
            best_fold_accuracy = self.evaluate_fold(args, X, y, le_species, le_header, fold, split_assignments)
            if best_fold_accuracy > best_accuracy:
                best_accuracy = best_fold_accuracy
                best_fold = fold
                print("\nFold {} did improve the model.".format(best_fold))
            else:
                print("\nFold {} did not improve the model.".format(fold))
            std_acc.append(best_fold_accuracy)
            avg_acc += best_fold_accuracy  # Accumulate the accuracy
            n_samples += 1  # Accumulate the number of samples
        end_evaluation = time.time()
        time_evaluation = end_evaluation - start_evaluation
        avg_acc /= n_samples  # Compute the average accuracy
        std_acc = torch.std(torch.FloatTensor(std_acc))
        print(f"\nMean {args.metric} and standard deviation after {args.n_folds} folds of {args.num_epochs} epochs: {avg_acc:.4f}% & {std_acc:.4f}.")  # Print evaluation results
        print(f"Total time of the evaluation: {time_evaluation:.2f}s.")  # Print evaluation time

    def train_model(self, args, X, y, le_species, le_header):
        """
        Train the model using the specified arguments and input data.

        Args:
            args (argparse.Namespace): Arguments for model training.
            X (scipy.sparse._csr.csr_matrix): Input features.
            y (numpy.ndarray): Target labels.
            le_species (sklearn.preprocessing._label.LabelEncoder): LabelEncoder for species.
            le_header (sklearn.preprocessing._label.LabelEncoder): LabelEncoder for header.
        """
        if args.rank:
            X = add_rank(X, len(le_species.classes_))  # Add rank-based features to the input data
        if args.binarization:
            X = add_binarization(X, len(le_species.classes_))  # Add binarized features to the input data
        if args.normalization:
            X = add_normalization(X, len(le_species.classes_))  # Add normalized features to the input data
        if args.endangered:
            eunis_red_list_crosswalks = get_eunis_red_list_crosswalks(args)
            endangered_red_list_habitats = get_endangered_red_list_habitats(args)  # Get endangered habitats
            endangered_eunis_habitats = get_endangered_eunis_habitats(eunis_red_list_crosswalks, endangered_red_list_habitats, le_header)
            X, y = add_endangered_habitats(X, y, endangered_eunis_habitats)  # Modify input data and labels accordingly
        if args.augmentation:
            if not args.endangered:
                labels = None
            else:
                labels = endangered_eunis_habitats  
            X, y = add_augmentation(X, y, labels)  # Add data augmentation to the input data and labels
        if args.dropout > 0:
            X = add_dropout(X, args.dropout, len(le_species.classes_))  # Add dropout to the input data
        if args.noise:
            X = add_noise(X, len(le_species.classes_))  # Add noise to the input data
        if args.log:
            X = add_log(X, len(le_species.classes_))  # Apply logarithmic transformation to the input data
        if args.features != ['all']:
            features = ''.join(args.features).replace(' ', '').split(',')
            X = remove_features(args, X, features)  # Remove specific features from the input data
        if args.standardization:
            X, scaler = add_standardization(X, scaler=None)  # Add standardization to the input data and save the scaler
            set_scaler(args, scaler)
        train_dataloader = add_dataloader(args, X, y, shuffle=True)
        model_parameters = get_model_parameters(args, X.shape[1], len(le_header.classes_))
        model = MLP(**model_parameters)
        if args.criterion == 'imbal-noised-top-k':
            LastLinearLayer = NormedLinear
            model.fc2 = LastLinearLayer(model.fc2.in_features, len(le_header.classes_))
        device = get_device(args)  # Set the device to run the model on to be GPU
        if args.data_parallelism:
            model = torch.nn.DataParallel(model)  # Run the model parallelly
        model.to(device)  # Use GPU for model
        criterion = get_criterion(args, y)  # Instantiate loss class
        criterion.to(device)
        optimizer = get_optimizer(args, model)  # Instantiate optimizer class
        scheduler = get_scheduler(args, optimizer)  # Instantiate step learning scheduler class
        best_accuracy = -np.inf
        best_epoch = -1
        best_model_state = model.state_dict()  # Set up a variable to store the best model's state
        no_improvement_epochs = 0  # Set up a counter for the number of epochs without improvement
        start_training = time.time()
        for epoch in range(args.num_epochs):
            epoch_accuracy, model = self.train_epoch(args, model, train_dataloader, epoch, device, optimizer, criterion, le_header)  # Train the model for one epoch
            scheduler.step(epoch_accuracy)  # Decay Learning Rate, pass validation accuracy for tracking at every epoch
            if epoch_accuracy > best_accuracy:  
                best_accuracy = epoch_accuracy  
                best_epoch = epoch
                best_model_state = model.state_dict()  # Update the best model state
                set_mlp_model(args, best_model_state)  # Save the model
                print("\nSuccessfully saved model at Models/MLP.pth")
                no_improvement_epochs = 0
            else:  # Check if the accuracy is not the best seen so far
                no_improvement_epochs += 1  # Increment the number of epochs without improvement
                print("\nEpoch {} did not improve the model.".format(epoch))
            if no_improvement_epochs == args.num_iter_no_change:  # Check if the model's accuracy on the test set has not improved for the num_iter_no_change number of epochs               
                print(f'\nEarly stopping occurred at epoch {epoch} with best epoch = {best_epoch} and best {args.metric} = {best_accuracy:.4f}%.')
                break  # Stop the training process
        end_training = time.time()
        time_training = end_training - start_training
        if no_improvement_epochs != args.num_iter_no_change:
            print(f'\nEarly stopping did not occur as best epoch = {best_epoch} and best {args.metric} = {best_accuracy:.4f}%.')
        print(f"Total time of the training: {time_training:.2f}s.")

    def predict_model(self, args, X, y, le_species, le_header):
        """
        Predict the habitat labels for given input data using a trained model.

        Args:
            args (argparse.Namespace): Command-line arguments.
            X (scipy.sparse._csr.csr_matrix): Input data features.
            y (numpy.ndarray): Input data labels.
            le_species (sklearn.preprocessing._label.LabelEncoder): LabelEncoder for species classes.
            le_header (sklearn.preprocessing._label.LabelEncoder): LabelEncoder for habitat classes.

        Returns:
            None
        """
        if args.rank:
            X = add_rank(X, len(le_species.classes_))  # Add rank features to input data
        if args.binarization:
            X = add_binarization(X, len(le_species.classes_))  # Add binarized features to input data
        if args.normalization:
            X = add_normalization(X, len(le_species.classes_))  # Add normalized features to input data
        if args.log:
            X = add_log(X, len(le_species.classes_))  # Add logarithmic features to input data
        if args.features != ['all']:
            features = ''.join(args.features).replace(' ', '').split(',')
            X = remove_features(args, X, features)  # Remove specified features from input data
        if args.standardization:
            scaler = get_scaler(args)
            X, _ = add_standardization(X, scaler=scaler)  # Add standardized features to input data using the scaler
        model_parameters = get_model_parameters(args, X.shape[1], len(le_header.classes_))  # Get model parameters
        model = MLP(**model_parameters)  # Create an instance of the MLP model
        if args.criterion == 'imbal-noised-top-k':
            LastLinearLayer = NormedLinear
            model.fc2 = LastLinearLayer(model.fc2.in_features, len(le_header.classes_))  # Replace the last linear layer with a custom NormedLinear layer
        device = get_device(args)  # Set the device to run the model on to be GPU
        if args.data_parallelism:
            model = torch.nn.DataParallel(model)  # Run the model in parallel if specified
        model = get_mlp_model(args, model)  # Load the trained model parameters
        model.to(device)  # Move the model to the specified device (GPU or CPU)
        X = torch.from_numpy(X.toarray()).float()  # Convert input data to a PyTorch tensor
        model.eval()  # Set the model to evaluation mode
        start_prediction = time.time()  # Start timing the prediction process
        with torch.no_grad():  # Turn off gradient computation to save memory
            predictions = model(X)  # Pass the input data through the model to obtain predictions
        end_prediction = time.time()  # End timing the prediction process
        time_prediction = end_prediction - start_prediction  # Compute the time taken for prediction
        predictions = scipy.special.softmax(predictions.cpu(), axis=1)  # Apply softmax function to convert logits to probabilities
        predictions = [np.argmax(sub_arr) for sub_arr in predictions]  # Get the indices of the maximum probability for each prediction
        predictions = add_predictions_decoding(predictions, le_header)  # Decode the predicted labels using the LabelEncoder for habitats
        set_predictions(args, predictions)  # Store the predictions in the specified location
        print("\nSuccessfully saved predictions at Data/predictions.txt")
        print(f"Total time of the prediction: {time_prediction:.2f}s.")  # Print the total time taken for prediction

    def evaluate_fold(self, args, X, y, le_species, le_header, fold, split_assignments):
        """
        Evaluate a single fold of the dataset for Habitat Distribution Modeling.

        Args:
            args (argparse.Namespace): Command-line arguments.
            X (scipy.sparse._csr.csr_matrix): Input data features.
            y (numpy.ndarray): Input data labels.
            le_species (sklearn.preprocessing._label.LabelEncoder): LabelEncoder for species classes.
            le_header (sklearn.preprocessing._label.LabelEncoder): LabelEncoder for habitat classes.
            fold (int): Fold number for evaluation.
            split_assignments (numpy.ndarray): Array indicating the fold assignments for each sample.

        Returns:
            float: Best accuracy achieved in the fold.
        """
        print('\n' + '*'*11)
        print('* Fold: {} *'.format(fold))  # Print fold number
        print('*'*11 + '\n' + '-'*30)
        X_train, X_test, y_train, y_test = add_fold_assignments(X, y, split_assignments, fold)  # Split data into training and test sets based on fold assignments
        if args.endangered:
            eunis_red_list_crosswalks = get_eunis_red_list_crosswalks(args)
            endangered_red_list_habitats = get_endangered_red_list_habitats(args)
            endangered_eunis_habitats = get_endangered_eunis_habitats(eunis_red_list_crosswalks, endangered_red_list_habitats, le_header)  # Get endangered habitats
            X_test, y_test = add_endangered_habitats(X_test, y_test, endangered_eunis_habitats)   # Add endangered habitats to the test set
        if args.augmentation:
            if not args.endangered:
                labels = None
            else:
                labels = endangered_eunis_habitats
            X_train, y_train = add_augmentation(X_train, y_train, labels)  # Augment the training set with additional samples
        if args.dropout > 0:
            X_test = add_dropout(X_test, args.dropout, len(le_species.classes_))  # Apply dropout to the test set
        if args.noise:
            X_test = add_noise(X_test, len(le_species.classes_))  # Add noise to the test set
        if args.log:
            X_train = add_log(X_train, len(le_species.classes_))  # Apply logarithmic transformation to the training set
            X_test = add_log(X_test, len(le_species.classes_))  # Apply logarithmic transformation to the test set
        if args.features != ['all']:
            features = ''.join(args.features).replace(' ', '').split(',')
            X_train = remove_features(args, X_train, features)  # Remove specific features from the training set
            X_test = remove_features(args, X_test, features)  # Remove specific features from the test set
        if args.standardization:
            X_train, scaler = add_standardization(X_train, scaler=None)  # Apply standardization to the training set
            X_test, scaler = add_standardization(X_test, scaler=scaler)  # Apply standardization to the test set using the same scaler
        train_dataloader = add_dataloader(args, X_train, y_train, shuffle=True)  # Create a dataloader for the training set
        test_dataloader = add_dataloader(args, X_test, y_test, shuffle=False)  # Create a dataloader for the test set
        model_parameters = get_model_parameters(args, X_train.shape[1], len(le_header.classes_))  # Get model parameters
        model = MLP(**model_parameters)  # Create an instance of the MLP model
        if args.criterion == 'imbal-noised-top-k':
            LastLinearLayer = NormedLinear
            model.fc2 = LastLinearLayer(model.fc2.in_features, len(le_header.classes_))  # Replace the last linear layer with a customized linear layer
        device = get_device(args)  # Set the device to run the model on (GPU or CPU)
        if args.data_parallelism:
            model = torch.nn.DataParallel(model)  # Use data parallelism to run the model on multiple GPUs
        model.to(device)  # Move the model to the specified device (GPU or CPU)
        criterion = get_criterion(args, y_train)  # Instantiate the loss function
        criterion.to(device)  # Move the loss function to the specified device (GPU or CPU)
        optimizer = get_optimizer(args, model)  # Instantiate the optimizer
        scheduler = get_scheduler(args, optimizer)  # Instantiate the learning rate scheduler
        best_fold_accuracy = -np.inf  # Initialize the best accuracy variable
        best_epoch = -1  # Initialize the best epoch variable
        no_improvement_epochs = 0  # Initialize the counter for epochs without improvement
        start_fold = time.time()  # Start timing the evaluation of the fold
        for epoch in range(args.num_epochs):
            accuracy, model = self.evaluate_epoch(args, model, train_dataloader, test_dataloader, epoch, device, optimizer, criterion, le_header)  # Evaluate the epoch and obtain the accuracy
            scheduler.step(accuracy)  # Decay Learning Rate, pass validation accuracy for tracking at every epoch
            if accuracy > best_fold_accuracy:  # Check if the model's state has the highest accuracy on the test set so far
                best_fold_accuracy = accuracy  # Update the best accuracy
                best_epoch = epoch  # Update the best epoch
                no_improvement_epochs = 0  # Reset the number of epochs without improvement
            else:  # Check if the test accuracy is not the best seen so far
                no_improvement_epochs += 1  # Increment the number of epochs without improvement
            if no_improvement_epochs == args.num_iter_no_change:  # Check if the model's accuracy on the test set has not improved for the num_iter_no_change number of epochs               
                print(f'\nEarly stopping occurred at epoch {epoch} with best epoch = {best_epoch} and best {args.metric} = {best_fold_accuracy}')
                break  # Stop the training process
        end_fold = time.time()  # End timing the evaluation of the fold
        time_fold = end_fold - start_fold  # Calculate the time taken for the evaluation of the fold
        if no_improvement_epochs != args.num_iter_no_change:
            print(f'\nEarly stopping did not occur as best epoch = {best_epoch} and best {args.metric} = {best_fold_accuracy}')
        print(f"Total time for the evaluation of the fold: {time_fold:.2f}s.")
        return best_fold_accuracy

    def train_epoch(self, args, model, train_dataloader, epoch, device, optimizer, criterion, le_header):
        """
        Train the model for a single epoch.

        Args:
            args (argparse.Namespace): Arguments passed to the training process.
            model (models.mlp.MLP | torch.nn.parallel.data_parallel.DataParallel): The deep learning model.
            train_dataloader (torch.utils.data.dataloader.DataLoader): Dataloader for the training dataset.
            epoch (int): Current epoch number.
            device (torch.device): Device to run the model on (GPU or CPU).
            optimizer (torch.optim.sgd.SGD | torch.optim.adam.Adam | torch.optim.adamw.AdamW): Optimizer for updating model parameters.
            criterion (torch.nn.modules.loss.CrossEntropyLoss | pytopk.noised_losses.BalNoisedTopK | pytopk.noised_losses.ImbalNoisedTopK): Loss function.
            le_header (sklearn.preprocessing._label.LabelEncoder): Label encoder for habitat classes.

        Returns:
            torch.Tensor: Accuracy achieved in the epoch.
            MLP: Updated model after training.
        """
        start_epoch = time.time()  # Start timing the epoch
        for batch, labels in train_dataloader:  # Iterate through train dataset
            batch = batch.requires_grad_().to(device)  # Load batches with gradient accumulation capabilities
            labels = labels.to(device)  # Use GPU for tensors
            optimizer.zero_grad()  # Clear gradients w.r.t. parameters
            outputs = model(batch)  # Forward pass to get output/logits
            loss = criterion(outputs, labels)  # Calculate Loss: softmax --> cross entropy loss
            loss.backward()  # Getting gradients w.r.t. parameters
            optimizer.step()  # Updating parameters
        end_epoch = time.time()  # End timing the epoch
        time_epoch = end_epoch - start_epoch  # Calculate the time taken for the epoch
        metric = get_metric(args, len(le_header.classes_))  # Instantiate the accuracy metric
        metric.to(device)  # Move the metric to the device
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():  # Disable gradient computation
            for batch, labels in train_dataloader:  # Iterate through the batches in the training dataset
                batch = batch.to(device)  # Move the batch to the device
                labels = labels.to(device)  # Move the labels to the device
                outputs = model(batch)  # Forward pass to get output/logits
                accuracy = metric(outputs, labels)  # Calculate the accuracy
        accuracy_epoch = metric.compute() * 100  # Compute the overall accuracy for the epoch
        print('\n' + '-'*30)
        print('Epoch {} completed.'.format(epoch))  # Print the epoch number
        print('LR: {}.'.format(optimizer.param_groups[0]['lr']))  # Print Learning Rate
        print(f"Time: {time_epoch:.2f}s.")  # Print Time
        print(f'Accuracy: {accuracy_epoch   :.4f}%.')  # Print Accuracy
        print('-'*30)
        model.train()  # Set the model back to training mode
        return accuracy_epoch, model

    def evaluate_epoch(self, args, model, train_dataloader, test_dataloader, epoch, device, optimizer, criterion, le_header):
        """
        Evaluate a single epoch of the training process.

        Args:
            args (argparse.Namespace): Training arguments.
            model (models.mlp.MLP | torch.nn.parallel.data_parallel.DataParallel): The deep learning model.
            train_dataloader (torch.utils.data.dataloader.DataLoader): Dataloader for the training dataset.
            test_dataloader (torch.utils.data.dataloader.DataLoader): Dataloader for the test dataset.
            epoch (int): Current epoch number.
            device (torch.device): Device to perform computations on (e.g., CPU or GPU).
            optimizer (torch.optim.sgd.SGD | torch.optim.adam.Adam | torch.optim.adamw.AdamW): Optimizer for updating model parameters.
            criterion (torch.nn.modules.loss.CrossEntropyLoss | pytopk.noised_losses.BalNoisedTopK | pytopk.noised_losses.ImbalNoisedTopK): Loss function criterion.
            le_header (sklearn.preprocessing._label.LabelEncoder): Label encoder for the header.

        Returns:
            tuple: A tuple containing the accuracy of the epoch and the updated model.
        """
        start_epoch = time.time()  # Start timing the epoch
        for batch, labels in train_dataloader:  # Iterate through train dataset
            batch = batch.requires_grad_().to(device)  # Load batches with gradient accumulation capabilities
            labels = labels.to(device)  # Use GPU for tensors
            optimizer.zero_grad()  # Clear gradients w.r.t. parameters
            outputs = model(batch)  # Forward pass to get output/logits
            loss = criterion(outputs, labels)  # Calculate Loss: softmax --> cross entropy loss
            loss.backward()  # Getting gradients w.r.t. parameters
            optimizer.step()  # Updating parameters
        end_epoch = time.time()  # End timing the epoch
        time_epoch = end_epoch - start_epoch  # Calculate the time taken for the epoch
        metric = get_metric(args, len(le_header.classes_))  # Instantiate the accuracy metric
        metric.to(device)  # Move the metric to the device
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():  # Disable gradient computation
            for batch, labels in test_dataloader:  # Iterate through the batches in the test dataset
                batch = batch.to(device)  # Move the batch to the device
                labels = labels.to(device)  # Move the labels to the device
                outputs = model(batch)  # Forward pass to get output/logits
                accuracy = metric(outputs, labels)  # Calculate the accuracy
        accuracy_epoch = metric.compute() * 100  # Compute the overall accuracy for the epoch
        print('Epoch {} completed.'.format(epoch))  # Print Epoch
        print('LR: {}.'.format(optimizer.param_groups[0]['lr']))  # Print Learning Rate
        print(f"Time: {time_epoch:.2f}s.")  # Print Time
        print(f'Accuracy: {accuracy_epoch:.4f}%.')  # Print Accuracy
        print('-'*30)
        model.train()  # Set the model back to training mode
        return accuracy_epoch, model

    def run(self, args, X, y, le_species, le_header):
        """
        Run the MLP model based on the specified pipeline.

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
