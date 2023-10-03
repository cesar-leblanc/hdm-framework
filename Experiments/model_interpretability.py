import torch
import multiprocessing
import torchmetrics
import numpy as np
import scipy.sparse
import pickle
import sklearn.preprocessing
import captum
import tqdm
import os
import sys

script_directory = os.path.dirname(os.path.abspath('__file__'))
framework_directory = os.path.abspath(os.path.join(script_directory, '..'))
sys.path.append(framework_directory)

from utils import SparseDataset

from models.mlp import MLP
from data.preprocess_data import add_rank, add_fold_assignments, add_standardization

X = scipy.sparse.load_npz(os.path.join(framework_directory, 'Data/input_data.npz'))
y = np.load(os.path.join(framework_directory, 'Data/target_values.npy'))
with open(os.path.join(framework_directory, 'Data/le_species.pkl'), 'rb') as f:
    le_species = pickle.load(f)
with open(os.path.join(framework_directory, 'Data/le_header.pkl'), 'rb') as f:
    le_header = pickle.load(f)
split_assignments = np.load(os.path.join(framework_directory, 'Data/split_assignments.npy'))

attributions = torch.zeros((2, len(le_header.classes_), X.shape[1]))
ablations = torch.zeros((2, len(le_header.classes_), 2))
ranks = torch.zeros((len(le_species.classes_))).cuda()
predictions = []
models = {}

X = add_rank(X, len(le_species.classes_))

for fold in range(2):
    print('\n' + '*'*11)
    print('* Fold: {} *'.format(fold))
    print('*'*11 + '\n')
    
    X_train, X_test, y_train, y_test = add_fold_assignments(X, y, split_assignments, fold)
    
    X_test_species = X_test[:, :len(le_species.classes_)]

    sorted_indices = []

    for i in range(X_test_species.shape[0]):
        data = X_test_species[i].data
        col_indices = X_test_species[i].indices
        sorted_idx = col_indices[np.argsort(data)][::-1]
        sorted_indices.append(torch.from_numpy(sorted_idx.copy()).cuda().long())

    X_train, scaler = add_standardization(X_train, scaler=None)
    X_test, scaler = add_standardization(X_test, scaler=scaler)
    
    train_dataloader = torch.utils.data.DataLoader(dataset=SparseDataset(X_train, y_train), batch_size=256, shuffle=True, num_workers=multiprocessing.cpu_count())
    test_dataloader = torch.utils.data.DataLoader(dataset=SparseDataset(X_test, y_test), batch_size=256, shuffle=False, num_workers=multiprocessing.cpu_count())

    model = MLP(input_dim=X.shape[1], output_dim=len(le_header.classes_))
    model = torch.nn.DataParallel(model)
    model.cuda()
    
    criterion = torch.nn.CrossEntropyLoss()
    criterion.cuda()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5)
    
    best_fold_accuracy = -np.inf
    best_epoch = -1
    no_improvement_epochs = 0
    
    for epoch in range(1):
        
        for batch, labels in train_dataloader:
            batch = batch.requires_grad_().cuda()
            labels = labels.cuda()
            optimizer.zero_grad()
            outputs = model(batch)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()    
            
        metric = torchmetrics.Accuracy(task="multiclass", num_classes=len(le_header.classes_), average='micro', top_k=1)
        metric.cuda()
        
        model.eval()
        
        with torch.no_grad():
            for batch, labels in test_dataloader:
                batch = batch.cuda()
                labels = labels.cuda()
                outputs = model(batch)
                accuracy = metric(outputs, labels)
        accuracy_epoch = metric.compute() * 100
        print('-'*30)
        print('Epoch {} completed.'.format(epoch))
        print(f'Accuracy: {accuracy_epoch:.4f}%.')
        print('-'*30 + '\n')
        model.train()
        
        scheduler.step(accuracy)
        
        if accuracy_epoch > best_fold_accuracy:
            best_fold_accuracy = accuracy_epoch
            best_epoch = epoch
            no_improvement_epochs = 0
        else:
            no_improvement_epochs += 1
        if no_improvement_epochs == 10:             
            print(f'Early stopping occurred at epoch {epoch} with best epoch = {best_epoch} and best accuracy = {best_fold_accuracy:.4f}%\n')
            break
            
    if no_improvement_epochs != 10:
        print(f'Early stopping dit not occur as best epoch = {best_epoch} and best accuracy = {best_fold_accuracy:.4f}%\n')

    models[f'fold_{fold}'] = model.state_dict()

    with torch.no_grad():
        predictions_fold = model(torch.from_numpy(X_test.toarray()).float())

    predictions_fold = scipy.special.softmax(predictions_fold.cpu(), axis=1)
    predictions_fold = [np.argmax(sub_arr) for sub_arr in predictions_fold]
    
    predictions.append(np.asarray(predictions_fold))

    X_test_tensor = torch.from_numpy(X_test.toarray()).float().cuda()
    y_test_tensor = torch.from_numpy(y_test).cuda()

    num_samples = len(X_test_tensor)
    num_classes = len(le_header.classes_)
    batch_size = 256

    integrated_gradients = captum.attr.IntegratedGradients(model)
    feature_ablation = captum.attr.FeatureAblation(model) 

    feature_mask = torch.zeros((X_test.shape[1]), dtype=torch.int).cuda()
    feature_mask[:len(le_species.classes_)] = 0 
    feature_mask[len(le_species.classes_):] = 1

    attributions_sums = torch.zeros(num_classes, X_test_tensor.shape[1]).cuda()
    ablations_sums = torch.zeros(num_classes, 2).cuda()

    class_counts = torch.zeros(num_classes).cuda()
    
    for i in tqdm.tqdm(range(0, num_samples, batch_size), desc=f'Interpretability of fold {fold}'):
        start_idx = i
        end_idx = min(i + batch_size, num_samples)

        batch_X = X_test_tensor[start_idx:end_idx]
        batch_y = y_test_tensor[start_idx:end_idx]
        sorted_indices_batch = sorted_indices[start_idx:end_idx]
        
        attributions_batch = integrated_gradients.attribute(batch_X, target=batch_y)
        ablations_batch = feature_ablation.attribute(batch_X, target=batch_y, feature_mask=feature_mask)
       
        attributions_batch = torch.abs(attributions_batch)
        ablations_batch = torch.abs(ablations_batch)

        ablations_internal = ablations_batch[:, :len(le_species.classes_)]
        ablations_external = ablations_batch[:, len(le_species.classes_):]

        mean_internal = torch.mean(ablations_internal, dim=1)
        mean_external = torch.mean(ablations_external, dim=1)

        ablations_batch = torch.stack((mean_internal, mean_external), dim=1)

        for j in range(num_classes):
            mask = batch_y == j
            class_counts[j] += mask.sum()
            attributions_sums[j] += attributions_batch[mask].sum(dim=0)
            ablations_sums[j] += ablations_batch[mask].sum(dim=0)

        attributions_ranks = attributions_batch[:, :len(le_species.classes_)]
        
        for j in range(len(attributions_ranks)):
            result = torch.zeros(len(le_species.classes_)).cuda()
            result[:len(sorted_indices[j])] = attributions_ranks[j][sorted_indices[j]]
            ranks += result

    species_per_plot = np.diff(X_test[:, :len(le_species.classes_)].indptr)
    plots_per_species = torch.zeros((len(le_species.classes_),), dtype=torch.int).cuda()
    
    for species_count in species_per_plot:
        plots_per_species[:species_count + 1] += 1

    plots_per_species[plots_per_species == 0] = 1

    ranks = ranks / plots_per_species

    class_counts[class_counts == 0] = 1

    attributions_fold = attributions_sums / class_counts.view(-1, 1)
    ablations_fold = ablations_sums / class_counts.view(-1, 1)    

    attributions[fold] = attributions_fold
    ablations[fold] = ablations_fold

habitats = torch.zeros((num_classes))

for fold in range(2):
    for habitat in range(num_classes):
        if np.count_nonzero(y[split_assignments == fold] == habitat): 
            habitats[habitat] += 1

attributions_transposed = attributions.permute(0, 2, 1)
ablations_transposed = ablations.permute(0, 2, 1)

expanded_habitats = habitats.unsqueeze(0).unsqueeze(1)

averaged_attributions = attributions_transposed / expanded_habitats
averaged_ablations = ablations_transposed / expanded_habitats

attributions = averaged_attributions.mean(dim=0)
ablations = averaged_ablations.mean(dim=0)

attributions = attributions.permute(1, 0)
ablations = ablations.permute(1, 0)

torch.save(attributions, "attributions.pt")
torch.save(ablations, "ablations.pt")
torch.save(ranks, "ranks.pt")
with open('predictions.pkl', 'wb') as file:
    pickle.dump(predictions, file)
with open('models.pkl', 'wb') as file:
    pickle.dump(models, file)
