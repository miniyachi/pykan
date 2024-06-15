import numpy as np
import os
from models.custom_dataset import CustomDataset
from torch.optim import Adam, SGD
from torchmetrics.classification import MulticlassAccuracy
from SFN import SFN

def check_opt(opt_name):
    if opt_name not in ['sgd', 'adam', 'sfn']:
        raise ValueError(f"Invalid optimizer name: {opt_name}")

def _get_train_val_test(data_folder, id):
    data_folder_id = os.path.join(data_folder, str(id)) 
    X_train = np.load(os.path.join(data_folder_id, 'X_train.npy'))
    X_val = np.load(os.path.join(data_folder_id, 'X_val.npy'))
    X_test = np.load(os.path.join(data_folder_id, 'X_test.npy'))
    y_train = np.load(os.path.join(data_folder_id, 'y_train.npy'), allow_pickle=True)
    y_val = np.load(os.path.join(data_folder_id, 'y_val.npy'), allow_pickle=True)
    y_test = np.load(os.path.join(data_folder_id, 'y_test.npy'), allow_pickle=True)

    return X_train, X_val, X_test, y_train, y_val, y_test

def get_tune_dataset(data_folder, id):
    X_train, X_val, _, y_train, y_val, _ = _get_train_val_test(data_folder, id)

    train_dataset = CustomDataset(X_train, y_train)
    val_dataset = CustomDataset(X_val, y_val)

    return train_dataset, val_dataset, X_train.shape[1], y_train

def get_final_dataset(data_folder, id):
    X_train, X_val, X_test, y_train, y_val, y_test = _get_train_val_test(data_folder, id)

    # Concatenate train and validation sets
    X_train = np.concatenate((X_train, X_val))
    y_train = np.concatenate((y_train, y_val))

    train_dataset = CustomDataset(X_train, y_train)
    test_dataset = CustomDataset(X_test, y_test)

    return train_dataset, test_dataset, X_train.shape[1], y_train


# Function to get the optimizer class
def get_opt(name):
    if name == 'sfn':
        return SFN  # Replace SFN with your custom optimizer class
    elif name == 'sgd':
        return SGD
    elif name == 'adam':
        return Adam
    # Add more optimizers as needed
    else:
        raise ValueError(f"Unknown optimizer name: {name}")


def compute_loss_acc(model, loss_fn, dataloader, num_classes, device):
    loss = 0
    n_samples = 0
    accuracy = MulticlassAccuracy(num_classes, average='macro').to(device) # Balanced accuracy

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        y_pred = model(x)

        loss += loss_fn(y_pred, y).item() * y.size(0)
        n_samples += y.size(0)

        pred_labels = y_pred.argmax(dim=1)
        accuracy.update(pred_labels, y)

    return loss / n_samples, accuracy.compute()