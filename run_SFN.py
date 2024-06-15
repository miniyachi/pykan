import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import sklearn
import wandb
import timeit

from torch.optim import SGD, Adam
from SFN import SFN
from utils import check_opt, get_final_dataset, get_opt, compute_loss_acc
from models.mlp import MLP

import argparse

# def get_best_lr(entity_name, tuning_name, metric, criteria):
#     direction = 'max' if metric == 'valid_acc' else 'min'

#     api = wandb.Api()
#     runs = api.runs(f"{entity_name}/{tuning_name}")
#     best_run = None
#     best_metric_val = 0 if direction == 'max' else float('inf')

#     for run in runs:
#         if run.state == 'finished':
#             if criteria == 'peak':
#                 metric_val = max(run.history()[metric]) if direction == 'max' else min(run.history()[metric])
#             elif criteria == 'final':
#                 metric_val = run.history()[metric].iloc[-1]

#             if (direction == 'max' and metric_val > best_metric_val) or (direction == 'min' and metric_val < best_metric_val):
#                 best_run = run
#                 best_metric_val = metric_val

#     return best_run.config['lr']

# Run function to test the MLP model with the custom SFN optimizer
def run(lr, proj_name, opt_name, run_name,
        input_size, output_size, hidden_layers, 
        loss_fn, 
        t_dl, t_dl_h, t_dl_2, tst_dl, 
        epochs, 
        device):
    run = wandb.init(project=proj_name, name=run_name, reinit=True)

    # Create neural net
    model = MLP(input_size, output_size, hidden_layers).to(device)

    # Get optimizer
    Opt = get_opt(opt_name)
    opt = None

    wandb.config.lr = lr

    if Opt == SFN:
        hes_update_freq = len(t_dl)
        opt = Opt(model.parameters(), lr=lr, rho=0.1, rank=10, chunk_size=1, verbose=False)
    elif Opt == SGD:
        opt = Opt(model.parameters(), lr=lr, momentum=0.9)
    elif Opt == Adam:
        opt = Opt(model.parameters(), lr=lr)

    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=15, T_mult=2)

    # Perform optimization
    n_iters = 0
    cum_time = 0
    for epoch in range(epochs):
        # Train the model
        model.train()
        epoch_start = timeit.default_timer()
        for x, y in t_dl:
            # Infrequent updating for SFN
            if isinstance(opt, SFN) and n_iters % hes_update_freq == 0:
                for x_h, y_h in t_dl_h:
                    x_h, y_h = x_h.to(device), y_h.to(device)
                    y_h_hat = model(x_h)
                    l_h = loss_fn(y_h_hat, y_h)
                    break
                grad_tuple = torch.autograd.grad(l_h, model.parameters(), create_graph=True)
                opt.update_preconditioner(grad_tuple)

            opt.zero_grad()
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            loss = loss_fn(y_hat, y)

            loss.backward()

            opt.step()
            n_iters += 1
            if n_iters % 100 == 0:
                print(f"Epoch: {epoch}, Iteration: {n_iters}, Loss: {loss.item()}")

        scheduler.step()
        cum_time += timeit.default_timer() - epoch_start

        model.eval()
        with torch.no_grad():
            train_loss, train_acc = compute_loss_acc(model, loss_fn, t_dl_2, output_size, device)
            test_loss, test_acc = compute_loss_acc(model, loss_fn, tst_dl, output_size, device)
            wandb.log({"epoch": epoch,
                       "time": cum_time,
                       "train_loss": train_loss,
                       "test_loss": test_loss,
                       "train_acc": train_acc,
                       "test_acc": test_acc})
    run.finish()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type = int, required = True)
    parser.add_argument('--data_folder', type = str, required = False, default = './data')
    parser.add_argument('--t_seed', type = int, required = False, default = 1729)
    parser.add_argument('--proj_name', type = str, required = True)
    parser.add_argument('--entity_name', type = str, required = True)
    parser.add_argument('--tuning_name', type = str, required = True)
    parser.add_argument('--metric', choices = ['valid_loss', 'valid_acc'], required = False, default = 'valid_acc')
    parser.add_argument('--criteria', choices = ['peak', 'final'], required = False, default = 'final')
    parser.add_argument('--opt', type = str, required = True)
    parser.add_argument('--epochs', type = int, required = False, default = 105)
    parser.add_argument('--n_trials', type = int, required = True)

    # Extract arguments
    args = parser.parse_args()
    # ids: volkert = 41166, Fashion-MNIST = 40996, Devnagari-Script = 40923
    id = args.id 
    data_folder = args.data_folder
    t_seed = args.t_seed
    proj_name = args.proj_name # Where to save results in wandb
    entity_name = args.entity_name # Where to obtain tuning results from wandb
    tuning_name = args.tuning_name # Where to obtain tuning results from wandb
    metric = args.metric # Which metric to use for tuning
    criteria = args.criteria # Whether to use the peak or final value of the tuning metric
    opt = args.opt
    epochs = args.epochs
    n_trials = args.n_trials

    # Check optimizer name
    check_opt(opt)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # The training set is the concatenation of the training and validation sets used for tuning
    train_dataset, test_dataset, input_size, y_train = get_final_dataset(data_folder, id)

    n_train = len(train_dataset)
    bh = int(n_train ** 0.5)
    bsz = 128

    train_dataloader = DataLoader(train_dataset, batch_size=bsz, shuffle=True)
    train_dataloader_hess = DataLoader(train_dataset, batch_size=bh, shuffle=True) # Used for updating the preconditioner
    train_dataloader2 = DataLoader(train_dataset, batch_size=4096, shuffle=False) # Used for computing metrics on the training set
    test_dataloader = DataLoader(test_dataset, batch_size=4096, shuffle=False)

    # Compute class weights using the training set
    n_classes = len(np.unique(y_train))
    class_weights = sklearn.utils.class_weight.compute_class_weight(
        'balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    hidden_layers = [512] * 9

    for lr in [1e-4, 1e-3, 1e-2, 1e-1]:
    # Get best learning rate by querying wandb
    # best_lr = get_best_lr(entity_name, tuning_name, metric, criteria)

        for i in range(n_trials):
            run_name = f"{opt}_{i}_lr={lr}"
            torch.manual_seed(t_seed + i) # Change random seed for each run
            run(lr,
                proj_name, 
                opt,
                run_name,
                input_size, 
                n_classes, 
                hidden_layers,
                loss_fn,
                train_dataloader, 
                train_dataloader_hess, 
                train_dataloader2, 
                test_dataloader,
                epochs,
                device)






# Example usage
if __name__ == "__main__":
    main()
    # data_folder = "./data"
    # id = 41166

    # # Define parameters
    # lr = 1e-4
    # proj_name = "MLP_SFN_Test"
    # opt_name = "Adam"
    # input_size = 784
    # output_size = 10
    # hidden_layers = [512] * 3
    
    # # The training set is the concatenation of the training and validation sets used for tuning
    # train_dataset, test_dataset, input_size, y_train = get_final_dataset(data_folder, id)

    # n_train = len(train_dataset)
    # bh = int(n_train ** 0.5)
    # bsz = 128

    # train_dataloader = DataLoader(train_dataset, batch_size=bsz, shuffle=True)
    # train_dataloader_hess = DataLoader(train_dataset, batch_size=bh, shuffle=True) # Used for updating the preconditioner
    # train_dataloader2 = DataLoader(train_dataset, batch_size=4096, shuffle=False) # Used for computing metrics on the training set
    # test_dataloader = DataLoader(test_dataset, batch_size=4096, shuffle=False)

    # epochs = 50
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # # Compute class weights using the training set
    # n_classes = len(np.unique(y_train))
    # class_weights = sklearn.utils.class_weight.compute_class_weight(
    #     'balanced',
    #     classes=np.unique(y_train),
    #     y=y_train
    # )
    # class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    # loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    # # Run the training
    # run(lr, proj_name, opt_name, 
    #     input_size, output_size, hidden_layers, 
    #     loss_fn, 
    #     train_dataloader, 
    #     train_dataloader_hess, 
    #     train_dataloader2, 
    #     test_dataloader, 
    #     epochs, 
    #     device)
