import torch
from torch import nn
import time
from evaluate import calculate_group_accuracies
import numpy as np
from sklearn.model_selection import train_test_split
import copy

# Train model

def train_model(net, x_train, y_train, train_color_labels, device, mean, std, lr, max_epochs, bs, mode, validation_split, weight_decay):
    if validation_split:
        x_train, x_val, y_train, y_val, train_color_labels, val_color_labels = train_test_split(x_train, y_train, train_color_labels, test_size=0.2, random_state=42)
    patience = 10
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=weight_decay)
    color_0_idx = np.where((train_color_labels == 0))
    color_1_idx = np.where((train_color_labels == 1))
    if mode == "reweight_loss":
        class_weights = torch.FloatTensor([1/len(color_0_idx), 1/len(color_1_idx)]).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()
    if mode == 'reweight_sampling':
        reweight_distribution = np.zeros(len(x_train))
        reweight_distribution[color_0_idx] = (0.5 / len(color_0_idx[0]))
        reweight_distribution[color_1_idx] = (0.5 / len(color_1_idx[0]))
    if mode == 'discard':
        # Discard the samples from the majority group
        if len(color_0_idx[0]) > len(color_1_idx[0]):
            x_train = np.delete(x_train, color_0_idx[0][len(color_1_idx[0]):], 0)
            y_train = np.delete(y_train, color_0_idx[0][len(color_1_idx[0]):], 0)
            train_color_labels = np.delete(train_color_labels, color_0_idx[0][len(color_1_idx[0]):], 0)
        else:
            x_train = np.delete(x_train, color_1_idx[0][len(color_0_idx[0]):], 0)
            y_train = np.delete(y_train, color_1_idx[0][len(color_0_idx[0]):], 0)
            train_color_labels = np.delete(train_color_labels, color_1_idx[0][len(color_0_idx[0]):], 0)
    lowest_val_loss = float('inf')
    patience_steps = 0
    best_params = copy.deepcopy(net.state_dict())
    for epoch in range(max_epochs):
        start = time.time()
        training_loss, training_acc, num_train_batches = 0, 0, 0
        shuffled_indices = torch.randperm(len(x_train))
        net.train()
        for idx in range(0, len(x_train), bs):
            optimizer.zero_grad()
            if mode == 'reweight_sampling':
                batch_indices = np.random.choice(np.arange(len(x_train)), bs, replace=False, p=reweight_distribution)
            elif mode == 'standard' or mode == 'discard' or mode == 'reweight_loss':
                batch_indices = shuffled_indices[idx:idx+bs]
            minibatch_data = x_train[batch_indices]
            minibatch_label = y_train[batch_indices]
            minibatch_data = minibatch_data.to(device)
            minibatch_label = minibatch_label.to(device)
            inputs = (minibatch_data - mean) / std
            inputs.requires_grad_()
            scores = net(inputs)
            loss = criterion(scores, minibatch_label)
            loss.backward()
            optimizer.step()
            training_loss += loss.detach().item()
            training_acc += torch.sum((torch.max(scores, 1)[1] == minibatch_label).float()).data.item()
            num_train_batches += 1
        if validation_split:
            net.eval()
            val_loss, val_acc, num_val_batches = 0, 0, 0
            with torch.no_grad():
                for idx in range(0, len(x_val), bs):
                    minibatch_data = x_val[idx:idx+bs]
                    minibatch_label = y_val[idx:idx+bs]
                    minibatch_data = minibatch_data.to(device)
                    minibatch_label = minibatch_label.to(device)
                    inputs = (minibatch_data - mean) / std
                    scores = net(inputs)
                    loss = criterion(scores, minibatch_label)
                    val_loss += loss.detach().item()
                    val_acc += torch.sum((torch.max(scores, 1)[1] == minibatch_label).float()).data.item()
                    num_val_batches += 1
            group_validation_accuracies, color_validation_accuracies = calculate_group_accuracies(net, x_val, y_val, val_color_labels, mean, std, device, bs)
            if val_loss > lowest_val_loss:
                patience_steps += 1
            else:
                best_params = copy.deepcopy(net.state_dict())
                patience_steps = 0
                lowest_val_loss = val_loss
            if patience_steps > patience:
                net.load_state_dict(best_params)
                break

        group_training_accuracies, color_training_accuracies = calculate_group_accuracies(net, x_train, y_train, train_color_labels, mean, std, device, bs)

        elapsed = (time.time() - start) / 60
        print(f'Epoch: {epoch+1}, Time: {elapsed:.5f} min')
        print(f'Training Loss: {training_loss/num_train_batches:.5f}, Training Accuracy: {training_acc/len(x_train):.5f}')
        print(f'Group Training Accuracy: {group_training_accuracies[0]:.5f}, {group_training_accuracies[1]:.5f}, {group_training_accuracies[2]:.5f}, {group_training_accuracies[3]:.5f}')
        print(f'Color Training Accuracy: {color_training_accuracies[0]:.5f}, {color_training_accuracies[1]:.5f}')
        if validation_split:
            print(f'Validation Loss: {val_loss/num_val_batches:.5f}, Validation Accuracy: {val_acc/len(x_val):.5f}')
            print(f'Group Validation Accuracy: {group_validation_accuracies[0]:.5f}, {group_validation_accuracies[1]:.5f}, {group_validation_accuracies[2]:.5f}, {group_validation_accuracies[3]:.5f}')
            print(f'Color Validation Accuracy: {color_validation_accuracies[0]:.5f}, {color_validation_accuracies[1]:.5f}\n')
    return net
