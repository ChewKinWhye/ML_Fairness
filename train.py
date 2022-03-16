import torch
from torch import nn
import time
from evaluate import calculate_group_accuracies
import numpy as np
from sklearn.model_selection import train_test_split
import copy
from dataset import obtain_color_idxs, obtain_class_color_idxs
# Train model

def train_model(net, x_train, y_train, train_color_labels, device, mean, std, lr, max_epochs, bs, mode, mode_grouping, validation_split, weight_decay):
    patience = 10
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(reduction='none')
    if mode == 'discard':
        if mode_grouping == "color":
            color_0_idx, color_1_idx = obtain_color_idxs(train_color_labels)
            min_size = min(len(color_0_idx[0]), len(color_1_idx[0]))
            delete_indxs = np.array(list(color_0_idx[0][min_size:]) + list(color_1_idx[0][min_size:]))
        else:
            class_0_color_0_idx, class_0_color_1_idx, class_1_color_0_idx, class_1_color_1_idx = obtain_class_color_idxs(y_train, train_color_labels)
            min_size = min(len(class_0_color_0_idx[0]), len(class_0_color_1_idx[0]),
                           len(class_1_color_0_idx[0]), len(class_1_color_1_idx[0]))
            delete_indxs = np.array(list(class_0_color_0_idx[0][min_size:]) + list(class_0_color_1_idx[0][min_size:]) +
                                    list(class_1_color_0_idx[0][min_size:]) + list(class_1_color_1_idx[0][min_size:]))
        x_train = np.delete(x_train, delete_indxs, 0)
        y_train = np.delete(y_train, delete_indxs, 0)
        train_color_labels = np.delete(train_color_labels, delete_indxs, 0)
    if mode == "reweight_sampling":
        if mode_grouping == "color":
            train_color_0_idx, train_color_1_idx = obtain_color_idxs(train_color_labels)
            samples_required = max(len(train_color_0_idx[0]), len(train_color_1_idx[0]))
            for i in range(int(samples_required/len(train_color_0_idx[0])) - 1):
                x_train = np.append(x_train, x_train[train_color_0_idx], axis=0)
                y_train = np.append(y_train, y_train[train_color_0_idx], axis=0)
                train_color_labels = np.append(train_color_labels, train_color_labels[train_color_0_idx], axis=0)
            for i in range(int(samples_required/len(train_color_1_idx[0])) - 1):
                x_train = np.append(x_train, x_train[train_color_1_idx], axis=0)
                y_train = np.append(y_train, y_train[train_color_1_idx], axis=0)
                train_color_labels = np.append(train_color_labels, train_color_labels[train_color_1_idx], axis=0)
        else:
            train_class_0_color_0_idx, train_class_0_color_1_idx, train_class_1_color_0_idx, train_class_1_color_1_idx = obtain_class_color_idxs(
                y_train, train_color_labels)
            samples_required = max(len(train_class_0_color_0_idx[0]), len(train_class_0_color_1_idx[0]), len(train_class_1_color_0_idx[0]), len(train_class_1_color_1_idx[0]))
            for i in range(int(samples_required/len(train_class_0_color_0_idx[0])) - 1):
                x_train = torch.cat((x_train, x_train[train_class_0_color_0_idx]), 0)
                y_train = torch.cat((y_train, y_train[train_class_0_color_0_idx]), 0)
                train_color_labels = torch.cat((train_color_labels, train_color_labels[train_class_0_color_0_idx]), 0)
            for i in range(int(samples_required/len(train_class_0_color_1_idx[0])) - 1):
                x_train = torch.cat((x_train, x_train[train_class_0_color_1_idx]), 0)
                y_train = torch.cat((y_train, y_train[train_class_0_color_1_idx]), 0)
                train_color_labels = torch.cat((train_color_labels, train_color_labels[train_class_0_color_1_idx]), 0)
            for i in range(int(samples_required/len(train_class_1_color_0_idx[0])) - 1):
                x_train = torch.cat((x_train, x_train[train_class_1_color_0_idx]), 0)
                y_train = torch.cat((y_train, y_train[train_class_1_color_0_idx]), 0)
                train_color_labels = torch.cat((train_color_labels, train_color_labels[train_class_1_color_0_idx]), 0)
            for i in range(int(samples_required/len(train_class_1_color_1_idx[0])) - 1):
                x_train = torch.cat((x_train, x_train[train_class_1_color_1_idx]), 0)
                y_train = torch.cat((y_train, y_train[train_class_1_color_1_idx]), 0)
                train_color_labels = torch.cat((train_color_labels, train_color_labels[train_class_1_color_1_idx]), 0)

    if mode_grouping == "color":
        color_0_idx, color_1_idx = obtain_color_idxs(train_color_labels)
        train_reweight_loss = np.zeros(len(x_train))
        train_reweight_loss[color_0_idx] = (len(x_train) * 0.5) / len(color_0_idx[0])
        train_reweight_loss[color_1_idx] = (len(x_train) * 0.5) / len(color_1_idx[0])
    else:
        class_0_color_0_idx, class_0_color_1_idx, class_1_color_0_idx, class_1_color_1_idx = obtain_class_color_idxs(
            y_train, train_color_labels)
        train_reweight_loss = np.zeros(len(x_train))

        train_reweight_loss[class_0_color_0_idx] = (len(x_train) * 0.25) / len(class_0_color_0_idx[0])
        train_reweight_loss[class_0_color_1_idx] = (len(x_train) * 0.25) / len(class_0_color_1_idx[0])
        train_reweight_loss[class_1_color_0_idx] = (len(x_train) * 0.25) / len(class_1_color_0_idx[0])
        train_reweight_loss[class_1_color_1_idx] = (len(x_train) * 0.25) / len(class_1_color_1_idx[0])

    if validation_split:
        x_train, x_val, y_train, y_val, train_color_labels, val_color_labels, train_reweight_loss, val_reweight_loss = train_test_split(x_train, y_train, train_color_labels, train_reweight_loss, test_size=0.2, random_state=42)

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
            batch_indices = shuffled_indices[idx:idx+bs]
            minibatch_data = x_train[batch_indices]
            minibatch_label = y_train[batch_indices]
            minibatch_data = minibatch_data.to(device)
            minibatch_label = minibatch_label.to(device)
            inputs = (minibatch_data - mean) / std
            inputs.requires_grad_()
            scores = net(inputs)
            loss = criterion(scores, minibatch_label)
            if mode == "reweight_loss":
                loss = torch.dot(loss, torch.tensor(train_reweight_loss[batch_indices]).float().to(device)) / bs
            else:
                loss = torch.mean(loss)
            loss.backward()
            optimizer.step()
            training_loss += loss.detach().item()
            training_acc += torch.sum((torch.max(scores, 1)[1] == minibatch_label).float()).data.item()
            num_train_batches += 1
        if validation_split:
            net.eval()
            val_loss, val_acc, num_val_batches = 0, 0, 0
            shuffled_indices = torch.randperm(len(x_val))
            with torch.no_grad():
                for idx in range(0, len(x_val), bs):
                    batch_indices = shuffled_indices[idx:idx + bs]
                    minibatch_data = x_val[batch_indices]
                    minibatch_label = y_val[batch_indices]
                    minibatch_data = minibatch_data.to(device)
                    minibatch_label = minibatch_label.to(device)
                    inputs = (minibatch_data - mean) / std
                    scores = net(inputs, bn_training=False)
                    loss = criterion(scores, minibatch_label)
                    if mode == "reweight_loss":
                        loss = torch.dot(loss, torch.tensor(val_reweight_loss[batch_indices]).float().to(device)) / bs
                    else:
                        loss = torch.mean(loss)
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
