import numpy as np
import torch


def calculate_group_accuracies(net, x, y, color_labels, mean, std, device, bs):
    group_accuracies = []
    for class_label in [0, 1]:
        for color_label in [0, 1]:
            group_acc = 0
            group_idx = np.where((y == class_label) & (color_labels == color_label))[0]
            x_group = x[group_idx]
            y_group = y[group_idx]
            for idx in range(0, len(x_group), bs):
                minibatch_data = x_group[idx:idx + bs]
                minibatch_label = y_group[idx:idx + bs]
                minibatch_data = minibatch_data.to(device)
                minibatch_label = minibatch_label.to(device)
                inputs = (minibatch_data - mean) / std
                scores = net(inputs, bn_training=False)
                group_acc += torch.sum((torch.max(scores, 1)[1] == minibatch_label).float()).data.item()
            group_accuracies.append(group_acc/len(group_idx))
            # print(f'Group: Class {class_label} Color {color_label}, Accuracy: {group_acc/len(group_idx)}')
    color_accuracies = []
    for color_label in [0, 1]:
        group_acc = 0
        group_idx = np.where((color_labels == color_label))[0]
        x_group = x[group_idx]
        y_group = y[group_idx]
        for idx in range(0, len(x_group), bs):
            minibatch_data = x_group[idx:idx + bs]
            minibatch_label = y_group[idx:idx + bs]
            minibatch_data = minibatch_data.to(device)
            minibatch_label = minibatch_label.to(device)
            inputs = (minibatch_data - mean) / std
            scores = net(inputs, bn_training=False)
            group_acc += torch.sum((torch.max(scores, 1)[1] == minibatch_label).float()).data.item()
        color_accuracies.append(group_acc/len(group_idx))
    return group_accuracies, color_accuracies
