from torchvision import datasets
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import pandas
from sklearn.utils import shuffle
from torchvision.utils import save_image
import os

# TODO: Make function callable and add hyperparameters

# Display torch tensor image
def display_image(tensor):
    if tensor.size()[0] == 3:
        image = transforms.ToPILImage()(tensor)
    else:
        image = Image.fromarray(np.array(tensor, dtype=np.uint8))
    image.show()


# Downloads/loads the raw dataset
def get_raw_dataset(dataset_name, dataset_path):
    if dataset_name == 'MNIST':
        raw_train = datasets.MNIST(dataset_path + '/MNIST', train=True, download=True)
        raw_test = datasets.MNIST(dataset_path + '/MNIST', train=False, download=True)
    elif dataset_name == 'FashionMNIST':
        raw_train = datasets.FashionMNIST(dataset_path + '/FashionMNIST', train=True, download=True)
        raw_test = datasets.FashionMNIST(dataset_path + '/FashionMNIST', train=False, download=True)
    else:
        raise Exception("Invalid dataset name")
    x_train = raw_train.data
    y_train = raw_train.targets
    x_test = raw_test.data
    y_test = raw_test.targets
    return x_train, y_train, x_test, y_test


# Returns data points in class 0 or class 1
def filter_classes(x_train, y_train, x_test, y_test, classes):
    train_idx = np.where((y_train == classes[0]) | (y_train == classes[1]))[0]
    test_idx = np.where((y_test == classes[0]) | (y_test == classes[1]))[0]
    return x_train[train_idx], y_train[train_idx], x_test[test_idx], y_test[test_idx]


# Converts greyscale image to rgb
def convert_to_rgb(x):
    x.unsqueeze_(1)
    x = x.repeat(1, 3, 1, 1)
    return x


# Adds color to image
def add_color(x, color, color_intensity):
    color_change_threshold = 10
    x_channel_0 = x[:, 0, :, :]
    x_channel_1 = x[:, 1, :, :]
    x_channel_2 = x[:, 2, :, :]
    if color == 'red':
        x_channel_0[x_channel_0 < color_change_threshold] = color_intensity
        x_channel_1[x_channel_1 < color_change_threshold] = 0
        x_channel_2[x_channel_2 < color_change_threshold] = 0
    elif color == 'green':
        x_channel_0[x_channel_0 < color_change_threshold] = 0
        x_channel_1[x_channel_1 < color_change_threshold] = color_intensity
        x_channel_2[x_channel_2 < color_change_threshold] = 0

    elif color == 'blue':
        x_channel_0[x_channel_0 < color_change_threshold] = 0
        x_channel_1[x_channel_1 < color_change_threshold] = 0
        x_channel_2[x_channel_2 < color_change_threshold] = color_intensity
    else:
        raise Exception("Invalid color")
    # display_image(x[0])
    return x


def add_color_imbalance(x_train, y_train, colors, color_fractions, color_intensity, classes, is_train):
    color_labels = []
    class_0_idx = np.where(y_train == classes[0])[0]
    class_1_idx = np.where(y_train == classes[1])[0]
    if is_train:
        class_0_fraction = color_fractions[0] + color_fractions[1]
        class_1_fraction = color_fractions[2] + color_fractions[3]
        if len(class_0_idx) / class_0_fraction < len(class_1_idx) / class_1_fraction:
            num_class_0 = len(class_0_idx)
            num_class_1 = int(num_class_0 * class_1_fraction / class_0_fraction)
        else:
            num_class_1 = len(class_1_idx)
            num_class_0 = int(num_class_1 * class_0_fraction / class_1_fraction)
        num_class_0_color_0 = int(num_class_0 * color_fractions[0] / class_0_fraction)
        num_class_1_color_0 = int(num_class_1 * color_fractions[2] / class_1_fraction)
    else:
        num_class_0 = len(class_0_idx)
        num_class_1 = len(class_1_idx)
        num_class_0_color_0 = int(num_class_0 * 0.5)
        num_class_1_color_0 = int(num_class_1 * 0.5)
    # Add color for class 0 color 0
    x_class_0_color_0 = add_color(x_train[class_0_idx[:num_class_0_color_0]], colors[0], color_intensity)
    y_class_0_color_0 = y_train[class_0_idx[:num_class_0_color_0]]
    color_labels.extend([0]*num_class_0_color_0)
    # Add color for class 0 color 1
    x_class_0_color_1 = add_color(x_train[class_0_idx[num_class_0_color_0:num_class_0]], colors[1], color_intensity)
    y_class_0_color_1 = y_train[class_0_idx[num_class_0_color_0:num_class_0]]
    color_labels.extend([1]*(num_class_0-num_class_0_color_0))
    # Add color for class 1 color 0
    x_class_1_color_0 = add_color(x_train[class_1_idx[:num_class_1_color_0]], colors[0], color_intensity)
    y_class_1_color_0 = y_train[class_1_idx[:num_class_1_color_0]]
    color_labels.extend([0]*num_class_1_color_0)
    # Add color for class 0 color 1
    x_class_1_color_1 = add_color(x_train[class_1_idx[num_class_1_color_0:num_class_1]], colors[1], color_intensity)
    y_class_1_color_1 = y_train[class_1_idx[num_class_1_color_0:num_class_1]]
    color_labels.extend([1]*(num_class_1-num_class_1_color_0))
    # Print sizes
    if is_train:
        sizes = [[f'{num_class_0_color_0} ({color_fractions[0]})', f'{num_class_0-num_class_0_color_0} ({color_fractions[1]})'],
                 [f'{num_class_1_color_0} ({color_fractions[2]})', f'{num_class_1-num_class_1_color_0} ({color_fractions[3]})']]
    else:
        sizes = [[f'{num_class_0_color_0}', f'{num_class_0-num_class_0_color_0}'],
                 [f'{num_class_1_color_0}', f'{num_class_1-num_class_1_color_0}']]

    row_labels = ['Class 0', 'Class 1']
    column_labels = ['Color 0', 'Color 1']
    df = pandas.DataFrame(sizes, columns=column_labels, index=row_labels)
    print(df)

    return torch.cat((x_class_0_color_0, x_class_0_color_1, x_class_1_color_0, x_class_1_color_1), 0),\
           torch.cat((y_class_0_color_0, y_class_0_color_1, y_class_1_color_0, y_class_1_color_1), 0), \
           torch.Tensor(color_labels)


def load_data(dataset_path, dataset_name, colors, color_intensity, color_fractions, save_modified_dataset, classes):
    assert sum(list(color_fractions)) == 1

    x_train, y_train, x_test, y_test = get_raw_dataset(dataset_name, dataset_path)
    # Shuffle both train and test as color is added by sequential indexing
    x_train, y_train = shuffle(x_train, y_train)
    x_test, y_test = shuffle(x_test, y_test)
    print('Raw Data Size:', x_train.size(), y_train.size(), x_test.size(), y_test.size())

    x_train, y_train, x_test, y_test = filter_classes(x_train, y_train, x_test, y_test, classes)
    print('Filtered Data Size:', x_train.size(), y_train.size(), x_test.size(), y_test.size())
    x_train, x_test = convert_to_rgb(x_train), convert_to_rgb(x_test)
    print('RGB Data Size:', x_train.size(), y_train.size(), x_test.size(), y_test.size())
    x_train, y_train, train_color_labels = add_color_imbalance(x_train, y_train, colors, color_fractions, color_intensity, classes, is_train=True)
    x_test, y_test, test_color_labels = add_color_imbalance(x_test, y_test, colors, color_fractions, color_intensity, classes, is_train=False)
    x_train, y_train, train_color_labels = shuffle(x_train, y_train, train_color_labels)
    print('Color Imbalanced Data Size:', x_train.size(), y_train.size(), x_test.size(), y_test.size())

    # Reindex labels
    y_train[np.where((y_train == classes[0]))[0]] = 0
    y_train[np.where((y_train == classes[1]))[0]] = 1
    y_test[np.where((y_test == classes[0]))[0]] = 0
    y_test[np.where((y_test == classes[1]))[0]] = 1

    # Save it to visually check dataset, only saving images
    if save_modified_dataset:
        modified_dataset_path = os.path.join(dataset_path, dataset_name + '_modified_train')
        if not os.path.exists(modified_dataset_path):
            os.makedirs(modified_dataset_path)
        for idx, image in enumerate(x_train):
            image_save_path = os.path.join(modified_dataset_path, f'img_{idx}.png')
            save_image(image/255, image_save_path)

        modified_dataset_path = os.path.join(dataset_path, dataset_name + '_modified_test')
        if not os.path.exists(modified_dataset_path):
            os.makedirs(modified_dataset_path)
        for idx, image in enumerate(x_test):
            image_save_path = os.path.join(modified_dataset_path, f'img_{idx}.png')
            save_image(image/255, image_save_path)

    return torch.Tensor.float(x_train), y_train, train_color_labels, torch.Tensor.float(x_test), y_test, test_color_labels