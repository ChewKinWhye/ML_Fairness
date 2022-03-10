from dataset import load_data
from model import Simple_CNN
from train import train_model
from evaluate import calculate_group_accuracies
import torch
import argparse

'''
python main.py --num_repeat 1 --dataset_name FashionMNIST --classes 2,4 --imbalance_ratio 0.45,0.05,0.05,0.45 --learning_rate 1e-2 --training_epochs 10 --batch_size 8 --weight_decay 1e-6 --CNN_channels 8 --mode standard
'''
def run(opts):
    dataset_path = 'data'
    colors = ('blue', 'green')

    device = torch.device("cuda")
    group_test_accuracies_average = [0, 0, 0, 0]
    color_test_accuracies_average = [0, 0]
    for _ in range(opts.num_repeat):
        # Data
        x_train, y_train, train_color_labels, x_test, y_test, test_color_labels = \
            load_data(dataset_path, opts.dataset_name, colors, opts.color_intensity, opts.imbalance_ratio, opts.to_save_dataset, opts.classes)
        # Model
        net = Simple_CNN(num_classes=2, width=opts.CNN_channels)
        print(net)
        # Train
        net = net.to(device)
        mean, std = x_train.mean().to(device), x_train.std().to(device)
        net = train_model(net, x_train, y_train, train_color_labels, device, mean, std, opts.learning_rate,
                          opts.training_epochs, opts.batch_size, opts.mode, opts.validation_split, opts.weight_decay)
        # Test
        group_test_accuracies, color_test_accuracies = calculate_group_accuracies(net, x_test, y_test, test_color_labels,
                                                                                  mean, std, device, opts.batch_size)
        group_test_accuracies_average = [x + y for x, y in zip(group_test_accuracies_average, group_test_accuracies)]
        color_test_accuracies_average = [x + y for x, y in zip(color_test_accuracies_average, color_test_accuracies)]
        print('\t Group Test Accuracy=', group_test_accuracies)
        print('\t Color Test Accuracy=', color_test_accuracies)
    group_test_accuracies_average = [x / opts.num_repeat for x in group_test_accuracies_average]
    color_test_accuracies_average = [x / opts.num_repeat for x in color_test_accuracies_average]
    print('\t Average Group Test Accuracy=', group_test_accuracies_average)
    print('\t Average Color Test Accuracy=', color_test_accuracies_average)
    return (group_test_accuracies_average[1] + group_test_accuracies_average[2]) / 2

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_repeat', help='Number of times to repeat experiment', type=int, default=1)
    parser.add_argument('--dataset_name', help='Which dataset to use, MNIST or FashionMNIST', type=str, default="MNIST")
    # "2,4" for FashionMNIST
    parser.add_argument('--classes', help='Which classes to use', type=str, default="3,5")
    parser.add_argument('--color_intensity', help='Intensity of color added, 0-255', type=int, default=150)
    parser.add_argument('--imbalance_ratio',
                        help='ratio of class, class_0_color_0, class_0_color_1, class_1_color_0, class__color_1',
                        type=str, default="0.4998,0.0002,0.0002,0.4998")
    parser.add_argument('--to_save_dataset', help='Whether to save the modified dataset', type=str, default="true")
    parser.add_argument('--validation_split', help='Whether to split train into train and val', type=str, default="true")
    parser.add_argument('--learning_rate', help='Learning rate for model', type=float, default=1e-2)
    parser.add_argument('--training_epochs', help='Number of training epochs', type=int, default=10)
    parser.add_argument('--batch_size', help='Training batch size', type=int, default=8)
    parser.add_argument('--weight_decay', help='Weight decay to use', type=float, default=1e-6)
    parser.add_argument('--CNN_channels', help='Number of channels for CNN first layer', type=int, default=16)
    parser.add_argument('--mode',
                        help='What mode to train model, can be [standard, reweight_sampling, discard, reweight_loss]',
                        type=str, default="standard")
    opts = parser.parse_args()
    opts.to_save_dataset = (opts.to_save_dataset == 'true')
    opts.validation_split = (opts.validation_split == 'true')
    opts.classes = tuple([int(x) for x in opts.classes.split(",")])
    opts.imbalance_ratio = tuple([float(x) for x in opts.imbalance_ratio.split(",")])
    run(opts)
