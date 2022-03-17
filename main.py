from dataset import load_data
from model import Simple_CNN
from train import train_model
from evaluate import calculate_group_accuracies
import torch
import argparse
from model import Learner, LoadLearner
from statistics import mean as ListMean

'''
python main.py --num_repeat 1 --dataset_name FashionMNIST --classes 2,4 --imbalance_ratio 0.45,0.05,0.05,0.45 --learning_rate 1e-2 --training_epochs 10 --batch_size 8 --weight_decay 1e-6 --CNN_channels 8 --mode standard
'''


# Default Arguments
def parse_args():
    parser = argparse.ArgumentParser()
    # Experiment Parameters
    parser.add_argument('--num_repeat', help='Number of times to repeat experiment', type=int, default=5)

    # Data parameters
    parser.add_argument('--dataset_name', help='Which dataset to use, MNIST or FashionMNIST', type=str, default="MNIST")
    parser.add_argument('--classes', help='Which classes to use', type=str, default="3,5")
    parser.add_argument('--color_intensity', help='Intensity of color added, 0-255', type=int, default=150)
    parser.add_argument('--imbalance_ratio',
                        help='ratio of class, class_0_color_0, class_0_color_1, class_1_color_0, class__color_1',
                        type=str, default="0.49,0.01,0.01,0.49")
    parser.add_argument('--to_save_dataset', help='Whether to save the modified dataset', type=str, default="true")
    parser.add_argument('--colors', help="What colors to color the image", type=str, default="blue,green")

    # Training Parameters
    # "2,4" for FashionMNIST
    parser.add_argument('--validation_split', help='Whether to split train into train and val', type=str,
                        default="true")
    parser.add_argument('--learning_rate', help='Learning rate for model', type=float, default=1e-2)
    parser.add_argument('--training_epochs', help='Number of training epochs', type=int, default=1000)
    parser.add_argument('--batch_size', help='Training batch size', type=int, default=8)
    parser.add_argument('--weight_decay', help='Weight decay to use', type=float, default=1e-4)
    parser.add_argument('--CNN_channels', help='Number of channels for CNN first layer', type=int, default=4)
    parser.add_argument('--mode',
                        help='What mode to train model, can be [standard, reweight_sampling, discard, reweight_loss]',
                        type=str, default="reweight_loss")
    parser.add_argument('--mode_grouping', help='How to group the data points, color or both', type=str, default="both")
    parser.add_argument('--maml', help='Whether to use maml, maml or none', type=str, default="none")
    parser.add_argument('--results_file', help='Where to save results', type=str, default="imbalance_ratio_results.csv")
    parser.add_argument('--worst_group', help='Which groups are the worst groups', type=str, default="1,2")

    opts = parser.parse_args()
    opts.to_save_dataset = (opts.to_save_dataset == 'true')
    opts.validation_split = (opts.validation_split == 'true')
    opts.classes = tuple([int(x) for x in opts.classes.split(",")])
    opts.imbalance_ratio = tuple([float(x) for x in opts.imbalance_ratio.split(",")])
    return opts


def run(opts):
    config = [
        ('conv2d', [opts.CNN_channels, 3, 3, 3, 2, 0]),
        ('relu', [True]),
        # ('bn', [opts.CNN_channels]),
        ('conv2d', [opts.CNN_channels, opts.CNN_channels, 3, 3, 2, 0]),
        ('relu', [True]),
        # ('bn', [opts.CNN_channels]),
        ('conv2d', [opts.CNN_channels, opts.CNN_channels, 3, 3, 2, 0]),
        ('relu', [True]),
        # ('bn', [opts.CNN_channels]),
        ('conv2d', [opts.CNN_channels, opts.CNN_channels, 2, 2, 1, 0]),
        ('relu', [True]),
        # ('bn', [opts.CNN_channels]),
        ('flatten', []),
        ('linear', [2, opts.CNN_channels])
    ]
    dataset_path = 'data'
    colors = tuple(opts.colors.split(","))

    device = torch.device("cuda")
    group_test_accuracies_list = []
    color_test_accuracies_list = []
    for _ in range(opts.num_repeat):
        # Data
        x_train, y_train, train_color_labels, x_test, y_test, test_color_labels = \
            load_data(dataset_path, opts.dataset_name, colors, opts.color_intensity, opts.imbalance_ratio,
                      opts.to_save_dataset, opts.classes)
        # Model
        if opts.maml == "maml":
            net = LoadLearner(config)
        else:
            net = Learner(config)
        # net = Simple_CNN(num_classes=2, width=opts.CNN_channels)
        # Train
        net = net.to(device)
        mean, std = x_train.mean().to(device), x_train.std().to(device)
        net = train_model(net, x_train, y_train, train_color_labels, device, mean, std, opts.learning_rate,
                          opts.training_epochs, opts.batch_size, opts.mode, opts.mode_grouping, opts.validation_split,
                          opts.weight_decay)
        # Test
        group_test_accuracies, color_test_accuracies = calculate_group_accuracies(net, x_test, y_test,
                                                                                  test_color_labels,
                                                                                  mean, std, device, opts.batch_size)
        print('\t Group Test Accuracy=', group_test_accuracies)
        print('\t Color Test Accuracy=', color_test_accuracies)
        group_test_accuracies_list.append(group_test_accuracies)
        color_test_accuracies_list.append(color_test_accuracies)
    worst_groups = [int(i) for i in opts.worst_group.split(",")]
    if len(worst_groups) == 2:
        worst_group_average_accuracy = ListMean([(i[worst_groups[0]] + i[worst_groups[1]]) / 2 for i in group_test_accuracies_list])
        worst_group_best_accuracy = max([(i[worst_groups[0]] + i[worst_groups[1]]) / 2 for i in group_test_accuracies_list])
        worst_group_worst_accuracy = min([(i[worst_groups[0]] + i[worst_groups[1]]) / 2 for i in group_test_accuracies_list])
    elif len(worst_groups) == 1:
        worst_group_average_accuracy = ListMean([i[worst_groups[0]] for i in group_test_accuracies_list])
        worst_group_best_accuracy = max([i[worst_groups[0]] for i in group_test_accuracies_list])
        worst_group_worst_accuracy = min([i[worst_groups[0]] for i in group_test_accuracies_list])

    balanced_group_average_accuracy = ListMean([(i[0] + i[1] + i[2] + i[3]) / 4 for i in group_test_accuracies_list])
    balanced_group_best_accuracy = max([(i[0] + i[1] + i[2] + i[3]) / 4 for i in group_test_accuracies_list])
    balanced_group_worst_accuracy = min([(i[0] + i[1] + i[2] + i[3]) / 4 for i in group_test_accuracies_list])

    return worst_group_average_accuracy, worst_group_best_accuracy, worst_group_worst_accuracy, balanced_group_average_accuracy, balanced_group_best_accuracy, balanced_group_worst_accuracy


if __name__ == "__main__":
    opts = parse_args()
    worst_group_average_accuracy, worst_group_best_accuracy, worst_group_worst_accuracy, \
        balanced_group_average_accuracy, balanced_group_best_accuracy, balanced_group_min_accuracy = run(opts)
