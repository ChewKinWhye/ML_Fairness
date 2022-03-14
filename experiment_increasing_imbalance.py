from main import run
import argparse
import csv

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_repeat', help='Number of times to repeat experiment', type=int, default=4)
    parser.add_argument('--dataset_name', help='Which dataset to use, MNIST or FashionMNIST', type=str, default="FashionMNIST")
    # "2,4" for FashionMNIST
    parser.add_argument('--classes', help='Which classes to use', type=str, default="2,4")
    parser.add_argument('--color_intensity', help='Intensity of color added, 0-255', type=int, default=150)
    parser.add_argument('--imbalance_ratio',
                        help='ratio of class, class_0_color_0, class_0_color_1, class_1_color_0, class__color_1',
                        type=str, default="0.48,0.02,0.02,0.48")
    parser.add_argument('--to_save_dataset', help='Whether to save the modified dataset', type=str, default="false")
    parser.add_argument('--validation_split', help='Whether to split train into train and val', type=str, default="true")
    parser.add_argument('--learning_rate', help='Learning rate for model', type=float, default=1e-3)
    parser.add_argument('--training_epochs', help='Number of training epochs', type=int, default=100)
    parser.add_argument('--batch_size', help='Training batch size', type=int, default=8)
    parser.add_argument('--weight_decay', help='Weight decay to use', type=float, default=1e-5)
    parser.add_argument('--CNN_channels', help='Number of channels for CNN first layer', type=int, default=16)
    parser.add_argument('--colors', help="What colors to color the image", type=str, default="blue,green")
    parser.add_argument('--mode',
                        help='What mode to train model, can be [standard, reweight_sampling, discard, reweight_loss]',
                        type=str, default="standard")
    opts = parser.parse_args()
    opts.to_save_dataset = (opts.to_save_dataset == 'true')
    opts.validation_split = (opts.validation_split == 'true')
    opts.classes = tuple([int(x) for x in opts.classes.split(",")])
    opts.imbalance_ratio = tuple([float(x) for x in opts.imbalance_ratio.split(",")])
    imbalance_ratios = [" ", 0.02, 0.05, 0.1, 0.2]
    experiment_results = []
    opts.CNN_channels = 4

    modes = ["standard", "reweight_sampling", "discard", "reweight_loss"]

    for idx, mode in enumerate(modes):
        experiment_results.append([mode])
        for imbalance_ratio in imbalance_ratios[1:]:
            opts.imbalance_ratio = (0.5-imbalance_ratio, imbalance_ratio, imbalance_ratio, 0.5-imbalance_ratio)
            result = run(opts)
            print(result)
            experiment_results[idx].append(result)
    print(imbalance_ratios)
    print(experiment_results)
    with open('imbalance_ratio_results.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(imbalance_ratios)
        for experiment_result in experiment_results:
            writer.writerow(experiment_result)
