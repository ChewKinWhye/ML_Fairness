from main import run, parse_args
import argparse
import csv

if __name__ == "__main__":
    opts = parse_args()
    experiment_results = []
    CNN_width_experiments = [" ", 2, 4, 8, 16, 32, 64]
    modes = ["standard", "reweight_sampling", "discard", "reweight_loss"]
    imbalance_ratio = 0.01

    for idx, mode in enumerate(modes):
        experiment_results.append([mode])
        for CNN_channels in CNN_width_experiments[1:]:
            opts.mode = mode
            opts.CNN_channels = CNN_channels
            opts.imbalance_ratio = ((1-imbalance_ratio)/3, (1-imbalance_ratio)/3, (1-imbalance_ratio)/3, imbalance_ratio)
            opts.worst_group = "3"
            result = run(opts)
            print(result)
            experiment_results[idx].append(result)
    print(CNN_width_experiments)
    print(experiment_results)
    with open(opts.results_file, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(CNN_width_experiments)
        for experiment_result in experiment_results:
            writer.writerow(experiment_result)
