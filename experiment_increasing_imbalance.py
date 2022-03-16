from main import run, parse_args
import argparse
import csv

if __name__ == "__main__":
    opts = parse_args()

    experiment_results = []
    imbalance_ratios = [" ", 0.01, 0.02, 0.05, 0.1, 0.2]
    modes = ["standard", "reweight_sampling", "discard", "reweight_loss"]

    for idx, mode in enumerate(modes):
        experiment_results.append([mode])
        for imbalance_ratio in imbalance_ratios[1:]:
            opts.mode = mode
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
