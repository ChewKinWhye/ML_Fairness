import pandas as pd
import csv
import matplotlib.pyplot as plt
import ast


if __name__ == "__main__":
    colors = ['b', 'g', 'r', 'c']
    # experiment = "model_size_results"
    experiment = "experiment_increasing_model_size_percy"
    plot = "average"
    data = []
    with open(f"{experiment}.csv") as f:
        read_tsv = csv.reader(f)
        for row in read_tsv:
            data.append(row)

    x = [float(i) for i in data[0][1:]]
    # x = data[0][1:]
    for color, y in zip(colors, data[1:]):
        mode = y[0]
        y_total = [ast.literal_eval(i) for i in y[1:]]
        average_worst_group = [float(i[0]) for i in y_total]
        best_worst_group = [float(i[1]) for i in y_total]
        worst_worst_group = [float(i[2]) for i in y_total]
        average_average_group = [float(i[3]) for i in y_total]
        best_average_group = [float(i[4]) for i in y_total]
        worst_average_group = [float(i[5]) for i in y_total]
        if plot == "average_group":
            plt.plot(x, average_average_group, color, label=mode)
        else:
            plt.plot(x, average_worst_group, color, label=mode)

        # plt.fill_between(x, worst_worst_group, best_worst_group, color=color, alpha=0.2)
    plt.xlabel(experiment)
    plt.ylabel('Worst Group Accuracy')
    plt.title(experiment)

    # show a legend on the plot
    plt.legend()

    # function to show the plot
    plt.show()
