import pandas as pd
import csv
import matplotlib.pyplot as plt

if __name__ == "__main__":
    experiment = "model_size_results"
    data = []
    with open(f"{experiment}.csv") as f:
        read_tsv = csv.reader(f, delimiter="\t")
        for row in read_tsv:
            data.append(row)

    x = data[0][1:]
    for y in data[1:]:
        label = y[0]
        data = [float(i) for i in y[1:]]
        plt.plot(x, data, label=label)
    plt.xlabel('Model Width')
    plt.ylabel('Worst Group Accuracy')
    plt.title(experiment)

    # show a legend on the plot
    plt.legend()

    # function to show the plot
    plt.show()
