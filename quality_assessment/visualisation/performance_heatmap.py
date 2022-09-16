import numpy as np


def visualise_heatmap(performance_data):
    pass


if __name__ == "__main__":

    algorithms = ["algo1", "algo2", "algo3", "algo4"]
    measures = ["M1", "M2", "M3", "M4", "M5", "M6"]

    performance_dict = dict()
    for algorithm in algorithms:
        performance_dict[algorithm] = {}
        for measure in measures:
            performance_dict[algorithm][measure] = np.random.random()

    visualise_heatmap(performance_dict)
