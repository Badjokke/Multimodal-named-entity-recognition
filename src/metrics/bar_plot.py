import matplotlib.pyplot as plt
import numpy as np


class PieChart:
    def __init__(self, data: list[float], labels: list[str], x_label: str = "class label", y_label: str= "class occurrence", plot_label: str = "class distribution"):
        assert len(data) == len(labels), "data and label count must be the same"
        self.data = data
        self.labels = labels
        self.x_label = x_label
        self.y_label = y_label
        self.plot_label = plot_label

    def plot(self):
        fig, ax = plt.subplots(layout='constrained')
        width = 0.25
        for i in range(len(self.data)):
            offset = width * i
            rects = ax.bar(i + offset, self.data[i], width)
            ax.bar_label(rects, padding=3)
        ax.set_xticks([i+width*i for i in range(len(self.labels))], self.labels)

        ax.set_xlabel(self.x_label)
        ax.set_ylabel(self.y_label)
        ax.set_title(self.plot_label)
        plt.show()

    def save(self, path):
        plt.savefig(path, bbox_inches='tight')
