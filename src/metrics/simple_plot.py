import matplotlib.pyplot as plt

"""
uses global object NOT THREAD SAFE
"""
class SimplePlot:
    def __init__(self, x: list[list[float]], y: list[list[float]], x_axis_label: str = None, y_axis_label: str = None,
                 plot_title: str = None,
                 labels: list[str] = [], colors: list[str] = [], fig_size: (int, int) = None):
        self.x = x
        self.y = y
        self.x_axis_label = x_axis_label
        self.y_axis_label = y_axis_label
        self.plot_title = plot_title
        self.labels = labels if len(labels) > 0 else [f"line_{i}" for i in range(len(x))]
        self.colors = colors
        self.fig_size = fig_size
        self.__default_colors = ["blue", "green", "red", "black"]
        self.__y_positions = []
        self.__threshold = 1.0
        self.__y_shift_up = 4
        self.__y_shift_down = 12

    def plot(self):
        assert len(self.x) == len(self.y), f"x and y must have same length. Received {len(self.x)}, {len(self.y)}"
        plt.figure(figsize=self.__get_fig_size(), clear=True)
        plt.xlabel(self.__get_x_axis_label())
        plt.ylabel(self.__get_y_axis_label())
        plt.title(self.__get_plot_title())
        self.__plot_data()
        plt.ylim(ymin=0)
        plt.xlim(xmin=0)
        plt.xticks(self.x[0])
        plt.yticks([i for i in range(0, 105, 5)])
        plt.legend(loc="lower right", labels=self.__get_legend_labels_with_macro_f1_information())
        return self
    def __get_legend_labels_with_macro_f1_information(self):
        legend_labels = []
        for i in range(len(self.y)):
            legend_labels.append(f"{self.labels[i]} macro-f1: {self.y[i][-1]:.2f}")
        return legend_labels
    @staticmethod
    def show():
        plt.show()

    @staticmethod
    def save(path: str) -> None:
        plt.savefig(path, bbox_inches='tight')
        plt.close()

    def __plot_data(self):
        for i in range(len(self.x)):
            plt.plot(self.x[i], self.y[i], label=self.__get_line_label(i), color=self.__get_line_color(i))
        #self.__add_text_value_to_last_points()

    def __add_text_value_to_last_points(self):
        for i in range(len(self.x)):
            last_x = self.x[i][-1]
            last_y = self.y[i][-1]
            last_y_position = self.__get_y_label_relaxed_position(last_y)
            plt.text(last_x, last_y_position, f"{self.labels[i]}: {last_y:.2f}", verticalalignment='bottom', horizontalalignment='right')
            self.__append_y_position_of_label(last_y_position)

    def __get_line_label(self, index: int) -> str:
        return f"line_{index + 1}" if index >= len(self.labels) else self.labels[index]

    def __get_line_color(self, index: int) -> str:
        return self.__default_colors[index % len(self.__default_colors)] if index >= len(self.colors) else self.colors[
            index]

    def __get_plot_title(self) -> str:
        return "Very cool plot of very cool things" if self.plot_title is None else self.plot_title

    def __get_x_axis_label(self) -> str:
        return "x-axis" if self.x_axis_label is None else self.x_axis_label

    def __get_y_axis_label(self) -> str:
        return "y-axis" if self.y_axis_label is None else self.y_axis_label
    """
    creates 500 by 400 figure (in pixels)
    configured in inches
    """
    def __get_fig_size(self) -> tuple[int, int]:
        return (5, 4) if self.fig_size is None else self.fig_size

    def __append_y_position_of_label(self, y):
        self.__y_positions.append(y)

    def __get_y_label_relaxed_position(self, y_curr):
        for y in self.__y_positions:
            if abs(y - y_curr) <= self.__threshold:
                return self.__shift_label_up_on_y_axis(y_curr) if y_curr > y else self.__shift_label_down_on_y_axis(y)
        return y_curr

    def __shift_label_down_on_y_axis(self, y):
        return y - self.__y_shift_down
    def __shift_label_up_on_y_axis(self, y):
        return y + self.__y_shift_up

if __name__ == "__main__":
    epochs = [i for i in range(1,8)]
    y_1 = [65, 68, 74, 75, 74, 78, 79]
    y_2 = [60, 63, 69, 70, 69, 73, 78]
    y_3 = [60, 63, 69, 70, 69, 73, 77]
    p = SimplePlot([epochs]*3, [y_1, y_2, y_3], x_axis_label="Epochs", y_axis_label="Value")
    p.plot().show()
