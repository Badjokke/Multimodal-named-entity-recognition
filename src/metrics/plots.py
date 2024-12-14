import matplotlib.pyplot as plt

class SimplePlot:
    def __init__(self, x: list[list[float]], y: list[list[float]], x_axis_label: str=None, y_axis_label: str=None, plot_title: str=None,
                 labels:list[str]=[], colors:list[str]=[], fig_size: (int, int)=()):
        self.x = x
        self.y = y
        self.x_axis_label = x_axis_label
        self.y_axis_label = y_axis_label
        self.plot_title = plot_title
        self.labels = labels
        self.colors = colors
        self.fig_size = fig_size
        self.__default_colors = ["blue","green","red","black"]


    def plot(self) -> None:
        assert len(self.x) == len(self.y), f"x and y must have same length. Received {len(self.x)}, {len(self.y)}"
        plt.xlabel(self.__get_x_axis_label())
        plt.ylabel(self.__get_y_axis_label())
        plt.title(self.__get_plot_title())
        self.__plot_data()
        plt.legend(loc="best")
        plt.show()

    def __plot_data(self):
        for i in range(len(self.x)):
            plt.plot(self.x[i], self.y[i], label=self.__get_line_label(i), color=self.__get_line_color(i))

    def __get_line_label(self, index: int) -> str:
        return f"line_{index+1}" if index >= len(self.labels) else self.labels[index]

    def __get_line_color(self, index: int) -> str:
        return self.__default_colors[index % len(self.__default_colors)] if index >= len(self.colors) else self.colors[index]

    def __get_plot_title(self) -> str:
        return "Very cool plot of very cool things" if self.plot_title is None else self.plot_title

    def __get_x_axis_label(self) -> str:
        return "x-axis" if self.x_axis_label is None else self.x_axis_label

    def __get_y_axis_label(self) -> str:
        return "y-axis" if self.y_axis_label is None else self.y_axis_label

    def __get_fig_size(self)->tuple[int, int]:
        return (800,800) if self.fig_size is not None else self.fig_size