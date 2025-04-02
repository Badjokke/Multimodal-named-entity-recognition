from metrics.simple_plot import SimplePlot



class PlotBuilder:
    def __init__(self):
        raise Exception("no instances")

    @staticmethod
    def build_simple_plot(x:list[list[float]], y:list[list[float]], **kwargs) -> SimplePlot:
        """
        kwargs:
            x_axis_label: str = x-axis label
            y_axis_label: str = y-axis label
            plot_title: str = plot title
            labels: list[str] = labels of different lines passed in x matrix
            colors: list[str] = colors of different lines passed in x matrix
            fig_size: tuple = (width, height) of the figure
        """
        return SimplePlot(x, y, **kwargs)
