from typing import Union
from metrics.bar_plot import PieChart
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

    @staticmethod
    def build_cake_plot(data: list[Union[float,int]], labels: list[str], **kwargs) -> PieChart:
        """
        kwargs:
            x_axis_label: str = x-axis label
            y_axis_label: str = y-axis label
            plot_title: str = plot title
            labels: list[str] = labels of different lines passed in x matrix
            colors: list[str] = colors of different lines passed in x matrix
            fig_size: tuple = (width, height) of the figure
        """
        return PieChart(data, labels,**kwargs)
