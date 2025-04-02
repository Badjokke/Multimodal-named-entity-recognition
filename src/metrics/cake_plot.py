import matplotlib.pyplot as plt
import numpy as np


class PieChart:
    def __init__(self, data: list[float], labels: list[str]):
        assert len(data) == len(labels), "data and label count must be the same"
        self.data = data
        self.labels = labels

    def plot(self):
        self.__preflight_check()
        plt.pie(np.array(self.data), labels=self.labels, autopct='%1.1f%%')

    def __preflight_check(self):
        numero_sum = 0
        for numero in self.data:
            if 0 <= numero <= 1 and (type(numero) == float or type(numero) == int):
                numero_sum += numero
                continue
            raise ValueError(f"numero {numero} is out of range. Needs to be in [0,1]")
        assert numero_sum == 1, "numero_sum must be equal to 1"
