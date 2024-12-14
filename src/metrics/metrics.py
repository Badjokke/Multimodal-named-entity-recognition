from torch import Tensor
import functools
from .confusion_matrix import ConfusionMatrix


class Metrics:
    def __init__(self, y_pred: list[Tensor], y_true: list[Tensor], label_count: int, id_to_label: dict[int, str]):
        self.y_pred = y_pred
        self.y_true = y_true
        self.label_count = label_count
        self.id_to_label = id_to_label

    def confusion_matrix(self, ) -> ConfusionMatrix:
        assert len(self.y_pred) == len(self.y_true), "y_pred and y_true must have same amount of samples"
        return ConfusionMatrix(self.y_pred, self.y_true, self.label_count, self.id_to_label)

    def f1(self, confusion_matrix: ConfusionMatrix, label: int) -> dict[str, float]:
        assert 0 <= label < self.label_count, f"Label must be between 0 and {self.label_count}"
        matrix = confusion_matrix.get_matrix()
        p = self.__precision(matrix, label)
        r = self.__recall(matrix, label)
        f1 = 2 * p * r / (p + r)
        return {'f1': f1, 'precision': p, 'recall': r}

    def macro_f1(self, confusion_matrix: ConfusionMatrix) -> float:
        n = self.label_count
        f1n = []
        for i in range(n):
            f1n.append(self.f1(confusion_matrix, i))
        print(f1n)
        f1_sum = functools.reduce(lambda x, y: x + y, map(lambda x: x['f1'], f1n))
        return f1_sum  / n

    def __precision(self, confusion_matrix: list[list[int]], label: int) -> float:
        tp = confusion_matrix[label][label]
        fp = 0
        for i in range(0, self.label_count):
            fp += confusion_matrix[i][label]
        fp -= tp
        return tp / (tp + fp)

    def __recall(self, confusion_matrix: list[list[int]], label: int) -> float:
        tp = confusion_matrix[label][label]
        fn = 0
        for i in range(0, self.label_count):
            fn += confusion_matrix[label][i]
        fn -= tp
        return tp / (tp + fn)
