from functools import reduce

from torch import Tensor

class Metrics:
    def __init__(self, y_pred: list[Tensor], y_true: list[Tensor], label_count:int ):
        self.y_pred = y_pred
        self.y_true = y_true
        self.label_count = label_count

    def __create_matrix(self) -> list[list[int]]:
        return [[0 for __ in range(self.label_count)] for _ in range(self.label_count)]

    def confusion_matrix(self):
        assert len(self.y_pred) == len(self.y_true), "y_pred and y_true must have same amount of samples"
        matrix = self.__create_matrix()
        for i in range(len(self.y_true)):
            for j in range(len(self.y_true[i])):
                true_label = self.y_true[i][j]
                pred_label = self.y_pred[i][j]
                matrix[true_label][pred_label] += 1
        return matrix

    def print_confusion_matrix(self, confusion_matrix:list[list[int]], id_to_labels:dict[int, str]):
        assert id_to_labels is not None, "id_to_labels arg cant be None"
        labels = " ".join([label for label in id_to_labels.values()])
        rows = ""
        for i in range(self.label_count):
            row_as_str = ", ".join((str(x) for x in confusion_matrix[i]))
            row = f"{id_to_labels[i]}\t{row_as_str}\n"
            rows += row
        print(f"\t {labels}\n{rows}")

    def f1(self, confusion_matrix:list[list[int]], id_to_labels:dict[int, str]):
        assert id_to_labels is not None, "id_to_labels arg cant be None"
        #labels = " ".join([label for label in id_to_labels.values()])
        p = self.__precision(confusion_matrix, 0)
        r = self.__recall(confusion_matrix, 0)
        f1 = 2 * p * r / (p + r)
        print(f"\t {p}\t{r}\t{f1}")



    def __precision(self,confusion_matrix: list[list[int]], label: int) -> float:
        tp = confusion_matrix[label][label]
        fp = 0
        for i in range(1, self.label_count):
            fp += confusion_matrix[i][label]
        return tp / (tp + fp)

    def __recall(self,confusion_matrix: list[list[int]], label: int) -> float:
        tp = confusion_matrix[label][label]
        fn = 0
        for i in range(1, self.label_count):
            fn += confusion_matrix[label][i]
        return tp / (tp + fn)