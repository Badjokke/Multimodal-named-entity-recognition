from torch import Tensor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class Metrics:
    def __init__(self, y_pred: Tensor, y_true: Tensor):
        self.y_pred = y_pred
        self.y_true = y_true

    def f1(self):
        f1 = f1_score(self.y_true, self.y_pred, average='macro')
        return f1

    def acc(self):
        acc = accuracy_score(self.y_true, self.y_pred)
        return acc

    def confusion_matrix(self):
        pass