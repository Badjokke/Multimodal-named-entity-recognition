class ConfusionMatrix:
    def __init__(self, y_pred, y_true, label_count:int, id_to_label:dict[int, str]):
        self.label_count = label_count
        self.id_to_label = id_to_label
        self.y_pred = y_pred
        self.y_true = y_true
        self.matrix = [[0 for __ in range(self.label_count)] for _ in range(self.label_count)]
        self.create()

    def print_matrix(self):
        labels = " ".join([label for label in self.id_to_label.values()])
        rows = ""
        for i in range(self.label_count):
            row_as_str = ", ".join((str(x) for x in self.matrix[i]))
            row = f"{self.id_to_label[i]}\t{row_as_str}\n"
            rows += row
        print(f"\t {labels}\n{rows}")


    def create(self) -> list[list[int]]:
        for i in range(len(self.y_true)):
            for batch in range(len(self.y_true[i])):
                for j in range(len(self.y_true[i][batch])):
                    true_label = self.y_true[i][batch][j]
                    pred_label = self.y_pred[i][batch][j]
                    self.matrix[true_label][pred_label] += 1
        return self.matrix.copy()


    def get_matrix(self) -> list[list[int]]:
        return self.matrix.copy()