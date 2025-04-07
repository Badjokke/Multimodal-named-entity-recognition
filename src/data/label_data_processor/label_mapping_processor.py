from concurrent.futures import ThreadPoolExecutor, Future

from data.data_processor import DataProcessor

class LabelMappingProcessor(DataProcessor):
    def __init__(self, label_map: dict[str, str]):
        super().__init__()
        self.io_pool_exec = ThreadPoolExecutor(max_workers=5)
        self.label_map = label_map

    def process_data(self, data: list[str]) -> Future[list[str]]:
        assert data is not None, "Data cannot be None"
        return self.io_pool_exec.submit(self.__map_labels, data)

    def __map_labels(self, labels: list[str]):
        return list(map(lambda lbl: lbl if lbl not in self.label_map else self.label_map[lbl], labels))
