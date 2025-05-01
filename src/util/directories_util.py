import os



class DirectoryUtil:
    def __init__(self, base_log_path, data_sets: set[str], models: set[str], pipeline: set[str]):
        self.base_path = base_log_path
        self.lstm_path = f"{base_log_path}/lstm"
        self.bert_path = f"{base_log_path}/bert"
        self.llama_path = f"{base_log_path}/llama"
        self.cnn_path = f"{base_log_path}/vit"
        self.vit_path = f"{base_log_path}/alex"
        self.data_sets = data_sets
        self.models = models
        self.pipeline = pipeline

    def create_directories_for_result_logs(self):
        model_paths = [self.lstm_path, self.bert_path, self.llama_path, self.cnn_path, self.vit_path]
        self.__create_t17_dataset_directories(model_paths)
        self.__create_t15_dataset_directories(model_paths)
        self.__create_soa_dataset_directories(model_paths)

    def __create_t17_dataset_directories(self,base_model_paths: list[str]):
        self.__create_dataset_directories("t17", base_model_paths)

    def __create_t15_dataset_directories(self, base_model_paths: list[str]):
        self.__create_dataset_directories("t15", base_model_paths)

    def __create_soa_dataset_directories(self, base_model_paths: list[str]):
        self.__create_dataset_directories("soa", base_model_paths)

    def __create_dataset_directories(self, dataset_dir: str, base_model_paths: list[str]):
        if dataset_dir.upper() not in self.data_sets:
            return
        for base_path in base_model_paths:
            prefix = f"{base_path}/{dataset_dir}"
            multimodal_prefix = f"{prefix}/multimodal"
            text_prefix = f"{prefix}/text"
            image_prefix = f"{prefix}/image"
            if "MULTIMODAL" in self.pipeline:
                DirectoryUtil.__mkdirs_no_err(f"{multimodal_prefix}/fig")
                DirectoryUtil.__mkdirs_no_err(f"{multimodal_prefix}/state_dict")
            if "TEXT" in self.pipeline:
                DirectoryUtil.__mkdirs_no_err(f"{text_prefix}/fig")
                DirectoryUtil.__mkdirs_no_err(f"{text_prefix}/state_dict")
            if "IMAGE" in self.pipeline:
                DirectoryUtil.__mkdirs_no_err(f"{image_prefix}/fig")
                DirectoryUtil.__mkdirs_no_err(f"{image_prefix}/state_dict")

    @staticmethod
    def __mkdirs_no_err(path: str):
        os.makedirs(path, exist_ok=True)

    @staticmethod
    def create_preprocessed_dataset_directories(root_path: str):
        DirectoryUtil.__mkdirs_no_err(f"{root_path}/text_preprocessed")
        DirectoryUtil.__mkdirs_no_err(f"{root_path}/image_preprocessed")
