import json


class Experiment:
    def __init__(self, data):
        """
        Initialize an experiment with dataset info and models.
        :param data: Dictionary with 'datasets' and 'models' keys.
        """
        self.datasets = {
            name: info["input_path"] for name, info in data.get("datasets", {}).items()
        }
        self.models = data.get("models")
        self.results_path = data["results_path"]
        self.pipeline = data["pipeline"]

    def get_datasets(self):
        return self.datasets

    def get_pipeline(self):
        return self.pipeline

    def get_results_path(self):
        return self.results_path

    def get_models(self):
        return self.models

    def contains_model(self, model: str) -> bool:
        model = model.upper()
        return model in self.models

    def contains_pipeline(self, pipeline: str) -> bool:
        pipeline = pipeline.upper()
        return pipeline in self.pipeline

    def contains_dataset(self, dataset: str) -> bool:
        dataset = dataset.upper()
        return dataset in self.datasets

class ConfigParser:
    def __init__(self, config_data):
        """
        Load JSON config from a dict or file path, and parse experiments.
        """
        if isinstance(config_data, str):
            with open(config_data, 'r') as f:
                config_data = json.load(f)
        elif not isinstance(config_data, dict):
            raise ValueError("Input must be a dict or a path to a JSON file.")

        self.experiments = [
            Experiment(exp_data) for exp_data in config_data.get("experiments", [])
        ]

    def get_experiments(self):
        return self.experiments
