import json
from typing import Union


class Util:
    @staticmethod
    def create_system_prompt(labels: list[str], entity_count: int) -> dict[str, str]:
        return {"role": "developer",
                "content": f"Perform Multimodal Named Entity Recognition and return only these classes.: {','.join(labels)} . Assign label to every word in the sentence. Use json format with 'entities' key for predicted classes."}

    @staticmethod
    def create_image_content(image_base64: str):
        return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}

    @staticmethod
    def create_user_prompt(data_row: tuple[list[str], list[str], list[int]]) -> dict[str, Union[str, list[str]]]:
        return {"role": "user", "content": Util._create_content(data_row[0], data_row[1])}

    @staticmethod
    def create_user_prompt_batch(data_row: list[tuple[list[str], list[str], list[int]]]):
        content = []
        for i in range(len(data_row)):
            single_layer_content = Util._create_content(data_row[i][0], data_row[i][1])
            # list compression for some reason flattens to generators
            for tmp in range(len(single_layer_content)):
                content.append(single_layer_content[tmp])
        return {"role": "user", "content": content}

    @staticmethod
    def _create_content(words: list[str], images: list[str]):
        content = []
        sentence = " ".join(words)
        content.append({"type": "text", "text": sentence})
        for image in images:
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}"}})
        return content

    @staticmethod
    def parse_json(text: str):
        return json.loads(text)

    @staticmethod
    def json_response_to_labels(content: list[dict[str, str]], label_map: dict[str, int]):
        y_pred = []
        for c in content:
            y_pred.append(label_map[c["label"]])
        return y_pred

    @staticmethod
    def pad_gpt_response_to_target(y_true:dict[str, Union[list[int],list[str]]], y_pred:dict[str, Union[list[int],list[str]]], pad_label: int):
        words = set(y_true["sentence"])
        labels = y_true["label"]
        pred_words = set(y_pred["sentence"])
        pred_labels = y_pred["label"]

        diff = words.difference(pred_words)
        for i in range(len(diff)):
            pred_labels.append(pad_label)
        return pred_labels


    @staticmethod
    def truncate_gpt_response_to_target(y_true: dict[str, Union[list[int],list[str]]], y_pred:dict[str, Union[list[int],list[str]]]):
        pass


