from closedai.client.open_api_client import OpenAIClient
from closedai.util import Util
from metrics.metrics import Metrics

class OpenAIEval:
    def __init__(self, client: OpenAIClient, labels: dict[str, int]):
        self.client = client
        self.labels = labels.keys()
        self.label_map = labels

    def call_api(self, data):
        user_prompt = Util.create_user_prompt(data)
        response = self.client.call_chat_completion_api(Util.create_system_prompt(self.labels, len(data[0])), user_prompt)
        return Util.parse_json(response)


    def eval_mner(self, eval_set: list[tuple[list[str], list[str], list[int]]]):
        y_true = []
        y_predicted = []
        invalid_res_counter = 0
        for i in range(len(eval_set)):
            json_res = self.call_api(eval_set[i])
            if json_res is None:
                invalid_res_counter += 1
                continue
            try:
                y_pred = Util.json_response_to_labels(json_res["entities"], self.label_map)
                if len(y_pred) < len(eval_set[i][2]):
                    y_pred = Util.pad_gpt_response_to_target({"labels":eval_set[i][2], "text":eval_set[i][0]}, {"labels":y_pred, "text": eval_set[i][0]}, self.label_map["O"])
                if len(y_pred) > len(eval_set[i][2]):
                    y_pred = y_pred[:len(eval_set[i][2])]
                y_predicted.append([y_pred])
                y_true.append([eval_set[i][2]])
            except KeyError as e:
                print(f"Key error {i}. Skipping")
        metrics = Metrics(y_predicted, y_true, len(self.label_map.keys()), {value:key for (key,value) in self.label_map.items()})
        macro_f1 = metrics.macro_f1(metrics.confusion_matrix())
        micro_f1 = metrics.micro_f1(metrics.confusion_matrix())
        acc = metrics.accuracy()
        print(f"macro f1: {macro_f1}, micro f1: {micro_f1}, acc: {acc}")