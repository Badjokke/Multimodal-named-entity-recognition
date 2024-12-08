import torch
import re


def save_model(model:torch.nn.Module, model_path: str):
    torch.save(model.state_dict(), model_path)


def load_and_filter_state_dict(model_path:str)->dict:
    state_dict = torch.load("./combined_model_e1.pth", weights_only=True)
    quantization_keywords = ["quant_map", "nested_absmax", "quant_state", "absmax"]
    filtered_state_dict = {}
    for key, value in state_dict.items():
        ignore = False
        for q_keywords in quantization_keywords:
            if re.match(f".*{q_keywords}.*", key) is not None:
                ignore = True
                break
        if not ignore:
            filtered_state_dict[key] = value
    return filtered_state_dict


def create_message(labels: dict, user_text: str)->list[dict[str,str]]:
    return [
        {
            "role" : "system",
            "content": f"You are a bot tasked with named entity recognition. Use these labels: {labels.keys()} a no else."
        },
        {
            "role": "user",
            "content": user_text
        }
    ]
"""
Text: "oikawa rly said i'll show them what real serves are then BOOM it goes out of bounds"
Task: Perform named entity recognition using only these entities: {'O': 0, 'B-LOC': 1, 'B-PER': 2, 'B-MIS': 3, 'I-PER': 4, 'B-ORG': 5, 'I-LOC': 6, 'I-MIS': 7, 'I-ORG': 8}. Answer in format 'token':'label' on new line for each token.
Answer: 
oikawa: 'B-PER'
rly: 'I-PER'
said: 'O'
i: 'B-PER'
ll: 'O'
show: 'O'
them: 'O'
what: 'O'
real: 'O'
serves: 'O'
are: 'O'
then: 'O'
BOOM: 'O'
it: 'O'
goes: 'O'
out: 'O'
of: 'O'
bounds:
"""
def answer_to_labels(text:str, labels:dict):
    answer = text[text.find("Answer:"):].split("\n")
    y_pred = []
    for i in range(1,len(answer), 1):
        token_prediction = answer[i].split(":")
        if len(token_prediction) != 2:
            print(f"Prediction {token_prediction} in answer {answer} not in valid format.")
            token_prediction.append("O")
        y_pred.append(labels[token_prediction[1].strip()[1:-1]])