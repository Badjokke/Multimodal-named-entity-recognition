import torch
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TrainingUtil:
    base_label_mapping = { "O": 0, "LOC": 1, "PER": 2, "MIS": 3, "ORG": 4 }

    @staticmethod
    def map_to_base_labels(y, labels_mapping):
        base = []
        for batch in range(len(y)):
            base_batch = []
            for label_id in y[batch]:
                label = labels_mapping[label_id]
                if label != "O":
                    label = label.split("-")[1]
                base_batch.append(TrainingUtil.base_label_mapping[label])
            base.append(base_batch)
        return base

    @staticmethod
    def contains_named_entity(y, other_label_id: int):
        return any(y) != other_label_id

    @staticmethod
    def compute_class_weights(labels):
        counts = torch.bincount(labels.flatten())
        total = counts.sum()
        # More aggressive weighting for minority classes
        weights = (total / counts) ** 1.5  # Exponential scaling
        return weights / weights.sum()

    @staticmethod
    def compute_inverse_freq_weights(y, alpha=0.5) -> torch.Tensor:
        weights = [1 / (np.bincount(x) ** alpha) for x in y]
        return torch.Tensor(weights).to(torch.float32).to(device)

    @staticmethod
    def compute_class_weights_rare_events(y) -> torch.Tensor:
        weights = compute_class_weight(class_weight="balanced", classes=np.unique(y), y=y)
        return torch.from_numpy(weights).to(torch.float32).to(device)

    @staticmethod
    def align_labels(word_ids, labels):
        padded_labels = []
        for word in word_ids:
            padded_labels.append(-100 if word is None else labels[word])
        return padded_labels

    @staticmethod
    def most_common(lst):
        return max(set(lst), key=lst.count)

    @staticmethod
    def decode_labels_majority_vote(word_ids, y_pred):
        y_predicted = y_pred
        batched_decoded_labels = []
        for batch in range(len(y_predicted)):
            current_word_labels = []
            decoded_labels = []
            current_word_id = word_ids[0]
            for i in range(len(word_ids)):
                w_id = word_ids[i]
                if w_id is None:
                    continue
                if current_word_id != w_id:
                    decoded_labels.append(TrainingUtil.most_common(current_word_labels))
                    current_word_labels = []
                    current_word_id = w_id
                current_word_labels.append(y_pred[batch][i])
            decoded_labels.append(TrainingUtil.most_common(current_word_labels))
            batched_decoded_labels.append(torch.tensor(decoded_labels, dtype=torch.long, device=device))
        return torch.stack(batched_decoded_labels)