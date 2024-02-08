import json

import torch
from torch_geometric.data import Data, Dataset
from transformers import BertTokenizer, BertModel
from tqdm.auto import tqdm


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BERT_TOKENIZER = BertTokenizer.from_pretrained("bert-base-uncased")
BERT_MODEL = BertModel.from_pretrained("bert-base-uncased", output_hidden_states=True).to(DEVICE)
EMOTIONS = ["neutral", "anger", "disgust", "fear", "joy", "sadness", "surprise"]
EMOTIONS_ONE_HOT = torch.zeros(len(EMOTIONS), len(EMOTIONS)).to(DEVICE)
for i, emotion in enumerate(EMOTIONS):
    EMOTIONS_ONE_HOT[i, i] = 1


def read_json(file_name: str) -> list:
    with open(file_name, "r") as f:
        return json.load(f)


def write_json(file_name: str, dataset: list) -> list:
    with open(file_name, "w") as f:
        return json.dump(dataset, f)


def get_cause_relations(dialogs: list) -> None:
    for d in tqdm(dialogs):
        d["cause_relations"] = []
        for target, cause in d["emotion-cause_pairs"]:
            cause_position = int(cause.split("_", maxsplit=1)[0]) - 1
            target_position = int(target.split("_", maxsplit=1)[0]) - 1
            if cause_position <= target_position:
                d["cause_relations"].append((cause_position, target_position))


def get_sentence_embeddings(sentences: list) -> torch.Tensor:
    encoded_input = BERT_TOKENIZER(sentences, padding=True, truncation=True, return_tensors="pt")
    encoded_input = {key: val.to(DEVICE) for key, val in encoded_input.items()}
    with torch.no_grad():
        outputs = BERT_MODEL(**encoded_input)
    embeddings = outputs.hidden_states[-2].mean(dim=1)
    return embeddings


def get_emotions_tensor(emotion_list: list) -> torch.Tensor:
    one_hot_matrix = torch.zeros(len(emotion_list), len(EMOTIONS)).to(DEVICE)
    for i, emotion_label in enumerate(emotion_list):
        one_hot_matrix[i, EMOTIONS.index(emotion_label)] = 1
    return one_hot_matrix


def get_speakers_tensor(speaker_list: list) -> torch.Tensor:
    speaker_to_idx = {speaker: idx for idx, speaker in enumerate(sorted(set(speaker_list)))}
    one_hot_encoded_fixed = torch.zeros((len(speaker_list), 9), dtype=torch.long)
    for i, speaker in enumerate(speaker_list):
        one_hot_encoded_fixed[i, speaker_to_idx[speaker]] = 1
    return one_hot_encoded_fixed


def get_graph(dialog: dict, test: bool = False, window=None) -> Data:
    texts_list = []
    emotions_list = []
    speakers_list = []
    emotional_turns = []

    for i, utterance in enumerate(dialog["conversation"]):
        texts_list.append(utterance["text"])
        emotions_list.append(utterance["emotion"])
        speakers_list.append(utterance["speaker"])
        if utterance["emotion"] != "neutral":
            emotional_turns.append(i)

    length = len(dialog["conversation"])
    x = get_sentence_embeddings(texts_list)
    if not window:
        window = length
    edge_index = (
        torch.tensor(
            [[i, j] for i in range(x.size(0)) for j in emotional_turns if (i <= j) & (j - i <= window)],
            dtype=torch.long,
        )
        .t()
        .contiguous()
    )
    edge_attr = get_speakers_tensor(speakers_list)
    emotions = get_emotions_tensor(emotions_list)
    if not emotional_turns:
        skip = True
    else:
        skip = False
    if not test:
        cause_relations_tensor = torch.tensor(dialog["cause_relations"])
        y = torch.zeros(edge_index.size(1), dtype=torch.int64).to(DEVICE)
        for i, edge in enumerate(edge_index.t()):
            if any(torch.all(edge == special_connection, dim=0) for special_connection in cause_relations_tensor):
                y[i] = 1
        data_entry = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, emotions=emotions, skip=skip).to(DEVICE)
    else:
        data_entry = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, emotions=emotions, skip=skip).to(DEVICE)

    return data_entry


class DialogDataset(Dataset):
    def __init__(self, data_list: list):
        super(DialogDataset, self).__init__()
        self.data_list = data_list

    def len(self):
        return len(self.data_list)

    def get(self, idx: int):
        return self.data_list[idx]


def get_dataset(data: dict, test: bool=False) -> DialogDataset:
    dataset = []
    for dialog in tqdm(data):
        if dialog["conversation"]:
            if test or dialog["emotion-cause_pairs"]:
                dataset.append(get_graph(dialog, window=None, test=test))
    dataset = DialogDataset(dataset)
    return dataset
