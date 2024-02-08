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
    """
    Reads a JSON file and returns its contents as a list.
    
    Parameters:
        file_name (str): The name of the JSON file to read.
        
    Returns:
        list: The contents of the JSON file as a list.
    """
    with open(file_name, "r") as f:
        return json.load(f)


def write_json(file_name: str, dataset: list) -> list:
    """
    Write a list of dictionaries to a JSON file.

    Args:
        file_name (str): The name of the JSON file to write.
        dataset (list): The list of dictionaries to write to the JSON file.

    Returns:
        list: The updated dataset.

    """
    with open(file_name, "w") as f:
        return json.dump(dataset, f)


def get_cause_relations(dialogs: list) -> None:
    """
    Adds cause relations to each dialog in the given list.

    Args:
        dialogs (list): A list of dialogs.

    Returns:
        None
    """
    for d in tqdm(dialogs):
        d["cause_relations"] = []
        for target, cause in d["emotion-cause_pairs"]:
            cause_position = int(cause.split("_", maxsplit=1)[0]) - 1
            target_position = int(target.split("_", maxsplit=1)[0]) - 1
            if cause_position <= target_position:
                d["cause_relations"].append((cause_position, target_position))


def get_sentence_embeddings(sentences: list) -> torch.Tensor:
    """
    Calculates the sentence embeddings for a list of sentences using a pre-trained BERT model.

    Args:
        sentences (list): A list of sentences.

    Returns:
        torch.Tensor: The sentence embeddings as a tensor.
    """
    encoded_input = BERT_TOKENIZER(sentences, padding=True, truncation=True, return_tensors="pt")
    encoded_input = {key: val.to(DEVICE) for key, val in encoded_input.items()}
    with torch.no_grad():
        outputs = BERT_MODEL(**encoded_input)
    embeddings = outputs.hidden_states[-2].mean(dim=1)
    return embeddings


def get_emotions_tensor(emotion_list: list) -> torch.Tensor:
    """
    Converts a list of emotion labels into a one-hot encoded tensor.

    Args:
        emotion_list (list): A list of emotion labels.

    Returns:
        torch.Tensor: A one-hot encoded tensor representing the emotions.
    """
    one_hot_matrix = torch.zeros(len(emotion_list), len(EMOTIONS)).to(DEVICE)
    for i, emotion_label in enumerate(emotion_list):
        one_hot_matrix[i, EMOTIONS.index(emotion_label)] = 1
    return one_hot_matrix


def get_speakers_tensor(speaker_list: list) -> torch.Tensor:
    """
    Converts a list of speakers into a one-hot encoded tensor.

    Args:
        speaker_list (list): A list of speakers.

    Returns:
        torch.Tensor: A one-hot encoded tensor representing the speakers.
    """
    speaker_to_idx = {speaker: idx for idx, speaker in enumerate(sorted(set(speaker_list)))}
    one_hot_encoded_fixed = torch.zeros((len(speaker_list), 9), dtype=torch.long)
    for i, speaker in enumerate(speaker_list):
        one_hot_encoded_fixed[i, speaker_to_idx[speaker]] = 1
    return one_hot_encoded_fixed


def get_graph(dialog: dict, test: bool = False, window=None) -> Data:
    """
    Converts a dialog dictionary into a graph representation.

    Args:
        dialog (dict): The dialog dictionary containing conversation information.
        test (bool, optional): Flag indicating whether the graph is for testing purposes. Defaults to False.
        window (int, optional): The maximum window size for connecting emotional turns. Defaults to None.

    Returns:
        Data: The graph representation of the dialog.

    """
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
        """
        Initializes a DialogDataset object.

        Args:
            data_list (list): A list containing the data for the dataset.
        """
        super(DialogDataset, self).__init__()
        self.data_list = data_list

    def len(self):
        """
        Returns the length of the dataset.

        Returns:
            int: The length of the dataset.
        """
        return len(self.data_list)

    def get(self, idx: int):
        """
        Returns the data at the specified index.

        Args:
            idx (int): The index of the data to retrieve.

        Returns:
            Any: The data at the specified index.
        """
        return self.data_list[idx]


def get_dataset(data: dict, test: bool=False) -> DialogDataset:
    """
    Create a DialogDataset from the given data.

    Args:
        data (dict): The input data containing dialog information.
        test (bool, optional): Flag indicating whether the dataset is for testing. Defaults to False.

    Returns:
        DialogDataset: The created DialogDataset object.
    """
    dataset = []
    for dialog in tqdm(data):
        if dialog["conversation"]:
            if test or dialog["emotion-cause_pairs"]:
                dataset.append(get_graph(dialog, window=None, test=test))
    dataset = DialogDataset(dataset)
    return dataset
