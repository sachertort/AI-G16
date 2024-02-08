import string

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from tqdm.auto import tqdm

from data_processing import read_json, write_json, get_dataset
from graph_models import CauseExtractor

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_span_position(span: str, utterance: str) -> list:
    """
    Get the position of a span within an utterance.

    Args:
        span (str): The span to search for.
        utterance (str): The utterance to search within.

    Returns:
        list: A list containing the begin and end positions of the span within the utterance.
              The positions are zero-based and the end position is exclusive.
    """
    begin_id, end_id = 0, 0
    cause_token = span.split()
    utterance_token = utterance.split()
    for wi in range(len(utterance_token)):
        if (wi+len(cause_token))<=len(utterance_token) and utterance_token[wi:wi+len(cause_token)] == cause_token:
            begin_id = wi
            end_id = wi+len(cause_token)
            break
    return [begin_id, end_id] # start from 0, [begin_id, end_id)

def clean_span(span: str) -> str:
    """
    Cleans the given span by removing leading and trailing punctuation characters.

    Args:
        span (str): The span to be cleaned.

    Returns:
        str: The cleaned span.
    """
    while 1:
        span = span.strip()
        if span[0] not in string.punctuation and span[-1] not in string.punctuation:
            break
        else:
            if span[0] in string.punctuation:
                span = span[1:]
            if span[-1] in string.punctuation:
                span = span[:-1]
    return span

def get_predictions(model: CauseExtractor, test_loader: DataLoader) -> list:
    """
    Generate predictions for a given model on test data.

    Args:
        model (torch.nn.Module): The trained model.
        test_loader (torch.utils.data.DataLoader): The data loader for the test data.

    Returns:
        pred_pairs (list): A list of predicted pairs for each dialog in the test data.
    """
    pred_pairs = []
    model.eval()
    with torch.no_grad():
        for data in tqdm(test_loader):
            if data.skip:
                pred_pairs.append([])
            else:
                dialog_pred_pairs = []
                data = data.to(DEVICE)

                y_preds = model(data)
                y_preds = torch.argmax(F.softmax(y_preds, dim=1), dim=1)
                y_preds = y_preds.cpu().numpy()

                edges = data.edge_index.cpu().numpy()

                for i in range(edges.shape[1]):
                    if y_preds[i] > 0:
                        cause_id, emotion_id = edges[:, i]
                        dialog_pred_pairs.append((cause_id, emotion_id))
                pred_pairs.append(dialog_pred_pairs)
    return pred_pairs

def main() -> None:
    """
    Main function that performs the prediction task.

    Reads the test data from a JSON file, creates a test dataset, loads the trained model,
    generates predictions for the test data, and updates the test data with the predicted
    emotion-cause pairs. Finally, writes the updated test data to a JSON file.

    Args:
        None

    Returns:
        None
    """
    test_data = read_json("data/Subtask_1_test_gpt.json")
    test_dataset = get_dataset(test_data, test=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    model = CauseExtractor().to(DEVICE)
    model.load_state_dict(torch.load("models/best_model.pth"))
    causes_emotions = get_predictions(model, test_loader)

    for i, dialog in enumerate(tqdm(test_data)):
        dialog["emotion-cause_pairs"] = []
        for cause, emotion in causes_emotions[i]:
            emotion_utterance_id = dialog["conversation"][emotion]["utterance_ID"]
            cause_utterance_text = dialog["conversation"][cause]["text"]
            cause_utterance_id = dialog["conversation"][cause]["utterance_ID"]
            cause_span = get_span_position(clean_span(cause_utterance_text), cause_utterance_text)
            emotion_type = dialog["conversation"][emotion_utterance_id - 1]["emotion"]
            dialog["emotion-cause_pairs"].append([f"{emotion_utterance_id}_{emotion_type}", f"{cause_utterance_id}_{cause_span[0]}_{cause_span[1]}"])

    write_json("data/Subtask_1_pred.json", test_data)

if __name__ == "__main__":
    main()
