from copy import deepcopy

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from deeppavlov import build_model
from sklearn.metrics import classification_report
from tqdm.auto import tqdm

from data_processing import read_json, get_dataset
from graph_models import ConversationGAT

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMOTIONS = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]


def get_predictions(model, test_loader, test_data, classification_model):
    pred_pairs = []
    model.eval()
    with torch.no_grad():
        for dialog_id, data in enumerate(tqdm(test_loader)):
            dialog_pred_pairs = []
            data = data.to(DEVICE)

            y_preds = model(data)
            y_preds = torch.argmax(F.softmax(y_preds, dim=1), dim=1)
            y_preds = y_preds.cpu().numpy()

            edges = data.edge_index.cpu().numpy()

            for i in range(edges.shape[1]):
                if y_preds[i] > 0:
                    cause_id, emotion_id = edges[:, i]
                    cause_utterance_text = test_data[dialog_id]["conversation"][cause_id]["text"]
                    emotion_utterance_text = test_data[dialog_id]["conversation"][emotion_id]["text"]
                    if cause_id == emotion_id:
                        emotion_type = classification_model([""], [emotion_utterance_text])[0]
                    else:
                        emotion_type = classification_model([cause_utterance_text], [emotion_utterance_text])[0]
                    dialog_pred_pairs.append((str(cause_id + 1), str(emotion_id + 1), emotion_type))  #
            pred_pairs.append(dialog_pred_pairs)
    return pred_pairs


def get_golds(data):
    gold_pairs = []
    for dialog in data:
        dialog_gold_pairs = []
        for emotion, cause in dialog["emotion-cause_pairs"]:
            emotion_utterance_id, emotion_type = emotion.split("_")
            cause_utterance_id = cause.split("_")[0]
            dialog_gold_pairs.append((cause_utterance_id, emotion_utterance_id, emotion_type))
        gold_pairs.append(dialog_gold_pairs)
    return gold_pairs


def preparee4evaluation(gold_pairs, pred_pairs):
    gold4eval = []
    pred4eval = []
    for gold, pred in zip(gold_pairs, pred_pairs):
        used = []
        intersection = set(gold) & set(pred)
        for gold_pair in gold:
            gold4eval.append(gold_pair[2])
            flag = False
            if gold_pair in list(intersection):
                pred4eval.append(gold_pair[2])
                flag = True
                used.append(pred.index(gold_pair))
            else:
                for emotion in EMOTIONS:
                    if (gold_pair[0], gold_pair[1], emotion) in pred:
                        pred4eval.append(emotion)
                        flag = True
                        used.append(pred.index((gold_pair[0], gold_pair[1], emotion)))
                        break
                if not flag:
                    pred4eval.append("neutral")
        for i, pred_pair in enumerate(pred):
            if i not in used:
                pred4eval.append(pred_pair[2])
                gold4eval.append("neutral")
    return gold4eval, pred4eval


def eval():
    test_data = read_json("data/Subtask_1_gold.json")
    test_dataset = get_dataset(test_data, test=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = ConversationGAT().to(DEVICE)
    model.load_state_dict(torch.load("models/best_model.pth"))
    classification_model = build_model("src/emotion_classifier/glue_emo.json")

    gold4eval, pred4eval = preparee4evaluation(
        get_golds(test_data), get_predictions(model, test_loader, test_data, classification_model)
    )
    print("All 6 emotions metrics:")
    print(classification_report(gold4eval, pred4eval, labels=EMOTIONS, digits=4))

    print("4 main emotions metrics:")
    reduced_emotions_list = deepcopy(EMOTIONS)
    reduced_emotions_list.remove("disgust")
    reduced_emotions_list.remove("fear")
    print(classification_report(gold4eval, pred4eval, labels=reduced_emotions_list, digits=4))


if __name__ == "__main__":
    eval()
