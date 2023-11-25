import json
import re
import requests
from copy import deepcopy
from spacy.lang.en import English
import time
from tqdm import tqdm

url = "http://0.0.0.0:8011/sentseg"
BATCH_SIZE = 64

nlp = English()
nlp.add_pipe("sentencizer")

fpath = "data/Subtask_1_train.json"
f = open(fpath)
train = json.load(f)
f.close()

for sample in train:
    conversation_splitted = list()
    for utterance in sample["conversation"]:
        subdata = utterance["text"].strip()
        if subdata:
            subdata = re.sub(r"\s([,\.!\?']+)", r"\1", re.sub(r" ’ ", r"’", subdata))
            subdata = re.sub(r"([,\.!\?]+)", r" ", subdata)
            subdata = re.sub(r"\s+", r" ", subdata).strip()
            conversation_splitted.append(subdata)

    sample["conversation_strip"] = conversation_splitted


def get_batch_punctuated(texts):
    sentences = {"sentences": texts}
    responses = requests.post(url, json=sentences).json()
    result = [response["segments"] for response in responses]
    return result


for sample in tqdm(train):
    sample["conversation_punctuated"] = get_batch_punctuated(sample["conversation_strip"])
    time.sleep(1)


for sample in train:
    dialog_length = len(sample["conversation_strip"])
    segmented_dialog = [segment for utterance in sample["conversation_punctuated"] for segment in utterance]
    sents = [utt["text"] for utt in sample["conversation"]]
    conversation_edu = list()
    if dialog_length < len(segmented_dialog):
        for i, s in enumerate(sents):
            new_sents = []
            new_s = deepcopy(s)
            try:
                while True:
                    comma_id = new_s.index(",")
                    substr = new_s[:comma_id].replace(" ,", "") + "."
                    substr = substr.replace("  ", " ")
                    substr = substr.replace(" .", ".")
                    after_substr = new_s[comma_id + 2 :].replace("  ", " ")
                    after_substr = after_substr.replace(" .", ".")
                    if substr in segmented_dialog or after_substr in segmented_dialog:
                        to_add = new_s[:comma_id] + "."
                        to_add = to_add.replace("  ", " , ")
                        to_add = to_add.replace(" .", " ,")
                        new_sents.append(to_add)
                        new_s = new_s[comma_id + 2 :]
                    else:
                        new_s = new_s[:comma_id] + new_s[comma_id + 1 :]

            except Exception:
                pass

            to_add = new_s.replace("  ", " , ")
            new_sents.append(to_add)
            for sent in new_sents:
                conversation_edu.append(
                    {
                        "utterance_ID": sample["conversation"][i]["utterance_ID"],
                        "text": sent,
                        "speaker": sample["conversation"][i]["speaker"],
                        "emotion": sample["conversation"][i]["emotion"],
                    }
                )

    sample["conversation_edu"] = conversation_edu


with open("train_with_edus.json", "w") as f:
    json.dump(train, f)
