import json
import random
from collections import Counter


def read_json(file_name: str) -> list:
    """
    Read a JSON file and return its contents as a list.

    Args:
        file_name (str): The name of the JSON file to read.

    Returns:
        list: The contents of the JSON file as a list.
    """
    with open(file_name, "r") as f:
        return json.load(f)

def write_jsonl(file_name: str, data: list) -> None:
    """
    Write a list of dictionaries to a JSONL file.

    Args:
        file_name (str): The name of the JSONL file to write.
        data (list): The list of dictionaries to write to the file.

    Returns:
        None
    """
    with open(file_name, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

def prepare4finetuning(data: list) -> list:
    """
    Prepares the data for fine-tuning by extracting conversation turns, previous turn text, target turn text, and emotion.

    Args:
        data (list): List of dictionaries representing the data.

    Returns:
        list: List of dictionaries containing the extracted information for each turn.
    """
    conversations = [dialog["conversation"] for dialog in data]

    for_prompts = []
    for conversation in conversations:
        for i, turn in enumerate(conversation):
            turn_dict = dict()
            if i == 0:
                turn_dict["previous"] = ""
            else:
                turn_dict["previous"] = conversation[i - 1]["text"]
            turn_dict["target"] = turn["text"]
            turn_dict["emotion"] = turn["emotion"]
            for_prompts.append(turn_dict)

    return for_prompts

def balancing(data: list) -> list:
    """
    Balances the given data by oversampling the non-neutral entries and undersampling the neutral entries.
    
    Args:
        data (list): The input data to be balanced.
        
    Returns:
        list: The balanced data.
    """
    emotions_counter = Counter([turn["emotion"] for turn in data if turn["emotion"] != "neutral"])
    max_count = max(emotions_counter.values())
    neutral_entries = [turn for turn in data if turn["emotion"] == "neutral"]
    non_neutral_entries = [turn for turn in data if turn["emotion"] != "neutral"]
    neutral_entries = neutral_entries[:max_count]
    new_data = non_neutral_entries + neutral_entries
    random.shuffle(new_data)
    return new_data

def prompts_construction(for_prompts: list) -> list:
    """
    Constructs prompts for fine-tuning data based on the given list of turns.

    Args:
        for_prompts (list): A list of turns, where each turn is a dictionary with "previous" and "target" keys.

    Returns:
        list: A list of examples, where each example is a dictionary with "messages" key. The "messages" key contains a list of dictionaries representing the system and assistant messages.
    """
    with open("data/prompt_main.txt", "r") as f:
        common_prompt = f.read()

    data4file = []
    for turn in for_prompts:
        prompt = common_prompt.replace("UTT_1", turn["previous"])
        prompt = prompt.replace("UTT_2", turn["target"])
        example = {"messages": [{"role": "system", "content": prompt}, 
                                {"role": "assistant", "content": turn["emotion"]}]}
        data4file.append(example)

    return data4file

def main():
    """
    Main function for processing data for fine-tuning.

    Reads JSON files, prepares data for fine-tuning, constructs prompts,
    and writes the prompts to JSONL files based on the file name.

    Args:
        None

    Returns:
        None
    """
    files = ["Subtask_1_train_real.json", "Subtask_1_dev.json"]
    for file in files:
        data = read_json(f"data/{file}")
        for_prompts = prepare4finetuning(data)
        if "train" in file:
            for_prompts = balancing(for_prompts)
            prompts = prompts_construction(for_prompts)
            write_jsonl("data/train_main_2.jsonl", prompts)
        else:
            prompts = prompts_construction(for_prompts)
            write_jsonl("data/test_main_2.jsonl", prompts)

if __name__ == "__main__":
    main()