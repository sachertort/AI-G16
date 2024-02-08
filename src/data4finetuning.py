import json


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
    files = ["Subtask_1_train.json", "Subtask_1_dev.json"]
    for file in files:
        data = read_json(f"data/{file}")
        for_prompts = prepare4finetuning(data)
        prompts = prompts_construction(for_prompts)
        if "train" in file:
            write_jsonl("data/train_main_2.jsonl", prompts)
        else:
            write_jsonl("data/test_main_2.jsonl", prompts)

if __name__ == "__main__":
    main()