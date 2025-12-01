# %%
import os
import json
from datasets import load_dataset


def mmlu_to_quiz_json(example):
    """Convert MMLU example to quiz JSON format."""
    question_text = example["question"]
    choices = example["choices"]
    correct_idx = example["answer"]

    answers = []
    for idx, choice in enumerate(choices):
        answers.append({"text": choice, "correct": (idx == correct_idx)})

    quiz_obj = {
        "title": "MMLU Quiz",
        "description": "A quiz question from the MMLU dataset",
        "questions": [{"question": question_text, "answers": answers}],
    }

    return json.dumps(quiz_obj, ensure_ascii=False)


def create_training_message(example):
    """Convert MMLU example to Llama3 chat template format."""
    subject = example.get("subject", "General Knowledge")
    user_content = f"Topic: {subject}\n\nGenerate a quiz question."
    quiz_json = mmlu_to_quiz_json(example)

    return {
        "messages": [
            # {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": quiz_json},
        ]
    }


dataset = (
    load_dataset("sbintuitions/MMLU", cache_dir="./artifacts/cache")["test"]
    .map(create_training_message)
    .remove_columns(column_names=["choices", "answer", "qid", "subject", "split", "tag", "description", "question"])
)

# %%
# Split the dataset into train 80%, validation 10%, and test 10%
splits = dataset.train_test_split(test_size=0.2, seed=42)  # 80% train, 20% temp
train_dataset = splits["train"]
temp_dataset = splits["test"]

# Split temp dataset into validation and test (50% each = 10% of total)
val_test_splits = temp_dataset.train_test_split(test_size=0.5, seed=42)
valid_dataset = val_test_splits["train"]
test_dataset = val_test_splits["test"]
# %%
# Save the datasets to the data directory
datadir = "./artifacts/data/mmlu"
os.makedirs(datadir, exist_ok=True)
# save to jsonl files
train_dataset.to_json(
    os.path.join(datadir, "train.jsonl"), lines=True, orient="records"
)
test_dataset.to_json(os.path.join(datadir, "test.jsonl"), lines=True, orient="records")
valid_dataset.to_json(
    os.path.join(datadir, "valid.jsonl"), lines=True, orient="records"
)
