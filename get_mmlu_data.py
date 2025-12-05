# %%
import os
import json
import random
from collections import defaultdict
from datasets import load_dataset, Dataset

def mmlu_item_to_question(example):
    answers = [
        {"text": choice, "correct": (i == example["answer"])}
        for i, choice in enumerate(example["choices"])
    ]
    return {"question": example["question"], "answers": answers}

def build_multi_question_quiz(examples, description):
    return json.dumps(
        {
            "title": "MMLU Quiz",
            "description": description,
            "questions": [mmlu_item_to_question(ex) for ex in examples],
        },
        ensure_ascii=False,
    )

def create_training_message_multi(examples, description):
    """Convert MMLU example to Llama3 chat template format."""
    num_questions = len(examples)
    user_prompt = (
        f"Topic: {description}\n"
        f"Generate a quiz with {num_questions} questions."
    )

    quiz_json = build_multi_question_quiz(examples, description)

    return {
        "messages": [
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": quiz_json},
        ]
    }

def group_by_description(dataset):
    groups = defaultdict(list)
    for ex in dataset:
        desc = ex.get("description", "General Knowledge")
        groups[desc].append(ex)
    return groups

def chunk_group(examples, min_q=1, max_q=5):
    chunks = []
    i = 0
    n_examples = len(examples)
    random.seed(42)   # prevent sequential bias

    while i < n_examples:
        n = random.randint(min_q, max_q)
        chunk = examples[i : i + n]
        if len(chunk) == 0:
            break
        chunks.append(chunk)
        i += n

    return chunks

raw = load_dataset("sbintuitions/MMLU", cache_dir="./artifacts/cache")["test"]

grouped = group_by_description(raw)

training_rows = []
for description, examples in grouped.items():
    chunks = chunk_group(examples, min_q=1, max_q=5)
    for chunk in chunks:
        training_rows.append(create_training_message_multi(chunk, description))
random.shuffle(training_rows)
dataset = Dataset.from_list(training_rows)
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
