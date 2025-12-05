# %%
import os
from datasets import load_dataset


def rename_keys(d: dict, mapping: dict) -> dict:
    "Takes a dictionary and renames the keys according to the mapping dictionary all keys must be present in the mapping dictionary."
    return {mapping[k]: v for k, v in d.items()}


def fix_roles(d: dict) -> dict:
    "Map the role values to the expected format"
    mapping = {"system": "system", "human": "user", "gpt": "assistant"}
    d["role"] = mapping[d["role"]]
    return d


dataset = (
    load_dataset("mlabonne/FineTome-100k")["train"]
    .rename_column("conversations", "messages")
    .remove_columns(column_names=["source", "score"])
)

# Format accepted by mlx_lm for training
mapping = {"from": "role", "value": "content"}
dataset = dataset.map(
    lambda x: {"messages": [fix_roles(rename_keys(m, mapping)) for m in x["messages"]]}
)
# %%
# Split the dataset into train 80%, test 10%, and valid 10%
splits = dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = splits["train"]
tmp_dataset = splits["test"]
tmp_splits = tmp_dataset.train_test_split(test_size=0.5, seed=42)
test_dataset = tmp_splits["test"]
valid_dataset = tmp_splits["train"]
# %%
# Save the datasets to the data directory
# %%
datadir = "./artifacts/data"
os.makedirs(datadir, exist_ok=True)
# save to jsonl files
train_dataset.to_json(
    os.path.join(datadir, "train.jsonl"), lines=True, orient="records"
)
test_dataset.to_json(os.path.join(datadir, "test.jsonl"), lines=True, orient="records")
valid_dataset.to_json(
    os.path.join(datadir, "valid.jsonl"), lines=True, orient="records"
)
