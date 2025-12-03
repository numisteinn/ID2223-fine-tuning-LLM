# %%
from mlx_embeddings import load as embed_load, generate as embed_generate
import numpy as np
import json
import mlx.core as mx
from mlx_lm.utils import load
from mlx_lm.generate import generate
from datasets import Dataset

# %%
dataset = []
with open("artifacts/data/train.jsonl", "r") as f:
    for i, line in enumerate(f):
        if i >= 10000:
            break
        dataset.append(json.loads(line))
sampled_dataset = Dataset.from_list(dataset)
sampled_dataset = sampled_dataset.shuffle(seed=49).select(range(20))
# sampled_dataset.to_json("eval-data/sampled_data.jsonl", lines=True, orient="records")
# %%
parsed_dataset = []
for example in sampled_dataset:
    convo = example["messages"]
    prompt = convo[0]["content"]
    response = convo[1]["content"]
    parsed_dataset.append({"prompt": prompt, "response": response})
parsed_dataset = Dataset.from_list(parsed_dataset)
parsed_dataset.to_json(
    "eval-data/sampled_data_valid.jsonl", lines=True, orient="records"
)

# %%
data = list(Dataset.from_json("eval-data/sampled_data_valid.jsonl"))
other_data = list(Dataset.from_json("reworded_valid_data.json"))
for i in range(len(data)):
    data[i]["reworded_prompt"] = other_data[i]["text"]
Dataset.from_list(data).to_json(
    "eval-data/sampled_data_reworded_valid.jsonl", lines=True, orient="records"
)


# %%
def get_model_embedding(model, tokenizer, text: str) -> np.ndarray:
    """Extract embedding from the model's last hidden state."""
    tokens = tokenizer.encode(text)
    inputs = mx.array(tokens)[None, :]
    outputs = model(inputs)
    # Use the mean of the last hidden state as the text embedding
    # Shape: (batch_size, seq_len, hidden_dim) -> (hidden_dim,)
    if hasattr(outputs, "last_hidden_state"):
        last_hidden_state = outputs.last_hidden_state
    else:
        # Fallback to the output itself if no last_hidden_state attribute
        last_hidden_state = outputs

    embedding = mx.mean(last_hidden_state, axis=1).squeeze()

    return np.array(embedding)


emb_model, emb_processor = embed_load("mlx-community/all-MiniLM-L6-v2-4bit")


def embed(text: str):
    out = embed_generate(emb_model, emb_processor, texts=[text])
    return out.text_embeds[0]  # already L2-normalized


def similarity(a, b):
    """Compute cosine similarity between two strings using model embeddings."""
    return float(mx.sum(a * b))


# def similarity(model, tokenizer, a: str, b: str) -> float:
#     embedding_a = get_model_embedding(model, tokenizer, a)
#     embedding_b = get_model_embedding(model, tokenizer, b)
#     dot_product = np.dot(embedding_a, embedding_b)
#     norm_a = np.linalg.norm(embedding_a)
#     norm_b = np.linalg.norm(embedding_b)
#
#     if norm_a == 0 or norm_b == 0:
#         return 0.0
#
#     return dot_product / (norm_a * norm_b)
#
# %%
def evaluate_model(model, tokenizer, dataset):
    out = []
    for example in dataset:
        generated_response = generate(
            model, tokenizer, example["reworded_prompt"], max_tokens=1024
        )
        similarity_score = similarity(
            embed(example["response"]), embed(generated_response)
        )

        out.append(
            {
                "original_prompt": example["prompt"],
                "reworded_prompt": example["reworded_prompt"],
                "response": example["response"],
                "generated_response": generated_response,
                "original_response": example["response"],
                "similarity": similarity_score,
            }
        )
    return out


dataset = Dataset.from_json("eval-data/sampled_data_reworded_valid.jsonl")
base_models = [
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
]

all_results = []
train = "_train1"

for base_model in base_models:
    print(f"\n=== Evaluating {base_model} ===")
    model_id = base_model.replace("/", "-")
    fine_tuned_model = f"artifacts/fused/{model_id}{train}"
    print(f"Loading base model: {base_model}")
    base_result = load(base_model)
    if len(base_result) == 2:
        base_model_loaded, base_tokenizer = base_result
    else:
        base_model_loaded, base_tokenizer, _ = base_result
    print(f"Loading fine-tuned model: {fine_tuned_model}")
    eval_result = load(fine_tuned_model)
    if len(eval_result) == 2:
        eval_model, eval_tokenizer = eval_result
    else:
        eval_model, eval_tokenizer, _ = eval_result

    print("Evaluating base model...")
    base_results = evaluate_model(base_model_loaded, base_tokenizer, dataset)
    print("Evaluating fine-tuned model...")
    fine_tuned_results = evaluate_model(eval_model, eval_tokenizer, dataset)

    # Combine results for comparison
    for i, (base_result, fine_tuned_result) in enumerate(
        zip(base_results, fine_tuned_results)
    ):
        comparison = {
            "model_pair": base_model,
            "example_index": i,
            "prompt": base_result["original_prompt"],
            "reworded_prompt": base_result["reworded_prompt"],
            "expected_response": base_result["response"],
            "base_model_response": base_result["generated_response"],
            "base_model_similarity": base_result["similarity"],
            "fine_tuned_model_response": fine_tuned_result["generated_response"],
            "fine_tuned_model_similarity": fine_tuned_result["similarity"],
            "similarity_delta": fine_tuned_result["similarity"]
            - base_result["similarity"],
        }
        all_results.append(comparison)

# Save raw results
with open("eval-data/human_eval_results_raw.jsonl", "w") as f:
    for result in all_results:
        f.write(json.dumps(result) + "\n")
# %%
# Create human-readable comparison report
with open("eval-data/human_eval_comparison.txt", "w") as f:
    f.write("Human Evaluation Comparison Report\n")
    f.write("=" * 50 + "\n\n")

    for base_model in base_models:
        model_results = [r for r in all_results if r["model_pair"] == base_model]
        if not model_results:
            continue

        f.write(f"Model: {base_model}\n")
        f.write("-" * 30 + "\n")

        base_avg = np.mean([r["base_model_similarity"] for r in model_results])
        fine_tuned_avg = np.mean(
            [r["fine_tuned_model_similarity"] for r in model_results]
        )
        avg_improvement = fine_tuned_avg - base_avg

        f.write(
            f"Average similarity - Base: {base_avg:.4f}, Fine-tuned: {fine_tuned_avg:.4f}, Improvement: {avg_improvement:.4f}\n\n"
        )

        for i, result in enumerate(model_results):
            f.write(f"Example {i + 1}:\n")
            f.write(f"Prompt: {result['prompt']}\n")
            f.write(f"Expected: {result['expected_response'][:100]}\n")
            f.write(
                f"Base Model (similarity: {result['base_model_similarity']:.4f}): {result['base_model_response'][:100]}\n"
            )
            f.write(
                f"Fine-tuned Model (similarity: {result['fine_tuned_model_similarity']:.4f}): {result['fine_tuned_model_response'][:100]}\n"
            )
            f.write(f"Improvement: {result['similarity_delta']:.4f}\n")
            f.write("\n" + "=" * 50 + "\n\n")

print("\nEvaluation complete! Results saved to:")
print("- Raw data: eval-data/human_eval_results_raw.jsonl")
print("- Human-readable report: eval-data/human_eval_comparison.txt")
