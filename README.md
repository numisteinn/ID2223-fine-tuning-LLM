# ID2223 - Lab 2: Fine tuning a LLM

A student project for the KTH course ID2223/FID3020 HT25 Scalable Machine Learning and Deep Learning created by Númi Steinn Baldursson and Márk Vince Varga, aka group Teki 🐢.

The goal of the project is to fine tune a LLM and create a UI for it.

## Usage

```bash
# Get the data
uv run get_data.py
```

### Set the desired model

```bash
BASE_LLM=meta-llama/Llama-3.2-1B-Instruct 
MODEL_ID=$(echo "${BASE_LLM}" | sed 's/\//-/g' | sed 's/:/-/g')
ARTIFACTS_ROOT="artifacts"
QUANTIZED_DIR="artifacts/models/${MODEL_ID}/quantized"
ADAPTER_DIR="artifacts/adapters/${MODEL_ID}"
FUSED_DIR="artifacts/fused/${MODEL_ID}" 
```

### [Optional] Quantize the chosen model to improve performance

```bash
uv run --env-file .env mlx_lm.convert --hf-path ${BASE_LLM} --mlx-path ${QUANTIZED_DIR} -q
```

### Run training

```bash
uv run --env-file .env mlx_lm.lora \
    $([ -d ${QUANTIZED_DIR} ] && echo "--model ${QUANTIZED_DIR}" || echo "--model ${BASE_LLM}") \
    --data ${ARTIFACTS_ROOT}/data \
    --adapter-path ${ADAPTER_DIR} \
    --train \
    --iters 500 \
    --save-every 50 \
    --batch-size 1 \
    --grad-accumulation-steps 3 \
    --max-seq-length 2048 \
    --num-layers 8 \
    $([ -f ${ADAPTER_DIR}/adapters.safetensors ] && echo "--resume-adapter-file ${ADAPTER_DIR}/adapters.safetensors")
```

### Further fine tune on multiple-choice question/answer dataset
```bash
uv run mlx_lm.lora \
    $([ -d ${QUANTIZED_DIR} ] && echo "--model ${QUANTIZED_DIR}" || echo "--model ${BASE_LLM}") \
    --data ${ARTIFACTS_ROOT}/data/mmlu \
    --adapter-path ${ADAPTER_DIR} \
    --train \
    --iters 500 \
    --save-every 50 \
    --batch-size 1 \
    --grad-accumulation-steps 3 \
    --max-seq-length 2048 \
    --num-layers 8 \
    $([ -f ${ADAPTER_DIR}/adapters.safetensors ] && echo "--resume-adapter-file ${ADAPTER_DIR}/adapters.safetensors")
```


### Export the tuned model


```bash
uv run mlx_lm.fuse \
    $([ -d ${QUANTIZED_DIR} ] && echo "--model ${QUANTIZED_DIR}" || echo "--model ${BASE_LLM}") \
    --save-path ${FUSED_DIR} \
    --adapter-path ${ADAPTER_DIR}
```


### Chat with the fine-tuned model (CLI)

```bash
uv run mlx_lm.chat \
    $([ -d ${QUANTIZED_DIR} ] && echo "--model ${QUANTIZED_DIR}" || echo "--model ${BASE_LLM}") \
    --adapter-path ${ADAPTER_DIR}
```

### Chat with the fine-tuned model (Streamlit UI)

```bash
uv run streamlit run app.py
```

This will launch a web interface at `http://localhost:8501` where you can:
- Chat with the fine-tuned model
- Adjust generation parameters (temperature, max tokens, top_p)
- View chat history
- Clear conversation and start fresh

### Upload the model to HuggingFace
```bash
uv run huggingface-cli upload markvincevarga/mouse ${FUSED_DIR} .
```


### Model Selection and Iteration

We evaluated several models to find the optimal balance between performance and capability:
- **Mistral 7B**: Provided good quality but, both inference and training were too slow for our use case
- **Llama 3.2 1B**: Fast but inconsistent at generating structured JSON output
- **Llama 3.2 3B**: Selected as the optimal middle ground with decent speed and reliable JSON generation

### Two-Phase Training Approach
We implemented a two-phase fine-tuning strategy to build both general instruction-following and specific quiz generation capabilities:

**Phase 1: Broad Instruction Fine-Tuning**
- Used the general-purpose finetome100k dataset to establish strong instruction-following foundations
- Focused on building conversational abilities and general knowledge comprehension

**Phase 2: Structure Polishing**
- Fine-tuned specifically on structured JSON quiz data to perfect output formatting
- Emphasized consistent multiple-choice question generation with proper schema adherence
- Used MMLU dataset to reinforce structured output capabilities

### Model-Centric and Data-Centric Iteration to Improve Model Performance

We employed both model-centric and data-centric approaches to enhance our fine-tuned LLM:

**Model-Centric Methods:**
- Evaluated our model on a validation set to measure performance
- Implemented quantization to improve inference speed and reduce memory usage
- Used LoRA (Low-Rank Adaptation) for efficient parameter fine-tuning

**Data-Centric Methods:**
- Identified and incorporated an additional dataset for structured output quiz specifically for multiple-choice quiz generation
- The dataset was derived from the MMLU dataset 
- Used the MMLU dataset to evaluate the model's ability to generate structured JSON output