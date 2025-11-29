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
BASE_LLM=Qwen/Qwen2.5-3B-Instruct
```

### [Optional] Quantize the chosen model to improve performance

```bash
uv run mlx_lm.convert --hf-path ${BASE_LLM} --mlx-path artifacts/quantized-model -q
```

### Run training

```bash
uv run mlx_lm.lora \
    $([ -d artifacts/quantized-model ] && echo "--model artifacts/quantized-model" || echo "--model ${BASE_LLM}") \
    --data artifacts/data \
    --adapter-path artifacts/adapter \
    --train \
    --iters 500 \
    --save-every 50 \
    --batch-size 1 \
    --grad-accumulation-steps 3 \
    --num-layers 4 \
    $([ -f artifacts/adapter/adapters.safetensors ] && echo "--resume-adapter-file artifacts/adapter/adapters.safetensors")
```

### Export the tuned model

```bash
uv run mlx_lm.fuse \
    $([ -d artifacts/quantized-model ] && echo "--model artifacts/quantized-model" || echo "--model ${BASE_LLM}") \
    --save-path artifacts/fused-model \
    --adapter-path artifacts/adapter
```

### Chat with the fine-tuned model (CLI)

```bash
uv run mlx_lm.chat \
    $([ -d artifacts/quantized-model ] && echo "--model artifacts/quantized-model" || echo "--model ${BASE_LLM}") \
    --adapter-path artifacts/adapter
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
uv run huggingface-cli upload markvincevarga/mouse artifacts/fused-model .
```
