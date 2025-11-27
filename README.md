# ID2223 - Lab 2: Fine tuning a LLM
A student project for the KTH course ID2223/FID3020 HT25 Scalable Machine Learning and Deep Learning created by Númi Steinn Baldursson and Márk Vince Varga, aka group Teki 🐢.

The goal of the project is to fine tune a LLM and create a UI for it.

## Model
Fine-tuned Llama 3 model on FineTome dataset, available in GGUF format at [numisteinn/ID2223-fine-tuning-LLM](https://huggingface.co/numisteinn/ID2223-fine-tuning-LLM)

## Setup

1. Install dependencies:
```bash
uv sync
```

## Running the Streamlit App

1. Make sure you have the correct GGUF filename in `app.py` (update `MODEL_FILE` constant)

2. Run the Streamlit app:
```bash
uv run streamlit run app.py
```

3. Open your browser to the URL shown (typically http://localhost:8501)

## Configuration

The app supports the following parameters (adjustable in the sidebar):
- **Max Tokens**: Maximum number of tokens to generate (128-2048)
- **Temperature**: Controls randomness in generation (0.0-2.0)

## Notes

- The model will be automatically downloaded from HuggingFace on first run and cached locally
- For GPU acceleration, install llama-cpp-python with CUDA support and adjust `n_gpu_layers` in `app.py`
