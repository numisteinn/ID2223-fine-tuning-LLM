import streamlit as st
from llama_cpp import Llama
from huggingface_hub import hf_hub_download
import os

# Page configuration
st.set_page_config(
    page_title="Fine-tuned Llama 3 Chat", page_icon="🦙", layout="centered"
)

# Constants
REPO_ID = "numisteinn/ID2223-fine-tuning-LLM"
MODEL_FILE = "model.gguf"  # Update this to match your actual GGUF filename


@st.cache_resource
def load_model():
    """Download and load the GGUF model from HuggingFace"""
    with st.spinner("Downloading model from HuggingFace..."):
        try:
            # Download model from HuggingFace
            model_path = hf_hub_download(
                repo_id=REPO_ID, filename=MODEL_FILE, cache_dir="./models", token = os.getenv("HUGGINGFACE_TOKEN")
            )

            # Load model with llama-cpp-python
            llm = Llama(
                model_path=model_path,
                n_ctx=2048,  # Context window
                n_threads=4,  # Number of CPU threads
                n_gpu_layers=0,  # Set to 0 for CPU, increase for GPU
            )
            return llm
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return None


def generate_response(llm, prompt, max_tokens=512, temperature=0.7):
    """Generate response from the model"""
    try:
        response = llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=0.95,
            repeat_penalty=1.1,
            stop=["</s>", "User:", "\n\n\n"],
            echo=False,
        )
        return response["choices"][0]["text"].strip()
    except Exception as e:
        return f"Error generating response: {str(e)}"


# Main UI
st.title("🦙 Fine-tuned Llama 3 Chat")
st.markdown("*Fine-tuned on FineTome Dataset*")

# Sidebar configuration
with st.sidebar:
    st.header("Configuration")
    max_tokens = st.slider("Max Tokens", 128, 2048, 512, 128)
    temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.1)

    st.markdown("---")
    st.markdown("### About")
    st.markdown(
        "This model is a fine-tuned version of Llama 3 trained on the FineTome dataset."
    )
    st.markdown(f"**Model**: [{REPO_ID}](https://huggingface.co/{REPO_ID})")

    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Load model
llm = load_model()

if llm is None:
    st.error("Failed to load model. Please check the model repository and filename.")
    st.stop()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask me anything..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Format the conversation history for the model
            conversation = ""
            for msg in st.session_state.messages:
                if msg["role"] == "user":
                    conversation += f"User: {msg['content']}\n"
                else:
                    conversation += f"Assistant: {msg['content']}\n"

            conversation += "Assistant: "

            # Generate response
            response = generate_response(llm, conversation, max_tokens, temperature)
            st.markdown(response)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

# Footer
st.markdown("---")
st.markdown("*Built by Númi Steinn Baldursson and Márk Vince Varga (Teki 🐢)*")
