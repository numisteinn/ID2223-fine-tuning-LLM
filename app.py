"""Streamlit app for chatting with the fine-tuned Mistral model using MLX."""

import streamlit as st
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

# Page configuration
st.set_page_config(
    page_title="Fine-tuned Mistral Chat",
    page_icon="💬",
    layout="wide"
)

# Model path
MODEL_PATH = "artifacts/fused-model"

@st.cache_resource
def load_model():
    """Load the fine-tuned model and tokenizer."""
    with st.spinner("Loading model... This may take a moment."):
        model, tokenizer = load(MODEL_PATH)
    return model, tokenizer

def format_chat_prompt(messages):
    """Format chat messages into a prompt for the model."""
    formatted = []
    for message in messages:
        role = message["role"]
        content = message["content"]
        if role == "user":
            formatted.append(f"[INST] {content} [/INST]")
        else:
            formatted.append(content)
    return "\n".join(formatted)

# Title and description
st.title("💬 Chat with Fine-tuned Mistral")
st.markdown("Interact with your locally fine-tuned Mistral-7B model using MLX")

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Generation Settings")
    
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=2.0,
        value=0.7,
        step=0.1,
        help="Higher values make output more random, lower values more deterministic"
    )
    
    max_tokens = st.slider(
        "Max Tokens",
        min_value=50,
        max_value=2048,
        value=512,
        step=50,
        help="Maximum number of tokens to generate"
    )
    
    top_p = st.slider(
        "Top P",
        min_value=0.0,
        max_value=1.0,
        value=0.9,
        step=0.05,
        help="Nucleus sampling threshold"
    )
    
    st.divider()
    
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
    
    st.divider()
    st.caption(f"Model: `{MODEL_PATH}`")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Load model
try:
    model, tokenizer = load_model()
    st.sidebar.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.info("Make sure the model exists at `artifacts/fused-model/`. Run the fine-tuning steps first if you haven't.")
    st.stop()

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Type your message here..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Format the prompt for Mistral's instruction format
            formatted_prompt = f"[INST] {prompt} [/INST]"
            
            try:
                # Create sampler with the specified parameters
                sampler = make_sampler(temp=temperature, top_p=top_p)
                
                # Generate response
                response = generate(
                    model,
                    tokenizer,
                    prompt=formatted_prompt,
                    max_tokens=max_tokens,
                    sampler=sampler,
                    verbose=False
                )
                
                # Extract just the response (remove the prompt)
                response_text = response.strip()
                if response_text.startswith(formatted_prompt):
                    response_text = response_text[len(formatted_prompt):].strip()
                
                st.markdown(response_text)
                
                # Add assistant response to chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response_text
                })
                
            except Exception as e:
                st.error(f"Error generating response: {e}")
                st.session_state.messages.pop()  # Remove the user message if generation failed

# Footer
st.divider()
st.caption("Powered by MLX and Streamlit")

