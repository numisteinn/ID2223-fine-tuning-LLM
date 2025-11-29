import os
from mlx_lm import load
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model(model_name_or_path: str):
    """
    Load a model from a local path or Hugging Face repo.
    
    Args:
        model_name_or_path (str): The name of the model or Hugging Face repo ID.
                                  If a local path 'artifacts/fused-model' exists, 
                                  it will be prioritized.
    
    Returns:
        model, tokenizer: The loaded model and tokenizer.
    """
    local_path = "artifacts/fused-model"
    
    if os.path.exists(local_path) and os.path.isdir(local_path):
        logger.info(f"Local model found at {local_path}. Loading from local directory...")
        path_to_load = local_path
    else:
        logger.info(f"Local model not found at {local_path}. Loading from Hugging Face: {model_name_or_path}...")
        path_to_load = model_name_or_path

    try:
        model, tokenizer = load(path_to_load)
        logger.info(f"Successfully loaded model from {path_to_load}")
        return model, tokenizer
    except Exception as e:
        logger.error(f"Failed to load model from {path_to_load}: {e}")
        raise

