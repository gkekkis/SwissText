"""Public module that initializes sentiment analysis HuggingFace model."""

from typing import Any, Tuple

from transformers import AutoModelForSequenceClassification, AutoTokenizer


def init_sentiment_analysis_model(model_name: str) -> Tuple[Any, Any]:
    """
    Initialize a sentiment analysis model and tokenizer from HuggingFace.

    Loads a pretrained sequence classification model and its tokenizer
    suitable for sentiment analysis tasks.

    Args:
    ----
    model_name (str):
        The HuggingFace model identifier or local path for the pretrained sentiment analysis model.

    Returns:
    -------
    Tuple[Any, Any]:
        A tuple containing:
        - model: The loaded sequence classification model.
        - tokenizer: The corresponding tokenizer.

    """
    # Load the pretrained sequence classification model for sentiment analysis
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    # Load the tokenizer that corresponds to the model
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return model, tokenizer
