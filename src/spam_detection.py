"""Public module that initializes spam detection HuggingFace model."""

from transformers import pipeline


def init_spam_detection_model(model_name: str) -> pipeline:
    """
    Initialize a text classification pipeline for spam detection using HuggingFace.

    Loads a pretrained model wrapped in a pipeline configured for text classification,
    suitable for detecting spam messages.

    Args:
    ----
    model_name (str):
        The HuggingFace model identifier or local path for the pretrained spam detection model.

    Returns:
    -------
    pipeline:
        A HuggingFace pipeline object configured for text classification (spam detection).

    """
    # Create a text classification pipeline with the specified spam detection model
    pipe = pipeline("text-classification", model=model_name)

    return pipe
