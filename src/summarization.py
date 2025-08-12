"""Public module that initializes summarization HuggingFace model."""

from transformers import pipeline


def init_summarization_model(model_name: str) -> pipeline:
    """
    Initialize a text summarization pipeline using a HuggingFace pretrained model.

    Creates a pipeline for automatic text summarization leveraging the specified model.

    Args:
    ----
    model_name (str):
        Identifier or path of the pretrained summarization model.

    Returns:
    -------
    pipeline:
        A HuggingFace pipeline object configured for text summarization.

    """
    # Initialize the summarization pipeline with the provided model
    summarizer = pipeline("summarization", model=model_name)

    return summarizer
