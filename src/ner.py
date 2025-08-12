"""Public module that initializes NER HuggingFace model and runs NER."""

import re
from typing import Any, Callable, Dict, Tuple

from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline


def init_ner_model(model_name: str) -> Tuple[Any, Any, Callable]:
    """
    Initialize a Named Entity Recognition (NER) model pipeline from HuggingFace.

    Loads the pretrained tokenizer and model for token classification, then
    creates a HuggingFace pipeline for NER.

    Args:
    ----
    model_name (str):
        The HuggingFace model identifier or local path for the pretrained NER model.

    Returns:
    -------
    Tuple[Any, Any, Callable]:
        A tuple containing:
        - model: The loaded token classification model.
        - tokenizer: The corresponding tokenizer.
        - nlp: A callable NER pipeline function that takes text input and returns entity predictions.

    """
    # Load pretrained tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load pretrained token classification model
    model = AutoModelForTokenClassification.from_pretrained(model_name)

    # Initialize NER pipeline with model and tokenizer
    nlp = pipeline("ner", model=model, tokenizer=tokenizer)

    return model, tokenizer, nlp


def get_ner_entities(text: str, nlp: Callable, threshold: float = 0.8) -> Dict[str, str]:
    """
    Extract named entities from a given text using the NER pipeline.

    Applies the NER model to the input text, filters entities by confidence
    score threshold, and cleans up the entity labels.

    Args:
    ----
    text (str):
        The input text to extract named entities from.
    nlp (Callable):
        The NER pipeline callable returned by `init_ner_model`.
    threshold (float, optional):
        Confidence score threshold to filter out low-confidence entities.
        Defaults to 0.8.

    Returns:
    -------
    Dict[str, str]:
        A dictionary mapping entity words to their cleaned entity labels.

    """
    # Run the NER pipeline on input text
    ner_results = nlp(text)

    # Filter entities by confidence and clean entity labels by removing word-boundary hyphens
    entities = {item["word"]: re.sub(r"\w\-", "", item["entity"]) for item in ner_results if item["score"] >= threshold}

    return entities
