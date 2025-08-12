"""Public module containing KeyphraseExtractionPipeline and initializes keyphrase extraction model."""

import nltk
import numpy as np
from transformers import AutoModelForTokenClassification, AutoTokenizer, TokenClassificationPipeline
from transformers.pipelines import AggregationStrategy

# Download necessary NLTK data packages for tokenization
nltk.download("punkt")
nltk.download("punkt_tab")


class KeyphraseExtractionPipeline(TokenClassificationPipeline):
    """
    Custom pipeline for keyphrase extraction using a token classification model.

    Inherits from Hugging Face's TokenClassificationPipeline and overrides
    the postprocess method to return a unique list of extracted keyphrases.

    Args:
    ----
    model (str or PreTrainedModel):
        The name or path of the pretrained model to use for token classification.
        Can be a string identifier from the Hugging Face hub or a local path.

    Additional args and kwargs are passed to the parent TokenClassificationPipeline.

    """

    def __init__(self, model, **kwargs):
        """
        Initialize the keyphrase extraction pipeline.

        Loads the pretrained model and tokenizer based on the given model name.

        Args:
        ----
        model (str):
            Pretrained model identifier or path.

        **kwargs:
            Additional keyword arguments passed to the parent class initializer.
            These can include configuration options for the pipeline.

        """
        super().__init__(
            model=AutoModelForTokenClassification.from_pretrained(model),
            tokenizer=AutoTokenizer.from_pretrained(model),
            **kwargs,
        )

    def postprocess(self, all_outputs):
        """
        Post-process the raw outputs from the token classification model.

        Applies aggregation strategy FIRST to merge token predictions,
        then extracts unique keyphrases.

        Args:
        ----
        all_outputs (list of dict):
            Raw token classification outputs from the model.

        Returns:
        -------
        numpy.ndarray:
            An array of unique keyphrases (strings) extracted from the input.

        """
        results = super().postprocess(all_outputs=all_outputs, aggregation_strategy=AggregationStrategy.FIRST)
        # Extract unique keyphrases stripping whitespace
        return np.unique([result.get("word").strip() for result in results])


def init_keyphrase_model(model_name: str) -> KeyphraseExtractionPipeline:
    """
    Initialize and return a KeyphraseExtractionPipeline with the specified model.

    Args:
    ----
    model_name (str):
        The pretrained model name or path to load for keyphrase extraction.

    Returns:
    -------
    KeyphraseExtractionPipeline:
        An instance of the KeyphraseExtractionPipeline ready for use.

    """
    extractor = KeyphraseExtractionPipeline(model=model_name)
    return extractor
