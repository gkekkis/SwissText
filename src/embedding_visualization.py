"""Public module that initializes embedding visualization model."""

from sentence_transformers import SentenceTransformer


def init_embedding_visualization_model(model_name: str) -> SentenceTransformer:
    """
    Initializes SentenceTransformer model for embedding visualizations.

    Args:
    ----
    model_name (str): HuggingFace model name/path.

    Returns:
    -------
    SentenceTransformer: SentenceTransformer instance.

    """
    model = SentenceTransformer(model_name)
    return model
