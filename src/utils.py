"""Public module containing all utility functions for the app's modules."""

import re
import string
from typing import Any, Dict, List, Tuple, Union

import nltk
import torch
import torch.nn.functional as f
from nltk.tokenize import sent_tokenize, word_tokenize, wordpunct_tokenize

nltk.download("punkt")
nltk.download("punkt_tab")


def clean_token(token: str) -> str:
    """
    Clean a subword token by removing the '##' prefix if present.

    This function is useful for reconstructing full words from subtokenized tokens
    produced by models like BERT.

    Args:
    ----
    token (str):
        The token string potentially starting with '##'.

    Returns:
    -------
    str:
        The cleaned token without the '##' prefix.

    """
    return token[2:] if token.startswith("##") else token


def join_subtokenized_tokens(tokens: List[Union[str, Tuple[str, str]]]) -> List[Union[str, Tuple[str, str]]]:
    """
    Join subtokenized tokens into full words, preserving optional entity labels.

    Subtokenized tokens are often split into pieces with '##' prefixes.
    This function reconstructs full tokens by concatenating these pieces.
    If tokens are tuples (token, label), it groups tokens with the same label together.

    Args:
    ----
    tokens (List[Union[str, Tuple[str, str]]]):
        List of tokens or (token, label) tuples to join.

    Returns:
    -------
    List[Union[str, Tuple[str, str]]]:
        List of full words as strings or tuples (word, label) if labels are present.

    """
    result = []
    buffer = ""
    current_label = None

    def flush():
        nonlocal buffer, current_label
        if buffer:
            # Append the buffered word with label if exists, else just the word
            if current_label:
                result.append((buffer.strip(), current_label))
            else:
                result.append(buffer.strip())
        buffer = ""
        current_label = None

    for token in tokens:
        if isinstance(token, tuple):
            word, label = token
        else:
            word, label = token, None

        if word.startswith("##"):
            # Append cleaned subtoken to buffer without space
            buffer += clean_token(word)
        else:
            if label:
                if current_label == label:
                    # Same label: concatenate with space
                    buffer += " " + clean_token(word)
                else:
                    # Different label: flush buffer and start new
                    flush()
                    buffer = clean_token(word)
                    current_label = label
            else:
                if buffer and current_label is None:
                    # No label and buffer active: concatenate with space
                    buffer += " " + clean_token(word)
                else:
                    # No label or new token: flush and start fresh
                    flush()
                    buffer = clean_token(word)
                    current_label = None

    flush()
    return result


def get_ner_annotations(tokens: List[str], entities: Dict[str, str]) -> List[Union[str, Tuple[str, str]]]:
    """
    Annotate tokens with their named entity labels where applicable.

    Given a list of tokens and a dictionary mapping tokens to entity labels,
    this function returns a list where tokens with entities are replaced by
    tuples (token, entity_label), and tokens without entities remain as strings.

    Args:
    ----
    tokens (List[str]):
        List of tokens (words) to annotate.

    entities (Dict[str, str]):
        Dictionary mapping tokens to their named entity labels.

    Returns:
    -------
    List[Union[str, Tuple[str, str]]]:
        List of tokens where tokens with entity labels are tuples (token, label).

    """
    return [(token.strip(), entities[token]) if token in entities else token for token in tokens]


def get_keyphrase_annotations(text: str) -> List[Union[str, Tuple[str, str]]]:
    """
    Tokenize the input text and create annotations for keyphrases.

    This function tokenizes the text using word punctuation tokenization,
    then returns a list where tokens containing underscores are converted into
    tuples with the underscore replaced by spaces and an empty string as the second element.
    Other tokens are returned as strings with a trailing space.

    Args:
    ----
    text (str):
        The input text to be tokenized and annotated.

    Returns:
    -------
    List[Union[str, Tuple[str, str]]]:
        List of tokens as strings or tuples where underscores in tokens
        indicate keyphrases split into tuples.

    """
    tokens = wordpunct_tokenize(text)
    return [f"{token} " if "_" not in token else (f"{token} ".replace("_", " "), "") for token in tokens]


def join_consecutive_tuples(tuples_list: List[Union[str, Tuple[str, str]]]) -> List[Union[str, Tuple[str, str]]]:
    """
    Merge consecutive tuples with the same second element by concatenating their first elements.

    This is useful for joining parts of keyphrases that are split into multiple tuples
    but share the same label or empty string.

    Args:
    ----
    tuples_list (List[Union[str, Tuple[str, str]]]):
        List containing strings or tuples to be joined.

    Returns:
    -------
    List[Union[str, Tuple[str, str]]]:
        List where consecutive tuples with the same label are merged into a single tuple.

    """
    result = []
    current_tuple = tuples_list[0]

    for next_tuple in tuples_list[1:]:
        # Check if both current and next are tuples with the same second element (label)
        if isinstance(current_tuple, tuple) and isinstance(next_tuple, tuple) and current_tuple[1] == next_tuple[1]:
            # Concatenate the first elements (strings) of the tuples
            current_tuple = (current_tuple[0] + next_tuple[0], current_tuple[1])
        else:
            # Append the current tuple/string and move on
            result.append(current_tuple)
            current_tuple = next_tuple

    # Append the last accumulated tuple/string
    result.append(current_tuple)
    return result


def remove_second_element(items_list: List[Union[str, Tuple[str, str]]]) -> List[Union[str, Tuple[str, str]]]:
    """
    Filter out tuples or strings starting with punctuation, preserving tuples that do not start with punctuation.

    This function removes tuples and strings where the first element starts with punctuation.
    Tuples that do not start with punctuation are preserved with their second element.
    Strings that do not start with punctuation are preserved as is.

    Args:
    ----
    items_list (List[Union[str, Tuple[str, str]]]):
        List containing strings or tuples to be filtered.

    Returns:
    -------
    List[Union[str, Tuple[str, str]]]:
        Filtered list without items starting with punctuation (except tuples preserved
        if they don't start with punctuation).

    """
    result: list[str | tuple[str, str]] = []
    for item in items_list:
        if (
            isinstance(item, tuple)
            and not item[0].startswith(tuple(string.punctuation))
            or isinstance(item, str)
            and not item.startswith(tuple(string.punctuation))
        ):
            result.append(item)

    return result


def split_string(string: str) -> List[str]:
    """
    Split input string into chunks of sentences with a maximum of 20 sentences per chunk.

    Args:
    ----
    string (str):
        The input text to be split into smaller chunks.

    Returns:
    -------
    List[str]:
        List of string chunks, each containing up to 20 sentences.

    """
    words = nltk.sent_tokenize(string)
    num_words = len(words)
    max_words_per_chunk = 20
    chunks = []

    for i in range(0, num_words, max_words_per_chunk):
        chunk = " ".join(words[i : i + max_words_per_chunk])  # noqa: E203
        chunks.append(chunk)
    return chunks


def remove_strings(list_of_items: List[Union[str, tuple]]) -> List[Union[str, tuple]]:
    """
    Remove string items from the list if they appear as the first element in any tuple within the same list.

    Args:
    ----
    list_of_items (List[Union[str, tuple]]):
        A list containing strings and tuples, where tuples have at least one element.

    Returns:
    -------
    List[Union[str, tuple]]:
        Filtered list with strings removed if they match any tuple's first element.

    """
    result = [
        item
        for item in list_of_items
        if not (isinstance(item, str) and item in [t[0] for t in list_of_items if isinstance(t, tuple)])
    ]
    return result


def split_string_into_batches(text: str, max_tokens: int, sentence: bool = True) -> List[str]:
    """
    Split a text into batches either by sentences or tokens, each batch containing up to max_tokens elements.

    Args:
    ----
    text (str):
        The input text to split.
    max_tokens (int):
        Maximum number of sentences or tokens per batch.
    sentence (bool, optional):
        Whether to split by sentences (True) or by tokens (False). Defaults to True.

    Returns:
    -------
    List[str]:
        List of batches, each batch is a string of sentences or tokens joined by spaces.

    """
    if sentence:
        sentences = sent_tokenize(text)
        num_sentences = len(sentences)
        num_batches = (num_sentences + max_tokens - 1) // max_tokens

        batches = []
        for i in range(num_batches):
            start_idx = i * max_tokens
            end_idx = (i + 1) * max_tokens
            batch = sentences[start_idx:end_idx]
            batches.append(" ".join(batch))

        return batches
    else:
        tokens = word_tokenize(text)
        num_tokens = len(tokens)
        num_batches = (num_tokens + max_tokens - 1) // max_tokens

        batches = []
        for i in range(num_batches):  # Fix: use range(num_batches), not range(len(num_batches))
            start_idx = i * max_tokens
            end_idx = (i + 1) * max_tokens
            batch = tokens[start_idx:end_idx]
            batches.append(" ".join(batch))
        return batches


def clean_token_for_annotation(token: str) -> str:
    """
    Clean a token string by removing or replacing unwanted characters
    commonly introduced by tokenizers or text processing pipelines.

    Args:
    ----
    token (str): The input token string to clean.

    Returns:
    -------
    str: The cleaned token string, with unwanted characters removed or replaced.

    """  # noqa: D205
    # Remove registered trademark symbol
    token = re.sub(r"[®]+", "", token).strip()
    # Replace end of sentence tokens with space
    token = re.sub(r"</s>", " ", token).strip()
    # Replace start of sentence tokens with space
    token = re.sub(r"<s>", " ", token).strip()
    # Replace special character 'ĉ' with space
    token = re.sub(r"ĉ", " ", token).strip()
    return token


def group_tokens_labels_confidences(
    tokens: List[str], labels: List[str], confidences: List[float]
) -> Tuple[List[str], List[str], List[str]]:
    """
    Group tokens, labels, and confidences by detecting special prefix characters
    that indicate token boundaries or continuations and concatenate accordingly.

    Args:
    ----
    tokens (List[str]): List of individual tokens output from the tokenizer.
    labels (List[str]): Corresponding labels predicted for each token.
    confidences (List[float]): Confidence scores for each token's label.

    Returns:
    -------
    Tuple[List[str], List[str], List[str]]:
        - Grouped tokens combined as strings.
        - Corresponding grouped labels combined as strings.
        - Corresponding grouped confidences concatenated as space-separated strings.

    """  # noqa: D205
    current_token = []
    current_label = []
    current_confidence = []

    for token, label, confidence in zip(tokens, labels, confidences):
        # If token starts with special char indicating new token group or contains specific chars,
        # treat as start of a new group
        if "Ġ" in token or any(x in token for x in ["â", "Ģ", "Ļ", "Â", "Ķ"]):
            current_token.append(token)
            current_label.append(label)
            current_confidence.append(str(confidence))
        else:
            # Otherwise, append token, label, and confidence to the last group
            if current_token:
                current_token[-1] += " " + token
                current_label[-1] += " " + label
                current_confidence[-1] += " " + str(confidence)

    return current_token, current_label, current_confidence


def clean_labels_from_multiple_o(labels: List[str]) -> List[str]:
    """
    Simplify label groups by collapsing sequences containing only 'O' labels
    to a single 'O', otherwise picking the first non-'O' label in the group.

    Args:
    ----
    labels (List[str]): List of label groups, each possibly containing multiple space-separated labels.

    Returns:
    -------
    List[str]: Cleaned list of labels with multiple 'O's collapsed and first non-'O' label retained.

    """  # noqa: D205
    clean_labels = []
    for label in labels:
        # If all labels in the group are 'O', simplify to single 'O'
        if set(label.split()) == {"O"}:
            clean_labels.append("O")
        else:
            # Otherwise, keep the first non-'O' label found
            for item in label.split():
                if item != "O":
                    clean_labels.append(item)
                    break
    return clean_labels


def average_confidences(confidences: List[str]) -> List[float]:
    """
    Convert string confidences which may be single values or space-separated
    multiple values into a list of averaged float confidence scores.

    Args:
    ----
    confidences (List[str]): List of confidence strings, either single or multiple values separated by spaces.

    Returns:
    -------
    List[float]: List of averaged confidence scores as floats.

    """  # noqa: D205
    new_confidences = []
    for score in confidences:
        if " " not in score:
            # Single confidence value, convert directly to float
            new_confidences.append(float(score))
        else:
            # Multiple confidence values, compute average
            scores = [float(scr) for scr in score.split()]
            avg_score = sum(scores) / len(scores)
            new_confidences.append(avg_score)
    return new_confidences


def _api_results(_model: Any, _tokenizer: Any, _text: str, _confidence_score: float) -> Tuple[List[str], List[str]]:
    """
    Process input text through the model and tokenizer to obtain tokens,
    predicted labels, and confidence scores; then clean and group tokens,
    adjust labels based on confidence threshold, and return final tokens and labels.

    Args:
    ----
    _model (Any): Pretrained HuggingFace sequence labeling model with id2label mapping.
    _tokenizer (Any): Corresponding tokenizer for the model.
    _text (str): Input text to process.
    _confidence_score (float): Threshold below which predicted labels are replaced with 'O'.

    Returns:
    -------
    Tuple[List[str], List[str]]:
        - List of cleaned and grouped tokens.
        - Corresponding list of cleaned labels after confidence filtering.

    """  # noqa: D205
    id2label = _model.config.id2label

    # Split input text into manageable batches for tokenization/model inference
    _text_batches = split_string_into_batches(_text, 20)

    words = []
    preds = []
    confidences = []

    for sentence in _text_batches:
        # Tokenize the sentence batch with truncation enabled
        tokens = _tokenizer(sentence, truncation=True)
        tokens = _tokenizer.convert_ids_to_tokens(tokens["input_ids"])
        words += tokens

        input_ids = _tokenizer.convert_tokens_to_ids(tokens)
        input_tensor = torch.tensor([input_ids])

        with torch.no_grad():
            outputs = _model(input_tensor)

        # Obtain predicted label indices and convert to label strings
        predictions = torch.argmax(outputs.logits, dim=2).squeeze().tolist()
        predicted_labels = [id2label[tag] for tag in predictions]
        preds += predicted_labels

        # Calculate confidence scores using softmax probabilities
        softmax_probs = f.softmax(outputs.logits, dim=2)
        confidence_scores = torch.max(softmax_probs, dim=2).values.squeeze().tolist()
        confidences += confidence_scores

    # Clean tokens by removing unwanted characters
    tokens = [clean_token_for_annotation(token) for token in words]

    # Group tokens, labels, and confidences based on token prefixes/special characters
    current_token, current_label, current_confidence = group_tokens_labels_confidences(tokens, preds, confidences)

    # Simplify labels, collapsing multiple 'O's and picking the first non-'O' label where applicable
    clean_labels = clean_labels_from_multiple_o(current_label)

    # Remove special characters from tokens
    new_tokens = [re.sub(r"[\sĠâĢĻÂĶ]+", "", token).strip() for token in current_token]

    # Average confidence scores for tokens with multiple sub-confidences
    new_confidences = average_confidences(current_confidence)

    # Apply confidence threshold: set label to 'O' if confidence below threshold
    for i, score in enumerate(new_confidences):
        if score <= _confidence_score:
            clean_labels[i] = "O"

    return new_tokens, clean_labels
