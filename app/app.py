"""Main app script."""

import os
import re
import sys

import pandas as pd
import plotly.express as px
import streamlit as st
import torch
import torch.nn.functional as f
from annotated_text import annotated_text
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

# Add project root to sys.path for local imports before importing other modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))  # noqa: E402

from src.config import LOGO_SIZE, TEXT_AREA_HEIGHT  # noqa: E402
from src.embedding_visualization import init_embedding_visualization_model  # noqa: E402
from src.error_messages import MODEL_NOT_FOUND  # noqa: E402
from src.keyword_extraction import init_keyphrase_model  # noqa: E402
from src.ner import get_ner_entities, init_ner_model  # noqa: E402
from src.sentiment_analysis import init_sentiment_analysis_model  # noqa: E402
from src.spam_detection import init_spam_detection_model  # noqa: E402
from src.summarization import init_summarization_model  # noqa: E402
from src.utils import get_keyphrase_annotations, get_ner_annotations, join_subtokenized_tokens  # noqa: E402

st.logo(image="app/assets/images/logo.png", size=LOGO_SIZE)
st.title("SwissText")
st.markdown("### 🧠 Explore various NLP models interactively")


if "model_initialized" not in st.session_state:
    st.session_state.model_initialized = False


(
    ner_tab,
    keyphrase_extraction_tab,
    sentiment_tab,
    spam_detection,
    summarization_tab,
    embedding_visualization_tab,
) = st.tabs(
    ["NER", "Keyphrase Extraction", "Sentiment Analysis", "Spam Detection", "Summarization", "Embedding Visualization"]
)


with st.spinner("Loading...", show_time=True), ner_tab:
    ner_model_name = st.selectbox(
        label="Choose NER model", options=["dslim/distilbert-NER", "boltuix/NeuroBERT-NER"], key="ner_options"
    )

    threshold = st.slider(label="Confidence score", min_value=0.0, max_value=1.0, step=0.05, value=0.85)

    text = st.text_area(
        label="Please insert some text", placeholder="Some text...", height=TEXT_AREA_HEIGHT, key="ner_text"
    )

    try:
        ner_model, ner_tokenizer, ner_nlp = init_ner_model(model_name=ner_model_name)
    except OSError:
        st.error(MODEL_NOT_FOUND.replace("__MODEL__NAME__", ner_model_name))
        st.stop()
    tokens = ner_tokenizer.tokenize(text)
    entities = get_ner_entities(text=text, nlp=ner_nlp, threshold=threshold)
    annotations = get_ner_annotations(tokens=tokens, entities=entities)
    joint_annotations = join_subtokenized_tokens(tokens=annotations)
    text_with_annotations = annotated_text(joint_annotations)

    if text_with_annotations:
        st.markdown(text_with_annotations)


with st.spinner("Loading...", show_time=True), keyphrase_extraction_tab:
    keyphrase_annotations = None
    keyphrase_model_name = st.selectbox(
        label="Choose KeyPhrase extractor model",
        options=["ml6team/keyphrase-extraction-distilbert-inspec", "ml6team/keyphrase-extraction-distilbert-openkp"],
        key="keyphrase_options",
        accept_new_options=True,
    )

    try:
        extractor = init_keyphrase_model(model_name=keyphrase_model_name)
    except OSError:
        st.error(MODEL_NOT_FOUND.replace("__MODEL__NAME__", keyphrase_model_name))
        st.stop()

    text = st.text_area(
        label="Please insert some text", placeholder="Some text...", height=TEXT_AREA_HEIGHT, key="keyphrase_text"
    )

    keyphrases = extractor(text)

    for phrase in keyphrases:
        text = re.sub(phrase, re.sub(r" +", "_", phrase), text, flags=re.IGNORECASE)

    keyphrase_annotation_list = get_keyphrase_annotations(text=text)

    keyphrases = list({keyphrase[0] for keyphrase in keyphrase_annotation_list if isinstance(keyphrase, tuple)})

    keyphrase_annotations = annotated_text(keyphrase_annotation_list)

    if text:
        st.markdown(keyphrase_annotations)

        keyphrase_statistics_df = pd.DataFrame({"Key Phrases": keyphrases}, index=list(range(len(keyphrases))))

        st.dataframe(keyphrase_statistics_df)


with st.spinner("Loading...", show_time=True), sentiment_tab:
    sentiment_model_name = st.selectbox(
        label="Choose Sentiment Analysis model",
        options=[
            "distilbert-base-uncased-finetuned-sst-2-english",
            "j-hartmann/emotion-english-distilroberta-base",
            "madhurjindal/autonlp-Gibberish-Detector-492513457",
            "s-nlp/roberta_toxicity_classifier",
            "facebook/roberta-hate-speech-dynabench-r4-target",
        ],
        key="sentiment_options",
        accept_new_options=True,
    )

    text = st.text_area(
        label="Please insert some text", placeholder="Some text...", height=TEXT_AREA_HEIGHT, key="sentiment_text"
    )

    if text:
        try:
            sentiment_analyzer, sentiment_tokenizer = init_sentiment_analysis_model(model_name=sentiment_model_name)
        except OSError:
            st.error(MODEL_NOT_FOUND.replace("__MODEL__NAME__", sentiment_model_name))
            st.stop()

        # Tokenize input
        inputs = sentiment_tokenizer(text, return_tensors="pt")

        # Get model outputs
        with torch.no_grad():
            logits = sentiment_analyzer(**inputs).logits

        # Apply softmax to get probabilities
        probs = f.softmax(logits, dim=1)[0]  # dim=1: apply across classes

        # Get predicted label
        predicted_class_id = logits.argmax().item()
        predicted_label = sentiment_analyzer.config.id2label[predicted_class_id]

        # Show predicted label
        st.success(f"**Predicted Sentiment:** {predicted_label.capitalize()}")

        # Show probabilities as bar chart
        prob_dict = {sentiment_analyzer.config.id2label[i].capitalize(): float(probs[i]) for i in range(len(probs))}
        st.subheader("Probabilities")
        st.bar_chart(prob_dict, horizontal=True)


with st.spinner("Loading...", show_time=True), spam_detection:
    spam_detection_model_name = st.selectbox(
        label="Choose Spam Detection model",
        options=["Titeiiko/OTIS-Official-Spam-Model", "mrm8488/bert-tiny-finetuned-sms-spam-detection"],
        key="spam_detection_options",
        accept_new_options=True,
    )

    text = st.text_area(
        label="Please insert some text", placeholder="Some text...", height=TEXT_AREA_HEIGHT, key="spam_detection_text"
    )

    if text:
        try:
            pipe = init_spam_detection_model(model_name=spam_detection_model_name)
        except OSError:
            st.error(MODEL_NOT_FOUND.replace("__MODEL__NAME__", spam_detection_model_name))
            st.stop()
        spam_detection_text = pipe(text)[0]
        spam_detection_score = spam_detection_text["score"]
        spam_detection_label = spam_detection_text["label"]
        predicted_spam_detection_label = "Not Spam" if spam_detection_label == "LABEL_0" else "Spam"
        other_spam_detection_label = "Spam" if predicted_spam_detection_label == "Not Spam" else "Not Spam"
        result = {
            predicted_spam_detection_label: spam_detection_score,
            other_spam_detection_label: 1 - spam_detection_score,
        }

        # Show predicted label
        st.success(f"**Predicted Label:** {predicted_spam_detection_label}")

        # Show probabilities as bar chart
        st.subheader("Probabilities")
        st.bar_chart(result, horizontal=True)


with st.spinner("Loading...", show_time=True), summarization_tab:
    summarization_model_name = st.selectbox(
        label="Choose Spam Detection model",
        options=["facebook/bart-large-cnn"],
        key="summarization_options",
        accept_new_options=True,
    )

    min_length = st.number_input(label="Insert min length", step=10, min_value=5, placeholder=30)
    max_length = st.number_input(label="Insert max length", step=10, min_value=10, placeholder=130)

    text = st.text_area(
        label="Please insert some text", placeholder="Some text...", height=TEXT_AREA_HEIGHT, key="summarization_text"
    )

    if text:
        try:
            pipe = init_summarization_model(model_name=summarization_model_name)
        except OSError:
            st.error(MODEL_NOT_FOUND.replace("__MODEL__NAME__", summarization_model_name))
            st.stop()
        try:
            summary = pipe(text, min_length=int(min_length), max_length=int(max_length))
            st.markdown(summary[0]["summary_text"])
        except IndexError:
            st.error("Length of input text is too large. Please try a smaller chunk.")

with st.spinner("Loading...", show_time=True), embedding_visualization_tab:
    embedding_visualization_model_name = st.selectbox(
        label="Choose Spam Detection model",
        options=["all-mpnet-base-v2", "all-distilroberta-v1", "all-MiniLM-L12-v2"],
        key="embedding_visualization_options_key",
        accept_new_options=True,
    )

    sentences = st.text_area(
        label="Enter one sentence per line",
        placeholder="Enter sentences...",
        key="embedding_visualization_text_area_key",
    )

    plot_dimensions = st.radio(
        label="Please enter plot dimensions",
        options=[2, 3],
        index=0,
        horizontal=True,
        key="embedding_visualization_dimensions_key",
    )

    n_clusters = st.number_input(
        label="Please enter number of clusters", min_value=2, key="embedding_visualization_n_clusters_key"
    )

    sentence_list = [sentence.strip() for sentence in sentences.split("\n")]
    n_samples = len(sentence_list)
    perplexity = min(30, n_samples - 1)
    perplexity = max(1, perplexity)

    if len(sentence_list) and len(sentence_list[0]):
        try:
            embedding_visualization_model = init_embedding_visualization_model(
                model_name=embedding_visualization_model_name
            )
            embeddings = embedding_visualization_model.encode(sentence_list)

            # Dimensionality reduction
            tsne = TSNE(n_components=plot_dimensions, perplexity=perplexity, random_state=42)
            reduced = tsne.fit_transform(embeddings)

            # Plot
            df = pd.DataFrame(reduced, columns=["x", "y"] if plot_dimensions == 2 else ["x", "y", "z"])
            kmeans = KMeans(n_clusters=int(n_clusters), random_state=42)
            df["cluster"] = kmeans.fit_predict(df[["x", "y"]] if plot_dimensions == 2 else df[["x", "y", "z"]])
            df["sentence"] = sentence_list

            if plot_dimensions == 2:
                fig = px.scatter(
                    df, x="x", y="y", title="t-SNE Embedding Visualization", color="cluster", hover_data="sentence"
                )
            else:
                fig = px.scatter_3d(df, x="x", y="y", z="z", color="cluster", hover_data="sentence")

            # Center the title
            fig.update_layout(
                title={
                    "text": "Sentence Embedding Visualization",
                    "x": 0.5,  # 0.5 centers the title
                    "xanchor": "center",
                },
                height=600,
            )

            # Plot with max width in Streamlit
            st.plotly_chart(fig, use_container_width=True)
        except OSError:
            st.error(MODEL_NOT_FOUND.replace("__MODEL__NAME__", embedding_visualization_model_name))
            st.stop()
