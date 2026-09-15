import streamlit as st
import torch
import joblib
import numpy as np
import pandas as pd
import json
import shap
import re
import plotly.graph_objects as go
import plotly.express as px
from scipy.sparse import hstack, csr_matrix
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, pipeline


# Set up the page title and layout
st.set_page_config(
    page_title="Amazon Products Review Analyser", layout="centered")
st.title("Review Analyser")
st.markdown("Analyse an Amazon review for **sentiment** and **suspicious behaviour** using two independent machine learning models.")


def get_review_frequency(reviewer_stats, reviewer_id):
    """
    This function figures out the review_frequency value to use at prediction time.
    Returns (review_frequency, source_label).
    Resolution order:
    Reviewer ID found in lookup table  (use historical count)
    Reviewer ID provided but not found (use training median and provide warning)
    No Reviewer ID provided            (use training median and  information note)
    """
    lookup = reviewer_stats.get("reviewer_lookup", {})
    median = reviewer_stats.get("median", 1)

    if reviewer_id.strip():
        if reviewer_id.strip() in lookup:
            freq = lookup[reviewer_id.strip()]
            source = f"Reviewer found in training data: frequency used: **{freq}**"
            found = True
        else:
            freq = median
            source = f"Reviewer ID not recognised: using training median (**{median}**)"
            found = False
    else:
        freq = median
        source = f"Reviewer history unavailable: review frequency estimated at population median (**{median}**)"
        found = False

    return freq, source, found


@st.cache_resource
def load_reviewer_stats(path="reviewer_stats.json"):
    """
    Loads pre-computed reviewer statistics from JSON.
    Falls back to safe defaults if the file is missing.
    """
    try:
        with open(path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        st.warning("reviewer_stats.json not found. Falling back to median = 1.")
        return {"median": 1, "mode": 1, "mean": 1.0, "reviewer_lookup": {}}


@st.cache_resource
def load_sentiment_model():
    """
    Loads the DistilBERT sentiment classifier from disk.
    Cached so it only loads once.
    """
    model = DistilBertForSequenceClassification.from_pretrained(
        "models/bert_model",
        output_attentions=True
    )
    tokenizer = DistilBertTokenizer.from_pretrained("models/bert_model")
    device = 0 if torch.cuda.is_available() else -1
    classifier = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=device
    )
    return {"classifier": classifier, "tokenizer": tokenizer, "model": model}


@st.cache_resource
def load_suspicious_model():
    """
    Loads the logistic regression model, TF-IDF vectorizer,
    and selected structured features list.
    """
    model = joblib.load("model_suspicious/logistic_model.pkl")
    tfidf = joblib.load("model_suspicious/logistic_tfidf.pkl")
    scaler = joblib.load("model_suspicious/logistic_scaler.pkl")
    background = joblib.load("model_suspicious/shap_background.pkl")
    with open("new_features/selected_features.json", "r") as f:
        selected_features = json.load(f)
    return model, tfidf, scaler, background, selected_features


@st.cache_resource
def load_shap_explainer(_model, _background):
    return shap.LinearExplainer(_model, _background)


def engineer_features(review, rating, verified, review_frequency):
    """
    Computes the four structured features used by the suspicious review model.

    Parameters:
        review (str): The review text.
        rating (int): Star rating given by the reviewer (1-5).
        verified (str): Whether the purchase was verified ('Yes', 'No', 'Unknown').
        review_frequency (int): Historical review count for this reviewer.

    Returns:
        dict: A dictionary of four engineered features.
    """
    if verified == "Yes":
        unverified_ratio = 0
    elif verified == "No":
        unverified_ratio = 1
    else:
        unverified_ratio = 0.5

    return {
        "review_frequency":     review_frequency,
        "unverified_ratio":     unverified_ratio,
        "extreme_rating_ratio": 1 if rating in [1, 5] else 0,
        "very_short_review":    1 if len(review.split()) < 20 else 0,
    }


def gauge_chart(value, title):
    """
    Renders a gauge chart showing model confidence.
    """
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(value * 100, 1),
        number={"suffix": "%"},
        title={"text": title},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#e63946" if value > 0.5 else "#2a9d8f"},
            "steps": [
                {"range": [0, 40],   "color": "#d4edda"},
                {"range": [40, 70],  "color": "#fff3cd"},
                {"range": [70, 100], "color": "#f8d7da"},
            ],
        }
    ))
    fig.update_layout(height=250, margin=dict(t=40, b=0, l=20, r=20))
    return fig


def get_word_importance(review, tokenizer, model, predicted_label):
    """
    Extracts word level importance scores from DistilBERT attention weights.

    Averages attention across all layers and heads, then filters out
    stopwords, punctuation, subword tokens, and short tokens.

    Parameters:
        review (str): The review text to analyse.
        tokenizer: HuggingFace tokenizer for DistilBERT.
        model: DistilBERT model with output_attentions=True.
        predicted_label (str): The predicted sentiment label (0 or 1).

    Returns:
        list: Top 15 (token, score) tuples sorted by absolute importance.
    """
    inputs = tokenizer(
        review,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        return_attention_mask=True
    )

    with torch.no_grad():
        outputs = model(
            **inputs,
            output_attentions=True  # explicitly request attentions
        )

    # Check attentions were returned
    if not outputs.attentions:
        return []

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    attentions = torch.stack(list(outputs.attentions))
    cls_attention = attentions.mean(dim=0)[0].mean(dim=0)[0]

    word_importance = []
    for token, attn in zip(tokens, cls_attention.numpy()):
        if token in ["[CLS]", "[SEP]", "[PAD]"]:
            continue
        score = float(attn) if predicted_label == "LABEL_1" else -float(attn)
        word_importance.append((token, score))

    NOISE = {
        "i", "a", "the", "and", "or", "is", "it", "in", "of", "to",
        "was", "be", "as", "at", "an", "if", "on", "he", "she", "we",
        "you", "my", "so", "do", "by", "its", "this", "that", "not",
        "are", "but", "for", "have", "had", "has", "with", "from",
        "they", "them", "their", "been", "will", "would", "could",
        "should", "more", "also", "just", "get", "got", "can", "did"
    }
    word_importance = [
        (token, score) for token, score in word_importance
        if token.lower() not in NOISE
        and len(token) > 2
        and not token.startswith("##")
        and re.match(r'^[a-zA-Z]+$', token)
    ]
    word_importance.sort(key=lambda x: abs(x[1]), reverse=True)
    return word_importance[:15]


def word_importance_chart(word_importance, predicted_label):
    """
    Renders a horizontal bar chart of word importance scores.
    Positive scores push toward the predicted sentiment.
    Negative scores push against it.
    """
    if not word_importance:
        return None

    words, scores = zip(*word_importance)
    colors = ["#e63946" if s > 0 else "#2a9d8f" for s in scores]
    sentiment_word = "Positive" if predicted_label == "LABEL_1" else "Negative"

    fig = go.Figure(go.Bar(
        x=list(scores),
        y=list(words),
        orientation="h",
        marker_color=colors,
        text=[f"{v:+.4f}" for v in scores],
        textposition="outside"
    ))
    fig.update_layout(
        title=f"Word Importance for {sentiment_word} Sentiment",
        xaxis_title=f"Importance Score (negative = pushes toward {sentiment_word})" if sentiment_word == "Negative" else f"Importance Score (positive = pushes toward {sentiment_word})",
        height=450,
        margin=dict(t=50, b=40, l=120, r=120),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        yaxis=dict(autorange="reversed")
    )
    return fig


def shap_bar_chart(shap_values, feature_names):
    """
    Renders a horizontal bar chart of SHAP values for the suspicious review model.
    Red = pushes toward suspicious. Teal = pushes away from suspicious.
    """
    df = pd.DataFrame({
        "Feature": feature_names,
        "Impact":  shap_values
    }).sort_values("Impact", ascending=True)

    colors = ["#e63946" if v > 0 else "#2a9d8f" for v in df["Impact"]]

    fig = go.Figure(go.Bar(
        x=df["Impact"],
        y=df["Feature"],
        orientation="h",
        marker_color=colors,
        text=[f"{v:+.4f}" for v in df["Impact"]],
        textposition="outside"
    ))
    fig.update_layout(
        title="Feature Impact (SHAP)",
        xaxis_title="SHAP Impact (positive = pushes toward Suspicious)",
        yaxis_title="",
        height=300,
        margin=dict(t=50, b=40, l=120, r=120),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white")
    )
    return fig


def feature_breakdown_chart(features):
    """
    Renders a horizontal bar chart showing the actual value
    of each structured feature for this specific review.
    """
    df = pd.DataFrame({
        "Feature": list(features.keys()),
        "Value":   list(features.values())
    })

    fig = px.bar(
        df, x="Value", y="Feature",
        orientation="h",
        title="Structured Feature Values for This Review",
        color="Value",
        color_continuous_scale=["#2a9d8f", "#e9c46a", "#e63946"],
        range_color=[0, 1]
    )
    fig.update_layout(
        height=300,
        margin=dict(t=50, b=40, l=10, r=20),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        coloraxis_showscale=False
    )
    return fig


def build_suspicious_bullets(features, rating):
    """
    Builds detailed bullet point explanations for each structured feature,
    explaining what the value means in the context of suspicious review detection.
    """
    bullets = []

    # Unverified ratio
    if features["unverified_ratio"] == 0:
        bullets.append(
            " **Verified purchase** : this review comes from a confirmed buyer, "
            "which reduces the likelihood of suspicious behaviour."
        )
    elif features["unverified_ratio"] == 1:
        bullets.append(
            " **Unverified purchase**: the reviewer did not verify they purchased "
            "this product. Unverified reviews are more commonly associated with "
            "suspicious or incentivised reviewing behaviour."
        )
    else:
        bullets.append(
            " **Purchase verification unknown**: verification status could not be "
            "determined. A neutral value has been used."
        )

    # Extreme rating
    if features["extreme_rating_ratio"] == 1:
        if rating == 5:
            bullets.append(
                f" **High extreme rating ({rating} stars)**: a 5 star rating was given. "
                "While extreme ratings are common in fake reviews, they are equally common "
                "among genuine enthusiastic buyers. On its own this is a weak signal, "
                "and the model treats it as a mild indicator rather than a strong flag."
            )
        else:
            bullets.append(
                f" **Low extreme rating ({rating} stars)**: a 1 star rating was given. "
                "Extremely low ratings can indicate genuine dissatisfaction or targeted "
                "negative review campaigns. The model treats this as a mild indicator "
                "that warrants consideration alongside other features."
            )
    else:
        bullets.append(
            f" **Moderate rating ({rating} stars)**: a mid range rating was given. "
            "Non extreme ratings are less commonly associated with suspicious behaviour "
            "and this pushes the model away from flagging this review."
        )

    # Very short review
    if features["very_short_review"] == 1:
        bullets.append(
            " **Very short review (fewer than 20 words)**: the review is unusually short. "
            "Fake reviews often lack detail because they are generated quickly or in bulk. "
            "Short reviews provide less textual evidence for the model to analyse."
        )
    else:
        bullets.append(
            "**Sufficient review length**: the review contains enough words to analyse. "
            "Longer reviews generally provide more genuine detail and are less commonly "
            "associated with suspicious behaviour."
        )

    # Review frequency (volume as proxy for suspicious activity)
    freq = features["review_frequency"]
    if freq == 1:
        bullets.append(
            f"**Low reviewer volume (frequency: {freq})**: this reviewer has posted "
            "very few reviews in total. Low volume reviewers are less likely to be "
            "engaged in coordinated or incentivised reviewing campaigns, which typically "
            "involve posting many reviews across multiple products."
        )
    elif freq <= 5:
        bullets.append(
            f"**Moderate reviewer volume (frequency: {freq})**: this reviewer has "
            f"posted {freq} reviews in total. This is within a normal range and does "
            "not strongly indicate suspicious behaviour on its own."
        )
    else:
        bullets.append(
            f" **High reviewer volume (frequency: {freq})**: this reviewer has posted "
            f"{freq} reviews in total. High volume reviewing was used as a signal when "
            "defining suspicious behaviour in the training data, as coordinated fake "
            "review campaigns tend to post many reviews across products in short periods."
        )

    return bullets


def interpret_shap_features(shap_df):
    """
    Generates a plain English interpretation of each SHAP value,
    explaining what direction and magnitude the feature pushed the prediction.
    """
    interpretations = []
    for _, row in shap_df.iterrows():
        feature = row["Feature"]
        impact = row["Impact"]
        if abs(impact) < 0.001:
            direction = "had negligible impact on this prediction"
            strength = ""
        elif impact > 0:
            direction = "pushed **toward** suspicious"
            strength = "strongly" if abs(impact) > 0.05 else "slightly"
        else:
            direction = "pushed **away** from suspicious"
            strength = "strongly" if abs(impact) > 0.05 else "slightly"

        if feature == "review_frequency":
            if impact > 0:
                explanation = "Higher reviewer volume increases suspicion as it mirrors the high frequency posting patterns seen in coordinated fake review campaigns."
            elif abs(impact) < 0.001:
                explanation = "Reviewer volume had no meaningful impact on this prediction."
            else:
                explanation = "Lower reviewer volume reduces suspicion as this reviewer has not posted at high enough volume to suggest coordinated or incentivised reviewing behaviour."
            interpretations.append(
                f"- **{feature}** ({abs(impact):.4f}): {strength} {direction}. {explanation}"
            )
        elif feature == "unverified_ratio":
            interpretations.append(
                f"- **{feature}** ({abs(impact):.4f}): {strength} {direction}. "
                "Unverified purchases are more commonly linked to suspicious reviewing "
                "behaviour where the reviewer may not have actually bought the product."
            )
        elif feature == "extreme_rating_ratio":
            if impact > 0:
                explanation = "An extreme rating was given. Extreme ratings (1 or 5 stars) appear more frequently in suspicious reviews, though they are equally common in genuine enthusiastic reviews. The model treats this as a mild signal rather than a strong flag."
            elif abs(impact) < 0.001:
                explanation = "Extreme rating had no meaningful impact on this prediction."
            else:
                explanation = "A non-extreme rating was given. Mid-range ratings are less commonly associated with suspicious behaviour and this pushed the model away from flagging this review."
            interpretations.append(
                f"- **{feature}** ({abs(impact):.4f}): {strength} {direction}. {explanation}"
            )
        elif feature == "very_short_review":
            interpretations.append(
                f"- **{feature}** ({abs(impact):.4f}): {strength} {direction}. "
                "Very short reviews lack the detail expected from genuine buyers and are "
                "commonly associated with bulk or incentivised reviewing."
            )

    return interpretations


# Load reviewer stats and models at startup

reviewer_stats = load_reviewer_stats()

with st.spinner("Loading models..."):
    sentiment_bundle = load_sentiment_model()
    sentiment_classifier = sentiment_bundle["classifier"]
    tokenizer = sentiment_bundle["tokenizer"]
    bert_model = sentiment_bundle["model"]
    suspicious_model, tfidf, scaler, shap_background, selected_features = load_suspicious_model()
    explainer = load_shap_explainer(suspicious_model, shap_background)

st.caption("Models loaded and ready")

# sidebar with app information and reviewer ID input

with st.sidebar:
    st.title("About This App")
    st.markdown(
        "This tool analyses Amazon product reviews using two independent "
        "machine learning models, one for sentiment and one for "
        "suspicious behaviour detection."
    )
    st.markdown("---")
    st.markdown("### Models")
    st.markdown(
        "**Sentiment Model**  \nDistilBERT: fine-tuned for binary "
        "text classification (Positive / Negative)\n\n"
        "**Suspicious Review Model**  \nLogistic Regression: trained on "
        "four engineered structured features"
    )
    st.markdown("---")
    st.markdown("### Dataset")
    st.markdown(
        "Trained on the Amazon Product Reviews dataset  \n"
        "**827,398** reviews  \n"
        "**630,543** unique reviewers"
    )
    st.markdown("---")
    st.markdown("### Team")
    st.markdown("IGP Team 4")
    st.markdown("---")
    st.markdown("### Links")
    st.markdown(
        "[Hugging Face Space](https://huggingface.co/spaces/IGPTeam4/IGP_webApp/tree/main)")


# user input section

st.subheader("Review Input")

review = st.text_area(
    "Enter a Review",
    height=150,
    placeholder="e.g. I bought this product last month and have been very happy with it. The quality is excellent and it arrived quickly. I would highly recommend it to anyone looking for a reliable and affordable option."
)

# word count indicator
word_count = len(review.split()) if review.strip() else 0
if word_count > 0:
    if word_count < 20:
        st.caption(
            f"Word count: **{word_count}**: this review is considered short (fewer than 20 words)")
    else:
        st.caption(f"Word count: **{word_count}**")

col1, col2 = st.columns(2)
with col1:
    rating = st.slider("Rating", 1, 5, 5)
with col2:
    verified = st.selectbox("Verified Purchase", ["Unknown", "Yes", "No"])

reviewer_id = st.text_input(
    "Reviewer ID (optional)",
    placeholder="e.g. A3SGXH7AUHU8GW",
    help="If provided, the app looks up this reviewer's historical review count from the training data. If not found or left blank, the population median is used instead."
)

# Resolve review_frequency before prediction
review_frequency, freq_source, freq_found = get_review_frequency(
    reviewer_stats, reviewer_id)


# analyse button

btn_col1, btn_col2 = st.columns([2, 1])
with btn_col1:
    analyse_clicked = st.button("Analyse Review", use_container_width=True)
with btn_col2:
    clear_clicked = st.button("Clear", use_container_width=True)

if clear_clicked:
    st.rerun()

if analyse_clicked:
    if review.strip() == "":
        st.warning("Please enter a review before analysing.")
    else:
        features = engineer_features(
            review, rating, verified, review_frequency)
        st.toast("Analysis complete")

        # run sentiment model before expanders so all variables are defined
        with st.spinner("Running sentiment model..."):
            sentiment_result = sentiment_classifier(
                review, truncation=True, max_length=512)[0]

        # Extract scores for both classes
        s_label = sentiment_result["label"]
        s_score = sentiment_result["score"]
        s_pct = round(s_score * 100, 2)
        s_word = "Positive" if s_label == "LABEL_1" else "Negative"

        # run suspicious model before expanders so all variables are defined
        structured_features = np.array(
            [[features[f] for f in selected_features]])
        structured_scaled = scaler.transform(structured_features)
        text_features = tfidf.transform([review])
        combined_input = hstack([text_features, csr_matrix(structured_scaled)])

        prob = suspicious_model.predict_proba(combined_input)[0][1]
        threshold = 0.4  # Adjusted threshold for better precision during model evaluation
        pred = 1 if prob > threshold else 0
        label = "Suspicious" if pred == 1 else "Not Suspicious"
        prob_pct = round(prob * 100, 1)
        likely = "likely" if prob > 0.4 else "likely not"

        # Show reviewer frequency note
        if not freq_found:
            st.info(f"{freq_source}")

        # summary card
        st.subheader("Summary")
        sum_col1, sum_col2 = st.columns(2)
        with sum_col1:
            if s_label == "LABEL_1":
                st.success(
                    f"Sentiment: **Positive**  \nConfidence: **{s_pct}%**")
            else:
                st.error(
                    f"Sentiment: **Negative**  \nConfidence: **{s_pct}%**")
        with sum_col2:
            if pred == 1:
                st.error(
                    f"Review: **Suspicious**  \nConfidence: **{prob_pct}%**")
            else:
                st.success(
                    f"Review: **Not Suspicious**  \nConfidence: **{prob_pct}%**")

        # side by side gauge charts
        gauge_col1, gauge_col2 = st.columns(2)
        with gauge_col1:
            st.plotly_chart(gauge_chart(
                s_score, "Sentiment Confidence"), use_container_width=True)
        with gauge_col2:
            st.plotly_chart(gauge_chart(
                prob, "Suspicion Confidence"), use_container_width=True)

        # sentiment detailed breakdown
        with st.expander("Sentiment Analysis: Detailed Breakdown", expanded=True):

            if s_label == "LABEL_1":
                st.success("Sentiment: **Positive**")
            else:
                st.error("Sentiment: **Negative**")

            # Interpretation sentence
            st.markdown(
                f"The model is **{s_pct}%** confident that this review sentiment is "
                f"**likely {s_word}** as a result of the following words:"
            )

            # Word importance
            with st.spinner("Computing word importance..."):
                word_importance = get_word_importance(
                    review, tokenizer, bert_model, s_label)

            if word_importance:
                word_fig = word_importance_chart(word_importance, s_label)
                seen = set()
                pos_words = []
                for w, v in word_importance:
                    if v > 0 and w.lower() not in seen:
                        seen.add(w.lower())
                        pos_words.append((w, v))
                    if len(pos_words) == 5:
                        break

                seen = set()
                neg_words = []
                for w, v in word_importance:
                    if v < 0 and w.lower() not in seen:
                        seen.add(w.lower())
                        neg_words.append((w, v))
                    if len(neg_words) == 5:
                        break

                if word_fig:
                    st.plotly_chart(word_fig, use_container_width=True)

                st.markdown("#### What the words mean for this prediction")

                if s_word == "Positive":
                    if pos_words:
                        word_list = ", ".join(
                            [f"**{w}**" for w, _ in pos_words])
                        st.markdown(
                            f"Words that most pushed the model toward **Positive**: {word_list}. "
                            "These words carry positive connotations that the model strongly associates "
                            "with genuine, satisfied reviews."
                        )
                    if neg_words:
                        word_list = ", ".join(
                            [f"**{w}**" for w, _ in neg_words])
                        st.markdown(
                            f"Words that pushed against **Positive** (toward Negative): {word_list}. "
                            "Despite these words, the overall sentiment remained positive."
                        )
                else:
                    if neg_words:
                        word_list = ", ".join(
                            [f"**{w}**" for w, _ in neg_words])
                        st.markdown(
                            f"Words that most pushed the model toward **Negative**: {word_list}. "
                            "These words carry negative connotations that the model strongly associates "
                            "with dissatisfied or critical reviews."
                        )
                    if pos_words:
                        word_list = ", ".join(
                            [f"**{w}**" for w, _ in pos_words])
                        st.markdown(
                            f"Words that pushed against **Negative** (toward Positive): {word_list}. "
                            "Despite these words, the overall sentiment remained negative."
                        )

        # suspicious review detection
        with st.expander("Suspicious Review Detection: Detailed Breakdown", expanded=True):

            if pred == 1:
                st.error(f"{label} Review")
            else:
                st.success(f"{label}")

            # Interpretation sentence
            st.markdown(
                f"The model is **{prob_pct}%** confident that this review is "
                f"**{likely} suspicious** as a result of the following features:"
            )

            # Detailed bullet point explanations per feature
            bullets = build_suspicious_bullets(features, rating)
            for bullet in bullets:
                st.markdown(bullet)

            # Feature value breakdown chart
            st.caption(
                "This chart shows the actual value each structured feature had for this specific review. "
                "Binary features (unverified_ratio, extreme_rating_ratio, very_short_review) are on a 0 to 1 scale , "
                "a value of 1 means the flag was fully triggered, 0 means it was not triggered, and 0.5 (shown in yellow) "
                "represents an unknown or neutral state, used when verification status is not provided. "
                "review_frequency reflects the reviewer's actual historical posting count and may exceed 1 "
                "when a reviewer ID is found in the training data."
            )

            st.plotly_chart(feature_breakdown_chart(
                features), use_container_width=True)

            # SHAP feature impact chart
            st.caption(
                "SHAP (SHapley Additive exPlanations) measures how much each feature "
                "contributed to pushing this specific prediction toward or away from suspicious. "
                "A red bar means that feature increased the suspicion score for this review. "
                "A teal bar means it reduced the suspicion score. "
                "The length of the bar shows how strong that influence was."
            )

            # shap_background: pre computed training sample used as SHAP baseline
            # stored in model_suspicious/shap_background.pkl
            shap_vals = explainer.shap_values(combined_input)
            shap_struct = shap_vals[0][-len(selected_features):]

            shap_df = pd.DataFrame({
                "Feature": selected_features,
                "Impact":  shap_struct
            }).sort_values("Impact", key=abs, ascending=False)

            st.plotly_chart(shap_bar_chart(
                shap_struct, selected_features), use_container_width=True)

            # Plain English interpretation of each SHAP value
            st.markdown(
                "#### What each feature contributed to this prediction")
            interpretations = interpret_shap_features(shap_df)
            for interp in interpretations:
                st.markdown(interp)


# Provenance:
# Streamlit app structure: https://docs.streamlit.io
# SHAP explainability: Lundberg and Lee (2017)
#   https://github.com/slundberg/shap
# Gauge chart: Plotly Graph Objects
#   https://plotly.com/python/gauge-charts/
# Attention based word importance adapted from:
#   https://huggingface.co/docs/transformers/en/main_classes/output
# Scipy sparse matrix combination (hstack, csr_matrix):
#   https://docs.scipy.org/doc/scipy/reference/sparse.html
