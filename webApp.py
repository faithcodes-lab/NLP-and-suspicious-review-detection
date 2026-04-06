import streamlit as st
import joblib
import numpy as np
import pandas as pd
import json
import shap
from scipy.sparse import hstack

# User interface configuration
st.set_page_config(
    page_title="Suspicious Review Detector",
    layout="centered"
)

st.title("Suspicious Review Detection")
st.markdown("Analyze whether a review is suspicious based on behavioural and textual patterns.")

# Load model, tfidf vectorizer and selected features
model = joblib.load("model_suspicious/logistic_model.pkl")
tfidf = joblib.load("model_suspicious/logistic_tfidf.pkl")

with open("new_features/selected_features.json", "r") as f:
    selected_features = json.load(f)

# Inputs UI 
with st.container():
    review = st.text_area("Enter a Review", height=150)

col1, col2 = st.columns(2)

with col1:
    rating = st.slider("Rating", 1, 5, 5)

with col2:
    verified = st.selectbox("Verified Purchase", ["Unknown", "Yes", "No"])

# Button setup for analysis 
if st.button("Analyze Review"):

    # Feature engineering based on input
    if verified == "Yes":
        unverified_ratio = 0
    elif verified == "No":
        unverified_ratio = 1
    else:
        unverified_ratio = 0.5

    extreme_rating_ratio = 1 if rating in [1, 5] else 0
    very_short_review = 1 if len(review.split()) < 20 else 0
    review_frequency = 1

    feature_dict = {
        "review_frequency": review_frequency,
        "unverified_ratio": unverified_ratio,
        "extreme_rating_ratio": extreme_rating_ratio,
        "very_short_review": very_short_review
    }

    structured_features = np.array([[feature_dict[f] for f in selected_features]])

    text_features = tfidf.transform([review])
    input = hstack([text_features, structured_features])

  
    # make predictions
    prob = model.predict_proba(input)[0][1]
    pred = model.predict(input)[0]

    label = "Suspicious" if pred == 1 else "Not Suspicious"

   # Display results

    st.markdown("### Prediction Result")

    if pred == 1:
        st.error(f"{label} Review")
    else:
        st.success(f"{label}")

    st.metric("Model Confidence", f"{prob*100:.1f}%")

    st.progress(float(prob))

    
    # Prediction explanation
    st.markdown("### Why this prediction?")

    explanations = []

    if unverified_ratio > 0.5:
        explanations.append("High unverified behaviour")

    if extreme_rating_ratio == 1:
        explanations.append("Extreme rating pattern")

    if very_short_review == 1:
        explanations.append("Short review length")

    if review_frequency == 1:
        explanations.append("Low review activity")

    for e in explanations:
        st.write(f"{e}")

    # Shap explanation for feature impact
    st.markdown("### Feature Impact (SHAP)")

    explainer = shap.Explainer(model)
    shap_values = explainer(input)

    # extract structured features
    shap_structured = shap_values.values[0][-len(selected_features):]

    shap_df = pd.DataFrame({
        "Feature": selected_features,
        "Impact": shap_structured
    })

    shap_df = shap_df.sort_values(by="Impact", key=abs, ascending=False)

    st.bar_chart(shap_df.set_index("Feature"))

    # -------------------------
    # Interpretation of shap values
    # -------------------------
    st.markdown("### Interpretation")

    top_feature = shap_df.iloc[0]

    st.write(
        f"The model's confidence is mainly influenced by **{top_feature['Feature']}**, "
        f"which had the strongest impact on this prediction."
    )

    
    # Disclaimer about feature estimation
    st.info("Note: Some behavioural features are estimated due to lack of user history.")