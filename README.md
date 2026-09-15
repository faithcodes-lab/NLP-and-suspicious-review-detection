# Sentiment Analysis and Suspicious Review Detection on Amazon Reviews
 
A machine learning system for sentiment analysis and suspicious review detection on Amazon product reviews, deployed as an interactive web application using Streamlit and Hugging Face Spaces.


## Project Overview

This project builds a dual-model pipeline that: 
1. Predict the **sentiment** (positive or negative) of amazon reviews product reviews using a fine-tuned DistilBERT transformer model.

2. Identify **suspicious or potentially fake reviews** using engineered behavioural, temporal, and linguistic features with a Logistic Regression classifier.  

3. Presents results through an interactive Streamlit web application with SHAP-based explainability and reviewer lookup 

## Dataset

The project uses the Amazon product Reviews dataset gotten from kaggle.
Due to its large size (8126.93 GB) the link to the dataset is provided below

[Dataset Link](https://www.kaggle.com/datasets/sahityasahu/amazon-review-dataset)


---            
##  Project Goals  
- Clean and prepare the product review text  
- Explore metadata such as rating, time, and verified ratio etc 
- Build and compare different machine learning model for sentiment classification  
- Build and compare different machine learning model for suspicious review detection  
- Add SHAP explainability to show the words or features that influenced predictions
- Deploy the final sentiment and suspicious review model in a simple Streamlit app
- Deploy the web app on a cloud platform so it is publicly accessible without needing to run the code locally.

---

## Project Pipeline

1. Data Preparation

The dataset goes through:

- Exploratory Data Analysis (ratings distribution, review length, class imbalance).

- Text cleaning: lowercasing, punctuation removal, stop word removal, standardisation

- Duplicate removal

- Metadata preparation: verified purchase status, vote count, reviewer ID


2. Label Creation (Weakly Supervised)

Suspicious labels were generated using a scoring system combining three independent signals:

- Sentiment–rating mismatch: predicted sentiment contradicts the star rating

- Extreme unverified: 1 or 5-star rating from an unverified purchase

- Burst review: review posted within 60 seconds of a previous review by the same user

3. Preprocessing

Two dedicated preprocessing notebooks were developed:

### Sentiment Preprocessing

- Sentiment labels derived from star ratings (≥ 4 = positive, < 4 = negative)

- Features: review text, review length, vote count, verified purchase status

- TF-IDF vectorisation: max_features=10,000, ngram_range=(1,2)
80/20 train/test split (random_state=42)
Shared files saved for reuse across model notebooks

### Suspicious Review Preprocessing

- Four feature groups: sentiment mismatch, behavioural, time-based, linguistic and  AI score

- 80/20 stratified split | TF-IDF (max_features=5,000) + raw text for BERT.

- Shared files saved for reproducibility across all model experiments	
---

4. 
| Category | Features |
|---|---|
| Behavioural | review_frequency, extreme_rating_ratio, unverified_ratio |
| Time-based | burst_review, review_frequency |
| Linguistic | very_short_review, exclamation_count, personal_pronoun_count, generic_word_flag, avg_word_length, lexical_diversity |
| AI likelihood | ai_probability_score |

5. Feature Ablation

A systematic feature ablation study was conducted for the suspicious review model, removing one feature at a time and measuring the impact on F1 and Recall for the suspicious class.

### Key findings:

- High impact: review_frequency, unverified_ratio, extreme_rating_ratio

- Harmful/noisy (removal improved performance): ai_probability_score, lexical_diversity

- Low signal: personal_pronoun_count, exclamation_count, generic_word_flag

Selected final features: review_frequency, unverified_ratio, extreme_rating_ratio, very_short_review

Class weight tuning ({0:1, 1:2}) and decision threshold tuning (best: 0.4) were applied based on ablation results.


## Machine Learning Models

Model 1: Sentiment Analysis

| Property | Detail |
|---|---|
| Architecture | DistilBERT (fine-tuned) |
| Task | Binary sentiment classification (Positive / Negative) |
| Accuracy | 90% |
| F1 Score (class 1) | 0.94 |
| F1 Score (class 0)  | 0.71|
| Recall | 0.90|
| Recall | 0.90 |


Model 2: Suspicious Review Detection
| Property | Detail |
|---|---|
| Architecture | Logistic Regression |
| Features |review_frequency, unverified_ratio, extreme_rating_ratio, very_short_review |
| Class weight | {0:1, 1:2} |
| Decision threshold | 0.4 |
| Accuracy | 94 % |
| F1 Score (class 1) | 0.47 (post-ablation) |
| Recall (class 1)| 0.53 (post-ablation) |

model 2 uses the sentiment scores from Model 1 as an input feature, creating a sequential pipeline.


### Web Application
Built with Streamlit, deployed on Hugging Face Spaces.
Features:

- Text input area with word count indicator

- Sentiment prediction with confidence gauge chart and plain-English interpretation

- Suspicious review detection with confidence gauge and structured feature breakdown

- SHAP horizontal bar chart showing feature impact direction and magnitude

- Reviewer ID lookup against 630,543 reviewers from training data, with transparent population median fallback when reviewer is unknown

- Sidebar with model information, dataset statistics, and team details

Development was structured across four iterative stages:

| Stage | Description |
|---|---|
| Stage 1 | Sentiment model only, running locally on localhost |
| Stage 2 | Both models integrated; first deployment to Hugging Face Spaces |
| Stage 3 | Explainability added: confidence gauges, SHAP analysis, plain-English outputs |
| Stage 4 | Model retraining with StandardScaler and filtered TF-IDF vocabulary; UI refinements |

**Live App:** [Open Web App](https://huggingface.co/spaces/Team4IGP/IGP_webApp)

**Hugging Face Repo:** [View Repository](https://huggingface.co/spaces/Team4IGP/IGP_webApp/tree/main)


### Explainability

SHAP (SHapley Additive exPlanations) is used to explain suspicious review predictions:

- Horizontal bar chart showing each feature's contribution direction and magnitude

- Plain-English interpretation for each feature's influence

- A representative sample of 100 training observations used as the SHAP background distribution


---

## Ethical Considerations

- No personal data is used

- The model may incorrectly flag genuine reviews; outputs should be treated as indicators, not verdicts

- SHAP explainability is included to improve transparency and support human oversight

- Dataset licence: CC BY-NC-SA 4.0

- This project is for academic use only


## Project Structure & Resources

Here are all key resources included in this repository:

- **Source Code (Python Scripts):**  
  [Code Folder](./Codes/)

- **Data / Processed Files:**  
  [Data Folder](./Data/)

- **Documents:**  
[Documents Folder](./Documents/) 

- **literature:**  
[Literature.md](./Literature.md/)

- **Meeting Minutes:**  
[Meeting Minutes Folder](./Meeting Minutes/) 

- **Project Flow Chart:**  
[Project_Flow_chart](./Project_Flow_chart.pdf/)

- **Project Process Documentation:**  
[Project Process Documentations Folder](./Project Process Documentation/) 

- **Proposal Document:**  
  [Open Proposal PDF](./Proposal/)



Feel free to explore each folder for detailed artefacts.

---


### Team members

- Faith Olan-George
- Khaing Su San
- Mateusz Marcinkowski
- Tanzim Siam
- Shayket Malaker
