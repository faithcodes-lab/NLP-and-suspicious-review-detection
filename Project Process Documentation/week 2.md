## Week 2: New Dataset Selection and Niche Identification (issue #11 )   
**Date:** Tuesday, 3 February 2026 to 10 february 2026
---

# 1. Following the meeting with supervisor after week 1 task , the team was assigned the following tasks:

- Continue searching for a suitable and accessible dataset.
- Research and document ground truth labeling strategies.
- Record project progress and decisions.

---

# 2. Task Assigned for Week 2

**Task 1:**  
Find a product review dataset in a specific niche for sentiment analysis and fake review detection.

## Task Requirements

The dataset needed to meet the following criteria:

- Product reviews (not app reviews)
- Clearly defined niche or category
- Sufficient data volume for analysis
- Recent review data
- Suitable structure for manual labeling

---

# 3. Dataset Search and Evaluation

Multiple datasets were identified and evaluated based on their relevance to the project objectives.

---

# 3.1 Amazon Alexa Review Dataset

**Dataset Source:** Kaggle  
**Author:** Sahitya Sahu  

Dataset Link:  
https://www.kaggle.com/datasets/sahityasahu/amazon-review-dataset  

## Dataset Description

This dataset contains large scale Amazon product reviews (~123GB).

The dataset includes millions of customer reviews along with multiple metadata fields.

Each review contains:

- ASIN (product identifier)
- Reviewer ID and reviewer name
- Star rating
- Review summary
- Full review text
- Review timestamps
- Verified purchase indicator
- Helpful vote counts

Due to the large size of the dataset, the JSON files were **converted into CSV batches** to improve readability and facilitate data processing.

---

## Suitability for Manual Labeling

The dataset is suitable for manual labeling because:

- Reviews contain **natural and detailed text**
- Data can be **sampled into smaller subsets** for annotation
- Star ratings can serve as **weak sentiment labels**
- Metadata supports **fake review detection analysis**

Reviewer behavior analysis can be conducted using fields such as:

- Reviewer identifiers
- Review timestamps
- Star ratings
- Verified purchase indicators
- Helpful vote counts

These fields enable analysis of behavioral patterns such as:

- Review frequency
- Rating consistency
- Verified purchase ratios
- Repeated reviewing behavior.

---

# 3.2 Amazon Product Reviews Dataset

Dataset Source: Kaggle  

Dataset Link:  
https://www.kaggle.com/datasets/gzdekzlkaya/amazon-product-reviews-dataset/data

## Category

General Amazon consumer product reviews.

---

## Justification

This dataset was considered because:

- It contains authentic customer reviews
- Reviews are unlabeled, making them suitable for manual annotation
- Dataset size is manageable for experimentation.

---

## Limitations

Some limitations were identified:

- Limited metadata compared to larger Amazon datasets
- May lack important indicators such as:
  - Verified purchase flags
  - Detailed reviewer history
  - Behavioral signals.

Additionally, the dataset contains approximately **4,900 reviews**, which may limit the performance of larger machine learning models.

---

# 3.3 Amazon Reviews 2023 Dataset

Dataset Source:  
https://amazon-reviews-2023.github.io/

## Category

Electronics product reviews.

---

## Justification

This dataset provides:

- Large-scale review data
- Recent reviews
- Rich metadata for behavioral analysis

It also allows filtering by specific product categories, enabling the definition of a clear niche.

---

# 3.4 Amazon Cell Phone Reviews Dataset

Dataset Link:  
https://www.kaggle.com/datasets/grikomsn/amazon-cell-phones-reviews

## Category

Mobile phones and related electronics.

---

## Limitations

Although the dataset includes useful metadata such as verified purchase indicators, it is relatively older, and some product links contained in the dataset are no longer functional.

---

# 4. Final Dataset Selection

After presenting the dataset options and discussing them with the supervisor, the team decided to proceed with the **Amazon Alexa Review Dataset**.

---

## Selected Dataset

**Dataset Source:** Kaggle  
**Author:** Sahitya Sahu  

Dataset Link:  
https://www.kaggle.com/datasets/sahityasahu/amazon-review-dataset

---

## Reasons for Selection

The dataset was selected because it satisfies the key project requirements:

- Contains real-world Amazon product reviews
- Includes large-scale and recent review data
- Provides rich metadata useful for fake review detection

The presence of fields such as **verified purchase indicators, reviewer IDs, timestamps, ratings, and helpful votes** allows the extraction of behavioral patterns commonly used in suspicious review detection research.

Additionally, the dataset can be **sampled and converted into manageable subsets**, making it practical for exploratory analysis and manual labeling.

---

# 5. Task 2 – Research Ground-Truth Labeling Strategies for Suspicious Reviews

- Conducted literature review to identify evidence-based labeling strategies for fake/suspicious reviews.

- Investigated gold-standard datasets (Ott et al., 2011) where fake reviews were intentionally written by crowd workers, providing true ground-truth labels.

- Reviewed behavioral heuristic approaches (Jindal & Liu, 2008; Mukherjee et al., 2013) that detect fake reviews using signals such as duplicate reviews, burst reviewing, abnormal reviewer activity, and extreme ratings.

- Examined verified purchase metadata as a weak proxy for genuine reviews, where verified purchases are more likely to represent authentic user experiences.

- Explored **synthetic fake review generation** methods using language models (e.g., GPT-2) to create balanced datasets for fake review detection.

- Reviewed semi-supervised approaches where a small manually labeled dataset is used to generate labels for a larger unlabeled dataset.

- Identified distant supervision for sentiment labeling, where star ratings are used as proxy labels (1–2 stars = negative, 4–5 stars = positive; 3 stars excluded).

- Proposed initial annotation guidelines based on linguistic cues (generic text, exaggerated language) and behavioral cues (non-verified purchase, extreme ratings, short reviews, repeated reviewer activity).

- Task remains open as further literature review and refinement of labeling criteria will continue throughout the project.


# 6. Outcome of Week 2

By the end of Week 2, the team achieved the following:

- Identified and evaluated multiple product review datasets
- Selected the Amazon Review Dataset for the project
- Researched academic approaches for ground-truth labeling of suspicious reviews
- Proposed initial labeling guidelines based on linguistic and behavioral signals.

The ground-truth labeling research task remains open, as additional literature will continue to be reviewed as the project progresses.