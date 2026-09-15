# Overview (issue #14)

Before performing sentiment analysis and suspicious review detection, the Amazon reviews dataset required systematic cleaning and validation. Raw user generated datasets frequently contain missing values, inconsistent formats, redundant variables, and noise that can negatively affect downstream analysis and machine learning performance.

The objective of the data cleaning phase was therefore to:

- Ensure dataset completeness and consistency
- Remove unusable observations
- Standardize variables
- Handle missing values appropriately
- Prepare features required for modelling

The final outcome of this process is a structured dataset suitable for natural language processing.

---

## Processing Environment and Workflow Strategy

## Initial Processing Using DuckDB

The original dataset contained *millions of records*, making it computationally inefficient to load directly into memory using traditional Python data processing libraries.

To address this limitation, the early stages of data inspection and preprocessing were conducted using *DuckDB*, an in process analytical database designed for efficient querying of large datasets.

DuckDB provides several advantages when working with large data:

- Efficient SQL based querying
- Ability to process large datasets without fully loading them into memory
- Fast aggregation and filtering operations
- Seamless integration with Python environments

This made DuckDB suitable for performing early tage tasks such as:

- inspecting dataset structure
- checking column distributions
- identifying missing values
- performing large scale filtering operations

Using DuckDB allowed the dataset to be explored and partially cleaned while maintaining efficient memory usage.

---

## Dataset Size Reduction

During the EDA, tope 30 most reviewed products were identified and selected  reducing the data drastically. In the cleaning phase, several operations were performed that slightly removed reduced the dataset size :

- removal of unnecessary columns
- elimination of empty review texts
- removal of exact duplicate records
- filtering unusable observations

overall, these steps reduced the dataset from *millions of rows to a smaller subset of several hundred thousand observations*, making it manageable for in-memory processing.

---

## Transition to Python Pandas

After the dataset size had been reduced, the workflow transitioned from DuckDB to *Python's Pandas library*.

Pandas is widely used in data science workflows because it provides:

- flexible data manipulation
- efficient feature engineering
- strong integration with machine learning libraries
- convenient support for text processing

This transition allowed the project to perform more detailed operations including:

- feature engineering
- text preprocessing
- creation of review length indicators
- preparation of the dataset for sentiment analysis models

The project therefore adopted a *hybrid processing strategy*:

1. *DuckDB* for scalable querying and early stage filtering  
2. *Python Pandas* for detailed cleaning and feature engineering

This approach ensured that the dataset could be processed efficiently without exceeding memory limitations.

---

# Dataset Overview

The dataset contains approximately *829,621 observations and 11 variables* representing customer reviews collected from Amazon product categories.

Key variables include:

| Variable | Description |
|--------|--------|
| asin | Unique identifier for each product |
| category | Product category |
| reviewerID | Unique identifier for each reviewer |
| reviewerName | Display name of reviewer |
| overall | Star rating assigned to the product |
| summary | Short review title |
| reviewText | Full textual review |
| reviewTime | Date of review |
| unixReviewTime | Timestamp |
| verified | Indicates verified purchase |
| vote | Number of helpful votes received |

Since the primary focus of the analysis is *review sentiment and suspicious review detection, particular attention was given to cleaning the **reviewText, **vote*, and reviewer related fields.

---

# Initial Data Inspection

The first stage involved exploring the structure of the dataset through:

- dataset shape inspection
- column name validation
- missing value detection
- verification of column data types

This ensured that all expected fields were present and that no structural corruption existed within the dataset.

---

# Missing Value Analysis

A dataset wide inspection was conducted to identify:

- NULL values
- empty strings
- whitespace only entries

Particular attention was given to *text fields*, since user generated text often contains irregular formatting.

The columns *asin, **category, **reviewerID, and **reviewTime* were found to contain no missing values. These variables serve as identifiers and timestamps and therefore required no intervention.

Minor inconsistencies were identified in the *reviewerName, **reviewText, and **vote* columns.

---

# Handling Missing Reviewer Names

Some records contained missing or blank values in the *reviewerName* column.

### Decision

Missing reviewer names were replaced with the placeholder value:
“Unknown”

### Justification

Removing these rows would unnecessarily reduce dataset size because:

- *reviewerID already uniquely identifies each reviewer*

- reviewerName is descriptive but not analytically critical

This approach preserves observations while maintaining consistent formatting.

### Alternative Considerations

Two alternative strategies were considered:

| Option | Reason Not Used |
|------|------|
| Delete rows | Would reduce dataset size unnecessarily |
| Impute names | No reliable method exists to infer missing reviewer names |

---

# Removal of the Summary Column

The *summary* column contains short titles or brief descriptions of reviews.

### Decision

The column was removed from the dataset.

### Justification

Several factors motivated this decision:

1. The summary often duplicates sentiment already present in the review text.
2. The *reviewText column contains richer contextual information*.
3. Retaining both variables may introduce redundant information during natural language processing.

Research in sentiment analysis suggests that full review texts generally provide stronger predictive signals than short summaries (Pang and Lee, 2008).

---

# Cleaning the Review Text Column

The *reviewText* column represents the primary textual content used for sentiment analysis.

### Identified Issues

Some records contained:

- NULL values
- empty strings
- whitespace only entries

These records contain no semantic information.

### Decision

Rows where reviewText was:

- NULL
- empty
- whitespace only

were removed from the dataset.

### Justification

Machine learning models require meaningful textual content to extract linguistic features such as:

- sentiment indicators
- token frequency patterns
- contextual expressions

Removing empty entries reduces noise and improves model reliability.

---

# Cleaning the Vote Column

The *vote* column represents the number of users who marked a review as helpful.

However, values were inconsistently formatted:
1,245
2,000
15

The comma separators caused the column to be interpreted as *text rather than numeric data*.

### Cleaning Procedure

1. Convert the column to string format
2. Remove whitespace characters
3. Remove comma separators
4. Convert the cleaned values to numeric format then integer.

### Justification

Standardizing the vote column allows it to be used for:
- helpfulness metrics
- suspicious review detection

---

# Duplicate Detection and Removal

Duplicate records were investigated to ensure that repeated observations did not bias the dataset.

The first stage involved identifying *exact duplicate rows* across all variables.

Exact duplicates may occur during data aggregation or web crawling processes when datasets are compiled from multiple sources.

### Decision

Exact duplicate rows were removed.

### Justification

Removing exact duplicates ensures that:

- each observation represents a unique review
- review counts are not artificially inflated
- downstream analysis is not distorted by repeated entries

---

# Investigation of Identical Timestamp Reviews

Following the removal of exact duplicates, further inspection revealed a pattern where multiple reviews shared the *same timestamp* while containing *different review texts and metadata*.

At first glance, this could suggest suspicious behaviour such as coordinated review bursts.

However, detailed inspection showed that:

- review text differed

Because the rows were not identical across fields, they were *not classified as duplicates*.

Instead, these observations likely represent *multiple snapshots of reviews captured during the web crawling process*, where several reviews were recorded at the same moment during dataset collection.

---

# Burst Behaviour Consideration

Identical timestamps can sometimes indicate *burst behaviour*, where many reviews are posted within a short period.

Such bursts may suggest:

- coordinated promotional campaigns
- spam activity
- manipulation of product ratings

However, the timestamp patterns observed during analysis did not display typical burst characteristics such as:

- repeated reviewer accounts
- identical review texts
- identical ratings posted simultaneously

Therefore, the observations were *not classified as burst behaviour*.

---

# Decision: Retaining Timestamp Matches

Since these rows contained *distinct textual content and metadata*, removing them would risk discarding legitimate information.

For this reason, these records were *retained in the dataset* and interpreted as potential *data collection artifacts rather than genuine duplicate records*.

---

# Feature Engineering

Additional variables were created to support downstream modelling tasks.

---

## Helpful Vote Indicator

A binary feature *has_vote* was created:
has_vote = 1 if vote > 0
has_vote = 0 otherwise

### Purpose

This variable captures whether a review received user engagement, which may help identify patterns related to:

- review credibility
- helpfulness signals
- potential manipulation.

---

## Review Length Features

Three additional variables were derived from the review text.

| Feature | Description |
|------|------|
| review_length | Number of characters in the review |
| word_count | Total number of words |
| short_review_flag | Indicator for extremely short reviews |

### Motivation

Review length features are commonly used in opinion mining research because extremely short reviews may indicate:

- spam
- automated content
- low informational value

Studies in opinion mining have shown that textual length can influence sentiment classification and credibility detection (Liu, 2012).

---

# Data Validation

After all cleaning steps were completed, the dataset was validated to confirm:

- no remaining missing values
- no empty textual fields
- consistent data types across variables
- removal of exact duplicate records

These checks ensured that the dataset was fully prepared for downstream analytical tasks.

---

# Outcome

The final cleaned dataset is:

- structurally consistent
- free of exact duplicate records
- free of empty review texts
- standardized across key variables
- enriched with additional analytical features

# Text Preprocessing

## Overview

After the structural data cleaning phase was completed, a second stage of processing was performed to prepare the review text for natural language processing and machine learning analysis.

User generated review text often contains noise such as:

- punctuation
- stop words
- inconsistent capitalization
- unnecessary symbols

These elements can introduce noise into machine learning models and reduce the effectiveness of sentiment analysis algorithms.

The goal of the text preprocessing phase was therefore to standardize the textual data and extract meaningful linguistic information from the review content.

---

# Stop Word Definition

Stop words are commonly occurring words that typically carry little semantic meaning in text analysis.

Examples include:
the, is, and, to, of

These words appear frequently across texts but usually do not contribute meaningful information for sentiment classification.

A stop word list was defined and used during preprocessing to remove these terms from the review text. Removing stop words helps:

- reduce dimensionality of the text data
- improve computational efficiency
- highlight more meaningful words for modelling

---

# Text Cleaning Function

A custom cleaning function was implemented to normalize the review text before applying advanced NLP processing.

The function performs several basic preprocessing steps:

- converting text to lowercase
- removing punctuation
- removing special characters
- removing extra whitespace
- standardizing text formatting

These steps ensure that words such as:
Good
good
GOOD
are treated as the same token during analysis.

This normalization step improves the consistency of textual features extracted during later modelling stages.

---

# Basic Text Cleaning Application

The cleaning function was applied to the review text column to generate a cleaned version of the dataset.

The cleaned text column removes unnecessary formatting while preserving the semantic content of each review.

This step ensures that the dataset contains standardized textual input before performing linguistic processing.

---

# spaCy Based Text Processing

To further prepare the text for analysis, the *spaCy natural language processing library* was used.

spaCy provides efficient tools for:

- tokenization
- lemmatization
- linguistic feature extraction

A preprocessing function was defined using spaCy to transform the cleaned review text into a normalized representation suitable for machine learning models.

---

# Lemmatization

One of the key steps performed using spaCy was *lemmatization*.

Lemmatization reduces words to their base or dictionary form.

Examples include:

| Original Word | Lemma |
|---------------|-------|
| running | run |
| better | good |
| bought | buy |

This process helps group similar word variations under a single representation, improving the effectiveness of text analysis.

For example, the words:
run, running, ran
are treated as the same concept after lemmatization.

---

# Generation of Processed Text Column

After spaCy preprocessing was applied, a new column containing the processed review text was created.

This processed column contains:

- cleaned tokens
- normalized word forms
- reduced linguistic noise

The resulting text representation is better suited for downstream tasks such as:

- sentiment classification
- suspicious review detection
- feature extraction for machine learning models.

---

# Validation of Preprocessed Text

After preprocessing, the resulting text columns were inspected to confirm that:

- punctuation and noise were removed
- stop words were filtered appropriately
- tokens were successfully lemmatized
- meaningful textual information remained intact

This validation ensured that the preprocessing pipeline produced a clean and linguistically meaningful dataset for further analysis.

---

# Outcome

The complete preprocessing workflow produced a dataset that is:

- structurally cleaned
- free of duplicate records
- standardized across key variables
- enriched with engineered features
- linguistically normalized for natural language processing

The final dataset is therefore suitable for the next stages of the project:

1. Sentiment analysis modelling  
2. Suspicious review detection  
3. Machine learning model training
---

## Pipeline Overview

The overall workflow can be summarized as:

- Raw Dataset  
 
- Large-scale inspection (DuckDB)  
  
- Structural data cleaning  
 
- Duplicate detection and validation  
 
- Dataset size reduction  
 
- Transition to Python Pandas  
  
- Text preprocessing (spaCy)  
 
- Feature engineering  
 
-  Analysis ready dataset

# References

Liu, B. (2012) Sentiment Analysis and Opinion Mining [online]. 1st ed. San Rafael, California: Morgan & Claypool Publishers. [Accessed 05 March 2026].

Pang, B. and Lee, L. (2008) Opinion Mining and Sentiment Analysis. Foundations and Trends in Information Retrieval [online]. 2 (2), p. 12. [Accessed 05 March 2026].