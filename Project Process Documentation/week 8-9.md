### Week 7: Suspicious review modelling  (issue #41)

Period: 19 march – 31 March 2026


### Overview

This project aimed to build and compare multiple machine learning and deep learning models for ssuspicious review detectio on Amazon reviews. The goal was to evaluate model performance while addressing computational and data related challenges.

--- 

## Preprocessing Overview

The preprocessing stage involved transforming the raw dataset into a structured format for suspicious review detection by generating behavioural, linguistic, and sentiment based features, alongside a weakly supervised suspicious label derived from multiple signals.

During this process, a key challenge was ensuring the reliability of the generated label while avoiding data leakage. Some features used in constructing the suspicious score risked indirectly revealing the target variable, which could lead to inflated model performance. To address this, careful feature selection was applied, and any potentially leaking or redundant features were removed, ensuring that the final dataset remained both meaningful and generalisable.

Additionally, the dataset exhibited class imbalance, which was handled at the modelling stage using class weights rather than altering the data distribution.

The pipeline was also structured to ensure consistency and reproducibility by separating preprocessing from modelling and saving all outputs for reuse.

Overall, the preprocessing pipeline was designed to be structured, reproducible, and aligned with the requirements of both traditional machine learning models and transformer-based approaches.

---

## Feature Ablation Analysis

### Objective  
Feature ablation was conducted to evaluate the contribution of each feature to the model’s performance.

---

### Baseline Model  

A baseline Logistic Regression model was first trained using all selected features

The performance of this model (F1-score and recall) was used as a reference point.

---

### Ablation Procedure  

Feature ablation was performed by removing one feature at a time and retraining the model using the remaining features.

For each iteration:

1. One feature was removed  
2. The model was retrained using the remaining features  
3. Performance metrics were recorded  

This process was repeated for all features.

---

### Evaluation Metrics  

The following metrics were used to evaluate performance:

- F1-score  
- Recall  

These metrics were chosen due to their importance in handling imbalanced classification problems.

---

### Comparison with Baseline  

Each ablation result was compared with the baseline model to assess the impact of removing the feature.

- A significant drop in performance indicates that the feature is important  
- Little or no change suggests the feature has minimal impact  
- An improvement suggests the feature may introduce noise  

---

### Key Findings  

The analysis showed that:

- Behavioural features such as review_frequency and unverified_ratio had a strong impact on performance  
- Some features had minimal contribution  
- Certain features reduced performance when included, indicating possible redundancy or noise  

---

### Conclusion  

Feature ablation demonstrated that the model relies heavily on a subset of key features. This highlights the importance of feature selection in improving model performance and reducing unnecessary complexity.

---
## Logistics Regression

Implemented a Logistic Regression model using TF-IDF text features and engineered features. Combined features, trained the model, evaluated performance using accuracy and F1-score, and saved the results for comparison with other models.

## Second Phase (Retraining)

### Model Training  

A Logistic Regression model was trained using the selected structured features from features ablation. The model was chosen due to its simplicity, interpretability, and effectiveness for structured data.

To improve performance on the imbalanced dataset, class weighting was introduced.

---

### Class Weight Tuning  

Different class weight configurations were tested to address class imbalance and improve detection of the minority class (suspicious reviews).

The following weight configurations were evaluated:

- {0:1, 1:1}  
- {0:1, 1:2}  
- {0:1, 1:3}  
- {0:1, 1:4}  

For each configuration, the model was trained and evaluated using prediction scores.

---

### Threshold Tuning  

Instead of relying on the default classification threshold (0.5), multiple thresholds were tested to optimise performance.

The following thresholds were evaluated:

- 0.3  
- 0.4  
- 0.5  
- 0.6  
- 0.7  

Predicted probabilities were converted into class labels using these thresholds.

---

### Model Evaluation  

For each combination of class weight and threshold, the following metrics were calculated:

- Precision  
- Recall  
- F1-score  

Special focus was placed on F1-score and recall, as they are more appropriate for imbalanced classification problems.

---

### Best Model Selection  

The optimal configuration was selected based on the highest F1-score while maintaining reasonable recall.

The best-performing model was identified as:

- Class weights: {0:1, 1:4}  
- Threshold: 0.3  

This configuration achieved the best balance between detecting suspicious reviews and limiting false positives.

---

### Final Model  

The final Logistic Regression model was retrained using the selected optimal parameters. The model was then used for further evaluation, feature importance analysis, and comparison with other models.

---


### Support Vector machine
The model was trained and tested on TF-IDF text and meta data like ai probability, generic words, very short review. The model got 86.2% accuracy score using the class weigth balanced due to big amount of reviews labeled genuine compared to the reviews labeled as fake.

---


### BERT