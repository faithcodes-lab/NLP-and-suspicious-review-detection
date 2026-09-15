 ## Week 7: Sentiment Analysis Modelling   (issue #32) 
**Period:** 12 march – 16 March 2026  

# Project Process Documentation: Sentiment Analysis Models

## Overview
This project aimed to build and compare multiple machine learning and deep learning models for sentiment analysis on Amazon reviews. The goal was to evaluate model performance while addressing computational and data related challenges.

---

## Key Challenges and Solutions

### 1. Class Imbalance
**Challenge:**  
The dataset contained significantly more positive reviews than negative reviews, which could bias the model toward predicting the majority class.

**Solution:**  
The dataset was balanced by downsampling the majority class (positive reviews) to match the number of negative reviews. This ensured fair learning across both classes.

---

### 2. High-Dimensional Text Data
**Challenge:**  
Text data is unstructured and high dimensional, making it difficult for traditional machine learning models to process efficiently.

**Solution:**  
TF-IDF vectorization was used to convert text into numerical features. Additionally, metadata features were incorporated to assesment whether the model performs better or not.

---

### 3. Model Training Time

**Challenge:**  
Training deep learning models such as BERT on a large dataset (approximately 186k samples) resulted in long training times (up to 12 hours), especially on limited hardware (e.g., Apple devices without CUDA support).

**Solution:**  
- Reduced token length (`max_length=128`) to decrease computation
- Switched from BERT to DistilBERT for efficiency
- Used batch processing and optimized training parameters

This reduced training time significantly (from 12 hours to 5 hours).

---

### 4. Hardware Limitations

**Challenge:**  
Training on CPU or Apple MPS devices is slower compared to NVIDIA GPU environments.

**Solution:**  
The project was designed to be hardware compatible by:
- Using efficient models (DistilBERT)
- Avoiding device-specific configurations
- Ensuring code runs across different systems

---
### 5. Logistic Regression 

Logistic Regression was implemented  using TF-IDF features extracted from processed review text. An extended version also incorporated metadata features such as review length, vote count, and verified purchase status.

Challenge:
The dataset was highly imbalanced, with significantly more positive reviews than negative ones. This caused the model to achieve high accuracy while underperforming on negative reviews.

Solution:

Class weighting was introduced to give more importance to the minority class. This improved the recall of negative reviews and resulted in a more balanced model performance.

---
### 6. Support Vector Machine

A Support Vector Machine (LinearSVC) model was trained using the same feature configurations as Logistic Regression to ensure fair comparison.

Challenge:
Similar to Logistic Regression, the model initially favored the majority class, leading to lower recall for negative reviews.

Solution:
Class weighting was applied to improve minority class detection. SVM demonstrated strong performance on high dimensional TF-IDF features and achieved the best balance between accuracy and F1-score among traditional machine learning models.

---
### 7. Naive Bayes
Implemented a Naive Bayes sentiment analysis model using two experiments: TF-IDF text only, and TF-IDF text combined with metadata features (review_length, vote, verified). The notebook evaluates both models using accuracy, precision, recall, and F1-score, and saves the comparison results to a CSV file for submission.

---

### 8. Random Forest
The model was trained on the TF-IDF text data and was tried with different configurations of the decision trees depth and the number of decision trees. Due to the small amount of negative reviews the model with less trees had better accuracy but with the class weigthing the bigger models had better accuracy. The model though had much lower accuracy than the other models and due to the long computation the meta data + text TF-IDF was not tested.
