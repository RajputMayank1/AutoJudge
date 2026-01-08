# AutoJudge: Predicting Programming Problem Difficulty

### An NLP-based Machine Learning System to Predict Problem Difficulty Class and Score

**Tasks:**
- Difficulty Classification → Easy / Medium / Hard  
- Difficulty Regression → Numerical Difficulty Score  

**Approach:** Text-based feature engineering + classical ML models  
## Problem Motivation

Online competitive programming platforms such as Codeforces, Kattis, and CodeChef assign
difficulty levels and scores to problems.  
However, this process often relies on **manual judgment and user feedback**, which can be subjective
and inconsistent.

The goal of this project is to build an **automated system** that can predict:
- the difficulty **class**
- the difficulty **score**

using **only the textual description of the problem**.
## Dataset Description

Each data sample represents a programming problem and contains the following fields:

- **Title**
- **Problem Description**
- **Input Description**
- **Output Description**
- **Difficulty Class** (Easy / Medium / Hard)
- **Difficulty Score** (numerical value)

Missing values are present in some text fields and are handled during preprocessing.
## Project Objectives

The main objectives of this project are:

1. Predict problem difficulty class (Easy / Medium / Hard)
2. Predict problem difficulty score (regression)
3. Use **only textual information**
4. Compare multiple machine learning models
5. Deploy the final system via a simple web interface
## Data Preprocessing

The preprocessing steps include:

- Handling missing values in text fields
- Merging all textual fields into a single representation
- Converting text to lowercase
- Removing unnecessary characters while **preserving mathematical symbols**

A single column `combined_text` is created to represent the full problem context.
## Feature Engineering

Two types of features are extracted:

### 1. TF-IDF Features
- Unigrams and bigrams
- Capture semantic meaning of the problem text
- Suitable for high-dimensional sparse text data

### 2. Hand-crafted Numerical Features
- Text length
- Word count
- Mathematical symbol count
- Algorithmic keyword frequencies (dp, graph, dfs, bfs, recursion, etc.)

The final feature vector is a **combination of TF-IDF and structural features**.
## Difficulty Classification Models

The following models were evaluated for predicting difficulty class:

- Dummy baseline (always predicts "Hard")
- Logistic Regression
- Random Forest Classifier
- **Linear Support Vector Machine (SVM)**

Linear SVM performed best due to its effectiveness on
high-dimensional sparse TF-IDF features.
## Classification Results

| Model | Accuracy |
|-----|---------|
| Dummy Baseline (Always Hard) | ~0.47 |
| Logistic Regression | ~0.41 |
| Random Forest | ~0.40 |
| **Linear SVM (Final)** | **~0.50+** |

Medium difficulty problems are the hardest to classify due to overlap
with both Easy and Hard categories.
## Difficulty Score Regression Models

The following regression models were evaluated:

- Mean baseline predictor
- Linear Regression
- Random Forest Regressor
- **Gradient Boosting Regressor**

Gradient Boosting achieved the lowest prediction error
by modeling non-linear relationships.
## Regression Results

| Model | MAE (Lower is Better) |
|----|----|
| Mean Baseline | ~1.3 |
| Linear Regression | ~1.9 |
| Random Forest Regressor | ~1.7 |
| **Gradient Boosting Regressor (Final)** | **~1.6** |
## Key Insights

- Difficulty prediction is inherently noisy due to overlapping problem structures
- Medium problems are linguistically ambiguous
- Linear SVM is well-suited for sparse text classification
- Gradient Boosting performs best for numerical difficulty estimation
- Structural features significantly improve performance
## Web Interface

A web-based interface was built using **Gradio**.

Users can:
- Paste a problem description
- Paste input and output descriptions
- Instantly receive:
  - Predicted difficulty class
  - Predicted difficulty score

This enables real-time usage of the trained models.
## Conclusion

This project demonstrates that programming problem difficulty can be
reasonably predicted using **only textual information**.

By combining NLP techniques with classical machine learning models,
the system achieves strong performance and practical usability.

Future improvements may include:
- Ordinal classification techniques
- Deeper semantic features
- Cross-platform deployment
# Mayank Rajput 21324016
# Link TO demo Video
```
https://drive.google.com/drive/folders/1RiCa5UEClGj5y4e1gVOXnaB4U5n6W3p_?usp=sharing
```
