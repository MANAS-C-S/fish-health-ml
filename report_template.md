# Assignment 2 — Build and Analyze a Machine Learning Model

## Title
**Fish Health Finding Classification Using Random Forest**

**Student:** ____________________  
**Course:** ____________________  
**Date:** ____________________

## 1. Problem Statement
This project explores whether machine learning can classify historical fish-health diagnostic findings using contextual information from public case records. The target is the recorded *Etiological Description*. Input features are limited to fields such as fish species, life stage, location and case date when available. The model predicts a recorded label; it is not intended to diagnose a live fish.

## 2. Dataset Description
**Source:** Fisheries and Oceans Canada, Fish Health Database  
**URL:** https://open.canada.ca/data/en/dataset/2ece9991-62aa-4b7a-bd7d-4f8f1052cd21  
**Licence:** Open Government Licence — Canada (verify the current source page).

The database contains historical diagnostic fish-health case records and pathogen findings from submissions in Canada's Pacific region. After running the code, fill in:
- Number of rows used: [from results_summary.csv]
- Number of columns: [inspect dataset]
- Target column: [from results_summary.csv]
- Input features: [from results_summary.csv]
- Number of target classes: [from results_summary.csv]

## 3. Preprocessing
The code reads the CSV using common encodings, cleans column names, removes rows without a target label, keeps target classes with at least ten records, imputes missing numerical and categorical features, and one-hot encodes categorical features. Case IDs and geographic coordinates are excluded. The target column is not used as an input feature.

## 4. Algorithm
A Random Forest Classifier is used. It combines multiple decision trees and can model non-linear relationships among categorical and numerical input features. A majority-class baseline is also calculated for comparison.

## 5. Training and Testing
The data is split into 75% training data and 25% test data using a fixed random seed and stratification. Preprocessing is fitted within a scikit-learn pipeline to reduce leakage from test data into training.

## 6. Evaluation Metrics
Use the actual outputs in `results_for_report.txt`:
- Accuracy
- Macro precision
- Macro recall
- Macro F1-score
- Weighted F1-score
- Per-class precision, recall and F1-score
- Confusion matrix

## 7. Graphs
Insert the generated files:
1. `plots/class_distribution.png`
2. `plots/confusion_matrix.png`
3. `plots/feature_importance.png`

## 8. Results and Analysis
**Complete this section only after running the code.** Compare Random Forest accuracy with the majority-class baseline. Discuss whether macro F1 is lower than weighted F1, which may indicate poorer performance on less common classes. Use the confusion matrix to identify commonly confused labels. Interpret feature importance as model association, not causal effect.

## 9. Limitations
The dataset consists of historical diagnostic submissions, so it may not represent all fish populations. Class imbalance can affect performance. Predicting recorded diagnostic labels from case context is not equivalent to diagnosing disease from fish measurements or images. The model is not a veterinary diagnostic tool.

## 10. Conclusion
[Write this after reviewing the actual metrics and graphs. Summarize performance honestly, state limitations, and suggest validation on independent cases as future work.]

## References
1. Fisheries and Oceans Canada. *Fish Health Database*. https://open.canada.ca/data/en/dataset/2ece9991-62aa-4b7a-bd7d-4f8f1052cd21
2. Scikit-learn documentation. https://scikit-learn.org/stable/
