
# Student Grade Predictor

## Problem Statement
Build an interactive Streamlit app to compare multiple classification models for predicting student grades using the UCI Student Performance dataset. The app should allow users to select the dataset, target column, and model, and display evaluation metrics, confusion matrix, and classification report.

## Dataset Description
This app uses the UCI Student Performance dataset, which contains student achievement data from secondary education in Portuguese schools. The data includes student grades, demographic, social, and school-related features collected from school reports and questionnaires. Two datasets are provided: one for Mathematics (student-mat.csv) and one for Portuguese language (student-por.csv). The target attributes (G1, G2, G3) are period grades, with G3 as the final grade. G3 is highly correlated with G1 and G2, as it is the final year grade, while G1 and G2 are from earlier periods. Predicting G3 without G1 and G2 is more challenging but more useful for real-world applications.

## Models Used
- Logistic Regression
- Decision Tree Classifier
- K-Nearest Neighbor Classifier
- Naive Bayes Classifier
- Random Forest (Ensemble)
- XGBoost (Ensemble)

## Comparison Table for student-mat.csv (for target = G3)
| ML Model Name        | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---------------------|----------|-----|-----------|--------|----|-----|
| Log Classifier      | 0.28 | Not computable | 0.34 | 0.28 | 0.27 | 0.22 |
| Decision Tree       | 0.33 | Not computable | 0.34 | 0.33 | 0.32 | 0.28 |
| kNN                 | 0.10 | Not computable | 0.06 | 0.10 | 0.07 | 0.03 |
| Naive Bayes         | 0.15 | Not computable | 0.15 | 0.15 | 0.12 | 0.11 |
| Random Forest       | 0.38 | Not computable | 0.33 | 0.38 | 0.34 | 0.33 |
| XGBoost             | 0.39 | Not computable | 0.41 | 0.39 | 0.38 | 0.34 |

## Comparison Table for student-por.csv (for target = G3)
| ML Model Name        | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---------------------|----------|-----|-----------|--------|----|-----|
| Log Classifier      | 0.27 | Not computable | 0.29 | 0.27 | 0.27 | 0.19 |
| Decision Tree       | 0.42 | Not computable | 0.43 | 0.42 | 0.41 | 0.35 |
| kNN                 | 0.18 | Not computable | 0.23 | 0.18 | 0.19 | 0.08 |
| Naive Bayes         | 0.10 | Not computable | 0.30 | 0.10 | 0.09 | 0.07 |
| Random Forest       | 0.45 | Not computable | 0.46 | 0.45 | 0.44 | 0.39 |
| XGBoost             | 0.45 | Not computable | 0.44 | 0.45 | 0.43 | 0.38 |


## Observations
| ML Model Name        | Observation about model performance |
|---------------------|-------------------------------------|
| Logistic Regression | Performs poorly due to many classes and linear boundaries. |
| Decision Tree       | Slightly better, but prone to overfitting and unstable splits. |
| kNN                 | Very low accuracy; struggles with high class cardinality. |
| Naive Bayes         | Performs poorly; strong independence assumptions not met. |
| Random Forest       | Best among all; handles multiclass and nonlinearity well. |
| XGBoost             | Comparable to Random Forest; robust to class imbalance. |

## How to Use
1. **Select Dataset:**
   - Use the sidebar to choose a dataset (student-mat.csv or student-por.csv).
2. **Read Dataset Info:**
   - The app displays a summary of the dataset and explains the target columns (G1, G2, G3).
3. **Select Target Column:**
   - Choose which grade (G1, G2, or G3) you want to predict.
4. **Select Model:**
   - Pick a classification model from the dropdown.
5. **Run Models:**
   - Click "Run Models" to train and evaluate the selected model. Results update automatically when you change the model or target.
6. **View Results:**
   - See evaluation metrics, confusion matrix, and classification report for the selected model and target.

## Notes
- The app automatically encodes categorical features and scales numeric features.
- All metrics are calculated for multiclass targets.
- If you encounter an AUC error, the app will display 'N/A' for AUC.

## Requirements
See requirements.txt for dependencies.
