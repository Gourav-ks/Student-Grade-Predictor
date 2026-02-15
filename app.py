

import streamlit as st
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

from model.data_utils import load_data, preprocess_data, split_data
from model.logistic_regression_classifier import LogisticRegressionClassifier
from model.decision_tree_classifier import DecisionTreeClassifierModel
from model.knn_classifier import KNNClassifier
from model.naive_bayes_classifier import NaiveBayesClassifier
from model.random_forest_classifier import RandomForestClassifierModel
from model.xgboost_classifier import XGBoostClassifier

import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Student Grade Predictor", layout="wide")
st.title("Student Grade Predictor")

#  Dataset Selection 
st.sidebar.header("Dataset Selection")
dataset_dir = os.path.join(os.path.dirname(__file__), "student_dataset")
csv_files = [f for f in os.listdir(dataset_dir) if f.endswith('.csv')]
dataset_file = st.sidebar.selectbox("Select Dataset", csv_files)
uploaded_file = os.path.join(dataset_dir, dataset_file) if dataset_file else None


if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, delimiter=';')
    st.markdown("""
    **About the Student Performance Dataset:**
    - Each row is a student record from Portuguese or Math courses.
        - student-mat.csv - Math course
        - student-por.csv - Portuguese course
    - Features: 30 attributes (demographics, family, school, etc.)
    - **Potential targets:**
        - **G1**: First period grade (0-20)
        - **G2**: Second period grade (0-20)
        - **G3**: Final grade (0-20, most commonly used for prediction)
    - See sidebar to select which target to predict.
    """)
    st.write("### Preview of Dataset", df.head())
    #  Sidebar selections 
    target_column = st.sidebar.selectbox(
        "Select Target Column",
        [col for col in df.columns if col in ["G1", "G2", "G3"]],
        index=["G1", "G2", "G3"].index("G3") if "G3" in df.columns else 0
    )
    model_options = {
        "Decision Tree": DecisionTreeClassifierModel,
        "KNN Classifier": KNNClassifier,
        "Naive Bayes": NaiveBayesClassifier,
        "Random Forest": RandomForestClassifierModel,
        "XGBoost": XGBoostClassifier,
        "Log Classification": LogisticRegressionClassifier,
    }
    model_name = st.sidebar.selectbox("Select Model", list(model_options.keys()))
    if st.sidebar.button("Run Models") or True:
        
        from sklearn.preprocessing import LabelEncoder
        y_raw = df[target_column]
        le_target = LabelEncoder()
        y_encoded = le_target.fit_transform(y_raw)
        X = df.drop(target_column, axis=1)
        from model.data_utils import preprocess_data, split_data
        X_scaled, _, label_encoders, scaler = preprocess_data(df, target_column)
        X_train, X_test, y_train, y_test = split_data(X_scaled, y_encoded)
        ModelClass = model_options[model_name]
        if model_name == "kNN":
            model = ModelClass(X_train, y_train, n_neighbors=5)
        elif model_name == "Random Forest":
            model = ModelClass(X_train, y_train, n_estimators=100)
        else:
            model = ModelClass(X_train, y_train)
        model.train()
        all_labels = np.unique(y_train)
        from sklearn.preprocessing import label_binarize
        def safe_evaluate(model, X_test, y_test, all_labels):
            import math
            metrics = model.evaluate(X_test, y_test)
            # Robust AUC calculation for multiclass
            try:
                y_proba = model.model.predict_proba(X_test)
                y_test_bin = label_binarize(y_test, classes=all_labels)
                auc = roc_auc_score(y_test_bin, y_proba, multi_class='ovr', average='macro')
                if auc is not None and not (isinstance(auc, float) and (math.isnan(auc) or math.isinf(auc))):
                    metrics['AUC'] = round(auc, 2)
                else:
                    metrics['AUC'] = 'Not computable'
            except Exception:
                metrics['AUC'] = 'Not computable'
            return metrics
        metrics = safe_evaluate(model, X_test, y_test, all_labels)
        
        st.write(f"## {model_name} Evaluation Metrics")
        metrics_df = pd.DataFrame([metrics]).round(2)
        styled_metrics_df = metrics_df.style.set_table_styles([
            {'selector': 'th', 'props': [('text-align', 'center')]},
            {'selector': 'td', 'props': [('text-align', 'center')]}
        ]).format(precision=2)
        st.table(styled_metrics_df)
        st.markdown('<div style="color: #888; font-size: 0.95em; margin-top: -1em; margin-bottom: 1em;"><em>*AUC is often "Not computable" for these targets because, in many train/test splits, not all grade classes are present in the test set or predicted by the model. Multiclass AUC requires every class to be represented in both true and predicted labels; otherwise, the metric is undefined.</em></div>', unsafe_allow_html=True)
        
        y_pred = model.predict(X_test)
        y_pred_decoded = le_target.inverse_transform(y_pred)
        y_test_decoded = le_target.inverse_transform(y_test)
        
        cm = confusion_matrix(y_test, y_pred, labels=all_labels)
        report_dict = classification_report(y_test, y_pred, labels=all_labels, output_dict=True)
        report_df = pd.DataFrame(report_dict).transpose()
        
        desired_rows = [str(lbl) for lbl in all_labels] + ['accuracy', 'macro avg', 'weighted avg']
        existing_rows = [row for row in desired_rows if row in report_df.index]
        report_df = report_df.loc[existing_rows]
        report_df = report_df.round(3)
        
        col1, col2 = st.columns([1,1])
        with col1:
            st.write("### Confusion Matrix")
            fig, ax = plt.subplots(figsize=(4, 3))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False, square=True)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            plt.tight_layout()
            st.pyplot(fig)
        with col2:
            st.write("### Classification Report")
            report_df_rounded = report_df.round(2)
            styled_report_df = report_df_rounded.style.set_table_styles([
                {'selector': 'th', 'props': [('text-align', 'center')]},
                {'selector': 'td', 'props': [('text-align', 'center')]}
            ]).format(precision=2)
            st.table(styled_report_df)
else:
    st.info("Please select a dataset to begin.")
