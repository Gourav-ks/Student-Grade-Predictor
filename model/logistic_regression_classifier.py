import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef
from .base_classifier import BaseClassifier

class LogisticRegressionClassifier(BaseClassifier):  # Log Classification
    def __init__(self, X_train, y_train):
        super().__init__(X_train, y_train)
        self.model = LogisticRegression(max_iter=1000)
    def train(self):
        self.model.fit(self.X_train, self.y_train)
    def predict(self, X):
        return self.model.predict(X)
    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        y_proba = self.model.predict_proba(X_test) if hasattr(self.model, 'predict_proba') else None
        auc = None
        all_labels = np.unique(self.y_train)
        if y_proba is not None:
            y_proba = np.asarray(y_proba)
            try:
                if y_proba.ndim == 2 and y_proba.shape[1] == 2:
                    auc = roc_auc_score(y_test, y_proba[:, 1], labels=all_labels)
                elif y_proba.ndim == 2:
                    auc = roc_auc_score(y_test, y_proba, multi_class='ovr', labels=all_labels)
            except Exception:
                auc = None
        metrics = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'AUC': auc,
            'Precision': precision_score(y_test, y_pred, average='weighted'),
            'Recall': recall_score(y_test, y_pred, average='weighted'),
            'F1': f1_score(y_test, y_pred, average='weighted'),
            'MCC': matthews_corrcoef(y_test, y_pred)
        }
        return metrics
