# /c:/Projects/CICD/src/test.py
import os
from typing import Optional, Any

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

# load model saved with skops
try:
    from skops import io as skops_io
except Exception:
    skops_io = None  


class ModelEvaluator:
    """
    Loads a single model (skops format) and provides simple classification metrics
    and a confusion-matrix visualization.

    Usage:
      evaluator = ModelEvaluator("models/model.skops")
      evaluator.load_model()
      metrics = evaluator.evaluate(X_test, y_test, plot_cm=True)
    """

    def __init__(self, model_path: str = "models/model.skops"):
        self.model_path = model_path
        self.model: Optional[Any] = None
        self.y_pred: Optional[np.ndarray] = None
        self.y_proba: Optional[np.ndarray] = None
        self.metrics_df: Optional[pd.DataFrame] = None
        self.trusted_types = [
            "lightgbm.sklearn.LGBMClassifier",
            "xgboost.sklearn.XGBClassifier",
            "imblearn.pipeline.Pipeline",
            "imblearn.over_sampling._smote.base.SMOTE",
            'collections.OrderedDict', 
            'lightgbm.basic.Booster', 
            'numpy.dtype', 
            'scipy.sparse._csr.csr_matrix', 
            'sklearn.utils._bunch.Bunch', 
            'xgboost.core.Booster'
        ]

    def load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        if skops_io is None:
            raise RuntimeError("skops.io not available. Install scikit-learn-skbio/skops.")
        self.model = skops_io.load(self.model_path, trusted=self.trusted_types)
        return self.model

    def predict(self, X):
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        self.y_pred = np.asarray(self.model.predict(X))
        # try predict_proba for ROC-AUC if available
        if hasattr(self.model, "predict_proba"):
            try:
                self.y_proba = np.asarray(self.model.predict_proba(X)[:, 1])
            except Exception:
                self.y_proba = None
        else:
            self.y_proba = None
        return self.y_pred

    def compute_metrics(self, y_true):
        if self.y_pred is None:
            raise RuntimeError("Predictions not available. Call predict(X) first.")
        acc = accuracy_score(y_true, self.y_pred)
        prec = precision_score(y_true, self.y_pred, zero_division=0)
        rec = recall_score(y_true, self.y_pred, zero_division=0)
        f1 = f1_score(y_true, self.y_pred, zero_division=0)
        roc_auc = None
        if self.y_proba is not None:
            try:
                roc_auc = roc_auc_score(y_true, self.y_proba)
            except Exception:
                roc_auc = None
        metrics = {
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1,
            "ROC-AUC": roc_auc,
        }
        self.metrics_df = pd.DataFrame.from_dict(metrics, orient="index", columns=["Score"])
        return self.metrics_df

    def plot_confusion_matrix(self, y_true, normalize: bool = False, cmap: str = "Blues", ax=None):
        if self.y_pred is None:
            raise RuntimeError("Predictions not available. Call predict(X) first.")
        cm = confusion_matrix(y_true, self.y_pred)
        if normalize:
            with np.errstate(all="ignore"):
                cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
                cm = np.nan_to_num(cm)
        if ax is None:
            _fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt=".2f" if normalize else "d",
            cmap=cmap,
            cbar=True,
            ax=ax,
            xticklabels=["Pred 0", "Pred 1"],
            yticklabels=["True 0", "True 1"],
        )
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.set_title("Confusion Matrix (normalized)" if normalize else "Confusion Matrix")
        plt.tight_layout()
        return ax

    def evaluate(self, X, y, plot_cm: bool = True, normalize_cm: bool = False):
        """
        Convenience method: runs prediction, computes metrics and optionally plots CM.
        Returns the metrics DataFrame.
        """
        self.predict(X)
        metrics = self.compute_metrics(y)
        if plot_cm:
            self.plot_confusion_matrix(y, normalize=normalize_cm)
            plt.show()
        metrics.to_json("metrics.json")
        return metrics




if __name__ == "__main__":
    # create a simple test dataset and provide a fallback model if loading fails
    from sklearn.datasets import make_classification
    from sklearn.dummy import DummyClassifier
    import warnings 

    warnings.filterwarnings("ignore")

    X_test, y_test = make_classification(
        n_samples=200,
        n_features=10,
        n_informative=5,
        n_redundant=0,
        n_classes=2,
        random_state=42,
    )

    evaluator = ModelEvaluator("models/model.skops")
  
    try:
        evaluator.load_model()
    except Exception:
        # If the saved model is not available or skops isn't installed, use a simple fallback
        dummy = DummyClassifier(strategy="most_frequent")
        dummy.fit(X_test, y_test)
        evaluator.model = dummy


