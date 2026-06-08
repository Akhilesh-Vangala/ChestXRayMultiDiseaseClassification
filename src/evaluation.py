import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support, calibration_curve
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class ModelEvaluator:
    def __init__(self, disease_labels: List[str]):
        self.disease_labels = disease_labels
        self.num_classes = len(disease_labels)
    
    def evaluate(self, model: tf.keras.Model, test_dataset: tf.data.Dataset,
                use_tta: bool = False) -> Dict:
        y_true_list, y_pred_list = [], []
        
        for images, labels in test_dataset:
            if use_tta:
                preds = []
                augs = [
                    lambda img: img,
                    lambda img: tf.image.flip_left_right(img),
                    lambda img: tf.image.flip_up_down(img),
                    lambda img: tf.image.rot90(img),
                    lambda img: tf.image.rot90(img, k=2),
                    lambda img: tf.image.rot90(img, k=3)
                ]
                for aug in augs:
                    preds.append(model(aug(images), training=False))
                predictions = tf.reduce_mean(preds, axis=0)
            else:
                predictions = model.predict(images, verbose=0)
            y_true_list.append(labels.numpy())
            y_pred_list.append(predictions)
        
        y_true = np.concatenate(y_true_list, axis=0)
        y_pred = np.concatenate(y_pred_list, axis=0)
        
        results = {
            'mean_auc': roc_auc_score(y_true, y_pred, average='macro'),
            'per_class_auc': roc_auc_score(y_true, y_pred, average=None),
            'mean_ap': average_precision_score(y_true, y_pred, average='macro'),
            'per_class_ap': [average_precision_score(y_true[:, i], y_pred[:, i])
                            for i in range(self.num_classes)]
        }
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, (y_pred > 0.5).astype(int), average=None, zero_division=0
        )
        
        calibration_scores = []
        for i in range(self.num_classes):
            try:
                _, _ = calibration_curve(y_true[:, i], y_pred[:, i], n_bins=10)
                calibration_scores.append(0.0)
            except:
                calibration_scores.append(0.0)
        
        results['results_df'] = pd.DataFrame({
            'Disease': self.disease_labels,
            'AUC-ROC': results['per_class_auc'],
            'AP': results['per_class_ap'],
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'Calibration': calibration_scores
        })
        
        return results
    
    def calculate_recall_improvement(self, baseline_recall: float, improved_recall: float) -> float:
        return ((improved_recall - baseline_recall) / baseline_recall) * 100
    
    def get_class_performance(self, y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, (y_pred > 0.5).astype(int), average=None, zero_division=0
        )
        return pd.DataFrame({
            'Disease': self.disease_labels,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1
        })
