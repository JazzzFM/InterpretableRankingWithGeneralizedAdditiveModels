import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss, ndcg_score,
    accuracy_score, precision_score, recall_score, f1_score
)
from scipy.stats import kendalltau, ks_2samp

def evaluate_ranking(y_true: np.ndarray, p: np.ndarray, k: int = 100, threshold: float = 0.5) -> dict:
    # Convert probabilities to binary predictions for classification metrics
    y_pred = (p >= threshold).astype(int)
    
    # Basic classification metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # ROC AUC
    roc_auc = roc_auc_score(y_true, p)
    
    # Kolmogorov-Smirnov statistic
    p_pos = p[y_true == 1]  # Probabilities for positive class
    p_neg = p[y_true == 0]  # Probabilities for negative class
    ks_stat, ks_pvalue = ks_2samp(p_pos, p_neg)
    
    # Ranking metrics
    rel = 1 - y_true  # mayor relevancia = no default
    ndcg = ndcg_score([rel], [1 - p], k=k)  # 1-p para que 'buenos' queden arriba
    tau, _ = kendalltau(-p, rel)  # orden por riesgo vs relevancia
    
    # Probability quality
    brier = brier_score_loss(y_true, p)
    pr_auc = average_precision_score(y_true, p)
    
    return {
        # Classification metrics
        "accuracy": float(accuracy),
        "precision": float(precision), 
        "recall": float(recall),
        "f1_score": float(f1),
        
        # Discrimination metrics
        "roc_auc": float(roc_auc),
        "ks_statistic": float(ks_stat),
        "ks_pvalue": float(ks_pvalue),
        
        # Probability quality
        "brier_score": float(brier),
        "pr_auc": float(pr_auc),
        
        # Ranking metrics  
        "ndcg@{}".format(k): float(ndcg),
        "kendall_tau": float(tau),
        
        # Legacy compatibility
        "brier": float(brier),
    }
