import os
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, classification_report
import numpy as np
import pandas as pd

def plot_to_base64():
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def plot_partial_effect(feature, x, y, ci):
    plt.figure()
    plt.plot(x, y, label="Efecto Parcial")
    if ci is not None and len(ci) == 2:
        # Ensure ci is a numpy array of the correct shape
        lower_ci = np.array(ci[0])
        upper_ci = np.array(ci[1])
        if lower_ci.shape == upper_ci.shape:
            plt.fill_between(x, lower_ci, upper_ci, alpha=0.2, label="IC 95%")
    plt.title(f"Efecto de {feature}")
    plt.xlabel(feature)
    plt.ylabel("Contribución (log-odds)")
    plt.legend()
    return plot_to_base64()

def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Matriz de Confusión')
    plt.xlabel('Predicho')
    plt.ylabel('Real')
    return plot_to_base64()

def plot_roc_curve(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, label='Curva ROC')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('Curva ROC')
    plt.xlabel('Tasa de Falsos Positivos')
    plt.ylabel('Tasa de Verdaderos Positivos')
    plt.legend()
    return plot_to_base64()

def plot_precision_recall_curve(y_true, y_prob):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    plt.figure()
    plt.plot(recall, precision, label='Curva Precisión-Recall')
    plt.title('Curva Precisión-Recall')
    plt.xlabel('Recall')
    plt.ylabel('Precisión')
    plt.legend()
    return plot_to_base64()

def plot_correlation_heatmap(df, numeric_cols, save_path=None):
    """Generate correlation heatmap for numeric variables"""
    plt.figure(figsize=(10, 8))
    correlation_matrix = df[numeric_cols].corr()
    
    # Create heatmap with annotations
    sns.heatmap(correlation_matrix, 
                annot=True, 
                cmap='coolwarm', 
                center=0, 
                square=True,
                fmt='.2f',
                cbar_kws={'label': 'Correlación'})
    
    plt.title('Heatmap de Correlación - Variables Numéricas', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        return plot_to_base64()

def plot_risk_by_purpose(df, purpose_col='purpose', target_col='credit_risk', save_path=None):
    """Generate risk distribution by credit purpose"""
    plt.figure(figsize=(12, 8))
    
    # Calculate risk proportion by purpose
    risk_by_purpose = df.groupby(purpose_col)[target_col].agg(['count', 'sum']).reset_index()
    risk_by_purpose['risk_rate'] = risk_by_purpose['sum'] / risk_by_purpose['count']
    risk_by_purpose = risk_by_purpose.sort_values('risk_rate', ascending=False)
    
    # Create stacked bar chart
    purpose_counts = df.groupby([purpose_col, target_col]).size().unstack(fill_value=0)
    purpose_counts = purpose_counts.loc[risk_by_purpose[purpose_col]]
    
    ax = purpose_counts.plot(kind='bar', 
                            stacked=True, 
                            figsize=(12, 8),
                            color=['#2E86AB', '#F24236'],
                            alpha=0.8)
    
    plt.title('Distribución del Riesgo por Propósito del Crédito', fontsize=14, fontweight='bold')
    plt.xlabel('Propósito del Crédito', fontsize=12)
    plt.ylabel('Número de Casos', fontsize=12)
    plt.legend(['Bajo Riesgo (Good)', 'Alto Riesgo (Bad)'], loc='upper right')
    plt.xticks(rotation=45, ha='right')
    
    # Add percentage labels
    for i, (idx, row) in enumerate(risk_by_purpose.iterrows()):
        total = row['count']
        risk_pct = row['risk_rate'] * 100
        plt.text(i, total + 5, f'{risk_pct:.1f}%', 
                ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        return plot_to_base64()

def plot_score_distribution(y_true, y_prob, save_path=None):
    """Generate score distribution by risk class"""
    plt.figure(figsize=(12, 8))
    
    # Create separate distributions for each class
    good_scores = y_prob[y_true == 0]  # Low risk (Good)
    bad_scores = y_prob[y_true == 1]   # High risk (Bad)
    
    # Plot overlapping histograms
    plt.hist(good_scores, bins=30, alpha=0.7, label='Bajo Riesgo (Good)', 
             color='#2E86AB', density=True, edgecolor='black', linewidth=0.5)
    plt.hist(bad_scores, bins=30, alpha=0.7, label='Alto Riesgo (Bad)', 
             color='#F24236', density=True, edgecolor='black', linewidth=0.5)
    
    # Add vertical lines for means
    plt.axvline(good_scores.mean(), color='#2E86AB', linestyle='--', 
                linewidth=2, label=f'Media Good: {good_scores.mean():.3f}')
    plt.axvline(bad_scores.mean(), color='#F24236', linestyle='--', 
                linewidth=2, label=f'Media Bad: {bad_scores.mean():.3f}')
    
    plt.title('Distribución de Scores por Clase de Riesgo', fontsize=14, fontweight='bold')
    plt.xlabel('Score de Riesgo (Probabilidad)', fontsize=12)
    plt.ylabel('Densidad', fontsize=12)
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    
    # Add statistics text box
    separation = abs(bad_scores.mean() - good_scores.mean())
    stats_text = f'Separación de Medias: {separation:.3f}\n'
    stats_text += f'Std Good: {good_scores.std():.3f}\n'
    stats_text += f'Std Bad: {bad_scores.std():.3f}'
    
    plt.text(0.02, 0.95, stats_text, transform=plt.gca().transAxes, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
             verticalalignment='top', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        return plot_to_base64()
