import os
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, classification_report
import numpy as np

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
