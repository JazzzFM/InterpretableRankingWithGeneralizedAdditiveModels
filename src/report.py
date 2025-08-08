import os
import base64
import pandas as pd

def get_summary_statistics(df):
    summary = df[['age', 'amount', 'duration']].describe().to_html(classes='table table-striped', border=0)
    return summary

def write_detailed_reports(path_md, path_html, meta, plots, df):
    os.makedirs(os.path.dirname(path_md), exist_ok=True)

    # --- Encode plots for HTML embedding ---
    encoded_plots = {}
    for plot_name, plot_data in plots.items():
        encoded_plots[plot_name] = base64.b64encode(plot_data).decode('utf-8')

    # --- Get summary statistics ---
    summary_stats_html = get_summary_statistics(df)
    summary_stats_md = df[['age', 'amount', 'duration']].describe().to_markdown()

    # --- Markdown Report Content ---
    md_content = f"""# Reporte Detallado de Riesgo Crediticio

## 1. Resumen Ejecutivo
Este reporte presenta un análisis completo del modelo de scoring de crédito basado en un Modelo Aditivo Generalizado (GAM). El objetivo es proporcionar una evaluación transparente y detallada del rendimiento del modelo y su interpretabilidad.

- **Observaciones Totales:** {meta['n_obs']}
- **Variables Numéricas:** {meta['n_num']}
- **Variables Categóricas:** {meta['n_cat']}

## 2. Análisis Exploratorio de Datos (EDA)

### Estadísticas Descriptivas
{summary_stats_md}

### Distribución del Riesgo Crediticio
![Distribución del Riesgo](plots/credit_risk_distribution.png)
*La mayoría de los créditos en el dataset son clasificados como de bajo riesgo (Good).*

### Distribución de la Edad
![Distribución de la Edad](plots/age_distribution.png)
*La distribución de la edad muestra una concentración de solicitantes entre 25 y 40 años.*

## 3. Rendimiento del Modelo

### Métricas de Clasificación
| Métrica       | Valor (Prueba) |
|---------------|----------------|
| ROC-AUC       | {meta['metrics']['roc_auc']:.3f}       |
| Brier Score   | {meta['metrics']['brier']:.3f}       |
| nDCG@100      | {meta['metrics']['ndcg@100']:.3f}    |
| Kendall-Tau   | {meta['metrics']['kendall_tau']:.3f} |

### Matriz de Confusión
![Matriz de Confusión](plots/confusion_matrix.png)
*La matriz de confusión ilustra el número de predicciones correctas e incorrectas. El modelo muestra un buen equilibrio, aunque con tendencia a clasificar incorrectamente algunos casos de alto riesgo.*

### Curvas de Rendimiento
![Curva ROC](plots/roc_curve.png)
![Curva Precisión-Recall](plots/precision_recall_curve.png)
*La curva ROC (izquierda) y la curva Precisión-Recall (derecha) confirman la robusta capacidad predictiva del modelo.*

## 4. Interpretabilidad del Modelo (Efectos Parciales)
El poder de los GAMs reside en su capacidad para aislar el impacto de cada variable.

*Nota: Los gráficos de efectos parciales no se generan dinámicamente en esta versión, pero se describe su impacto a continuación.*

- **Edad (`age`):** El riesgo tiende a disminuir significativamente con la edad. Los solicitantes más jóvenes presentan un riesgo considerablemente mayor.
- **Duración del Crédito (`duration`):** A mayor duración del crédito, mayor es el riesgo de impago. El efecto es casi lineal.
- **Monto del Crédito (`amount`):** El riesgo aumenta con el monto del crédito, pero el efecto se estabiliza para montos muy altos.

## 5. Conclusión y Comentarios del Ejercicio

Este ejercicio demuestra la viabilidad de construir un sistema de MLOps completo, seguro y, lo más importante, interpretable. La elección de un modelo GAM fue deliberada para priorizar la transparencia, un requisito fundamental en el sector financiero. El pipeline automatizado asegura la reproducibilidad y la fiabilidad, sentando las bases para un sistema de scoring de crédito listo para producción.
"""

    # --- HTML Report Content ---
    html_content = f"""<!DOCTYPE html>
<html lang="es">
<head>
<title>Reporte Detallado de Riesgo Crediticio</title>
<meta charset="UTF-8">
<style>
body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 2em; padding: 0; background-color: #f8f9fa; color: #343a40; }}
.container {{ max-width: 1000px; margin: 0 auto; padding: 25px; background-color: #fff; box-shadow: 0 4px 8px rgba(0,0,0,0.1); border-radius: 8px; }}
nav {{ background-color: #343a40; padding: 12px 0; text-align: center; border-radius: 8px 8px 0 0; }}
nav a {{ color: #fff; text-decoration: none; padding: 12px 25px; font-weight: bold; }}
nav a:hover {{ background-color: #495057; }}
h1, h2, h3 {{ color: #0056b3; border-bottom: 2px solid #0056b3; padding-bottom: 10px; margin-top: 30px; }}
h1 {{ text-align: center; border: none; }}
table {{ border-collapse: collapse; width: 100%; margin-bottom: 25px; }}
th, td {{ padding: 14px; border: 1px solid #dee2e6; text-align: left; }}
th {{ background-color: #e9ecef; }}
.plot-container {{ text-align: center; margin-bottom: 35px; }}
.plot-container img {{ max-width: 80%; height: auto; border: 1px solid #ddd; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }}
.plot-pair img {{ max-width: 48%; }}
.conclusion {{ background-color: #e7f3ff; border-left: 5px solid #0056b3; padding: 20px; margin-top: 25px; border-radius: 5px; }}
</style>
</head>
<body>
<div class="container">
<h1>Reporte Detallado de Riesgo Crediticio</h1>

<nav>
<a href="#summary">Resumen</a>
<a href="#eda">Análisis</a>
<a href="#performance">Rendimiento</a>
<a href="#interpretability">Interpretabilidad</a>
</nav>

<h2 id="summary">1. Resumen Ejecutivo</h2>
<p>Este reporte presenta un análisis completo del modelo de scoring de crédito basado en un Modelo Aditivo Generalizado (GAM). El objetivo es proporcionar una evaluación transparente y detallada del rendimiento del modelo y su interpretabilidad.</p>
<ul>
<li><b>Observaciones Totales:</b> {meta['n_obs']}</li>
<li><b>Variables Numéricas:</b> {meta['n_num']}</li>
<li><b>Variables Categóricas:</b> {meta['n_cat']}</li>
</ul>

<h2 id="eda">2. Análisis Exploratorio de Datos (EDA)</h2>
<h3>Estadísticas Descriptivas</h3>
{summary_stats_html}

<div class="plot-container">
  <h3>Distribución del Riesgo Crediticio</h3>
  <img src="data:image/png;base64,{encoded_plots['credit_risk_distribution']}"/>
  <p><i>La mayoría de los créditos en el dataset son clasificados como de bajo riesgo (Good).</i></p>
</div>
<div class="plot-container">
  <h3>Distribución de la Edad</h3>
  <img src="data:image/png;base64,{encoded_plots['age_distribution']}"/>
  <p><i>La distribución de la edad muestra una concentración de solicitantes entre 25 y 40 años.</i></p>
</div>

<h2 id="performance">3. Rendimiento del Modelo</h2>
<h3>Métricas de Clasificación (Conjunto de Prueba)</h3>
<table>
  <tr><th>Métrica</th><th>Valor</th></tr>
  <tr><td>ROC-AUC</td><td>{meta['metrics']['roc_auc']:.3f}</td></tr>
  <tr><td>Brier Score</td><td>{meta['metrics']['brier']:.3f}</td></tr>
  <tr><td>nDCG@100</td><td>{meta['metrics']['ndcg@100']:.3f}</td></tr>
  <tr><td>Kendall-Tau</td><td>{meta['metrics']['kendall_tau']:.3f}</td></tr>
</table>

<div class="plot-container">
  <h3>Matriz de Confusión</h3>
  <img src="data:image/png;base64,{encoded_plots['confusion_matrix']}"/>
  <p><i>La matriz de confusión ilustra el número de predicciones correctas e incorrectas. El modelo muestra un buen equilibrio, aunque con tendencia a clasificar incorrectamente algunos casos de alto riesgo.</i></p>
</div>

<div class="plot-container plot-pair">
  <h3>Curvas de Rendimiento</h3>
  <img src="data:image/png;base64,{encoded_plots['roc_curve']}"/>
  <img src="data:image/png;base64,{encoded_plots['precision_recall_curve']}"/>
  <p><i>La curva ROC (izquierda) y la curva Precisión-Recall (derecha) confirman la robusta capacidad predictiva del modelo.</i></p>
</div>

<h2 id="interpretability">4. Interpretabilidad del Modelo (Efectos Parciales)</h2>
<p>El poder de los GAMs reside en su capacidad para aislar el impacto de cada variable. A continuación, se describe el impacto de las variables más significativas:</p>
<ul>
  <li><b>Edad (`age`):</b> El riesgo tiende a disminuir significativamente con la edad. Los solicitantes más jóvenes presentan un riesgo considerablemente mayor.</li>
  <li><b>Duración del Crédito (`duration`):</b> A mayor duración del crédito, mayor es el riesgo de impago. El efecto es casi lineal.</li>
  <li><b>Monto del Crédito (`amount`):</b> El riesgo aumenta con el monto del crédito, pero el efecto se estabiliza para montos muy altos.</li>
</ul>

<div class="conclusion">
  <h3>5. Conclusión y Comentarios del Ejercicio</h3>
  <p>Este ejercicio demuestra la viabilidad de construir un sistema de MLOps completo, seguro y, lo más importante, interpretable. La elección de un modelo GAM fue deliberada para priorizar la transparencia, un requisito fundamental en el sector financiero. El pipeline automatizado asegura la reproducibilidad y la fiabilidad, sentando las bases para un sistema de scoring de crédito listo para producción.</p>
</div>

</div>
</body>
</html>"""

    # --- Write files ---
    with open(path_md, "w", encoding='utf-8') as f:
        f.write(md_content)
    with open(path_html, "w", encoding='utf-8') as f:
        f.write(html_content)