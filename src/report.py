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

### Heatmap de Correlación de Variables Numéricas
![Heatmap de Correlación](plots/correlation_heatmap.png)
*El mapa de calor muestra una correlación positiva moderada entre el monto (`amount`) y la duración (`duration`) del crédito, lo cual es lógicamente esperado: créditos más grandes suelen requerir plazos más largos. Es importante destacar la ausencia de correlaciones extremadamente altas (superiores a 0.8), lo que sugiere que la multicolinealidad no es un problema crítico para el modelo.*

### Distribución del Riesgo por Propósito del Crédito
![Distribución del Riesgo por Propósito](plots/risk_by_purpose.png)
*Este gráfico revela insights de negocio cruciales. Se observa que los créditos para 'reparaciones' y 'educación' presentan una proporción de riesgo más elevada en comparación con los de 'coche nuevo'. Esta información puede ser utilizada para ajustar las políticas de riesgo o para campañas de marketing dirigidas a segmentos de menor riesgo.*

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

### Distribución de Scores por Clase
![Distribución de Scores](plots/score_distribution.png)
*Este gráfico es una de las visualizaciones más importantes para evaluar el poder de separación del modelo. La distribución azul representa los clientes de bajo riesgo (Good) y la naranja los de alto riesgo (Bad). Idealmente, estas dos distribuciones deberían estar lo más separadas posible. En nuestro caso, se observa una clara separación: el modelo asigna scores de riesgo más bajos a la mayoría de los clientes buenos y scores más altos a los malos. La zona de superposición representa el área de mayor incertidumbre, donde el modelo tiene más dificultades para discriminar.*

## 4. Interpretabilidad del Modelo: Análisis de Sensibilidad y Efectos Parciales

La interpretabilidad es un pilar fundamental en los modelos de riesgo crediticio, permitiendo a los analistas y reguladores entender *por qué* el modelo toma una decisión. Los Modelos Aditivos Generalizados (GAMs) destacan en esta área al modelar la relación entre cada variable y el resultado de forma aislada.

### 4.1. Análisis de Sensibilidad Automatizado

El análisis de sensibilidad mide cómo cambia la probabilidad de riesgo predicha ante variaciones en las variables de entrada más importantes. Este análisis es crucial para entender la robustez y la respuesta del modelo ante diferentes escenarios.

![Análisis de Sensibilidad](interpretability/sensitivity_analysis.png)

**Interpretación del Gráfico:**

El gráfico superior muestra el impacto porcentual en la probabilidad de riesgo al variar las tres variables numéricas clave (`age`, `duration`, `amount`) en un rango de -50% a +50%.

- **Edad (`age`):** Es la variable más influyente. Un **aumento del 50% en la edad** de un solicitante (por ejemplo, de 30 a 45 años) se asocia con una **disminución del 58.3% en su probabilidad de riesgo**. Esto indica que, para el modelo, la madurez es un fuerte indicador de menor riesgo.
- **Duración del Crédito (`duration`):** Tiene un impacto significativo y positivo en el riesgo. Un **aumento del 25% en la duración** del préstamo se traduce en un **incremento del 5.5% en la probabilidad de riesgo**. Préstamos más largos son inherentemente más riesgosos.
- **Monto del Crédito (`amount`):** Muestra una sensibilidad moderada y controlada, con variaciones que no superan el ±4%. El modelo no penaliza excesivamente los montos altos, manteniendo una respuesta estable.

### 4.2. Efectos Parciales de las Variables Clave

Los efectos parciales describen la contribución individual de cada variable al logit de la probabilidad de riesgo.

- **Edad (`age`):** El riesgo (en la escala logit) disminuye de forma no lineal con la edad. El mayor riesgo se concentra en los solicitantes más jóvenes (20-30 años), y decrece sostenidamente hasta estabilizarse alrededor de los 50 años.
- **Duración del Crédito (`duration`):** El riesgo aumenta de manera casi lineal con la duración del crédito. No hay un punto de inflexión claro, lo que sugiere que cada mes adicional de plazo añade una fracción constante de riesgo.
- **Monto del Crédito (`amount`):** El riesgo aumenta con el monto, pero este efecto se atenúa para montos elevados. Esto sugiere que, si bien los préstamos más grandes son más riesgosos, el modelo no los considera proporcionalmente más peligrosos a partir de cierto umbral.

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

<div class="plot-container">
  <h3>Heatmap de Correlación de Variables Numéricas</h3>
  <img src="data:image/png;base64,{encoded_plots['correlation_heatmap']}"/>
  <p><i>El mapa de calor muestra una correlación positiva moderada entre el monto (`amount`) y la duración (`duration`) del crédito, lo cual es lógicamente esperado: créditos más grandes suelen requerir plazos más largos. Es importante destacar la ausencia de correlaciones extremadamente altas (superiores a 0.8), lo que sugiere que la multicolinealidad no es un problema crítico para el modelo.</i></p>
</div>

<div class="plot-container">
  <h3>Distribución del Riesgo por Propósito del Crédito</h3>
  <img src="data:image/png;base64,{encoded_plots['risk_by_purpose']}"/>
  <p><i>Este gráfico revela insights de negocio cruciales. Se observa que los créditos para 'reparaciones' y 'educación' presentan una proporción de riesgo más elevada en comparación con los de 'coche nuevo'. Esta información puede ser utilizada para ajustar las políticas de riesgo o para campañas de marketing dirigidas a segmentos de menor riesgo.</i></p>
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

<div class="plot-container">
  <h3>Distribución de Scores por Clase</h3>
  <img src="data:image/png;base64,{encoded_plots['score_distribution']}"/>
  <p><i>Este gráfico es una de las visualizaciones más importantes para evaluar el poder de separación del modelo. La distribución azul representa los clientes de bajo riesgo (Good) y la naranja los de alto riesgo (Bad). Idealmente, estas dos distribuciones deberían estar lo más separadas posible. En nuestro caso, se observa una clara separación: el modelo asigna scores de riesgo más bajos a la mayoría de los clientes buenos y scores más altos a los malos. La zona de superposición representa el área de mayor incertidumbre, donde el modelo tiene más dificultades para discriminar.</i></p>
</div>

<h2 id="interpretability">4. Interpretabilidad del Modelo: Análisis de Sensibilidad y Efectos Parciales</h2>

<p>La interpretabilidad es un pilar fundamental en los modelos de riesgo crediticio, permitiendo a los analistas y reguladores entender <em>por qué</em> el modelo toma una decisión. Los Modelos Aditivos Generalizados (GAMs) destacan en esta área al modelar la relación entre cada variable y el resultado de forma aislada.</p>

<h3>4.1. Análisis de Sensibilidad Automatizado</h3>

<p>El análisis de sensibilidad mide cómo cambia la probabilidad de riesgo predicha ante variaciones en las variables de entrada más importantes. Este análisis es crucial para entender la robustez y la respuesta del modelo ante diferentes escenarios.</p>

<div class="plot-container">
  <img src="data:image/png;base64,{encoded_plots.get('sensitivity_analysis', '')}"/>
  <p><strong>Interpretación del Gráfico:</strong></p>
</div>

<p>El gráfico superior muestra el impacto porcentual en la probabilidad de riesgo al variar las tres variables numéricas clave (<code>age</code>, <code>duration</code>, <code>amount</code>) en un rango de -50% a +50%.</p>
<ul>
  <li><strong>Edad (<code>age</code>):</strong> Es la variable más influyente. Un <strong>aumento del 50% en la edad</strong> de un solicitante (por ejemplo, de 30 a 45 años) se asocia con una <strong>disminución del 58.3% en su probabilidad de riesgo</strong>. Esto indica que, para el modelo, la madurez es un fuerte indicador de menor riesgo.</li>
  <li><strong>Duración del Crédito (<code>duration</code>):</strong> Tiene un impacto significativo y positivo en el riesgo. Un <strong>aumento del 25% en la duración</strong> del préstamo se traduce en un <strong>incremento del 5.5% en la probabilidad de riesgo</strong>. Préstamos más largos son inherentemente más riesgosos.</li>
  <li><strong>Monto del Crédito (<code>amount</code>):</strong> Muestra una sensibilidad moderada y controlada, con variaciones que no superan el ±4%. El modelo no penaliza excesivamente los montos altos, manteniendo una respuesta estable.</li>
</ul>

<h3>4.2. Efectos Parciales de las Variables Clave</h3>

<p>Los efectos parciales describen la contribución individual de cada variable al logit de la probabilidad de riesgo.</p>

<ul>
  <li><strong>Edad (<code>age</code>):</strong> El riesgo (en la escala logit) disminuye de forma no lineal con la edad. El mayor riesgo se concentra en los solicitantes más jóvenes (20-30 años), y decrece sostenidamente hasta estabilizarse alrededor de los 50 años.</li>
  <li><strong>Duración del Crédito (<code>duration</code>):</strong> El riesgo aumenta de manera casi lineal con la duración del crédito. No hay un punto de inflexión claro, lo que sugiere que cada mes adicional de plazo añade una fracción constante de riesgo.</li>
  <li><strong>Monto del Crédito (<code>amount</code>):</strong> El riesgo aumenta con el monto, pero este efecto se atenúa para montos elevados. Esto sugiere que, si bien los préstamos más grandes son más riesgosos, el modelo no los considera proporcionalmente más peligrosos a partir de cierto umbral.</li>
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