import os
from textwrap import dedent

def write_markdown(path, meta: dict, plots: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    md = f"""# Reporte de Ranking de Riesgo Crediticio

## 1. Resumen Ejecutivo
- Observaciones: {meta['n_obs']}
- Variables Numéricas: {meta['n_num']}
- Variables Categóricas: {meta['n_cat']}
- Métricas de Prueba:
  - ROC-AUC: {meta['metrics']['roc_auc']:.3f}
  - PR-AUC: {meta.get('pr_auc', 'N/A')}
  - Brier: {meta['metrics']['brier']:.3f}
  - nDCG@100: {meta['metrics']['ndcg@100']:.3f}
  - Kendall τ: {meta['metrics']['kendall_tau']:.3f}

## 2. Detalles del Reporte

Un reporte HTML detallado con gráficos interactivos y análisis en profundidad está disponible en [reports/report.html](reports/report.html).
"""
    with open(path, "w") as f:
        f.write(md)


def write_html(path, meta: dict, plots: dict, top_md: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    html = f"""<!DOCTYPE html>
<html>
<head>
<title>Reporte de Ranking de Riesgo Crediticio</title>
<meta charset="UTF-8">
<style>
body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 0; background-color: #f4f4f4; color: #333; }}
.container {{ max-width: 1200px; margin: 0 auto; padding: 20px; background-color: #fff; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
nav {{ background-color: #333; padding: 10px 0; text-align: center; }}
nav a {{ color: #fff; text-decoration: none; padding: 10px 20px; }}
nav a:hover {{ background-color: #555; }}
h1, h2, h3 {{ color: #333; }}
table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
th, td {{ padding: 12px; border: 1px solid #ddd; text-align: left; }}
th {{ background-color: #f2f2f2; }}
.plot {{ text-align: center; margin-bottom: 30px; }}
.plot img {{ max-width: 100%; height: auto; border: 1px solid #ddd; }}
</style>
</head>
<body>
<nav>
<a href="#resumen">Resumen</a>
<a href="#rendimiento">Rendimiento</a>
<a href="#efectos">Efectos Parciales</a>
</nav>
<div class="container">
<h1 id="resumen">Reporte de Ranking de Riesgo Crediticio</h1>

<h2>1. Resumen Ejecutivo</h2>
<ul>
<li>Observaciones: {meta['n_obs']}</li>
<li>Variables Numéricas: {meta['n_num']}</li>
<li>Variables Categóricas: {meta['n_cat']}</li>
<li>Métricas de Prueba:
  <ul>
  <li>ROC-AUC: {meta['metrics']['roc_auc']:.3f}</li>
  <li>PR-AUC: {meta.get('pr_auc', 'N/A')}</li>
  <li>Brier: {meta['metrics']['brier']:.3f}</li>
  <li>nDCG@100: {meta['metrics']['ndcg@100']:.3f}</li>
  <li>Kendall τ: {meta['metrics']['kendall_tau']:.3f}</li>
  </ul>
</li>
</ul>

<h2 id="rendimiento">2. Rendimiento del Modelo</h2>
<div class="plot">
<h3>Matriz de Confusión</h3>
<img src="data:image/png;base64,{plots['confusion_matrix']}"/>
<p>La matriz de confusión muestra el rendimiento del modelo en la clasificación del riesgo crediticio. Los elementos diagonales representan el número de puntos para los cuales la etiqueta predicha es igual a la etiqueta real, mientras que los elementos fuera de la diagonal son aquellos que el clasificador etiqueta incorrectamente.</p>
</div>
<div class="plot">
<h3>Curva ROC</h3>
<img src="data:image/png;base64,{plots['roc_curve']}"/>
<p>La curva ROC es un gráfico que ilustra la capacidad de diagnóstico de un sistema clasificador binario a medida que varía su umbral de discriminación. La curva se crea trazando la tasa de verdaderos positivos (TPR) frente a la tasa de falsos positivos (FPR) en varios ajustes de umbral.</p>
</div>
<div class="plot">
<h3>Curva Precisión-Recall</h3>
<img src="data:image/png;base64,{plots['precision_recall_curve']}"/>
<p>La curva de precisión-recall muestra el equilibrio entre la precisión y el recall para diferentes umbrales. Un área alta bajo la curva representa tanto un alto recall como una alta precisión, donde una alta precisión se relaciona con una baja tasa de falsos positivos y un alto recall se relaciona con una baja tasa de falsos negativos.</p>
</div>

<h2 id="efectos">3. Efectos Parciales por Variable</h2>
"""
    for feat, img in plots.items():
        if feat not in ["confusion_matrix", "roc_curve", "precision_recall_curve"]:
            html += f"""<div class="plot">
<h3>{feat}</h3>
<img src="data:image/png;base64,{img}"/>
<p>Este gráfico muestra el efecto parcial de la característica en la predicción del riesgo crediticio. Ilustra cómo cambia la predicción a medida que cambia el valor de la característica, manteniendo constantes todas las demás características.</p>
</div>"""

    html += """</div>
</body>
</html>"""
    with open(path, "w") as f:
        f.write(html)
