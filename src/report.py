import os
import base64

def write_markdown(path, meta: dict, plots: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    md = f"""# Credit Risk Ranking Report

## 1. Executive Summary
- Observations: {meta['n_obs']}
- Numerical Variables: {meta['n_num']}
- Categorical Variables: {meta['n_cat']}

## 2. Dataset Analysis

### Credit Risk Distribution
![Credit Risk Distribution](plots/credit_risk_distribution.png)

### Age Distribution
![Age Distribution](plots/age_distribution.png)

## 3. Model Performance
- ROC-AUC: {meta['metrics']['roc_auc']:.3f}
- PR-AUC: {meta.get('pr_auc', 'N/A')}
- Brier: {meta['metrics']['brier']:.3f}
- nDCG@100: {meta['metrics']['ndcg@100']:.3f}
- Kendall τ: {meta['metrics']['kendall_tau']:.3f}

## 4. Comments on the Exercise

This report was generated as part of a technical exercise to demonstrate MLOps capabilities. The data used is the German Credit Dataset, which is a well-known dataset for credit scoring tasks.

The exercise involved building a complete MLOps pipeline, including data validation, model training, interpretability analysis, and report generation. The pipeline is designed to be production-ready, with a focus on security, scalability, and maintainability.

## 5. Detailed Report

A detailed HTML report with interactive graphs and in-depth analysis is available at [report.html](report.html).
"""
    
    with open(path, "w") as f:
        f.write(md)

def write_html(path, meta: dict, plots: dict, top_md: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Encode plots in base64
    encoded_plots = {}
    for plot_name, plot_data in plots.items():
        encoded_plots[plot_name] = base64.b64encode(plot_data).decode('utf-8')

    html = f"""<!DOCTYPE html>
<html>
<head>
<title>Credit Risk Ranking Report</title>
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
<a href="#summary">Summary</a>
<a href="#performance">Performance</a>
<a href="#effects">Partial Effects</a>
</nav>
<div class="container">
<h1 id="summary">Credit Risk Ranking Report</h1>

<h2>1. Executive Summary</h2>
<ul>
<li>Observations: {meta['n_obs']}</li>
<li>Numerical Variables: {meta['n_num']}</li>
<li>Categorical Variables: {meta['n_cat']}</li>
</ul>

<h2>2. Model Performance</h2>
<ul>
<li>ROC-AUC: {meta['metrics']['roc_auc']:.3f}</li>
<li>PR-AUC: {meta.get('pr_auc', 'N/A')}</li>
<li>Brier: {meta['metrics']['brier']:.3f}</li>
<li>nDCG@100: {meta['metrics']['ndcg@100']:.3f}</li>
<li>Kendall τ: {meta['metrics']['kendall_tau']:.3f}</li>
</ul>

<h2 id="performance">3. Performance Plots</h2>
<div class="plot">
<h3>Confusion Matrix</h3>
<img src="data:image/png;base64,{encoded_plots['confusion_matrix']}"/>
</div>
<div class="plot">
<h3>ROC Curve</h3>
<img src="data:image/png;base64,{encoded_plots['roc_curve']}"/>
</div>
<div class="plot">
<h3>Precision-Recall Curve</h3>
<img src="data:image/png;base64,{encoded_plots['precision_recall_curve']}"/>
</div>

<h2 id="effects">4. Partial Effects by Variable</h2>
<div class="plot">
<h3>Credit Risk Distribution</h3>
<img src="data:image/png;base64,{encoded_plots['credit_risk_distribution']}"/>
</div>
<div class="plot">
<h3>Age Distribution</h3>
<img src="data:image/png;base64,{encoded_plots['age_distribution']}"/>
</div>

<h2>5. Comments on the Exercise</h2>
<p>This report was generated as part of a technical exercise to demonstrate MLOps capabilities. The data used is the German Credit Dataset, which is a well-known dataset for credit scoring tasks.</p>
<p>The exercise involved building a complete MLOps pipeline, including data validation, model training, interpretability analysis, and report generation. The pipeline is designed to be production-ready, with a focus on security, scalability, and maintainability.</p>

</div>
</body>
</html>"""
    
    with open(path, "w") as f:
        f.write(html)
