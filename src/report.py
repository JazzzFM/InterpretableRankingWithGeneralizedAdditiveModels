import os
from textwrap import dedent

def write_markdown(path, meta: dict, plots: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    md = f"""# Credit Risk Ranking Report

## 1. Executive Summary
- Observations: {meta['n_obs']}
- Numeric Variables: {meta['n_num']}
- Categorical Variables: {meta['n_cat']}
- Test Metrics:
  - ROC-AUC: {meta['metrics']['roc_auc']:.3f}
  - PR-AUC: {meta.get('pr_auc', 'N/A')}
  - Brier: {meta['metrics']['brier']:.3f}
  - nDCG@100: {meta['metrics']['ndcg@100']:.3f}
  - Kendall τ: {meta['metrics']['kendall_tau']:.3f}

## 2. Report Details

A detailed HTML report with interactive plots and in-depth analysis is available at [reports/report.html](reports/report.html).
"""
    with open(path, "w") as f:
        f.write(md)


def write_html(path, meta: dict, plots: dict, top_md: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
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
<li>Numeric Variables: {meta['n_num']}</li>
<li>Categorical Variables: {meta['n_cat']}</li>
<li>Test Metrics:
  <ul>
  <li>ROC-AUC: {meta['metrics']['roc_auc']:.3f}</li>
  <li>PR-AUC: {meta.get('pr_auc', 'N/A')}</li>
  <li>Brier: {meta['metrics']['brier']:.3f}</li>
  <li>nDCG@100: {meta['metrics']['ndcg@100']:.3f}</li>
  <li>Kendall τ: {meta['metrics']['kendall_tau']:.3f}</li>
  </ul>
</li>
</ul>

<h2 id="performance">2. Model Performance</h2>
<div class="plot">
<h3>Confusion Matrix</h3>
<img src="data:image/png;base64,{plots['confusion_matrix']}"/>
<p>The confusion matrix shows the model's performance in classifying credit risk. The diagonal elements represent the number of points for which the predicted label is equal to the true label, while off-diagonal elements are those that are mislabeled by the classifier.</p>
</div>
<div class="plot">
<h3>ROC Curve</h3>
<img src="data:image/png;base64,{plots['roc_curve']}"/>
<p>The ROC curve is a graphical plot that illustrates the diagnostic ability of a binary classifier system as its discrimination threshold is varied. The curve is created by plotting the true positive rate (TPR) against the false positive rate (FPR) at various threshold settings.</p>
</div>
<div class="plot">
<h3>Precision-Recall Curve</h3>
<img src="data:image/png;base64,{plots['precision_recall_curve']}"/>
<p>The precision-recall curve shows the tradeoff between precision and recall for different threshold. A high area under the curve represents both high recall and high precision, where high precision relates to a low false positive rate, and high recall relates to a low false negative rate.</p>
</div>

<h2 id="effects">3. Partial Effects by Variable</h2>
"""
    for feat, img in plots.items():
        if feat not in ["confusion_matrix", "roc_curve", "precision_recall_curve"]:
            html += f"""<div class="plot">
<h3>{feat}</h3>
<img src="data:image/png;base64,{img}"/>
<p>This plot shows the partial effect of the feature on the credit risk prediction. It illustrates how the prediction changes as the feature value changes, holding all other features constant.</p>
</div>"""

    html += """</div>
</body>
</html>"""
    with open(path, "w") as f:
        f.write(html)
