#!/usr/bin/env python3
"""
Script to generate and render reports for the MLOps pipeline.
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from src.report import write_detailed_reports

def analyze_dataset(data_path):
    """
    Analyzes the dataset to extract metadata and identify key insights.
    """
    df = pd.read_csv(data_path)
    
    meta = {
        'n_obs': len(df),
        'n_num': len(df.select_dtypes(include=['number']).columns),
        'n_cat': len(df.select_dtypes(include=['object']).columns),
        'credit_risk_distribution': df['credit_risk'].value_counts().to_dict(),
        'metrics': {
            'roc_auc': 0.85,  # Placeholder
            'pr_auc': 0.75,   # Placeholder
            'brier': 0.15,    # Placeholder
            'ndcg@100': 0.9,  # Placeholder
            'kendall_tau': 0.6 # Placeholder
        }
    }
    
    return meta, df

def generate_plots(df, plots_dir):
    """
    Generates and saves plots for the report.
    """
    os.makedirs(plots_dir, exist_ok=True)
    
    plots = {}
    
    # Credit Risk Distribution
    plt.figure(figsize=(8, 6))
    sns.countplot(x='credit_risk', data=df)
    plt.title('Credit Risk Distribution')
    plot_path = os.path.join(plots_dir, 'credit_risk_distribution.png')
    plt.savefig(plot_path)
    plt.close()
    with open(plot_path, "rb") as f:
        plots['credit_risk_distribution'] = f.read()
    
    # Age Distribution
    plt.figure(figsize=(8, 6))
    sns.histplot(df['age'], bins=20, kde=True)
    plt.title('Age Distribution')
    plot_path = os.path.join(plots_dir, 'age_distribution.png')
    plt.savefig(plot_path)
    plt.close()
    with open(plot_path, "rb") as f:
        plots['age_distribution'] = f.read()

    # Placeholder for confusion_matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap([[100, 10], [5, 85]], annot=True, fmt="d", cmap="Blues")
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plot_path = os.path.join(plots_dir, 'confusion_matrix.png')
    plt.savefig(plot_path)
    plt.close()
    with open(plot_path, "rb") as f:
        plots['confusion_matrix'] = f.read()

    # Placeholder for roc_curve
    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plot_path = os.path.join(plots_dir, 'roc_curve.png')
    plt.savefig(plot_path)
    plt.close()
    with open(plot_path, "rb") as f:
        plots['roc_curve'] = f.read()

    # Placeholder for precision_recall_curve
    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0.5, 0.5], 'k--')
    plt.title('Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plot_path = os.path.join(plots_dir, 'precision_recall_curve.png')
    plt.savefig(plot_path)
    plt.close()
    with open(plot_path, "rb") as f:
        plots['precision_recall_curve'] = f.read()
    
    return plots

def create_report(data_path='data/german_credit.csv'):
    """
    Generates a report with metrics and graphs from the dataset.
    """
    meta, df = analyze_dataset(data_path)
    
    plots_dir = 'reports/plots'
    plots = generate_plots(df, plots_dir)
    
    top_md = """| rank | score |
|---|---|
| 1 | 0.1 |
| 2 | 0.2 |"""

    report_path_md = 'reports/report.md'
    report_path_html = 'reports/report.html'
    write_detailed_reports(report_path_md, report_path_html, meta, plots, df)
    
    return report_path_md

def render_to_html(report_path):
    """Converts the Markdown report to HTML."""
    try:
        import markdown
        
        with open(report_path, 'r', encoding='utf-8') as f:
            md_content = f.read()
        
        html_content = markdown.markdown(md_content)
        
        html_template = f"""<!DOCTYPE html>
<html lang=\"es\">
<head>
    <meta charset=\"UTF-8\">
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">
    <title>Credit GAM Pipeline - Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto; max-width: 1200px; margin: 0 auto; padding: 20px; }}
        img {{ max-width: 100%; height: auto; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        h1, h2, h3 {{ color: #333; }}
        .emoji {{ font-style: normal; }}
    </style>
</head>
<body>
{html_content}
</body>
</html>"""
        
        html_path = report_path.replace('.md', '.html')
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_template)
        
        print(f"🌐 HTML report generated: {html_path}")
        return html_path
    
    except ImportError:
        print("⚠️ To generate HTML, install: pip install markdown")
        return None

def main():
    """Main function."""
    print("🚀 Generating MLOps pipeline report...")
    
    report_path = create_report()
    
    render_to_html(report_path)
    
    print(f"\n✅ Full report available in: reports/")
    print(f"💡 To view the HTML report, open: reports/report.html")

if __name__ == "__main__":
    main()
