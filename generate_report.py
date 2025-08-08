#!/usr/bin/env python3
"""
Script para generar y renderizar reportes del pipeline MLOps
"""
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from src.report import write_markdown, write_html

def create_sample_report():
    """Genera un reporte de muestra con métricas y gráficos"""
    
    # Crear directorio de reportes si no existe
    os.makedirs('reports', exist_ok=True)
    os.makedirs('reports/plots', exist_ok=True)
    
    # Dummy data for report generation
    meta = {
        'n_obs': 1000,
        'n_num': 3,
        'n_cat': 5,
        'metrics': {
            'roc_auc': 0.85,
            'pr_auc': 0.75,
            'brier': 0.15,
            'ndcg@100': 0.9,
            'kendall_tau': 0.6
        }
    }
    plots = {
        'age': 'reports/plots/age.png',
        'amount': 'reports/plots/amount.png',
        'duration': 'reports/plots/duration.png',
        'confusion_matrix': 'reports/plots/confusion_matrix.png',
        'roc_curve': 'reports/plots/roc_curve.png',
        'precision_recall_curve': 'reports/plots/precision_recall_curve.png'
    }
    top_md = "| rank | score |\n|---||\n| 1 | 0.1 |\n| 2 | 0.2 |"

    # Generar reporte en Markdown
    write_markdown('reports/report.md', meta, plots, top_md)
    print(f"📄 Reporte generado: reports/report.md")

    # Generar reporte en HTML
    write_html('reports/report.html', meta, plots, top_md)
    print(f"🌐 Reporte HTML generado: reports/report.html")
    
    return 'reports/report.md'

def render_to_html(report_path):
    """Convierte el reporte Markdown a HTML"""
    try:
        import markdown
        
        with open(report_path, 'r', encoding='utf-8') as f:
            md_content = f.read()
        
        html_content = markdown.markdown(md_content)
        
        # Template HTML básico
        html_template = f"""<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Credit GAM Pipeline - Reporte</title>
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
        
        print(f"🌐 Reporte HTML generado: {html_path}")
        return html_path
    
    except ImportError:
        print("⚠️  Para generar HTML instala: pip install markdown")
        return None

def render_to_pdf(report_path):
    """Convierte el reporte a PDF usando pandoc si está disponible"""
    pdf_path = report_path.replace('.md', '.pdf')
    
    # Intentar con pandoc
    pandoc_cmd = f"pandoc {report_path} -o {pdf_path} --pdf-engine=xelatex"
    if os.system("pandoc --version > /dev/null 2>&1") == 0:
        result = os.system(pandoc_cmd)
        if result == 0:
            print(f"📄 Reporte PDF generado: {pdf_path}")
            return pdf_path
        else:
            print("⚠️  Error generando PDF con pandoc")
    
    # Fallback: intentar con markdown y weasyprint
    try:
        html_path = render_to_html(report_path)
        if html_path:
            import weasyprint
            weasyprint.HTML(filename=html_path).write_pdf(pdf_path)
            print(f"📄 Reporte PDF generado: {pdf_path}")
            return pdf_path
    except ImportError:
        print("⚠️  Para generar PDF instala: pip install weasyprint")
    
    return None

def main():
    """Función principal"""
    print("🚀 Generando reporte del pipeline MLOps...")
    
    # Generar reporte
    report_path = create_sample_report()
    
    # Renderizar a diferentes formatos
    render_to_html(report_path)
    render_to_pdf(report_path)
    
    print(f"\n✅ Reporte completo disponible en: reports/")
    print(f"💡 Para ver el reporte HTML, abre: reports/report.html")

if __name__ == "__main__":
    main()