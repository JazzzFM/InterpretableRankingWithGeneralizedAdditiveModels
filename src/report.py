import os
from textwrap import dedent

def write_markdown(path, meta: dict, plots: dict, top_md: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    md = f"""# Credit Risk Ranking Report

## 1. Resumen ejecutivo
- Observaciones: {meta['n_obs']}
- Variables numéricas: {meta['n_num']}
- Variables categóricas: {meta['n_cat']}
- Métricas (test):
  - ROC-AUC: {meta['metrics']['roc_auc']:.3f}
  - PR-AUC: {meta['metrics']['pr_auc']:.3f}
  - Brier: {meta['metrics']['brier']:.3f}
  - nDCG@100: {meta['metrics']['ndcg@100']:.3f}
  - Kendall τ: {meta['metrics']['kendall_tau']:.3f}

## 2. Top 10 solicitudes (menor riesgo primero)
{top_md}

## 3. Efectos parciales por variable
"""
    for feat, img in plots.items():
        rel = os.path.relpath(img, os.path.dirname(path))
        md += f"\n### {feat}\n![]({rel})\n"

    md += dedent("""

## 4. Sensibilidad (what-if)
Se realizaron perturbaciones ±10–20% en variables clave y se observó el cambio en el score.

## 5. Calibración
Se evalúa Brier y curvas de confiabilidad. Si Brier elevado, considerar calibración isotónica.

## 6. Consideraciones éticas
Revisar efectos de variables sensibles/proxies. Evitar uso directo de atributos protegidos.

## 7. Limitaciones
Dataset UCI; no incluye reject inference. Resultados sujetos a *concept drift*.
""")
    with open(path, "w") as f:
        f.write(md)
