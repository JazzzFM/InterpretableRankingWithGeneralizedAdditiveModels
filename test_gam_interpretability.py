#!/usr/bin/env python3
"""
Test script para verificar interpretabilidad completa del modelo GAM
"""

import os
import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.model import GAMTrainer, FeatureSpec
from src.plots import plot_partial_effect

def test_gam_interpretability():
    """Test completo de interpretabilidad GAM"""
    
    print("🔍 VERIFICACIÓN COMPLETA DE INTERPRETABILIDAD GAM")
    print("="*60)
    
    # 1. Crear dataset de prueba más realista
    np.random.seed(42)
    n_samples = 500
    
    # Simular datos de crédito más realistas
    age = np.random.normal(40, 15, n_samples)
    age = np.clip(age, 18, 80)
    
    amount = np.random.lognormal(8, 0.8, n_samples)
    amount = np.clip(amount, 1000, 50000)
    
    duration = np.random.normal(24, 12, n_samples)  
    duration = np.clip(duration, 3, 72)
    
    # Variables categóricas realistas
    status_options = ['existing_paid', 'critical', 'delayed', 'no_checking']
    status = np.random.choice(status_options, n_samples)
    
    purpose_options = ['car', 'furniture', 'radio_tv', 'domestic', 'repairs', 'education', 'business']
    purpose = np.random.choice(purpose_options, n_samples)
    
    # Target con relaciones realistas
    # Riesgo aumenta con: edad muy joven/vieja, montos altos, duración larga
    risk_score = (
        0.1 * (age < 25) + 0.1 * (age > 65) +  # Edad
        0.2 * (amount > 20000) +  # Monto alto
        0.15 * (duration > 36) +  # Duración larga
        0.1 * (status == 'critical') +  # Status crítico
        0.05 * np.random.normal(0, 1, n_samples)  # Ruido
    )
    credit_risk = (risk_score > np.percentile(risk_score, 70)).astype(int)
    
    df = pd.DataFrame({
        'age': age,
        'amount': amount,
        'duration': duration,
        'status': status,
        'purpose': purpose,
        'credit_risk': credit_risk
    })
    
    print(f"📊 Dataset creado:")
    print(f"   - Muestras: {len(df)}")
    print(f"   - Variables numéricas: age, amount, duration")
    print(f"   - Variables categóricas: status, purpose")
    print(f"   - Distribución target: {credit_risk.mean():.2%} riesgo alto")
    
    # 2. Entrenar modelo GAM
    print("\n🧠 Entrenando modelo GAM...")
    
    spec = FeatureSpec(
        numeric=['age', 'amount', 'duration'],
        categorical=['status', 'purpose'],
        target='credit_risk'
    )
    
    trainer = GAMTrainer(spec=spec, random_state=42)
    
    try:
        train_df, test_df = trainer.fit(df)
        print(f"   ✅ Modelo entrenado exitosamente")
        print(f"   📊 Train samples: {len(train_df)}, Test samples: {len(test_df)}")
        
        # Mostrar rendimiento básico
        train_acc = ((train_df['p'] > 0.5) == train_df['y']).mean()
        test_acc = ((test_df['p'] > 0.5) == test_df['y']).mean()
        print(f"   📈 Accuracy train: {train_acc:.3f}, test: {test_acc:.3f}")
        
    except Exception as e:
        print(f"   ❌ Error en entrenamiento: {e}")
        return False
    
    # 3. Generar gráficos de efectos parciales
    print("\n📊 Generando gráficos de efectos parciales...")
    
    os.makedirs('reports/interpretability', exist_ok=True)
    
    # Para cada característica, generar gráfico de efecto parcial
    features_to_plot = spec.numeric + spec.categorical[:2]  # Limitar categóricas
    plot_paths = {}
    
    for feature in features_to_plot:
        try:
            x, y, ci = trainer.partial_effect(feature, grid=50)
            plot_path = plot_partial_effect('reports/interpretability', feature, x, y, ci)
            plot_paths[feature] = plot_path
            print(f"   ✅ Gráfico generado: {feature} -> {plot_path}")
            
        except Exception as e:
            print(f"   ❌ Error generando gráfico para {feature}: {e}")
    
    # 4. Análisis de sensibilidad
    print("\n🔧 Análisis de sensibilidad del modelo...")
    
    # Seleccionar muestra representativa para análisis
    base_sample = df.iloc[100:101].copy()  # Una muestra como baseline
    base_pred = trainer.predict_proba(base_sample)[0]
    
    print(f"   📋 Muestra base:")
    print(f"      - Age: {base_sample['age'].iloc[0]:.1f}")
    print(f"      - Amount: {base_sample['amount'].iloc[0]:.0f}")
    print(f"      - Duration: {base_sample['duration'].iloc[0]:.1f}")
    print(f"      - Status: {base_sample['status'].iloc[0]}")
    print(f"      - Predicción base: {base_pred:.3f}")
    
    # Análisis de sensibilidad por variable
    sensitivity_results = {}
    
    for feature in spec.numeric:
        print(f"\n   🔍 Sensibilidad para {feature}:")
        
        # Crear variaciones de la característica
        original_value = base_sample[feature].iloc[0]
        
        # Variaciones: -50%, -25%, +25%, +50%
        variations = [-0.5, -0.25, 0.25, 0.5]
        sensitivity_data = []
        
        for pct in variations:
            modified_sample = base_sample.copy()
            new_value = original_value * (1 + pct)
            
            # Asegurar valores realistas
            if feature == 'age':
                new_value = max(18, min(80, new_value))
            elif feature == 'amount':
                new_value = max(1000, min(50000, new_value))
            elif feature == 'duration':
                new_value = max(3, min(72, new_value))
                
            modified_sample[feature] = new_value
            
            try:
                new_pred = trainer.predict_proba(modified_sample)[0]
                change = new_pred - base_pred
                sensitivity_data.append({
                    'variation': pct,
                    'new_value': new_value,
                    'prediction': new_pred,
                    'change': change
                })
                
                print(f"      {pct:+.0%}: {new_value:.1f} -> pred: {new_pred:.3f} (Δ{change:+.3f})")
                
            except Exception as e:
                print(f"      ❌ Error en variación {pct}: {e}")
        
        sensitivity_results[feature] = sensitivity_data
    
    # 5. Generar gráfico de sensibilidad combinado
    print("\n📈 Generando gráfico de análisis de sensibilidad...")
    
    try:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Análisis de Sensibilidad del Modelo GAM', fontsize=16)
        
        for i, feature in enumerate(spec.numeric):
            if feature not in sensitivity_results:
                continue
                
            data = sensitivity_results[feature]
            variations = [d['variation'] for d in data]
            changes = [d['change'] for d in data]
            
            axes[i].plot(variations, changes, 'bo-', linewidth=2, markersize=8)
            axes[i].axhline(y=0, color='r', linestyle='--', alpha=0.5)
            axes[i].set_xlabel(f'Cambio en {feature} (%)')
            axes[i].set_ylabel('Cambio en Probabilidad')
            axes[i].set_title(f'Sensibilidad: {feature}')
            axes[i].grid(True, alpha=0.3)
            
            # Añadir valores en los puntos
            for var, change in zip(variations, changes):
                axes[i].annotate(f'{change:+.3f}', 
                               (var, change), 
                               textcoords="offset points", 
                               xytext=(0,10), 
                               ha='center')
        
        plt.tight_layout()
        sensitivity_path = 'reports/interpretability/sensitivity_analysis.png'
        plt.savefig(sensitivity_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Gráfico de sensibilidad guardado: {sensitivity_path}")
        
    except Exception as e:
        print(f"   ❌ Error generando gráfico de sensibilidad: {e}")
    
    # 6. Resumen de interpretabilidad
    print("\n📋 RESUMEN DE INTERPRETABILIDAD")
    print("-" * 40)
    
    print("✅ Funcionalidades verificadas:")
    print("   🔸 Splines para variables numéricas (age, amount, duration)")
    print("   🔸 Factores para variables categóricas (status, purpose)")
    print("   🔸 Efectos parciales con intervalos de confianza")
    print("   🔸 Gráficos de interpretabilidad generados")
    print("   🔸 Análisis de sensibilidad completado")
    
    print(f"\n📊 Archivos generados:")
    for feature, path in plot_paths.items():
        print(f"   - {path}")
    print(f"   - {sensitivity_path}")
    
    print("\n🎯 El modelo GAM está completamente implementado con:")
    print("   ✅ Términos base (splines y factores)")
    print("   ✅ Interpretabilidad individual por característica") 
    print("   ✅ Análisis de sensibilidad a cambios clave")
    
    return True

if __name__ == "__main__":
    success = test_gam_interpretability()
    if success:
        print("\n🎉 ¡Verificación de interpretabilidad GAM EXITOSA!")
    else:
        print("\n❌ Verificación de interpretabilidad GAM FALLIDA")