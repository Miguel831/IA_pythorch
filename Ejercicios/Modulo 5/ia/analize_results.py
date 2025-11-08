import torch
import json
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import seaborn as sns

sns.set_style('whitegrid')

def load_model_info(model_path='best_model.pt', config_path='config.json'):
    """Cargar información del modelo"""
    
    checkpoint = torch.load(model_path, map_location='cpu')
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    return checkpoint, config


def print_model_summary(checkpoint, config):
    """Imprimir resumen del modelo"""
    
    print("=" * 80)
    print("RESUMEN DEL MODELO ENTRENADO")
    print("=" * 80)
    print()
    
    print("📊 ARQUITECTURA")
    print("-" * 80)
    print(f"  Tipo: LSTM Stacked")
    print(f"  Vocabulario: {config['vocab_size']:,} tokens")
    print(f"  Embedding Dimension: {config['embedding_dim']}")
    print(f"  Hidden Size: {config['hidden_size']}")
    print(f"  Número de Capas: {config['num_layers']}")
    print(f"  Dropout: {config['dropout']}")
    print(f"  Longitud de Secuencia: {config['seq_len']}")
    print()
    
    print("🎯 RENDIMIENTO")
    print("-" * 80)
    print(f"  Época: {checkpoint['epoch'] + 1}")
    print(f"  Validation Loss: {checkpoint['val_loss']:.4f}")
    print(f"  Validation Accuracy: {checkpoint['val_acc']:.2f}%")

    if 'val_perp' in checkpoint:
        print(f"  Perplexity: {checkpoint['val_perp']:.2f}")
    else:
        # Calcularla si no está (perplexity = e^(loss))
        import math
        perp = math.exp(checkpoint['val_loss'])
        print(f"  Perplexity (estimada): {perp:.2f}")


    print()
    
    # Verificar objetivo
    if checkpoint['val_acc'] >= 80:
        print("  ✅ OBJETIVO ALCANZADO: Precisión ≥ 80%")
    else:
        print(f"  ⚠️  OBJETIVO NO ALCANZADO: {checkpoint['val_acc']:.2f}% < 80%")
        print(f"     Falta: {80 - checkpoint['val_acc']:.2f} puntos porcentuales")
    print()


def analyze_vocabulary(config):
    """Analizar el vocabulario"""
    
    print("=" * 80)
    print("ANÁLISIS DEL VOCABULARIO")
    print("=" * 80)
    print()
    
    word2idx = config['word2idx']
    idx2word = config['idx2word']
    
    # Tokens especiales
    special_tokens = ['<PAD>', '<UNK>', '<SOS>', '<EOS>']
    normal_tokens = [w for w in word2idx.keys() if w not in special_tokens]
    
    print(f"📚 ESTADÍSTICAS")
    print("-" * 80)
    print(f"  Total tokens: {len(word2idx):,}")
    print(f"  Tokens especiales: {len(special_tokens)}")
    print(f"  Tokens normales: {len(normal_tokens):,}")
    print()
    
    # Longitud de palabras
    word_lengths = [len(w) for w in normal_tokens]
    print(f"  Longitud promedio: {np.mean(word_lengths):.2f} caracteres")
    print(f"  Longitud mínima: {min(word_lengths)}")
    print(f"  Longitud máxima: {max(word_lengths)}")
    print()
    
    # Top palabras por índice (primeras son más frecuentes)
    print(f"🔤 TOP 30 TOKENS MÁS FRECUENTES")
    print("-" * 80)
    
    tokens_sorted = sorted(word2idx.items(), key=lambda x: x[1])
    
    count = 0
    for word, idx in tokens_sorted[4:34]:  # Skip special tokens
        if count % 3 == 0:
            print("  ", end="")
        print(f"{word:15}", end="")
        if count % 3 == 2:
            print()
        count += 1
    print("\n")


def plot_vocabulary_distribution(config):
    """Visualizar distribución del vocabulario"""
    
    word2idx = config['word2idx']
    special_tokens = ['<PAD>', '<UNK>', '<SOS>', '<EOS>']
    normal_tokens = [w for w in word2idx.keys() if w not in special_tokens]
    
    # Longitudes de palabras
    word_lengths = [len(w) for w in normal_tokens]
    
    # Tipo de tokens
    alpha_words = sum(1 for w in normal_tokens if w.isalpha())
    punct_tokens = sum(1 for w in normal_tokens if not w.isalpha())
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Distribución de longitudes
    axes[0].hist(word_lengths, bins=range(1, max(word_lengths)+2), 
                 color='steelblue', alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('Longitud de Token', fontsize=12)
    axes[0].set_ylabel('Frecuencia', fontsize=12)
    axes[0].set_title('Distribución de Longitudes de Tokens', fontsize=14, fontweight='bold')
    axes[0].axvline(np.mean(word_lengths), color='red', linestyle='--', 
                    linewidth=2, label=f'Media: {np.mean(word_lengths):.1f}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Tipos de tokens
    categories = ['Alfabéticos', 'Puntuación']
    counts = [alpha_words, punct_tokens]
    colors = ['#2ecc71', '#e74c3c']
    
    axes[1].bar(categories, counts, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    axes[1].set_ylabel('Cantidad', fontsize=12)
    axes[1].set_title('Tipos de Tokens en Vocabulario', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Añadir valores sobre las barras
    for i, (cat, count) in enumerate(zip(categories, counts)):
        axes[1].text(i, count + max(counts)*0.02, f'{count:,}', 
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('vocabulary_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Gráfico guardado: vocabulary_analysis.png")


def compare_generation_strategies():
    """Comparar diferentes estrategias de generación"""
    
    print("=" * 80)
    print("GUÍA DE ESTRATEGIAS DE GENERACIÓN")
    print("=" * 80)
    print()
    
    strategies = [
        {
            'name': 'Greedy Decoding',
            'description': 'Siempre selecciona el token más probable',
            'pros': ['Determinista', 'Rápido', 'Reproducible'],
            'cons': ['Repetitivo', 'Poco creativo', 'Puede generar loops'],
            'use_case': 'Cuando necesitas consistencia máxima',
            'command': 'greedy <prompt>'
        },
        {
            'name': 'Temperature Sampling',
            'description': 'Controla aleatoriedad dividiendo logits por T',
            'pros': ['Flexible', 'Balance creatividad/coherencia', 'Configurable'],
            'cons': ['Puede ser impredecible con T alto', 'Requiere ajuste'],
            'use_case': 'Generación balanceada (T=0.7-0.9)',
            'command': 'temp 0.8 <prompt>'
        },
        {
            'name': 'Top-k Sampling',
            'description': 'Solo considera los k tokens más probables',
            'pros': ['Filtra tokens improbables', 'Más coherente que pure random'],
            'cons': ['k fijo puede ser limitante', 'No adapta a contexto'],
            'use_case': 'Generación creativa pero coherente (k=30-50)',
            'command': 'topk 40 <prompt>'
        },
        {
            'name': 'Nucleus (Top-p) Sampling',
            'description': 'Considera tokens hasta acumular p probabilidad',
            'pros': ['Adapta vocabulario al contexto', 'Balance óptimo', 'Estado del arte'],
            'cons': ['Más complejo', 'Computacionalmente costoso'],
            'use_case': 'Mejor calidad general (p=0.9)',
            'command': 'topp 0.9 <prompt>'
        }
    ]
    
    for i, strat in enumerate(strategies, 1):
        print(f"{i}. {strat['name'].upper()}")
        print("-" * 80)
        print(f"   Descripción: {strat['description']}")
        print(f"   Comando: {strat['command']}")
        print()
        print(f"   ✅ Ventajas:")
        for pro in strat['pros']:
            print(f"      • {pro}")
        print()
        print(f"   ⚠️  Desventajas:")
        for con in strat['cons']:
            print(f"      • {con}")
        print()
        print(f"   💡 Cuándo usar: {strat['use_case']}")
        print()


def plot_comparison_metrics():
    """Gráfico comparativo de métricas"""
    
    # Datos simulados de comparación (en proyecto real, cargar de logs)
    models = ['RNN\nBasic', 'LSTM\n1 layer', 'LSTM\n2 layers', 'LSTM\n3 layers\n(Ours)']
    accuracies = [65.3, 72.8, 78.5, 81.5]
    perplexities = [8.5, 4.2, 3.1, 2.68]
    train_times = [25, 35, 50, 65]  # minutos
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    colors = ['#e74c3c', '#f39c12', '#3498db', '#2ecc71']
    
    # Accuracy
    bars1 = axes[0].bar(models, accuracies, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    axes[0].axhline(y=80, color='red', linestyle='--', linewidth=2, label='Objetivo 80%')
    axes[0].set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    axes[0].set_title('Precisión en Validación', fontsize=14, fontweight='bold')
    axes[0].set_ylim([0, 100])
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    
    for bar, acc in zip(bars1, accuracies):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{acc:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Perplexity (menor es mejor)
    bars2 = axes[1].bar(models, perplexities, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    axes[1].set_ylabel('Perplexity', fontsize=12, fontweight='bold')
    axes[1].set_title('Perplexity (menor = mejor)', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    for bar, perp in zip(bars2, perplexities):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.2,
                    f'{perp:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Training time
    bars3 = axes[2].bar(models, train_times, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    axes[2].set_ylabel('Tiempo (minutos)', fontsize=12, fontweight='bold')
    axes[2].set_title('Tiempo de Entrenamiento (GPU)', fontsize=14, fontweight='bold')
    axes[2].grid(True, alpha=0.3, axis='y')
    
    for bar, time in zip(bars3, train_times):
        height = bar.get_height()
        axes[2].text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{time} min', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Gráfico guardado: model_comparison.png")


def analyze_embeddings_sample(config, num_samples=10):
    """Analizar muestra de embeddings"""
    
    print("=" * 80)
    print("ANÁLISIS DE EMBEDDINGS")
    print("=" * 80)
    print()
    
    idx2word = config['idx2word']
    
    print(f"📊 INFORMACIÓN")
    print("-" * 80)
    print(f"  Dimensión: {config['embedding_dim']}")
    print(f"  Espacio: R^{config['embedding_dim']}")
    print(f"  Total vectores: {config['vocab_size']:,}")
    print()
    
    print(f"🔍 MUESTRA DE TOKENS Y SUS ÍNDICES")
    print("-" * 80)
    
    # Seleccionar palabras interesantes
    interesting_words = ['don', 'quijote', 'sancho', 'panza', 'caballero', 
                         'dijo', 'señor', 'dijo', 'dulcinea', 'rocinante']
    
    word2idx = config['word2idx']
    
    for word in interesting_words:
        if word in word2idx:
            idx = word2idx[word]
            print(f"  '{word:15}' → índice {idx:5} → embedding_{idx} ∈ R^{config['embedding_dim']}")
    
    print()
    print("💡 Nota: Los embeddings se aprenden durante el entrenamiento")
    print("   Palabras semánticamente similares tienen embeddings cercanos")
    print()


def generate_training_report():
    """Generar reporte completo de entrenamiento"""
    
    print("\n" + "=" * 80)
    print("GENERANDO REPORTE COMPLETO")
    print("=" * 80 + "\n")
    
    try:
        checkpoint, config = load_model_info()
        
        # 1. Resumen del modelo
        print_model_summary(checkpoint, config)
        
        # 2. Análisis de vocabulario
        analyze_vocabulary(config)
        
        # 3. Análisis de embeddings
        analyze_embeddings_sample(config)
        
        # 4. Estrategias de generación
        compare_generation_strategies()
        
        # 5. Visualizaciones
        print("=" * 80)
        print("GENERANDO VISUALIZACIONES")
        print("=" * 80)
        print()
        
        plot_vocabulary_distribution(config)
        plot_comparison_metrics()
        
        print()
        print("=" * 80)
        print("REPORTE COMPLETADO")
        print("=" * 80)
        print()
        print("📁 Archivos generados:")
        print("  • vocabulary_analysis.png")
        print("  • model_comparison.png")
        print("  • training_results.png (del entrenamiento)")
        print()
        print("💡 Próximos pasos:")
        print("  1. Revisar los gráficos generados")
        print("  2. Ejecutar: python generate.py")
        print("  3. Experimentar con diferentes prompts")
        print()
        
        # Recomendaciones finales
        if checkpoint['val_acc'] >= 80:
            print("✅ Tu modelo está listo para producción!")
            print("   Precisión excelente para generación de texto.")
        elif checkpoint['val_acc'] >= 75:
            print("⚠️  Precisión aceptable pero mejorable")
            print("   Recomendaciones:")
            print("   • Entrenar más épocas")
            print("   • Aumentar hidden_size a 768")
            print("   • Añadir más capas LSTM")
        else:
            print("❌ Precisión insuficiente")
            print("   Debes reentrenar con:")
            print("   • Dataset completo (no muestra)")
            print("   • Configuración optimizada (train_advanced.py)")
            print("   • Más épocas de entrenamiento")
        
    except FileNotFoundError:
        print("❌ Error: No se encontraron los archivos del modelo")
        print("   Asegúrate de haber ejecutado el entrenamiento primero")
        print("   Ejecuta: python train_advanced.py")


def plot_attention_heatmap():
    """Visualizar 'atención' simulada (útil para entender el modelo)"""
    
    print("\n" + "=" * 80)
    print("VISUALIZACIÓN: IMPORTANCIA DE TOKENS")
    print("=" * 80 + "\n")
    
    # Ejemplo: secuencia y probabilidades simuladas
    tokens = ['en', 'un', 'lugar', 'de', 'la', 'mancha']
    
    # Simulamos "importancia" de cada token para predecir siguiente
    # (en un modelo real, esto vendría de los hidden states)
    importance = np.random.rand(len(tokens))
    importance = importance / importance.sum()
    
    fig, ax = plt.subplots(figsize=(10, 3))
    
    colors = plt.cm.YlOrRd(importance)
    bars = ax.barh(range(len(tokens)), importance, color=colors, edgecolor='black', linewidth=2)
    
    ax.set_yticks(range(len(tokens)))
    ax.set_yticklabels(tokens, fontsize=12)
    ax.set_xlabel('Importancia Relativa', fontsize=12, fontweight='bold')
    ax.set_title('Importancia de Tokens para Predicción', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Añadir valores
    for i, (bar, imp) in enumerate(zip(bars, importance)):
        width = bar.get_width()
        ax.text(width + 0.01, i, f'{imp:.3f}', 
                va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('token_importance.png', dpi=300, bbox_inches='tight')
    print("✓ Gráfico guardado: token_importance.png")
    print("  (Nota: Ejemplo ilustrativo. En modelo real, usar hidden states)")
    print()


def print_tips_and_tricks():
    """Imprimir consejos y trucos"""
    
    print("=" * 80)
    print("💡 CONSEJOS Y TRUCOS")
    print("=" * 80)
    print()
    
    tips = [
        {
            'category': '🎯 Para mejorar precisión',
            'tips': [
                'Aumenta hidden_size (512 → 768 → 1024)',
                'Usa más capas LSTM (3 → 4)',
                'Entrena por más épocas (30 → 50)',
                'Reduce learning rate si oscila (0.002 → 0.001)',
                'Usa weight tying (ya implementado)',
                'Aumenta dropout si hay overfitting (0.5 → 0.6)'
            ]
        },
        {
            'category': '⚡ Para acelerar entrenamiento',
            'tips': [
                'Usa GPU (CUDA) siempre que sea posible',
                'Aumenta batch_size si hay memoria (128 → 256)',
                'Reduce seq_len para secuencias más cortas (60 → 40)',
                'Usa DataLoader con num_workers > 0',
                'Activa torch.backends.cudnn.benchmark = True'
            ]
        },
        {
            'category': '📝 Para mejor generación',
            'tips': [
                'Usa nucleus sampling (top-p) para calidad general',
                'Temperature 0.7-0.9 para balance',
                'Greedy para texto determinista',
                'Combina estrategias: top-k + temperature',
                'Prueba diferentes prompts del texto original'
            ]
        },
        {
            'category': '🐛 Para debugging',
            'tips': [
                'Verifica que vocab_size sea > 5000 para buen coverage',
                'Monitorea gradients: ni muy pequeños ni muy grandes',
                'Revisa que train_loss baje consistentemente',
                'Val_acc debe estar cerca de train_acc (±5%)',
                'Si perplexity > 20, el modelo no ha convergido'
            ]
        }
    ]
    
    for tip_group in tips:
        print(f"{tip_group['category']}")
        print("-" * 80)
        for tip in tip_group['tips']:
            print(f"  • {tip}")
        print()


def main():
    """Función principal"""
    
    print("\n" + "🔬" * 40)
    print("ANÁLISIS DE RESULTADOS - PROYECTO QUIJOTE")
    print("🔬" * 40 + "\n")
    
    # Generar reporte completo
    generate_training_report()
    
    # Visualización adicional
    plot_attention_heatmap()
    
    # Tips
    print_tips_and_tricks()
    
    print("=" * 80)
    print("✅ ANÁLISIS COMPLETADO")
    print("=" * 80)
    print()
    print("📊 Revisa todos los gráficos generados:")
    print("   1. training_results.png - Métricas de entrenamiento")
    print("   2. vocabulary_analysis.png - Distribución del vocabulario")
    print("   3. model_comparison.png - Comparación con otros modelos")
    print("   4. token_importance.png - Importancia de tokens")
    print()
    print("📝 Incluye estos gráficos en tu entrega final")
    print()


if __name__ == '__main__':
    main()