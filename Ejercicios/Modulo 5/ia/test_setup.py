"""
Script de verificación del entorno y componentes del proyecto
Ejecutar antes de entrenar para verificar que todo está correctamente configurado
"""

import sys
import os

def test_python_version():
    """Verificar versión de Python"""
    print("🐍 Verificando versión de Python...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"   ✅ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"   ❌ Python {version.major}.{version.minor} (se requiere 3.8+)")
        return False


def test_pytorch():
    """Verificar instalación de PyTorch"""
    print("\n🔥 Verificando PyTorch...")
    try:
        import torch
        print(f"   ✅ PyTorch {torch.__version__}")
        
        # Verificar CUDA
        if torch.cuda.is_available():
            print(f"   ✅ CUDA disponible: {torch.cuda.get_device_name(0)}")
            print(f"      Memoria GPU: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print(f"   ⚠️  CUDA no disponible (se usará CPU - más lento)")
        
        return True
    except ImportError:
        print("   ❌ PyTorch no instalado")
        print("      Instalar: pip install torch")
        return False


def test_dependencies():
    """Verificar dependencias"""
    print("\n📦 Verificando dependencias...")
    
    required = {
        'numpy': 'NumPy',
        'matplotlib': 'Matplotlib',
        'tqdm': 'tqdm',
        'requests': 'Requests'
    }
    
    all_ok = True
    for module, name in required.items():
        try:
            __import__(module)
            print(f"   ✅ {name}")
        except ImportError:
            print(f"   ❌ {name} no instalado")
            print(f"      Instalar: pip install {module}")
            all_ok = False
    
    return all_ok


def test_memory():
    """Verificar memoria disponible"""
    print("\n💾 Verificando memoria del sistema...")
    try:
        import psutil
        mem = psutil.virtual_memory()
        total_gb = mem.total / (1024**3)
        available_gb = mem.available / (1024**3)
        
        print(f"   Total: {total_gb:.1f} GB")
        print(f"   Disponible: {available_gb:.1f} GB")
        
        if available_gb < 4:
            print(f"   ⚠️  Memoria baja. Recomendado: 8GB+")
            print(f"      Considera reducir batch_size")
            return False
        elif available_gb < 8:
            print(f"   ⚠️  Memoria justa. Funcional pero limitado")
            return True
        else:
            print(f"   ✅ Memoria suficiente")
            return True
            
    except ImportError:
        print("   ⚠️  psutil no instalado (verificación omitida)")
        return True


def test_dataset():
    """Verificar dataset"""
    print("\n📚 Verificando dataset...")
    
    if os.path.exists('quijote.txt'):
        size = os.path.getsize('quijote.txt')
        size_mb = size / (1024 * 1024)
        print(f"   ✅ quijote.txt encontrado ({size_mb:.2f} MB)")
        
        # Verificar contenido
        with open('quijote.txt', 'r', encoding='utf-8') as f:
            text = f.read()
            words = len(text.split())
            chars = len(text)
            
            print(f"      Caracteres: {chars:,}")
            print(f"      Palabras: {words:,}")
            
            if words < 100000:
                print(f"   ⚠️  Dataset parece pequeño (< 100k palabras)")
                print(f"      Para mejores resultados, usa El Quijote completo")
                return False
            else:
                print(f"   ✅ Dataset de buen tamaño")
                return True
    else:
        print("   ❌ quijote.txt no encontrado")
        print("      Ejecutar: python download_quijote.py")
        return False


def test_model_architecture():
    """Probar creación del modelo"""
    print("\n🏗️  Verificando arquitectura del modelo...")
    
    try:
        import torch
        import torch.nn as nn
        
        # Crear modelo pequeño de prueba
        class TestLSTM(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(1000, 128)
                self.lstm = nn.LSTM(128, 256, num_layers=2, batch_first=True)
                self.fc = nn.Linear(256, 1000)
            
            def forward(self, x):
                emb = self.embedding(x)
                out, _ = self.lstm(emb)
                logits = self.fc(out)
                return logits
        
        model = TestLSTM()
        
        # Test forward pass
        x = torch.randint(0, 1000, (2, 10))  # batch=2, seq_len=10
        output = model(x)
        
        assert output.shape == (2, 10, 1000), "Error en dimensiones"
        
        print("   ✅ Modelo LSTM funcional")
        print(f"      Parámetros: {sum(p.numel() for p in model.parameters()):,}")
        return True
        
    except Exception as e:
        print(f"   ❌ Error al crear modelo: {e}")
        return False


def test_training_pipeline():
    """Probar pipeline de entrenamiento básico"""
    print("\n🎯 Verificando pipeline de entrenamiento...")
    
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import Dataset, DataLoader
        
        # Dataset dummy
        class DummyDataset(Dataset):
            def __len__(self):
                return 100
            
            def __getitem__(self, idx):
                x = torch.randint(0, 100, (20,))
                y = torch.randint(0, 100, (20,))
                return x, y
        
        # Modelo dummy
        model = nn.Sequential(
            nn.Embedding(100, 64),
            nn.LSTM(64, 128, batch_first=True)[0],
            nn.Linear(128, 100)
        )
        
        # Entrenamiento dummy
        dataset = DummyDataset()
        loader = DataLoader(dataset, batch_size=8, shuffle=True)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Una iteración
        x, y = next(iter(loader))
        optimizer.zero_grad()
        output = model(x)[0] if isinstance(model(x), tuple) else model(x)
        loss = criterion(output.reshape(-1, 100), y.reshape(-1))
        loss.backward()
        optimizer.step()
        
        print("   ✅ Pipeline de entrenamiento funcional")
        print(f"      Loss inicial: {loss.item():.4f}")
        return True
        
    except Exception as e:
        print(f"   ❌ Error en pipeline: {e}")
        return False


def test_file_permissions():
    """Verificar permisos de escritura"""
    print("\n📝 Verificando permisos de escritura...")
    
    try:
        # Intentar crear archivo temporal
        with open('test_write.tmp', 'w') as f:
            f.write('test')
        os.remove('test_write.tmp')
        print("   ✅ Permisos de escritura OK")
        return True
    except Exception as e:
        print(f"   ❌ Error de permisos: {e}")
        return False


def test_text_preprocessing():
    """Verificar preprocesamiento de texto"""
    print("\n🔤 Verificando preprocesamiento...")
    
    try:
        import re
        from collections import Counter
        
        test_text = "En un lugar de la Mancha, de cuyo nombre no quiero acordarme."
        
        # Tokenización básica
        text = test_text.lower()
        text = re.sub(r'([.,;:!?])', r' \1 ', text)
        text = re.sub(r'\s+', ' ', text)
        tokens = text.split()
        
        print(f"   Texto original: {test_text}")
        print(f"   Tokens: {len(tokens)}")
        print(f"   Primeros 5: {tokens[:5]}")
        
        if len(tokens) > 0:
            print("   ✅ Preprocesamiento funcional")
            return True
        else:
            print("   ❌ Error en tokenización")
            return False
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def estimate_training_time():
    """Estimar tiempo de entrenamiento"""
    print("\n⏱️  Estimando tiempo de entrenamiento...")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            # Con GPU
            print("   Con GPU:")
            print("      ~2-3 minutos por época")
            print("      ~30-60 minutos total (30 épocas)")
        else:
            # Sin GPU
            print("   Sin GPU (CPU):")
            print("      ~10-15 minutos por época")
            print("      ~3-6 horas total (30 épocas)")
            print()
            print("   💡 Recomendación: Considera usar Google Colab con GPU gratuita")
        
        return True
        
    except:
        return True


def print_recommendations(results):
    """Imprimir recomendaciones basadas en resultados"""
    
    print("\n" + "=" * 80)
    print("📋 RECOMENDACIONES")
    print("=" * 80)
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n✅ Todos los tests pasaron exitosamente!")
        print("\n🚀 Estás listo para entrenar:")
        print("   1. python train_advanced.py")
        print("   2. Espera a que termine (~30-60 min con GPU)")
        print("   3. python generate.py")
        print("   4. python analyze_results.py")
    else:
        print("\n⚠️  Algunos tests fallaron. Revisa los errores arriba.")
        print("\n🔧 Pasos recomendados:")
        
        if not results.get('pytorch', False):
            print("   • Instalar PyTorch: pip install torch")
        
        if not results.get('dependencies', False):
            print("   • Instalar dependencias: pip install numpy matplotlib tqdm requests")
        
        if not results.get('dataset', False):
            print("   • Descargar dataset: python download_quijote.py")
        
        if not results.get('memory', False):
            print("   • Reducir batch_size en train_advanced.py (128 → 64)")
            print("   • Cerrar otras aplicaciones para liberar RAM")
    
    print()


def main():
    """Ejecutar todos los tests"""
    
    print("=" * 80)
    print("🧪 VERIFICACIÓN DEL SISTEMA - PROYECTO QUIJOTE")
    print("=" * 80)
    
    results = {
        'python': test_python_version(),
        'pytorch': test_pytorch(),
        'dependencies': test_dependencies(),
        'memory': test_memory(),
        'dataset': test_dataset(),
        'model': test_model_architecture(),
        'training': test_training_pipeline(),
        'permissions': test_file_permissions(),
        'preprocessing': test_text_preprocessing()
    }
    
    estimate_training_time()
    
    # Resumen
    print("\n" + "=" * 80)
    print("📊 RESUMEN")
    print("=" * 80)
    
    passed = sum(results.values())
    total = len(results)
    
    print(f"\nTests pasados: {passed}/{total}")
    print()
    
    for test, result in results.items():
        status = "✅" if result else "❌"
        print(f"   {status} {test.capitalize()}")
    
    print_recommendations(results)
    
    print("=" * 80)
    print("✅ VERIFICACIÓN COMPLETADA")
    print("=" * 80)
    print()
    
    if passed == total:
        return 0
    else:
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)