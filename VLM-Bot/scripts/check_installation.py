"""
Script de vérification de l'installation - VLM-Bot
Vérifie toutes les dépendances et la configuration GPU.
"""

import sys
from pathlib import Path

print("="*80)
print("🔍 VLM-Bot - Vérification de l'installation")
print("="*80)

# ============================================================================
# 1. Vérifier Python
# ============================================================================
print(f"\n✅ Python: {sys.version}")
python_version = sys.version_info
if python_version.major == 3 and python_version.minor >= 10:
    print(f"   Version OK (3.{python_version.minor})")
else:
    print(f"   ⚠️  Recommandé: Python 3.10 ou 3.11")

# ============================================================================
# 2. Vérifier PyTorch et CUDA
# ============================================================================
print("\n" + "-"*80)
print("PyTorch & CUDA:")
print("-"*80)

try:
    import torch
    print(f"✅ PyTorch: {torch.__version__}")
    
    # CUDA
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        print(f"✅ CUDA disponible: {torch.version.cuda}")
        print(f"   Device: {torch.cuda.get_device_name(0)}")
        
        # Mémoire GPU
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        
        print(f"   VRAM totale: {total_memory:.2f} GB")
        print(f"   VRAM utilisée: {allocated:.2f} GB")
        print(f"   VRAM réservée: {reserved:.2f} GB")
        
        if total_memory <= 4.5:
            print(f"   ⚠️  4GB VRAM détecté - Quantisation 4-bit OBLIGATOIRE")
    else:
        print("⚠️  CUDA non disponible - Mode CPU uniquement")
        print("   Installation: conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia")
        
except ImportError:
    print("❌ PyTorch non installé!")
    print("   Installation: conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia")
    sys.exit(1)

# ============================================================================
# 3. Vérifier les dépendances critiques
# ============================================================================
print("\n" + "-"*80)
print("Dépendances critiques:")
print("-"*80)

critical_deps = {
    'transformers': 'Modèles VLM',
    'accelerate': 'Accélération GPU',
    'bitsandbytes': 'Quantisation 4-bit',
    'peft': 'LoRA adapters',
    'langchain': 'RAG framework',
    'langchain_community': 'RAG vectorstores',
    'faiss': 'Index vectoriel (faiss-cpu)',
    'sentence_transformers': 'Embeddings',
    'datasets': 'Datasets HuggingFace',
    'cv2': 'OpenCV (opencv-python-headless)',
    'gradio': 'Interface web',
    'PIL': 'Images (Pillow)',
    'sklearn': 'ML utils (scikit-learn)',
    'yaml': 'Config (pyyaml)',
    'dotenv': 'Env vars (python-dotenv)',
}

missing = []
for module, desc in critical_deps.items():
    try:
        __import__(module)
        print(f"✅ {module:25s} - {desc}")
    except ImportError:
        print(f"❌ {module:25s} - {desc} (MANQUANT)")
        missing.append(module)

if missing:
    print(f"\n⚠️  {len(missing)} dépendances manquantes!")
    print("   Installation: pip install -r requirements.txt")
else:
    print(f"\n✅ Toutes les {len(critical_deps)} dépendances critiques sont installées!")

# ============================================================================
# 4. Vérifier dépendances optionnelles
# ============================================================================
print("\n" + "-"*80)
print("Dépendances optionnelles:")
print("-"*80)

optional_deps = {
    'matplotlib': 'Visualisations',
    'psutil': 'Monitoring système',
    'gputil': 'Monitoring GPU',
}

for module, desc in optional_deps.items():
    try:
        __import__(module)
        print(f"✅ {module:15s} - {desc}")
    except ImportError:
        print(f"⚪ {module:15s} - {desc} (optionnel)")

# ============================================================================
# 5. Vérifier la structure du projet
# ============================================================================
print("\n" + "-"*80)
print("Structure du projet:")
print("-"*80)

required_files = [
    'config.yaml',
    '.env',
    'src/app.py',
    'src/services/vlm_service.py',
    'src/services/rag_service.py',
    'src/utils/helpers.py',
    'scripts/build_index.py',
]

for file_path in required_files:
    if Path(file_path).exists():
        print(f"✅ {file_path}")
    else:
        print(f"❌ {file_path} (MANQUANT)")
        if file_path == '.env':
            print(f"   → Copier .env.example vers .env et configurer HF_TOKEN")

# ============================================================================
# 6. Vérifier l'index RAG
# ============================================================================
print("\n" + "-"*80)
print("Index RAG:")
print("-"*80)

index_path = Path('data/processed/faiss_index')
if index_path.exists():
    print(f"✅ Index FAISS trouvé: {index_path}")
else:
    print(f"⚠️  Index FAISS non trouvé: {index_path}")
    print(f"   → Exécuter: python scripts/build_index.py")

# ============================================================================
# 7. Test de quantisation 4-bit
# ============================================================================
print("\n" + "-"*80)
print("Test de quantisation 4-bit:")
print("-"*80)

try:
    from transformers import BitsAndBytesConfig
    import torch
    
    config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    print("✅ Configuration 4-bit OK")
    print("   Prêt pour charger Phi-3-Vision sur 4GB VRAM")
    
except Exception as e:
    print(f"❌ Erreur quantisation: {e}")
    print("   Vérifier bitsandbytes et CUDA")

# ============================================================================
# Résumé final
# ============================================================================
print("\n" + "="*80)
if missing:
    print("⚠️  INSTALLATION INCOMPLÈTE")
    print(f"   → Installer les dépendances manquantes")
    print(f"   → pip install -r requirements.txt")
elif not Path('.env').exists():
    print("⚠️  CONFIGURATION MANQUANTE")
    print(f"   → Copier .env.example vers .env")
    print(f"   → Ajouter votre HF_TOKEN dans .env")
elif not index_path.exists():
    print("⚠️  INDEX RAG MANQUANT")
    print(f"   → Exécuter: python scripts/build_index.py")
else:
    print("✅ INSTALLATION COMPLÈTE!")
    print(f"   → Prêt à lancer: python src/app.py")
print("="*80)
