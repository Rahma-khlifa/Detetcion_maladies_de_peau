# VLM-Bot - Système d'Analyse Dermatologique

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Système modulaire combinant **Vision-Language Model (Phi-3-Vision)** et **RAG** pour l'analyse automatique de lésions dermatologiques.

---

## 🚀 Fonctionnalités

- **🤖 VLM Phi-3-Vision**: Analyse visuelle 4.2B avec quantisation 4-bit (optimisé pour 4GB VRAM)
 - **📚 RAG**: Diagnostic basé sur la littérature médicale avec citations
- **🎨 Gradio**: Interface web interactive

---

## 📋 Prérequis

- **Python**: 3.10 ou 3.11 (recommandé: **3.10**)
- **GPU**: NVIDIA avec CUDA (minimum 4GB VRAM)
  - ✅ Testé sur: RTX 3050 Laptop (4GB), RTX 3060, RTX 4070
  - ⚠️ 4GB VRAM: Quantisation 4-bit obligatoire
- **RAM**: Minimum 8GB système
- **OS**: Windows 10/11, Linux, macOS
- **CUDA**: 11.8+ (compatible avec drivers 450+)

---

## ⚙️ Installation

### 1️⃣ Créer l'environnement Conda

```bash
# Créer l'environnement 'rag' avec Python 3.10
conda create -n rag python=3.10 -y

# Activer l'environnement
conda activate rag
```

### 2️⃣ Installer PyTorch avec CUDA

```bash
# ✅ RECOMMANDÉ: CUDA 12.1 (compatible avec tous les drivers modernes)
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Alternative: CUDA 11.8 (plus stable sur certains systèmes)
# conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Note: PyTorch officiel supporte CUDA 11.8 et 12.1
# Votre driver 591.59 (CUDA 13.1) est compatible avec ces versions
```

### 3️⃣ Installer les dépendances

```bash
# Installer toutes les dépendances depuis requirements.txt
pip install -r requirements.txt
```

### 4️⃣ Configuration

```bash
# Copier le fichier d'exemple
cp .env.example .env

# Éditer .env et ajouter votre token HuggingFace
# HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxx
```

🔑 **Obtenir un token HuggingFace**:
1. Aller sur [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Créer un nouveau token (Read access suffit)
3. Copier le token dans `.env`

---

## 🏗️ Construction de l'Index RAG

Avant la première utilisation, construisez l'index FAISS:

```bash
python scripts/build_index.py
```

Cela télécharge le dataset médical et crée l'index vectoriel (~2-3 minutes).

---

## 🎯 Utilisation

### Interface Gradio

```bash
python src/app.py
```

Puis ouvrir: http://localhost:7860

### API Python

```python
from src.services.vlm_service import VLMService
from src.services.rag_service import RAGService
from src.utils.helpers import load_config
from PIL import Image

# Charger config
config = load_config()

# Initialiser les services
vlm = VLMService(config['models']['vlm'])
rag = RAGService(config['rag'])

# Charger modèles
vlm.load_model()
rag.load_index('data/processed/faiss_index')

# Analyser une image
image = Image.open('lesion.jpg')
rag_results = rag.search("melanoma diagnostic criteria")
rag_context = rag.format_context(rag_results)

# Générer diagnostic
from src.utils.helpers import format_prompt
prompt = format_prompt("", rag_context)
diagnosis = vlm.generate_diagnosis(image, prompt)

print(diagnosis)
```

---

## 📁 Structure du Projet

```
VLM-Bot/
├── config.yaml              # Configuration centrale
├── requirements.txt         # Dépendances Python
├── .env                     # Variables d'environnement (à créer)
├── .env.example             # Template
├── README.md
│
├── src/
│   ├── app.py              # Application Gradio principale
│   ├── services/
│   │   ├── vlm_service.py      # Service VLM (Phi-3-Vision)
│   │   ├── rag_service.py      # Service RAG (FAISS)
│   │   └── (optional) opencv_service.py   # Service OpenCV (removed from runtime)
│   └── utils/
│       └── helpers.py          # Fonctions utilitaires
│
├── scripts/
│   └── build_index.py      # Construction de l'index RAG
│
├── data/
│   ├── raw/                # Données brutes (optionnel)
│   └── processed/          # Index FAISS généré
│       └── faiss_index/
│
└── dermatology_diagnosis_system.ipynb  # Notebook original
```

---

## ⚡ Optimisation pour 4GB VRAM

Le modèle Llava 7B est configuré avec:
- ✅ Quantisation 4-bit (bitsandbytes)
- ✅ Double quantisation
- ✅ `device_map="auto"` (offload intelligent CPU/GPU)
- ✅ `max_memory` limit

Si vous rencontrez des OOM (Out of Memory):

```yaml
# Dans config.yaml, ajuster:
models:
  vlm:
    max_memory:
      0: "3.5GB"  # Réduire la limite GPU
      "cpu": "10GB"  # Augmenter CPU offload
```

---

##1. Vérifier l'installation complète
python scripts/check_installation.py

# 2. Tester l'index RAG
python scripts/build_index.py

# 3. (Optional) If you need classical CV measurements, the OpenCV service is available but not used by default.

# 4. Vérifier GPU et CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}'

# Tester le service OpenCV
python -c "from src.services.opencv_service import OpenCVService; from src.utils.helpers import load_config; from PIL import Image; svc = OpenCVService(load_config()['opencv']); print(svc.analyze_lesion(Image.new('RGB', (512, 512), 'red'))['description'][:200])"
```

---

## 🐛 Dépannage

### Erreur CUDA / PyTorch

```bash
# Vérifier CUDA
nvidia-smi

# Réinstaller PyTorch
pip uninstall torch torchvision torchaudio
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

### Erreur HuggingFace Token

```bash
# Login manuel
huggingface-cli login
```

### Out of Memory GPU

```python
# Forcer CPU dans config.yaml
models:
  vlm:
    device_map: "cpu"
```

---

## 📖 Documentation

- [Phi-3-Vision Model](https://huggingface.co/microsoft/Phi-3-vision-128k-instruct)
- [FAISS](https://github.com/facebookresearch/faiss)
- [LangChain](https://python.langchain.com/)
- [Gradio](https://www.gradio.app/)

---

## ⚠️ Disclaimer

**Usage éducatif et recherche uniquement**. Ce système n'est PAS un dispositif médical et ne remplace en AUCUN CAS un diagnostic professionnel par un dermatologue qualifié.

---

## 📝 Licence

MIT License - voir [LICENSE](LICENSE)

---

## 🤝 Contribution

Les contributions sont bienvenues! Merci d'ouvrir une issue avant de soumettre une PR.

---

**Auteurs**: VLM-Bot Team  
**Version**: 1.0.0  
**Python**: 3.10+
