#!/bin/bash
# 🚀 Script d'Installation Automatique pour Serveur GPU
# =====================================================

set -e  # Arrêter en cas d'erreur

echo "🚀 CONFIGURATION SERVEUR GPU POUR SURVEILLANCE INTELLIGENTE"
echo "============================================================"

# Couleurs pour l'affichage
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Fonction d'affichage
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# 1. Vérification GPU
echo -e "\n${YELLOW}🔍 VÉRIFICATION GPU${NC}"
if nvidia-smi &> /dev/null; then
    print_status "GPU NVIDIA détecté"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits
else
    print_warning "GPU NVIDIA non détecté - Installation CPU"
fi

# 2. Mise à jour du système
echo -e "\n${YELLOW}📦 MISE À JOUR SYSTÈME${NC}"
sudo apt-get update -y
sudo apt-get install -y python3-pip python3-venv python3-dev build-essential

# 3. Configuration environnement Python
echo -e "\n${YELLOW}🐍 CONFIGURATION PYTHON${NC}"
if [ ! -d "venv" ]; then
    python3 -m venv venv
    print_status "Environnement virtuel créé"
fi

source venv/bin/activate
print_status "Environnement virtuel activé"

# 4. Installation dépendances de base
echo -e "\n${YELLOW}📚 INSTALLATION DÉPENDANCES DE BASE${NC}"
pip install --upgrade pip setuptools wheel

# 5. Installation PyTorch avec support GPU
echo -e "\n${YELLOW}🔥 INSTALLATION PYTORCH GPU${NC}"
if nvidia-smi &> /dev/null; then
    # Version GPU
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    print_status "PyTorch GPU installé"
else
    # Version CPU
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    print_status "PyTorch CPU installé"
fi

# 6. Installation dépendances principales
echo -e "\n${YELLOW}🛠️ INSTALLATION DÉPENDANCES PRINCIPALES${NC}"
pip install \
    ultralytics \
    opencv-python \
    transformers \
    accelerate \
    pydantic \
    loguru \
    rich \
    numpy \
    pillow \
    scikit-learn \
    matplotlib \
    seaborn \
    pandas \
    fastapi \
    uvicorn \
    websockets \
    aiofiles \
    python-multipart \
    jinja2 \
    python-dotenv \
    psutil

print_status "Toutes les dépendances installées"

# 7. Installation dépendances pour modèles VLM
echo -e "\n${YELLOW}🧠 INSTALLATION DÉPENDANCES VLM${NC}"
pip install \
    huggingface-hub \
    tokenizers \
    safetensors \
    datasets \
    evaluate

print_status "Dépendances VLM installées"

# 8. Test de l'installation
echo -e "\n${YELLOW}🧪 TEST DE L'INSTALLATION${NC}"

# Test PyTorch GPU
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA disponible: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU détecté: {torch.cuda.get_device_name(0)}')
    print(f'Mémoire GPU: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
else:
    print('Mode CPU activé')
"

# Test Ultralytics
python -c "
from ultralytics import YOLO
print('✅ Ultralytics importé avec succès')
"

# Test du système
python -c "
import sys
sys.path.insert(0, '.')
from src.core.types import Detection, BoundingBox, AnalysisRequest
print('✅ Types système importés avec succès')
"

print_status "Installation validée"

# 9. Téléchargement des modèles
echo -e "\n${YELLOW}📥 TÉLÉCHARGEMENT MODÈLES${NC}"

# YOLO11
python -c "
from ultralytics import YOLO
model = YOLO('yolov11n.pt')
print('✅ YOLO11n téléchargé')
"

print_status "Modèles téléchargés"

# 10. Configuration des permissions
echo -e "\n${YELLOW}🔐 CONFIGURATION PERMISSIONS${NC}"
chmod +x *.py
chmod +x tests/*.py
chmod +x examples/*.py
print_status "Permissions configurées"

# 11. Tests finaux
echo -e "\n${YELLOW}🎯 TESTS FINAUX${NC}"

# Test des corrections de base
if python test_basic_corrections.py > /dev/null 2>&1; then
    print_status "Tests de base: RÉUSSIS"
else
    print_warning "Tests de base: À vérifier"
fi

# 12. Informations finales
echo -e "\n${GREEN}🎉 INSTALLATION TERMINÉE !${NC}"
echo "============================================================"
echo "📋 COMMANDES POUR TESTER LE SYSTÈME:"
echo ""
echo "# 1. Activer l'environnement"
echo "source venv/bin/activate"
echo ""
echo "# 2. Test basique"
echo "python test_basic_corrections.py"
echo ""
echo "# 3. Test système complet" 
echo "python test_system_fixed.py"
echo ""
echo "# 4. Test vidéo (webcam)"
echo "python test_full_system_video.py --video webcam --max-frames 50"
echo ""
echo "# 5. Test optimisation"
echo "python examples/tool_optimization_demo.py --mode balanced"
echo ""
echo "# 6. Lancement production"
echo "python main.py --video webcam"
echo ""
echo "🚀 VOTRE SERVEUR GPU EST PRÊT POUR LA SURVEILLANCE INTELLIGENTE !"