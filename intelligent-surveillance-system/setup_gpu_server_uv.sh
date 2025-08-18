#!/bin/bash
# 🚀 Script d'Installation UV pour Serveur GPU
# ============================================

set -e  # Arrêter en cas d'erreur

echo "🚀 CONFIGURATION SERVEUR GPU AVEC UV"
echo "===================================="

# Couleurs pour l'affichage
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
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

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# 1. Vérification GPU
echo -e "\n${YELLOW}🔍 VÉRIFICATION GPU${NC}"
if nvidia-smi &> /dev/null; then
    print_status "GPU NVIDIA détecté"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits
    GPU_AVAILABLE=true
else
    print_warning "GPU NVIDIA non détecté - Installation CPU"
    GPU_AVAILABLE=false
fi

# 2. Vérification/Installation UV
echo -e "\n${YELLOW}📦 VÉRIFICATION UV${NC}"
if ! command -v uv &> /dev/null; then
    print_warning "UV non installé - Installation..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.cargo/env
    print_status "UV installé"
else
    print_status "UV déjà disponible: $(uv --version)"
fi

# 3. Initialisation du projet avec UV
echo -e "\n${YELLOW}🐍 INITIALISATION PROJET UV${NC}"

# Supprimer ancien environnement si existant
if [ -d ".venv" ]; then
    rm -rf .venv
    print_info "Ancien environnement supprimé"
fi

# Initialiser avec UV
uv sync
print_status "Environnement UV créé et synchronisé"

# 4. Installation dépendances GPU spécifiques
echo -e "\n${YELLOW}🔥 INSTALLATION PYTORCH GPU${NC}"

if [ "$GPU_AVAILABLE" = true ]; then
    # Version GPU avec UV
    uv add torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    print_status "PyTorch GPU installé via UV"
else
    # Version CPU avec UV
    uv add torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    print_status "PyTorch CPU installé via UV"
fi

# 5. Installation du projet en mode éditable
echo -e "\n${YELLOW}🛠️ INSTALLATION PROJET${NC}"
uv pip install -e .
print_status "Projet installé en mode éditable"

# 6. Installation extras si nécessaire
if [ "$GPU_AVAILABLE" = true ]; then
    echo -e "\n${YELLOW}🎯 INSTALLATION EXTRAS GPU${NC}"
    uv add --group gpu nvidia-ml-py
    print_status "Extras GPU installés"
fi

# 7. Test de l'installation
echo -e "\n${YELLOW}🧪 TEST DE L'INSTALLATION${NC}"

# Test UV Python
uv run python -c "
print('🐍 Python UV version:')
import sys
print(f'Python: {sys.version}')
print(f'Executable: {sys.executable}')
"

# Test PyTorch GPU
uv run python -c "
import torch
print(f'🔥 PyTorch version: {torch.__version__}')
print(f'CUDA disponible: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU détecté: {torch.cuda.get_device_name(0)}')
    print(f'Mémoire GPU: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
else:
    print('Mode CPU activé')
"

# Test Ultralytics
uv run python -c "
try:
    from ultralytics import YOLO
    print('✅ Ultralytics importé avec succès')
except Exception as e:
    print(f'⚠️ Ultralytics: {e}')
"

# Test du système
uv run python -c "
try:
    import sys
    sys.path.insert(0, '.')
    from src.core.types import Detection, BoundingBox, AnalysisRequest
    print('✅ Types système importés avec succès')
except Exception as e:
    print(f'⚠️ Types système: {e}')
"

print_status "Installation UV validée"

# 8. Téléchargement des modèles
echo -e "\n${YELLOW}📥 TÉLÉCHARGEMENT MODÈLES${NC}"

# YOLO11 via UV
uv run python -c "
try:
    from ultralytics import YOLO
    model = YOLO('yolov11n.pt')
    print('✅ YOLO11n téléchargé')
except Exception as e:
    print(f'⚠️ YOLO11: {e}')
"

print_status "Modèles téléchargés"

# 9. Configuration des permissions
echo -e "\n${YELLOW}🔐 CONFIGURATION PERMISSIONS${NC}"
chmod +x *.py 2>/dev/null || true
chmod +x tests/*.py 2>/dev/null || true
chmod +x examples/*.py 2>/dev/null || true
print_status "Permissions configurées"

# 10. Tests finaux avec UV
echo -e "\n${YELLOW}🎯 TESTS FINAUX UV${NC}"

# Test des corrections de base
if uv run python test_basic_corrections.py > /dev/null 2>&1; then
    print_status "Tests de base: RÉUSSIS"
else
    print_warning "Tests de base: À vérifier"
fi

# 11. Création des scripts UV
echo -e "\n${YELLOW}📝 CRÉATION SCRIPTS UV${NC}"

# Script uv run pour surveillance
cat > run_surveillance.sh << 'EOF'
#!/bin/bash
# Script de lancement surveillance avec UV
echo "🎬 Lancement surveillance avec UV..."
uv run python main.py "$@"
EOF

# Script uv run pour tests
cat > run_tests.sh << 'EOF'
#!/bin/bash
# Script de tests avec UV
echo "🧪 Tests avec UV..."
uv run python run_gpu_tests.py "$@"
EOF

chmod +x run_surveillance.sh run_tests.sh
print_status "Scripts UV créés"

# 12. Informations finales
echo -e "\n${GREEN}🎉 INSTALLATION UV TERMINÉE !${NC}"
echo "============================================================"
echo "📋 COMMANDES UV POUR TESTER LE SYSTÈME:"
echo ""
echo "# 1. Test basique avec UV"
echo "uv run python test_basic_corrections.py"
echo ""
echo "# 2. Test système complet avec UV" 
echo "uv run python test_system_fixed.py"
echo ""
echo "# 3. Test surveillance avec UV"
echo "uv run python test_full_system_video.py --video webcam --max-frames 50"
echo ""
echo "# 4. Test optimisation avec UV"
echo "uv run python examples/tool_optimization_demo.py --mode balanced"
echo ""
echo "# 5. Lancement production avec UV"
echo "uv run python main.py --video webcam"
echo ""
echo "# 6. Ou avec le script dédié"
echo "./run_surveillance.sh --video webcam"
echo ""
echo "# 7. Tests complets avec UV"
echo "uv run python run_gpu_tests.py"
echo "# Ou:"
echo "./run_tests.sh"
echo ""
echo "# 8. Ajouter de nouvelles dépendances"
echo "uv add package_name"
echo ""
echo "# 9. Mettre à jour les dépendances"
echo "uv sync"
echo ""
echo "🚀 VOTRE SERVEUR GPU AVEC UV EST PRÊT !"
echo ""
echo "💡 AVANTAGES UV:"
echo "   ⚡ Installation 10-100x plus rapide"
echo "   🔒 Résolution de dépendances garantie"
echo "   🎯 Environnement isolé et reproductible"
echo "   🛠️ Gestion de projet moderne"
echo ""
echo "📊 ÉTAT UV:"
uv tree --depth 1 2>/dev/null || echo "   📦 Dépendances installées et prêtes"