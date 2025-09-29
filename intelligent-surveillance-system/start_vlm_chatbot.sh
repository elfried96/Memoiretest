#!/bin/bash
# 🚀 Script de Démarrage Rapide - VLM Chatbot Symbiosis
# =====================================================

echo "🧠 VLM Chatbot Symbiosis - Démarrage Intelligent"
echo "================================================"

# Configuration environnement
export PYTHONPATH="${PWD}/src:${PWD}/dashboard:${PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export AUTO_INIT_VLM=true
export FORCE_REAL_PIPELINE=true

# Vérification GPU
echo "🔍 Vérification GPU..."
if command -v nvidia-smi &> /dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | head -1)
    echo "✅ GPU détecté: $GPU_INFO"
    export CUDA_VISIBLE_DEVICES=0
    export VLM_MODE="gpu"
else
    echo "⚠️ Aucun GPU détecté - Mode CPU/Simulation"
    export VLM_MODE="cpu"
fi

# Vérification dépendances Python
echo "🔍 Vérification dépendances..."
python3 -c "
try:
    import streamlit, torch, transformers, PIL, matplotlib, loguru
    print('✅ Dépendances principales OK')
except ImportError as e:
    print(f'❌ Dépendance manquante: {e}')
    exit(1)
"

if [ $? -ne 0 ]; then
    echo "💡 Installation dépendances manquantes..."
    pip install streamlit torch torchvision transformers pillow matplotlib loguru numpy pandas plotly
fi

# Test chatbot VLM
echo "🧪 Test rapide VLM Chatbot..."
timeout 30s python3 test_vlm_chatbot.py || echo "⚠️ Test timeout - Continuing..."

# Choix mode lancement
echo ""
echo "🎯 Choisissez le mode de lancement:"
echo "1) 🎮 Dashboard complet avec VLM Chatbot"
echo "2) 🧠 Test interactif chatbot VLM"
echo "3) ⚡ Qwen2.5-VL-32B optimisé (GPU requis)"
echo "4) 🔧 Mode développement/debug"

read -p "Votre choix (1-4): " choice

case $choice in
    1)
        echo "🎮 Lancement Dashboard VLM Chatbot..."
        cd dashboard/
        echo "🌐 Dashboard sera disponible sur: http://localhost:8501"
        echo "💬 Chat VLM disponible dans onglets 'Surveillance' et 'Upload Vidéo'"
        streamlit run production_dashboard.py
        ;;
    2)
        echo "🧠 Mode test interactif chatbot..."
        python3 -c "
import asyncio
import sys
sys.path.append('dashboard')
from vlm_chatbot_symbiosis import get_vlm_chatbot, process_vlm_chat_query

async def interactive_chat():
    print('🤖 VLM Chatbot Interactif - Tapez \"quit\" pour sortir')
    chatbot = get_vlm_chatbot()
    
    mock_context = {
        'stats': {'frames_processed': 42, 'current_performance_score': 0.89},
        'detections': [], 'optimizations': []
    }
    
    while True:
        question = input('\\n👤 Vous: ')
        if question.lower() in ['quit', 'exit', 'q']:
            break
            
        try:
            response = await process_vlm_chat_query(
                question=question,
                chat_type='surveillance', 
                vlm_context=mock_context
            )
            print(f'\\n🧠 VLM: {response.get(\"response\", \"Erreur réponse\")}')
            
            if response.get('thinking'):
                print(f'💭 Thinking: {response[\"thinking\"][:200]}...')
                
        except Exception as e:
            print(f'❌ Erreur: {e}')

asyncio.run(interactive_chat())
        "
        ;;
    3)
        echo "⚡ Lancement Qwen2.5-VL-32B optimisé..."
        if [ "$VLM_MODE" = "cpu" ]; then
            echo "❌ GPU requis pour Qwen2.5-VL-32B"
            exit 1
        fi
        
        echo "🔧 Configuration GPU optimale..."
        export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
        
        # Test vidéo avec Qwen2.5-VL-32B
        if [ -f "videos/test.mp4" ]; then
            python3 launch_qwen_32b.py --video videos/test.mp4 --max-frames 10
        else
            echo "⚠️ Aucune vidéo test trouvée dans videos/"
            echo "💡 Créez un dossier 'videos/' avec des fichiers MP4 pour tester"
        fi
        ;;
    4)
        echo "🔧 Mode développement/debug..."
        echo "📊 Variables environnement:"
        echo "   - VLM_MODE: $VLM_MODE"
        echo "   - CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-non_défini}"
        echo "   - PYTHONPATH: $PYTHONPATH"
        
        echo "🧪 Tests disponibles:"
        echo "   python3 test_vlm_chatbot.py        # Test complet chatbot"
        echo "   python3 launch_qwen_32b.py --help  # Aide Qwen2.5-VL-32B"
        
        echo "🎮 Dashboard:"
        echo "   cd dashboard/ && streamlit run production_dashboard.py"
        
        echo "🐍 Shell interactif Python:"
        python3 -c "
import sys
sys.path.append('dashboard')
sys.path.append('src')

print('🔬 Modules disponibles:')
try:
    from vlm_chatbot_symbiosis import get_vlm_chatbot
    print('  ✅ vlm_chatbot_symbiosis')
except:
    print('  ❌ vlm_chatbot_symbiosis')

try:
    from real_pipeline_integration import get_real_pipeline
    print('  ✅ real_pipeline_integration')
except:
    print('  ❌ real_pipeline_integration')

print('\\n💡 Commandes utiles:')
print('  chatbot = get_vlm_chatbot()')
print('  pipeline = get_real_pipeline()')
"
        bash
        ;;
    *)
        echo "❌ Choix invalide"
        exit 1
        ;;
esac