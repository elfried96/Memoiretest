#!/bin/bash

echo "🔒 Dashboard de Surveillance Intelligente"
echo "========================================"

# Navigation vers le répertoire du projet
cd /home/elfried-kinzoun/PycharmProjects/intelligent-surveillance-system

# Activation de l'environnement virtuel
echo "🔄 Activation de l'environnement virtuel..."
source .venv/bin/activate

# Navigation vers le dossier dashboard
cd dashboard

echo "🚀 Lancement du dashboard..."
echo "🌐 Le dashboard sera accessible sur :"
echo "   http://localhost:8501"
echo "   http://0.0.0.0:8501"
echo ""
echo "💡 Pour arrêter : Ctrl+C"
echo "========================================"

# Lancement du dashboard
streamlit run enhanced_dashboard.py --server.port=8501 --server.address=0.0.0.0 --browser.gatherUsageStats=false