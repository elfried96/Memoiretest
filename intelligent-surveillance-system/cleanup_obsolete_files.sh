#!/bin/bash
"""
Script de nettoyage des fichiers obsolètes après refactoring
Usage: ./cleanup_obsolete_files.sh
"""

echo "🧹 NETTOYAGE DES FICHIERS OBSOLÈTES"
echo "=================================="

# Scripts de contournement obsolètes
echo "📝 Suppression des scripts de contournement..."
rm -f fix_memory_gpu.py
rm -f run_without_questions.py
rm -f auto_kimi.expect
rm -f test_ultra_leger.sh
rm -f config_fast.py
rm -f run_unit_tests_cpu_only.py
rm -f memory_dump_test.json

# Fichiers de cache et temporaires
echo "🗑️ Nettoyage cache et temporaires..."
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null
find . -name "*.pyc" -delete 2>/dev/null
find . -name ".pytest_cache" -type d -exec rm -rf {} + 2>/dev/null

# Anciens outputs spécifiques (optionnel)
echo "📊 Suppression anciens outputs spécifiques..."
rm -rf surveillance_output_qwen_only/ 2>/dev/null
rm -f simple_surveillance_*.json 2>/dev/null

# Fichiers de backup vim/emacs
echo "📝 Suppression fichiers d'éditeur..."
find . -name "*~" -delete 2>/dev/null
find . -name "*.swp" -delete 2>/dev/null
find . -name ".#*" -delete 2>/dev/null

echo ""
echo "✅ Nettoyage terminé!"
echo ""
echo "📋 FICHIERS GARDÉS (à vérifier manuellement si nécessaires):"
echo "  - main_headless.py (backup de l'ancien, supprimer quand refactorisé validé)"
echo "  - main_Qwen.py (version spécifique, évaluer si intégration complète)"  
echo "  - main.py (version GUI, garder si utilisée)"
echo ""
echo "🎯 PROCHAINES ÉTAPES:"
echo "  1. Tester main_headless_refactored.py"
echo "  2. Valider que tout fonctionne avec nouvelle architecture"
echo "  3. Supprimer manuellement les anciens main_*.py si OK"
echo "  4. Mettre à jour README.md avec nouvelles commandes"