#!/bin/bash
# Script de nettoyage des fichiers de test obsolètes

echo "🧹 Nettoyage des fichiers de test obsolètes..."

# Créer dossier de sauvegarde
mkdir -p old_tests

# Déplacer les anciens tests
echo "📦 Archivage des anciens tests..."
mv test_basic_corrections.py old_tests/ 2>/dev/null
mv test_corrections.py old_tests/ 2>/dev/null  
mv test_system_fixed.py old_tests/ 2>/dev/null
mv test_simple_vlm.py old_tests/ 2>/dev/null
mv test_tool_optimization.py old_tests/ 2>/dev/null
mv simple_test.py old_tests/ 2>/dev/null
mv check_gpu_system.py old_tests/ 2>/dev/null
mv check_gpu_system_uv.py old_tests/ 2>/dev/null
mv fix_all_imports.py old_tests/ 2>/dev/null
mv run_all_tests.py old_tests/ 2>/dev/null
mv run_gpu_tests.py old_tests/ 2>/dev/null
mv run_gpu_tests_uv.py old_tests/ 2>/dev/null
mv download_and_test_models.py old_tests/ 2>/dev/null

# Garder uniquement les tests essentiels
echo "✅ Tests conservés:"
echo "  - test_simple_kimi_only.py (nouveau test minimal)"
echo "  - test_kimi_vl_only.py (benchmark Kimi)"  
echo "  - test_qwen2_vl_only.py (benchmark Qwen)"
echo "  - compare_vlm_models.py (comparaison)"
echo "  - test_video_analysis.py (analyse vidéo)"
echo "  - test_full_system_video.py (système complet)"

echo "📁 Anciens tests archivés dans old_tests/"

# Nettoyer workspace si possible
echo "🗑️  Nettoyage workspace..."
rm -rf /workspace/transformers_cache/* 2>/dev/null
rm -rf /workspace/huggingface/* 2>/dev/null

echo "✅ Nettoyage terminé!"