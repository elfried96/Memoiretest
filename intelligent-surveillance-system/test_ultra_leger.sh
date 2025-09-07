#!/bin/bash

# Script test ultra-léger pour valider que ça marche

echo "🚀 TEST ULTRA-LÉGER KIMI-VL 2506"
echo "================================"

# Nettoyer mémoire d'abord  
python fix_memory_gpu.py

echo ""
echo "🧪 Test 1: Kimi-VL 2506 minimal (1 frame)"
python main_headless.py \
  --model kimi-vl-a3b-thinking-2506 \
  --video videos/surveillance01.mp4 \
  --max-frames 1 \
  --frame-skip 10 \
  --vlm-mode smart \
  --summary-interval 120

echo ""
echo "🧪 Test 2: Si échec, Qwen2-VL (plus stable)"
python main_Qwen.py \
  --video videos/surveillance01.mp4 \
  --max-frames 3 \
  --frame-skip 5 \
  --vlm-mode smart

echo ""
echo "✅ Tests terminés - Vérifiez les résultats !"