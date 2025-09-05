#!/usr/bin/env python3
"""
Configuration rapide pour surveillance temps réel
==============================================

Configuration optimisée pour des performances maximales
avec Kimi-VL en mode surveillance.
"""

# Paramètres VLM optimisés pour vitesse
FAST_VLM_CONFIG = {
    "max_new_tokens": 100,      # Réponses très courtes
    "temperature": 0.1,         # Très déterministe = rapide
    "do_sample": False,         # Greedy = plus rapide
    "use_cache": True,          # Cache actif
}

# Mode surveillance économique  
SURVEILLANCE_CONFIG = {
    "vlm_mode": "smart",                # Mode intelligent (pas continuous)
    "frame_skip": 3,                    # 1 frame sur 3
    "vlm_cooldown_seconds": 10,         # 10s entre analyses VLM
    "summary_interval_seconds": 60,     # Résumés toutes les minutes
    "max_concurrent_tools": 2,          # Moins d'outils simultanés
}

# Paramètres YOLO optimisés
YOLO_FAST_CONFIG = {
    "conf_threshold": 0.7,      # Plus strict = moins de détections
    "iou_threshold": 0.5,       # Moins de boxes
    "model_size": "n",          # nano = plus rapide
}

def get_fast_args():
    """Arguments recommandés pour mode rapide."""
    return {
        "vlm_mode": "smart",
        "frame_skip": 3,
        "max_frames": 50,
        "mode": "FAST",
        "summary_interval": 60
    }

if __name__ == "__main__":
    print("Configuration rapide pour surveillance:")
    print("python main_headless.py --video videos/surveillance01.mp4 \\")
    print("  --vlm-mode smart --frame-skip 3 --max-frames 50 \\")
    print("  --mode FAST --summary-interval 60")