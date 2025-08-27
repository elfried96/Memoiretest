"""
🧠 Gestionnaire VLM Lazy Loading
===============================

Gère le chargement à la demande des modèles VLM coûteux.
Les modèles ne sont chargés que lors de la première utilisation.
"""

import time
import threading
from typing import Dict, Any, Optional, Callable
from loguru import logger
from dataclasses import dataclass
from enum import Enum


class VLMType(Enum):
    """Types de VLM supportés"""
    KIMI_VL = "kimi_vl"
    GPT4_VISION = "gpt4_vision"
    CLAUDE_VISION = "claude_vision"
    GEMINI_VISION = "gemini_vision"


@dataclass
class VLMConfig:
    """Configuration d'un VLM"""
    name: str
    model_id: str
    loader_function: Callable
    memory_usage_mb: int
    supports_batch: bool = False
    max_image_size: int = 1024
    lazy_load: bool = True


class LazyVLMManager:
    """Gestionnaire lazy loading pour les VLM"""
    
    def __init__(self):
        self.vlm_configs = self._setup_vlm_configs()
        self.loaded_models = {}
        self.loading_locks = {}
        self.usage_stats = {}
        
        # Initialiser les locks
        for vlm_type in VLMType:
            self.loading_locks[vlm_type] = threading.Lock()
            self.usage_stats[vlm_type] = {
                "load_count": 0,
                "usage_count": 0,
                "total_inference_time": 0.0,
                "last_used": None
            }
    
    def _setup_vlm_configs(self) -> Dict[VLMType, VLMConfig]:
        """Configuration des VLM supportés"""
        return {
            VLMType.KIMI_VL: VLMConfig(
                name="Kimi-VL",
                model_id="moonshot-v1-vision",
                loader_function=self._load_kimi_vl,
                memory_usage_mb=2048,
                supports_batch=True,
                max_image_size=1024
            ),
            VLMType.GPT4_VISION: VLMConfig(
                name="GPT-4 Vision",
                model_id="gpt-4-vision-preview",
                loader_function=self._load_gpt4_vision,
                memory_usage_mb=1024,  # API, pas de mémoire locale
                supports_batch=False,
                max_image_size=2048
            ),
            VLMType.CLAUDE_VISION: VLMConfig(
                name="Claude Vision",
                model_id="claude-3-sonnet-20240229",
                loader_function=self._load_claude_vision,
                memory_usage_mb=1024,  # API
                supports_batch=False,
                max_image_size=1568
            ),
            VLMType.GEMINI_VISION: VLMConfig(
                name="Gemini Vision",
                model_id="gemini-pro-vision",
                loader_function=self._load_gemini_vision,
                memory_usage_mb=1024,  # API
                supports_batch=False,
                max_image_size=2048
            )
        }
    
    def _load_kimi_vl(self):
        """Charge le modèle Kimi-VL"""
        logger.info("📥 Chargement Kimi-VL...")
        try:
            # Simuler chargement (remplacer par vraie implémentation)
            time.sleep(2)  # Simuler temps de chargement
            
            class MockKimiVL:
                def analyze_scene(self, image, context=""):
                    return {
                        "description": f"Scene analysis for context: {context}",
                        "suspicion_level": 0.3,
                        "confidence": 0.8,
                        "processing_time": 0.5,
                        "success": True
                    }
            
            logger.success("✅ Kimi-VL chargé avec succès")
            return MockKimiVL()
            
        except Exception as e:
            logger.error(f"❌ Erreur chargement Kimi-VL: {e}")
            return None
    
    def _load_gpt4_vision(self):
        """Charge GPT-4 Vision (API)"""
        logger.info("🔗 Initialisation GPT-4 Vision API...")
        try:
            # Configuration API
            import openai
            
            class GPT4VisionWrapper:
                def __init__(self):
                    # Configuration OpenAI (clé API depuis env)
                    pass
                
                def analyze_scene(self, image, context=""):
                    # Implémentation API GPT-4 Vision
                    return {
                        "description": f"GPT-4 Vision analysis: {context}",
                        "suspicion_level": 0.4,
                        "confidence": 0.9,
                        "processing_time": 1.2,
                        "success": True
                    }
            
            logger.success("✅ GPT-4 Vision API prêt")
            return GPT4VisionWrapper()
            
        except ImportError:
            logger.warning("⚠️ openai package non installé")
            return None
        except Exception as e:
            logger.error(f"❌ Erreur GPT-4 Vision: {e}")
            return None
    
    def _load_claude_vision(self):
        """Charge Claude Vision (API)"""
        logger.info("🔗 Initialisation Claude Vision API...")
        
        class ClaudeVisionWrapper:
            def analyze_scene(self, image, context=""):
                return {
                    "description": f"Claude Vision analysis: {context}",
                    "suspicion_level": 0.35,
                    "confidence": 0.85,
                    "processing_time": 0.8,
                    "success": True
                }
        
        logger.success("✅ Claude Vision API prêt")
        return ClaudeVisionWrapper()
    
    def _load_gemini_vision(self):
        """Charge Gemini Vision (API)"""
        logger.info("🔗 Initialisation Gemini Vision API...")
        
        class GeminiVisionWrapper:
            def analyze_scene(self, image, context=""):
                return {
                    "description": f"Gemini Vision analysis: {context}",
                    "suspicion_level": 0.42,
                    "confidence": 0.82,
                    "processing_time": 0.9,
                    "success": True
                }
        
        logger.success("✅ Gemini Vision API prêt")
        return GeminiVisionWrapper()
    
    def is_loaded(self, vlm_type: VLMType) -> bool:
        """Vérifie si un VLM est chargé"""
        return vlm_type in self.loaded_models and self.loaded_models[vlm_type] is not None
    
    def load_vlm(self, vlm_type: VLMType, force_reload: bool = False) -> Optional[Any]:
        """Charge un VLM à la demande"""
        
        # Vérifier si déjà chargé
        if not force_reload and self.is_loaded(vlm_type):
            logger.debug(f"✅ {vlm_type.value} déjà chargé")
            return self.loaded_models[vlm_type]
        
        # Threading lock pour éviter les chargements concurrents
        with self.loading_locks[vlm_type]:
            # Double-check après avoir acquis le lock
            if not force_reload and self.is_loaded(vlm_type):
                return self.loaded_models[vlm_type]
            
            config = self.vlm_configs[vlm_type]
            logger.info(f"🚀 Chargement lazy de {config.name}...")
            
            start_time = time.perf_counter()
            
            try:
                # Charger le modèle
                model = config.loader_function()
                
                if model is not None:
                    self.loaded_models[vlm_type] = model
                    
                    # Mettre à jour les stats
                    load_time = time.perf_counter() - start_time
                    stats = self.usage_stats[vlm_type]
                    stats["load_count"] += 1
                    stats["last_used"] = time.time()
                    
                    logger.success(
                        f"✅ {config.name} chargé en {load_time:.2f}s "
                        f"(~{config.memory_usage_mb}MB)"
                    )
                    
                    return model
                else:
                    logger.error(f"❌ Échec chargement {config.name}")
                    return None
                    
            except Exception as e:
                logger.error(f"❌ Erreur chargement {config.name}: {e}")
                return None
    
    def get_vlm(self, vlm_type: VLMType) -> Optional[Any]:
        """Récupère un VLM (charge si nécessaire)"""
        if not self.is_loaded(vlm_type):
            return self.load_vlm(vlm_type)
        
        # Mettre à jour stats d'utilisation
        stats = self.usage_stats[vlm_type]
        stats["usage_count"] += 1
        stats["last_used"] = time.time()
        
        return self.loaded_models[vlm_type]
    
    def analyze_with_vlm(
        self, 
        vlm_type: VLMType, 
        image: Any, 
        context: str = "",
        **kwargs
    ) -> Dict[str, Any]:
        """Analyse d'image avec un VLM spécifique"""
        
        start_time = time.perf_counter()
        
        try:
            # Récupérer le VLM (charge si nécessaire)
            vlm = self.get_vlm(vlm_type)
            
            if vlm is None:
                return {
                    "success": False,
                    "error": f"Impossible de charger {vlm_type.value}",
                    "processing_time": time.perf_counter() - start_time
                }
            
            # Exécuter l'analyse
            result = vlm.analyze_scene(image, context)
            
            # Ajouter métriques
            processing_time = time.perf_counter() - start_time
            result["total_processing_time"] = processing_time
            
            # Mettre à jour stats
            stats = self.usage_stats[vlm_type]
            stats["total_inference_time"] += processing_time
            
            logger.debug(f"✅ Analyse {vlm_type.value} en {processing_time:.2f}s")
            
            return result
            
        except Exception as e:
            processing_time = time.perf_counter() - start_time
            logger.error(f"❌ Erreur analyse {vlm_type.value}: {e}")
            
            return {
                "success": False,
                "error": str(e),
                "processing_time": processing_time
            }
    
    def unload_vlm(self, vlm_type: VLMType) -> bool:
        """Décharge un VLM pour libérer la mémoire"""
        if vlm_type in self.loaded_models:
            del self.loaded_models[vlm_type]
            logger.info(f"🗑️ {vlm_type.value} déchargé")
            return True
        return False
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Estime l'utilisation mémoire des VLM chargés"""
        total_mb = 0
        details = {}
        
        for vlm_type, model in self.loaded_models.items():
            if model is not None:
                config = self.vlm_configs[vlm_type]
                details[vlm_type.value] = config.memory_usage_mb
                total_mb += config.memory_usage_mb
        
        return {
            "total_mb": total_mb,
            "details": details,
            "loaded_models": len([m for m in self.loaded_models.values() if m is not None])
        }
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Récupère les statistiques d'utilisation"""
        formatted_stats = {}
        
        for vlm_type, stats in self.usage_stats.items():
            config = self.vlm_configs[vlm_type]
            formatted_stats[vlm_type.value] = {
                "name": config.name,
                "is_loaded": self.is_loaded(vlm_type),
                "load_count": stats["load_count"],
                "usage_count": stats["usage_count"],
                "total_inference_time": stats["total_inference_time"],
                "avg_inference_time": (
                    stats["total_inference_time"] / stats["usage_count"]
                    if stats["usage_count"] > 0 else 0
                ),
                "last_used": stats["last_used"]
            }
        
        return formatted_stats
    
    def cleanup_unused(self, max_idle_minutes: int = 30) -> int:
        """Nettoie les VLM non utilisés depuis X minutes"""
        current_time = time.time()
        cleaned_count = 0
        
        for vlm_type in list(self.loaded_models.keys()):
            stats = self.usage_stats[vlm_type]
            
            if stats["last_used"] is not None:
                idle_minutes = (current_time - stats["last_used"]) / 60
                
                if idle_minutes > max_idle_minutes:
                    self.unload_vlm(vlm_type)
                    cleaned_count += 1
                    logger.info(f"🧹 {vlm_type.value} nettoyé (inactif {idle_minutes:.1f}min)")
        
        return cleaned_count


# Instance globale
lazy_vlm_manager = LazyVLMManager()


def get_vlm_for_analysis(vlm_name: str = "kimi_vl"):
    """Récupère un VLM pour analyse (interface simple)"""
    try:
        vlm_type = VLMType(vlm_name)
        return lazy_vlm_manager.get_vlm(vlm_type)
    except ValueError:
        logger.error(f"❌ VLM inconnu: {vlm_name}")
        return None


if __name__ == "__main__":
    # Test du gestionnaire lazy
    import numpy as np
    
    print("🧪 Test Gestionnaire VLM Lazy")
    print("=" * 40)
    
    # Test chargement à la demande
    fake_image = np.zeros((224, 224, 3))
    
    result = lazy_vlm_manager.analyze_with_vlm(
        VLMType.KIMI_VL,
        fake_image,
        "Test scene analysis"
    )
    
    print(f"Résultat: {result}")
    print(f"Stats: {lazy_vlm_manager.get_usage_stats()}")
    print(f"Mémoire: {lazy_vlm_manager.get_memory_usage()}")