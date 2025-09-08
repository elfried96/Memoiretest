"""Registre des modèles VLM supportés avec configuration dynamique."""

from enum import Enum
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


class VLMModelType(Enum):
    """Types de modèles VLM supportés."""
    KIMI_VL = "kimi_vl"
 
    QWEN = "qwen"


@dataclass
class ModelConfig:
    """Configuration pour un modèle VLM."""
    model_name: str
    model_type: VLMModelType
    processor_class: str
    model_class: str
    default_params: Dict[str, Any]
    supports_tool_calling: bool = True
    supports_batch: bool = True
    memory_efficient: bool = True
    description: str = ""


class VLMModelRegistry:
    """Registre centralisé des modèles VLM disponibles."""
    
    def __init__(self):
        self.models = self._initialize_supported_models()
        self.current_model = None
    
    def _initialize_supported_models(self) -> Dict[str, ModelConfig]:
        """Initialisation des modèles supportés avec leurs configurations."""
        
        models = {}
        
        # === Kimi-VL Models (Moonshot AI) ===
        
        # Kimi-VL-A3B-Thinking-2506 (NOUVEAU - Version améliorée 2024)
        models["kimi-vl-a3b-thinking-2506"] = ModelConfig(
            model_name="moonshotai/Kimi-VL-A3B-Thinking-2506",
            model_type=VLMModelType.KIMI_VL,
            processor_class="AutoProcessor",
            model_class="AutoModelForVision2Seq", 
            default_params={
                "max_new_tokens": 50,  # ✅ SURVEILLANCE TEMPS RÉEL
                "temperature": 0.8,    # ✅ Doc officielle Kimi Thinking  
                "do_sample": True,
                "use_cache": True      # ✅ ACTIVER cache pour performance 2025
            },
            description="NOUVEAU - Kimi-VL 2506 avec meilleurs vision/reasoning/agent (fix erreurs cache)"
        )
        
        # Kimi-VL-A3B-Thinking (LEGACY - Problèmes DynamicCache)
        models["kimi-vl-a3b-thinking"] = ModelConfig(
            model_name="moonshotai/Kimi-VL-A3B-Thinking", 
            model_type=VLMModelType.KIMI_VL,
            processor_class="AutoProcessor",
            model_class="AutoModelForCausalLM",
            default_params={
                "torch_dtype": "auto",
                "device_map": "auto", 
                "trust_remote_code": True,
                "max_new_tokens": 768,
                "temperature": 0.8,  # Optimal pour surveillance ET raisonnement
                "do_sample": True
            },
            supports_tool_calling=True,
            supports_batch=True,
            memory_efficient=True,
            description="PRINCIPAL - Kimi-VL Thinking pour surveillance et analyse (61.7 MMMU, 2.8B activés)"
        )
        
        
        # === Qwen2-VL Models (Alibaba) ===
        
        # Qwen2-VL-7B-Instruct (FALLBACK UNIQUEMENT)
        models["qwen2-vl-7b-instruct"] = ModelConfig(
            model_name="Qwen/Qwen2-VL-7B-Instruct",
            model_type=VLMModelType.QWEN,
            processor_class="Qwen2VLProcessor",
            model_class="Qwen2VLForConditionalGeneration",
            default_params={
                "torch_dtype": "bfloat16",
                "device_map": "auto",
                "load_in_4bit": True, 
                "trust_remote_code": True,
                "max_new_tokens": 512,
                "temperature": 0.1,
                "do_sample": True
            },
            supports_tool_calling=True,
            supports_batch=True,
            memory_efficient=True,
            description="FALLBACK - Qwen2-VL 7B robuste si Kimi indisponible"
        )
        
        # Qwen2.5-VL-32B-Instruct (NOUVELLE VERSION PUISSANTE)
        models["qwen2.5-vl-32b-instruct"] = ModelConfig(
            model_name="Qwen/Qwen2.5-VL-32B-Instruct",
            model_type=VLMModelType.QWEN,
            processor_class="AutoProcessor",
            model_class="Qwen2_5_VLForConditionalGeneration",
            default_params={
                "torch_dtype": "auto",
                "device_map": "auto",
                "trust_remote_code": True,
                "max_new_tokens": 1024,
                # "temperature": 0.6,  # ✅ Supprimé car incompatible avec do_sample=False
                "top_p": 0.95,
                "do_sample": True
            },
            supports_tool_calling=True,
            supports_batch=True,
            memory_efficient=True,
            description="Qwen2.5-VL 32B - Excellent pour surveillance et analyse comportementale"
        )

        
        return models
    
    def get_model_config(self, model_id: str) -> Optional[ModelConfig]:
        """Récupère la configuration d'un modèle."""
        return self.models.get(model_id)
    
    def list_available_models(self) -> Dict[str, ModelConfig]:
        """Liste tous les modèles disponibles."""
        return self.models.copy()
    
    def get_models_by_type(self, model_type: VLMModelType) -> Dict[str, ModelConfig]:
        """Récupère les modèles par type."""
        return {
            model_id: config for model_id, config in self.models.items()
            if config.model_type == model_type
        }
    
    def get_recommended_model(self, use_case: str = "surveillance") -> str:
        """Récupère le modèle recommandé - SIMPLIFIÉ: Kimi-VL-Thinking pour TOUT."""
        
        # Architecture simplifiée: Kimi-VL-Thinking pour tous les cas d'usage
        # Fallback automatique vers Qwen2-VL si nécessaire
        return "kimi-vl-a3b-thinking"
    
    def validate_model_availability(self, model_id: str) -> Tuple[bool, str]:
        """Valide la disponibilité d'un modèle."""
        
        if model_id not in self.models:
            return False, f"Modèle {model_id} non supporté"
        
        config = self.models[model_id]
        
        # Vérification des dépendances selon le type
        try:
            if config.model_type == VLMModelType.KIMI_VL:
                # Vérification Kimi-VL (Moonshot AI)
                try:
                    import transformers
                    from transformers import AutoProcessor, AutoModelForCausalLM
                    return True, "Kimi-VL disponible (via transformers>=4.48.2)"
                except ImportError:
                    return False, "Transformers>=4.48.2 requis pour Kimi-VL"
            
            
            elif config.model_type == VLMModelType.QWEN:
                import transformers
                # NOTE: Qwen2-VL pourrait nécessiter une version récente
                return True, "Qwen2-VL disponible" 
                
        except ImportError as e:
            return False, f"Dépendance manquante: {e}"
        
        return True, "Modèle disponible"
    
    def get_model_recommendations(self) -> Dict[str, str]:
        """Retourne les recommandations - SIMPLIFIÉ: Architecture dual-VLM."""
        
        return {
            "surveillance": "kimi-vl-a3b-thinking",    # Principal pour TOUT
            "thinking": "kimi-vl-a3b-thinking",       # Principal pour TOUT
            "high_performance": "kimi-vl-a3b-thinking", # Principal pour TOUT
            "reasoning": "kimi-vl-a3b-thinking",      # Principal pour TOUT
            "fallback": "qwen2-vl-7b-instruct",      # Fallback uniquement
            "default": "kimi-vl-a3b-thinking"         # Principal par défaut
        }
    
    def get_model_comparison(self) -> Dict[str, Any]:
        """Comparaison des modèles disponibles."""
        
        comparison = {
            "models": {},
            "summary": {
                "total_models": len(self.models),
                "by_type": {},
                "memory_efficient": [],
                "high_performance": [],
                "tool_calling_capable": []
            }
        }
        
        # Analyse par modèle
        for model_id, config in self.models.items():
            comparison["models"][model_id] = {
                "type": config.model_type.value,
                "name": config.model_name,
                "memory_efficient": config.memory_efficient,
                "supports_tool_calling": config.supports_tool_calling,
                "supports_batch": config.supports_batch,
                "description": config.description
            }
            
            # Catégorisation
            if config.memory_efficient:
                comparison["summary"]["memory_efficient"].append(model_id)
            
            if not config.memory_efficient:  # Modèles haute perf = moins memory efficient
                comparison["summary"]["high_performance"].append(model_id)
            
            if config.supports_tool_calling:
                comparison["summary"]["tool_calling_capable"].append(model_id)
        
        # Comptage par type
        for model_type in VLMModelType:
            count = len(self.get_models_by_type(model_type))
            comparison["summary"]["by_type"][model_type.value] = count
        
        return comparison