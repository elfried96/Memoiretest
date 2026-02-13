"""
VLM Orchestrateur avec Tool-Calling.
Analyse contextuelle des comportements suspects pour réduire les faux positifs.
"""

import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import torch
from loguru import logger
from PIL import Image

from src.core.config import settings


@dataclass
class VLMResponse:
    """Réponse du VLM."""
    success: bool
    final_suspicion_score: float
    is_threat_confirmed: bool
    reasoning: str
    observations: list[str]
    recommended_action: str  # "none", "monitor", "alert", "intervene"
    tools_used: list[str]
    inference_time_ms: float
    tokens_used: int
    raw_response: str


class VLMOrchestrator:
    """
    Orchestrateur VLM pour analyse contextuelle.
    
    Utilise Qwen2.5-VL-7B-Instruct pour analyser les frames suspects
    et fournir une décision contextuelle réduisant les faux positifs.
    
    Usage:
        vlm = VLMOrchestrator()
        await vlm.initialize()
        result = await vlm.analyze(image_paths, context)
    """
    
    SYSTEM_PROMPT = """Tu es un expert en surveillance de sécurité pour la prévention du vol en grande distribution.

MISSION:
- Analyser les images de surveillance pour détecter les comportements suspects
- Fournir une évaluation contextuelle pour réduire les faux positifs
- Distinguer comportements légitimes atypiques vs tentatives de vol

COMPORTEMENTS SUSPECTS:
- Dissimulation de produits (vêtements, sacs)
- Mouvements nerveux ou furtifs
- Évitement des zones de caisse
- Retrait d'étiquettes de sécurité
- Stationnement prolongé devant produits de valeur
- Gestes de surveillance (regarder autour avant d'agir)

COMPORTEMENTS LÉGITIMES À NE PAS CONFONDRE:
- Client consultant son téléphone
- Parent surveillant ses enfants
- Personne comparant des produits
- Employé en réassort
- Client indécis

RÈGLES IMPORTANTES:
- Privilégie la PRÉCISION (éviter les faux positifs)
- Score < 0.3 = comportement normal
- Score 0.3-0.6 = à surveiller
- Score 0.6-0.85 = attention requise  
- Score > 0.85 = intervention recommandée

RÉPONDS EN JSON:
{
  "suspicion_score": 0.0-1.0,
  "is_threat": true/false,
  "reasoning": "Explication détaillée",
  "observations": ["observation1", "observation2"],
  "recommended_action": "none|monitor|alert|intervene"
}"""

    def __init__(self):
        self.model = None
        self.processor = None
        self.is_initialized = False
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    async def initialize(self) -> None:
        """Charge le modèle VLM."""
        if self.is_initialized:
            return
        
        model_name = settings.vlm.model_name
        logger.info(f"Initializing VLM: {model_name}")
        
        try:
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, BitsAndBytesConfig
            
            # Configuration quantization
            quant_config = None
            if settings.vlm.quantization == "4bit":
                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
            elif settings.vlm.quantization == "8bit":
                quant_config = BitsAndBytesConfig(load_in_8bit=True)
            
            # Charger processor
            self.processor = AutoProcessor.from_pretrained(
                model_name,
                trust_remote_code=True
            )
            
            # Charger modèle
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name,
                quantization_config=quant_config,
                device_map=settings.vlm.device_map,
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
            
            self.is_initialized = True
            logger.info(f"VLM ready - device: {self.device}, quantization: {settings.vlm.quantization}")
            
        except Exception as e:
            logger.error(f"Failed to initialize VLM: {e}")
            raise
    
    async def analyze(
        self,
        image_paths: list[str],
        context: dict,
        initial_suspicion: float = 0.5
    ) -> VLMResponse:
        """
        Analyse les images avec le VLM.
        
        Args:
            image_paths: Chemins vers les images à analyser
            context: Contexte de l'alerte (trajectoire, durée, zones)
            initial_suspicion: Score de suspicion initial du pipeline CV
            
        Returns:
            VLMResponse avec décision et raisonnement
        """
        if not self.is_initialized:
            raise RuntimeError("VLM not initialized")
        
        start_time = time.time()
        
        try:
            # Charger les images
            images = []
            for path in image_paths[:4]:  # Max 4 images
                if Path(path).exists():
                    images.append(Image.open(path).convert("RGB"))
            
            if not images:
                return self._error_response("No valid images found", initial_suspicion)
            
            # Construire le prompt
            user_prompt = self._build_prompt(context, initial_suspicion)
            
            # Préparer les messages
            messages = [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        *[{"type": "image", "image": img} for img in images],
                        {"type": "text", "text": user_prompt}
                    ]
                }
            ]
            
            # Tokenizer
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            inputs = self.processor(
                text=[text],
                images=images,
                padding=True,
                return_tensors="pt"
            ).to(self.device)
            
            # Génération
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=settings.vlm.max_new_tokens,
                    temperature=settings.vlm.temperature,
                    do_sample=True
                )
            
            # Décoder
            response_text = self.processor.decode(
                outputs[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )
            
            # Parser la réponse
            parsed = self._parse_response(response_text, initial_suspicion)
            
            inference_time = (time.time() - start_time) * 1000
            
            return VLMResponse(
                success=True,
                final_suspicion_score=parsed["suspicion_score"],
                is_threat_confirmed=parsed["is_threat"],
                reasoning=parsed["reasoning"],
                observations=parsed.get("observations", []),
                recommended_action=parsed.get("recommended_action", "monitor"),
                tools_used=[],
                inference_time_ms=inference_time,
                tokens_used=outputs.shape[1],
                raw_response=response_text
            )
            
        except Exception as e:
            logger.error(f"VLM analysis failed: {e}")
            return self._error_response(str(e), initial_suspicion)
    
    def _build_prompt(self, context: dict, initial_suspicion: float) -> str:
        """Construit le prompt utilisateur."""
        return f"""Analyse cette séquence de surveillance.

CONTEXTE DÉTECTÉ PAR LE SYSTÈME:
- Score de suspicion initial: {initial_suspicion:.2f}
- Durée de présence: {context.get('duration_seconds', 0):.0f} secondes
- Zones visitées: {', '.join(context.get('zones_visited', ['inconnues']))}
- Changements de direction: {context.get('direction_changes', 0)}
- Raisons de suspicion: {', '.join(context.get('suspicion_reasons', ['aucune']))}

Analyse les images et fournis ton évaluation au format JSON demandé.
Sois précis et évite les faux positifs."""
    
    def _parse_response(self, response: str, default_score: float) -> dict:
        """Parse la réponse JSON du VLM."""
        try:
            # Trouver le JSON dans la réponse
            start = response.find("{")
            end = response.rfind("}") + 1
            
            if start >= 0 and end > start:
                json_str = response[start:end]
                data = json.loads(json_str)
                
                return {
                    "suspicion_score": float(data.get("suspicion_score", default_score)),
                    "is_threat": bool(data.get("is_threat", False)),
                    "reasoning": data.get("reasoning", ""),
                    "observations": data.get("observations", []),
                    "recommended_action": data.get("recommended_action", "monitor")
                }
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse VLM response: {e}")
        
        # Fallback
        return {
            "suspicion_score": default_score,
            "is_threat": default_score > 0.7,
            "reasoning": response[:500],
            "observations": [],
            "recommended_action": "monitor"
        }
    
    def _error_response(self, error: str, score: float) -> VLMResponse:
        """Retourne une réponse d'erreur."""
        return VLMResponse(
            success=False,
            final_suspicion_score=score,
            is_threat_confirmed=False,
            reasoning=f"Analysis failed: {error}",
            observations=[],
            recommended_action="monitor",
            tools_used=[],
            inference_time_ms=0,
            tokens_used=0,
            raw_response=""
        )
    
    async def shutdown(self) -> None:
        """Libère les ressources."""
        if self.model:
            del self.model
        if self.processor:
            del self.processor
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.is_initialized = False
        logger.info("VLM shut down")
    
    @property
    def stats(self) -> dict:
        """Statistiques du VLM."""
        gpu_memory = 0
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024**3  # GB
        
        return {
            "initialized": self.is_initialized,
            "model": settings.vlm.model_name,
            "quantization": settings.vlm.quantization,
            "gpu_memory_gb": round(gpu_memory, 2),
            "device": self.device
        }


# === SINGLETON ===

_vlm: Optional[VLMOrchestrator] = None


def get_vlm_orchestrator() -> VLMOrchestrator:
    """Retourne le singleton du VLM."""
    global _vlm
    if _vlm is None:
        _vlm = VLMOrchestrator()
    return _vlm