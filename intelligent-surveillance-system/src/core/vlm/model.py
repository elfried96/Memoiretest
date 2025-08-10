"""Modèle Vision-Language avec capacités d'orchestration d'outils."""

import json
import base64
from io import BytesIO
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import asyncio

import torch
import torch.nn.functional as F
from transformers import (
    AutoProcessor, 
    AutoModelForVision2Seq,
    LlavaNextProcessor, 
    LlavaNextForConditionalGeneration,
    BlipProcessor,
    BlipForConditionalGeneration
)
from PIL import Image
import numpy as np
from loguru import logger

from ..types import (
    AnalysisRequest, 
    AnalysisResponse, 
    SuspicionLevel, 
    ActionType, 
    ToolType,
    ToolResult
)
from ...utils.exceptions import ModelError, ProcessingError


class VisionLanguageModel:
    """
    Modèle Vision-Language avec capacités d'orchestration d'outils.
    
    Supporte plusieurs architectures open source:
    - LLaVA-NeXT (recommandé pour tool-calling)
    - BLIP-2 (rapide, léger)
    - Kosmos-2 (multimodal avancé)
    """
    
    def __init__(
        self, 
        model_name: str = "llava-hf/llava-v1.6-mistral-7b-hf",
        device: str = "auto",
        load_in_4bit: bool = True,
        max_tokens: int = 512
    ):
        self.model_name = model_name
        self.device = self._setup_device(device)
        self.max_tokens = max_tokens
        self.load_in_4bit = load_in_4bit
        
        # État du modèle
        self.model = None
        self.processor = None
        self.is_loaded = False
        
        # Cache pour optimisation
        self._tool_descriptions_cache = {}
        self._prompt_templates_cache = {}
        
        logger.info(f"Initialisation VLM: {model_name} sur {self.device}")
    
    def _setup_device(self, device: str) -> torch.device:
        """Configuration du device de calcul."""
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        
        return torch.device(device)
    
    async def load_model(self) -> None:
        """Chargement asynchrone du modèle."""
        try:
            logger.info(f"Chargement du modèle {self.model_name}...")
            
            # Configuration pour optimisation mémoire
            model_kwargs = {
                "torch_dtype": torch.float16 if self.device.type == "cuda" else torch.float32,
                "device_map": "auto" if self.device.type == "cuda" else None,
            }
            
            if self.load_in_4bit and self.device.type == "cuda":
                from transformers import BitsAndBytesConfig
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
            
            # Chargement selon l'architecture
            if "llava" in self.model_name.lower():
                self.processor = LlavaNextProcessor.from_pretrained(self.model_name)
                self.model = LlavaNextForConditionalGeneration.from_pretrained(
                    self.model_name, **model_kwargs
                )
            elif "blip" in self.model_name.lower():
                self.processor = BlipProcessor.from_pretrained(self.model_name)
                self.model = BlipForConditionalGeneration.from_pretrained(
                    self.model_name, **model_kwargs
                )
            else:
                # Fallback générique
                self.processor = AutoProcessor.from_pretrained(self.model_name)
                self.model = AutoModelForVision2Seq.from_pretrained(
                    self.model_name, **model_kwargs
                )
            
            if not model_kwargs.get("device_map"):
                self.model = self.model.to(self.device)
            
            self.model.eval()
            self.is_loaded = True
            
            logger.success(f"Modèle {self.model_name} chargé avec succès")
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement du modèle: {e}")
            raise ModelError(f"Impossible de charger le modèle: {e}")
    
    def _prepare_image(self, image_data: str) -> Image.Image:
        """Préparation de l'image à partir des données base64."""
        try:
            # Décodage base64
            image_bytes = base64.b64decode(image_data)
            image = Image.open(BytesIO(image_bytes)).convert("RGB")
            
            # Redimensionnement pour optimisation
            max_size = 768
            if max(image.size) > max_size:
                ratio = max_size / max(image.size)
                new_size = tuple(int(dim * ratio) for dim in image.size)
                image = image.resize(new_size, Image.Resampling.LANCZOS)
            
            return image
            
        except Exception as e:
            raise ProcessingError(f"Erreur préparation image: {e}")
    
    def _build_tool_calling_prompt(
        self, 
        context: Dict[str, Any],
        available_tools: List[str]
    ) -> str:
        """Construction du prompt pour l'orchestration d'outils."""
        
        # Template de base pour tool-calling
        base_prompt = """Tu es un système de surveillance intelligent spécialisé dans la détection de vol en grande distribution.

CONTEXTE:
- Localisation: {location}
- Heure: {timestamp}
- Détections précédentes: {previous_detections}

OUTILS DISPONIBLES:
{tools_description}

TÂCHE:
Analyse cette image de surveillance et détermine:
1. Le niveau de suspicion (LOW/MEDIUM/HIGH/CRITICAL)
2. Le type d'action observé
3. Les outils à utiliser pour validation
4. Tes recommandations

IMPORTANT:
- Privilégie la précision sur la rapidité
- Évite les faux positifs
- Utilise plusieurs outils pour validation croisée si suspicion élevée
- Réponds au format JSON strict

FORMAT DE RÉPONSE:
```json
{
    "suspicion_level": "LOW|MEDIUM|HIGH|CRITICAL",
    "action_type": "normal_shopping|suspicious_movement|item_concealment|potential_theft|confirmed_theft",
    "confidence": 0.85,
    "description": "Description détaillée de ce qui est observé",
    "tools_to_use": ["tool1", "tool2"],
    "recommendations": ["action1", "action2"]
}
```
"""
        
        # Génération des descriptions d'outils
        tools_description = self._get_tools_description(available_tools)
        
        # Formatage du prompt
        formatted_prompt = base_prompt.format(
            location=context.get("location", "Zone inconnue"),
            timestamp=context.get("timestamp", datetime.now().isoformat()),
            previous_detections=json.dumps(context.get("previous_detections", []), indent=2),
            tools_description=tools_description
        )
        
        return formatted_prompt
    
    def _get_tools_description(self, available_tools: List[str]) -> str:
        """Génération des descriptions d'outils disponibles."""
        
        tools_info = {
            "object_detector": "Détection d'objets avec YOLO - identifie personnes, produits, contenants",
            "tracker": "Suivi de personnes - analyse les mouvements et trajectoires",
            "behavior_analyzer": "Analyse comportementale - détecte gestes suspects",
            "context_validator": "Validation contextuelle - vérifie cohérence avec environnement",
            "false_positive_filter": "Filtre anti-faux positifs - validation finale"
        }
        
        descriptions = []
        for tool in available_tools:
            if tool in tools_info:
                descriptions.append(f"- {tool}: {tools_info[tool]}")
        
        return "\n".join(descriptions)
    
    async def analyze_frame(
        self, 
        request: AnalysisRequest,
        tools_results: Optional[Dict[str, ToolResult]] = None
    ) -> AnalysisResponse:
        """Analyse d'un frame avec orchestration d'outils."""
        
        if not self.is_loaded:
            await self.load_model()
        
        try:
            # Préparation de l'image
            image = self._prepare_image(request.frame_data)
            
            # Construction du prompt avec contexte outils
            prompt = self._build_tool_calling_prompt(
                request.context,
                request.tools_available
            )
            
            # Ajout des résultats d'outils si disponibles
            if tools_results:
                tools_info = self._format_tools_results(tools_results)
                prompt += f"\n\nRÉSULTATS D'OUTILS PRÉCÉDENTS:\n{tools_info}"
            
            # Génération avec le modèle
            response_text = await self._generate_response(image, prompt)
            
            # Parse et validation de la réponse
            analysis_result = self._parse_model_response(response_text)
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Erreur analyse frame: {e}")
            # Réponse par défaut en cas d'erreur
            return AnalysisResponse(
                suspicion_level=SuspicionLevel.LOW,
                action_type=ActionType.NORMAL_SHOPPING,
                confidence=0.0,
                description=f"Erreur d'analyse: {str(e)}",
                tools_used=[],
                recommendations=["Vérification manuelle recommandée"]
            )
    
    async def _generate_response(self, image: Image.Image, prompt: str) -> str:
        """Génération de réponse avec le modèle VLM."""
        
        try:
            # Préparation des inputs
            inputs = self.processor(
                text=prompt,
                images=image,
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            # Génération
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_tokens,
                    temperature=0.1,
                    do_sample=True,
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                    num_return_sequences=1
                )
            
            # Décodage
            generated_text = self.processor.batch_decode(
                generated_ids, 
                skip_special_tokens=True
            )[0]
            
            # Extraction de la réponse (après le prompt)
            if prompt in generated_text:
                response = generated_text.split(prompt)[-1].strip()
            else:
                response = generated_text.strip()
            
            return response
            
        except Exception as e:
            logger.error(f"Erreur génération VLM: {e}")
            raise ProcessingError(f"Erreur génération: {e}")
    
    def _format_tools_results(self, tools_results: Dict[str, ToolResult]) -> str:
        """Formatage des résultats d'outils pour le contexte."""
        
        formatted_results = []
        
        for tool_name, result in tools_results.items():
            status = "✓" if result.success else "✗"
            confidence = f" (confiance: {result.confidence:.2f})" if result.confidence else ""
            
            formatted_results.append(
                f"{status} {tool_name}{confidence}: {json.dumps(result.data, indent=2)}"
            )
        
        return "\n".join(formatted_results)
    
    def _parse_model_response(self, response_text: str) -> AnalysisResponse:
        """Parse et validation de la réponse du modèle."""
        
        try:
            # Extraction du JSON de la réponse
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start == -1 or json_end == 0:
                raise ValueError("Pas de JSON trouvé dans la réponse")
            
            json_text = response_text[json_start:json_end]
            
            # Parse JSON
            parsed = json.loads(json_text)
            
            # Validation et conversion
            return AnalysisResponse(
                suspicion_level=SuspicionLevel(parsed.get("suspicion_level", "LOW")),
                action_type=ActionType(parsed.get("action_type", "normal_shopping")),
                confidence=float(parsed.get("confidence", 0.0)),
                description=parsed.get("description", ""),
                tools_used=parsed.get("tools_to_use", []),
                recommendations=parsed.get("recommendations", [])
            )
            
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.warning(f"Erreur parse réponse VLM: {e}. Réponse brute: {response_text}")
            
            # Fallback: analyse heuristique simple
            return self._fallback_analysis(response_text)
    
    def _fallback_analysis(self, response_text: str) -> AnalysisResponse:
        """Analyse de fallback en cas d'erreur de parsing."""
        
        text_lower = response_text.lower()
        
        # Détection de mots-clés pour suspicion
        if any(word in text_lower for word in ["vol", "theft", "suspicious", "steal"]):
            suspicion = SuspicionLevel.HIGH
            action = ActionType.SUSPICIOUS_MOVEMENT
            confidence = 0.6
        elif any(word in text_lower for word in ["caché", "conceal", "hidden"]):
            suspicion = SuspicionLevel.MEDIUM
            action = ActionType.ITEM_CONCEALMENT
            confidence = 0.4
        else:
            suspicion = SuspicionLevel.LOW
            action = ActionType.NORMAL_SHOPPING
            confidence = 0.3
        
        return AnalysisResponse(
            suspicion_level=suspicion,
            action_type=action,
            confidence=confidence,
            description=f"Analyse de fallback: {response_text[:200]}...",
            tools_used=[],
            recommendations=["Analyse manuelle recommandée"]
        )
    
    def unload_model(self) -> None:
        """Déchargement du modèle pour libérer la mémoire."""
        if self.model is not None:
            del self.model
            del self.processor
            
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            
            self.model = None
            self.processor = None
            self.is_loaded = False
            
            logger.info("Modèle VLM déchargé")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Informations sur le modèle chargé."""
        return {
            "model_name": self.model_name,
            "device": str(self.device),
            "is_loaded": self.is_loaded,
            "max_tokens": self.max_tokens,
            "load_in_4bit": self.load_in_4bit,
            "memory_usage_mb": self._get_memory_usage()
        }
    
    def _get_memory_usage(self) -> Optional[float]:
        """Usage mémoire GPU/CPU du modèle."""
        if not self.is_loaded:
            return None
        
        try:
            if self.device.type == "cuda":
                return torch.cuda.memory_allocated(self.device) / 1024**2
            else:
                # Pour CPU, approximation basée sur les paramètres
                if hasattr(self.model, 'num_parameters'):
                    return self.model.num_parameters() * 4 / 1024**2  # 4 bytes par paramètre
        except:
            pass
        
        return None