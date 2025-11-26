"""Modèle VLM refactorisé avec intégration des outils avancés."""

import base64
from io import BytesIO
from typing import Dict, List, Optional, Any
import asyncio

import torch
from transformers import (
    AutoProcessor, 
    AutoModelForVision2Seq,
    BlipProcessor,
    BlipForConditionalGeneration
)
from PIL import Image
import numpy as np
from loguru import logger

from ..types import AnalysisRequest, AnalysisResponse, ToolResult
from ...utils.exceptions import ModelError, ProcessingError

from .prompt_builder import PromptBuilder
from .response_parser import ResponseParser
from .tools_integration import AdvancedToolsManager


class VisionLanguageModel:
    """
    Modèle VLM modernisé avec intégration complète des 8 outils avancés.
    
    Architecture modulaire:
    - PromptBuilder: Construction des prompts
    - ResponseParser: Parsing des réponses 
    - AdvancedToolsManager: Gestion des outils avancés
    """
    
    def __init__(
        self, 
        model_name: str = "moonshotai/Kimi-VL-A3B-Thinking",
        device: str = "auto",
        load_in_4bit: bool = True,
        max_tokens: int = 512
    ):
        self.model_name = model_name
        self.device = self._setup_device(device)
        self.max_tokens = max_tokens
        self.load_in_4bit = load_in_4bit
        
        # Composants modulaires
        self.prompt_builder = PromptBuilder()
        self.response_parser = ResponseParser()
        self.tools_manager = AdvancedToolsManager()
        
        # État du modèle VLM
        self.model = None
        self.processor = None
        self.is_loaded = False
        
        logger.info(f"VLM initialisé: {model_name} sur {self.device}")
        logger.info(f"Outils avancés disponibles: {self.tools_manager.get_available_tools()}")
    
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
        """Chargement asynchrone du modèle VLM."""
        try:
            logger.info(f"Chargement VLM {self.model_name}...")
            
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
            if "blip" in self.model_name.lower():
                self.processor = BlipProcessor.from_pretrained(self.model_name)
                self.model = BlipForConditionalGeneration.from_pretrained(
                    self.model_name, **model_kwargs
                )
            else:
                self.processor = AutoProcessor.from_pretrained(self.model_name)
                self.model = AutoModelForVision2Seq.from_pretrained(
                    self.model_name, **model_kwargs
                )
            
            if not model_kwargs.get("device_map"):
                self.model = self.model.to(self.device)
            
            self.model.eval()
            self.is_loaded = True
            
            logger.success(f"VLM {self.model_name} chargé avec succès")
            
        except Exception as e:
            logger.error(f"Erreur chargement VLM: {e}")
            raise ModelError(f"Impossible de charger le VLM: {e}")
    
    async def analyze_with_tools(
        self, 
        request: AnalysisRequest,
        use_advanced_tools: bool = True
    ) -> AnalysisResponse:
        """Analyse complète avec orchestration des outils avancés."""
        
        if not self.is_loaded:
            await self.load_model()
        
        try:
            # 1. Préparation de l'image
            image = self._prepare_image(request.frame_data)
            image_array = np.array(image)
            
            # 2. Exécution des outils si demandés
            tools_results = {}
            if use_advanced_tools and request.tools_available:
                logger.info(f"Exécution outils: {request.tools_available}")
                tools_results = await self.tools_manager.execute_tools(
                    image_array, 
                    request.tools_available,
                    request.context
                )
                logger.debug(f"Résultats outils: {list(tools_results.keys())}")
            
            # 3. Construction du prompt avec contexte enrichi
            video_context = request.context.get('video_context_metadata', None)
            prompt = self.prompt_builder.build_surveillance_prompt(
                request.context,
                request.tools_available,
                tools_results,
                video_context
            )
            
            # 4. Génération VLM
            response_text = await self._generate_response(image, prompt)
            
            # 5. Parse de la réponse
            analysis_result = self.response_parser.parse_vlm_response(response_text)
            
            logger.success(f"Analyse terminée - Suspicion: {analysis_result.suspicion_level.value}")
            return analysis_result
            
        except Exception as e:
            logger.error(f"Erreur analyse avec outils: {e}")
            return self._emergency_fallback(str(e))
    
    async def _generate_response(self, image: Image.Image, prompt: str) -> str:
        """Génération de réponse avec le VLM."""
        
        try:
            # Préparation des inputs
            inputs = self.processor(
                text=prompt,
                images=image,
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            # Génération avec contrôle de température
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_tokens,
                    temperature=0.1,  # Température basse pour cohérence
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
            
            logger.debug(f"Réponse VLM générée: {response[:200]}...")
            return response
            
        except Exception as e:
            logger.error(f"Erreur génération VLM: {e}")
            raise ProcessingError(f"Erreur génération: {e}")
    
    def _prepare_image(self, image_data: str) -> Image.Image:
        """Préparation optimisée de l'image."""
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
    
    def _emergency_fallback(self, error: str) -> AnalysisResponse:
        """Réponse d'urgence en cas d'erreur critique."""
        from ..types import SuspicionLevel, ActionType
        
        return AnalysisResponse(
            suspicion_level=SuspicionLevel.LOW,
            action_type=ActionType.NORMAL_SHOPPING,
            confidence=0.0,
            description=f"Erreur système VLM: {error}",
            tools_used=[],
            recommendations=["Vérification système urgente", "Analyse manuelle requise"]
        )
    
    def get_system_status(self) -> Dict[str, Any]:
        """État complet du système VLM + outils."""
        return {
            "vlm_model": {
                "name": self.model_name,
                "loaded": self.is_loaded,
                "device": str(self.device),
                "memory_mb": self._get_memory_usage()
            },
            "tools_status": self.tools_manager.get_tools_status(),
            "available_tools": self.tools_manager.get_available_tools()
        }
    
    def _get_memory_usage(self) -> Optional[float]:
        """Usage mémoire GPU/CPU du VLM."""
        if not self.is_loaded:
            return None
        
        try:
            if self.device.type == "cuda":
                return torch.cuda.memory_allocated(self.device) / 1024**2
            else:
                if hasattr(self.model, 'num_parameters'):
                    return self.model.num_parameters() * 4 / 1024**2
        except:
            pass
        return None
    
    def unload_model(self) -> None:
        """Déchargement complet du système."""
        if self.model is not None:
            del self.model
            del self.processor
            
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            
            self.model = None
            self.processor = None
            self.is_loaded = False
            
            logger.info("VLM et outils déchargés")