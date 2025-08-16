"""VLM avec support dynamique de KIM, LLaVA et Qwen2-VL."""

import asyncio
from typing import Dict, List, Optional, Any
import torch
from PIL import Image
import base64
from io import BytesIO
from loguru import logger

from ..types import AnalysisRequest, AnalysisResponse, ToolResult
from ...utils.exceptions import ModelError, ProcessingError
from .model_registry import VLMModelRegistry, VLMModelType
from .prompt_builder import PromptBuilder
from .response_parser import ResponseParser
from .tools_integration import AdvancedToolsManager


class DynamicVisionLanguageModel:
    """
    VLM dynamique supportant KIM, LLaVA et Qwen2-VL avec switch à chaud.
    
    Fonctionnalités:
    - Switch entre modèles sans redémarrer
    - Configuration optimisée par modèle
    - Fallbacks intelligents
    - Monitoring de performance
    """
    
    def __init__(
        self,
        default_model: str = "kimi-vl-a3b-thinking",
        device: str = "auto",
        enable_fallback: bool = True
    ):
        self.device = self._setup_device(device)
        self.enable_fallback = enable_fallback
        
        # Registre des modèles
        self.model_registry = VLMModelRegistry()
        
        # État actuel
        self.current_model_id = None
        self.current_config = None
        self.model = None
        self.processor = None
        self.is_loaded = False
        
        # Composants modulaires
        self.prompt_builder = PromptBuilder()
        self.response_parser = ResponseParser()
        self.tools_manager = AdvancedToolsManager()
        
        # Cache et performance
        self._model_cache = {}
        self._performance_stats = {}
        
        # Chargement du modèle par défaut
        self.default_model = default_model
        logger.info(f"VLM Dynamique initialisé - Modèle par défaut: {default_model}")
    
    def _setup_device(self, device: str) -> torch.device:
        """Configuration du device."""
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        return torch.device(device)
    
    async def load_model(self, model_id: str = None) -> bool:
        """Chargement d'un modèle spécifique."""
        
        model_id = model_id or self.default_model
        
        # Vérification disponibilité
        is_available, message = self.model_registry.validate_model_availability(model_id)
        if not is_available:
            logger.error(f"Modèle {model_id} non disponible: {message}")
            if self.enable_fallback:
                return await self._load_fallback_model()
            raise ModelError(f"Modèle {model_id} indisponible: {message}")
        
        # Récupération config
        config = self.model_registry.get_model_config(model_id)
        if not config:
            raise ModelError(f"Configuration manquante pour {model_id}")
        
        try:
            logger.info(f"Chargement {config.model_type.value.upper()} : {model_id}")
            
            # Chargement selon le type
            success = await self._load_model_by_type(config)
            
            if success:
                self.current_model_id = model_id
                self.current_config = config
                self.is_loaded = True
                
                logger.success(f"✅ {model_id} chargé avec succès")
                return True
            else:
                logger.error(f"❌ Échec chargement {model_id}")
                if self.enable_fallback:
                    return await self._load_fallback_model()
                return False
                
        except Exception as e:
            logger.error(f"Erreur chargement {model_id}: {e}")
            if self.enable_fallback:
                return await self._load_fallback_model()
            raise ModelError(f"Impossible de charger {model_id}: {e}")
    
    async def _load_model_by_type(self, config) -> bool:
        """Chargement selon le type de modèle."""
        
        try:
            # Configuration commune
            model_kwargs = {
                "torch_dtype": getattr(torch, config.default_params.get("torch_dtype", "float16")),
                "device_map": config.default_params.get("device_map", "auto")
            }
            
            # Quantization si supportée
            if config.default_params.get("load_in_4bit") and self.device.type == "cuda":
                from transformers import BitsAndBytesConfig
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
            elif config.default_params.get("load_in_8bit") and self.device.type == "cuda":
                from transformers import BitsAndBytesConfig
                model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            
            # Trust remote code si nécessaire
            if config.default_params.get("trust_remote_code"):
                model_kwargs["trust_remote_code"] = True
            
            # === CHARGEMENT PAR TYPE ===
            
            if config.model_type == VLMModelType.KIMI_VL:
                success = await self._load_kimi_vl_model(config, model_kwargs)
            
            
            elif config.model_type == VLMModelType.QWEN:
                success = await self._load_qwen_model(config, model_kwargs)
            
            else:
                logger.error(f"Type de modèle non supporté: {config.model_type}")
                return False
            
            # Configuration finale
            if success and self.model:
                if not model_kwargs.get("device_map"):
                    self.model = self.model.to(self.device)
                self.model.eval()
            
            return success
            
        except Exception as e:
            logger.error(f"Erreur chargement modèle: {e}")
            return False
    
    async def _load_kimi_vl_model(self, config, model_kwargs) -> bool:
        """Chargement spécifique Kimi-VL (Moonshot AI)."""
        try:
            # Kimi-VL utilise transformers>=4.48.2 avec trust_remote_code
            from transformers import AutoProcessor, AutoModelForCausalLM
            
            logger.info(f"Chargement Kimi-VL : {config.model_name}")
            
            # Chargement du processor 
            self.processor = AutoProcessor.from_pretrained(
                config.model_name, 
                trust_remote_code=True
            )
            
            # Chargement du modèle (AutoModelForCausalLM pour Kimi-VL)
            self.model = AutoModelForCausalLM.from_pretrained(
                config.model_name, 
                **model_kwargs
            )
            
            logger.info("✅ Kimi-VL chargé avec succès")
            return True
            
        except Exception as e:
            logger.error(f"Erreur chargement Kimi-VL: {e}")
            logger.warning("💡 Vérifiez: transformers>=4.48.2 et trust_remote_code=True")
            return False
    
    
    async def _load_qwen_model(self, config, model_kwargs) -> bool:
        """Chargement spécifique Qwen2-VL."""
        try:
            # Qwen2-VL utilise transformers mais avec trust_remote_code
            from transformers import AutoProcessor, AutoModelForVision2Seq
            
            self.processor = AutoProcessor.from_pretrained(
                config.model_name, 
                trust_remote_code=True
            )
            self.model = AutoModelForVision2Seq.from_pretrained(
                config.model_name, 
                **model_kwargs
            )
            
            logger.info("✅ Qwen2-VL chargé avec succès")
            return True
            
        except Exception as e:
            logger.error(f"Erreur chargement Qwen2-VL: {e}")
            return False
    
    async def _load_fallback_model(self) -> bool:
        """Chargement d'un modèle de fallback."""
        logger.warning("Tentative de chargement du modèle de fallback...")
        
        # Ordre de priorité pour fallback
        fallback_models = [
            "qwen2-vl-7b-instruct",    # Qwen principal
            "kimi-vl-a3b-instruct",    # Kimi alternatif
        ]
        
        for fallback_id in fallback_models:
            try:
                logger.info(f"Tentative fallback: {fallback_id}")
                success = await self.load_model(fallback_id)
                if success:
                    logger.warning(f"✅ Fallback réussi avec {fallback_id}")
                    return True
            except Exception as e:
                logger.error(f"Fallback {fallback_id} échoué: {e}")
                continue
        
        logger.error("❌ Aucun modèle de fallback disponible")
        return False
    
    async def switch_model(self, new_model_id: str) -> bool:
        """Switch vers un nouveau modèle à chaud."""
        
        if new_model_id == self.current_model_id:
            logger.info(f"Modèle {new_model_id} déjà chargé")
            return True
        
        logger.info(f"Switch: {self.current_model_id} → {new_model_id}")
        
        # Sauvegarde de l'état actuel
        old_model_id = self.current_model_id
        
        # Déchargement propre
        self._unload_current_model()
        
        # Chargement du nouveau modèle
        success = await self.load_model(new_model_id)
        
        if success:
            logger.success(f"✅ Switch réussi vers {new_model_id}")
            return True
        else:
            logger.error(f"❌ Switch échoué, retour à {old_model_id}")
            # Tentative de retour à l'ancien modèle
            if old_model_id:
                await self.load_model(old_model_id)
            return False
    
    def _unload_current_model(self):
        """Déchargement propre du modèle actuel."""
        if self.model is not None:
            del self.model
            del self.processor
            
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
        
        self.model = None
        self.processor = None
        self.is_loaded = False
        self.current_model_id = None
        self.current_config = None
    
    async def analyze_with_tools(
        self, 
        request: AnalysisRequest,
        use_advanced_tools: bool = True
    ) -> AnalysisResponse:
        """Analyse avec le modèle actuellement chargé."""
        
        if not self.is_loaded:
            success = await self.load_model()
            if not success:
                raise ModelError("Aucun modèle VLM disponible")
        
        try:
            # Préparation de l'image
            image = self._prepare_image(request.frame_data)
            image_array = self._pil_to_numpy(image)
            
            # Exécution des outils si demandés
            tools_results = {}
            if use_advanced_tools and request.tools_available:
                logger.info(f"Exécution outils avec {self.current_model_id}: {request.tools_available}")
                tools_results = await self.tools_manager.execute_tools(
                    image_array, 
                    request.tools_available,
                    request.context
                )
            
            # Construction du prompt optimisé pour le modèle
            prompt = self._build_model_specific_prompt(request, tools_results)
            
            # Génération avec le modèle actuel
            response_text = await self._generate_response(image, prompt)
            
            # Parse de la réponse
            analysis_result = self.response_parser.parse_vlm_response(response_text)
            
            # Ajout d'informations sur le modèle utilisé
            analysis_result.description += f" | Modèle: {self.current_model_id}"
            
            logger.success(f"Analyse {self.current_model_id}: {analysis_result.suspicion_level.value}")
            return analysis_result
            
        except Exception as e:
            logger.error(f"Erreur analyse {self.current_model_id}: {e}")
            return self._create_error_response(str(e))
    
    def _build_model_specific_prompt(self, request: AnalysisRequest, tools_results: Dict) -> str:
        """Construction de prompt optimisé selon le modèle."""
        
        # Prompt de base
        base_prompt = self.prompt_builder.build_surveillance_prompt(
            request.context,
            request.tools_available,
            tools_results
        )
        
        # Optimisations spécifiques par modèle
        if self.current_config and self.current_config.model_type == VLMModelType.KIM:
            # KIM préfère des prompts structurés avec étapes
            base_prompt += "\n\nRaisonne étape par étape pour cette analyse de surveillance."
        
        elif self.current_config and self.current_config.model_type == VLMModelType.QWEN:
            # Qwen2-VL excelle avec des instructions précises
            base_prompt += "\n\nSois précis et factuel dans ton analyse. Justifie chaque conclusion."
        
        
        return base_prompt
    
    async def _generate_response(self, image: Image.Image, prompt: str) -> str:
        """Génération optimisée selon le modèle."""
        
        try:
            # Préparation des inputs
            inputs = self.processor(
                text=prompt,
                images=image,
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            # Paramètres de génération du modèle actuel
            gen_params = {
                "max_new_tokens": self.current_config.default_params.get("max_new_tokens", 512),
                "temperature": self.current_config.default_params.get("temperature", 0.1),
                "do_sample": self.current_config.default_params.get("do_sample", True),
                "pad_token_id": self.processor.tokenizer.eos_token_id if hasattr(self.processor, 'tokenizer') else None
            }
            
            # Génération
            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, **gen_params)
            
            # Décodage
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # Extraction de la réponse
            if prompt in generated_text:
                response = generated_text.split(prompt)[-1].strip()
            else:
                response = generated_text.strip()
            
            return response
            
        except Exception as e:
            logger.error(f"Erreur génération {self.current_model_id}: {e}")
            raise ProcessingError(f"Génération échouée: {e}")
    
    def _prepare_image(self, image_data: str) -> Image.Image:
        """Préparation de l'image."""
        try:
            image_bytes = base64.b64decode(image_data)
            image = Image.open(BytesIO(image_bytes)).convert("RGB")
            
            # Redimensionnement optimal selon le modèle
            max_size = 768  # Standard pour la plupart des VLM
            if self.current_config and self.current_config.model_type == VLMModelType.QWEN:
                max_size = 1024  # Qwen2-VL supporte des images plus grandes
            
            if max(image.size) > max_size:
                ratio = max_size / max(image.size)
                new_size = tuple(int(dim * ratio) for dim in image.size)
                image = image.resize(new_size, Image.Resampling.LANCZOS)
            
            return image
            
        except Exception as e:
            raise ProcessingError(f"Erreur préparation image: {e}")
    
    def _pil_to_numpy(self, image: Image.Image):
        """Conversion PIL vers numpy pour les outils."""
        import numpy as np
        return np.array(image)
    
    def _create_error_response(self, error: str) -> AnalysisResponse:
        """Réponse d'erreur."""
        from ..types import SuspicionLevel, ActionType
        
        return AnalysisResponse(
            suspicion_level=SuspicionLevel.LOW,
            action_type=ActionType.NORMAL_SHOPPING,
            confidence=0.0,
            description=f"Erreur VLM ({self.current_model_id}): {error}",
            tools_used=[],
            recommendations=["Vérification système requise"]
        )
    
    def get_system_status(self) -> Dict[str, Any]:
        """État complet du système multi-modèles."""
        
        # Statut des outils
        tools_status = self.tools_manager.get_tools_status()
        available_tools = self.tools_manager.get_available_tools()
        
        # Modèles disponibles
        available_models = {}
        for model_id, config in self.model_registry.list_available_models().items():
            is_available, message = self.model_registry.validate_model_availability(model_id)
            available_models[model_id] = {
                "available": is_available,
                "message": message,
                "type": config.model_type.value,
                "description": config.description
            }
        
        return {
            "current_model": {
                "model_id": self.current_model_id,
                "model_type": self.current_config.model_type.value if self.current_config else None,
                "is_loaded": self.is_loaded,
                "device": str(self.device),
                "supports_tool_calling": self.current_config.supports_tool_calling if self.current_config else False
            },
            "available_models": available_models,
            "tools_status": tools_status,
            "available_tools": available_tools,
            "system": {
                "enable_fallback": self.enable_fallback,
                "device": str(self.device),
                "cuda_available": torch.cuda.is_available()
            }
        }
    
    def get_model_recommendations(self) -> Dict[str, str]:
        """Recommandations de modèles selon les cas d'usage."""
        return {
            "surveillance_principal": self.model_registry.get_recommended_model("surveillance"),
            "haute_performance": self.model_registry.get_recommended_model("high_performance"),
            "economie_memoire": self.model_registry.get_recommended_model("memory_efficient"),
            "raisonnement_avance": self.model_registry.get_recommended_model("reasoning"),
            "flagship": self.model_registry.get_recommended_model("flagship")
        }
    
    async def shutdown(self):
        """Arrêt propre."""
        logger.info("Arrêt du VLM dynamique...")
        self._unload_current_model()
        logger.info("VLM dynamique arrêté")