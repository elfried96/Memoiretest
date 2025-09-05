"""VLM avec architecture dual-VLM simplifiée: Kimi-VL-Thinking + Qwen2-VL fallback."""

import asyncio
from typing import Dict, List, Optional, Any
import torch
from PIL import Image
import base64
from io import BytesIO
from loguru import logger

# Import pour Qwen2-VL processing
try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    def process_vision_info(conversations):
        """Fallback function si qwen_vl_utils n'est pas disponible."""
        logger.warning("qwen_vl_utils non disponible, utilisation fallback")
        return [], []

from ..types import AnalysisRequest, AnalysisResponse, ToolResult
from ...utils.exceptions import ModelError, ProcessingError
from .model_registry import VLMModelRegistry, VLMModelType
from .prompt_builder import PromptBuilder
from .response_parser import ResponseParser
from .tools_integration import AdvancedToolsManager
from .tool_schema import ToolSchemaBuilder


class DynamicVisionLanguageModel:
    """
    VLM avec architecture mono-VLM selon standards 2025.
    
    Architecture:
    - UNIQUE: Kimi-VL-A3B-Thinking avec Tool Calling natif
    - AUCUN FALLBACK: Échec propre selon standards 2025
    - Tool Calling officiel avec schémas JSON stricts
    - Monitoring de performance intégré
    """
    
    def __init__(
        self,
        default_model: str = "kimi-vl-a3b-thinking",
        device: str = "auto",
        enable_fallback: bool = False  # SUPPRIMÉ: Plus de fallback selon standards 2025
    ):
        self.device = self._setup_device(device)
        # SUPPRIMÉ: enable_fallback selon standards 2025
        
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
        
        # Performance stats seulement (pas de cache par défaut pour économiser mémoire)
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
        """Chargement d'un modèle spécifique - Standards 2025 sans fallback."""
        return await self._load_model_direct(model_id, enable_fallback=False)
    
    async def _load_model_direct(self, model_id: str = None, enable_fallback: bool = False) -> bool:
        """Chargement direct d'un modèle sans fallback automatique."""
        
        model_id = model_id or self.default_model
        
        # Vérification disponibilité
        is_available, message = self.model_registry.validate_model_availability(model_id)
        if not is_available:
            logger.error(f"Modèle {model_id} non disponible: {message}")
            # SUPPRIMÉ: Pas de fallback selon standards 2025
            logger.error("❌ Modèle requis indisponible - arrêt selon standards 2025")
            return False
        
        # Récupération config
        config = self.model_registry.get_model_config(model_id)
        if not config:
            logger.error(f"Configuration manquante pour {model_id}")
            # SUPPRIMÉ: Pas de fallback selon standards 2025
            logger.error("❌ Modèle requis indisponible - arrêt selon standards 2025")
            return False
        
        try:
            logger.info(f"Chargement {config.model_type.value.upper()} : {model_id}")
            
            # Chargement selon le type
            success = await self._load_model_by_type(config)
            
            if success:
                self.current_model_id = model_id
                self.current_config = config
                self.is_loaded = True
                
                logger.info(f"✅ {model_id} chargé avec succès")
                return True
            else:
                logger.error(f"❌ Échec chargement {model_id}")
                if enable_fallback and self.enable_fallback:
                    return await self._load_fallback_model(exclude_models=[model_id])
                return False
                
        except Exception as e:
            logger.error(f"Erreur chargement {model_id}: {e}")
            # SUPPRIMÉ: Pas de fallback selon standards 2025
            logger.error("❌ Modèle requis indisponible - arrêt selon standards 2025")
            return False
    
    async def _load_model_by_type(self, config) -> bool:
        """Chargement selon le type de modèle."""
        
        try:
            # Configuration commune - FIX: torch.float16 au lieu de torch.auto
            torch_dtype_str = config.default_params.get("torch_dtype", "float16")
            if torch_dtype_str == "auto":
                torch_dtype = torch.float16 if self.device.type == "cuda" else torch.float32
            else:
                torch_dtype = getattr(torch, torch_dtype_str, torch.float16)
                
            model_kwargs = {
                "torch_dtype": torch_dtype,
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
            
            # Forcer l'usage CPU pour éviter les problèmes SDPA
            model_kwargs_fixed = model_kwargs.copy()
            model_kwargs_fixed["attn_implementation"] = "eager"  # Évite SDPA
            
            # Chargement du processor 
            self.processor = AutoProcessor.from_pretrained(
                config.model_name, 
                trust_remote_code=True
            )
            
            # Chargement du modèle (AutoModelForCausalLM pour Kimi-VL)
            self.model = AutoModelForCausalLM.from_pretrained(
                config.model_name, 
                **model_kwargs_fixed
            )
            
            # Fix pour l'attribut _supports_sdpa manquant
            if not hasattr(self.model, '_supports_sdpa'):
                self.model._supports_sdpa = False
            
            logger.info("✅ Kimi-VL chargé avec succès")
            return True
            
        except Exception as e:
            logger.error(f"Erreur chargement Kimi-VL: {e}")
            logger.warning("💡 Vérifiez: transformers>=4.48.2 et trust_remote_code=True")
            return False
    
    
    async def _load_qwen_model(self, config, model_kwargs) -> bool:
        """Chargement spécifique Qwen2-VL et Qwen2.5-VL."""
        try:
            from transformers import AutoProcessor
            
            # Support Qwen2.5-VL et Qwen2-VL
            if "qwen2.5-vl" in config.model_name.lower():
                from transformers import Qwen2_5_VLForConditionalGeneration
                model_class = Qwen2_5_VLForConditionalGeneration
                logger.info(f"Chargement Qwen2.5-VL : {config.model_name}")
            else:
                from transformers import AutoModelForVision2Seq
                model_class = AutoModelForVision2Seq
                logger.info(f"Chargement Qwen2-VL : {config.model_name}")
            
            self.processor = AutoProcessor.from_pretrained(
                config.model_name, 
                trust_remote_code=True
            )
            self.model = model_class.from_pretrained(
                config.model_name, 
                **model_kwargs
            )
            
            logger.info("✅ Qwen VL chargé avec succès")
            return True
            
        except Exception as e:
            logger.error(f"Erreur chargement Qwen VL: {e}")
            return False
    
    # SUPPRIMÉ: Méthode _load_fallback_model selon standards 2025 - pas de fallback
    
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
            logger.error(f"❌ Switch échoué selon standards 2025 - pas de retour automatique")
            return False
    
    def _unload_current_model(self):
        """Déchargement propre du modèle actuel avec nettoyage mémoire optimisé."""
        if self.model is not None:
            # Libération de la mémoire
            if hasattr(self.model, 'cpu'):
                self.model.cpu()
            del self.model
            
        if self.processor is not None:
            del self.processor
            
        # Nettoyage mémoire avancé
        import gc
        gc.collect()  # Garbage collection Python
            
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        self.model = None
        self.processor = None
        self.is_loaded = False
        self.current_model_id = None
        self.current_config = None
        
        logger.info("✅ Modèle déchargé et mémoire libérée (optimisé)")
    
    async def analyze_with_tools(
        self, 
        request: AnalysisRequest,
        use_advanced_tools: bool = True
    ) -> AnalysisResponse:
        """Analyse avec le modèle actuellement chargé."""
        
        # Vérification plus robuste du modèle chargé
        if not self.is_loaded or self.model is None or self.current_model_id != (request.preferred_model or self.default_model):
            logger.info(f"Rechargement nécessaire: loaded={self.is_loaded}, model={self.model is not None}, current={self.current_model_id}")
            success = await self.load_model(request.preferred_model or self.default_model)
            if not success:
                raise ModelError("Aucun modèle VLM disponible")
        else:
            logger.info(f"✅ Modèle {self.current_model_id} déjà chargé - réutilisation du cache")
        
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
            
            # Affichage détaillé des décisions VLM
            self._log_vlm_decision(response_text, analysis_result)
            
            logger.success(f"Analyse {self.current_model_id}: {analysis_result.suspicion_level.value}")
            return analysis_result
            
        except Exception as e:
            logger.error(f"Erreur analyse {self.current_model_id}: {e}")
            return self._create_error_response(str(e))
    
    async def analyze_with_custom_prompt(
        self, 
        frame_data: str,
        custom_prompt: str,
        context: Dict[str, Any] = None
    ) -> AnalysisResponse:
        """Analyse avec prompt personnalisé pour résumés cumulatifs."""
        
        # Créer une requête simplifiée
        request = AnalysisRequest(
            frame_data=frame_data,
            context=context or {},
            tools_available=[],  # Pas d'outils pour les résumés
            preferred_model=self.current_model_id
        )
        
        try:
            # Vérifier que le modèle est chargé
            if not self.is_loaded:
                success = await self.load_model(request.preferred_model or self.default_model)
                if not success:
                    raise ModelError("Aucun modèle VLM disponible")
            
            # Préparation de l'image
            image = self._prepare_image(request.frame_data)
            
            # Génération avec prompt personnalisé
            response_text = await self._generate_response(image, custom_prompt)
            
            # Parse de la réponse (simple car pas d'outils)
            analysis_result = self.response_parser.parse_vlm_response(response_text)
            
            logger.success(f"Analyse custom {self.current_model_id}: {analysis_result.suspicion_level.value}")
            return analysis_result
            
        except Exception as e:
            logger.error(f"Erreur analyse custom {self.current_model_id}: {e}")
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
            logger.debug(f"Début génération {self.current_model_id}")
            
            # Préparation des inputs avec correction Qwen2-VL
            logger.debug("Préparation des inputs...")
            
            if self.current_config and self.current_config.model_type == VLMModelType.QWEN:
                # Traitement spécifique pour Qwen2-VL
                conversation = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": prompt}
                        ]
                    }
                ]
                inputs = self.processor.apply_chat_template(
                    conversation, tokenize=False, add_generation_prompt=True
                )
                image_inputs, video_inputs = process_vision_info([conversation])
                inputs = self.processor(
                    text=[inputs],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt"
                ).to(self.device)
            elif self.current_config and self.current_config.model_type == VLMModelType.KIMI_VL:
                # Traitement spécifique pour Kimi-VL selon documentation officielle
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": prompt}
                        ]
                    }
                ]
                # Format officiel Kimi-VL avec apply_chat_template
                text = self.processor.apply_chat_template(
                    messages, add_generation_prompt=True, return_tensors="pt"
                )
                inputs = self.processor(
                    images=image, 
                    text=text, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True
                ).to(self.device)
            else:
                # Traitement standard pour les autres modèles
                inputs = self.processor(
                    text=prompt,
                    images=image,
                    return_tensors="pt",
                    padding=True
                ).to(self.device)
            
            logger.debug(f"Inputs préparés, device: {self.device}")
            
            # Paramètres de génération selon documentation officielle - OPTIMISÉS SURVEILLANCE
            if self.current_config and self.current_config.model_type == VLMModelType.KIMI_VL:
                # Paramètres Kimi-VL OPTIMISÉS pour surveillance temps réel
                gen_params = {
                    "max_new_tokens": 150,  # ✅ Réduit pour vitesse (512→150)
                    "temperature": 0.3,     # ✅ Réduit pour vitesse et cohérence (0.8→0.3)  
                    "do_sample": True,
                    "pad_token_id": self.processor.tokenizer.eos_token_id if hasattr(self.processor, 'tokenizer') else None,
                    "use_cache": True,      # ✅ ACTIVER cache pour vitesse
                    "return_dict_in_generate": False,
                    "output_attentions": False,
                    "output_hidden_states": False
                }
            else:
                # Paramètres standards pour autres modèles
                gen_params = {
                    "max_new_tokens": self.current_config.default_params.get("max_new_tokens", 512),
                    "temperature": self.current_config.default_params.get("temperature", 0.1),
                    "do_sample": self.current_config.default_params.get("do_sample", True),
                    "pad_token_id": self.processor.tokenizer.eos_token_id if hasattr(self.processor, 'tokenizer') else None,
                    "use_cache": False,  # CRITIQUE: Désactiver cache pour éviter DynamicCache error
                    "return_dict_in_generate": False,  # Simplifier retour
                    "output_attentions": False,
                    "output_hidden_states": False
                }
            logger.debug(f"Paramètres de génération: {gen_params}")
            
            # Génération avec gestion d'erreurs améliorée
            logger.debug("Début génération du modèle...")
            with torch.no_grad():
                try:
                    generated_ids = self.model.generate(**inputs, **gen_params)
                except Exception as gen_error:
                    # Réessayer avec des paramètres plus sûrs
                    logger.warning(f"Erreur génération, réessai avec paramètres basiques: {gen_error}")
                    
                    # Paramètres ultra-sécurisés pour éviter DynamicCache
                    safe_params = {
                        "max_new_tokens": 128,
                        "do_sample": False,  # Greedy decoding
                        "num_beams": 1,
                        "early_stopping": True,
                        "use_cache": False,  # CRITIQUE: Désactiver cache
                        "return_dict_in_generate": False,
                        "output_attentions": False,
                        "output_hidden_states": False
                    }
                    
                    # Ajouter pad_token_id seulement s'il existe
                    if hasattr(self.processor, 'tokenizer') and self.processor.tokenizer.eos_token_id is not None:
                        safe_params["pad_token_id"] = self.processor.tokenizer.eos_token_id
                    
                    try:
                        generated_ids = self.model.generate(**inputs, **safe_params)
                    except Exception as final_error:
                        logger.error(f"Échec génération définitif: {final_error}")
                        # Standards 2025: Échec propre sans fallback
                        raise ProcessingError(f"Génération VLM 2025 échouée définitivement: {final_error}")
            logger.debug("Génération terminée")
            
            # Décodage
            logger.debug("Début décodage...")
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            logger.debug(f"Décodage terminé, longueur: {len(generated_text)}")
            
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
                "standards_2025": True,  # SUPPRIMÉ: enable_fallback selon standards 2025
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
    
    def _log_vlm_decision(self, response_text: str, analysis_result) -> None:
        """Affichage détaillé de la décision VLM."""
        print("\n" + "="*80)
        print("🧠 ANALYSE VLM DÉTAILLÉE")
        print("="*80)
        
        # Essayer d'extraire le thinking du JSON
        try:
            import json
            # Chercher le JSON dans la réponse
            if "thinking" in response_text.lower():
                start = response_text.find("{")
                end = response_text.rfind("}")
                if start != -1 and end != -1:
                    json_str = response_text[start:end+1]
                    parsed = json.loads(json_str)
                    
                    if "thinking" in parsed:
                        print("🤔 THINKING PROCESSUS:")
                        print(parsed["thinking"])
                        print()
                    
                    if "observations" in parsed:
                        print("👁️ OBSERVATIONS:")
                        obs = parsed["observations"]
                        if isinstance(obs, dict):
                            for key, value in obs.items():
                                print(f"  • {key}: {value}")
                        else:
                            print(f"  {obs}")
                        print()
                    
                    if "decision_final" in parsed:
                        print("⚖️ DÉCISION FINALE:")
                        print(parsed["decision_final"])
                        print()
                        
        except Exception:
            # Si parsing JSON échoue, afficher la réponse brute
            print("📄 RÉPONSE COMPLÈTE VLM:")
            print(response_text[:1000] + "..." if len(response_text) > 1000 else response_text)
            print()
        
        # Toujours afficher le résumé structuré
        print("📊 RÉSUMÉ DE DÉCISION:")
        print(f"  🚨 Suspicion: {analysis_result.suspicion_level.value}")
        print(f"  🎯 Action: {analysis_result.action_type}")
        print(f"  📈 Confiance: {analysis_result.confidence:.2f}")
        print(f"  💭 Raisonnement: {analysis_result.reasoning}")
        print(f"  📋 Recommandations: {', '.join(analysis_result.recommendations)}")
        print("="*80 + "\n")

    async def shutdown(self):
        """Arrêt propre."""
        logger.info("Arrêt du VLM dynamique...")
        self._unload_current_model()
        logger.info("VLM dynamique arrêté")
    
    # ==================== NOUVELLE IMPLÉMENTATION TOOL CALLING OFFICIELLE 2024 ====================
    
    async def analyze_with_native_tool_calling(
        self, 
        request: AnalysisRequest,
        enable_tools: bool = True
    ) -> AnalysisResponse:
        """
        🛠️ ANALYSE VLM AVEC TOOL CALLING NATIF 2025 - SANS FALLBACK
        
        Implémentation pure selon standards 2025:
        - Schémas JSON stricts
        - Parsing natif des tool calls
        - Exécution directe des outils
        - AUCUN fallback vers ancien système
        """
        # Validation stricte - échec si pas de support tool calling
        if not self.current_config or not self.current_config.supports_tool_calling:
            raise ModelError("❌ Tool Calling requis mais modèle ne le supporte pas")
        
        # Vérifier que le modèle est chargé
        if not self.is_loaded:
            success = await self.load_model(request.preferred_model or self.default_model)
            if not success:
                raise ModelError("❌ Aucun modèle VLM disponible")
        
        logger.info("🛠️ TOOL CALLING NATIF 2025 - Mode pur sans fallback")
        
        # Préparer l'image
        image = self._prepare_image(request.frame_data)
        
        # OUTILS OBLIGATOIRES avec schémas JSON 2025
        tools_schema = []
        if enable_tools:
            tools_schema = ToolSchemaBuilder.get_surveillance_tools_schema()
            logger.info(f"🔧 {len(tools_schema)} outils Tool Calling: {[t['function']['name'] for t in tools_schema]}")
        
        # Prompt optimisé pour Tool Calling 2025
        surveillance_prompt = """Tu es un expert en surveillance retail avec capacités Tool Calling avancées.

MISSION: Analyser cette scène pour détecter comportements suspects.

OUTILS DISPONIBLES (utilise-les intelligemment):
- sam2_segmentator: Segmentation précise d'objets
- dino_features: Features visuelles robustes  
- pose_estimator: Analyse postures/gestes
- trajectory_analyzer: Patterns de mouvement
- multimodal_fusion: Agrégation intelligente
- adversarial_detector: Détection manipulations

DIRECTIVE: Utilise les outils pertinents puis fournis ton analyse finale."""
        
        # Messages format Kimi-VL 2025
        if self.current_config.model_type == VLMModelType.KIMI_VL:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": surveillance_prompt}
                    ]
                }
            ]
            
            # Template avec tools intégrés selon documentation Kimi-VL 2025
            conversation_text = self.processor.apply_chat_template(
                messages, 
                add_generation_prompt=True,
                tokenize=False  # Obtenir le texte d'abord
            )
            
            inputs = self.processor(
                images=image,
                text=conversation_text,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.device)
            
            # Paramètres Tool Calling 2025 optimisés pour Kimi-VL
            gen_params = {
                "max_new_tokens": 1024,  # Plus de tokens pour tool calls
                "temperature": 0.7,  # Optimisé pour tool calling 2025
                "do_sample": True,
                "use_cache": False,  # Éviter DynamicCache errors
                "pad_token_id": self.processor.tokenizer.eos_token_id if hasattr(self.processor, 'tokenizer') else None,
                "return_dict_in_generate": False,
                "output_attentions": False,
                "output_hidden_states": False
            }
        
        # Génération avec Tool Calling natif
        with torch.no_grad():
            try:
                generated_ids = self.model.generate(**inputs, **gen_params)
                response_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                
                logger.debug(f"Réponse brute: {response_text[:200]}...")
                
                # PARSING TOOL CALLS 2025 (format Kimi natif)
                if "<|tool_calls_section_begin|>" in response_text:
                    logger.info("🔧 Tool calls détectés - Exécution native")
                    return await self._execute_native_tool_calls(response_text, image, request.context)
                else:
                    logger.info("📄 Réponse directe sans tool calls")
                    return self.response_parser.parse_vlm_response(response_text)
                    
            except Exception as e:
                logger.error(f"❌ Erreur Tool Calling natif: {e}")
                # PAS DE FALLBACK - Erreur pure
                raise ModelError(f"Tool Calling 2025 échoué: {e}")
    
    async def _execute_native_tool_calls(
        self, 
        response_text: str, 
        image, 
        context: dict
    ) -> AnalysisResponse:
        """Exécution native des tool calls sans fallback."""
        
        import re
        
        # Parser les tool calls format Kimi 2025
        tool_calls = []
        
        # Extraction des tool calls entre markers
        tool_section_match = re.search(
            r'<\|tool_calls_section_begin\|>(.*?)<\|tool_calls_section_end\|>', 
            response_text, 
            re.DOTALL
        )
        
        if not tool_section_match:
            logger.warning("⚠️ Markers tool calls trouvés mais pas de contenu")
            return self._create_error_response("Tool calls mal formatés")
        
        tool_section = tool_section_match.group(1)
        
        # Parser chaque tool call individuel
        individual_calls = re.findall(
            r'<\|tool_call_begin\|>(.*?)<\|tool_call_end\|>', 
            tool_section, 
            re.DOTALL
        )
        
        logger.info(f"🔧 {len(individual_calls)} tool calls à exécuter")
        
        # Exécuter chaque tool call
        executed_results = {}
        for call_text in individual_calls:
            try:
                # Parser JSON du tool call
                import json
                call_data = json.loads(call_text.strip())
                
                tool_name = call_data.get("name")
                tool_params = call_data.get("parameters", {})
                
                logger.info(f"🛠️ Exécution: {tool_name} avec {tool_params}")
                
                # Exécuter l'outil via le gestionnaire
                if hasattr(self, 'tools_manager') and self.tools_manager:
                    # Convertir image en numpy array si nécessaire
                    import numpy as np
                    if hasattr(image, 'convert'):
                        image_array = np.array(image.convert('RGB'))
                    else:
                        image_array = image
                    
                    # Préparer contexte avec paramètres
                    tool_context = {**context, **tool_params}
                    
                    # Exécuter l'outil spécifique
                    result = await self.tools_manager.execute_tools(
                        image_array, 
                        [tool_name], 
                        tool_context
                    )
                    
                    executed_results[tool_name] = result.get(tool_name)
                    logger.info(f"✅ {tool_name} exécuté: {result.get(tool_name, {}).get('success', False)}")
                
            except Exception as e:
                logger.error(f"❌ Erreur exécution tool call {call_text[:50]}: {e}")
                executed_results[f"error_{len(executed_results)}"] = {"error": str(e)}
        
        # Générer réponse finale basée sur les résultats des outils
        final_analysis = self._synthesize_tool_results(executed_results, response_text)
        
        return final_analysis
    
    def _synthesize_tool_results(self, tool_results: dict, original_response: str) -> AnalysisResponse:
        """Synthèse finale des résultats d'outils."""
        
        from ..types import SuspicionLevel, ActionType
        
        # Analyse des résultats d'outils
        tools_used = list(tool_results.keys())
        success_count = sum(1 for r in tool_results.values() if r and r.get('success', False))
        
        # Calcul suspicion basé sur convergence des outils
        suspicion_score = 0.5  # Base
        
        # Ajustements basés sur outils exécutés
        if 'sam2_segmentator' in tool_results:
            # Segmentation réussie augmente la confiance
            if tool_results['sam2_segmentator'].get('success'):
                suspicion_score += 0.1
        
        if 'pose_estimator' in tool_results:
            # Poses suspectes détectées
            pose_data = tool_results['pose_estimator'].get('data', {})
            behavior_score = pose_data.get('behavior_score', 0.0)
            suspicion_score += behavior_score * 0.3
        
        if 'adversarial_detector' in tool_results:
            # Détection d'attaque
            adv_data = tool_results['adversarial_detector'].get('data', {})
            if adv_data.get('is_adversarial', False):
                suspicion_score += 0.4
        
        # Normaliser
        suspicion_score = max(0.0, min(1.0, suspicion_score))
        
        # Déterminer niveau
        if suspicion_score < 0.3:
            suspicion_level = SuspicionLevel.LOW
            action_type = ActionType.NORMAL_SHOPPING
        elif suspicion_score < 0.6:
            suspicion_level = SuspicionLevel.MEDIUM
            action_type = ActionType.SUSPICIOUS_MOVEMENT
        elif suspicion_score < 0.8:
            suspicion_level = SuspicionLevel.HIGH
            action_type = ActionType.ITEM_CONCEALMENT
        else:
            suspicion_level = SuspicionLevel.CRITICAL
            action_type = ActionType.POTENTIAL_THEFT
        
        return AnalysisResponse(
            suspicion_level=suspicion_level,
            action_type=action_type,
            confidence=0.85,
            description=f"Analyse Tool Calling 2025: {success_count}/{len(tools_used)} outils exécutés avec succès",
            tools_used=tools_used,
            recommendations=[f"Outils utilisés: {', '.join(tools_used)}"]
        )