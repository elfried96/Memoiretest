#!/usr/bin/env python3
"""
🎯 MAIN HEADLESS - Pipeline Sans Interface Graphique
===================================================

Version adaptée du main.py pour environnements sans display (serveurs, SSH, etc.)
Génère des logs détaillés et sauvegarde optionnelle des frames avec détections.
"""

import asyncio
import cv2
import time
import base64
import json
import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from io import BytesIO
import numpy as np

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Imports du système de surveillance
from src.core.vlm.dynamic_model import DynamicVisionLanguageModel
from src.core.orchestrator.vlm_orchestrator import (
    ModernVLMOrchestrator, 
    OrchestrationConfig, 
    OrchestrationMode
)
from src.core.types import DetectedObject, BoundingBox
from src.detection.yolo_detector import YOLODetector
from src.detection.tracking.byte_tracker import BYTETracker
from src.core.vlm.memory_system import vlm_memory


class AlertLevel(Enum):
    """Niveaux d'alerte du système."""
    NORMAL = "normal"
    ATTENTION = "attention" 
    ALERTE = "alerte"
    CRITIQUE = "critique"


@dataclass
class SurveillanceResult:
    """Résultat de surveillance pour export JSON."""
    frame_id: int
    timestamp: float
    detections_count: int
    persons_detected: int
    alert_level: str
    vlm_analysis: Optional[Dict] = None
    actions_taken: List[str] = None
    processing_time: float = 0.0
    cumulative_summary: Optional[Dict] = None  # Résumé cumulatif toutes les 30s


class HeadlessSurveillanceSystem:
    """Système de surveillance pour environnement sans interface graphique."""
    
    def __init__(
        self,
        video_source: str = 0,
        vlm_model: str = "kimi-vl-a3b-thinking",  # Kimi-VL par défaut
        orchestration_mode: OrchestrationMode = OrchestrationMode.BALANCED,
        save_results: bool = True,
        save_frames: bool = False,
        frame_skip: int = 1,
        vlm_analysis_mode: str = "continuous"
    ):
        self.video_source = video_source
        self.vlm_model = vlm_model  # Utilise le modèle spécifié
        self.orchestration_mode = orchestration_mode
        self.save_results = save_results
        self.save_frames = save_frames
        self.frame_skip = frame_skip
        self.vlm_analysis_mode = vlm_analysis_mode
        
        # Dossiers de sortie
        self.output_dir = Path("surveillance_output")
        self.output_dir.mkdir(exist_ok=True)
        
        if save_frames:
            self.frames_dir = self.output_dir / "frames"
            self.frames_dir.mkdir(exist_ok=True)
        
        # === COMPOSANTS PRINCIPAUX ===
        
        # 1. Détection et tracking
        self.yolo_detector = YOLODetector(
            model_path="yolov11n.pt",
            device="auto"
        )
        self.tracker = BYTETracker()
        
        # 2. VLM et orchestration 
        self.vlm = DynamicVisionLanguageModel(
            default_model=vlm_model,
            enable_fallback=True
        )
        
        config = OrchestrationConfig(
            mode=orchestration_mode,
            enable_advanced_tools=True,
            max_concurrent_tools=4,
            confidence_threshold=0.7
        )
        
        self.orchestrator = ModernVLMOrchestrator(
            vlm_model_name=vlm_model,
            config=config
        )
        self.vlm_enabled = True
        logger.info(f"✅ VLM activé: {vlm_model}")
        
        # === ÉTAT DU SYSTÈME ===
        self.frame_count = 0
        self.results_log = []
        self.processing_stats = {
            "total_frames": 0,
            "detected_objects": 0,
            "persons_detected": 0,
            "vlm_analyses": 0,
            "vlm_triggered": 0,  # Nouveau: combien de fois le VLM a été déclenché
            "alerts_triggered": 0,
            "total_processing_time": 0.0,
            "average_fps": 0.0
        }
        
        # === SYSTÈME DE DÉCLENCHEMENT INTELLIGENT ===
        self.last_vlm_trigger_time = 0
        self.vlm_cooldown_seconds = 2 if vlm_analysis_mode == "continuous" else 5
        self.person_count_history = []  # Historique du nombre de personnes
        self.alert_history = []  # Historique des alertes
        
        # === DESCRIPTIONS CUMULATIVES ===
        self.cumulative_descriptions = []  # Historique des descriptions
        self.last_summary_time = 0
        self.summary_interval_seconds = 30  # Résumé toutes les 30s
        self.video_start_time = time.time()
        self.current_period_descriptions = []  # Descriptions de la période courante
        
        logger.info(f"🎯 Système headless initialisé - VLM: {vlm_model}, Mode: {orchestration_mode.value}")
        logger.info(f"📁 Résultats sauvés dans: {self.output_dir}")
    
    async def initialize(self):
        """Initialisation asynchrone des composants."""
        logger.info("🚀 Initialisation du système headless...")
        
        if self.vlm_enabled:
            # Chargement du VLM
            logger.info(f"⏳ Chargement {self.vlm_model}...")
            vlm_loaded = await self.vlm.load_model()
            
            if vlm_loaded:
                logger.info(f"✅ {self.vlm_model} chargé avec succès")
            else:
                logger.warning("⚠️ VLM principal échec, fallback activé")
        
        # Test des composants
        logger.info("🔍 Tests des composants...")
        
        # Test YOLO
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        test_detections = self.yolo_detector.detect(test_frame)
        logger.info(f"✅ YOLO opérationnel - {len(test_detections)} détections test")
        
        if self.vlm_enabled:
            # Statut complet
            status = await self.orchestrator.health_check()
            logger.info(f"📊 Health check: {status}")
        
        logger.info("🎯 Système prêt pour surveillance headless!")
    
    def create_detections_list(self, yolo_results) -> List[DetectedObject]:
        """Convertit les résultats YOLO en liste de Detection."""
        detections = []
        
        for result in yolo_results:
            if hasattr(result, 'boxes') and result.boxes is not None:
                for box in result.boxes:
                    # Extraction des coordonnées
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    cls = int(box.cls[0].cpu().numpy())
                    
                    # Nom de classe
                    class_names = result.names
                    class_name = class_names.get(cls, f"class_{cls}")
                    
                    # Création de la détection
                    detection = DetectedObject(
                        class_id=cls,
                        class_name=class_name,
                        bbox=BoundingBox(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2)),
                        confidence=float(conf),
                        track_id=None
                    )
                    detections.append(detection)
        
        return detections
    
    def _should_trigger_vlm_analysis(self, detections: List[DetectedObject], persons_count: int, context: Dict[str, Any]) -> bool:
        """
        Détermine si le VLM doit être déclenché.
        Mode 'continuous': Analyse très fréquente pour descriptions complètes
        Mode 'smart': Analyse intelligente économique
        """
        current_time = time.time()
        
        # MODE CONTINUOUS - Analyse beaucoup plus fréquente
        if self.vlm_analysis_mode == "continuous":
            # 1. Vérifier le cooldown court
            if current_time - self.last_vlm_trigger_time < self.vlm_cooldown_seconds:
                return False
            
            # 2. TOUJOURS analyser s'il y a des personnes
            if persons_count > 0:
                logger.info(f"📄 VLM CONTINU déclenché: {persons_count} personne(s) présente(s)")
                return True
            
            # 3. TOUJOURS analyser s'il y a des objets suspects
            suspicious_objects = ["backpack", "handbag", "suitcase", "umbrella", "sports ball", "bag", "bottle", "cup"]
            for detection in detections:
                if detection.class_name in suspicious_objects:
                    logger.info(f"📄 VLM CONTINU déclenché: objet '{detection.class_name}' détecté")
                    return True
            
            # 4. Analyse périodique même sans personnes (pour capture environnement)
            if current_time - self.last_vlm_trigger_time > 10:  # Toutes les 10s minimum
                logger.info("📄 VLM CONTINU déclenché: analyse périodique environnement")
                return True
            
            return False
        
        # MODE SMART - Analyse intelligente économique (ancien comportement)
        else:
            # 1. Vérifier le cooldown
            if current_time - self.last_vlm_trigger_time < self.vlm_cooldown_seconds:
                return False
            
            # 2. Toujours déclencher si beaucoup de personnes
            if persons_count >= 3:
                logger.info(f"🚨 VLM déclenché: {persons_count} personnes détectées")
                return True
            
            # 3. Maintenir historique du nombre de personnes
            self.person_count_history.append(persons_count)
            if len(self.person_count_history) > 10:
                self.person_count_history.pop(0)
            
            # 4. Détecter changement soudain de population
            if len(self.person_count_history) >= 5:
                recent_avg = sum(self.person_count_history[-5:]) / 5
                if persons_count > recent_avg + 1:
                    logger.info(f"📈 VLM déclenché: augmentation population {recent_avg:.1f} → {persons_count}")
                    return True
            
            # 5. Objets suspects détectés
            suspicious_objects = ["backpack", "handbag", "suitcase", "umbrella", "sports ball", "bag"]
            for detection in detections:
                if detection.class_name in suspicious_objects:
                    logger.info(f"👜 VLM déclenché: objet suspect '{detection.class_name}' détecté")
                    return True
            
            # 6. Personne seule qui reste longtemps
            if persons_count == 1 and len(self.person_count_history) >= 8:
                if all(count == 1 for count in self.person_count_history[-8:]):
                    logger.info("🕐 VLM déclenché: personne seule depuis longtemps")
                    return True
            
            # 7. Déclenchement périodique
            if not context.get("test_mode", False):
                if (current_time - self.last_vlm_trigger_time > 60 and persons_count > 0):
                    logger.info("⏰ VLM déclenché: contrôle périodique de sécurité")
                    return True
            
            return False
    
    def encode_frame_to_base64(self, frame: np.ndarray) -> str:
        """Encode un frame OpenCV en base64 pour le VLM."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        _, buffer = cv2.imencode('.jpg', frame_rgb, [cv2.IMWRITE_JPEG_QUALITY, 85])
        frame_b64 = base64.b64encode(buffer).decode('utf-8')
        return frame_b64
    
    def should_generate_cumulative_summary(self) -> bool:
        """Vérifie si il faut générer un résumé cumulatif."""
        current_time = time.time()
        elapsed_since_last_summary = current_time - self.last_summary_time
        return elapsed_since_last_summary >= self.summary_interval_seconds
    
    async def generate_cumulative_summary(self, frame_b64: str, current_context: Dict[str, Any]) -> Optional[Dict]:
        """Génère un résumé cumulatif des 30 dernières secondes."""
        try:
            current_time = time.time()
            elapsed_total = current_time - self.video_start_time
            period_number = int(elapsed_total // self.summary_interval_seconds) + 1
            
            # Construire le contexte cumulatif
            cumulative_context = current_context.copy()
            cumulative_context.update({
                "summary_type": "cumulative_video_analysis",
                "period_number": period_number,
                "total_elapsed_seconds": elapsed_total,
                "period_start": f"{(period_number-1) * 30}s",
                "period_end": f"{period_number * 30}s",
                "previous_descriptions": self.cumulative_descriptions[-5:] if self.cumulative_descriptions else [],
                "current_period_descriptions": self.current_period_descriptions
            })
            
            # Prompt spécial pour description cumulative
            cumulative_prompt = f"""
            ANALYSE CUMULATIVE KIMI-VL - PÉRIODE {period_number} [HEADLESS]
            ============================================================
            
            📊 CONTEXTE:
            - Période: {cumulative_context['period_start']} à {cumulative_context['period_end']}
            - Temps total écoulé: {elapsed_total:.0f} secondes
            - Frame actuel: {current_context.get('frame_id', 0)}
            - Mode: Headless (sans interface graphique)
            
            📋 DESCRIPTIONS PRÉCÉDENTES:
            {chr(10).join([f"- Période {i+1}: {desc}" for i, desc in enumerate(self.cumulative_descriptions[-3:])]) if self.cumulative_descriptions else "Aucune description précédente"}
            
            🔍 OBSERVATIONS ACTUELLES ({len(self.current_period_descriptions)} descriptions):
            {chr(10).join([f"- {desc}" for desc in self.current_period_descriptions[-10:]]) if self.current_period_descriptions else "Aucune observation dans cette période"}
            
            🎥 TÂCHE KIMI-VL:
            Analysez l'image et générez une DESCRIPTION CUMULATIVE de cette période de 30s.
            
            Votre réponse doit inclure:
            1. 📝 RÉSUMÉ PÉRIODE: Activités principales de ces 30 secondes
            2. 👥 PERSONNES: Nombre, actions, comportements observés
            3. 🎨 ÉVÉNEMENTS: Actions significatives et changements
            4. 📋 CONTINUITÉ: Lien avec les périodes précédentes
            5. ⚠️ POINTS NOTABLES: Éléments d'intérêt ou alertes
            
            Soyez DÉTAILLÉ, PRÉCIS et FACTUEL pour une surveillance efficace.
            """
            
            # Utiliser l'orchestrateur avec prompt personnalisé
            if hasattr(self.orchestrator.vlm, 'analyze_with_custom_prompt'):
                cumulative_analysis = await self.orchestrator.vlm.analyze_with_custom_prompt(
                    frame_data=frame_b64,
                    custom_prompt=cumulative_prompt,
                    context=cumulative_context
                )
            else:
                # Fallback: utiliser l'analyse normale
                cumulative_analysis = await self.orchestrator.analyze_surveillance_frame(
                    frame_data=frame_b64,
                    detections=[],
                    context=cumulative_context
                )
            
            if cumulative_analysis:
                # Extraire la description
                if hasattr(cumulative_analysis, 'description'):
                    description_text = cumulative_analysis.description
                elif isinstance(cumulative_analysis, dict):
                    description_text = cumulative_analysis.get('description', '')
                else:
                    description_text = str(cumulative_analysis)
                
                summary_data = {
                    "period_number": period_number,
                    "period_range": f"{cumulative_context['period_start']}-{cumulative_context['period_end']}",
                    "timestamp": current_time,
                    "frame_id": current_context.get('frame_id', 0),
                    "description": description_text,
                    "total_elapsed": elapsed_total,
                    "analysis_details": cumulative_analysis
                }
                
                # Ajouter à l'historique cumulatif
                self.cumulative_descriptions.append(description_text)
                
                # Réinitialiser pour la prochaine période
                self.current_period_descriptions.clear()
                self.last_summary_time = current_time
                
                logger.info(f"📋 [KIMI-VL HEADLESS] Résumé Période {period_number} généré ({elapsed_total:.0f}s total)")
                logger.info(f"📝 Description: {description_text[:200]}...") 
                
                return summary_data
            
        except Exception as e:
            logger.error(f"❌ Erreur génération résumé cumulatif KIMI-VL: {e}")
        
        return None
    
    def save_frame_with_detections(self, frame: np.ndarray, detections: List[DetectedObject], frame_id: int):
        """Sauvegarde un frame avec les détections dessinées."""
        if not self.save_frames:
            return
            
        overlay_frame = frame.copy()
        
        # Dessiner les détections
        for detection in detections:
            bbox = detection.bbox
            color = (0, 255, 0) if detection.class_name == "person" else (255, 0, 0)
            
            # Rectangle
            cv2.rectangle(
                overlay_frame,
                (int(bbox.x1), int(bbox.y1)),
                (int(bbox.x2), int(bbox.y2)),
                color, 2
            )
            
            # Label
            label = f"{detection.class_name}: {detection.confidence:.2f}"
            cv2.putText(
                overlay_frame, label,
                (int(bbox.x1), int(bbox.y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )
        
        # Sauvegarder
        frame_path = self.frames_dir / f"frame_{frame_id:06d}.jpg"
        cv2.imwrite(str(frame_path), overlay_frame)
        logger.debug(f"🖼️ Frame sauvé: {frame_path}")
    
    async def process_frame(self, frame: np.ndarray) -> SurveillanceResult:
        """Traite un frame complet en mode headless."""
        
        start_time = time.time()
        self.frame_count += 1
        
        # === ÉTAPE 1: DÉTECTION YOLO ===
        yolo_results = self.yolo_detector.detect(frame)
        detections = self.create_detections_list(yolo_results)
        
        # === ÉTAPE 1.5: TRACKING ===
        tracked_objects = self.tracker.update(detections)
        
        persons_count = len([d for d in detections if d.class_name == "person"])
        
        self.processing_stats["detected_objects"] += len(detections)
        self.processing_stats["persons_detected"] += persons_count
        
        # === ÉTAPE 2: ANALYSE VLM (conditionnelle) ===
        vlm_analysis = None
        alert_level = AlertLevel.NORMAL
        actions_taken = []
        
        # Vérifier que le VLM est chargé avant de continuer
        vlm_ready = (
            self.vlm_enabled and 
            hasattr(self.vlm, 'model') and 
            self.vlm.model is not None
        )
        
        if vlm_ready:
            logger.debug(f"🧠 Frame {self.frame_count} - Analyse VLM...")
            
            # Encodage pour VLM
            frame_b64 = self.encode_frame_to_base64(frame)
            
            # Contexte enrichi avec mémoire historique
            memory_context = vlm_memory.get_context_for_vlm()
            
            context = {
                "frame_id": self.frame_count,
                "timestamp": time.time(),
                "location": "Store Main Area",
                "camera": "CAM_01",
                "person_count": persons_count,
                "total_objects": len(detections),
                "time_of_day": time.strftime("%H:%M:%S"),
                **memory_context  # Ajouter le contexte de mémoire
            }
            
            # Vérifier si il faut générer un résumé cumulatif (toutes les 30s)
            cumulative_summary = None
            if self.should_generate_cumulative_summary():
                cumulative_summary = await self.generate_cumulative_summary(frame_b64, context)
            
            # DÉCLENCHEMENT INTELLIGENT DU VLM
            should_trigger_vlm = self._should_trigger_vlm_analysis(detections, persons_count, context)
            
            try:
                if should_trigger_vlm:
                    # Marquer le déclenchement
                    self.last_vlm_trigger_time = time.time()
                    self.processing_stats["vlm_triggered"] += 1
                    
                    # Analyse orchestrée seulement si conditions remplies
                    vlm_analysis = await self.orchestrator.analyze_surveillance_frame(
                        frame_data=frame_b64,
                        detections=detections,
                        context=context
                    )
                    
                    # Ajouter cette analyse à la période courante pour le cumul
                    if vlm_analysis and hasattr(vlm_analysis, 'description'):
                        self.current_period_descriptions.append(vlm_analysis.description)
                    
                    self.processing_stats["vlm_analyses"] += 1
                else:
                    # Pas de déclenchement VLM - analyse légère seulement
                    vlm_analysis = None
                
                # Décisions basées sur l'analyse
                if vlm_analysis is not None:
                    if vlm_analysis.suspicion_level.value in ["HIGH", "CRITICAL"]:
                        alert_level = AlertLevel.ALERTE
                        actions_taken = ["alert_triggered", "recording_started"]
                        self.processing_stats["alerts_triggered"] += 1
                    elif vlm_analysis.suspicion_level.value == "MEDIUM":
                        alert_level = AlertLevel.ATTENTION
                        actions_taken = ["increased_monitoring"]
                else:
                    # Analyse de base sans VLM (surveillance continue rapide)
                    if persons_count >= 3:
                        alert_level = AlertLevel.ATTENTION
                        actions_taken = ["high_occupancy_basic"]
                    elif len(detections) > 5:  # Beaucoup d'objets détectés
                        alert_level = AlertLevel.ATTENTION
                        actions_taken = ["many_objects_detected"]
                
            except Exception as e:
                logger.error(f"❌ Erreur analyse VLM frame {self.frame_count}: {e}")
        
        # Analyse simple si pas de VLM
        if not self.vlm_enabled:
            if persons_count > 3:
                alert_level = AlertLevel.ALERTE
                actions_taken = ["high_occupancy_detected"]
                self.processing_stats["alerts_triggered"] += 1
            elif persons_count > 1:
                alert_level = AlertLevel.ATTENTION
                actions_taken = ["multiple_persons_detected"]
        
        # === SAUVEGARDE FRAME SI DEMANDÉE ===
        if len(detections) > 0 or alert_level != AlertLevel.NORMAL:
            self.save_frame_with_detections(frame, detections, self.frame_count)
        
        # === CRÉATION DU RÉSULTAT ===
        processing_time = time.time() - start_time
        
        # Sérialisation sécurisée de vlm_analysis
        vlm_analysis_dict = None
        if vlm_analysis:
            try:
                if hasattr(vlm_analysis, 'to_dict'):
                    vlm_analysis_dict = vlm_analysis.to_dict()
                elif hasattr(vlm_analysis, '__dict__'):
                    vlm_analysis_dict = vlm_analysis.__dict__.copy()
                    # Convertir les enums en strings
                    for key, value in vlm_analysis_dict.items():
                        if hasattr(value, 'value'):  # C'est un enum
                            vlm_analysis_dict[key] = value.value
            except Exception:
                vlm_analysis_dict = None
        
        result = SurveillanceResult(
            frame_id=self.frame_count,
            timestamp=time.time(),
            detections_count=len(detections),
            persons_detected=persons_count,
            alert_level=alert_level.value,
            vlm_analysis=vlm_analysis_dict,
            actions_taken=actions_taken,
            processing_time=processing_time,
            cumulative_summary=cumulative_summary
        )
        
        # === MISE À JOUR DES STATISTIQUES ===
        self.processing_stats["total_frames"] += 1
        self.processing_stats["total_processing_time"] += processing_time
        
        if self.processing_stats["total_frames"] > 0:
            avg_time = self.processing_stats["total_processing_time"] / self.processing_stats["total_frames"]
            self.processing_stats["average_fps"] = 1.0 / avg_time if avg_time > 0 else 0
        
        # === ENREGISTREMENT EN MÉMOIRE ===
        vlm_triggered_for_memory = (vlm_analysis is not None)
        vlm_memory.add_frame(
            frame_id=self.frame_count,
            detections=detections,
            vlm_triggered=vlm_triggered_for_memory,
            vlm_analysis=vlm_analysis,
            alert_level=alert_level.value,
            actions_taken=actions_taken
        )
        
        # === LOGGING DÉTAILLÉ ===
        if persons_count > 0 or alert_level != AlertLevel.NORMAL:
            # Ajouter les stats de mémoire aux logs importants
            memory_stats = vlm_memory.get_memory_stats()
            
            logger.info(f"📊 Frame {self.frame_count}: "
                       f"{len(detections)} objs, {persons_count} personnes, "
                       f"Alert: {alert_level.value}, "
                       f"Actions: {actions_taken}, "
                       f"Temps: {processing_time:.2f}s, "
                       f"Mémoire: {memory_stats['current_frames_in_memory']} frames, "
                       f"{memory_stats['active_persons']} personnes actives")
        elif self.frame_count % 60 == 0:
            # Calcul du taux de déclenchement intelligent + stats mémoire
            trigger_rate = (self.processing_stats['vlm_triggered'] / self.processing_stats['total_frames'] * 100) if self.processing_stats['total_frames'] > 0 else 0
            memory_stats = vlm_memory.get_memory_stats()
            
            logger.info(f"📈 Frame {self.frame_count}: "
                       f"FPS: {self.processing_stats['average_fps']:.1f}, "
                       f"Total objets: {self.processing_stats['detected_objects']}, "
                       f"VLM déclenché: {self.processing_stats['vlm_triggered']}/{self.processing_stats['total_frames']} ({trigger_rate:.1f}%), "
                       f"Analyses VLM: {self.processing_stats['vlm_analyses']}, "
                       f"Patterns détectés: {memory_stats['patterns_detected']}")
        
        return result
    
    async def run_surveillance(self, max_frames: int = None):
        """Lance la surveillance en mode headless."""
        logger.info("🎬 Démarrage surveillance headless...")
        
        # Ouverture de la source vidéo avec optimisations
        cap = cv2.VideoCapture(self.video_source)
        
        if not cap.isOpened():
            logger.error(f"❌ Impossible d'ouvrir source vidéo: {self.video_source}")
            return
        
        # 🎯 OPTIMISATIONS EXTRACTION FRAMES
        # Backend optimisé selon la plateforme
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Réduire buffer pour temps réel
        
        # Qualité maximale (sans compression)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))  # MJPEG pour moins de compression
        
        # Informations vidéo
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"📹 Source vidéo: {self.video_source}")
        logger.info(f"📏 Résolution: {width}x{height} | FPS: {fps:.2f} | Frames: {total_frames}")
        
        try:
            frame_processed = 0
            frame_read_count = 0  # Compteur pour frame_skip
            
            while True:
                # Lecture optimisée du frame
                ret, frame = cap.read()
                frame_read_count += 1
                
                if not ret:
                    logger.warning("📹 Fin de vidéo ou erreur lecture")
                    break
                
                # Application du frame_skip
                if frame_read_count % self.frame_skip != 0:
                    continue  # Skip ce frame
                
                # 🎯 VALIDATION ET AMÉLIORATION DE LA QUALITÉ DU FRAME
                if frame is None or frame.size == 0:
                    logger.warning("⚠️ Frame vide détecté")
                    continue
                
                # Amélioration de la qualité (optionnel)
                # frame = cv2.convertScaleAbs(frame, alpha=1.1, beta=10)  # Augmenter contraste/luminosité
                
                # Débruitage léger si nécessaire
                # frame = cv2.fastNlMeansDenoisingColored(frame, None, 3, 3, 7, 21)
                
                # Vérification des dimensions
                if frame.shape[0] < 100 or frame.shape[1] < 100:
                    logger.warning(f"⚠️ Frame trop petit: {frame.shape}")
                    continue
                
                # Traitement complet du frame
                result = await self.process_frame(frame)
                self.results_log.append(result)
                
                frame_processed += 1
                
                # Limite de frames si spécifiée
                if max_frames and frame_processed >= max_frames:
                    logger.info(f"🏁 Limite de {max_frames} frames atteinte")
                    break
        
        except KeyboardInterrupt:
            logger.info("🛑 Arrêt par interruption clavier")
        
        finally:
            # Nettoyage
            cap.release()
            
            # Sauvegarde des résultats
            if self.save_results:
                self.save_results_to_json()
            
            # Statistiques finales
            self.print_final_statistics()
    
    def _serialize_result(self, result):
        """Sérialise un résultat en évitant les erreurs JSON avec enums et datetime."""
        import datetime
        
        def serialize_value(value):
            if hasattr(value, 'value'):  # C'est un enum
                return value.value
            elif isinstance(value, datetime.datetime):
                return value.isoformat()
            elif isinstance(value, dict):
                return {k: serialize_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [serialize_value(item) for item in value]
            else:
                return value
        
        try:
            result_dict = asdict(result)
            # Convertir récursivement tous les types problématiques
            return {key: serialize_value(value) for key, value in result_dict.items()}
        except Exception as e:
            # Fallback: conversion manuelle
            return {
                "frame_id": getattr(result, 'frame_id', 0),
                "timestamp": getattr(result, 'timestamp', 0),
                "detections_count": getattr(result, 'detections_count', 0),
                "persons_detected": getattr(result, 'persons_detected', 0),
                "alert_level": getattr(result, 'alert_level', 'normal'),
                "actions_taken": getattr(result, 'actions_taken', []),
                "processing_time": getattr(result, 'processing_time', 0),
                "vlm_analysis": None
            }
    
    def save_results_to_json(self):
        """Sauvegarde les résultats en JSON."""
        results_file = self.output_dir / f"surveillance_results_{int(time.time())}.json"
        
        output_data = {
            "metadata": {
                "video_source": str(self.video_source),
                "vlm_model": self.vlm_model,
                "model_type": "kimi-vl-headless",
                "orchestration_mode": self.orchestration_mode.value,
                "vlm_analysis_mode": self.vlm_analysis_mode,
                "summary_interval_seconds": self.summary_interval_seconds,
                "total_frames": len(self.results_log),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            "statistics": self.processing_stats,
            "cumulative_video_summary": {
                "total_periods": len(self.cumulative_descriptions),
                "periods_descriptions": self.cumulative_descriptions,
                "interval_seconds": self.summary_interval_seconds
            },
            "results": [self._serialize_result(result) for result in self.results_log]
        }
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Résultats sauvés: {results_file}")
    
    def print_final_statistics(self):
        """Affiche les statistiques finales."""
        logger.info("=" * 60)
        logger.info("📈 STATISTIQUES FINALES DE SURVEILLANCE")
        logger.info("=" * 60)
        
        # Statistiques générales
        for key, value in self.processing_stats.items():
            if isinstance(value, float):
                logger.info(f"  • {key}: {value:.2f}")
            else:
                logger.info(f"  • {key}: {value}")
        
        # 🚀 STATISTIQUES D'EFFICACITÉ DU DÉCLENCHEMENT INTELLIGENT
        logger.info("")
        logger.info("🧠 EFFICACITÉ DU SYSTÈME DE DÉCLENCHEMENT INTELLIGENT:")
        logger.info("-" * 60)
        
        total_frames = self.processing_stats.get("total_frames", 1)
        vlm_triggered = self.processing_stats.get("vlm_triggered", 0)
        vlm_analyses = self.processing_stats.get("vlm_analyses", 0)
        
        # Calcul du taux de déclenchement intelligent
        trigger_rate = (vlm_triggered / total_frames * 100) if total_frames > 0 else 0
        success_rate = (vlm_analyses / vlm_triggered * 100) if vlm_triggered > 0 else 0
        frames_saved = total_frames - vlm_triggered
        
        logger.info(f"  🎯 Frames total traités: {total_frames}")
        logger.info(f"  ⚡ VLM déclenché seulement: {vlm_triggered} fois ({trigger_rate:.1f}%)")
        logger.info(f"  ✅ Analyses VLM réussies: {vlm_analyses} ({success_rate:.1f}%)")
        logger.info(f"  🚀 Frames économisés (traitement rapide): {frames_saved}")
        logger.info(f"  💡 Économie de traitement: {(100 - trigger_rate):.1f}%")
        
        # Performance comparative
        if trigger_rate < 50:
            logger.info("  ⭐ EXCELLENT: Système de déclenchement très efficace!")
        elif trigger_rate < 80:
            logger.info("  ✅ BON: Système de déclenchement efficace")
        else:
            logger.info("  ⚠️ À AMÉLIORER: Déclenchements fréquents détectés")
        
        # 🧠 STATISTIQUES DU SYSTÈME DE MÉMOIRE
        logger.info("")
        logger.info("🧠 SYSTÈME DE MÉMOIRE CONTEXTUELLE:")
        logger.info("-" * 60)
        
        memory_stats = vlm_memory.get_memory_stats()
        logger.info(f"  💾 Frames en mémoire: {memory_stats['current_frames_in_memory']}")
        logger.info(f"  👥 Personnes actives trackées: {memory_stats['active_persons']}")
        logger.info(f"  🔍 Patterns suspects détectés: {memory_stats['patterns_detected']}")
        logger.info(f"  📊 Analyses VLM stockées: {memory_stats['vlm_analyses_stored']}")
        logger.info(f"  🧮 Taille mémoire: {memory_stats['memory_size_mb']:.2f} MB")
        
        if memory_stats['patterns_detected'] > 0:
            logger.info("  ⚠️ Des patterns suspects ont été détectés durant la session")
        else:
            logger.info("  ✅ Aucun pattern suspect majeur détecté")
        
        # Analyse des alertes
        alerts = [r for r in self.results_log if r.alert_level != "normal"]
        if alerts:
            logger.info(f"🚨 ALERTES DÉTECTÉES: {len(alerts)}")
            for alert in alerts[-5:]:  # 5 dernières alertes
                logger.info(f"   Frame {alert.frame_id}: {alert.alert_level} - {alert.actions_taken}")
        
        # Statistiques descriptions cumulatives
        logger.info("")
        logger.info("📋 DESCRIPTIONS CUMULATIVES KIMI-VL:")
        logger.info("-" * 60)
        logger.info(f"  📄 Mode VLM: {self.vlm_analysis_mode}")
        logger.info(f"  🕒 Intervalle résumés: {self.summary_interval_seconds}s")
        logger.info(f"  📝 Résumés générés: {len(self.cumulative_descriptions)}")
        logger.info(f"  📊 Descriptions courantes: {len(self.current_period_descriptions)}")
        
        # Afficher les résumés cumulatifs
        if self.cumulative_descriptions:
            logger.info("")
            logger.info("📜 RÉSUMÉS DE LA VIDÉO KIMI-VL:")
            logger.info("=" * 60)
            for i, description in enumerate(self.cumulative_descriptions):
                period_start = i * self.summary_interval_seconds
                period_end = (i + 1) * self.summary_interval_seconds
                logger.info(f"  🕰️ Période {i+1} ({period_start}s-{period_end}s):")
                logger.info(f"    {description[:300]}{'...' if len(description) > 300 else ''}")
                logger.info("")
        
        # Analyse des personnes
        frames_with_persons = [r for r in self.results_log if r.persons_detected > 0]
        if frames_with_persons:
            max_persons = max(r.persons_detected for r in frames_with_persons)
            logger.info(f"👥 Max personnes simultanées: {max_persons}")
            logger.info(f"👥 Frames avec personnes: {len(frames_with_persons)}/{len(self.results_log)}")


def parse_arguments():
    """Parse les arguments de la ligne de commande."""
    parser = argparse.ArgumentParser(
        description="🎯 Système de Surveillance Headless",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--video", "-v", default="webcam",
                       help="Source vidéo")
    parser.add_argument("--model", "-m", default="kimi-vl-a3b-thinking",
                       help="Modèle VLM")
    parser.add_argument("--mode", default="BALANCED",
                       choices=["FAST", "BALANCED", "THOROUGH"],
                       help="Mode orchestration")
    parser.add_argument("--max-frames", type=int, default=None,
                       help="Nombre max de frames")
    parser.add_argument("--save-frames", action="store_true",
                       help="Sauvegarder frames avec détections")
    parser.add_argument("--no-save", action="store_true",
                       help="Ne pas sauvegarder les résultats")
    parser.add_argument("--frame-skip", type=int, default=1,
                       help="Traiter 1 frame sur N (ex: 2 = une frame sur deux)")
    parser.add_argument("--vlm-mode", default="continuous",
                       choices=["continuous", "smart"],
                       help="Mode analyse VLM: continuous (fréquent) ou smart (économique)")
    parser.add_argument("--summary-interval", type=int, default=30,
                       help="Intervalle en secondes pour résumés cumulatifs")
    
    return parser.parse_args()


async def main():
    """Point d'entrée principal."""
    args = parse_arguments()
    
    print(f"""
🎯 SYSTÈME DE SURVEILLANCE HEADLESS
====================================

Configuration:
📹 Source vidéo  : {args.video}
🤖 Modèle VLM    : {args.model} (KIMI-VL par défaut)
⚙️ Mode          : {args.mode}
💾 Sauvegarde    : {'Activée' if not args.no_save else 'Désactivée'}
🖼️ Frames        : {'Sauvées' if args.save_frames else 'Non sauvées'}
📊 Max frames    : {args.max_frames or 'Illimité'}
🔢 Frame skip    : {args.frame_skip} (traite 1 frame sur {args.frame_skip})
📄 Mode VLM      : {args.vlm_mode} ({'Analyse continue' if args.vlm_mode == 'continuous' else 'Analyse économique'})
📋 Résumés       : Toutes les {args.summary_interval}s

WORKFLOW HEADLESS:
1. 📹 Capture vidéo → logs détaillés
2. 🔍 Détection YOLO → comptage objets  
3. 🧠 Analyse VLM → évaluation suspicion
4. 🚨 Prise décisions → actions automatisées
5. 💾 Export JSON → résultats structurés
6. 📊 Statistiques finales → rapport complet
""")
    
    # Configuration du mode
    mode_mapping = {
        "FAST": OrchestrationMode.FAST,
        "BALANCED": OrchestrationMode.BALANCED,
        "THOROUGH": OrchestrationMode.THOROUGH
    }
    
    # Validation source vidéo
    video_source = args.video
    if video_source.lower() == "webcam":
        video_source = 0
    
    # Initialisation système
    system = HeadlessSurveillanceSystem(
        video_source=video_source,
        vlm_model=args.model,
        orchestration_mode=mode_mapping[args.mode],
        save_results=not args.no_save,
        save_frames=args.save_frames,
        frame_skip=args.frame_skip,
        vlm_analysis_mode=args.vlm_mode
    )
    
    # Configurer l'intervalle de résumé
    system.summary_interval_seconds = args.summary_interval
    
    try:
        # Initialisation et démarrage
        await system.initialize()
        await system.run_surveillance(max_frames=args.max_frames)
        
    except Exception as e:
        logger.error(f"❌ Erreur système: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())