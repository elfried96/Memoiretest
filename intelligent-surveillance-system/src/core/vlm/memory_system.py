"""
🧠 Système de mémoire contextuelle pour le VLM
==============================================

Permet au VLM de se souvenir des analyses précédentes et de prendre
des décisions plus intelligentes basées sur l'historique.
"""

import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import deque
import json
from loguru import logger

from ..types import AnalysisResponse, DetectedObject, SuspicionLevel, ActionType


@dataclass
class MemoryFrame:
    """Représente une frame stockée en mémoire."""
    timestamp: float
    frame_id: int
    detections: List[Dict[str, Any]]  # Détections simplifiées
    persons_count: int
    vlm_triggered: bool
    vlm_analysis: Optional[Dict[str, Any]]  # Analyse VLM si disponible
    alert_level: str
    suspicion_score: float
    actions_taken: List[str]


@dataclass
class PersonMemory:
    """Mémoire d'une personne trackée."""
    person_id: str
    first_seen: float
    last_seen: float
    total_frames: int
    positions: List[Tuple[float, float]]  # (x, y) centers
    suspicious_behaviors: List[str]
    confidence_scores: List[float]
    

class VLMMemorySystem:
    """Système de mémoire contextuelle pour le VLM."""
    
    def __init__(self, max_frames: int = 50, max_persons: int = 20):
        self.max_frames = max_frames
        self.max_persons = max_persons
        
        # Mémoire des frames récentes (FIFO)
        self.frame_memory: deque = deque(maxlen=max_frames)
        
        # Mémoire des personnes trackées
        self.person_memory: Dict[str, PersonMemory] = {}
        
        # Historique des patterns suspects
        self.suspicious_patterns: List[Dict[str, Any]] = []
        
        # Métriques de performance
        self.memory_stats = {
            "total_frames_processed": 0,
            "vlm_analyses_stored": 0,
            "patterns_detected": 0,
            "memory_size_mb": 0.0
        }
        
        logger.info("🧠 Système de mémoire VLM initialisé")
    
    def add_frame(self, 
                  frame_id: int, 
                  detections: List[DetectedObject],
                  vlm_triggered: bool = False,
                  vlm_analysis: Optional[AnalysisResponse] = None,
                  alert_level: str = "normal",
                  actions_taken: List[str] = None) -> None:
        """Ajoute une nouvelle frame à la mémoire."""
        
        current_time = time.time()
        actions_taken = actions_taken or []
        
        # Simplifier les détections pour stockage
        simplified_detections = []
        persons_count = 0
        
        for detection in detections:
            simplified_det = {
                "class_name": detection.class_name,
                "confidence": detection.confidence,
                "bbox": [detection.bbox.x1, detection.bbox.y1, detection.bbox.x2, detection.bbox.y2],
                "track_id": detection.track_id
            }
            simplified_detections.append(simplified_det)
            
            if detection.class_name == "person":
                persons_count += 1
                self._update_person_memory(detection, current_time)
        
        # Simplifier l'analyse VLM si disponible
        vlm_analysis_dict = None
        suspicion_score = 0.0
        
        if vlm_analysis:
            vlm_analysis_dict = {
                "suspicion_level": vlm_analysis.suspicion_level.value,
                "confidence": vlm_analysis.confidence,
                "description": vlm_analysis.description[:200],  # Limiter la taille
                "action_type": vlm_analysis.action_type.value,
                "tools_used": vlm_analysis.tools_used
            }
            suspicion_score = self._calculate_suspicion_score(vlm_analysis.suspicion_level)
        
        # Créer la frame mémoire
        memory_frame = MemoryFrame(
            timestamp=current_time,
            frame_id=frame_id,
            detections=simplified_detections,
            persons_count=persons_count,
            vlm_triggered=vlm_triggered,
            vlm_analysis=vlm_analysis_dict,
            alert_level=alert_level,
            suspicion_score=suspicion_score,
            actions_taken=actions_taken
        )
        
        # Ajouter à la mémoire
        self.frame_memory.append(memory_frame)
        
        # Mettre à jour les statistiques
        self.memory_stats["total_frames_processed"] += 1
        if vlm_analysis:
            self.memory_stats["vlm_analyses_stored"] += 1
        
        # Analyser les patterns suspects
        self._detect_suspicious_patterns()
        
        # Nettoyer la mémoire des personnes anciennes
        self._cleanup_old_persons()
        
        logger.debug(f"Frame {frame_id} ajoutée à la mémoire (VLM: {vlm_triggered})")
    
    def _update_person_memory(self, detection: DetectedObject, current_time: float) -> None:
        """Met à jour la mémoire d'une personne."""
        
        if detection.track_id is None:
            return
            
        person_id = str(detection.track_id)
        
        # Calculer la position centrale
        center_x = (detection.bbox.x1 + detection.bbox.x2) / 2
        center_y = (detection.bbox.y1 + detection.bbox.y2) / 2
        
        if person_id in self.person_memory:
            # Mettre à jour la mémoire existante
            person = self.person_memory[person_id]
            person.last_seen = current_time
            person.total_frames += 1
            person.positions.append((center_x, center_y))
            person.confidence_scores.append(detection.confidence)
            
            # Limiter la taille de l'historique
            if len(person.positions) > 100:
                person.positions = person.positions[-50:]
                person.confidence_scores = person.confidence_scores[-50:]
        else:
            # Créer nouvelle mémoire de personne
            if len(self.person_memory) < self.max_persons:
                self.person_memory[person_id] = PersonMemory(
                    person_id=person_id,
                    first_seen=current_time,
                    last_seen=current_time,
                    total_frames=1,
                    positions=[(center_x, center_y)],
                    suspicious_behaviors=[],
                    confidence_scores=[detection.confidence]
                )
    
    def _calculate_suspicion_score(self, suspicion_level: SuspicionLevel) -> float:
        """Convertit le niveau de suspicion en score numérique."""
        mapping = {
            SuspicionLevel.LOW: 0.2,
            SuspicionLevel.MEDIUM: 0.5,
            SuspicionLevel.HIGH: 0.8,
            SuspicionLevel.CRITICAL: 1.0
        }
        return mapping.get(suspicion_level, 0.0)
    
    def _detect_suspicious_patterns(self) -> None:
        """Détecte des patterns suspects dans l'historique récent."""
        
        if len(self.frame_memory) < 10:  # Besoin d'au moins 10 frames
            return
            
        recent_frames = list(self.frame_memory)[-10:]  # 10 dernières frames
        
        # Pattern 1: Augmentation soudaine de population
        person_counts = [frame.persons_count for frame in recent_frames]
        if len(person_counts) >= 5:
            recent_avg = sum(person_counts[-5:]) / 5
            older_avg = sum(person_counts[:5]) / 5
            
            if recent_avg > older_avg + 2:  # Augmentation significative
                pattern = {
                    "type": "population_spike",
                    "timestamp": time.time(),
                    "description": f"Augmentation population: {older_avg:.1f} → {recent_avg:.1f}",
                    "severity": "medium"
                }
                self.suspicious_patterns.append(pattern)
                self.memory_stats["patterns_detected"] += 1
                logger.info(f"🚨 Pattern détecté: {pattern['description']}")
        
        # Pattern 2: Activité VLM fréquente (système nerveux)
        vlm_triggers = [frame.vlm_triggered for frame in recent_frames]
        if sum(vlm_triggers) >= 7:  # 7+ déclenchements sur 10 frames
            pattern = {
                "type": "high_vlm_activity",
                "timestamp": time.time(),
                "description": f"Activité VLM élevée: {sum(vlm_triggers)}/10 frames",
                "severity": "high"
            }
            self.suspicious_patterns.append(pattern)
            self.memory_stats["patterns_detected"] += 1
            logger.warning(f"⚠️ Pattern détecté: {pattern['description']}")
        
        # Pattern 3: Scores de suspicion croissants
        suspicion_scores = [frame.suspicion_score for frame in recent_frames if frame.suspicion_score > 0]
        if len(suspicion_scores) >= 3:
            if all(suspicion_scores[i] <= suspicion_scores[i+1] for i in range(len(suspicion_scores)-1)):
                pattern = {
                    "type": "escalating_suspicion",
                    "timestamp": time.time(),
                    "description": f"Suspicion croissante: {suspicion_scores[0]:.2f} → {suspicion_scores[-1]:.2f}",
                    "severity": "high"
                }
                self.suspicious_patterns.append(pattern)
                self.memory_stats["patterns_detected"] += 1
                logger.warning(f"📈 Pattern détecté: {pattern['description']}")
        
        # Limiter la taille de l'historique des patterns
        if len(self.suspicious_patterns) > 20:
            self.suspicious_patterns = self.suspicious_patterns[-10:]
    
    def _cleanup_old_persons(self) -> None:
        """Nettoie la mémoire des personnes non vues depuis longtemps."""
        
        current_time = time.time()
        cutoff_time = 30.0  # 30 secondes
        
        old_persons = [
            person_id for person_id, person in self.person_memory.items()
            if current_time - person.last_seen > cutoff_time
        ]
        
        for person_id in old_persons:
            del self.person_memory[person_id]
            logger.debug(f"Personne {person_id} supprimée de la mémoire (ancienne)")
    
    def get_context_for_vlm(self) -> Dict[str, Any]:
        """Génère le contexte historique pour le VLM."""
        
        if not self.frame_memory:
            return {"previous_detections": [], "memory_summary": "Aucun historique disponible"}
        
        # Résumé des dernières 5 frames
        recent_frames = list(self.frame_memory)[-5:]
        
        previous_detections = []
        for frame in recent_frames:
            frame_summary = {
                "frame_id": frame.frame_id,
                "timestamp": frame.timestamp,
                "persons_count": frame.persons_count,
                "alert_level": frame.alert_level,
                "vlm_triggered": frame.vlm_triggered
            }
            
            if frame.vlm_analysis:
                frame_summary["suspicion_level"] = frame.vlm_analysis["suspicion_level"]
                frame_summary["description"] = frame.vlm_analysis["description"]
            
            previous_detections.append(frame_summary)
        
        # Résumé des personnes actives
        active_persons = []
        for person_id, person in self.person_memory.items():
            if time.time() - person.last_seen < 10:  # Vue dans les 10 dernières secondes
                person_summary = {
                    "person_id": person_id,
                    "duration_seconds": person.last_seen - person.first_seen,
                    "total_frames": person.total_frames,
                    "avg_confidence": sum(person.confidence_scores) / len(person.confidence_scores),
                    "suspicious_behaviors": person.suspicious_behaviors
                }
                active_persons.append(person_summary)
        
        # Résumé des patterns suspects récents
        recent_patterns = [
            pattern for pattern in self.suspicious_patterns
            if time.time() - pattern["timestamp"] < 60  # Dernière minute
        ]
        
        # Tendances
        if len(self.frame_memory) >= 10:
            last_10_frames = list(self.frame_memory)[-10:]
            person_trend = [frame.persons_count for frame in last_10_frames]
            avg_persons = sum(person_trend) / len(person_trend)
            vlm_activity = sum(frame.vlm_triggered for frame in last_10_frames)
        else:
            avg_persons = 0
            vlm_activity = 0
        
        memory_summary = {
            "active_persons": len(active_persons),
            "avg_persons_last_10_frames": round(avg_persons, 1),
            "vlm_activity_last_10_frames": vlm_activity,
            "recent_patterns": len(recent_patterns),
            "memory_frames": len(self.frame_memory)
        }
        
        context = {
            "previous_detections": previous_detections,
            "active_persons": active_persons,
            "suspicious_patterns": recent_patterns,
            "memory_summary": memory_summary
        }
        
        return context
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques de la mémoire."""
        
        # Calcul approximatif de la taille mémoire
        memory_size = 0
        memory_size += len(str(self.frame_memory)) / 1024  # Approximation en KB
        memory_size += len(str(self.person_memory)) / 1024
        memory_size += len(str(self.suspicious_patterns)) / 1024
        
        self.memory_stats["memory_size_mb"] = memory_size / 1024
        
        return {
            **self.memory_stats,
            "current_frames_in_memory": len(self.frame_memory),
            "active_persons": len(self.person_memory),
            "recent_patterns": len(self.suspicious_patterns)
        }
    
    def clear_memory(self) -> None:
        """Vide complètement la mémoire."""
        self.frame_memory.clear()
        self.person_memory.clear()
        self.suspicious_patterns.clear()
        logger.info("🧠 Mémoire VLM vidée")
    
    def export_memory_dump(self) -> str:
        """Exporte un dump JSON de la mémoire pour debugging."""
        
        dump_data = {
            "timestamp": time.time(),
            "stats": self.memory_stats,
            "frame_memory": [asdict(frame) for frame in self.frame_memory],
            "person_memory": {
                person_id: asdict(person) 
                for person_id, person in self.person_memory.items()
            },
            "suspicious_patterns": self.suspicious_patterns
        }
        
        return json.dumps(dump_data, indent=2, default=str)


# Instance globale
vlm_memory = VLMMemorySystem()