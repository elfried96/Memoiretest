"""
Système de surveillance headless refactorisé.
"""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import defaultdict, deque

from .video_processor import VideoProcessor
from .frame_analyzer import FrameAnalyzer
from .result_models import SurveillanceResult, AlertLevel, SessionSummary
from ..orchestrator.vlm_orchestrator import OrchestrationConfig, OrchestrationMode

logger = logging.getLogger(__name__)


class HeadlessSurveillanceSystem:
    """Système de surveillance optimisé pour environnement headless."""

    def __init__(
        self,
        video_source: str = 0,
        vlm_model: str = "kimi-vl-a3b-thinking",
        orchestration_mode: OrchestrationMode = OrchestrationMode.BALANCED,
        save_results: bool = True,
        save_frames: bool = False,
        frame_skip: int = 1,
        max_frames: Optional[int] = None,
        vlm_analysis_mode: str = "smart",
        output_dir: str = "surveillance_output"
    ):
        """
        Initialise le système de surveillance headless.
        
        Args:
            video_source: Source vidéo (fichier ou webcam)
            vlm_model: Modèle VLM à utiliser
            orchestration_mode: Mode d'orchestration
            save_results: Sauvegarde des résultats JSON
            save_frames: Sauvegarde des frames
            frame_skip: Nombre de frames à ignorer
            max_frames: Limite de frames à traiter
            vlm_analysis_mode: Mode d'analyse VLM
            output_dir: Répertoire de sortie
        """
        # Configuration
        self.video_source = video_source
        self.vlm_model = vlm_model
        self.orchestration_mode = orchestration_mode
        self.save_results = save_results
        self.save_frames = save_frames
        self.vlm_analysis_mode = vlm_analysis_mode
        
        # Répertoires
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        if save_frames:
            self.frames_dir = self.output_dir / "frames"
            self.frames_dir.mkdir(exist_ok=True)

        # Composants
        self.video_processor = VideoProcessor(
            source=video_source,
            frame_skip=frame_skip,
            max_frames=max_frames
        )
        
        orchestration_config = OrchestrationConfig(mode=orchestration_mode)
        self.frame_analyzer = FrameAnalyzer(
            vlm_model=vlm_model,
            orchestration_config=orchestration_config
        )
        
        # État de la session
        self.results: List[SurveillanceResult] = []
        self.session_start_time = time.time()
        self.is_running = False
        
        # Métriques en temps réel
        self.alert_counts = defaultdict(int)
        self.recent_results = deque(maxlen=100)  # 100 derniers résultats
        
        logger.info(f"🎯 Système headless initialisé")
        logger.info(f"📹 Source: {video_source}")
        logger.info(f"🧠 VLM: {vlm_model}")
        logger.info(f"⚙️ Mode: {orchestration_mode}")

    async def run_surveillance(self) -> SessionSummary:
        """
        Lance la surveillance complète.
        
        Returns:
            Résumé de la session de surveillance
        """
        logger.info("🚀 Démarrage surveillance headless")
        self.is_running = True
        
        try:
            # Traitement des frames
            async for result in self._process_frames():
                self._update_metrics(result)
                
                if self.save_results:
                    self.results.append(result)
                
                # Sauvegarde frame si demandée
                if self.save_frames and result.alert_level != AlertLevel.NORMAL:
                    await self._save_critical_frame(result)
                
                # Log périodique
                if result.frame_id % 30 == 0:
                    self._log_progress(result)
            
            # Génération du résumé
            summary = self._generate_session_summary()
            
            # Sauvegarde finale
            if self.save_results:
                await self._save_session_results(summary)
            
            logger.info("✅ Surveillance terminée avec succès")
            return summary
            
        except Exception as e:
            logger.error(f"❌ Erreur surveillance: {e}")
            raise
        finally:
            self.is_running = False
            await self._cleanup()

    async def _process_frames(self):
        """Générateur de traitement des frames."""
        try:
            for frame_id, frame in self.video_processor.frames_generator():
                timestamp = time.time()
                
                # Analyse de la frame
                result = await self.frame_analyzer.analyze_frame(
                    frame=frame,
                    frame_id=frame_id,
                    timestamp=timestamp,
                    vlm_analysis_mode=self.vlm_analysis_mode
                )
                
                yield result
                
        except Exception as e:
            logger.error(f"❌ Erreur traitement frames: {e}")
            raise

    def _update_metrics(self, result: SurveillanceResult) -> None:
        """Met à jour les métriques de la session."""
        # Compteurs d'alertes
        self.alert_counts[result.alert_level.value] += 1
        
        # Historique récent
        self.recent_results.append(result)
        
        # Log des alertes importantes
        if result.alert_level in [AlertLevel.ALERTE, AlertLevel.CRITIQUE]:
            logger.warning(
                f"🚨 {result.alert_level.value.upper()} Frame {result.frame_id}: "
                f"{result.persons_detected} personnes détectées"
            )

    async def _save_critical_frame(self, result: SurveillanceResult) -> None:
        """Sauvegarde les frames critiques."""
        try:
            # Note: La frame actuelle n'est pas directement disponible ici
            # Dans une implémentation complète, il faudrait la passer en paramètre
            logger.debug(f"📸 Frame critique {result.frame_id} marquée pour sauvegarde")
        except Exception as e:
            logger.warning(f"⚠️ Échec sauvegarde frame {result.frame_id}: {e}")

    def _log_progress(self, result: SurveillanceResult) -> None:
        """Log de progression périodique."""
        elapsed = time.time() - self.session_start_time
        
        # Statistiques récentes
        recent_avg_time = sum(r.processing_time for r in self.recent_results) / len(self.recent_results)
        total_persons = sum(r.persons_detected for r in self.recent_results)
        
        logger.info(
            f"📊 Frame {result.frame_id} | "
            f"Temps: {elapsed:.1f}s | "
            f"Traitement moy: {recent_avg_time:.2f}s | "
            f"Personnes (100 dernières): {total_persons}"
        )

    def _generate_session_summary(self) -> SessionSummary:
        """Génère le résumé de la session."""
        session_duration = time.time() - self.session_start_time
        
        if not self.results:
            return SessionSummary(
                total_frames=0,
                total_detections=0,
                total_persons=0,
                alerts_by_level=dict(self.alert_counts),
                average_processing_time=0.0,
                session_duration=session_duration
            )
        
        # Statistiques globales
        total_frames = len(self.results)
        total_detections = sum(r.detections_count for r in self.results)
        total_persons = sum(r.persons_detected for r in self.results)
        avg_processing_time = sum(r.processing_time for r in self.results) / total_frames
        
        # Événements clés
        key_events = []
        for result in self.results:
            if result.alert_level in [AlertLevel.ALERTE, AlertLevel.CRITIQUE]:
                key_events.append({
                    "frame_id": result.frame_id,
                    "timestamp": result.timestamp,
                    "alert_level": result.alert_level.value,
                    "persons": result.persons_detected,
                    "actions": result.actions_taken
                })
        
        summary = SessionSummary(
            total_frames=total_frames,
            total_detections=total_detections,
            total_persons=total_persons,
            alerts_by_level=dict(self.alert_counts),
            average_processing_time=avg_processing_time,
            session_duration=session_duration,
            key_events=key_events
        )
        
        # Log du résumé
        logger.info("📋 RÉSUMÉ DE SESSION:")
        logger.info(f"  • Frames traitées: {total_frames}")
        logger.info(f"  • Détections totales: {total_detections}")
        logger.info(f"  • Personnes totales: {total_persons}")
        logger.info(f"  • Temps de traitement moyen: {avg_processing_time:.2f}s")
        logger.info(f"  • Durée session: {session_duration:.1f}s")
        logger.info(f"  • Alertes par niveau: {dict(self.alert_counts)}")
        
        return summary

    async def _save_session_results(self, summary: SessionSummary) -> None:
        """Sauvegarde les résultats de la session."""
        try:
            timestamp = int(time.time())
            
            # Fichier résumé
            summary_file = self.output_dir / f"session_summary_{timestamp}.json"
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary.to_dict(), f, indent=2, ensure_ascii=False)
            
            # Fichier détaillé
            if self.results:
                details_file = self.output_dir / f"session_details_{timestamp}.json"
                results_data = [result.to_dict() for result in self.results]
                
                with open(details_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        "session_info": summary.to_dict(),
                        "frame_results": results_data,
                        "video_info": self.video_processor.get_video_info(),
                        "analyzer_stats": self.frame_analyzer.get_performance_stats()
                    }, f, indent=2, ensure_ascii=False)
            
            logger.info(f"💾 Résultats sauvés: {summary_file}")
            
        except Exception as e:
            logger.error(f"❌ Erreur sauvegarde résultats: {e}")

    async def _cleanup(self) -> None:
        """Nettoyage des ressources."""
        try:
            self.video_processor.release()
            await self.frame_analyzer.cleanup()
            logger.info("🧹 Nettoyage terminé")
        except Exception as e:
            logger.warning(f"⚠️ Erreur nettoyage: {e}")

    def get_real_time_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques en temps réel."""
        if not self.recent_results:
            return {}
        
        elapsed = time.time() - self.session_start_time
        recent_avg_time = sum(r.processing_time for r in self.recent_results) / len(self.recent_results)
        
        return {
            "is_running": self.is_running,
            "elapsed_time": elapsed,
            "frames_processed": len(self.results),
            "recent_avg_processing_time": recent_avg_time,
            "alert_counts": dict(self.alert_counts),
            "recent_persons": sum(r.persons_detected for r in list(self.recent_results)[-10:])
        }