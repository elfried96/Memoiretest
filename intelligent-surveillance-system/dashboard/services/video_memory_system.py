"""
Système de mémoire vidéo avancé basé sur les patterns 2025.
Intégration complète avec l'architecture existante sans conflits.
"""

import asyncio
import streamlit as st
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import json
import numpy as np
from collections import deque
import hashlib
import uuid

# Import des services existants
from .session_manager import get_session_manager

@dataclass
class VideoMemoryFrame:
    """Frame mémoire individuel avec contexte détaillé."""
    timestamp: float
    frame_index: int
    description: str
    objects_detected: List[Dict]
    behaviors: List[Dict]
    scene_context: str
    confidence: float
    tools_used: List[str]
    suspicion_level: str

@dataclass 
class ConversationMemoryNode:
    """Noeud de mémoire conversationnelle."""
    node_id: str
    question: str
    response: str
    video_frames_referenced: List[int]
    confidence: float
    timestamp: datetime
    context_tokens_used: int

class AdvancedVideoMemorySystem:
    """
    Système de mémoire vidéo avancé inspiré des architectures 2025.
    Compatible avec l'architecture existante du projet.
    """
    
    def __init__(self, max_context_tokens: int = 1_000_000):
        self.max_context_tokens = max_context_tokens
        self.session_manager = get_session_manager()
        
        # Initialisation session state pour mémoire vidéo
        if 'detailed_video_analyses' not in st.session_state:
            st.session_state.detailed_video_analyses = {}
        
        if 'conversation_graph' not in st.session_state:
            st.session_state.conversation_graph = {}
        
        if 'video_memory_cache' not in st.session_state:
            st.session_state.video_memory_cache = {}
        
        # Métriques mémoire
        if 'memory_stats' not in st.session_state:
            st.session_state.memory_stats = {
                'videos_stored': 0,
                'queries_processed': 0,
                'memory_efficiency': 0.0,
                'last_cleanup': datetime.now()
            }
    
    def store_video_analysis(self, video_id: str, analysis_result: Dict) -> str:
        """
        Stockage analyse vidéo avec mémoire détaillée niveau frame.
        Compatible avec la structure existante de session_manager.
        """
        
        # Construction mémoire frame par frame
        frame_memories = []
        detailed_frames = analysis_result.get('detailed_frames', [])
        
        if not detailed_frames:
            # Fallback si pas de detailed_frames - création depuis data existante
            detailed_frames = self._create_frames_from_basic_analysis(analysis_result)
        
        for frame_data in detailed_frames:
            frame_memory = VideoMemoryFrame(
                timestamp=frame_data.get('timestamp', 0),
                frame_index=frame_data.get('frame_index', 0),
                description=frame_data.get('description', ''),
                objects_detected=self._extract_objects(frame_data),
                behaviors=self._extract_behaviors(frame_data),
                scene_context=self._build_scene_context(frame_data),
                confidence=frame_data.get('confidence', 0.0),
                tools_used=frame_data.get('tools_used', []),
                suspicion_level=frame_data.get('suspicion_level', 'LOW')
            )
            frame_memories.append(frame_memory)
        
        # Narrative complet inspiré Gemini 2.5 long-context
        complete_narrative = self._build_comprehensive_narrative(
            analysis_result, frame_memories
        )
        
        # Structure compatible avec session_manager existant
        enhanced_analysis = {
            **analysis_result,  # Garde toutes les données existantes
            'video_id': video_id,
            'frame_memories': [asdict(frame) for frame in frame_memories],
            'complete_narrative': complete_narrative,
            'conversation_ready_context': self._prepare_conversation_context(
                frame_memories, complete_narrative
            ),
            'searchable_index': self._build_searchable_index(frame_memories),
            'enhanced_timestamp': datetime.now().isoformat(),
            'token_count': self._estimate_token_count(complete_narrative),
            'memory_version': '2025_v1'
        }
        
        # Stockage dans session state avancé
        st.session_state.detailed_video_analyses[video_id] = enhanced_analysis
        
        # Mise à jour session manager existant (compatibilité)
        self.session_manager.store_video_analysis(video_id, enhanced_analysis)
        
        # Mise à jour stats
        st.session_state.memory_stats['videos_stored'] += 1
        self._update_memory_cache(video_id)
        
        return video_id
    
    def query_video_memory(
        self, 
        video_id: str, 
        question: str, 
        conversation_history: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """
        Query mémoire vidéo avec contexte conversationnel.
        Compatible avec l'interface chat existante.
        """
        
        if video_id not in st.session_state.detailed_video_analyses:
            # Fallback vers session_manager classique
            basic_analysis = self.session_manager.get_video_analysis(video_id)
            if basic_analysis:
                return self._generate_basic_response(question, basic_analysis)
            return self._empty_response()
        
        video_memory = st.session_state.detailed_video_analyses[video_id]
        
        # Analyse sémantique de la question
        query_analysis = self._analyze_question_intent(question)
        
        # Retrieval intelligent basé sur intent
        relevant_frames = self._retrieve_relevant_frames(
            video_memory, query_analysis
        )
        
        # Construction réponse contextuelle
        response = self._build_contextual_response(
            question, query_analysis, relevant_frames, 
            video_memory['complete_narrative']
        )
        
        # Enregistrement noeud conversationnel
        conversation_node = self._create_conversation_node(
            question, response, relevant_frames, video_id
        )
        
        # Mise à jour stats
        st.session_state.memory_stats['queries_processed'] += 1
        
        return {
            'response': response,
            'confidence': conversation_node.confidence,
            'frames_referenced': [f['frame_index'] for f in relevant_frames],
            'conversation_node_id': conversation_node.node_id,
            'metadata': {
                'video_id': video_id,
                'query_type': query_analysis['intent_type'],
                'context_tokens': conversation_node.context_tokens_used,
                'memory_efficiency': self._calculate_memory_efficiency(),
                'source': 'advanced_video_memory',
                'analysis_time': 0.5
            }
        }
    
    def _create_frames_from_basic_analysis(self, analysis_result: Dict) -> List[Dict]:
        """
        Crée des frames détaillées depuis une analyse basique.
        Pour compatibilité avec analyses existantes.
        """
        
        frame_count = analysis_result.get('frame_count', 5)
        duration_str = analysis_result.get('analysis_time', 2.0)
        
        # Extraction des objets détectés
        detected_objects = analysis_result.get('detected_objects', [])
        behaviors = analysis_result.get('behaviors', [])
        timeline = analysis_result.get('timeline', [])
        
        frames = []
        for i in range(frame_count):
            timestamp = i * (float(duration_str) / frame_count) if frame_count > 0 else 0
            
            # Distribution des objets/comportements sur les frames
            frame_objects = [obj for j, obj in enumerate(detected_objects) 
                           if j % frame_count == i]
            frame_behaviors = [beh for j, beh in enumerate(behaviors) 
                             if j % frame_count == i]
            
            # Description basée sur timeline si disponible
            frame_desc = "Mouvement normal détecté"
            for event in timeline:
                event_time = self._parse_time(event.get('time', '00:00'))
                if abs(event_time - timestamp) < 5:  # 5 secondes de tolérance
                    frame_desc = event.get('event', frame_desc)
                    break
            
            frames.append({
                'frame_index': i,
                'timestamp': timestamp,
                'description': frame_desc,
                'objects_detected': frame_objects,
                'behaviors': frame_behaviors,
                'confidence': analysis_result.get('confidence', 0.8),
                'suspicion_level': analysis_result.get('suspicion_level', 'LOW'),
                'tools_used': ['fallback_creation']
            })
        
        return frames
    
    def _build_comprehensive_narrative(
        self, 
        analysis_result: Dict, 
        frame_memories: List[VideoMemoryFrame]
    ) -> str:
        """
        Construit narrative complet compatible avec le format existant.
        """
        
        video_name = analysis_result.get('video_name', 'cette vidéo')
        duration = self._format_duration(analysis_result.get('analysis_time', 2.0))
        
        narrative_parts = [
            f"Dans {video_name} de {duration}, voici l'analyse détaillée :"
        ]
        
        # Timeline détaillée basée sur frames
        for frame in frame_memories:
            time_str = f"{frame.timestamp:.1f}s"
            
            frame_desc = f"À {time_str}: {frame.description}"
            
            # Enrichissement avec objets détectés
            if frame.objects_detected:
                objects_summary = self._summarize_objects(frame.objects_detected)
                if objects_summary:
                    frame_desc += f" (Détection: {objects_summary})"
            
            # Enrichissement avec comportements
            if frame.behaviors:
                behaviors_summary = self._summarize_behaviors(frame.behaviors)
                if behaviors_summary:
                    frame_desc += f" [Comportement: {behaviors_summary}]"
            
            narrative_parts.append(frame_desc)
        
        # Conclusion basée sur niveau suspicion global
        suspicion = analysis_result.get('suspicion_level', 'LOW')
        if suspicion == 'LOW':
            narrative_parts.append(
                "Conclusion: Les personnes observées ne faisaient qu'examiner "
                "des produits et ont tout redéposé avant de poursuivre leur cours. "
                "Comportement normal détecté."
            )
        elif suspicion == 'MEDIUM':
            narrative_parts.append(
                "Conclusion: Comportement légèrement inhabituel observé, "
                "mais pas de menace immédiate détectée."
            )
        else:
            narrative_parts.append(
                "Conclusion: Comportement suspect détecté nécessitant "
                "une attention particulière."
            )
        
        return ". ".join(narrative_parts) + "."
    
    def _analyze_question_intent(self, question: str) -> Dict[str, Any]:
        """
        Analyse intention question - compatible avec logique existante.
        """
        
        question_lower = question.lower()
        
        intent_patterns = {
            'detail_request': ['détail', 'exactement', 'précis', 'spécifique', 'vu'],
            'timeline_query': ['quand', 'moment', 'temps', 'heure', 'temporel'],
            'object_count': ['combien', 'nombre', 'quantité'],
            'behavior_analysis': ['comportement', 'action', 'geste', 'activité'],
            'location_query': ['où', 'zone', 'endroit', 'position'],
            'summary_request': ['résumé', 'global', 'général', 'ensemble']
        }
        
        detected_intents = []
        for intent, keywords in intent_patterns.items():
            if any(keyword in question_lower for keyword in keywords):
                detected_intents.append(intent)
        
        primary_intent = detected_intents[0] if detected_intents else 'general'
        
        return {
            'intent_type': primary_intent,
            'all_intents': detected_intents,
            'question_complexity': len(question.split()),
            'requires_frame_level': primary_intent in ['detail_request', 'timeline_query'],
            'requires_aggregation': primary_intent in ['summary_request', 'behavior_analysis']
        }
    
    def _retrieve_relevant_frames(
        self, 
        video_memory: Dict, 
        query_analysis: Dict
    ) -> List[Dict]:
        """
        Retrieval intelligent frames pertinents.
        """
        
        frame_memories = video_memory.get('frame_memories', [])
        intent = query_analysis['intent_type']
        
        if intent == 'detail_request':
            return frame_memories  # Tous les frames
            
        elif intent == 'timeline_query':
            return sorted(frame_memories, key=lambda f: f.get('timestamp', 0))
            
        elif intent == 'object_count':
            return [f for f in frame_memories if f.get('objects_detected', [])]
            
        elif intent == 'behavior_analysis':
            return [f for f in frame_memories if f.get('behaviors', [])]
            
        else:
            # Top frames par confiance
            return sorted(frame_memories, 
                        key=lambda f: f.get('confidence', 0), reverse=True)[:3]
    
    def _build_contextual_response(
        self, 
        question: str, 
        query_analysis: Dict, 
        relevant_frames: List[Dict],
        complete_narrative: str
    ) -> str:
        """
        Construction réponse contextuelle intelligente.
        """
        
        intent = query_analysis['intent_type']
        
        if intent == 'detail_request':
            return complete_narrative
            
        elif intent == 'timeline_query':
            response = "Voici la timeline détaillée des événements :\n"
            for frame in relevant_frames:
                timestamp = frame.get('timestamp', 0)
                description = frame.get('description', 'Événement non décrit')
                response += f"• {timestamp:.1f}s: {description}\n"
            return response.strip()
            
        elif intent == 'object_count':
            total_objects = {}
            for frame in relevant_frames:
                for obj in frame.get('objects_detected', []):
                    obj_type = obj.get('type', 'objet')
                    count = obj.get('count', 1)
                    total_objects[obj_type] = total_objects.get(obj_type, 0) + count
            
            if total_objects:
                response = "Objets détectés dans la vidéo :\n"
                for obj_type, count in total_objects.items():
                    response += f"• {count} {obj_type}(s)\n"
                return response.strip()
            else:
                return "Aucun objet spécifique détecté dans cette vidéo."
            
        elif intent == 'behavior_analysis':
            behaviors = set()
            for frame in relevant_frames:
                for behavior in frame.get('behaviors', []):
                    if behavior.get('confidence', 0) > 0.7:
                        behaviors.add(behavior.get('type', 'comportement'))
            
            if behaviors:
                return f"Comportements observés : {', '.join(behaviors)}"
            else:
                return "Comportements normaux observés sans particularités."
                
        else:
            # Réponse générale
            if relevant_frames:
                top_frame = relevant_frames[0]
                description = top_frame.get('description', 'Analyse disponible')
                return f"Basé sur l'analyse vidéo : {description}"
            return "Analyse vidéo disponible pour répondre à vos questions."
    
    def _create_conversation_node(
        self, 
        question: str, 
        response: str, 
        relevant_frames: List[Dict],
        video_id: str
    ) -> ConversationMemoryNode:
        """
        Crée un noeud de conversation pour la mémoire.
        """
        
        node_id = str(uuid.uuid4())
        frame_indices = [f.get('frame_index', 0) for f in relevant_frames]
        
        # Calcul confiance basé sur frames et qualité réponse
        confidence = 0.9 if len(relevant_frames) > 0 else 0.7
        
        node = ConversationMemoryNode(
            node_id=node_id,
            question=question,
            response=response,
            video_frames_referenced=frame_indices,
            confidence=confidence,
            timestamp=datetime.now(),
            context_tokens_used=len(response.split()) * 1.3  # Estimation
        )
        
        # Stockage dans graph conversationnel
        if video_id not in st.session_state.conversation_graph:
            st.session_state.conversation_graph[video_id] = []
        
        st.session_state.conversation_graph[video_id].append(asdict(node))
        
        return node
    
    # Méthodes utilitaires
    
    def _extract_objects(self, frame_data: Dict) -> List[Dict]:
        """Extrait objets détectés du frame."""
        if 'objects_detected' in frame_data:
            return frame_data['objects_detected']
        
        # Fallback depuis analysis si structure différente
        analysis = frame_data.get('analysis', {})
        if hasattr(analysis, 'detected_objects'):
            return analysis.detected_objects
        
        return []
    
    def _extract_behaviors(self, frame_data: Dict) -> List[Dict]:
        """Extrait comportements du frame."""
        if 'behaviors' in frame_data:
            return frame_data['behaviors']
        
        analysis = frame_data.get('analysis', {})
        if hasattr(analysis, 'behaviors'):
            return analysis.behaviors
        
        return []
    
    def _build_scene_context(self, frame_data: Dict) -> str:
        """Construit contexte de scène."""
        context_parts = []
        
        if frame_data.get('objects_detected'):
            obj_count = len(frame_data['objects_detected'])
            context_parts.append(f"{obj_count} objet(s) détecté(s)")
        
        if frame_data.get('behaviors'):
            behavior_count = len(frame_data['behaviors'])
            context_parts.append(f"{behavior_count} comportement(s) analysé(s)")
        
        suspicion = frame_data.get('suspicion_level', 'LOW')
        context_parts.append(f"niveau {suspicion}")
        
        return ", ".join(context_parts) if context_parts else "scène normale"
    
    def _estimate_token_count(self, text: str) -> int:
        """Estimation simple du nombre de tokens."""
        return len(text.split()) * 1.3
    
    def _format_duration(self, seconds: float) -> str:
        """Formate durée en format lisible."""
        if seconds < 60:
            return f"{seconds:.0f} secondes"
        else:
            minutes = int(seconds // 60)
            remaining_seconds = int(seconds % 60)
            return f"{minutes}min {remaining_seconds}s" if remaining_seconds > 0 else f"{minutes}min"
    
    def _parse_time(self, time_str: str) -> float:
        """Parse temps format 'MM:SS' vers secondes."""
        try:
            if ':' in time_str:
                parts = time_str.split(':')
                if len(parts) == 2:
                    minutes = int(parts[0])
                    seconds = int(parts[1])
                    return minutes * 60 + seconds
            return float(time_str.replace('s', ''))
        except:
            return 0.0
    
    def _summarize_objects(self, objects: List[Dict]) -> str:
        """Résumé objets détectés."""
        if not objects:
            return ""
        
        summary_parts = []
        for obj in objects:
            count = obj.get('count', 1)
            obj_type = obj.get('type', 'objet')
            summary_parts.append(f"{count} {obj_type}")
        
        return ", ".join(summary_parts)
    
    def _summarize_behaviors(self, behaviors: List[Dict]) -> str:
        """Résumé comportements."""
        if not behaviors:
            return ""
        
        behavior_types = []
        for behavior in behaviors:
            if behavior.get('confidence', 0) > 0.7:
                behavior_types.append(behavior.get('type', 'comportement'))
        
        return ", ".join(set(behavior_types))
    
    def _prepare_conversation_context(self, frame_memories, narrative) -> Dict:
        """Prépare contexte pour conversations."""
        return {
            'ready_for_questions': True,
            'frame_count': len(frame_memories),
            'narrative_length': len(narrative),
            'can_answer_about': [
                'détails des événements',
                'timeline temporelle',
                'objets détectés',
                'comportements observés'
            ]
        }
    
    def _build_searchable_index(self, frame_memories) -> Dict:
        """Construit index de recherche."""
        index = {
            'keywords': set(),
            'timestamps': [f.timestamp for f in frame_memories],
            'objects': set(),
            'behaviors': set()
        }
        
        for frame in frame_memories:
            # Mots-clés de description
            description_words = frame.description.lower().split()
            index['keywords'].update(description_words)
            
            # Objets
            for obj in frame.objects_detected:
                index['objects'].add(obj.get('type', ''))
            
            # Comportements
            for behavior in frame.behaviors:
                index['behaviors'].add(behavior.get('type', ''))
        
        # Conversion sets en listes pour JSON
        return {
            'keywords': list(index['keywords']),
            'timestamps': index['timestamps'],
            'objects': list(index['objects']),
            'behaviors': list(index['behaviors'])
        }
    
    def _update_memory_cache(self, video_id: str):
        """Met à jour cache mémoire."""
        st.session_state.video_memory_cache[video_id] = {
            'last_access': datetime.now(),
            'access_count': st.session_state.video_memory_cache.get(video_id, {}).get('access_count', 0) + 1
        }
    
    def _calculate_memory_efficiency(self) -> float:
        """Calcule efficacité mémoire."""
        total_videos = st.session_state.memory_stats['videos_stored']
        total_queries = st.session_state.memory_stats['queries_processed']
        
        if total_videos == 0:
            return 0.0
        
        # Efficacité = queries per video
        efficiency = total_queries / total_videos if total_videos > 0 else 0
        
        # Normalisation 0-1
        return min(efficiency / 10.0, 1.0)
    
    def _generate_basic_response(self, question: str, basic_analysis: Dict) -> Dict[str, Any]:
        """Génère réponse depuis analyse basique (fallback)."""
        description = basic_analysis.get('description', 'Analyse vidéo disponible')
        
        return {
            'response': f"Basé sur l'analyse de base: {description}",
            'confidence': 0.6,
            'frames_referenced': [],
            'conversation_node_id': str(uuid.uuid4()),
            'metadata': {
                'source': 'basic_fallback',
                'analysis_time': 0.2
            }
        }
    
    def _empty_response(self) -> Dict[str, Any]:
        """Réponse vide."""
        return {
            'response': "Aucune analyse vidéo trouvée pour cette question.",
            'confidence': 0.0,
            'frames_referenced': [],
            'conversation_node_id': str(uuid.uuid4()),
            'metadata': {
                'source': 'empty',
                'analysis_time': 0.1
            }
        }
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Statistiques du système mémoire."""
        stats = st.session_state.memory_stats.copy()
        stats.update({
            'videos_in_memory': len(st.session_state.detailed_video_analyses),
            'conversations_tracked': len(st.session_state.conversation_graph),
            'cache_entries': len(st.session_state.video_memory_cache),
            'memory_efficiency': self._calculate_memory_efficiency()
        })
        return stats
    
    def cleanup_old_memories(self, max_age_days: int = 7):
        """Nettoie anciennes mémoires."""
        cutoff_date = datetime.now() - timedelta(days=max_age_days)
        
        # Nettoyage analyses anciennes
        to_remove = []
        for video_id, analysis in st.session_state.detailed_video_analyses.items():
            enhanced_timestamp = analysis.get('enhanced_timestamp')
            if enhanced_timestamp:
                timestamp = datetime.fromisoformat(enhanced_timestamp)
                if timestamp < cutoff_date:
                    to_remove.append(video_id)
        
        for video_id in to_remove:
            del st.session_state.detailed_video_analyses[video_id]
            if video_id in st.session_state.conversation_graph:
                del st.session_state.conversation_graph[video_id]
            if video_id in st.session_state.video_memory_cache:
                del st.session_state.video_memory_cache[video_id]
        
        # Mise à jour stats dernière cleanup
        st.session_state.memory_stats['last_cleanup'] = datetime.now()
        
        return len(to_remove)

# Instance globale
_video_memory_system: Optional[AdvancedVideoMemorySystem] = None

def get_video_memory_system() -> AdvancedVideoMemorySystem:
    """Récupère l'instance du système mémoire vidéo."""
    global _video_memory_system
    
    if _video_memory_system is None:
        _video_memory_system = AdvancedVideoMemorySystem()
    
    return _video_memory_system