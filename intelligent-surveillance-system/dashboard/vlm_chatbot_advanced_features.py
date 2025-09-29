"""
[ADVANCED] VLM Chatbot Advanced Features
===============================

Fonctionnalités avancées pour le chatbot VLM:
- Mode conversation contextuelle multi-tours
- Analyse comparative inter-temporelle  
- Recommandations proactives intelligentes
- Intégration avec alertes système
- Mode expertise technique approfondi
- Analyse prédictive tendances
- Export rapports conversationnels
"""

import asyncio
import json
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
import pandas as pd
from loguru import logger

# Analytics et ML
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn non disponible - Analytics limités")


class ConversationMode(Enum):
    """Modes de conversation avancés."""
    STANDARD = "standard"
    CONTEXTUAL = "contextual" 
    ANALYTICAL = "analytical"
    PREDICTIVE = "predictive"
    EXPERT = "expert"
    TUTORIAL = "tutorial"


class AlertPriority(Enum):
    """Niveaux de priorité des alertes."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ConversationContext:
    """Contexte de conversation multi-tours."""
    session_id: str
    user_intent: str
    conversation_mode: ConversationMode
    topics_discussed: List[str]
    context_memory: Dict[str, Any]
    expertise_level: str  # beginner, intermediate, expert
    preferred_detail_level: str  # brief, standard, detailed
    last_interaction: datetime
    
    def is_stale(self, max_age_minutes: int = 60) -> bool:
        """Vérifie si le contexte est périmé."""
        return datetime.now() - self.last_interaction > timedelta(minutes=max_age_minutes)


class SmartAlert:
    """Alerte intelligente générée par l'analyse VLM."""
    
    def __init__(
        self,
        alert_id: str,
        title: str,
        description: str,
        priority: AlertPriority,
        category: str,
        data_source: Dict[str, Any],
        recommendations: List[str],
        auto_generated: bool = True
    ):
        self.alert_id = alert_id
        self.title = title
        self.description = description
        self.priority = priority
        self.category = category
        self.data_source = data_source
        self.recommendations = recommendations
        self.auto_generated = auto_generated
        self.timestamp = datetime.now()
        self.acknowledged = False
        self.resolved = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'alert_id': self.alert_id,
            'title': self.title,
            'description': self.description,
            'priority': self.priority.value,
            'category': self.category,
            'recommendations': self.recommendations,
            'timestamp': self.timestamp.isoformat(),
            'acknowledged': self.acknowledged,
            'resolved': self.resolved,
            'auto_generated': self.auto_generated
        }


class TrendAnalyzer:
    """Analyseur de tendances pour données VLM."""
    
    def __init__(self, history_size: int = 1000):
        self.history_size = history_size
        self.performance_history: List[Dict[str, Any]] = []
        self.detection_history: List[Dict[str, Any]] = []
        self.optimization_history: List[Dict[str, Any]] = []
    
    def add_performance_data(self, data: Dict[str, Any]):
        """Ajoute données de performance à l'historique."""
        data['timestamp'] = datetime.now()
        self.performance_history.append(data)
        
        if len(self.performance_history) > self.history_size:
            self.performance_history = self.performance_history[-self.history_size:]
    
    def analyze_performance_trend(self, window_hours: int = 24) -> Dict[str, Any]:
        """Analyse la tendance de performance sur une fenêtre temporelle."""
        
        cutoff_time = datetime.now() - timedelta(hours=window_hours)
        recent_data = [
            d for d in self.performance_history
            if d.get('timestamp', datetime.min) > cutoff_time
        ]
        
        if len(recent_data) < 5:
            return {"trend": "insufficient_data", "confidence": 0.0}
        
        # Analyse tendance performance score
        scores = [d.get('performance_score', 0) for d in recent_data]
        processing_times = [d.get('avg_processing_time', 0) for d in recent_data]
        
        # Calcul tendances
        score_trend = self._calculate_trend(scores)
        time_trend = self._calculate_trend(processing_times)
        
        return {
            "trend": self._classify_trend(score_trend, time_trend),
            "performance_change": score_trend,
            "speed_change": -time_trend,  # Négatif car moins de temps = mieux
            "confidence": min(1.0, len(recent_data) / 20),
            "data_points": len(recent_data),
            "window_hours": window_hours
        }
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calcule la tendance d'une série de valeurs."""
        if len(values) < 2:
            return 0.0
        
        x = np.arange(len(values))
        coeffs = np.polyfit(x, values, 1)
        return coeffs[0]  # Coefficient de tendance
    
    def _classify_trend(self, score_trend: float, time_trend: float) -> str:
        """Classifie la tendance globale."""
        if score_trend > 0.01 and time_trend < -0.1:
            return "improving"
        elif score_trend < -0.01 and time_trend > 0.1:
            return "degrading"
        elif abs(score_trend) < 0.01 and abs(time_trend) < 0.1:
            return "stable"
        else:
            return "mixed"
    
    def predict_next_performance(self) -> Dict[str, Any]:
        """Prédiction basique de la prochaine performance."""
        
        if len(self.performance_history) < 10:
            return {"prediction": "insufficient_data"}
        
        recent_scores = [d.get('performance_score', 0) for d in self.performance_history[-10:]]
        recent_times = [d.get('avg_processing_time', 0) for d in self.performance_history[-10:]]
        
        # Prédiction simple par moyenne pondérée
        score_prediction = np.average(recent_scores, weights=range(1, len(recent_scores) + 1))
        time_prediction = np.average(recent_times, weights=range(1, len(recent_times) + 1))
        
        return {
            "predicted_performance_score": round(score_prediction, 3),
            "predicted_processing_time": round(time_prediction, 2),
            "confidence": 0.7,  # Confidence basique
            "prediction_horizon": "next_analysis"
        }


class ProactiveRecommendationEngine:
    """Moteur de recommandations proactives basé sur l'analyse VLM."""
    
    def __init__(self):
        self.trend_analyzer = TrendAnalyzer()
        self.recommendation_history: List[Dict[str, Any]] = []
        self.ignored_recommendations: List[str] = []
    
    def generate_proactive_recommendations(
        self, 
        current_stats: Dict[str, Any],
        recent_detections: List[Any],
        optimization_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Génère des recommandations proactives intelligentes."""
        
        recommendations = []
        
        # Analyse performance
        performance_recommendations = self._analyze_performance_issues(current_stats)
        recommendations.extend(performance_recommendations)
        
        # Analyse détections
        detection_recommendations = self._analyze_detection_patterns(recent_detections)
        recommendations.extend(detection_recommendations)
        
        # Analyse optimisations
        optimization_recommendations = self._analyze_optimization_opportunities(optimization_results)
        recommendations.extend(optimization_recommendations)
        
        # Filtrage recommandations déjà ignorées
        filtered = [r for r in recommendations if r['id'] not in self.ignored_recommendations]
        
        # Tri par priorité
        filtered.sort(key=lambda x: x.get('priority_score', 0), reverse=True)
        
        return filtered[:5]  # Top 5 recommandations
    
    def _analyze_performance_issues(self, stats: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyse des problèmes de performance."""
        recommendations = []
        
        avg_time = stats.get('average_processing_time', 0)
        performance_score = stats.get('current_performance_score', 1.0)
        frames_processed = stats.get('frames_processed', 0)
        
        # Temps de traitement élevé
        if avg_time > 5.0:
            recommendations.append({
                'id': 'high_processing_time',
                'type': 'performance_optimization',
                'title': 'Temps de traitement élevé détecté',
                'description': f'Temps moyen: {avg_time:.1f}s (optimal: <3s)',
                'recommendations': [
                    'Activer quantization 4-bit pour réduire VRAM',
                    'Réduire résolution images d\'entrée',
                    'Optimiser sélection outils (moins d\'outils simultanés)'
                ],
                'priority_score': 0.8,
                'data_source': {'avg_processing_time': avg_time}
            })
        
        # Score de performance faible
        if performance_score < 0.7:
            recommendations.append({
                'id': 'low_performance_score',
                'type': 'model_optimization',
                'title': 'Score de performance sous-optimal',
                'description': f'Score actuel: {performance_score:.2f} (objectif: >0.8)',
                'recommendations': [
                    'Lancer cycle d\'optimisation adaptative',
                    'Vérifier qualité données d\'entrée',
                    'Calibrer seuils de confiance'
                ],
                'priority_score': 0.9,
                'data_source': {'performance_score': performance_score}
            })
        
        # Peu de frames traitées (possible problème pipeline)
        if frames_processed < 10 and avg_time > 0:
            recommendations.append({
                'id': 'low_throughput',
                'type': 'pipeline_issue',
                'title': 'Débit de traitement faible',
                'description': f'Seulement {frames_processed} frames traitées',
                'recommendations': [
                    'Vérifier source vidéo/caméra',
                    'Diagnostiquer goulots d\'étranglement pipeline',
                    'Optimiser configuration hardware'
                ],
                'priority_score': 0.7,
                'data_source': {'frames_processed': frames_processed}
            })
        
        return recommendations
    
    def _analyze_detection_patterns(self, detections: List[Any]) -> List[Dict[str, Any]]:
        """Analyse des patterns dans les détections."""
        recommendations = []
        
        if not detections:
            return recommendations
        
        # Analyse confiance des détections
        confidences = [getattr(d, 'confidence', 0) for d in detections]
        avg_confidence = np.mean(confidences) if confidences else 0
        
        if avg_confidence < 0.8:
            recommendations.append({
                'id': 'low_detection_confidence',
                'type': 'detection_quality',
                'title': 'Confiance détections sous-optimale',
                'description': f'Confiance moyenne: {avg_confidence:.1%} (objectif: >80%)',
                'recommendations': [
                    'Améliorer qualité/résolution images',
                    'Ajuster éclairage environnement',
                    'Recalibrer seuils de détection'
                ],
                'priority_score': 0.6,
                'data_source': {'avg_confidence': avg_confidence, 'detection_count': len(detections)}
            })
        
        # Détections répétitives (possible faux positifs)
        descriptions = [getattr(d, 'description', '') for d in detections]
        if len(set(descriptions)) < len(descriptions) * 0.7:  # 70% de diversité minimum
            recommendations.append({
                'id': 'repetitive_detections',
                'type': 'false_positive_prevention',
                'title': 'Détections répétitives détectées',
                'description': 'Possible surajustement ou faux positifs récurrents',
                'recommendations': [
                    'Analyser patterns de faux positifs',
                    'Ajuster sensibilité détection',
                    'Diversifier données d\'entraînement'
                ],
                'priority_score': 0.5,
                'data_source': {'unique_descriptions': len(set(descriptions)), 'total_detections': len(descriptions)}
            })
        
        return recommendations
    
    def _analyze_optimization_opportunities(self, optimizations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyse des opportunités d'optimisation."""
        recommendations = []
        
        if not optimizations:
            recommendations.append({
                'id': 'no_recent_optimization',
                'type': 'optimization_needed',
                'title': 'Aucune optimisation récente',
                'description': 'Le système pourrait bénéficier d\'une optimisation adaptative',
                'recommendations': [
                    'Lancer cycle d\'optimisation manuel',
                    'Activer optimisation automatique périodique',
                    'Évaluer nouvelles combinaisons d\'outils'
                ],
                'priority_score': 0.4
            })
            return recommendations
        
        # Analyse amélioration récente
        recent_improvement = optimizations[-1].get('performance_improvement', 0)
        
        if recent_improvement < 0.05:  # Moins de 5% d'amélioration
            recommendations.append({
                'id': 'low_optimization_gain',
                'type': 'optimization_plateau',
                'title': 'Gains d\'optimisation limités',
                'description': f'Dernière amélioration: {recent_improvement:.1%}',
                'recommendations': [
                    'Explorer nouveaux outils/configurations',
                    'Analyser données pour patterns non détectés',
                    'Considérer mise à jour modèle VLM'
                ],
                'priority_score': 0.6,
                'data_source': {'recent_improvement': recent_improvement}
            })
        
        return recommendations


class ConversationalAnalytics:
    """Analytics avancées des conversations VLM."""
    
    def __init__(self):
        self.conversations: List[Dict[str, Any]] = []
        self.topic_clustering_enabled = SKLEARN_AVAILABLE
        
        if SKLEARN_AVAILABLE:
            self.vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
            self.topic_clusters = None
    
    def add_conversation(self, conversation_data: Dict[str, Any]):
        """Ajoute une conversation aux analytics."""
        conversation_data['timestamp'] = datetime.now()
        self.conversations.append(conversation_data)
        
        # Limitation taille historique
        if len(self.conversations) > 1000:
            self.conversations = self.conversations[-1000:]
    
    def analyze_conversation_patterns(self) -> Dict[str, Any]:
        """Analyse les patterns dans les conversations."""
        
        if not self.conversations:
            return {"analysis": "no_conversations"}
        
        analysis = {}
        
        # Fréquence questions
        questions = [c.get('question', '') for c in self.conversations]
        question_lengths = [len(q.split()) for q in questions]
        
        analysis['question_stats'] = {
            'total_conversations': len(self.conversations),
            'avg_question_length': np.mean(question_lengths),
            'most_common_words': self._extract_common_words(questions)
        }
        
        # Types de réponses
        response_types = [c.get('response_type', 'unknown') for c in self.conversations]
        type_counts = pd.Series(response_types).value_counts().to_dict()
        
        analysis['response_patterns'] = {
            'type_distribution': type_counts,
            'vlm_thinking_rate': type_counts.get('vlm_thinking', 0) / len(self.conversations),
            'fallback_rate': type_counts.get('fallback', 0) / len(self.conversations)
        }
        
        # Clustering topics si disponible
        if self.topic_clustering_enabled and len(questions) > 10:
            analysis['topic_analysis'] = self._analyze_topics(questions)
        
        return analysis
    
    def _extract_common_words(self, texts: List[str]) -> Dict[str, int]:
        """Extrait les mots les plus fréquents."""
        
        # Mots simples sans ML
        word_counts = {}
        for text in texts:
            words = text.lower().split()
            for word in words:
                if len(word) > 3:  # Ignorer mots courts
                    word_counts[word] = word_counts.get(word, 0) + 1
        
        return dict(sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:10])
    
    def _analyze_topics(self, questions: List[str]) -> Dict[str, Any]:
        """Analyse des topics avec clustering."""
        
        try:
            # Vectorisation TF-IDF
            tfidf_matrix = self.vectorizer.fit_transform(questions)
            
            # Clustering K-means
            n_clusters = min(5, len(questions) // 3)  # Max 5 clusters
            if n_clusters < 2:
                return {"topics": "insufficient_data"}
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(tfidf_matrix)
            
            # Extraction topics
            topics = {}
            feature_names = self.vectorizer.get_feature_names_out()
            
            for i in range(n_clusters):
                cluster_center = kmeans.cluster_centers_[i]
                top_indices = cluster_center.argsort()[-5:][::-1]
                top_words = [feature_names[idx] for idx in top_indices]
                
                topics[f"topic_{i}"] = {
                    'keywords': top_words,
                    'question_count': np.sum(cluster_labels == i)
                }
            
            return {"topics": topics, "clustering_quality": "basic"}
            
        except Exception as e:
            logger.error(f"Erreur clustering topics: {e}")
            return {"topics": "clustering_failed", "error": str(e)}


class VLMChatbotAdvancedFeatures:
    """Gestionnaire des fonctionnalités avancées du chatbot VLM."""
    
    def __init__(self):
        self.conversation_contexts: Dict[str, ConversationContext] = {}
        self.smart_alerts: List[SmartAlert] = []
        self.recommendation_engine = ProactiveRecommendationEngine()
        self.analytics = ConversationalAnalytics()
        self.trend_analyzer = TrendAnalyzer()
        
        # Configuration features
        self.features_enabled = {
            'proactive_recommendations': True,
            'contextual_memory': True,
            'trend_analysis': True,
            'smart_alerts': True,
            'conversation_analytics': True
        }
    
    def initialize_conversation_context(
        self, 
        session_id: str,
        user_intent: str = "general",
        expertise_level: str = "intermediate"
    ) -> ConversationContext:
        """Initialise un contexte de conversation."""
        
        context = ConversationContext(
            session_id=session_id,
            user_intent=user_intent,
            conversation_mode=ConversationMode.CONTEXTUAL,
            topics_discussed=[],
            context_memory={},
            expertise_level=expertise_level,
            preferred_detail_level="standard",
            last_interaction=datetime.now()
        )
        
        self.conversation_contexts[session_id] = context
        return context
    
    def update_conversation_context(
        self, 
        session_id: str,
        question: str,
        response: Dict[str, Any],
        topics: List[str] = None
    ):
        """Met à jour le contexte conversationnel."""
        
        if session_id not in self.conversation_contexts:
            self.initialize_conversation_context(session_id)
        
        context = self.conversation_contexts[session_id]
        context.last_interaction = datetime.now()
        
        if topics:
            context.topics_discussed.extend(topics)
            context.topics_discussed = list(set(context.topics_discussed))  # Déduplique
        
        # Mémorisation éléments clés
        if response.get('confidence', 0) > 0.8:
            context.context_memory[question[:50]] = {
                'response_summary': response.get('response', '')[:100],
                'key_insights': response.get('technical_details', '')[:100],
                'timestamp': datetime.now().isoformat()
            }
        
        # Nettoyage mémoire ancienne
        if len(context.context_memory) > 20:
            oldest_keys = list(context.context_memory.keys())[:5]
            for key in oldest_keys:
                del context.context_memory[key]
    
    def generate_contextual_prompt_enhancement(
        self, 
        session_id: str,
        base_prompt: str
    ) -> str:
        """Améliore le prompt avec le contexte conversationnel."""
        
        if session_id not in self.conversation_contexts:
            return base_prompt
        
        context = self.conversation_contexts[session_id]
        
        enhancement = f"""
CONTEXTE CONVERSATIONNEL:
- Session ID: {session_id}
- Niveau expertise utilisateur: {context.expertise_level}
- Niveau détail préféré: {context.preferred_detail_level}
- Topics discutés: {', '.join(context.topics_discussed[-5:])}
- Mémoire conversation: {len(context.context_memory)} éléments mémorisés

INSTRUCTIONS ADAPTATIVES:
- Adapter niveau technique selon expertise: {context.expertise_level}
- Référencer conversations précédentes si pertinent
- Maintenir cohérence avec topics déjà abordés
- Personnaliser selon préférences utilisateur
"""
        
        return base_prompt + enhancement
    
    def analyze_and_generate_insights(
        self, 
        current_stats: Dict[str, Any],
        recent_detections: List[Any],
        optimization_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Génère insights avancés basés sur l'analyse des données."""
        
        insights = {
            'timestamp': datetime.now().isoformat(),
            'analysis_type': 'comprehensive_insights'
        }
        
        # Analyse tendances
        if self.features_enabled['trend_analysis']:
            self.trend_analyzer.add_performance_data(current_stats)
            trend_analysis = self.trend_analyzer.analyze_performance_trend()
            insights['trend_analysis'] = trend_analysis
            
            # Prédictions
            predictions = self.trend_analyzer.predict_next_performance()
            insights['performance_prediction'] = predictions
        
        # Recommandations proactives
        if self.features_enabled['proactive_recommendations']:
            recommendations = self.recommendation_engine.generate_proactive_recommendations(
                current_stats, recent_detections, optimization_results
            )
            insights['proactive_recommendations'] = recommendations
        
        # Analytics conversationnelles
        if self.features_enabled['conversation_analytics']:
            conversation_patterns = self.analytics.analyze_conversation_patterns()
            insights['conversation_analytics'] = conversation_patterns
        
        return insights
    
    def get_feature_status(self) -> Dict[str, Any]:
        """État des fonctionnalités avancées."""
        
        return {
            'features_enabled': self.features_enabled,
            'active_contexts': len(self.conversation_contexts),
            'smart_alerts': len([a for a in self.smart_alerts if not a.resolved]),
            'conversation_history': len(self.analytics.conversations),
            'trend_data_points': len(self.trend_analyzer.performance_history)
        }


# Instance globale features avancées
_advanced_features = None

def get_advanced_features() -> VLMChatbotAdvancedFeatures:
    """Récupère l'instance des fonctionnalités avancées."""
    global _advanced_features
    if _advanced_features is None:
        _advanced_features = VLMChatbotAdvancedFeatures()
    return _advanced_features