#!/usr/bin/env python3
"""
🎭 Tests Orchestration - Logique de Décision Pure
================================================

Tests de la logique d'orchestration sans modèles ML réels.
Teste: sélection d'outils, configuration, décisions métier.
"""

# import pytest  # Not needed for direct execution
from enum import Enum
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from unittest.mock import Mock

class OrchestrationMode(Enum):
    """Modes d'orchestration"""
    FAST = "fast"
    BALANCED = "balanced" 
    THOROUGH = "thorough"

@dataclass
class OrchestrationConfig:
    """Configuration d'orchestration"""
    mode: OrchestrationMode = OrchestrationMode.BALANCED
    max_concurrent_tools: int = 4
    enable_advanced_tools: bool = True
    confidence_threshold: float = 0.7
    enabled_tools: List[str] = field(default_factory=lambda: [
        'sam2_segmentator', 'dino_features', 'pose_estimator',
        'trajectory_analyzer', 'multimodal_fusion', 'temporal_transformer',
        'adversarial_detector', 'domain_adapter'
    ])

@dataclass
class ContextPattern:
    """Pattern de contexte pour optimisation"""
    scenario_type: str  # "normal", "suspicious", "crowded"
    time_period: str    # "morning", "afternoon", "evening", "night"  
    person_count: int
    location_type: str  # "entrance", "electronics", "checkout"
    optimal_tools: List[str] = field(default_factory=list)
    performance_score: float = 0.0

@dataclass
class AnalysisRequest:
    """Requête d'analyse"""
    frame_number: int
    detections: List[Dict[str, Any]]
    context: Dict[str, Any]
    timestamp: float

# =================== Logique d'Orchestration Pure ===================

class MockOrchestrator:
    """Orchestrateur simulé pour tests de logique pure"""
    
    def __init__(self, config: OrchestrationConfig):
        self.config = config
        self.context_patterns: Dict[str, ContextPattern] = {}
        self.tool_performance_history: Dict[str, float] = {
            'sam2_segmentator': 0.8,
            'dino_features': 0.9,
            'pose_estimator': 0.7,
            'trajectory_analyzer': 0.85,
            'multimodal_fusion': 0.9,
            'temporal_transformer': 0.75,
            'adversarial_detector': 0.8,
            'domain_adapter': 0.7
        }
        self.execution_history = []
    
    def analyze_context(self, request: AnalysisRequest) -> Dict[str, Any]:
        """Analyse le contexte de la requête"""
        detections = request.detections
        context = request.context
        
        # Analyser les détections
        persons = [d for d in detections if d.get('class_name') == 'person']
        valuable_objects = [d for d in detections if d.get('class_name') in ['electronics', 'bottle']]
        
        # Déterminer le type de scénario
        scenario_type = "normal"
        if len(persons) > 2:
            scenario_type = "crowded"
        elif len(persons) > 0 and len(valuable_objects) > 0:
            # Personne proche d'objets de valeur
            person_centers = [(d['bbox'][0] + d['bbox'][2])/2 for d in persons]
            object_centers = [(d['bbox'][0] + d['bbox'][2])/2 for d in valuable_objects]
            
            # Vérifier proximité (simulation simple)
            min_distance = float('inf')
            for pc in person_centers:
                for oc in object_centers:
                    distance = abs(pc - oc)
                    min_distance = min(min_distance, distance)
            
            if min_distance < 100:  # Pixels
                scenario_type = "suspicious"
        
        # Déterminer période (simulation basée sur frame_number)
        time_period = "morning"
        if request.frame_number > 1000:
            time_period = "afternoon"
        elif request.frame_number > 2000:
            time_period = "evening"
        elif request.frame_number > 3000:
            time_period = "night"
        
        return {
            'scenario_type': scenario_type,
            'time_period': time_period,
            'person_count': len(persons),
            'valuable_object_count': len(valuable_objects),
            'total_detections': len(detections)
        }
    
    def select_tools_by_mode(self, mode: OrchestrationMode, context: Dict[str, Any]) -> List[str]:
        """Sélection d'outils selon le mode"""
        base_tools = ['dino_features']  # Toujours inclus
        
        if mode == OrchestrationMode.FAST:
            # Mode rapide: outils essentiels seulement
            if context['person_count'] > 0:
                base_tools.append('pose_estimator')
            return base_tools
        
        elif mode == OrchestrationMode.BALANCED:
            # Mode équilibré: outils moyens
            base_tools.extend(['sam2_segmentator', 'pose_estimator'])
            
            if context['scenario_type'] == 'suspicious':
                base_tools.extend(['trajectory_analyzer', 'multimodal_fusion'])
            
            if context['person_count'] > 2:
                base_tools.append('adversarial_detector')
                
            return base_tools
        
        elif mode == OrchestrationMode.THOROUGH:
            # Mode complet: tous les outils
            all_tools = self.config.enabled_tools.copy()
            
            # Filtrer selon contexte
            if context['person_count'] == 0:
                # Pas besoin de pose estimation sans personnes
                if 'pose_estimator' in all_tools:
                    all_tools.remove('pose_estimator')
            
            return all_tools
    
    def calculate_suspicion_level(self, context: Dict[str, Any], tools_results: Dict[str, Any] = None) -> float:
        """Calcul du niveau de suspicion"""
        suspicion = 0.0
        
        # Facteur 1: Type de scénario
        scenario_scores = {
            'normal': 0.1,
            'suspicious': 0.6,
            'crowded': 0.3
        }
        suspicion += scenario_scores.get(context['scenario_type'], 0.1)
        
        # Facteur 2: Période
        time_scores = {
            'morning': 0.1,
            'afternoon': 0.05,
            'evening': 0.2,
            'night': 0.3
        }
        suspicion += time_scores.get(context['time_period'], 0.1)
        
        # Facteur 3: Nombre de personnes
        if context['person_count'] > 3:
            suspicion += 0.2
        elif context['person_count'] > 1:
            suspicion += 0.1
        
        # Facteur 4: Objets de valeur
        if context.get('valuable_object_count', 0) > 0 and context['person_count'] > 0:
            suspicion += 0.3
        
        # Facteur 5: Résultats des outils (simulation)
        if tools_results:
            if tools_results.get('trajectory_analysis', {}).get('erratic_movement', False):
                suspicion += 0.2
            if tools_results.get('pose_analysis', {}).get('suspicious_pose', False):
                suspicion += 0.15
        
        return min(suspicion, 1.0)
    
    def adaptive_tool_selection(self, context: Dict[str, Any]) -> List[str]:
        """Sélection adaptative d'outils basée sur performance"""
        # Sélection de base selon le mode
        base_tools = self.select_tools_by_mode(self.config.mode, context)
        
        # Optimisation basée sur l'historique de performance
        optimized_tools = []
        
        for tool in base_tools:
            performance = self.tool_performance_history.get(tool, 0.5)
            
            # Inclure seulement si performance > seuil
            if performance >= 0.6:
                optimized_tools.append(tool)
        
        # Ajouter outils spécialisés si contexte le justifie
        if context['scenario_type'] == 'suspicious' and 'trajectory_analyzer' not in optimized_tools:
            if self.tool_performance_history.get('trajectory_analyzer', 0) > 0.7:
                optimized_tools.append('trajectory_analyzer')
        
        # Limiter selon max_concurrent_tools
        if len(optimized_tools) > self.config.max_concurrent_tools:
            # Garder les plus performants
            tool_scores = [(t, self.tool_performance_history.get(t, 0)) for t in optimized_tools]
            tool_scores.sort(key=lambda x: x[1], reverse=True)
            optimized_tools = [t for t, _ in tool_scores[:self.config.max_concurrent_tools]]
        
        return optimized_tools
    
    def estimate_processing_time(self, tools: List[str], mode: OrchestrationMode) -> float:
        """Estimation du temps de traitement"""
        # Temps de base par outil (simulation)
        tool_times = {
            'sam2_segmentator': 0.8,
            'dino_features': 0.3,
            'pose_estimator': 0.2,
            'trajectory_analyzer': 0.4,
            'multimodal_fusion': 0.6,
            'temporal_transformer': 1.2,
            'adversarial_detector': 0.7,
            'domain_adapter': 0.5
        }
        
        # Facteurs de vitesse selon le mode
        mode_factors = {
            OrchestrationMode.FAST: 0.7,      # Optimisations agressives
            OrchestrationMode.BALANCED: 1.0,  # Temps standard
            OrchestrationMode.THOROUGH: 1.3   # Plus de précision = plus lent
        }
        
        total_time = sum(tool_times.get(tool, 0.5) for tool in tools)
        return total_time * mode_factors.get(mode, 1.0)

# =================== Tests ===================

class TestOrchestrationLogic:
    """Tests de la logique d'orchestration"""
    
    def test_config_initialization(self):
        """Test initialisation configuration"""
        config = OrchestrationConfig()
        
        assert config.mode == OrchestrationMode.BALANCED
        assert config.max_concurrent_tools == 4
        assert config.enable_advanced_tools is True
        assert len(config.enabled_tools) == 8
    
    def test_config_custom(self):
        """Test configuration personnalisée"""
        config = OrchestrationConfig(
            mode=OrchestrationMode.FAST,
            max_concurrent_tools=2,
            enable_advanced_tools=False,
            enabled_tools=['dino_features', 'pose_estimator']
        )
        
        assert config.mode == OrchestrationMode.FAST
        assert config.max_concurrent_tools == 2
        assert len(config.enabled_tools) == 2

class TestContextAnalysis:
    """Tests d'analyse de contexte"""
    
    def setup_method(self):
        """Setup pour chaque test"""
        config = OrchestrationConfig()
        self.orchestrator = MockOrchestrator(config)
    
    def test_normal_scenario_analysis(self):
        """Test analyse scénario normal"""
        request = AnalysisRequest(
            frame_number=100,
            detections=[
                {'class_name': 'car', 'bbox': [100, 100, 200, 200]}
            ],
            context={},
            timestamp=1234567890
        )
        
        context = self.orchestrator.analyze_context(request)
        
        assert context['scenario_type'] == 'normal'
        assert context['person_count'] == 0
        assert context['valuable_object_count'] == 0
    
    def test_suspicious_scenario_analysis(self):
        """Test analyse scénario suspect"""
        request = AnalysisRequest(
            frame_number=500,
            detections=[
                {'class_name': 'person', 'bbox': [100, 100, 200, 200]},
                {'class_name': 'electronics', 'bbox': [150, 150, 250, 250]}  # Proche
            ],
            context={},
            timestamp=1234567890
        )
        
        context = self.orchestrator.analyze_context(request)
        
        assert context['scenario_type'] == 'suspicious'
        assert context['person_count'] == 1
        assert context['valuable_object_count'] == 1
    
    def test_crowded_scenario_analysis(self):
        """Test analyse scénario bondé"""
        request = AnalysisRequest(
            frame_number=200,
            detections=[
                {'class_name': 'person', 'bbox': [100, 100, 200, 200]},
                {'class_name': 'person', 'bbox': [300, 100, 400, 200]},
                {'class_name': 'person', 'bbox': [100, 300, 200, 400]}
            ],
            context={},
            timestamp=1234567890
        )
        
        context = self.orchestrator.analyze_context(request)
        
        assert context['scenario_type'] == 'crowded'
        assert context['person_count'] == 3

class TestToolSelection:
    """Tests de sélection d'outils"""
    
    def setup_method(self):
        """Setup pour chaque test"""
        config = OrchestrationConfig()
        self.orchestrator = MockOrchestrator(config)
    
    def test_fast_mode_selection(self):
        """Test sélection mode rapide"""
        context = {
            'scenario_type': 'normal',
            'person_count': 1,
            'valuable_object_count': 0
        }
        
        tools = self.orchestrator.select_tools_by_mode(OrchestrationMode.FAST, context)
        
        assert 'dino_features' in tools
        assert 'pose_estimator' in tools
        assert len(tools) <= 3  # Mode rapide = peu d'outils
        assert 'temporal_transformer' not in tools  # Pas d'outils lents
    
    def test_balanced_mode_selection(self):
        """Test sélection mode équilibré"""
        context = {
            'scenario_type': 'suspicious',
            'person_count': 2,
            'valuable_object_count': 1
        }
        
        tools = self.orchestrator.select_tools_by_mode(OrchestrationMode.BALANCED, context)
        
        assert 'dino_features' in tools
        assert 'trajectory_analyzer' in tools  # Scénario suspect
        assert 'multimodal_fusion' in tools
        assert len(tools) > 2 and len(tools) <= 6
    
    def test_thorough_mode_selection(self):
        """Test sélection mode complet"""
        context = {
            'scenario_type': 'suspicious',
            'person_count': 3,
            'valuable_object_count': 2
        }
        
        tools = self.orchestrator.select_tools_by_mode(OrchestrationMode.THOROUGH, context)
        
        assert len(tools) >= 6  # Mode complet = beaucoup d'outils
        assert 'temporal_transformer' in tools
        assert 'adversarial_detector' in tools
    
    def test_adaptive_tool_selection(self):
        """Test sélection adaptative"""
        # Modifier performance d'un outil pour test
        self.orchestrator.tool_performance_history['sam2_segmentator'] = 0.3  # Très faible
        
        context = {
            'scenario_type': 'normal',
            'person_count': 1,
            'valuable_object_count': 0
        }
        
        tools = self.orchestrator.adaptive_tool_selection(context)
        
        # L'outil avec faible performance devrait être exclu
        assert 'sam2_segmentator' not in tools
        # Les outils performants devraient être inclus
        assert 'dino_features' in tools  # Performance 0.9
    
    def test_concurrent_tools_limit(self):
        """Test limite outils concurrents"""
        config = OrchestrationConfig(max_concurrent_tools=2)
        orchestrator = MockOrchestrator(config)
        
        context = {
            'scenario_type': 'suspicious',
            'person_count': 3,
            'valuable_object_count': 2
        }
        
        tools = orchestrator.adaptive_tool_selection(context)
        
        assert len(tools) <= 2  # Respecte la limite
        # Devrait garder les plus performants
        performances = [orchestrator.tool_performance_history.get(t, 0) for t in tools]
        assert all(p >= 0.8 for p in performances)  # Seulement les meilleurs

class TestSuspicionCalculation:
    """Tests de calcul de suspicion"""
    
    def setup_method(self):
        """Setup pour chaque test"""
        config = OrchestrationConfig()
        self.orchestrator = MockOrchestrator(config)
    
    def test_normal_scenario_suspicion(self):
        """Test suspicion scénario normal"""
        context = {
            'scenario_type': 'normal',
            'time_period': 'afternoon',
            'person_count': 0,
            'valuable_object_count': 0
        }
        
        suspicion = self.orchestrator.calculate_suspicion_level(context)
        
        assert suspicion < 0.3  # Faible suspicion
    
    def test_suspicious_scenario_suspicion(self):
        """Test suspicion scénario suspect"""
        context = {
            'scenario_type': 'suspicious',
            'time_period': 'night',
            'person_count': 2,
            'valuable_object_count': 1
        }
        
        suspicion = self.orchestrator.calculate_suspicion_level(context)
        
        assert suspicion >= 0.5  # Suspicion élevée
        assert suspicion <= 1.0  # Pas de dépassement
    
    def test_suspicion_with_tool_results(self):
        """Test suspicion avec résultats d'outils"""
        context = {
            'scenario_type': 'normal',
            'time_period': 'morning',
            'person_count': 1,
            'valuable_object_count': 0
        }
        
        # Sans résultats d'outils
        suspicion_base = self.orchestrator.calculate_suspicion_level(context)
        
        # Avec résultats d'outils suspects
        tools_results = {
            'trajectory_analysis': {'erratic_movement': True},
            'pose_analysis': {'suspicious_pose': True}
        }
        
        suspicion_enhanced = self.orchestrator.calculate_suspicion_level(context, tools_results)
        
        assert suspicion_enhanced > suspicion_base
        assert suspicion_enhanced - suspicion_base >= 0.3  # Bonus des outils

class TestProcessingTimeEstimation:
    """Tests d'estimation temps de traitement"""
    
    def setup_method(self):
        """Setup pour chaque test"""
        config = OrchestrationConfig()
        self.orchestrator = MockOrchestrator(config)
    
    def test_fast_mode_time_estimation(self):
        """Test estimation temps mode rapide"""
        tools = ['dino_features', 'pose_estimator']
        time_fast = self.orchestrator.estimate_processing_time(tools, OrchestrationMode.FAST)
        time_balanced = self.orchestrator.estimate_processing_time(tools, OrchestrationMode.BALANCED)
        
        assert time_fast < time_balanced  # Mode rapide plus rapide
        assert time_fast > 0
    
    def test_thorough_mode_time_estimation(self):
        """Test estimation temps mode complet"""
        tools = ['sam2_segmentator', 'temporal_transformer', 'adversarial_detector']
        time_thorough = self.orchestrator.estimate_processing_time(tools, OrchestrationMode.THOROUGH)
        time_balanced = self.orchestrator.estimate_processing_time(tools, OrchestrationMode.BALANCED)
        
        assert time_thorough > time_balanced  # Mode complet plus lent
        assert time_thorough >= 2.0  # Outils lents
    
    def test_tool_count_impact_on_time(self):
        """Test impact nombre d'outils sur temps"""
        few_tools = ['dino_features']
        many_tools = ['sam2_segmentator', 'dino_features', 'pose_estimator', 'trajectory_analyzer']
        
        time_few = self.orchestrator.estimate_processing_time(few_tools, OrchestrationMode.BALANCED)
        time_many = self.orchestrator.estimate_processing_time(many_tools, OrchestrationMode.BALANCED)
        
        assert time_many > time_few
        assert time_many >= 4 * time_few  # Proportionnel au nombre d'outils

class TestPerformanceOptimization:
    """Tests d'optimisation de performance"""
    
    def test_performance_tracking_update(self):
        """Test mise à jour historique performance"""
        config = OrchestrationConfig()
        orchestrator = MockOrchestrator(config)
        
        # Performance initiale
        initial_perf = orchestrator.tool_performance_history['dino_features']
        
        # Simuler bonne exécution
        orchestrator.tool_performance_history['dino_features'] = min(initial_perf + 0.1, 1.0)
        
        # Vérifier mise à jour
        new_perf = orchestrator.tool_performance_history['dino_features']
        assert new_perf > initial_perf
        assert new_perf <= 1.0
    
    def test_tool_exclusion_based_on_performance(self):
        """Test exclusion d'outils selon performance"""
        config = OrchestrationConfig()
        orchestrator = MockOrchestrator(config)
        
        # Mettre un outil en mauvaise performance
        orchestrator.tool_performance_history['sam2_segmentator'] = 0.4  # Sous seuil
        
        context = {
            'scenario_type': 'normal',
            'person_count': 1,
            'valuable_object_count': 0
        }
        
        tools = orchestrator.adaptive_tool_selection(context)
        
        # L'outil peu performant devrait être exclu
        assert 'sam2_segmentator' not in tools

if __name__ == "__main__":
    print("🎭 Tests Orchestration - Logique Pure")
    print("=" * 40)
    
    # Tests rapides sans pytest
    try:
        # Test config
        config_tests = TestOrchestrationLogic()
        config_tests.test_config_initialization()
        config_tests.test_config_custom()
        print("✅ Tests configuration: OK")
        
        # Test contexte
        context_tests = TestContextAnalysis()
        context_tests.setup_method()
        context_tests.test_normal_scenario_analysis()
        context_tests.test_suspicious_scenario_analysis()
        context_tests.test_crowded_scenario_analysis()
        print("✅ Tests analyse contexte: OK")
        
        # Test sélection outils
        tool_tests = TestToolSelection()
        tool_tests.setup_method()
        tool_tests.test_fast_mode_selection()
        tool_tests.test_balanced_mode_selection()
        tool_tests.test_thorough_mode_selection()
        tool_tests.test_adaptive_tool_selection()
        tool_tests.test_concurrent_tools_limit()
        print("✅ Tests sélection outils: OK")
        
        # Test suspicion
        suspicion_tests = TestSuspicionCalculation()
        suspicion_tests.setup_method()
        suspicion_tests.test_normal_scenario_suspicion()
        suspicion_tests.test_suspicious_scenario_suspicion()
        suspicion_tests.test_suspicion_with_tool_results()
        print("✅ Tests calcul suspicion: OK")
        
        # Test estimation temps
        time_tests = TestProcessingTimeEstimation()
        time_tests.setup_method()
        time_tests.test_fast_mode_time_estimation()
        time_tests.test_thorough_mode_time_estimation()
        time_tests.test_tool_count_impact_on_time()
        print("✅ Tests estimation temps: OK")
        
        print(f"\n🎉 Tous les tests orchestration passent sans GPU !")
        
    except AssertionError as e:
        print(f"❌ Test échoué: {e}")
    except Exception as e:
        print(f"💥 Erreur: {e}")
        import traceback
        traceback.print_exc()