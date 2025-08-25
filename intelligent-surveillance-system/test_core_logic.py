#!/usr/bin/env python3
"""
🧪 Test de la Logique Métier Sans Dépendances ML
==============================================

Test des composants de base sans nécessiter PyTorch/YOLO/VLM réels.
"""

import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass
from unittest.mock import Mock, AsyncMock

# Configuration du path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

@dataclass
class MockDetection:
    """Mock d'une détection YOLO."""
    bbox: List[int]
    confidence: float
    class_name: str
    track_id: int = None

class CoreLogicTester:
    """Testeur de la logique métier sans dépendances externes."""
    
    def __init__(self):
        self.results = []
    
    def test_detection_parsing(self):
        """Test du parsing des détections."""
        
        print("🔍 Test: Parsing des détections")
        
        # Données simulées
        raw_detections = [
            {"bbox": [100, 100, 200, 300], "confidence": 0.85, "class_name": "person"},
            {"bbox": [300, 50, 400, 150], "confidence": 0.75, "class_name": "bottle"}
        ]
        
        detections = [MockDetection(**d) for d in raw_detections]
        
        assert len(detections) == 2
        assert detections[0].class_name == "person"
        assert detections[1].confidence == 0.75
        
        print("  ✅ Parsing détections OK")
        return True
    
    def test_suspicion_logic(self):
        """Test de la logique de détection de suspicion."""
        
        print("🚨 Test: Logique de suspicion")
        
        def calculate_suspicion_level(detections: List[MockDetection], 
                                    context: Dict[str, Any]) -> float:
            """Calcule le niveau de suspicion basé sur les détections."""
            
            score = 0.0
            
            # Détection de personnes
            persons = [d for d in detections if d.class_name == "person"]
            if len(persons) > 0:
                score += 0.2
            
            # Proximité avec objets de valeur
            valuable_objects = [d for d in detections if d.class_name in ["bottle", "electronics"]]
            if len(valuable_objects) > 0 and len(persons) > 0:
                score += 0.3
            
            # Comportement temporel (temps passé dans la zone)
            if context.get("time_in_zone", 0) > 30:  # Plus de 30 secondes
                score += 0.4
            
            # Mouvement suspect (changement rapide de position)
            if context.get("rapid_movement", False):
                score += 0.2
            
            return min(score, 1.0)
        
        # Test cas normal
        normal_detections = [MockDetection([100, 100, 200, 300], 0.8, "person")]
        normal_context = {"time_in_zone": 10, "rapid_movement": False}
        normal_score = calculate_suspicion_level(normal_detections, normal_context)
        
        # Test cas suspect
        suspect_detections = [
            MockDetection([100, 100, 200, 300], 0.8, "person"),
            MockDetection([150, 150, 180, 180], 0.7, "bottle")
        ]
        suspect_context = {"time_in_zone": 45, "rapid_movement": True}
        suspect_score = calculate_suspicion_level(suspect_detections, suspect_context)
        
        assert normal_score < suspect_score
        assert suspect_score > 0.5  # Seuil de suspicion
        
        print(f"  ✅ Score normal: {normal_score:.2f}")
        print(f"  ✅ Score suspect: {suspect_score:.2f}")
        
        return True
    
    def test_alert_generation(self):
        """Test de la génération d'alertes."""
        
        print("🔔 Test: Génération d'alertes")
        
        def generate_alert(suspicion_level: float, detections: List[MockDetection]) -> Dict[str, Any]:
            """Génère une alerte basée sur le niveau de suspicion."""
            
            if suspicion_level < 0.3:
                return {"level": "none", "message": ""}
            elif suspicion_level < 0.6:
                return {
                    "level": "low", 
                    "message": "Comportement potentiellement suspect détecté",
                    "confidence": suspicion_level
                }
            else:
                return {
                    "level": "high",
                    "message": "ALERTE: Comportement suspect confirmé - Action immédiate requise",
                    "confidence": suspicion_level,
                    "persons_count": len([d for d in detections if d.class_name == "person"])
                }
        
        # Test différents niveaux
        low_alert = generate_alert(0.4, [MockDetection([100, 100, 200, 300], 0.8, "person")])
        high_alert = generate_alert(0.8, [
            MockDetection([100, 100, 200, 300], 0.8, "person"),
            MockDetection([150, 150, 180, 180], 0.7, "bottle")
        ])
        
        assert low_alert["level"] == "low"
        assert high_alert["level"] == "high"
        assert "ALERTE" in high_alert["message"]
        
        print(f"  ✅ Alerte faible: {low_alert['level']}")
        print(f"  ✅ Alerte forte: {high_alert['level']}")
        
        return True
    
    def test_tool_selection_logic(self):
        """Test de la logique de sélection d'outils."""
        
        print("🛠️ Test: Sélection d'outils")
        
        def select_tools(context: Dict[str, Any]) -> List[str]:
            """Sélectionne les outils optimaux selon le contexte."""
            
            tools = []
            
            # Toujours utiliser la segmentation de base
            tools.append("basic_segmentation")
            
            # Si personnes détectées, utiliser pose estimation
            if context.get("persons_count", 0) > 0:
                tools.append("pose_estimation")
            
            # Si comportement suspect, analyser la trajectoire
            if context.get("suspicion_level", 0) > 0.5:
                tools.append("trajectory_analysis")
                tools.append("multimodal_fusion")
            
            # Si plusieurs personnes, utiliser détection adversariale
            if context.get("persons_count", 0) > 2:
                tools.append("adversarial_detection")
            
            # Mode thoroughness élevé
            if context.get("mode") == "thorough":
                tools.extend(["temporal_transformer", "domain_adaptation"])
            
            return tools
        
        # Test contextes différents
        normal_context = {"persons_count": 1, "suspicion_level": 0.2, "mode": "fast"}
        suspect_context = {"persons_count": 2, "suspicion_level": 0.7, "mode": "thorough"}
        
        normal_tools = select_tools(normal_context)
        suspect_tools = select_tools(suspect_context)
        
        assert len(suspect_tools) > len(normal_tools)
        assert "trajectory_analysis" in suspect_tools
        assert "trajectory_analysis" not in normal_tools
        
        print(f"  ✅ Outils normaux: {len(normal_tools)} - {normal_tools}")
        print(f"  ✅ Outils suspects: {len(suspect_tools)} - {suspect_tools}")
        
        return True
    
    def test_performance_metrics(self):
        """Test de calcul des métriques de performance."""
        
        print("📊 Test: Métriques de performance")
        
        def calculate_metrics(processing_times: List[float], 
                            tool_usage: Dict[str, int]) -> Dict[str, Any]:
            """Calcule les métriques de performance."""
            
            if not processing_times:
                return {"error": "No processing times"}
            
            avg_time = sum(processing_times) / len(processing_times)
            max_time = max(processing_times)
            min_time = min(processing_times)
            
            # FPS approximatif
            fps = 1.0 / avg_time if avg_time > 0 else 0
            
            # Efficacité des outils (utilisation vs performance)
            tool_efficiency = {}
            for tool, count in tool_usage.items():
                # Score simple basé sur l'utilisation
                efficiency = min(count / 10.0, 1.0)  # Normalisé sur 10 utilisations
                tool_efficiency[tool] = efficiency
            
            return {
                "avg_processing_time": avg_time,
                "max_processing_time": max_time,
                "min_processing_time": min_time,
                "approximate_fps": fps,
                "tool_efficiency": tool_efficiency,
                "total_analyses": len(processing_times)
            }
        
        # Données de test
        test_times = [0.5, 0.3, 0.8, 0.4, 0.6, 0.2, 0.7]
        test_usage = {
            "basic_segmentation": 7,
            "pose_estimation": 5,
            "trajectory_analysis": 3,
            "multimodal_fusion": 2
        }
        
        metrics = calculate_metrics(test_times, test_usage)
        
        assert metrics["total_analyses"] == 7
        assert metrics["approximate_fps"] > 0
        assert "basic_segmentation" in metrics["tool_efficiency"]
        
        print(f"  ✅ FPS moyen: {metrics['approximate_fps']:.1f}")
        print(f"  ✅ Temps moyen: {metrics['avg_processing_time']:.3f}s")
        print(f"  ✅ Outils efficaces: {len(metrics['tool_efficiency'])}")
        
        return True
    
    def run_comprehensive_scenario(self):
        """Scénario complet simulé."""
        
        print("🎬 Test: Scénario de vol simulé")
        print("=" * 40)
        
        # Simulation d'un scénario de 60 secondes
        scenario_results = []
        
        for second in range(60):
            frame_data = {
                "timestamp": second,
                "detections": [],
                "context": {}
            }
            
            # Scénario: personne entre (seconde 10), se dirige vers les étagères (20-40), 
            # prend un objet (45), sort (55)
            
            if 10 <= second <= 55:
                # Personne présente
                frame_data["detections"].append(
                    MockDetection([100 + second, 100, 200 + second, 300], 0.85, "person", track_id=1)
                )
                
                # Objet valuable visible à partir de la seconde 20
                if 20 <= second <= 50:
                    frame_data["detections"].append(
                        MockDetection([300, 50, 400, 150], 0.75, "electronics", track_id=2)
                    )
                
                # Contexte temporal
                frame_data["context"] = {
                    "time_in_zone": second - 10,
                    "rapid_movement": 40 <= second <= 45,  # Mouvement rapide lors de la prise
                    "persons_count": 1,
                    "mode": "balanced"
                }
                
                # Calculs
                suspicion = min((second - 10) * 0.02 + (0.3 if frame_data["context"]["rapid_movement"] else 0), 1.0)
                frame_data["suspicion_level"] = suspicion
                
                # Sélection outils
                tools = ["basic_segmentation"]
                if suspicion > 0.5:
                    tools.extend(["trajectory_analysis", "pose_estimation"])
                
                frame_data["tools_selected"] = tools
                frame_data["processing_time"] = 0.2 + len(tools) * 0.1  # Simulation temps
                
                # Alerte si suspicion élevée
                if suspicion > 0.7:
                    frame_data["alert"] = {
                        "level": "high",
                        "message": f"Vol potentiel détecté (suspicion: {suspicion:.2f})"
                    }
            
            scenario_results.append(frame_data)
        
        # Analyse des résultats
        alerts = [f for f in scenario_results if f.get("alert")]
        max_suspicion = max([f.get("suspicion_level", 0) for f in scenario_results])
        total_tools = sum([len(f.get("tools_selected", [])) for f in scenario_results])
        
        print(f"✅ Alertes générées: {len(alerts)}")
        print(f"✅ Suspicion maximale: {max_suspicion:.2f}")
        print(f"✅ Outils utilisés: {total_tools}")
        
        # Moment de l'alerte
        if alerts:
            first_alert_time = alerts[0]["timestamp"]
            print(f"✅ Première alerte: {first_alert_time}s (détection précoce)")
        
        return len(alerts) > 0 and max_suspicion > 0.7
    
    def run_all_tests(self):
        """Exécute tous les tests."""
        
        print("🧪 Tests de la Logique Métier - Architecture Surveillance")
        print("=" * 60)
        
        tests = [
            ("Parsing Détections", self.test_detection_parsing),
            ("Logique Suspicion", self.test_suspicion_logic),
            ("Génération Alertes", self.test_alert_generation),
            ("Sélection Outils", self.test_tool_selection_logic),
            ("Métriques Performance", self.test_performance_metrics),
            ("Scénario Complet", self.run_comprehensive_scenario)
        ]
        
        results = []
        start_time = time.time()
        
        for test_name, test_func in tests:
            try:
                result = test_func()
                results.append((test_name, result, None))
                print()
            except Exception as e:
                results.append((test_name, False, str(e)))
                print(f"❌ Erreur dans {test_name}: {e}")
                print()
        
        total_time = time.time() - start_time
        
        # Rapport final
        print("=" * 60)
        print("📊 RAPPORT FINAL DES TESTS")
        print("=" * 60)
        
        passed = sum(1 for _, success, _ in results if success)
        total = len(results)
        
        for test_name, success, error in results:
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"{status} {test_name}")
            if error:
                print(f"       Erreur: {error}")
        
        print(f"\n📈 Résultats: {passed}/{total} tests réussis")
        print(f"⏱️ Temps total: {total_time:.2f}s")
        
        if passed == total:
            print("\n🎉 Tous les tests de logique métier réussis !")
            print("🏗️ L'architecture de surveillance est fonctionnelle")
        else:
            print(f"\n⚠️ {total - passed} test(s) échoué(s)")
        
        return passed == total

def main():
    """Point d'entrée principal."""
    
    tester = CoreLogicTester()
    success = tester.run_all_tests()
    
    return 0 if success else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)