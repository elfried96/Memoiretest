#!/usr/bin/env python3
"""
🚀 Tests d'Intégration - Modèles GPU Réels
==========================================

Tests approfondis des vrais modèles pour valider leur fonctionnement.
Complément aux tests CPU-only pour validation complète du système.
"""

import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import traceback

# Import optionnel de numpy
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    # Mock simple pour numpy
    class MockNumpy:
        @staticmethod
        def zeros(shape, dtype=None):
            if len(shape) == 3:  # Image
                return [[[0 for _ in range(shape[2])] for _ in range(shape[1])] for _ in range(shape[0])]
            return 0
        @staticmethod
        def array(data, dtype=None):
            return data
        @staticmethod
        def random():
            class MockRandom:
                @staticmethod
                def rand(*args):
                    import random
                    if args:
                        return [random.random() for _ in range(args[0])]
                    return random.random()
            return MockRandom()
        uint8 = int
        int32 = int
    np = MockNumpy()

# Import optionnel de cv2
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    # Mock simple pour cv2 si pas disponible
    class MockCV2:
        @staticmethod
        def rectangle(img, pt1, pt2, color, thickness): pass
        @staticmethod 
        def circle(img, center, radius, color, thickness): pass
        @staticmethod
        def fillPoly(img, pts, color): pass
    cv2 = MockCV2()

# Import optionnel de torch
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    class MockTorch:
        @staticmethod
        def cuda():
            class MockCuda:
                @staticmethod
                def is_available(): return False
                @staticmethod
                def get_device_properties(idx): return MockDevice()
            return MockCuda()
        
        class MockDevice:
            total_memory = 0
            
    torch = MockTorch()

# Configuration du path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

class IntegrationTestRunner:
    """Runner de tests d'intégration avec GPU"""
    
    def __init__(self):
        self.results = {
            'total_tests': 0,
            'passed_tests': 0,
            'failed_tests': 0,
            'errors': [],
            'execution_time': 0,
            'gpu_available': TORCH_AVAILABLE and torch.cuda.is_available(),
            'gpu_memory': torch.cuda.get_device_properties(0).total_memory if (TORCH_AVAILABLE and torch.cuda.is_available()) else 0
        }
        
    def create_test_image(self, width: int = 640, height: int = 480):
        """Créer une image de test"""
        # Image avec des formes géométriques pour tester la détection
        img = np.zeros((height, width, 3), dtype=np.uint8)
        
        if CV2_AVAILABLE:
            # Rectangle (simule une personne)
            cv2.rectangle(img, (100, 100), (200, 300), (255, 0, 0), -1)
            
            # Cercle (simule un objet)
            cv2.circle(img, (400, 200), 50, (0, 255, 0), -1)
            
            # Polygone (simule electronics)
            pts = np.array([[300, 50], [350, 50], [375, 100], [325, 120], [275, 100]], np.int32)
            cv2.fillPoly(img, [pts], (0, 0, 255))
        
        return img
    
    def test_yolo_real_model(self) -> Dict[str, Any]:
        """Test du modèle YOLO réel"""
        print("\n🎯 TEST YOLO RÉEL")
        print("=" * 40)
        
        test_result = {
            'name': 'YOLO Real Model',
            'passed': False,
            'execution_time': 0,
            'details': {},
            'errors': []
        }
        
        start_time = time.time()
        
        try:
            # Import du modèle YOLO
            from src.detection.yolo_detector import YOLODetector
            
            # Initialisation
            print("📥 Chargement du modèle YOLO...")
            detector = YOLODetector()
            
            # Image de test
            test_image = self.create_test_image()
            print(f"🖼️ Image de test créée: {test_image.shape}")
            
            # Test de détection
            print("🔍 Exécution détection...")
            detections = detector.detect(test_image)
            
            # Validation des résultats
            test_result['details'] = {
                'model_loaded': True,
                'detections_count': len(detections),
                'detections': detections[:3],  # Premiers résultats
                'image_processed': True,
                'gpu_used': TORCH_AVAILABLE and hasattr(detector, 'model') and torch.cuda.is_available()
            }
            
            # Critères de succès
            if len(detections) > 0:
                print(f"✅ Détections trouvées: {len(detections)}")
                for i, det in enumerate(detections[:3]):
                    print(f"   {i+1}. {det.get('class_name', 'unknown')}: {det.get('confidence', 0):.2f}")
                test_result['passed'] = True
            else:
                print("⚠️ Aucune détection trouvée")
                test_result['passed'] = False
                
        except ImportError as e:
            error_msg = f"Module YOLO non trouvé: {e}"
            print(f"❌ {error_msg}")
            test_result['errors'].append(error_msg)
        except Exception as e:
            error_msg = f"Erreur YOLO: {e}"
            print(f"❌ {error_msg}")
            test_result['errors'].append(error_msg)
            traceback.print_exc()
        
        test_result['execution_time'] = time.time() - start_time
        print(f"⏱️ Temps d'exécution: {test_result['execution_time']:.2f}s")
        
        return test_result
    
    def test_sam2_real_model(self) -> Dict[str, Any]:
        """Test du modèle SAM2 réel"""
        print("\n🎯 TEST SAM2 RÉEL")
        print("=" * 40)
        
        test_result = {
            'name': 'SAM2 Real Model',
            'passed': False,
            'execution_time': 0,
            'details': {},
            'errors': []
        }
        
        start_time = time.time()
        
        try:
            # Import du modèle SAM2
            from src.advanced_tools.sam2_segmentation import SAM2Segmentation
            
            # Initialisation
            print("📥 Chargement du modèle SAM2...")
            segmentator = SAM2Segmentation()
            
            # Image de test
            test_image = self.create_test_image()
            print(f"🖼️ Image de test créée: {test_image.shape}")
            
            # Test de segmentation
            print("🔍 Exécution segmentation...")
            segments = segmentator.segment_everything(test_image)
            
            # Validation des résultats
            test_result['details'] = {
                'model_loaded': True,
                'segments_count': len(segments.get('masks', [])),
                'total_segments': segments.get('total_segments', 0),
                'processing_time': segments.get('processing_time', 0),
                'image_processed': True
            }
            
            # Critères de succès
            if segments.get('total_segments', 0) > 0:
                print(f"✅ Segments trouvés: {segments.get('total_segments', 0)}")
                print(f"   Temps processing: {segments.get('processing_time', 0)*1000:.1f}ms")
                test_result['passed'] = True
            else:
                print("⚠️ Aucun segment trouvé")
                test_result['passed'] = False
                
        except ImportError as e:
            error_msg = f"Module SAM2 non trouvé: {e}"
            print(f"❌ {error_msg}")
            test_result['errors'].append(error_msg)
        except Exception as e:
            error_msg = f"Erreur SAM2: {e}"
            print(f"❌ {error_msg}")
            test_result['errors'].append(error_msg)
            traceback.print_exc()
        
        test_result['execution_time'] = time.time() - start_time
        print(f"⏱️ Temps d'exécution: {test_result['execution_time']:.2f}s")
        
        return test_result
    
    def test_dino_v2_real_model(self) -> Dict[str, Any]:
        """Test du modèle DINO v2 réel"""
        print("\n🎯 TEST DINO V2 RÉEL")
        print("=" * 40)
        
        test_result = {
            'name': 'DINO v2 Real Model',
            'passed': False,
            'execution_time': 0,
            'details': {},
            'errors': []
        }
        
        start_time = time.time()
        
        try:
            # Import du modèle DINO v2
            from src.advanced_tools.dino_features import DinoFeatures
            
            # Initialisation
            print("📥 Chargement du modèle DINO v2...")
            extractor = DinoFeatures()
            
            # Image de test
            test_image = self.create_test_image()
            print(f"🖼️ Image de test créée: {test_image.shape}")
            
            # Test d'extraction de features
            print("🔍 Exécution extraction features...")
            features = extractor.extract_features(test_image)
            
            # Validation des résultats
            test_result['details'] = {
                'model_loaded': True,
                'features_extracted': features.get('success', False),
                'feature_dimension': len(features.get('features', [])),
                'confidence': features.get('confidence', 0),
                'processing_time': features.get('processing_time', 0)
            }
            
            # Critères de succès
            if features.get('success', False) and len(features.get('features', [])) > 0:
                print(f"✅ Features extraites: {len(features.get('features', []))} dimensions")
                print(f"   Confiance: {features.get('confidence', 0):.2f}")
                print(f"   Temps processing: {features.get('processing_time', 0)*1000:.1f}ms")
                test_result['passed'] = True
            else:
                print("⚠️ Échec extraction features")
                test_result['passed'] = False
                
        except ImportError as e:
            error_msg = f"Module DINO v2 non trouvé: {e}"
            print(f"❌ {error_msg}")
            test_result['errors'].append(error_msg)
        except Exception as e:
            error_msg = f"Erreur DINO v2: {e}"
            print(f"❌ {error_msg}")
            test_result['errors'].append(error_msg)
            traceback.print_exc()
        
        test_result['execution_time'] = time.time() - start_time
        print(f"⏱️ Temps d'exécution: {test_result['execution_time']:.2f}s")
        
        return test_result
    
    def test_kimi_vlm_real_model(self) -> Dict[str, Any]:
        """Test du modèle Kimi VLM réel"""
        print("\n🎯 TEST KIMI VLM RÉEL")  
        print("=" * 40)
        
        test_result = {
            'name': 'Kimi VLM Real Model',
            'passed': False,
            'execution_time': 0,
            'details': {},
            'errors': []
        }
        
        start_time = time.time()
        
        try:
            # Import du modèle Kimi VLM
            from src.core.vlm.model import VLMModel
            
            # Initialisation
            print("📥 Initialisation du modèle Kimi VLM...")
            analyzer = VLMModel()
            
            # Image de test
            test_image = self.create_test_image()
            print(f"🖼️ Image de test créée: {test_image.shape}")
            
            # Test d'analyse de scène
            print("🔍 Exécution analyse de scène...")
            analysis = analyzer.analyze_scene(test_image, "person detected with objects")
            
            # Validation des résultats
            test_result['details'] = {
                'model_loaded': True,
                'analysis_completed': analysis.get('success', False),
                'suspicion_level': analysis.get('suspicion_level', 0),
                'confidence': analysis.get('confidence', 0),
                'processing_time': analysis.get('processing_time', 0),
                'description': analysis.get('description', '')[:100]  # Premiers 100 caractères
            }
            
            # Critères de succès
            if analysis.get('success', False):
                print(f"✅ Analyse complétée")
                print(f"   Niveau suspicion: {analysis.get('suspicion_level', 0):.2f}")
                print(f"   Confiance: {analysis.get('confidence', 0):.2f}")
                print(f"   Temps processing: {analysis.get('processing_time', 0)*1000:.1f}ms")
                test_result['passed'] = True
            else:
                print("⚠️ Échec analyse de scène")
                test_result['passed'] = False
                
        except ImportError as e:
            error_msg = f"Module Kimi VLM non trouvé: {e}"
            print(f"❌ {error_msg}")
            test_result['errors'].append(error_msg)
        except Exception as e:
            error_msg = f"Erreur Kimi VLM: {e}"
            print(f"❌ {error_msg}")
            test_result['errors'].append(error_msg)
            traceback.print_exc()
        
        test_result['execution_time'] = time.time() - start_time
        print(f"⏱️ Temps d'exécution: {test_result['execution_time']:.2f}s")
        
        return test_result
    
    def test_complete_pipeline(self) -> Dict[str, Any]:
        """Test du pipeline complet d'orchestration"""
        print("\n🎯 TEST PIPELINE COMPLET")
        print("=" * 40)
        
        test_result = {
            'name': 'Complete Pipeline',
            'passed': False,
            'execution_time': 0,
            'details': {},
            'errors': []
        }
        
        start_time = time.time()
        
        try:
            # Import de l'orchestrateur
            from src.core.orchestrator.adaptive_orchestrator import AdaptiveOrchestrator
            
            # Initialisation
            print("📥 Initialisation pipeline complet...")
            orchestrator = AdaptiveOrchestrator()
            
            # Image de test
            test_image = self.create_test_image()
            print(f"🖼️ Image de test créée: {test_image.shape}")
            
            # Test du pipeline complet
            print("🔍 Exécution pipeline complet...")
            result = orchestrator.analyze_frame(test_image)
            
            # Validation des résultats
            test_result['details'] = {
                'pipeline_executed': result.get('success', False),
                'tools_used': len(result.get('tools_results', {})),
                'total_detections': len(result.get('detections', [])),
                'suspicion_score': result.get('suspicion_score', 0),
                'processing_time': result.get('processing_time', 0),
                'mode_used': result.get('mode_used', 'unknown')
            }
            
            # Critères de succès
            if result.get('success', False):
                print(f"✅ Pipeline exécuté avec succès")
                print(f"   Outils utilisés: {len(result.get('tools_results', {}))}")
                print(f"   Détections: {len(result.get('detections', []))}")
                print(f"   Score suspicion: {result.get('suspicion_score', 0):.2f}")
                print(f"   Mode: {result.get('mode_used', 'unknown')}")
                print(f"   Temps total: {result.get('processing_time', 0)*1000:.1f}ms")
                test_result['passed'] = True
            else:
                print("⚠️ Échec du pipeline complet")
                test_result['passed'] = False
                
        except ImportError as e:
            error_msg = f"Module orchestrateur non trouvé: {e}"
            print(f"❌ {error_msg}")
            test_result['errors'].append(error_msg)
        except Exception as e:
            error_msg = f"Erreur pipeline: {e}"
            print(f"❌ {error_msg}")
            test_result['errors'].append(error_msg)
            traceback.print_exc()
        
        test_result['execution_time'] = time.time() - start_time
        print(f"⏱️ Temps d'exécution: {test_result['execution_time']:.2f}s")
        
        return test_result
    
    def generate_integration_report(self, test_results: List[Dict[str, Any]]):
        """Génère le rapport d'intégration"""
        print("\n" + "=" * 60)
        print("📊 RAPPORT TESTS D'INTÉGRATION GPU")
        print("=" * 60)
        
        # Statistiques GPU
        print(f"\n🖥️ ENVIRONNEMENT GPU:")
        print(f"   GPU disponible: {'✅' if self.results['gpu_available'] else '❌'}")
        if self.results['gpu_available']:
            print(f"   Mémoire GPU: {self.results['gpu_memory'] / (1024**3):.1f} GB")
            if TORCH_AVAILABLE:
                print(f"   Device: {torch.cuda.get_device_name(0)}")
        else:
            print(f"   PyTorch disponible: {'✅' if TORCH_AVAILABLE else '❌'}")
        
        # Résultats par modèle
        total_passed = sum(1 for test in test_results if test['passed'])
        total_tests = len(test_results)
        
        print(f"\n📈 RÉSULTATS GLOBAUX:")
        print(f"   Tests exécutés: {total_tests}")
        print(f"   Tests réussis: {total_passed}")
        print(f"   Tests échoués: {total_tests - total_passed}")
        print(f"   Taux de succès: {(total_passed/total_tests*100):.1f}%")
        
        print(f"\n🎯 DÉTAILS PAR MODÈLE:")
        for test in test_results:
            status = "✅" if test['passed'] else "❌"
            print(f"   {status} {test['name']}: {test['execution_time']:.2f}s")
            
            if test['details']:
                for key, value in test['details'].items():
                    if isinstance(value, bool):
                        print(f"      • {key}: {'✅' if value else '❌'}")
                    elif isinstance(value, (int, float)):
                        print(f"      • {key}: {value}")
            
            if test['errors']:
                for error in test['errors']:
                    print(f"      ⚠️ {error}")
        
        # Recommandations
        print(f"\n💡 RECOMMANDATIONS:")
        if not self.results['gpu_available']:
            print("   • Installer CUDA pour accélération GPU")
            print("   • Tests exécutés en mode CPU uniquement")
        
        failed_tests = [test for test in test_results if not test['passed']]
        if failed_tests:
            print("   • Modules manquants à installer:")
            for test in failed_tests:
                if test['errors']:
                    print(f"     - {test['name']}")
        else:
            print("   • ✅ Tous les modèles fonctionnent correctement")
            print("   • ✅ Pipeline d'intégration validé")
            print("   • ✅ Prêt pour déploiement production")
    
    def run_all_integration_tests(self):
        """Exécute tous les tests d'intégration"""
        overall_start_time = time.time()
        
        print("🚀 TESTS D'INTÉGRATION - MODÈLES GPU RÉELS")
        print("Validation fonctionnement des modèles d'IA")
        print("=" * 60)
        
        # Liste des tests à exécuter
        test_results = []
        
        # Test YOLO
        test_results.append(self.test_yolo_real_model())
        
        # Test SAM2
        test_results.append(self.test_sam2_real_model())
        
        # Test DINO v2
        test_results.append(self.test_dino_v2_real_model())
        
        # Test Kimi VLM
        test_results.append(self.test_kimi_vlm_real_model())
        
        # Test pipeline complet
        test_results.append(self.test_complete_pipeline())
        
        # Rapport final
        overall_time = time.time() - overall_start_time
        print(f"\n⏱️ Temps total d'exécution: {overall_time:.2f} secondes")
        
        self.generate_integration_report(test_results)
        
        # Retourner résultat global
        all_passed = all(test['passed'] for test in test_results)
        return all_passed

def main():
    """Point d'entrée principal"""
    print("🚀 Lancement des Tests d'Intégration GPU...")
    
    runner = IntegrationTestRunner()
    
    try:
        success = runner.run_all_integration_tests()
        
        if success:
            print(f"\n✨ Tous les tests d'intégration ont réussi!")
            return 0
        else:
            print(f"\n💥 Certains tests d'intégration ont échoué")
            return 1
            
    except KeyboardInterrupt:
        print(f"\n⚠️ Tests interrompus par l'utilisateur")
        return 130
    except Exception as e:
        print(f"\n💥 Erreur inattendue: {e}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)