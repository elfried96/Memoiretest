#!/usr/bin/env python3
"""Test benchmark Qwen2-VL uniquement."""

import os
import sys
import asyncio
import time
from pathlib import Path
from PIL import Image
import numpy as np

# Configuration cache - utiliser un dossier différent pour éviter les conflits
os.environ['HF_HOME'] = '/home/elfried-kinzoun/.cache/huggingface_qwen'
os.environ['TRANSFORMERS_CACHE'] = '/home/elfried-kinzoun/.cache/transformers_qwen'

# Path
sys.path.append(str(Path(__file__).parent / "src"))

from src.core.vlm.dynamic_model import DynamicVisionLanguageModel
from src.core.types import AnalysisRequest


class Qwen2VLBenchmark:
    """Benchmark spécialisé pour Qwen2-VL."""
    
    def __init__(self):
        # Configuration VLM avec Qwen2-VL uniquement
        self.vlm = DynamicVisionLanguageModel(
            default_model="qwen2-vl-7b-instruct",
            enable_fallback=False  # Pas de fallback
        )
        self.results = {}
    
    async def setup(self):
        """Initialisation du modèle."""
        print("🚀 Benchmark Qwen2-VL")
        print("=" * 40)
        
        # Nettoyer le cache Kimi pour libérer de l'espace
        print("🧹 Nettoyage cache Kimi-VL...")
        os.system("rm -rf ~/.cache/huggingface/hub/models--moonshotai--Kimi-VL-A3B-* 2>/dev/null")
        
        print("⏳ Chargement Qwen2-VL-7B-Instruct...")
        start_time = time.time()
        
        success = await self.vlm.switch_model("qwen2-vl-7b-instruct")
        load_time = time.time() - start_time
        
        if success:
            print(f"✅ Qwen2-VL chargé en {load_time:.1f}s")
            print(f"📊 Modèle: {self.vlm.current_model_id}")
            print(f"🔧 Device: {self.vlm.device}")
            return True
        else:
            print("❌ Échec du chargement")
            return False
    
    def create_test_image(self, scenario: str) -> Image.Image:
        """Créer une image de test selon le scénario."""
        
        # Image simple 224x224 RGB
        if scenario == "simple":
            # Image unie bleue
            img_array = np.ones((224, 224, 3), dtype=np.uint8) * [100, 100, 255]
        
        elif scenario == "complex":
            # Motif complexe différent
            img_array = np.zeros((224, 224, 3), dtype=np.uint8)
            # Damier coloré
            for i in range(224):
                for j in range(224):
                    if (i // 28 + j // 28) % 2 == 0:
                        img_array[i, j] = [255, 200, 100]
                    else:
                        img_array[i, j] = [100, 200, 255]
        
        elif scenario == "surveillance":
            # Scène surveillance différente
            img_array = np.ones((224, 224, 3), dtype=np.uint8) * [40, 60, 40]
            # Formes "objets suspects"
            img_array[30:80, 30:60] = [255, 100, 100]  # Objet rouge
            img_array[150:200, 160:190] = [100, 255, 100]  # Objet vert
            # Ligne "barrière" 
            img_array[110:115, :] = [200, 200, 200]
        
        elif scenario == "text":
            # Test OCR - texte simulé
            img_array = np.ones((224, 224, 3), dtype=np.uint8) * 255
            # Barres noires simulant du texte
            img_array[50:70, 50:150] = [0, 0, 0]
            img_array[80:100, 50:120] = [0, 0, 0]
            img_array[110:130, 50:180] = [0, 0, 0]
            img_array[140:160, 50:140] = [0, 0, 0]
        
        else:
            img_array = np.ones((224, 224, 3), dtype=np.uint8) * 128
        
        return Image.fromarray(img_array)
    
    async def test_scenario(self, scenario_name: str, prompt: str, image: Image.Image):
        """Tester un scénario spécifique."""
        
        print(f"\n📋 Test: {scenario_name}")
        print("-" * 30)
        
        # Créer la requête
        request = AnalysisRequest(
            image=image,
            prompt=prompt,
            enable_advanced_tools=True,
            max_tokens=256
        )
        
        # Mesure des performances
        start_time = time.time()
        
        try:
            response = await self.vlm.analyze_image(request)
            
            analysis_time = time.time() - start_time
            
            # Résultats
            result = {
                "temps_analyse": analysis_time,
                "succes": True,
                "suspicion": response.suspicion_level,
                "action": response.recommended_action,
                "confiance": response.confidence_score,
                "description": response.description[:100] + "..." if len(response.description) > 100 else response.description,
                "outils_utilises": len(response.tools_used),
                "erreur": None
            }
            
            print(f"⏱️  Temps: {analysis_time:.2f}s")
            print(f"🎯 Suspicion: {response.suspicion_level}")
            print(f"🔧 Action: {response.recommended_action}")
            print(f"📊 Confiance: {response.confidence_score:.2f}")
            print(f"🛠️  Outils: {len(response.tools_used)}")
            print(f"📝 Description: {result['description']}")
            
        except Exception as e:
            analysis_time = time.time() - start_time
            result = {
                "temps_analyse": analysis_time,
                "succes": False,
                "erreur": str(e),
                "suspicion": None,
                "action": None,
                "confiance": 0.0,
                "description": None,
                "outils_utilises": 0
            }
            print(f"❌ Erreur ({analysis_time:.2f}s): {e}")
        
        self.results[scenario_name] = result
        return result
    
    async def run_benchmark(self):
        """Exécuter tous les tests."""
        
        # Vérification du setup
        if not await self.setup():
            return
        
        # Scénarios de test (adaptés aux forces de Qwen2-VL)
        scenarios = [
            {
                "name": "surveillance_avancee",
                "prompt": "Analyze this surveillance scene. Detect any objects, people, or suspicious activities. Provide detailed reasoning.",
                "image_type": "surveillance"
            },
            {
                "name": "raisonnement_visuel", 
                "prompt": "Examine this image carefully. What patterns, relationships, or logical structures can you identify?",
                "image_type": "complex"
            },
            {
                "name": "detection_fine",
                "prompt": "Perform detailed object detection and classification in this image. List all elements you can identify.",
                "image_type": "simple"
            },
            {
                "name": "ocr_analyse",
                "prompt": "Analyze any text-like elements in this image. Can you identify writing, symbols, or structured information?",
                "image_type": "text"
            },
            {
                "name": "evaluation_securite_complexe",
                "prompt": "Assess the security implications of this scene. Consider multiple factors and provide risk analysis.",
                "image_type": "surveillance"
            }
        ]
        
        print(f"\n🧪 Lancement de {len(scenarios)} tests...")
        
        for scenario in scenarios:
            image = self.create_test_image(scenario["image_type"])
            await self.test_scenario(
                scenario["name"],
                scenario["prompt"], 
                image
            )
            
            # Pause entre tests
            await asyncio.sleep(2)  # Plus long car Qwen peut être plus lent
    
    def generate_report(self):
        """Génerer rapport de performance."""
        
        print("\n📊 RAPPORT QWEN2-VL")
        print("=" * 50)
        
        if not self.results:
            print("❌ Aucun résultat disponible")
            return
        
        # Statistiques globales
        total_tests = len(self.results)
        succes_tests = sum(1 for r in self.results.values() if r["succes"])
        temps_total = sum(r["temps_analyse"] for r in self.results.values())
        temps_moyen = temps_total / total_tests if total_tests > 0 else 0
        
        print(f"📈 Tests réalisés: {total_tests}")
        print(f"✅ Taux de succès: {(succes_tests/total_tests)*100:.1f}%")
        print(f"⏱️  Temps moyen: {temps_moyen:.2f}s")
        print(f"⚡ Temps total: {temps_total:.2f}s")
        
        # Analyse de la stabilité
        if succes_tests > 0:
            temps_succes = [r["temps_analyse"] for r in self.results.values() if r["succes"]]
            temps_min = min(temps_succes)
            temps_max = max(temps_succes)
            print(f"📊 Plage temps: {temps_min:.2f}s - {temps_max:.2f}s")
        
        # Détail par test
        print(f"\n📋 Détail des tests:")
        for name, result in self.results.items():
            status = "✅" if result["succes"] else "❌"
            print(f"  {status} {name}: {result['temps_analyse']:.2f}s")
            if result["succes"]:
                print(f"     → Suspicion: {result['suspicion']}, Confiance: {result['confiance']:.2f}")
            else:
                print(f"     → Erreur: {result['erreur']}")
        
        # Points forts de Qwen2-VL
        print(f"\n💡 Analyse Qwen2-VL:")
        
        # Évaluation performance
        if succes_tests == total_tests:
            print("  🎯 Fiabilité excellente - Aucune erreur")
        elif succes_tests >= total_tests * 0.8:
            print("  👍 Fiabilité correcte - Majoritairement stable")
        else:
            print("  ⚠️  Fiabilité faible - Instabilité détectée")
        
        # Évaluation vitesse
        if temps_moyen < 3.0:
            print("  ⚡ Rapidité excellente")
        elif temps_moyen < 8.0:
            print("  🐌 Vitesse modérée mais acceptable")
        else:
            print("  🐌 Lenteur significative")
        
        # Forces spécifiques
        print("\n🎯 Points forts attendus de Qwen2-VL:")
        print("  • Raisonnement visuel sophistiqué")
        print("  • Analyse détaillée des scènes complexes")
        print("  • Capacité OCR intégrée")
        print("  • Stabilité des réponses")


async def main():
    """Point d'entrée principal."""
    
    benchmark = Qwen2VLBenchmark()
    
    try:
        await benchmark.run_benchmark()
        benchmark.generate_report()
        
        print("\n💾 Pour comparer avec Kimi-VL, lancez:")
        print("   python test_kimi_vl_only.py")
        
    except KeyboardInterrupt:
        print("\n⏹️  Test interrompu")
    except Exception as e:
        print(f"\n❌ Erreur globale: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())