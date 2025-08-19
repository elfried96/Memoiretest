#!/usr/bin/env python3
"""Test benchmark Kimi-VL uniquement."""

import os
import sys
import asyncio
import time
from pathlib import Path
from PIL import Image
import numpy as np

# Configuration cache
os.environ['HF_HOME'] = '/home/elfried-kinzoun/.cache/huggingface'
os.environ['TRANSFORMERS_CACHE'] = '/home/elfried-kinzoun/.cache/transformers'

# Path
sys.path.append(str(Path(__file__).parent / "src"))

from src.core.vlm.dynamic_model import DynamicVisionLanguageModel
from src.core.types import AnalysisRequest


class KimiVLBenchmark:
    """Benchmark spécialisé pour Kimi-VL."""
    
    def __init__(self):
        # Configuration VLM avec Kimi-VL uniquement
        self.vlm = DynamicVisionLanguageModel(
            default_model="kimi-vl-a3b-thinking",
            enable_fallback=False  # Pas de fallback
        )
        self.results = {}
    
    async def setup(self):
        """Initialisation du modèle."""
        print("🚀 Benchmark Kimi-VL")
        print("=" * 40)
        
        print("⏳ Chargement Kimi-VL-A3B-Thinking...")
        start_time = time.time()
        
        success = await self.vlm.switch_model("kimi-vl-a3b-thinking")
        load_time = time.time() - start_time
        
        if success:
            print(f"✅ Kimi-VL chargé en {load_time:.1f}s")
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
            # Image unie rouge
            img_array = np.ones((224, 224, 3), dtype=np.uint8) * [255, 100, 100]
        
        elif scenario == "complex":
            # Motif plus complexe
            img_array = np.zeros((224, 224, 3), dtype=np.uint8)
            # Gradients
            for i in range(224):
                for j in range(224):
                    img_array[i, j] = [i % 255, j % 255, (i + j) % 255]
        
        elif scenario == "surveillance":
            # Simulation scène surveillance (rectangles = personnes/objets)
            img_array = np.ones((224, 224, 3), dtype=np.uint8) * [50, 50, 50]
            # Rectangles "personnes"
            img_array[50:100, 50:80] = [200, 150, 100]  # Personne 1
            img_array[120:170, 140:170] = [180, 160, 120]  # Personne 2
        
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
        
        # Scénarios de test
        scenarios = [
            {
                "name": "surveillance_normale",
                "prompt": "Analysez cette scène de surveillance. Détectez les personnes et évaluez les risques potentiels.",
                "image_type": "surveillance"
            },
            {
                "name": "detection_simple", 
                "prompt": "Que voyez-vous dans cette image ? Décrivez les objets présents.",
                "image_type": "simple"
            },
            {
                "name": "analyse_complexe",
                "prompt": "Analysez cette image complexe en détail. Identifiez les motifs, couleurs et formes.",
                "image_type": "complex"
            },
            {
                "name": "evaluation_securite",
                "prompt": "Évaluez le niveau de sécurité de cette scène. Y a-t-il des comportements suspects ?",
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
            await asyncio.sleep(1)
    
    def generate_report(self):
        """Génerer rapport de performance."""
        
        print("\n📊 RAPPORT KIMI-VL")
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
        
        # Détail par test
        print(f"\n📋 Détail des tests:")
        for name, result in self.results.items():
            status = "✅" if result["succes"] else "❌"
            print(f"  {status} {name}: {result['temps_analyse']:.2f}s")
            if result["succes"]:
                print(f"     → Suspicion: {result['suspicion']}, Confiance: {result['confiance']:.2f}")
            else:
                print(f"     → Erreur: {result['erreur']}")
        
        # Recommandations
        print(f"\n💡 Analyse Kimi-VL:")
        if succes_tests == total_tests:
            print("  🎯 Performance excellente - Modèle fiable")
        elif succes_tests >= total_tests * 0.8:
            print("  👍 Performance correcte - Quelques améliorations possibles")
        else:
            print("  ⚠️  Performance faible - Vérification nécessaire")
        
        if temps_moyen < 2.0:
            print("  ⚡ Vitesse excellente")
        elif temps_moyen < 5.0:
            print("  🐌 Vitesse correcte")
        else:
            print("  🐌 Lent - Optimisation recommandée")


async def main():
    """Point d'entrée principal."""
    
    benchmark = KimiVLBenchmark()
    
    try:
        await benchmark.run_benchmark()
        benchmark.generate_report()
        
    except KeyboardInterrupt:
        print("\n⏹️  Test interrompu")
    except Exception as e:
        print(f"\n❌ Erreur globale: {e}")


if __name__ == "__main__":
    asyncio.run(main())